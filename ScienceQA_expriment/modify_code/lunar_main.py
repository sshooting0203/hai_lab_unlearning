import os, torch, random, hydra, copy, json
import numpy as np
from itertools import chain
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from src_lunar.generate_directions import generate_candidate_directions
from src_lunar.dataset_utils import (
    load_dataset_to_get_direction, 
    split_raw_dataset_for_forget, 
    prepare_estimated_net_lora_list, 
    prepare_trainset
)
from src_lunar.model_utils.model_loader import load_model
from src_lunar.estimated_net_utils import (
    load_lora_only, save_lora_only, train_multiple_layers, ActivationDataset_multiple_layers
)
from src_lunar.eval_util import custom_evaluate

os.environ["WANDB_DISABLED"] = "true"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

@hydra.main(version_base=None, config_path="config", config_name="forget")
def train(cfg):
    print(cfg)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positions = cfg.positions
    layer_idx_list = cfg.layer_modified
    coeff_list = cfg.coeff_list

    print(f'[LOAD] loading model from {cfg.model_family} at {cfg.model_path}')
    model_base = load_model(cfg.model_family, cfg.model_path, device)
    updated_model = copy.deepcopy(model_base) 

    ################## generate data ####################
    ref_train, forget_train = load_dataset_to_get_direction(  
        cfg,
        instructions_only=True,
        use_harmful=cfg.use_harmful,
        use_unverified=cfg.use_unverified,
    )
    print(f"[DATASET] forget_dataset: {len(forget_train)}") # 아래 forget_dataset이랑 동일
    print(f"[DATASET] ref_dataset: {len(ref_train)}")

    forget_dataset, retain_dataset = split_raw_dataset_for_forget(
        cfg,
        model_base,
        instructions_only=True,
        torch_reformat=False,
    ) 
    print(f"[DATASET] forget_dataset: {len(forget_dataset)}")
    print(f"[DATASET] retain_dataset: {len(retain_dataset)}")
   
    ################## load LoRA ####################
    if getattr(cfg, "load_unlearned_model", False):
        updated_model = load_lora_only( # 이미 t이전에 저장된 lora model을 load 
            updated_model,
            load_path=cfg.load_unlearned_model_path, # load할 lora의 경로 
            layer_idxs=layer_idx_list, # unlearn할 target layer
        )
    else:
    ################## train LoRA ####################
        candidate_directions = generate_candidate_directions(  # uv = mean_activations_harmful - mean_activations_harmless
            cfg, model_base, ref_train, forget_train
        )  # ['n_pos n_layer d_model']

        direction = []
        for layer_index in layer_idx_list: # 1) cadidnate_direction는 모든 layer를 수집, 그 중 config layer만 선택, 2) position에 해당하는 위치만 수집 
            direction.append(candidate_directions[positions, layer_index+1, :] # +1 because direction is calcuated using pre-hook
        )  # ['n_pos selected_layer d_model']

        # 실제 LORA의 train loss에 사용할 activation extract
        (
            forget_input_list, 
            forget_target_list,
            remain_input_list,  
            remain_target_list, 
            estimated_net_list, 
        ) = prepare_trainset( 
            layer_idx_list,
            model_base,
            forget_dataset,
            retain_dataset, 
            direction,
            coeff_list,
        )

        # 지정한 layer에 해당하는 proj 또는 lora를 준비
        # down proj : Linear(in_features=11008, out_features=4096, bias=False)
        estimated_net_list = prepare_estimated_net_lora_list( 
            device=device,
            layer_idx_list=layer_idx_list,
            model_base=model_base,
            init_model_list=None, # @ 수정 : 인자로 지정하여 layer에 해당하는 t개의 lora를 로드 가능(default는 1개의 unlearn lora)
        ) 

        # prepare the training dataset
        train_dataset_forget = ActivationDataset_multiple_layers(
            forget_input_list, forget_target_list
        )
        train_dataset_remain = ActivationDataset_multiple_layers(
            remain_input_list, remain_target_list
        )
        combined_dataset = ConcatDataset([train_dataset_forget, train_dataset_remain]) # @ 수정
        train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True) # = train_dataset_forget(input, target) + train_dataset_remain(input, target)

        # training lora
        optimizer = optim.AdamW(
            chain(*[model.parameters() for model in estimated_net_list]), lr=cfg.lr
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        updated_estimated_net_list = train_multiple_layers( # selection된 layer로 훈련 시작하여 mlp down proj 리턴 
            estimated_net_list, 
            train_loader, # train_loader는 input_list와 target_list로 구성 
            optimizer,
            scheduler,
            device=device,
            num_epochs=cfg.num_epochs,
        )

        # 2) no merge, save adapter
        for i, layer_idx in enumerate(layer_idx_list):
            updated_model.model_block_modules[layer_idx].mlp.down_proj = updated_estimated_net_list[i]
        if cfg.save_unlearned_model:
            save_lora_only(
                updated_model, 
                layer_idx_list, 
                cfg.save_unlearned_model_path
            )
        print("[EVAL] start evaluate")

    eval_jobs = [
        # SD (forget)
        dict(
            name="forget",
            eval_target="forget_edge",
            data_path=cfg.dist_forget_path,
        ),
        # RD (retain)
        dict(
            name="retained",
            eval_target="retained_edge",
            data_path=cfg.dist_retain_path,
        ),
        # COMMONQA (factual)
        dict(
            name="commonqa",
            eval_target="factual_edge",
            data_path=cfg.dist_commonqa_path,
        ),
        # OPENBOOKQA (factual)
        dict(
            name="openbookqa",
            eval_target="factual_edge",
            data_path=cfg.dist_openbookqa_path,
        ),
    ]

    all_eval_logs = []
    for job in eval_jobs:
        print(f"[EVAL] {job['name']} | target={job['eval_target']} | path={job['data_path']}")
        eval_logs = custom_evaluate(
            cfg=cfg,
            name=job["name"],
            data_path=job["data_path"],
            tokenizer=model_base.tokenizer,   
            model=updated_model,
            eval_target=job["eval_target"],
            output_es_score=cfg.compute_es_score,
        )
        all_eval_logs.append(eval_logs)

    save_file = os.path.join(cfg.save_path, f"forget_{cfg.task_id}.json")
    save_dir = os.path.dirname(save_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"Saving completions to {save_file}")
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(eval_logs, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    train()


#===================================================

# # Generate refusal directions
# harmful_train, forget_train = load_dataset_to_get_direction( # direction을 구하기 위해 forget과 harmful을 로드
#     cfg,
#     data_path,
#     instructions_only=True,
#     use_harmful=cfg.use_harmful,
#     use_unverified=cfg.use_unverified,
# )
# candidate_directions = generate_candidate_directions( # uv 벡터 계산 : mean_activations_harmful - mean_activations_harmless
#     cfg, model_base, harmful_train, forget_train
# )  # ['n_pos n_layer d_model']
# direction = []
# for layer_index in layer_idx_list: # selected된 layer에 해당하는 direction uv 벡터만 로드
#     direction.append(candidate_directions[positions, layer_index+1, :]) # +1 because direction is calcuated using pre-hook
# # ['n_pos selected_layer d_model']

# forget_dataset, _ = split_raw_dataset_for_forget(
#     cfg,
#     data_path,
#     model_base,
#     instructions_only=True,
#     torch_reformat=False,
# ) # unlearn 데이터 로드 -> "Context: Read the description of a trait.\nLogan has blue eyes.\nQuestion: What information supports the conclusion that Logan inherited this trait?\nOptions: (A) Logan's mother has blue eyes. She passed this trait down to Logan. (B) Logan likes to wear a blue sweater to match his blue eyes.\n"
# print(f"forget_dataset: {len(forget_dataset)}")
# updated_model = copy.deepcopy(model_base)

    # # 실제 LORA의 train loss에 사용할 activation을 준비(11008->4096) # inp : torch.Size([108673, 11008]), tar: torch.Size([108673, 4096])
# (
#     forget_input_list, 
#     forget_target_list,
#     # remain_input_list,
#     # remain_target_list,
#     estimated_net_list, 
# ) = prepare_trainset( 
#     layer_idx_list,
#     model_base,
#     forget_dataset,
#     # retain_dataset,
#     direction,
#     coeff_list,
#     device,
# )

# estimated_net_list = prepare_estimated_net_lora_list( # 지정한 layer에 해당하는 proj 또는 lora를 준비 - down proj : Linear(in_features=11008, out_features=4096, bias=False)
#     cfg,
#     device=device,
#     layer_idx_list=layer_idx_list,
#     model_base=model_base,
#     init_model_list=None, # @ 수정 : 인자로 지정하여 layer에 해당하는 t개의 lora를 로드 가능(default는 1개의 unlearn lora)
# ) # prepare_estimated_net_list

# # prepare the training dataset
# train_dataset_forget = ActivationDataset_multiple_layers(
#     forget_input_list, forget_target_list
# )
# # train_dataset_remain = ActivationDataset_multiple_layers(
# #     remain_input_list, remain_target_list
# # )
# # combined_dataset = ConcatDataset([train_dataset_forget, train_dataset_remain]) # @ 수정
# combined_dataset = train_dataset_forget
# train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True) # = train_dataset_forget(input, target) + train_dataset_remain(input, target)

# # Training
# optimizer = optim.AdamW(
#     chain(*[model.parameters() for model in estimated_net_list]), lr=cfg.lr
# )
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

#===================================================

# 0) 원래 코드
# update the new weight in the base model
# for i, layer_idx in enumerate(layer_idx_list):
#     updated_model.model_block_modules[layer_idx].mlp.down_proj.weight.data = ( # 모델 down proj 바꿔치기 
#         updated_estimated_net_list[i].down_proj.weight.data
#     )

# 1) model merge
# for i, layer_idx in enumerate(layer_idx_list):
#     lora_net = updated_model.model_block_modules[layer_idx].mlp.down_proj  # LUNAR_LoRA_net
#     if isinstance(lora_net, LUNAR_LoRA_net):
#         lora_net.merge_weights() # 1) LoRA를 base weight에 병합
#         in_dim = lora_net.linear.in_features
#         out_dim = lora_net.linear.out_features
#         device = lora_net.linear.weight.device
#         dtype  = lora_net.linear.weight.dtype # 2) 다시 nn.Linear로 되돌리고 
#         merged_linear = torch.nn.Linear(in_dim, out_dim, bias=False, device=device, dtype=dtype)
#         with torch.no_grad():
#             merged_linear.weight.copy_(lora_net.linear.weight)
#         updated_model.model_block_modules[layer_idx].mlp.down_proj = merged_linear # 3) proj에 weight 설정
# if cfg.save_unlearned_model:
#     if not os.path.exists(os.path.dirname(cfg.save_unlearned_model_path)):
#         os.makedirs(os.path.dirname(cfg.save_unlearned_model_path))
#     print(f"Saving unlearned model to {cfg.save_unlearned_model_path}")
#     # Save model and tokenizer
#     updated_model._save_pretrained(cfg.save_unlearned_model_path)
