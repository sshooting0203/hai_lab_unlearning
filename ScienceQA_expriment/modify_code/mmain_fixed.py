# mmain_fixed.py
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
    prepare_estimated_net_moe_lora_list, 
    prepare_trainset
)
from src_lunar.estimated_net_utils import ActivationDataset_moe_multiple_layers
from src_lunar.estimated_net_moe_utils import (
    load_lora_moe_only,
    save_lora_moe_only,
    expand_lora_moe_experts,
)
from src_lunar.train_moe import expand_per_layer
from src_lunar.model_utils.model_loader import load_model
from src_lunar.eval_util import custom_evaluate
from src_lunar.train_moe import train_moe_multiple_layers

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
   
    ################## prepare dataset ####################
    candidate_directions = generate_candidate_directions(  # uv = mean_activations_harmful - mean_activations_harmless
        cfg, model_base, ref_train, forget_train
    )  # ['n_pos n_layer d_model']

    direction = [] 
    # 처음 수집된건 eoi token에 따라 5개가 맞지만 -> forget.yaml의 1개로 필터링됨
    # torch.Size([5, 32, 4096]) -> torch.Size([4096])
    for layer_index in layer_idx_list: # 1) cadidnate_direction는 모든 layer를 수집, 그 중 config layer만 선택, 2) position에 해당하는 위치만 수집 
        direction.append(candidate_directions[positions, layer_index+1, :] # +1 because direction is calcuated using pre-hook
    )  # ['n_pos selected_layer d_model']

    # 실제 LORA의 train loss에 사용할 activation extract
    (
        forget_input_list, 
        forget_target_list,
        remain_input_list,  
        remain_target_list, 
    ) = prepare_trainset( 
        layer_idx_list,
        model_base,
        forget_dataset,
        retain_dataset, 
        direction,
        coeff_list,
    )

    ################## load MoE layer ####################
    if getattr(cfg, "load_unlearned_model", False):
        print("\n[INFO] Load previous MoE LoRA model and expand experts")
        # (A) 이전 task까지 학습된 MoE load
        updated_model = load_lora_moe_only(
            model=updated_model,
            load_path=cfg.load_unlearned_model_path,
            layer_idxs=layer_idx_list,
            device=device,
            dtype=torch.bfloat16,
        )
        # (B) 새 task용 expert 1개 추가 -> forget용 expert 1개
        num_new_experts = getattr(cfg.moe, "experts_per_task", 1)
        estimated_net_list, new_expert_map = expand_lora_moe_experts(
            model=updated_model,
            layer_idxs=layer_idx_list,
            num_new_experts=num_new_experts,
            init_std=getattr(cfg.moe, "new_expert_init_std", 0.01),
        )
        print(f"[INFO] New expert map: {new_expert_map}")
        forget_label_idx = estimated_net_list[0].n_experts - 1
    else:
        print("\n[INFO] Initialize fresh MoE LoRA layers (t=0)")
        new_expert_map = None
        estimated_net_list = prepare_estimated_net_moe_lora_list(
            cfg=cfg,
            device=device,
            layer_idx_list=layer_idx_list,
            model_base=model_base,
            init_model_list=None,
        )
        forget_label_idx = getattr(cfg.moe, "initial_forget_expert_idx", 1)

    # forget index, retain index(0) 설정
    print("[INFO] forget_label_idx : ", forget_label_idx) # task별로 1 -> 2 -> 3 -> 4 ...
    domain_label_forget = torch.full((forget_input_list[0].size(0),), fill_value=forget_label_idx, dtype=torch.long)
    domain_label_remain = torch.zeros(remain_input_list[0].size(0), dtype=torch.long)

    train_dataset_forget = ActivationDataset_moe_multiple_layers(
        forget_input_list, forget_target_list, domain_label_forget # @ 수정
    )
    train_dataset_remain = ActivationDataset_moe_multiple_layers(
        remain_input_list, remain_target_list, domain_label_remain # @ 수정
    )
    combined_dataset = ConcatDataset([train_dataset_forget, train_dataset_remain]) # @ 수정
    train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True) # = train_dataset_forget(input, target) + train_dataset_remain(input, target)

    # training lora
    optimizer = optim.AdamW(
        chain(*[model.parameters() for model in estimated_net_list]), lr=cfg.lr
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    ################## train MoE layer ####################
    print("[TRAIN] start train")
    updated_estimated_net_list = train_moe_multiple_layers(
        model_list=estimated_net_list,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        current_forget_idx=forget_label_idx,
        num_epochs=cfg.num_epochs,
        expert_weight=cfg.moe.expert_weight,
        router_weight=cfg.moe.router_weight,
    )

    ################## save MoE layer ####################
    for i, layer_idx in enumerate(layer_idx_list): # no merge, save adapter
        updated_model.model_block_modules[layer_idx].mlp.down_proj = updated_estimated_net_list[i]

    if cfg.save_unlearned_model:
        save_lora_moe_only(
            model=updated_model,
            save_path=cfg.save_unlearned_model_path,
            layer_idxs=layer_idx_list,
        )
        
    print("[INFO] saved New MoE Model")

    print("[EVAL] start evaluate")
    data_name = str(cfg.data_name).lower()

    if data_name == "scienceqa":
        utility_jobs = [
            dict(
                name="commonqa",
                eval_target="factual_edge",
                data_path=cfg.utiliy_data_path1,
            ),
            dict(
                name="openbookqa",
                eval_target="factual_edge",
                data_path=cfg.utiliy_data_path2,
            ),
        ]
    elif data_name == "clincqa":
        utility_jobs = [
            dict(
                name="mrpc",
                eval_target="factual_edge",
                data_path=cfg.utiliy_data_path1,
            ),
            dict(
                name="rte",
                eval_target="factual_edge",
                data_path=cfg.utiliy_data_path2,
            ),
        ]
    else:
        raise ValueError(
            f"Unsupported cfg.data_name={cfg.data_name}. "
            f"Expected one of ['scienceqa', 'clincqa']"
        )

    eval_jobs = [
        # SD (forget)
        dict(
            name="forget",
            eval_target="forget_edge",
            data_path=cfg.sampled_data_path,
        ),
        # RD (retain)
        dict(
            name="retained",
            eval_target="retained_edge",
            data_path=cfg.retain_data_path,
        ),
        *utility_jobs,
    ]

    all_eval_logs = {}
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
            forget_label_idx=forget_label_idx
        )
        all_eval_logs[job["name"]] = {
            "eval_target": job["eval_target"],
            "data_path": job["data_path"],
            "results": eval_logs
        }

    save_file = os.path.join(cfg.save_path, f"forget_{cfg.task_id}.json")
    save_dir = os.path.dirname(save_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"Saving completions to {save_file}")
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(all_eval_logs, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    train()