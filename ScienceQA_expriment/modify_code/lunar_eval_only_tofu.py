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
    prepare_trainset,
)
from src_lunar.model_utils.model_loader import load_model
from src_lunar.estimated_net_utils import (
    load_lora_only, save_lora_only, train_multiple_layers, ActivationDataset_multiple_layers
)
from src_lunar.eval_util_tofu import custom_evaluate_tofu

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


@hydra.main(version_base=None, config_path="config", config_name="forget_tofu")
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
   
    # ################## load LoRA ####################
    if getattr(cfg, "load_unlearned_model", False):
        updated_model = load_lora_only( # 이미 t이전에 저장된 lora model을 load 
            updated_model,
            load_path=cfg.load_unlearned_model_path, # load할 lora의 경로 
            layer_idxs=layer_idx_list, # unlearn할 target layer
        )
    else:
        candidate_directions = generate_candidate_directions(  # uv = mean_activations_harmful - mean_activations_harmless
            cfg, model_base, ref_train, forget_train
        )  # ['n_pos n_layer d_model'] : torch.Size([1, 32, 4096])

    direction = []
    for layer_index in layer_idx_list: 
        direction.append(candidate_directions[positions, layer_index+1, :]) # [n_pos, n_layer, d_model]

    # ################## load LoRA ####################
    if getattr(cfg, "load_unlearned_model", False):
        updated_model = load_lora_only( # 이미 t이전에 저장된 lora model을 load 
            updated_model,
            load_path=cfg.load_unlearned_model_path, # load할 lora의 경로 
            layer_idxs=layer_idx_list, # unlearn할 target layer
        )
    else:
    ################## train LoRA ####################
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

        # 지정한 layer에 해당하는 proj 또는 lora를 준비
        # down proj : Linear(in_features=11008, out_features=4096, bias=False)
        estimated_net_list = prepare_estimated_net_lora_list( 
            device=device,
            layer_idx_list=layer_idx_list,
            model_base=model_base,
            init_model_list=None, 
        ) 

        # prepare the training dataset
        train_dataset_forget = ActivationDataset_multiple_layers(
            forget_input_list, forget_target_list
        )
        train_dataset_remain = ActivationDataset_multiple_layers(
            remain_input_list, remain_target_list
        )
        combined_dataset = ConcatDataset([train_dataset_forget, train_dataset_remain]) 
        train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True) 

        # training lora
        optimizer = optim.AdamW(
            chain(*[model.parameters() for model in estimated_net_list]), lr=cfg.lr
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        updated_estimated_net_list = train_multiple_layers( # selection된 layer로 훈련 시작하여 mlp down proj 리턴 
            estimated_net_list, 
            train_loader, 
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
            data_path=cfg.sampled_data_path,
        ),
        # RD (retain)
        dict(
            name="retained",
            eval_target="retained_edge",
            data_path=cfg.retain_data_path,
        ),
        # COMMONQA (factual)
        dict(
            name="real_authors",
            eval_target="factual_edge",
            data_path=cfg.utiliy_data_path1,
        ),
        # OPENBOOKQA (factual)
        dict(
            name="world_facts",
            eval_target="factual_edge",
            data_path=cfg.utiliy_data_path2,
        ),
    ]

    all_eval_logs = {}
    for job in eval_jobs:
        print(f"[EVAL] {job['name']} | target={job['eval_target']} | path={job['data_path']}")
        eval_logs = custom_evaluate_tofu(
            cfg=cfg,
            name=job["name"],
            data_path=job["data_path"],
            tokenizer=model_base.tokenizer,   
            model=updated_model,
            eval_target=job["eval_target"],
            output_es_score=cfg.compute_es_score,
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
