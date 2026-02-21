# Copyright (c) Meta Platforms, Inc. and affiliates.
import json, os, torch 
from torch.utils.data import DataLoader
from src_lunar.data_loader import QuestionsDataset
from src_lunar.hook_for_unlearn import (
    get_activations,
    perturb_post_block_activations_forget,
)
from src_lunar.estimated_net_utils import EstimatedNet, LUNAR_LoRA_net


def load_dataset_to_get_direction(
    cfg, 
    instructions_only=True, 
    use_harmful=True, 
    use_unverified=False
):
    # Load the forget dataset 
    data_path = cfg.data_name
    with open(data_path, "r") as f:
        forget_dataset = json.load(f)

    # Load the harmful dataset
    if use_harmful:
        harmful_file_path = os.path.join("dataset/splits", "harmful.json")
        print(f'loading harmful dataset from {harmful_file_path}')
        with open(harmful_file_path, "r") as f:
            harmful_dataset = json.load(f)
    elif use_unverified:
        unverified_file_path = os.path.join("dataset/splits", "unverified.json")
        print(f'loading unverified dataset from {unverified_file_path}')
        with open(unverified_file_path, "r") as f:
            harmful_dataset = json.load(f)

    if instructions_only:
        forget_train = [d["text_input"] for d in forget_dataset] # @@@ model input(forget data)
        harmful_train = [d["instruction"] for d in harmful_dataset] # @@@ model input(harmful data)
    return harmful_train, forget_train


def split_raw_dataset_for_forget(
    cfg,
    model_base,
    instructions_only=True,
    torch_reformat=False,
):
    # def torch_reformat(cfg, input_QA_list, tokenizer):
    #     max_length = 500
    #     torch_format_dataset = QuestionsDataset(
    #         input_QA_list=input_QA_list,
    #         tokenizer=tokenizer,
    #         configs=cfg,
    #         max_length=max_length,
    #         split="train",
    #     )
    #     torch_format_dataset = DataLoader(
    #         torch_format_dataset,
    #         batch_size=1,
    #         shuffle=False,
    #     )
    #     return torch_format_dataset

    with open(cfg.r_data_path, "r") as f:
        retain_dataset = json.load(f)
    
    with open(cfg.data_name, "r") as f:
        forget_dataset = json.load(f)

    if instructions_only:
        def pack(d):
            if "topic" not in d:
                return {
                    "text_input": d["text_input"],
                    "topic": d["domain"],   
                }
            return {
                "text_input": d["text_input"],
                "topic": d["topic"],   
            } 
        forget_dataset = [pack(d) for d in forget_dataset] # @@@ model input(forget data)
        retain_dataset = [pack(d) for d in retain_dataset] # @@@ model input(retain data)
    else:
        if torch_reformat:
            forget_dataset = torch_reformat(cfg, forget_dataset, model_base.tokenizer)
            retain_dataset = torch_reformat(cfg, retain_dataset, model_base.tokenizer)
            
    return forget_dataset, retain_dataset


def prepare_trainset(
    layer_idx_list,
    model_base,
    forget_dataset,
    retain_dataset,
    direction,
    coeff_list,
):
    estimated_net_list = []
    forget_input_list = []
    forget_target_list = []
    remain_input_list = []
    remain_target_list = [] # torch.Size([layer, 4096])

    for i, layer_idx in enumerate(layer_idx_list): 
        (
            post_block_activation_forget, # [S, T, D] : torch.Size([1, 98, 4096]) -> batch 샘플 개수 x 토큰 개수 x 차원
            post_block_activation_remain,
            pre_post_attention_layernorm_activation_forget, # [S, T, D] : torch.Size([1, 98, 4096])
            pre_post_attention_layernorm_activation_remain,
            pre_down_proj_activation_forget, # [S, T, D] : torch.Size([1, 98, 14336])
            pre_down_proj_activation_remain,
        ) = get_activations(model_base, layer_idx, forget_dataset, retain_dataset)

        # perturb the post block activations for forget data(direction 생성할 때 cfg.position으로 끝에서부터 얼마나 변형시킬지 결정)
        post_block_activation_forget = perturb_post_block_activations_forget( # [S, T, D] : torch.Size([1, 98, 4096])
            post_block_activation_forget,
            direction[i],
            coeff=coeff_list[i],
        )

        # prepare the data for training estimated net
        concat_forget_input, concat_forget_target, input_remain, target_remain = (
            prepare_trainset_raw(
                pre_down_proj_activation_forget,
                post_block_activation_forget, # forget : y에 해당 -> activation 변형을 사용 
                pre_post_attention_layernorm_activation_forget,
                pre_down_proj_activation_remain,
                post_block_activation_remain,
                pre_post_attention_layernorm_activation_remain,
            )
        )
        forget_input_list.append(concat_forget_input)
        forget_target_list.append(concat_forget_target)
        remain_input_list.append(input_remain)
        remain_target_list.append(target_remain)

    return (
        forget_input_list,
        forget_target_list,
        remain_input_list,
        remain_target_list,
        estimated_net_list, # 별 의미 x 
    )

# 입력 x -> (LayerNorm1)
# self-attn -> (+x)
# output o -> (LayerNorm2)
# mlp -> (+o)
# output y
def prepare_trainset_raw(
    pre_down_proj_activation_forget, # forget : mlp 내부의 down proj의 입력에 해당
    post_block_activation_forget, # forget : y에 해당 
    pre_post_attention_layernorm_activation_forget, # forget : mlp의 입력에 해당 
    pre_down_proj_activation_remain, # remain : mlp 내부의 down proj의 입력에 해당
    post_block_activation_remain, # remain : y에 해당 
    pre_post_attention_layernorm_activation_remain, # remain : mlp의 입력에 해당 
):
    # prepare the forget data
    inputs_forget = [item.detach() for item in pre_down_proj_activation_forget] # forget : mlp 내부의 down proj의 입력에 해당
    post_mlp_activation_forget = [
        x - y
        for x, y in zip(
            post_block_activation_forget, pre_post_attention_layernorm_activation_forget # y에 해당 - mlp의 입력에 해당
            # 최종 mlp의 출력에서 residual 출력을 빼기, down_proj만 조절
        )
    ] 
    targets_forget = [item.detach() for item in post_mlp_activation_forget]

    # prepare the remain data
    pre_down_proj_activation_remain = [
        item.view(-1, pre_down_proj_activation_remain[0].size()[-1])
        for item in pre_down_proj_activation_remain
    ]
    post_block_activation_remain = [ 
        item.view(-1, post_block_activation_remain[0].size()[-1])
        for item in post_block_activation_remain
    ]
    pre_post_attention_layernorm_activation_remain = [ 
        item.view(-1, pre_post_attention_layernorm_activation_remain[0].size()[-1])
        for item in pre_post_attention_layernorm_activation_remain
    ]

    inputs_remain = torch.cat(pre_down_proj_activation_remain, dim=0) # remain : mlp 내부의 down proj의 입력에 해당
    targets_remain = torch.cat(post_block_activation_remain, dim=0) - torch.cat(
        pre_post_attention_layernorm_activation_remain, dim=0 # y에 해당 - mlp의 입력에 해당 
    )

    concat_forget_input = torch.cat(
        [activation.squeeze(0) for activation in inputs_forget], dim=0
    )
    concat_forget_target = torch.cat(
        [activation.squeeze(0) for activation in targets_forget], dim=0
    )

    return (
        concat_forget_input, 
        concat_forget_target, 
        inputs_remain, 
        targets_remain,
    )
# remain : input == target # forget : intput != target


#==============prepare Linear layer=============
def prepare_estimated_net_list(
    device, layer_idx_list, model_base, init_model_list=None
):
    estimated_net_list = []
    init_weight_list = []

    if init_model_list is None: # 초기 기준 weight는 베이스 모델의 down_proj weight로
        print("initialize the estimated net list with the model base MLP weight")
        for layer_idx in layer_idx_list:
            init_weight_list.append(
                model_base.model_block_modules[layer_idx].mlp.down_proj.weight.clone()
            )
    else: # init_model_list가 주어진 경우 그 weight로 
        print("initialize the estimated net list with the previous unlearning MLP weight")
        for estimante_model in init_model_list:
            init_weight_list.append(
                estimante_model.down_proj.weight.clone()
            )

    for weight_parameter in init_weight_list:
        down_proj_in_features = weight_parameter.shape[1]
        down_proj_out_features = weight_parameter.shape[0]
        print(f"down_proj_in_features: {down_proj_in_features}")
        print(f"down_proj_out_features: {down_proj_out_features}")
        estimated_down_proj = EstimatedNet(
            in_features=down_proj_in_features,
            out_features=down_proj_out_features,
            bias=False,
            original_down_proj_weight=weight_parameter,
        ).to(device, dtype=torch.bfloat16)
        estimated_net_list.append(estimated_down_proj)

    return estimated_net_list


#==============prepare LoRA layer=============
def prepare_estimated_net_lora_list(
    device, layer_idx_list, model_base, init_model_list=None
):
    estimated_net_list = []
    init_weight_list = []

    if init_model_list is None:  # 초기 기준 weight는 베이스 모델의 down_proj weight로
        print("initialize the estimated net list with the model base MLP weight")
        for layer_idx in layer_idx_list:
            init_weight_list.append(
                model_base.model_block_modules[layer_idx].mlp.down_proj.weight.clone()
            )
    else: # init_model_list가 주어진 경우 그 weight로 
        print("initialize the estimated net list with the previous unlearning MLP weight")
        for estimante_model in init_model_list: # 
            init_weight_list.append(
                estimante_model.down_proj.weight.clone()
            )

    for weight_parameter in init_weight_list: # 내부에 원본 weight(W0)를 기준으로 두고 학습 가능한 LoRA 저랭크 업데이트(ΔW = BA)를 추가
        down_proj_in_features = weight_parameter.shape[1]
        down_proj_out_features = weight_parameter.shape[0]
        print(f"down_proj_in_features: {down_proj_in_features}") # 11008
        print(f"down_proj_out_features: {down_proj_out_features}") # 4096
        estimated_down_proj = LUNAR_LoRA_net( 
            input_dim=down_proj_in_features,
            output_dim=down_proj_out_features,
            lora_rank=8,
            pretrained_weight=weight_parameter,
        ).to(device, dtype=torch.bfloat16)
        estimated_net_list.append(estimated_down_proj)
    return estimated_net_list


def load_dataset_json(dataset_name):
    data_path = os.path.join("dataset/unlearning", f"{dataset_name}.json")
    print(f"Loading dataset from {data_path}")
    with open(data_path, "r") as f:
        dataset_full = json.load(f)
    return dataset_full