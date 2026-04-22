# train_moe.py
from collections.abc import Mapping
from distutils.util import strtobool
from omegaconf import ListConfig
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence
import torch
import torch.nn.functional as F 
from huggingface_hub import Repository, create_repo
from packaging import version
from torch import nn
from transformers import __version__
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_pt_utils import (
    nested_detach,
)
from tqdm import tqdm

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

# def zero_expert_grads(model, freeze_indices: List[int]):
#     if not freeze_indices:
#         return
#     with torch.no_grad():
#         if model.lora_A.grad is not None:
#             for idx in freeze_indices:
#                 model.lora_A.grad[idx].zero_()
#         if model.lora_B.grad is not None:
#             for idx in freeze_indices:
#                 model.lora_B.grad[idx].zero_()
#         if hasattr(model, "gate"):
#             gate = model.gate
#             if hasattr(gate, "weight") and gate.weight.grad is not None:
#                 for idx in freeze_indices:
#                     gate.weight.grad[idx].zero_()
#             if hasattr(gate, "bias") and gate.bias is not None and gate.bias.grad is not None:
#                 for idx in freeze_indices:
#                     gate.bias.grad[idx].zero_()

def zero_expert_param_grads(model, freeze_indices):
    if not freeze_indices:
        return
    with torch.no_grad():
        if hasattr(model, "lora_A") and model.lora_A.grad is not None: # lora A 행렬 freeze
            for idx in freeze_indices:
                model.lora_A.grad[idx].zero_()
        if hasattr(model, "lora_B") and model.lora_B.grad is not None: # lora B 행렬 freeze
            for idx in freeze_indices:
                model.lora_B.grad[idx].zero_()
        if hasattr(model, "expert_W") and model.expert_W.grad is not None: # lora말고 expert W 행렬 freeze
            for idx in freeze_indices:
                model.expert_W.grad[idx].zero_()

def zero_gate_row_grads(model, freeze_indices):
    if not freeze_indices:
        return
    with torch.no_grad():
        ####################################
        if hasattr(model, "gate_hidden") and model.gate_hidden > 0:
            # model.gate is nn.Sequential
            out_layer = model.gate[-1] # 마지막 Linear 층
            # 이전 expert들의 classification weights 업데이트 방지
            if out_layer.weight.grad is not None:
                for idx in freeze_indices:
                    out_layer.weight.grad[idx].zero_()
            if out_layer.bias is not None and out_layer.bias.grad is not None:
                for idx in freeze_indices:
                    out_layer.bias.grad[idx].zero_()
            
            # 마지막 이전 layer를 freeze
            # for param in model.gate[:-1].parameters():
            #     if param.grad is not None:
            #         param.grad.zero_()
        ####################################
        else:
            gate = model.gate
            if hasattr(gate, "weight") and gate.weight.grad is not None:
                for idx in freeze_indices:
                    gate.weight.grad[idx].zero_()
            if hasattr(gate, "bias") and gate.bias is not None and gate.bias.grad is not None:
                for idx in freeze_indices:
                    gate.bias.grad[idx].zero_()


def train_moe_multiple_layers(
    model_list: List[torch.nn.Module],
    train_loader,
    optimizer,
    scheduler,
    device,
    current_forget_idx: int,        # index of current-task forget expert
    num_epochs: int = 100,
    expert_weight: float = 1.0,     # weight applied to the combined expert loss
    router_weight: float = 1.0,     # weight applied to router CE loss
    aux_weight: float = 0.01,       # 사용하지 않음
    num_experts: Optional[Sequence[int]] = None,
    top_k: Optional[Sequence[int]] = None,
    debug_every: int = 0,
    max_grad_norm: Optional[float] = None,
):
    print("[Debug] loss is modified!!!")
    assert current_forget_idx >= 1, (
        f"current_forget_idx must be >= 1 (0 is always the retain expert). "
        f"Got {current_forget_idx}."
    )

    for model in model_list:
        model.train()
        model.to(device)

    mse_loss_fn = torch.nn.MSELoss()
    ce_loss_fn  = torch.nn.CrossEntropyLoss()

    frozen_expert_indices = [0] + list(range(1, current_forget_idx)) # retain expert와 이전 expert들의 weight는 freeze
    frozen_gate_indices   = list(range(1, current_forget_idx)) # 이전 expert들의 gate weight는 freeze
    
    print(f"[MoE Train] current_forget_idx={current_forget_idx} | frozen expert indices={frozen_expert_indices}")
    global_step = 0

    for epoch in range(num_epochs):
        running_loss   = 0.0
        running_lunar  = 0.0
        running_router = 0.0  

        with tqdm(total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]") as pbar:
            for batch_idx, batch in enumerate(train_loader):
                inputs_list, targets_list, domain_labels = batch
                # print("domain_labels is : ", domain_labels)

                domain_labels = domain_labels.to(device)          
                inputs_list   = [x.to(device) for x in inputs_list] # lunar activation input
                targets_list  = [y.to(device) for y in targets_list] # lunar activation target

                retain_mask = (domain_labels == 0)                  # domain_labels==0 : retain 데이터에 해당
                forget_mask = (domain_labels == current_forget_idx) # domain_labels==current_forget_idx : 현재 forget 데이터에 해당

                # has_retain = retain_mask.any() 
                has_forget = forget_mask.any()
                optimizer.zero_grad(set_to_none=True)

                outputs_mixture   = []  
                router_logits_list = [] 
                lunar_loss = torch.tensor(0.0, device=device)

                for layer_i, (model, x, y_tgt) in enumerate(zip(model_list, inputs_list, targets_list)):
                    # Optimized forward: returns (y_mix, logits, BAx, base)
                    y_mix, logits, BAx, base = model(x, return_router_logits=True, return_all_experts=True)
                    outputs_mixture.append(y_mix)
                    router_logits_list.append(logits)

                    if has_forget:
                        # y_forget_expert = base[forget_mask] + scaling * expert_current(x[forget_mask])
                        # BAx[forget_mask, current_forget_idx] is exactly expert_current(x[forget_mask])
                        y_forget_expert = base[forget_mask] + model.scaling * BAx[forget_mask, current_forget_idx]
                        lunar_loss = lunar_loss + mse_loss_fn(
                            y_forget_expert, y_tgt[forget_mask]
                        )

                # 1) Router CE loss
                router_loss = torch.tensor(0.0, device=device)
                if router_weight > 0.0:
                    for layer_i, logits in enumerate(router_logits_list):
                        E = logits.size(-1)
                        assert 0 <= current_forget_idx < E, (
                            f"[Layer {layer_i}] current_forget_idx={current_forget_idx} out of range [0, {E})"
                        )

                        targets_router = domain_labels.clone()
                        # 현재 task에서 router CE를 걸 샘플만 유효하게 사용
                        valid_mask = (targets_router == 0) | (targets_router == current_forget_idx)
                        assert valid_mask.any(), (
                            f"[Layer {layer_i}] No valid samples for router CE. "
                            f"domain_labels unique: {domain_labels.unique().tolist()}"
                        )
                        logits_valid = logits[valid_mask].float() # mask 적용
                        targets_valid = targets_router[valid_mask] # mask 적용
                        # full logits [B_valid, E] 유지
                        # 단, retain(0) / current_forget_idx 외 클래스는 softmax 분모에서 제외
                        keep_mask = torch.zeros(E, dtype=torch.bool, device=logits.device)
                        keep_mask[0] = True
                        keep_mask[current_forget_idx] = True
                        masked_logits = logits_valid.masked_fill(~keep_mask.unsqueeze(0), -1e9)
                        router_loss = router_loss + ce_loss_fn(masked_logits, targets_valid)

                loss = expert_weight * lunar_loss + router_weight * router_loss
                
                if not loss.requires_grad:
                    optimizer.zero_grad(set_to_none=True)
                    pbar.update(1)
                    continue

                loss.backward()

                for model in model_list:
                    zero_expert_param_grads(model, frozen_expert_indices) # expert 파라미터 freeze
                    zero_gate_row_grads(model, frozen_gate_indices) # gate 일부 행 파라미터 freeze

                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for model in model_list for p in model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )
                optimizer.step() # optimize

                running_loss   += loss.detach().item()
                running_lunar  += lunar_loss.detach().item()
                running_router += router_loss.detach().item()
                global_step += 1

            if scheduler is not None:
                scheduler.step()

        n_batches  = max(len(train_loader), 1)
        epoch_loss   = running_loss   / n_batches
        epoch_lunar  = running_lunar  / n_batches
        epoch_router = running_router / n_batches

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Loss: {epoch_loss:.4f} | "
            f"LUNAR(MSE): {epoch_lunar:.4f} | "
            f"Router(CE): {epoch_router:.4f} | "
            f"expert_w={expert_weight} router_w={router_weight}"
        )
    print("MoE Training completed!")
    return model_list


def expand_per_layer(value, n_layers, name):
    if value is None:
        return None

    if isinstance(value, (list, tuple, ListConfig)):
        value = list(value)
        if len(value) == 0:
            raise ValueError(f"{name} is empty")
        if len(value) == 1:
            return [int(value[0])] * n_layers
        if len(value) < n_layers:
            raise ValueError(
                f"{name} length must be 1 or >= number of layers. "
                f"Got len={len(value)}, n_layers={n_layers}"
            )
        return [int(v) for v in value[:n_layers]]

    return [int(value)] * n_layers


"""
Computes ST-MoE router z loss (https://arxiv.org/abs/2202.08906)
See equation (5) on page 7

import torch
# constants
B = 16     # batch size
C = 256    # sequence length
n_exp = 8  # number of experts
# create input tensor for router z-loss
router_logits = torch.rand(B, C, n_exp) # [B, C, n_exp]
# exponentiate logits, sum logits of each expert, take log, and square
# code below is equivalent to the following:
# z_loss = torch.exp(router_logits)
# z_loss = torch.sum(z_loss, dim=-1)
# z_loss = torch.log(z_loss) ** 2.0
router_z_loss = torch.logsumexp(router_logits, dim=-1) ** 2.0  # [B, C]
# sum over all tokens and divide by total number of tokens
router_z_loss = torch.mean(router_z_loss)
"""