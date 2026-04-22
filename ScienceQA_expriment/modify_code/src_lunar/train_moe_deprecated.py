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
        if model.lora_A.grad is not None:
            for idx in freeze_indices:
                model.lora_A.grad[idx].zero_()
        if model.lora_B.grad is not None:
            for idx in freeze_indices:
                model.lora_B.grad[idx].zero_()

def zero_gate_row_grads(model, freeze_indices):
    if not freeze_indices:
        return
    with torch.no_grad():
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

                outputs_mixture   = []  # mixture output for retain MSE
                router_logits_list = [] # experts pre-softmax score

                for layer_i, (model, x) in enumerate(zip(model_list, inputs_list)):
                    # TODO: 문제가 이 부분에서 logits이 full router의 logits이라는 사실
                    y_mix, logits = model(x, return_router_logits=True)  # y_mix : mixture output, logits : softmax 이전의 logit !
                    outputs_mixture.append(y_mix) # [B,D] : [64, 4096]
                    router_logits_list.append(logits) # [B,D] : [64, experts]

                # 1) Router CE loss
                router_loss = torch.tensor(0.0, device=device)
                if router_weight > 0.0:
                    for layer_i, logits in enumerate(router_logits_list): # 여러 layer에서 나온 router logits 모음
                        E = logits.size(-1) # 현재 expert 개수
                        targets_router = domain_labels.clone() # domain_labels : 0 (retain) or current_forget_idx (forget)
                        targets_router = targets_router.clamp(0, E - 1)

                        assert (targets_router >= 0).all() and (targets_router < E).all(), (
                            f"[Layer {layer_i}] Router target out of range [0, {E}). "
                            f"domain_labels unique: {domain_labels.unique().tolist()}"
                        )
                        flat_logits  = logits.float()
                        flat_targets = targets_router
                        router_loss = router_loss + ce_loss_fn(flat_logits, flat_targets)

                lunar_loss = torch.tensor(0.0, device=device) # 2) LUNAR MSE Loss : 원래 original MSE loss를 사용(current forget expert output - LUNAR-redirected target)
                if has_forget:                                # forced_expert_idx를 사용하여 현재 forget할 expert에 대해서만 output에 contribute(no gradient through other experts or the router weights)
                    for model, x, y_tgt in zip(model_list, inputs_list, targets_list):
                        y_forget_expert, _ = model(
                            x[forget_mask],
                            return_router_logits=True,
                            forced_expert_idx=current_forget_idx,
                        )
                        lunar_loss = lunar_loss + mse_loss_fn(
                            y_forget_expert, y_tgt[forget_mask]
                        )

                loss = (
                    expert_weight * lunar_loss
                    + router_weight * router_loss
                )
                loss.backward() # backward

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