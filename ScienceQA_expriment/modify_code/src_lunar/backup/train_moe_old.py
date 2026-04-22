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
    current_forget_idx: int,        # index of current-task forget expert (E-1 after expansion)
    num_epochs: int = 100,
    expert_weight: float = 1.0,     # weight applied to the combined expert loss
    router_weight: float = 1.0,     # weight applied to router CE loss
    aux_weight: float = 0.01,       # weight applied to load-balancing aux loss
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

    frozen_expert_indices = [0] + list(range(1, current_forget_idx))
    frozen_gate_indices   = list(range(1, current_forget_idx))
    # retain expert와 이전 expert들의 weight는 freeze
    print(f"[MoE Train] current_forget_idx={current_forget_idx} | frozen expert indices={frozen_expert_indices}")

    global_step = 0

    for epoch in range(num_epochs):
        running_loss   = 0.0
        running_retain = 0.0
        running_lunar  = 0.0
        running_router = 0.0  

        with tqdm(total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]") as pbar:
            for batch_idx, batch in enumerate(train_loader):
                inputs_list, targets_list, domain_labels = batch
                # print("domain_labels is : ", domain_labels)

                domain_labels = domain_labels.to(device)          # [B]
                inputs_list   = [x.to(device) for x in inputs_list]
                targets_list  = [y.to(device) for y in targets_list]

                retain_mask = (domain_labels == 0)                  # [B] bool
                forget_mask = (domain_labels == current_forget_idx) # [B] bool

                # has_retain = retain_mask.any() 
                has_retain = False
                has_forget = forget_mask.any()

                optimizer.zero_grad(set_to_none=True)

                outputs_mixture   = []  # mixture output for retain MSE
                router_logits_list = [] # experts pre-softmax score

                for layer_i, (model, x) in enumerate(zip(model_list, inputs_list)):
                    y_mix, logits = model(x, return_router_logits=True) 
                    outputs_mixture.append(y_mix) # [B,D] : [64, 4096]
                    router_logits_list.append(logits) # [B,D] : [64, experts]

                # 1) Router CE loss
                router_loss = torch.tensor(0.0, device=device)
                if router_weight > 0.0:
                    for layer_i, logits in enumerate(router_logits_list):
                        E = logits.size(-1) # 현재 expert 개수

                        # Per-layer target: 0 (retain) or current_forget_idx (forget)
                        # Clamp to [0, E-1] as a safety guard then assert validity.
                        targets_router = domain_labels.clone()
                        targets_router = targets_router.clamp(0, E - 1)

                        # Validation: any label out of range is a caller bug
                        assert (targets_router >= 0).all() and (targets_router < E).all(), (
                            f"[Layer {layer_i}] Router target out of range [0, {E}). "
                            f"domain_labels unique: {domain_labels.unique().tolist()}"
                        )

                        flat_logits  = logits.float()
                        flat_targets = targets_router
                        router_loss = router_loss + ce_loss_fn(flat_logits, flat_targets)

                # 2) Expert losses
                # 2a) Retain loss: MSE between mixture output and retain target
                #     Expert 0 is the retain expert; the mixture is dominated by it
                #     when the router correctly routes retain samples to expert 0.
                retain_loss = torch.tensor(0.0, device=device)
                if has_retain:
                    for y_mix, y_tgt in zip(outputs_mixture, targets_list):
                        retain_loss = retain_loss + mse_loss_fn(y_mix[retain_mask], y_tgt[retain_mask])

                # 2b) LUNAR loss: MSE between current forget-expert output and
                #     the LUNAR-redirected target. Uses forced_expert_idx so that
                #     ONLY expert `current_forget_idx` contributes to the output
                #     (no gradient through other experts or the router weights).
                lunar_loss = torch.tensor(0.0, device=device)
                if has_forget:
                    for model, x, y_tgt in zip(model_list, inputs_list, targets_list):
                        y_forget_expert, _ = model(
                            x[forget_mask],
                            return_router_logits=True,
                            forced_expert_idx=current_forget_idx,
                        )
                        lunar_loss = lunar_loss + mse_loss_fn(
                            y_forget_expert, y_tgt[forget_mask]
                        )

                expert_loss = retain_loss + lunar_loss
                loss = (
                    expert_weight * expert_loss
                    + router_weight * router_loss
                )
                loss.backward()

                for model in model_list:
                    zero_expert_param_grads(model, frozen_expert_indices)
                    zero_gate_row_grads(model, frozen_gate_indices)

                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for model in model_list for p in model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )

                optimizer.step()

                running_loss   += loss.detach().item()
                running_retain += retain_loss.detach().item()
                running_lunar  += lunar_loss.detach().item()
                running_router += router_loss.detach().item()
                global_step += 1

                if debug_every > 0 and global_step % debug_every == 0:
                    with torch.no_grad():
                        if retain_mask.any():
                            x_ret = inputs_list[0][retain_mask]
                            x_ret = x_ret[: min(4, x_ret.size(0))]
                            base_ret = model_list[0].linear(x_ret)
                            out_ret, _ = model_list[0](x_ret, return_router_logits=True)
                            print("retain diff:", (out_ret - base_ret).abs().mean().item())

                        if forget_mask.any():
                            x_for = inputs_list[0][forget_mask]
                            x_for = x_for[: min(4, x_for.size(0))]
                            base_for = model_list[0].linear(x_for)
                            out_for, _ = model_list[0](x_for, return_router_logits=True)
                            print("forget diff:", (out_for - base_for).abs().mean().item())

                        debug_msg = []
                        for li, logits in enumerate(router_logits_list):
                            probs      = torch.softmax(logits.float(), dim=-1)
                            probs_flat = probs.reshape(-1, probs.shape[-1])  # [N, E]
                            mean_prob  = probs_flat.mean(dim=0)              # [E]
                            top1       = probs_flat.argmax(dim=-1)
                            counts     = torch.bincount(top1, minlength=probs_flat.shape[-1]).float()
                            counts     = counts / counts.sum().clamp_min(1e-9)
                            debug_msg.append(
                                f"L{li}: mean_prob={mean_prob.cpu().tolist()}, "
                                f"top1_frac={counts.cpu().tolist()}"
                            )
                        # domain label distribution
                        r_cnt = int(retain_mask.sum())
                        f_cnt = int(forget_mask.sum())
                        print(
                            f"\n[Debug step {global_step}] "
                            f"retain_n={r_cnt}, forget_n={f_cnt} | "
                            + " | ".join(debug_msg)
                        )
                        for li, model in enumerate(model_list):
                            print(f"[Layer {li}] expert0 norm:",
                                model.lora_A[0].abs().mean().item(),
                                model.lora_B[0].abs().mean().item())

                pbar.update(1)
                pbar.set_postfix(
                    loss   =f"{loss.item():.4f}",
                    retain =f"{retain_loss.item():.4f}",
                    lunar  =f"{lunar_loss.item():.4f}",
                    router =f"{router_loss.item():.4f}",
                )

            if scheduler is not None:
                scheduler.step()

        n_batches  = max(len(train_loader), 1)
        epoch_loss   = running_loss   / n_batches
        epoch_retain = running_retain / n_batches
        epoch_lunar  = running_lunar  / n_batches
        epoch_router = running_router / n_batches

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Loss: {epoch_loss:.4f} | "
            f"Retain(MSE): {epoch_retain:.4f} | "
            f"LUNAR: {epoch_lunar:.4f} | "
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