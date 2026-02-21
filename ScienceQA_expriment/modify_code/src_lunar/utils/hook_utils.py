# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch, contextlib, functools
from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor

from src_lunar.estimated_net_utils import LUNAR_LoRA_net

@contextlib.contextmanager
def add_hooks( 
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def get_direction_ablation_input_pre_hook(direction: Tensor):
    def hook_fn(module, input):
        nonlocal direction
        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_direction_ablation_output_hook(direction: Tensor):
    def hook_fn(module, input, output):
        nonlocal direction

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def get_all_direction_ablation_hooks(
    model_base,
    direction: Float[Tensor, "d_model"],
):
    fwd_pre_hooks = [
        (
            model_base.model_block_modules[layer],
            get_direction_ablation_input_pre_hook(direction=direction),
        )
        for layer in range(model_base.model.config.num_hidden_layers)
    ]
    fwd_hooks = [
        (
            model_base.model_attn_modules[layer],
            get_direction_ablation_output_hook(direction=direction),
        )
        for layer in range(model_base.model.config.num_hidden_layers)
    ]
    fwd_hooks += [
        (
            model_base.model_mlp_modules[layer],
            get_direction_ablation_output_hook(direction=direction),
        )
        for layer in range(model_base.model.config.num_hidden_layers)
    ]

    return fwd_pre_hooks, fwd_hooks


def get_directional_patching_input_pre_hook(
    direction: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]
):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction
        activation += coeff * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation

    return hook_fn


def get_activation_addition_input_pre_hook(
    vector: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]
):
    def hook_fn(module, input):
        nonlocal vector

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        vector = vector.to(activation)
        activation += coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation

    return hook_fn

# @ 수정 : hook 생성
def make_gate_inject_pre_hook(gate_vec, alpha_best):
    def _hook(module, args):
        x = args[0]
        if gate_vec is not None and gate_vec.size(0) != x.size(0):
            raise RuntimeError(f"gate_vec B={gate_vec.size(0)} but x B={x.size(0)}")
        module._gate_vec = gate_vec # gate vec 설정
        module._alpha_best = alpha_best # alpha_best 설정   
        return  
    return _hook

# @ 수정 : hook 추가
def build_gate_pre_hooks(cfg, model, gate_vec, alpha_best):
    if gate_vec is None:
        return []
    hooks = []
    for layer_idx in cfg.layer_modified:
        down_proj = model.model.layers[layer_idx].mlp.down_proj  
        if isinstance(down_proj, LUNAR_LoRA_net):
            hooks.append((down_proj, make_gate_inject_pre_hook(gate_vec, alpha_best))) # down_proj 접근하여 hook register
    return hooks

# LUNAR_LoRA_net(
# (linear): Linear(in_features=11008, out_features=4096, bias=False)
# (lora_dropout): Dropout(p=0.0, inplace=False)
# )