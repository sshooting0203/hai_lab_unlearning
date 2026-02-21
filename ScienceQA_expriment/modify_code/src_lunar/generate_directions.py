# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch, os
from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from src_lunar.utils.hook_utils import add_hooks
from src_lunar.model_utils.model_base import ModelBase


def get_mean_activations_pre_hook(
    layer, 
    cache: Float[Tensor, "pos layer d_model"], 
    n_samples, 
    positions: List[int]
):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = (
            input[0].clone().to(cache)
        ) # torch.Size([5, 32, 4096])
        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0) # token 5개, layer축으로 sum
    return hook_fn


def get_mean_activations(
    model,
    tokenizer,
    instructions,
    tokenize_instructions_fn,
    block_modules: List[torch.nn.Module],
    batch_size=32,
    positions=[-1],
):
    torch.cuda.empty_cache()
    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers # 32
    n_samples = len(instructions) 
    d_model = model.config.hidden_size # 4096
    # print("[POS] : ", positions)
    # print("[POS LEN] : ", n_positions)

    mean_activations = torch.zeros(
        (n_positions, n_layers, d_model), dtype=torch.float64, device=model.device
    )

    fwd_pre_hooks = [
        (
            block_modules[layer],
            get_mean_activations_pre_hook(
                layer=layer,
                cache=mean_activations,
                n_samples=n_samples,
                positions=positions,
            ),
        )
        for layer in range(n_layers)
    ]

    for i in tqdm(range(0, len(instructions), batch_size)): 
        inputs = tokenize_instructions_fn(instructions=instructions[i : i + batch_size])
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )
    return mean_activations # 끝부분 여러 토큰(-5, -4, -3, -2, -1)의 위치별 평균 계산 ex. [/INST] : len(5)
    # 여기서는 모든 layer에 대한 activation hook를 다 넘기는 것으로 추정


def get_mean_diff(
    cfg,
    model,
    tokenizer,
    harmful_instructions,
    harmless_instructions,
    tokenize_instructions_fn,
    block_modules: List[torch.nn.Module],
    batch_size=32,
    positions=[-1],
):
    mean_activations_harmful = get_mean_activations(
        model,
        tokenizer,
        harmful_instructions,
        tokenize_instructions_fn,
        block_modules,
        batch_size=batch_size,
        positions=positions,
    ) # torch.Size([5, 32, 4096]) : position, layer, hidden_dim
    mean_activations_harmless = get_mean_activations(
        model,
        tokenizer,
        harmless_instructions,
        tokenize_instructions_fn,
        block_modules,
        batch_size=batch_size,
        positions=positions,
    ) # torch.Size([5, 32, 4096]) : position, layer, hidden_dim

    mean_diff: Float[Tensor, "n_positions n_layers d_model"] = (
        cfg.alpha_ref * mean_activations_harmful - cfg.alpha_for * mean_activations_harmless
    ) # @ 수정 : UV 벡터 생성 = D_ref - D_f

    alpha = float(cfg.alpha_ref)
    beta = float(cfg.alpha_for)
    print("[alpha] : ", alpha, ", [beta] : ", beta)
    return mean_diff


def generate_directions(
    cfg,
    model_base: ModelBase,
    harmful_instructions, # ref
    harmless_instructions, # forget
):
    mean_diffs = get_mean_diff(
        cfg,
        model_base.model,
        model_base.tokenizer,
        harmful_instructions, # harmful
        harmless_instructions, # forget
        model_base.tokenize_instructions_fn, # @@@
        model_base.model_block_modules,
        positions=list(range(-len(model_base.eoi_toks), 0)),
    )
    assert mean_diffs.shape == (
        len(model_base.eoi_toks),
        model_base.model.config.num_hidden_layers,
        model_base.model.config.hidden_size,
    )
    assert not mean_diffs.isnan().any()
    return mean_diffs


def generate_candidate_directions(cfg, model_base, ref_train, forget_train):
    # Generate and save candidate directions.
    mean_diffs = generate_directions(
        cfg,
        model_base,
        ref_train,
        forget_train,
    )
    print("[DEBUG] mean_diffs.shape =", mean_diffs.shape) # last token에서 갯수만큼 데리고 옴 : [5, 32, 4096]
    return mean_diffs
