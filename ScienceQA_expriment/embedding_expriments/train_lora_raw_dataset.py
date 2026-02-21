#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batched LoRA training on DATASET-1 (HF dataset on disk), not cache-jsonl.

Dataset-1 example schema (ScienceQA-style):
- text_input (optional, recommended if present)
- instruction or question
- choices: list[str] (len 2..4)
- answer: "A"/"B"/"C"/"D" or numeric label (0..3 or 1..4)

Loss:
  L = L_ce_wrong + lam_repel * L_repel

- L_ce_wrong:
    CE on next-token logits at last prompt position,
    restricted to valid letters (A.. up to #choices),
    with RANDOM WRONG target letter (!= GT).

- L_repel (optional):
    At steer_layer, pre-letter position (last token of prompt),
    repel from GT prototype direction:
      ReLU( cos(h_hat, mu_hat_gt) - margin_m )

Notes:
- One tokenizer call per batch
- One model forward per batch
- Works with variable-length prompts via attention_mask last index
"""

import argparse
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

LETTERS = ["A", "B", "C", "D"]
EPS = 1e-12


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm_answer_to_letter(a) -> str:
    if a is None:
        return ""
    s = str(a).strip().upper()
    # numeric label handling: 0..3 or 1..4
    if s.isdigit():
        i = int(s)
        if 0 <= i < len(LETTERS):
            return LETTERS[i]
        if 1 <= i <= len(LETTERS):
            return LETTERS[i - 1]
    for ch in s:
        if ch in LETTERS:
            return ch
    return ""


def valid_letters_from_choices(ex: Dict) -> List[str]:
    choices = ex.get("choices", [])
    if not isinstance(choices, list):
        return []
    n = len(choices)
    if n < 2:
        return []
    n = min(n, len(LETTERS))
    return LETTERS[:n]


def build_prompt_from_example(ex: Dict) -> str:
    """
    Prefer text_input if present; otherwise format question+choices.
    Must end with 'The answer is ' (trailing space important).
    """
    ti = ex.get("text_input", "")
    if isinstance(ti, str) and ti.strip():
        base = ti.strip()
        if "The answer is" in base:
            base = base.split("The answer is")[0].rstrip()
        return base + "\nThe answer is "

    q = ex.get("instruction") or ex.get("question")
    choices = ex.get("choices")
    if q is None or choices is None:
        raise KeyError(f"Missing fields. keys={list(ex.keys())}")

    valid = LETTERS[: min(len(choices), len(LETTERS))]
    s = f"Question: {q}\nChoices:\n"
    for i, c in enumerate(choices[: len(valid)]):
        s += f"({valid[i]}) {c}\n"
    s += f"Answer with a single letter among {', '.join(valid)}.\nThe answer is "
    return s


def get_global_letter_token_ids(tokenizer) -> torch.Tensor:
    """
    Return token ids for [" A"," B"," C"," D"] as a tensor [4].
    Requires each to be a single token.
    """
    ids = []
    for L in LETTERS:
        tok = tokenizer(f" {L}", add_special_tokens=False).input_ids
        if len(tok) != 1:
            raise ValueError(f'" {L}" is not a single token: {tok}')
        ids.append(tok[0])
    return torch.tensor(ids, dtype=torch.long)


def choose_wrong_target_random(valid_n: int, gt_local: int) -> int:
    if valid_n <= 1:
        return 0
    candidates = [i for i in range(valid_n) if i != gt_local]
    return random.choice(candidates)


# -------------------------
# Prototypes (for repel)
# -------------------------
def load_proto_unit(stats_path: str, min_count: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expects:
      x["stats"]["S_unit"] : [L, C(=4), D]
      x["stats"]["N"]      : [L, C]
    """
    x = torch.load(stats_path, map_location="cpu")
    S_unit = x["stats"]["S_unit"].float()
    N = x["stats"]["N"].long()
    proto = S_unit / (S_unit.norm(dim=-1, keepdim=True) + EPS)
    mask = (N >= min_count).unsqueeze(-1)
    proto = torch.where(mask, proto, torch.zeros_like(proto))
    return proto, N


# -------------------------
# LoRA helpers
# -------------------------
def apply_lora(model, target_modules: List[str], r: int, alpha: int, dropout: float):
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return get_peft_model(model, cfg)


def freeze_lora_outside_layer_window(model, layer_start: int, layer_end: int):
    for name, p in model.named_parameters():
        if "lora_" not in name:
            continue
        keep = False
        for li in range(layer_start, layer_end + 1):
            if f"model.layers.{li}." in name or f"layers.{li}." in name:
                keep = True
                break
        p.requires_grad = bool(keep)


def count_trainable_params(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


@torch.no_grad()
def topk_next_tokens_batch(logits_2d: torch.Tensor, tokenizer, k: int = 10):
    v, idx = torch.topk(logits_2d, k, dim=-1)
    out = []
    for b in range(logits_2d.size(0)):
        toks = [tokenizer.decode([int(i)]) for i in idx[b]]
        out.append(list(zip(toks, [float(x) for x in v[b]])))
    return out


# -------------------------
# Collate: build a filtered micro-batch from HF examples
# -------------------------
def make_collate_fn(proto_layer_cpu: torch.Tensor):
    """
    proto_layer_cpu: [4, D] on CPU (for existence check)
    """
    def collate(examples: List[Dict]):
        prompts: List[str] = []
        valid_n_list: List[int] = []
        gt_local_list: List[int] = []
        gt_global_list: List[int] = []
        tgt_local_list: List[int] = []

        for ex in examples:
            valid = valid_letters_from_choices(ex)
            if not valid:
                continue

            gt = norm_answer_to_letter(ex.get("answer"))
            if gt not in valid:
                continue

            gt_local = valid.index(gt)
            gt_global = LETTERS.index(gt)

            # prototype existence check
            mu = proto_layer_cpu[gt_global]
            if float(mu.norm()) < 1e-6:
                continue

            prompt = build_prompt_from_example(ex)
            vn = len(valid)
            tgt_local = choose_wrong_target_random(vn, gt_local)

            prompts.append(prompt)
            valid_n_list.append(vn)
            gt_local_list.append(gt_local)
            gt_global_list.append(gt_global)
            tgt_local_list.append(tgt_local)

        return {
            "prompts": prompts,
            "valid_n_list": valid_n_list,
            "gt_local_list": gt_local_list,
            "gt_global_list": gt_global_list,
            "tgt_local_list": tgt_local_list,
        }

    return collate


# -------------------------
# Training
# -------------------------
def train(
    model_name: str,
    dataset_path: str,
    split: str,
    out_dir: str,
    stats_path: str,
    steer_layer: int,
    min_count_proto: int,
    margin_m: float,
    lam_repel: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_layer_start: int,
    lora_layer_end: int,
    target_modules: List[str],
    lr: float,
    batch_size: int,
    grad_accum: int,
    max_steps: int,
    log_every: int,
    topk_log: int,
    device: str,
    dtype_str: str,
    seed: int,
    max_items: Optional[int],
    shuffle: bool,
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    if dtype_str == "float16":
        dtype = torch.float16
    elif dtype_str == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Load prototypes (CPU)
    proto_unit, counts = load_proto_unit(stats_path, min_count=min_count_proto)
    proto_layer_cpu = proto_unit[steer_layer].float()  # [4, D] on CPU

    # Load dataset-1 (HF on disk)
    ds = load_from_disk(dataset_path)
    if isinstance(ds, dict) or hasattr(ds, "keys"):  # DatasetDict
        if split not in ds:
            raise ValueError(f"split={split} not in dataset. available={list(ds.keys())}")
        split_ds = ds[split]
    else:
        # If user saved a plain Dataset (no dict), treat as one split
        split_ds = ds

    if max_items is not None and max_items > 0:
        max_items = min(max_items, len(split_ds))
        split_ds = split_ds.select(range(max_items))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model + LoRA
    backbone = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device.startswith("cuda") else None,
        torch_dtype=dtype,
    )
    model = apply_lora(backbone, target_modules=target_modules, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    freeze_lora_outside_layer_window(model, lora_layer_start, lora_layer_end)

    total_params, trainable_params = count_trainable_params(model)
    print(f"[MODEL] total_params={total_params:,} trainable_params={trainable_params:,}")
    print(f"[DATA] dataset_path={dataset_path} split={split} size={len(split_ds)}")
    print(f"[LoRA] layer_window=[{lora_layer_start},{lora_layer_end}] target_modules={target_modules}")
    print(f"[LOSS] L = CE(random-wrong) + lam_repel*ReLU(cos_gt - m) | lam_repel={lam_repel} m={margin_m}")
    print(f"[SEED] {seed}")

    # Move proto layer to model device
    model_device = model.device if hasattr(model, "device") else torch.device("cpu")
    proto_layer = proto_layer_cpu.to(model_device)  # [4, D]

    # Precompute global letter token ids [" A"," B"," C"," D"]
    letter_ids_global = get_global_letter_token_ids(tokenizer).to(model_device)  # [4]

    # DataLoader
    collate_fn = make_collate_fn(proto_layer_cpu=proto_layer_cpu)
    loader = DataLoader(
        split_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
    )

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    model.train()
    step = 0
    optim.zero_grad(set_to_none=True)

    for epoch in range(10_000_000):
        for pack in loader:
            prompts = pack["prompts"]
            valid_n_list = pack["valid_n_list"]
            gt_local_list = pack["gt_local_list"]
            gt_global_list = pack["gt_global_list"]
            tgt_local_list = pack["tgt_local_list"]

            used = len(prompts)
            if used == 0:
                continue

            # Tokenize batch
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(model_device)
            attention_mask = enc["attention_mask"].to(model_device)

            # last index per sample
            last_idx = (attention_mask.sum(dim=1) - 1).clamp(min=0)  # [B]
            B = input_ids.size(0)
            batch_arange = torch.arange(B, device=model_device)

            need_hidden = lam_repel > 0.0
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=need_hidden,
            )

            # logits at last position: [B, V]
            logits_last = out.logits[batch_arange, last_idx, :]  # [B, V]

            # letter logits A..D: [B,4]
            logits_letters = logits_last.index_select(dim=1, index=letter_ids_global)

            valid_n = torch.tensor(valid_n_list, device=model_device, dtype=torch.long)  # [B]
            pos = torch.arange(4, device=model_device).unsqueeze(0).expand(B, 4)
            mask_valid = pos < valid_n.unsqueeze(1)

            neg_inf = torch.finfo(logits_letters.dtype).min
            logits_letters_masked = torch.where(mask_valid, logits_letters, torch.full_like(logits_letters, neg_inf))
            logp_letters = torch.log_softmax(logits_letters_masked, dim=-1)

            tgt_local = torch.tensor(tgt_local_list, device=model_device, dtype=torch.long)
            gt_local = torch.tensor(gt_local_list, device=model_device, dtype=torch.long)
            gt_global = torch.tensor(gt_global_list, device=model_device, dtype=torch.long)

            # CE(wrong): -log p(tgt)
            loss_ce = (-logp_letters[batch_arange, tgt_local]).mean()

            # Repel (optional)
            if need_hidden:
                # hidden_states[0]=emb, so layer i output is hidden_states[i+1]
                h = out.hidden_states[steer_layer + 1][batch_arange, last_idx, :].float()  # [B,D]
                h_hat = F.normalize(h, dim=-1)

                mu = proto_layer.index_select(dim=0, index=gt_global).float()  # [B,D]
                mu_hat = F.normalize(mu, dim=-1)

                cos_gt = (h_hat * mu_hat).sum(dim=-1)  # [B]
                loss_repel = F.relu(cos_gt - margin_m).mean()
            else:
                cos_gt = None
                loss_repel = torch.tensor(0.0, device=model_device)

            loss = loss_ce + lam_repel * loss_repel

            # backward (grad accumulation)
            (loss / grad_accum).backward()
            step += 1

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)

            # Logging
            if step % log_every == 0:
                with torch.no_grad():
                    p_letters = torch.softmax(logits_letters_masked, dim=-1)
                    p_gt = p_letters[batch_arange, gt_local]
                    p_tgt = p_letters[batch_arange, tgt_local]

                    logp_full = torch.log_softmax(logits_last, dim=-1)
                    logp_letters_full = logp_full.index_select(dim=1, index=letter_ids_global)
                    logp_letters_full_masked = torch.where(
                        mask_valid, logp_letters_full, torch.full_like(logp_letters_full, neg_inf)
                    )
                    p_valid_mass = torch.exp(torch.logsumexp(logp_letters_full_masked, dim=-1))

                    topk_list = topk_next_tokens_batch(logits_last.detach().float().cpu(), tokenizer, k=topk_log)

                    j = B - 1
                    gt_chr = LETTERS[int(gt_local_list[j])]
                    tgt_chr = LETTERS[int(tgt_local_list[j])]
                    vn = int(valid_n_list[j])
                    valid_letters = LETTERS[:vn]
                    print(f"[TRAIN] step={step} loss={float(loss.item()):.6f} used={used} (batch={batch_size})")
                    print(
                        f"  gt={gt_chr} tgt={tgt_chr} valid={valid_letters} | "
                        f"lm p_gt={float(p_gt[j]):.4f} p_tgt={float(p_tgt[j]):.4f} "
                        f"p_valid_mass={float(p_valid_mass[j]):.4f}"
                        + (f" | cos_gt={float(cos_gt[j]):.4f}" if cos_gt is not None else "")
                    )
                    if need_hidden:
                        print(f"  losses: ce={float(loss_ce.item()):.4f} repel={float(loss_repel.item()):.4f} (lam_repel={lam_repel})")
                    else:
                        print(f"  losses: ce={float(loss_ce.item()):.4f} (lam_repel={lam_repel})")
                    print(f"  prompt_tail: {repr(prompts[j][-220:])}")
                    print(f"  top{topk_log} next tokens: {topk_list[j]}")

            # Save + exit
            if step >= max_steps:
                save_path = Path(out_dir) / "lora_adapter"
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"[SAVED] {save_path.resolve()}")
                return


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--dataset_path", type=str, required=True, help="e.g., data/scienceqa/scienceqa_biology_all.hf")
    ap.add_argument("--split", type=str, default="train", help="train/validation/test")
    ap.add_argument("--max_items", type=int, default=0, help="0=all, else cap items for debug")
    ap.add_argument("--shuffle", action="store_true", help="shuffle HF dataset in DataLoader")

    ap.add_argument("--stats", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="out_ds1_cewrong_repel")

    ap.add_argument("--steer_layer", type=int, default=17)
    ap.add_argument("--min_count_proto", type=int, default=50)

    ap.add_argument("--margin_m", type=float, default=0.05)
    ap.add_argument("--lam_repel", type=float, default=0.0)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_layer_start", type=int, default=14)
    ap.add_argument("--lora_layer_end", type=int, default=19)

    ap.add_argument("--target_modules", type=str, default="q_proj,v_proj")

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=2000)

    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--topk_log", type=int, default=10)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    ap.add_argument("--seed", type=int, default=0)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    max_items = None if args.max_items <= 0 else args.max_items

    train(
        model_name=args.model,
        dataset_path=args.dataset_path,
        split=args.split,
        out_dir=args.out_dir,
        stats_path=args.stats,
        steer_layer=args.steer_layer,
        min_count_proto=args.min_count_proto,
        margin_m=args.margin_m,
        lam_repel=args.lam_repel,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layer_start=args.lora_layer_start,
        lora_layer_end=args.lora_layer_end,
        target_modules=target_modules,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        log_every=args.log_every,
        topk_log=args.topk_log,
        device=args.device,
        dtype_str=args.dtype,
        seed=args.seed,
        max_items=max_items,
        shuffle=args.shuffle,
    )
