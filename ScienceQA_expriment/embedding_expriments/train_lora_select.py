#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batched LoRA training (random wrong-target CE):
  L = L_ce_wrong + lambda_repel * L_repel

- L_ce_wrong:
    - Use next-token logits at the last prompt position.
    - Restrict to valid letters per sample (A.. up to #choices).
    - Choose random WRONG letter (!= GT) among valid letters, and maximize it via CE.

- L_repel (optional; only if lam_repel > 0):
    - At steer_layer, pre-letter position (last token of the prompt),
      repel from GT prototype direction:
        ReLU( cos(h_hat, mu_hat_gt) - margin_m )

This version is truly batched:
- One tokenizer call per batch
- One model forward per batch

Notes:
- Works with variable-length prompts by using attention_mask to find last index.
- Uses global letter token ids for [" A"," B"," C"," D"], then masks invalid letters per sample.
- UPDATED: Instead of a contiguous layer window, you can specify arbitrary layers via --lora_layers
  e.g., "14,16,19" or "14-16,19,22-23"
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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


def parse_layer_list(s: str) -> List[int]:
    """
    Parse layer spec like:
      "14,16,19" or "14-19" or "14-16,19,22-23"
    Returns sorted unique layer indices.
    """
    if s is None or str(s).strip() == "":
        return []
    s = s.strip()
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a.strip())
            b = int(b.strip())
            lo, hi = (a, b) if a <= b else (b, a)
            for x in range(lo, hi + 1):
                out.add(x)
        else:
            out.add(int(part))
    return sorted(out)


def norm_answer_to_letter(a) -> str:
    if a is None:
        return ""
    s = str(a).strip().upper()
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


def valid_letters_from_choices(r: Dict) -> List[str]:
    choices = r.get("choices", [])
    if not isinstance(choices, list):
        return []
    n = len(choices)
    if n < 2:
        return []
    n = min(n, len(LETTERS))
    return LETTERS[:n]


def build_prompt_from_record(r: Dict) -> str:
    base = r.get("text_input", "")
    if isinstance(base, str) and base.strip():
        base = base.strip()
        if "The answer is" in base:
            base = base.split("The answer is")[0].rstrip()
        return base + "\nThe answer is "  # trailing space is important
    raise KeyError("Record missing text_input (expected in your cache jsonl).")


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
    """
    valid_n: number of valid letters among A..D (2..4)
    gt_local: index in [0, valid_n)
    returns: index in [0, valid_n) and != gt_local
    """
    if valid_n <= 1:
        return 0
    candidates = [i for i in range(valid_n) if i != gt_local]
    return random.choice(candidates)


# -------------------------
# Prototypes (for repel)
# -------------------------
def load_proto_unit(stats_path: str, min_count: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.load(stats_path, map_location="cpu")
    S_unit = x["stats"]["S_unit"].float()  # [L,C,D]
    N = x["stats"]["N"].long()             # [L,C]
    proto = S_unit / (S_unit.norm(dim=-1, keepdim=True) + EPS)
    mask = (N >= min_count).unsqueeze(-1)
    proto = torch.where(mask, proto, torch.zeros_like(proto))
    return proto, N


# -------------------------
# Dataset
# -------------------------
class JsonlCacheDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


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


def freeze_lora_outside_layers(model, keep_layers: List[int]):
    """
    keep_layers: list of layer indices to train (e.g., [14, 16, 19])
    Sets requires_grad=True only for LoRA params inside those layers.
    """
    keep_set = set(int(x) for x in keep_layers)
    for name, p in model.named_parameters():
        if "lora_" not in name:
            continue

        keep = False
        # LLaMA-style: model.layers.{i}.
        # Some variants: layers.{i}.
        for li in keep_set:
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


# -------------------------
# Diagnostics
# -------------------------
@torch.no_grad()
def topk_next_tokens_batch(logits_2d: torch.Tensor, tokenizer, k: int = 10):
    """
    logits_2d: [B, V] (already selected at last positions)
    returns: list of list of (tok, logit) per sample
    """
    v, idx = torch.topk(logits_2d, k, dim=-1)
    out = []
    for b in range(logits_2d.size(0)):
        toks = [tokenizer.decode([int(i)]) for i in idx[b]]
        out.append(list(zip(toks, [float(x) for x in v[b]])))
    return out


# -------------------------
# Batched training
# -------------------------
def train(
    model_name: str,
    train_jsonl: str,
    stats_path: str,
    out_dir: str,
    steer_layer: int,
    min_count_proto: int,
    tau_unused: float,          # kept for CLI compatibility; not used here
    margin_m: float,
    lam_repel: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_layers: List[int],
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

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load prototypes (CPU)
    proto_unit, counts = load_proto_unit(stats_path, min_count=min_count_proto)
    proto_layer_cpu = proto_unit[steer_layer].float()  # [C=4, D]

    # LoRA model
    backbone = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device.startswith("cuda") else None,
        torch_dtype=dtype,
    )
    model = apply_lora(backbone, target_modules=target_modules, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    if not lora_layers:
        raise ValueError("Empty --lora_layers. Provide something like '14,16,19' or '14-19'.")

    freeze_lora_outside_layers(model, lora_layers)

    total_params, trainable_params = count_trainable_params(model)
    print(f"[MODEL] total_params={total_params:,} trainable_params={trainable_params:,}")
    print(f"[LoRA] layers={sorted(lora_layers)} target_modules={target_modules}")
    print(f"[LOSS] L = CE(random-wrong) + lam_repel*ReLU(cos_gt - m) | lam_repel={lam_repel} m={margin_m}")
    print(f"[SEED] {seed}")

    # Determine device used by the model's outputs (more reliable under device_map="auto")
    # We'll set tensors on this device for safety.
    # Note: model.device may be misleading when sharded; using output device is safest.
    dummy_device = None
    try:
        dummy_device = next(model.parameters()).device
    except StopIteration:
        dummy_device = torch.device("cpu")

    # Move prototype layer and letter ids to a working device (will still be ok if outputs differ;
    # we will re-align to logits device after forward if needed)
    proto_layer = proto_layer_cpu.to(dummy_device)  # [4, D]
    letter_ids_global = get_global_letter_token_ids(tokenizer).to(dummy_device)  # [4]

    # Data
    train_ds = JsonlCacheDataset(train_jsonl)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x,
    )

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    model.train()
    step = 0
    optim.zero_grad(set_to_none=True)

    for epoch in range(10_000_000):
        for raw_batch in train_loader:
            # --------
            # Filter + prepare batch metadata (on CPU)
            # --------
            prompts: List[str] = []
            valid_n_list: List[int] = []
            gt_local_list: List[int] = []
            gt_global_list: List[int] = []
            tgt_local_list: List[int] = []

            for r in raw_batch:
                valid = valid_letters_from_choices(r)
                if not valid:
                    continue
                gt = norm_answer_to_letter(r.get("answer"))
                if gt not in valid:
                    continue

                gt_local = valid.index(gt)
                gt_global = LETTERS.index(gt)

                # proto existence check (CPU)
                mu = proto_layer_cpu[gt_global]
                if float(mu.norm()) < 1e-6:
                    continue

                prompt = build_prompt_from_record(r)

                vn = len(valid)
                tgt_local = choose_wrong_target_random(vn, gt_local)

                prompts.append(prompt)
                valid_n_list.append(vn)
                gt_local_list.append(gt_local)
                gt_global_list.append(gt_global)
                tgt_local_list.append(tgt_local)

            used = len(prompts)
            if used == 0:
                continue

            # --------
            # Batched tokenization
            # --------
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )

            # We will place inputs on the same device as the model's first parameter.
            # This works in most sharded setups; model forward will handle internal dispatch.
            input_ids = enc["input_ids"].to(dummy_device)
            attention_mask = enc["attention_mask"].to(dummy_device)

            # last index per sample = sum(attn)-1
            last_idx = attention_mask.sum(dim=1) - 1  # [B]
            last_idx = last_idx.clamp(min=0)

            # --------
            # Forward (optionally with hidden states)
            # --------
            need_hidden = lam_repel > 0.0
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=need_hidden,
            )

            # Align helper tensors to logits device (robust under device_map="auto")
            logits_device = out.logits.device
            if proto_layer.device != logits_device:
                proto_layer = proto_layer.to(logits_device)
            if letter_ids_global.device != logits_device:
                letter_ids_global = letter_ids_global.to(logits_device)

            # logits at last position for each sample: [B, V]
            B = input_ids.size(0)
            batch_arange = torch.arange(B, device=logits_device)
            last_idx = last_idx.to(logits_device)
            logits_last = out.logits[batch_arange, last_idx, :]  # [B, V]

            # extract letter logits for A,B,C,D: [B, 4]
            logits_letters = logits_last.index_select(dim=1, index=letter_ids_global)  # [B,4]

            # mask invalid letters per sample (valid_n can be 2..4)
            valid_n = torch.tensor(valid_n_list, device=logits_device, dtype=torch.long)  # [B]
            pos = torch.arange(4, device=logits_device).unsqueeze(0).expand(B, 4)  # [B,4]
            mask_valid = pos < valid_n.unsqueeze(1)  # [B,4] bool

            neg_inf = torch.finfo(logits_letters.dtype).min
            logits_letters_masked = torch.where(
                mask_valid, logits_letters, torch.full_like(logits_letters, neg_inf)
            )

            logp_letters = torch.log_softmax(logits_letters_masked, dim=-1)  # [B,4]

            tgt_local = torch.tensor(tgt_local_list, device=logits_device, dtype=torch.long)  # [B]
            gt_local = torch.tensor(gt_local_list, device=logits_device, dtype=torch.long)    # [B]
            gt_global = torch.tensor(gt_global_list, device=logits_device, dtype=torch.long)  # [B]

            # CE(wrong): -log p(tgt)
            loss_ce_vec = -logp_letters[batch_arange, tgt_local]  # [B]
            loss_ce = loss_ce_vec.mean()

            # Repel (optional)
            if need_hidden:
                # hidden_states: list length (n_layers+1). layers are 1-indexed there.
                h = out.hidden_states[steer_layer + 1][batch_arange, last_idx, :].float()  # [B,D]
                h_hat = F.normalize(h, dim=-1)

                mu = proto_layer.index_select(dim=0, index=gt_global).float()  # [B,D]
                mu_hat = F.normalize(mu, dim=-1)

                cos_gt = (h_hat * mu_hat).sum(dim=-1)  # [B]
                loss_repel_vec = F.relu(cos_gt - margin_m)
                loss_repel = loss_repel_vec.mean()
            else:
                cos_gt = None
                loss_repel = torch.tensor(0.0, device=logits_device)

            loss = loss_ce + lam_repel * loss_repel

            # backward (grad accumulation)
            (loss / grad_accum).backward()
            step += 1

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)

            # --------
            # Logging
            # --------
            if step % log_every == 0:
                with torch.no_grad():
                    p_letters = torch.softmax(logits_letters_masked, dim=-1)  # [B,4]
                    p_gt = p_letters[batch_arange, gt_local]  # [B]
                    p_tgt = p_letters[batch_arange, tgt_local]  # [B]

                    # valid mass in full vocab (sum over valid letter ids)
                    logp_full = torch.log_softmax(logits_last, dim=-1)  # [B,V]
                    logp_letters_full = logp_full.index_select(dim=1, index=letter_ids_global)  # [B,4]
                    logp_letters_full_masked = torch.where(
                        mask_valid, logp_letters_full, torch.full_like(logp_letters_full, neg_inf)
                    )
                    p_valid_mass = torch.exp(torch.logsumexp(logp_letters_full_masked, dim=-1))  # [B]

                    topk_list = topk_next_tokens_batch(
                        logits_last.detach().float().cpu(), tokenizer, k=topk_log
                    )

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
                        print(
                            f"  losses: ce={float(loss_ce.item()):.4f} "
                            f"repel={float(loss_repel.item()):.4f} (lam_repel={lam_repel})"
                        )
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
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--stats", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="out_cewrong_repel")

    ap.add_argument("--steer_layer", type=int, default=17)
    ap.add_argument("--min_count_proto", type=int, default=50)

    ap.add_argument("--margin_m", type=float, default=0.05)
    ap.add_argument("--lam_repel", type=float, default=0.0)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument(
        "--lora_layers",
        type=str,
        default="14-19",
        help='Layers to train, e.g. "14,16,19" or "14-16,19,22-23".',
    )

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
    lora_layers = parse_layer_list(args.lora_layers)

    train(
        model_name=args.model,
        train_jsonl=args.train_jsonl,
        stats_path=args.stats,
        out_dir=args.out_dir,
        steer_layer=args.steer_layer,
        min_count_proto=args.min_count_proto,
        tau_unused=0.0,
        margin_m=args.margin_m,
        lam_repel=args.lam_repel,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=lora_layers,
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
    )
