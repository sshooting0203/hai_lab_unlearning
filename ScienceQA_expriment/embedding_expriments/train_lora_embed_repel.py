#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LoRA training with EMBEDDING-ONLY losses to MINIMIZE accuracy (make model wrong).

Loss = lambda_notgt * L_notgt + lambda_repel * L_repel (+ optional beta_norm * L_norm)

- L_notgt = -log(1 - p_y) where p_y is GT prob under prototype-softmax at a chosen layer/position.
- L_repel = ReLU( cos(h_hat, mu_hat_gt) - m )

We compute h at PRE-LETTER position (last token of prompt "... The answer is")
from a chosen steering layer (default 17). We do NOT use token-level CE.

Inputs required:
- train_pred_cache.jsonl / validation_pred_cache.jsonl created earlier
- stats_pre_full.pt (or compatible stats .pt) containing:
    payload["stats"]["S_unit"] : [L,C,D]
    payload["stats"]["N"]      : [L,C]
(we use proto_unit = normalize(S_unit) as class prototypes for A/B/C/D)

Run example:
python train_lora_embed_losses.py \
  --model meta-llama/Llama-3.1-8B \
  --train_jsonl data/scienceqa/splits_biology_pred/train_pred_cache.jsonl \
  --val_jsonl data/scienceqa/splits_biology_pred/validation_pred_cache.jsonl \
  --stats stats_pre_full.pt \
  --out_dir out_embed_notgt_repel \
  --steer_layer 17 \
  --min_count_proto 50 \
  --tau 0.07 \
  --margin_m 0.05 \
  --lambda_notgt 1.0 \
  --lambda_repel 1.0 \
  --beta_norm 1e-4 \
  --target_modules q_proj,v_proj \
  --lora_layer_start 14 --lora_layer_end 19 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --lr 1e-4 --batch_size 4 --grad_accum 4 \
  --max_steps 2000 --eval_every 200 \
  --dtype float16
"""

import argparse
import json
import os
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
# Text / label utilities
# -------------------------
def norm_answer_to_letter(a) -> str:
    """Normalize record['answer'] into A/B/C/D (or '' if unknown)."""
    if a is None:
        return ""
    s = str(a).strip().upper()
    if s.isdigit():
        i = int(s)
        if 0 <= i < len(LETTERS):
            return LETTERS[i]
        if 0 <= i < 5:
            return "ABCDE"[i]
    for ch in s:
        if ch in LETTERS:
            return ch
        if ch in "ABCDE":
            return ch
    return ""


def valid_letters_from_choices(r: Dict) -> List[str]:
    choices = r.get("choices", [])
    if not isinstance(choices, list):
        return []
    n = len(choices)
    if n < 2:
        return []
    if n > 4:
        n = 4
    return LETTERS[:n]


def build_prompt_from_record(r: Dict, strip_existing_answer_is: bool = True) -> str:
    """
    Prefer r['text_input'] and ensure it ends with 'The answer is'.
    """
    base = r.get("text_input", "")
    if isinstance(base, str) and base.strip():
        base = base.strip()
        if strip_existing_answer_is and "The answer is" in base:
            base = base.split("The answer is")[0].rstrip()
        if base.endswith("The answer is"):
            return base
        return base + "\nThe answer is"

    # fallback formatting
    q = r.get("instruction") or r.get("question")
    choices = r.get("choices", [])
    if q is None or not isinstance(choices, list) or len(choices) < 2:
        raise KeyError(f"Missing text_input/question/choices. keys={list(r.keys())}")

    valid = LETTERS[:len(choices)]
    s = f"Question: {q}\nChoices:\n"
    for i, c in enumerate(choices):
        s += f"({valid[i]}) {c}\n"
    s += f"Answer with a single letter among {', '.join(valid)}.\nThe answer is"
    return s


def letter_token_ids(tokenizer, valid_letters: List[str]) -> torch.Tensor:
    ids = []
    for L in valid_letters:
        tok = tokenizer(f" {L}", add_special_tokens=False).input_ids
        if len(tok) != 1:
            raise ValueError(f'" {L}" is not a single token for this tokenizer: {tok}')
        ids.append(tok[0])
    return torch.tensor(ids, dtype=torch.long)


# -------------------------
# Stats / prototypes
# -------------------------
def load_proto_unit(stats_path: str, min_count: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load proto unit vectors from stats pt.
    Returns:
      proto_unit: [L,C,D] float32 CPU
      counts:     [L,C] long CPU
    """
    x = torch.load(stats_path, map_location="cpu")
    S_unit = x["stats"]["S_unit"].float()  # [L,C,D]
    N = x["stats"]["N"].long()             # [L,C]

    proto = S_unit / (S_unit.norm(dim=-1, keepdim=True) + EPS)  # mean direction
    mask = (N >= min_count).unsqueeze(-1)
    proto = torch.where(mask, proto, torch.zeros_like(proto))
    return proto, N


# -------------------------
# Dataset
# -------------------------
class JsonlCacheDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_records(batch: List[Dict]):
    return batch


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
    """
    Freeze LoRA params outside the specified transformer layer window.
    Works for typical HF LLaMA naming: 'model.layers.{i}.'.
    """
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


# -------------------------
# Monitoring: next-token accuracy (we want LOW)
# -------------------------
@torch.no_grad()
def eval_next_token_acc(model, tokenizer, records: List[Dict], device: str, max_items: int = 200) -> float:
    model.eval()
    correct = 0
    total = 0
    for r in records[:max_items]:
        valid = valid_letters_from_choices(r)
        if not valid:
            continue
        gt = norm_answer_to_letter(r.get("answer"))
        if gt not in valid:
            continue

        prompt = build_prompt_from_record(r, strip_existing_answer_is=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        out = model(**inputs)
        logits = out.logits[0, -1, :]  # next token logits

        cand_ids = letter_token_ids(tokenizer, valid).to(device)
        scores = logits[cand_ids]
        pred = valid[int(torch.argmax(scores).item())]

        total += 1
        correct += int(pred == gt)

    return correct / max(total, 1)


# -------------------------
# Training
# -------------------------
def train(
    model_name: str,
    train_jsonl: str,
    val_jsonl: str,
    stats_path: str,
    out_dir: str,
    steer_layer: int,
    min_count_proto: int,
    tau: float,
    margin_m: float,
    lambda_notgt: float,
    lambda_repel: float,
    beta_norm: float,
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
    eval_every: int,
    device: str,
    dtype_str: str,
):
    os.makedirs(out_dir, exist_ok=True)
    device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    # dtype
    if dtype_str == "float16":
        dtype = torch.float16
    elif dtype_str == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device.startswith("cuda") else None,
        dtype=dtype,
    )
    base_model.eval()

    # Apply LoRA
    model = apply_lora(base_model, target_modules=target_modules, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    freeze_lora_outside_layer_window(model, lora_layer_start, lora_layer_end)

    total_params, trainable_params = count_trainable_params(model)
    print(f"[MODEL] total_params={total_params:,} trainable_params={trainable_params:,}")
    print(f"[LoRA] layer_window=[{lora_layer_start},{lora_layer_end}] target_modules={target_modules}")

    # Load prototypes
    proto_unit, counts = load_proto_unit(stats_path, min_count=min_count_proto)
    proto_layer = proto_unit[steer_layer].float()  # [C,D] CPU

    # Datasets
    train_ds = JsonlCacheDataset(train_jsonl)
    val_ds = JsonlCacheDataset(val_jsonl) if val_jsonl and Path(val_jsonl).exists() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_records
    )

    # Optimizer
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    # Baseline eval
    if val_ds is not None:
        acc0 = eval_next_token_acc(model, tokenizer, val_ds.items, device=device, max_items=200)
        print(f"[EVAL] baseline val next-token acc (we want LOW after training): {acc0:.4f}")

    model.train()
    step = 0
    optim.zero_grad(set_to_none=True)

    for epoch in range(10_000_000):
        for batch in train_loader:
            loss_sum = 0.0
            used = 0

            for r in batch:
                valid = valid_letters_from_choices(r)
                if not valid:
                    continue

                gt = norm_answer_to_letter(r.get("answer"))
                if gt not in valid:
                    continue

                gt_global = LETTERS.index(gt)  # 0..3
                mu_gt = proto_layer[gt_global]  # [D] CPU
                if float(mu_gt.norm()) < 1e-6:
                    continue

                prompt = build_prompt_from_record(r, strip_existing_answer_is=True)
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

                if device.startswith("cuda"):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                else:
                    inputs = {k: v.cpu() for k, v in inputs.items()}

                # PRE-LETTER position = last token of prompt
                pos = inputs["input_ids"].shape[1] - 1

                # forward
                out = model(**inputs, output_hidden_states=True)
                hs = out.hidden_states

                h = hs[steer_layer + 1][0, pos, :]  # [D]
                h_f = h.float()
                h_hat = F.normalize(h_f, dim=-1)

                # Build prototypes for *valid letters* only
                n_valid = len(valid)
                proto_valid = proto_layer[:n_valid].to(h_hat.device).float()
                proto_valid = F.normalize(proto_valid, dim=-1)  # [n_valid, D]

                # local gt index in valid set
                gt_local = valid.index(gt)

                # -------- L_not-gt = -log(1 - p_y) --------
                logits_proto = (h_hat.unsqueeze(0) @ proto_valid.t()) / tau  # [1,n_valid]
                p = torch.softmax(logits_proto, dim=-1)[0]                   # [n_valid]
                p_y = p[gt_local]
                loss_notgt = -torch.log(torch.clamp(1.0 - p_y, min=1e-6))

                # -------- L_repel = ReLU(cos(h, mu_gt) - m) --------
                mu_hat_gt = proto_valid[gt_local]  # [D]
                cos = torch.dot(h_hat, mu_hat_gt)
                loss_repel = F.relu(cos - margin_m)

                # -------- optional norm stabilizer --------
                if beta_norm > 0:
                    r0 = h_f.detach().norm(p=2)
                    loss_norm = (h_f.norm(p=2) - r0) ** 2
                else:
                    loss_norm = 0.0

                loss = lambda_notgt * loss_notgt + lambda_repel * loss_repel + beta_norm * loss_norm
                loss_sum = loss_sum + loss
                used += 1

            if used == 0:
                continue

            loss_mean = loss_sum / used
            (loss_mean / grad_accum).backward()
            step += 1

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)

            if step % 50 == 0:
                print(f"[TRAIN] step={step} loss={loss_mean.item():.6f} used={used}")

            if val_ds is not None and step % eval_every == 0:
                acc = eval_next_token_acc(model, tokenizer, val_ds.items, device=device, max_items=200)
                print(f"[EVAL] step={step} val next-token acc (we want LOW): {acc:.4f}")

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
    ap.add_argument("--val_jsonl", type=str, default="")
    ap.add_argument("--stats", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="out_embed_notgt_repel")

    ap.add_argument("--steer_layer", type=int, default=17)
    ap.add_argument("--min_count_proto", type=int, default=50)

    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--margin_m", type=float, default=0.05)
    ap.add_argument("--lambda_notgt", type=float, default=1.0)
    ap.add_argument("--lambda_repel", type=float, default=1.0)
    ap.add_argument("--beta_norm", type=float, default=1e-4)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_layer_start", type=int, default=14)
    ap.add_argument("--lora_layer_end", type=int, default=19)

    ap.add_argument("--target_modules", type=str, default="q_proj,v_proj",
                    help="comma-separated: q_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=200)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]

    train(
        model_name=args.model,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        stats_path=args.stats,
        out_dir=args.out_dir,
        steer_layer=args.steer_layer,
        min_count_proto=args.min_count_proto,
        tau=args.tau,
        margin_m=args.margin_m,
        lambda_notgt=args.lambda_notgt,
        lambda_repel=args.lambda_repel,
        beta_norm=args.beta_norm,
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
        eval_every=args.eval_every,
        device=args.device,
        dtype_str=args.dtype,
    )
