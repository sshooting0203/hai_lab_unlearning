#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimal LoRA training:
  L = L_ce_wrong + lambda_repel * L_repel

- L_ce_wrong: CE on next-token logits restricted to valid letters, target = wrong letter
             chosen from BASE (top-2 if base top-1 == GT else base top-1).
- L_repel: ReLU( cos(h_hat, mu_hat_gt) - margin_m ) at steer_layer, pre-letter position.

Includes essential diagnostics (every log_every):
- chosen target (wrong) vs gt
- lm probs on letters: p_gt, p_tgt, p_valid_mass
- proto cosine: cos_gt
- hidden norm and delta vs base
- topk next tokens
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
# Utilities
# -------------------------
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
        # trailing space is important
        return base + "\nThe answer is "
    raise KeyError("Record missing text_input (expected in your cache jsonl).")


def letter_token_ids(tokenizer, valid_letters: List[str]) -> torch.Tensor:
    ids = []
    for L in valid_letters:
        tok = tokenizer(f" {L}", add_special_tokens=False).input_ids
        if len(tok) != 1:
            raise ValueError(f'" {L}" is not a single token: {tok}')
        ids.append(tok[0])
    return torch.tensor(ids, dtype=torch.long)


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
# Diagnostics
# -------------------------
@torch.no_grad()
def topk_next_tokens(logits_1d: torch.Tensor, tokenizer, k: int = 10):
    v, idx = torch.topk(logits_1d, k)
    toks = [tokenizer.decode([int(i)]) for i in idx]
    return list(zip(toks, [float(x) for x in v]))


def choose_wrong_target_from_base(base_logits_letters: torch.Tensor, gt_local: int) -> int:
    order = torch.argsort(base_logits_letters, descending=True)
    top1 = int(order[0].item())
    if top1 != gt_local:
        return top1
    if len(order) >= 2:
        return int(order[1].item())
    return 0 if gt_local != 0 else 1


# -------------------------
# Training
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
):
    os.makedirs(out_dir, exist_ok=True)
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

    # BASE (frozen) for choosing wrong target
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device.startswith("cuda") else None,
        torch_dtype=dtype,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    # LoRA model
    backbone = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device.startswith("cuda") else None,
        torch_dtype=dtype,
    )
    model = apply_lora(backbone, target_modules=target_modules, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    freeze_lora_outside_layer_window(model, lora_layer_start, lora_layer_end)

    total_params, trainable_params = count_trainable_params(model)
    print(f"[MODEL] total_params={total_params:,} trainable_params={trainable_params:,}")
    print(f"[LoRA] layer_window=[{lora_layer_start},{lora_layer_end}] target_modules={target_modules}")
    print(f"[LOSS] L = CE(wrong) + lam_repel*ReLU(cos_gt - m) | lam_repel={lam_repel} m={margin_m}")

    # Prototypes
    proto_unit, counts = load_proto_unit(stats_path, min_count=min_count_proto)
    proto_layer_cpu = proto_unit[steer_layer].float()  # [C,D] CPU

    # Data
    train_ds = JsonlCacheDataset(train_jsonl)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_records
    )

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    model.train()
    step = 0
    optim.zero_grad(set_to_none=True)

    for epoch in range(10_000_000):
        for batch in train_loader:
            loss_sum = 0.0
            used = 0
            last_log_pack = None

            for r in batch:
                valid = valid_letters_from_choices(r)
                if not valid:
                    continue

                gt = norm_answer_to_letter(r.get("answer"))
                if gt not in valid:
                    continue
                gt_local = valid.index(gt)
                gt_global = LETTERS.index(gt)

                mu_gt = proto_layer_cpu[gt_global]
                if float(mu_gt.norm()) < 1e-6:
                    continue

                prompt = build_prompt_from_record(r)
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

                if device.startswith("cuda"):
                    inputs_lora = {k: v.to(model.device) for k, v in inputs.items()}
                    inputs_base = {k: v.to(base_model.device) for k, v in inputs.items()}
                else:
                    inputs_lora = {k: v.cpu() for k, v in inputs.items()}
                    inputs_base = inputs_lora

                pos = inputs_lora["input_ids"].shape[1] - 1

                # BASE forward for wrong-target selection (no grad)
                with torch.no_grad():
                    out_b = base_model(**inputs_base)
                    logits_b = out_b.logits[0, -1, :]
                    cand_ids_b = letter_token_ids(tokenizer, valid).to(logits_b.device)
                    logits_b_letters = logits_b[cand_ids_b]
                    tgt_local = choose_wrong_target_from_base(logits_b_letters, gt_local)

                    # also get base hidden for delta norm logging
                    out_b_h = base_model(**inputs_base, output_hidden_states=True)
                    h_b = out_b_h.hidden_states[steer_layer + 1][0, pos, :].float()

                # LoRA forward (grad)
                out = model(**inputs_lora, output_hidden_states=True)

                # --- CE(wrong) on next-token restricted to letters ---
                logits = out.logits[0, -1, :]
                cand_ids = letter_token_ids(tokenizer, valid).to(logits.device)
                logits_letters = logits[cand_ids]
                logp_letters = torch.log_softmax(logits_letters, dim=-1)
                loss_ce = -logp_letters[tgt_local]

                # --- repel in proto space at steer_layer ---
                h = out.hidden_states[steer_layer + 1][0, pos, :].float()
                h_hat = F.normalize(h, dim=-1)
                mu_hat_gt = F.normalize(proto_layer_cpu[gt_global].to(h.device).float(), dim=-1)
                cos_gt = torch.dot(h_hat, mu_hat_gt)
                loss_repel = F.relu(cos_gt - margin_m)

                loss = loss_ce + lam_repel * loss_repel

                loss_sum = loss_sum + loss
                used += 1

                # diagnostics pack
                with torch.no_grad():
                    p_letters = torch.softmax(logits_letters, dim=-1)
                    p_gt = float(p_letters[gt_local].item())
                    p_tgt = float(p_letters[tgt_local].item())
                    # valid mass in full vocab
                    logp_full = torch.log_softmax(logits, dim=-1)
                    p_valid_mass = float(torch.exp(torch.logsumexp(logp_full[cand_ids], dim=0)).item())
                    # norms
                    h_norm = float(h.norm(p=2).item())
                    h_b_norm = float(h_b.norm(p=2).item())
                    delta_norm = float((h - h_b.to(h.device)).norm(p=2).item())
                    # topk next tokens
                    topk = topk_next_tokens(logits.detach().float().cpu(), tokenizer, k=topk_log)

                    last_log_pack = dict(
                        valid=valid,
                        gt=gt,
                        tgt=valid[tgt_local],
                        p_gt=p_gt,
                        p_tgt=p_tgt,
                        p_valid_mass=p_valid_mass,
                        cos_gt=float(cos_gt.item()),
                        loss_ce=float(loss_ce.item()),
                        loss_repel=float(loss_repel.item()),
                        h_norm=h_norm,
                        h_base_norm=h_b_norm,
                        delta_norm=delta_norm,
                        topk=topk,
                        prompt_tail=prompt[-220:],
                    )

            if used == 0:
                continue

            loss_mean = loss_sum / used
            (loss_mean / grad_accum).backward()
            step += 1

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)

            if step % log_every == 0:
                print(f"[TRAIN] step={step} loss={loss_mean.item():.6f} used={used}")
                if last_log_pack is not None:
                    x = last_log_pack
                    print(
                        f"  gt={x['gt']} tgt={x['tgt']} valid={x['valid']} | "
                        f"lm p_gt={x['p_gt']:.4f} p_tgt={x['p_tgt']:.4f} p_valid_mass={x['p_valid_mass']:.4f} | "
                        f"cos_gt={x['cos_gt']:.4f}"
                    )
                    print(
                        f"  losses: ce={x['loss_ce']:.4f} repel={x['loss_repel']:.4f} (lam_repel={lam_repel})"
                    )
                    print(
                        f"  ||h||={x['h_norm']:.3f} ||h_base||={x['h_base_norm']:.3f} ||delta||={x['delta_norm']:.3f}"
                    )
                    print(f"  prompt_tail: {repr(x['prompt_tail'])}")
                    print(f"  top{topk_log} next tokens: {x['topk']}")

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
    ap.add_argument("--lam_repel", type=float, default=1.0)

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

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]

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
    )
