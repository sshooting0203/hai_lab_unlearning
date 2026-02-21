#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Layer-window sensitivity probing experiment (multi-probe vs full training correlation).

What it does
------------
- Split 32-layer LLM into contiguous windows of size W (default 4) with stride S (default 4).
- For each window [l_s, l_e]:
    - Attach LoRA to target modules.
    - Freeze all params except LoRA params inside [l_s, l_e].
    - Train on forget-train (your cache JSONL) with:
        L = CE(wrong-target) + lam_repel * ReLU(cos_gt - margin_m)  (optional)
    - Evaluate on forget-eval at multiple checkpoints within SAME run:
        - probe_steps_list (e.g., 50,100,150,200)
        - full_steps  (final training)
    - Record score = drop in mean log p(gt):
        delta_t = base_logp_gt - logp_gt_at_step_t

- After running all windows:
    - Compute Pearson & Spearman correlation between probe deltas (each probe step) and final delta
    - Report top-k overlap / rank agreement (each probe step vs final)
    - Save CSV summary

Data format (JSONL)
-------------------
Each line is a JSON object containing at least:
  - "text_input": prompt string
  - "choices": list of answer choices
  - "answer": "A"/"B"/"C"/"D" or numeric label (0-based/1-based)
"""

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

LETTERS = ["A", "B", "C", "D"]
EPS = 1e-12


# -------------------------
# Repro / utils
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
        # trailing space is important so that next token tends to be " A"/" B"...
        return base + "\nThe answer is "
    raise KeyError("Record missing text_input.")


def get_global_letter_token_ids(tokenizer) -> torch.Tensor:
    ids = []
    for L in LETTERS:
        tok = tokenizer(f" {L}", add_special_tokens=False).input_ids
        if len(tok) != 1:
            raise ValueError(f'" {L}" is not a single token: {tok}')
        ids.append(tok[0])
    return torch.tensor(ids, dtype=torch.long)


def choose_wrong_target_random(valid_n: int, gt_local: int) -> int:
    candidates = [i for i in range(valid_n) if i != gt_local]
    return random.choice(candidates)


def last_idx_from_attention(attention_mask: torch.Tensor) -> torch.Tensor:
    last_idx = attention_mask.sum(dim=1) - 1
    return last_idx.clamp(min=0)


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
# Dataset (JSONL)
# -------------------------

class JsonlCacheDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
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


def freeze_all_params(model):
    for p in model.parameters():
        p.requires_grad = False


def enable_lora_in_layer_window(model, layer_start: int, layer_end: int):
    """
    Enable LoRA params only for layers in [layer_start, layer_end].
    Assumes LoRA params contain "lora_" and layer path contains "layers.<i>."
    """
    enabled = 0
    for name, p in model.named_parameters():
        if "lora_" not in name:
            continue
        keep = False
        for li in range(layer_start, layer_end + 1):
            if f"model.layers.{li}." in name or f"layers.{li}." in name:
                keep = True
                break
        p.requires_grad = bool(keep)
        if keep:
            enabled += p.numel()
    if enabled == 0:
        raise ValueError(
            f"Enabled LoRA params = 0 for window [{layer_start},{layer_end}]. "
            "Check parameter naming / model architecture."
        )


def count_trainable_params(model) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


# -------------------------
# Eval: mean log p(gt) on forget-eval
# -------------------------

@torch.no_grad()
def eval_mean_logp_gt(
    model,
    tokenizer,
    eval_loader: DataLoader,
    letter_ids_global: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    tot = 0.0
    n = 0
    use_amp = device.type == "cuda"

    for raw_batch in eval_loader:
        prompts = []
        valid_n_list = []
        gt_local_list = []

        for r in raw_batch:
            valid = valid_letters_from_choices(r)
            if not valid:
                continue
            gt = norm_answer_to_letter(r.get("answer"))
            if gt not in valid:
                continue
            prompts.append(build_prompt_from_record(r))
            valid_n_list.append(len(valid))
            gt_local_list.append(valid.index(gt))

        if len(prompts) == 0:
            continue

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        last_idx = last_idx_from_attention(attention_mask)
        B = input_ids.size(0)
        ar = torch.arange(B, device=device)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
            logits_last = out.logits[ar, last_idx, :]  # [B,V]

        logits_letters = logits_last.index_select(dim=1, index=letter_ids_global)  # [B,4]

        valid_n = torch.tensor(valid_n_list, device=device, dtype=torch.long)  # [B]
        pos = torch.arange(4, device=device).unsqueeze(0).expand(B, 4)
        mask_valid = pos < valid_n.unsqueeze(1)
        neg_inf = torch.finfo(logits_letters.dtype).min
        logits_letters_masked = torch.where(mask_valid, logits_letters, torch.full_like(logits_letters, neg_inf))
        logp_letters = torch.log_softmax(logits_letters_masked, dim=-1)  # [B,4]

        gt_local = torch.tensor(gt_local_list, device=device, dtype=torch.long)
        logp_gt = logp_letters[ar, gt_local].clamp_min(-1e9)
        tot += float(logp_gt.sum().item())
        n += int(logp_gt.numel())

    return tot / max(n, 1)


# -------------------------
# Train one window with multi-probe checkpoints + final checkpoint
# -------------------------

def train_one_window(
    model_name: str,
    train_jsonl: str,
    eval_jsonl: str,
    stats_path: str,
    steer_layer: int,
    min_count_proto: int,
    margin_m: float,
    lam_repel: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    layer_start: int,
    layer_end: int,
    lr: float,
    batch_size: int,
    grad_accum: int,
    probe_steps_list: List[int],
    full_steps: int,
    log_every: int,
    device_str: str,
    dtype_str: str,
    seed: int,
) -> Dict[str, float]:
    set_seed(seed)

    device = torch.device("cuda" if (device_str.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    if dtype_str == "float16":
        amp_dtype = torch.float16
    elif dtype_str == "bfloat16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # prototypes for repel
    proto_unit, _counts = load_proto_unit(stats_path, min_count=min_count_proto)
    proto_layer_cpu = proto_unit[steer_layer].float()  # [4,D]

    # backbone + LoRA
    backbone = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,
        torch_dtype=amp_dtype if device.type == "cuda" else torch.float32,
    ).to(device)

    model = apply_lora(backbone, target_modules=target_modules, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    # Ensure ONLY LoRA in window is trainable (and base is frozen)
    freeze_all_params(model)
    enable_lora_in_layer_window(model, layer_start, layer_end)

    total_params, trainable_params = count_trainable_params(model)

    # move proto to device
    proto_layer = proto_layer_cpu.to(device)  # [4,D]
    letter_ids_global = get_global_letter_token_ids(tokenizer).to(device)

    # DataLoaders (raw dict batches)
    train_ds = JsonlCacheDataset(train_jsonl)
    eval_ds = JsonlCacheDataset(eval_jsonl)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=lambda x: x)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda x: x)

    # base eval logp for this run (consistent baseline)
    base_logp = eval_mean_logp_gt(backbone, tokenizer, eval_loader, letter_ids_global, device, amp_dtype)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    model.train()
    step = 0
    optim.zero_grad(set_to_none=True)

    probe_steps_set = set(probe_steps_list)
    probe_logp_map: Dict[int, float] = {}
    final_logp: float = float("nan")

    # Train loop until full_steps
    for epoch in range(10_000_000):
        for raw_batch in train_loader:
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

                # proto existence (if repel is on)
                if lam_repel > 0.0:
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

            if len(prompts) == 0:
                continue

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            last_idx = last_idx_from_attention(attention_mask)

            B = input_ids.size(0)
            ar = torch.arange(B, device=device)

            need_hidden = lam_repel > 0.0
            use_amp = device.type == "cuda"

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=need_hidden,
                )
                logits_last = out.logits[ar, last_idx, :]  # [B,V]
                logits_letters = logits_last.index_select(dim=1, index=letter_ids_global)  # [B,4]

                valid_n = torch.tensor(valid_n_list, device=device, dtype=torch.long)
                pos = torch.arange(4, device=device).unsqueeze(0).expand(B, 4)
                mask_valid = pos < valid_n.unsqueeze(1)

                neg_inf = torch.finfo(logits_letters.dtype).min
                logits_letters_masked = torch.where(mask_valid, logits_letters, torch.full_like(logits_letters, neg_inf))
                logp_letters = torch.log_softmax(logits_letters_masked, dim=-1)

                tgt_local = torch.tensor(tgt_local_list, device=device, dtype=torch.long)
                loss_ce = (-logp_letters[ar, tgt_local]).mean()

                if need_hidden:
                    h = out.hidden_states[steer_layer + 1][ar, last_idx, :].float()  # [B,D]
                    h_hat = F.normalize(h, dim=-1)

                    gt_global = torch.tensor(gt_global_list, device=device, dtype=torch.long)
                    mu = proto_layer.index_select(dim=0, index=gt_global).float()     # [B,D]
                    mu_hat = F.normalize(mu, dim=-1)
                    cos_gt = (h_hat * mu_hat).sum(dim=-1)  # [B]
                    loss_repel = F.relu(cos_gt - margin_m).mean()
                else:
                    loss_repel = torch.tensor(0.0, device=device)

                loss = loss_ce + lam_repel * loss_repel

            (loss / grad_accum).backward()
            step += 1

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)

            if log_every > 0 and step % log_every == 0:
                print(f"[WIN {layer_start:02d}-{layer_end:02d}] step={step} loss={float(loss.item()):.6f} "
                      f"(trainable={trainable_params:,}/{total_params:,})")

            # Multi-probe evaluations
            if step in probe_steps_set and step not in probe_logp_map:
                probe_logp_map[step] = eval_mean_logp_gt(
                    model, tokenizer, eval_loader, letter_ids_global, device, amp_dtype
                )

            # Final evaluation
            if step == full_steps:
                final_logp = eval_mean_logp_gt(
                    model, tokenizer, eval_loader, letter_ids_global, device, amp_dtype
                )
                break

            if step >= full_steps:
                break

        if step >= full_steps:
            break

    # Fill missing probe checkpoints (should be rare)
    for s in probe_steps_list:
        if s not in probe_logp_map:
            probe_logp_map[s] = eval_mean_logp_gt(
                model, tokenizer, eval_loader, letter_ids_global, device, amp_dtype
            )

    if final_logp != final_logp:  # NaN check
        final_logp = eval_mean_logp_gt(model, tokenizer, eval_loader, letter_ids_global, device, amp_dtype)

    # Build result dict
    res: Dict[str, float] = {
        "base_logp": float(base_logp),
        "final_logp": float(final_logp),
        "final_delta": float(base_logp - final_logp),
        "trainable_params": float(trainable_params),
    }
    for s in probe_steps_list:
        logp_s = float(probe_logp_map[s])
        res[f"probe_logp_{s}"] = logp_s
        res[f"probe_delta_{s}"] = float(base_logp - logp_s)

    # cleanup
    del model
    del backbone
    torch.cuda.empty_cache()

    return res


# -------------------------
# Correlation helpers (no numpy/scipy dependency)
# -------------------------

def pearson_corr(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n == 0:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return cov / ((vx ** 0.5) * (vy ** 0.5))


def rankdata(values: List[float]) -> List[float]:
    # average rank for ties
    sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    r = 1
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[sorted_idx[j + 1]] == values[sorted_idx[i]]:
            j += 1
        avg_rank = (r + r + (j - i)) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        r += (j - i + 1)
        i = j + 1
    return ranks


def spearman_corr(xs: List[float], ys: List[float]) -> float:
    rx = rankdata(xs)
    ry = rankdata(ys)
    return pearson_corr(rx, ry)


def topk_overlap(idx_a: List[int], idx_b: List[int], k: int) -> float:
    A = set(idx_a[:k])
    B = set(idx_b[:k])
    return len(A & B) / max(k, 1)


# -------------------------
# Main: enumerate windows, run, compute correlation
# -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--eval_jsonl", type=str, required=True)
    ap.add_argument("--stats", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/window_probe_corr")

    ap.add_argument("--num_layers", type=int, default=32)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--min_start", type=int, default=0)

    ap.add_argument("--steer_layer", type=int, default=17)
    ap.add_argument("--min_count_proto", type=int, default=50)

    ap.add_argument("--margin_m", type=float, default=0.05)
    ap.add_argument("--lam_repel", type=float, default=0.0)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="q_proj,v_proj")

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)

    ap.add_argument("--probe_steps_list", type=str, default="50,100,150,200",
                    help="comma-separated probe steps, e.g., 50,100,150,200")
    ap.add_argument("--full_steps", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=50)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--topk", type=int, default=3, help="report top-k overlap between probe and final rankings")

    args = ap.parse_args()

    # Parse probe steps
    probe_steps_list = [int(x) for x in args.probe_steps_list.split(",") if x.strip()]
    probe_steps_list = sorted(list(set(probe_steps_list)))
    if len(probe_steps_list) == 0:
        raise ValueError("--probe_steps_list produced empty list")
    if any(s <= 0 for s in probe_steps_list):
        raise ValueError("probe steps must be positive")
    if args.full_steps <= max(probe_steps_list):
        raise ValueError("--full_steps must be larger than max(probe_steps_list)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]

    # Generate windows
    windows: List[Tuple[int, int]] = []
    s = args.min_start
    while s < args.num_layers:
        e = s + args.window - 1
        if e < args.num_layers:
            windows.append((s, e))
        s += args.stride
    if len(windows) == 0:
        raise ValueError("No windows generated. Check --min_start/--window/--stride/--num_layers.")

    print(f"[SETUP] num_layers={args.num_layers} window={args.window} stride={args.stride} min_start={args.min_start}")
    print(f"[SETUP] windows={windows}")
    print(f"[SETUP] probe_steps_list={probe_steps_list} full_steps={args.full_steps}")
    print(f"[SETUP] target_modules={target_modules} lam_repel={args.lam_repel} steer_layer={args.steer_layer}")

    rows: List[Dict[str, float]] = []
    for wi, (l_s, l_e) in enumerate(windows):
        seed = args.seed + 1000 * wi  # decorrelate wrong-target sampling across windows

        print("\n" + "=" * 90)
        print(f"[RUN] window {wi+1}/{len(windows)}: [{l_s},{l_e}] seed={seed}")

        res = train_one_window(
            model_name=args.model,
            train_jsonl=args.train_jsonl,
            eval_jsonl=args.eval_jsonl,
            stats_path=args.stats,
            steer_layer=args.steer_layer,
            min_count_proto=args.min_count_proto,
            margin_m=args.margin_m,
            lam_repel=args.lam_repel,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            layer_start=l_s,
            layer_end=l_e,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            probe_steps_list=probe_steps_list,
            full_steps=args.full_steps,
            log_every=args.log_every,
            device_str=args.device,
            dtype_str=args.dtype,
            seed=seed,
        )

        row = {"l_s": float(l_s), "l_e": float(l_e), **res}
        rows.append(row)

        probes_str = " ".join([f"Δ@{s}={res[f'probe_delta_{s}']:.3f}" for s in probe_steps_list])
        print(f"[RESULT] [{l_s},{l_e}] final_Δ={res['final_delta']:.3f} | {probes_str}")

    # Compute correlations probe@step vs final
    final = [r["final_delta"] for r in rows]
    idx_final = sorted(range(len(rows)), key=lambda i: rows[i]["final_delta"], reverse=True)

    print("\n" + "#" * 90)
    for s in probe_steps_list:
        probe = [r[f"probe_delta_{s}"] for r in rows]
        pear = pearson_corr(probe, final)
        spear = spearman_corr(probe, final)
        idx_probe = sorted(range(len(rows)), key=lambda i: rows[i][f"probe_delta_{s}"], reverse=True)
        ov = topk_overlap(idx_probe, idx_final, k=args.topk)

        print(f"[CORR@{s:4d}] Pearson={pear:.4f} Spearman={spear:.4f} | top-{args.topk} overlap={ov:.3f}")
        print(f"          probe top: {[(int(rows[i]['l_s']), int(rows[i]['l_e']), rows[i][f'probe_delta_{s}']) for i in idx_probe[:args.topk]]}")
        print(f"          final top: {[(int(rows[i]['l_s']), int(rows[i]['l_e']), rows[i]['final_delta']) for i in idx_final[:args.topk]]}")

    # Save CSV (dynamic columns)
    csv_path = out_dir / "window_probe_multi_vs_final.csv"
    headers: List[str] = ["l_s", "l_e", "base_logp"]
    for s in probe_steps_list:
        headers += [f"probe_logp_{s}", f"probe_delta_{s}"]
    headers += ["final_logp", "final_delta", "trainable_params"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            row_out: List[Any] = [int(r["l_s"]), int(r["l_e"]), f"{r['base_logp']:.8f}"]
            for s in probe_steps_list:
                row_out += [f"{r[f'probe_logp_{s}']:.8f}", f"{r[f'probe_delta_{s}']:.8f}"]
            row_out += [f"{r['final_logp']:.8f}", f"{r['final_delta']:.8f}", int(r["trainable_params"])]
            w.writerow(row_out)

    meta_path = out_dir / "summary.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"windows={[(int(a), int(b)) for (a,b) in windows]}\n")
        f.write(f"probe_steps_list={probe_steps_list}\n")
        f.write(f"full_steps={args.full_steps}\n")
        f.write(f"target_modules={target_modules}\n")
        f.write(f"lam_repel={args.lam_repel} steer_layer={args.steer_layer}\n")
        f.write(f"topk={args.topk}\n")

    print(f"\n[SAVED] {csv_path.resolve()}")
    print(f"[SAVED] {meta_path.resolve()}")


if __name__ == "__main__":
    main()
