#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forget-only LoRA layer-range search for MCQ (A/B/C/D) prompts.

Changes vs earlier version:
1) JSON / JSONL auto support (load_json_or_jsonl).
2) Robust LoRA injection:
   - Do NOT rely on PEFT layers_to_transform/layers_pattern (which often breaks across model variants).
   - Inject LoRA into all *existing* target modules.
   - Freeze LoRA params outside [l_s, l_e] by parameter-name filtering.

3) Search candidate start layer >= --min_start (e.g., 8).

Metric:
  base_logp = mean log p(gt) on forget-eval for base model
  eval_logp = mean log p(gt) on forget-eval for LoRA model
  delta     = base_logp - eval_logp   (bigger is better = GT prob decreases)
"""

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable, Optional, Set

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

LETTERS = ["A", "B", "C", "D"]
EPS = 1e-12


# ----------------------------
# JSON / JSONL loader
# ----------------------------

def load_json_or_jsonl(path: str):
    """
    Supports:
      - JSON list/object in a single file
      - JSONL (one JSON object per line)

    Returns:
      - If JSON list -> list
      - If JSON object -> dict
      - If JSONL -> list[dict]
    """
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    if txt == "":
        return []

    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSONL at {path}:{line_no}: {e}") from e
    return data


# ----------------------------
# Utilities: answer normalization / prompting
# ----------------------------

def norm_answer_to_letter(a: Any) -> str:
    if a is None:
        return ""
    s = str(a).strip().upper()

    if s.isdigit():
        i = int(s)
        if 0 <= i < 4:
            return LETTERS[i]
        if 1 <= i <= 4:
            return LETTERS[i - 1]
        return ""

    for ch in s:
        if ch in "ABCD":
            return ch
    return ""


def build_prompt(r: Dict[str, Any]) -> str:
    base = r.get("text_input") or r.get("prompt") or r.get("instruction") or r.get("question") or ""
    base = str(base).strip()

    key = "The answer is"
    if key in base:
        base = base.split(key)[0].rstrip()

    return base + "\nThe answer is"


def valid_letters(r: Dict[str, Any]) -> List[str]:
    ch = r.get("choices", None)
    if isinstance(ch, list) and len(ch) > 0:
        n = min(len(ch), 4)
        return LETTERS[:n]
    return LETTERS


# ----------------------------
# Dataset
# ----------------------------

@dataclass
class Example:
    prompt: str
    gt_letter: str
    valid: List[str]


class ForgetMCQDataset(Dataset):
    def __init__(self, path: str):
        data = load_json_or_jsonl(path)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list/dict or JSONL; got {type(data)} from {path}")

        exs: List[Example] = []
        for r in data:
            if not isinstance(r, dict):
                continue
            gt = norm_answer_to_letter(r.get("answer"))
            if gt == "":
                continue
            v = valid_letters(r)
            if gt not in v:
                continue
            exs.append(Example(prompt=build_prompt(r), gt_letter=gt, valid=v))

        if len(exs) == 0:
            raise ValueError(f"No valid examples parsed from {path}. Check fields/answers.")
        self.exs = exs

    def __len__(self):
        return len(self.exs)

    def __getitem__(self, idx):
        return self.exs[idx]


def make_collate(tokenizer, max_len: int):
    def collate(batch: List[Example]) -> Dict[str, Any]:
        prompts = [b.prompt for b in batch]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "gt_letters": [b.gt_letter for b in batch],
            "valids": [b.valid for b in batch],
            "prompts": prompts,
        }
    return collate


# ----------------------------
# Letter token ids
# ----------------------------

def get_letter_token_ids(tokenizer) -> Dict[str, int]:
    ids = {}
    for L in LETTERS:
        tok = tokenizer(f" {L}", add_special_tokens=False)["input_ids"]
        if len(tok) != 1:
            raise ValueError(
                f"Tokenization for ' {L}' produced {tok} (len={len(tok)}). "
                "This script assumes single-token letters."
            )
        ids[L] = tok[0]
    return ids


def last_token_index(attn: torch.Tensor) -> torch.Tensor:
    return attn.long().sum(dim=1) - 1


# ----------------------------
# Eval metric: mean log p(gt)
# ----------------------------

@torch.no_grad()
def mean_logp_gt(model, dl: DataLoader, letter2id: Dict[str, int], device: str, dtype: str) -> float:
    model.eval()
    total = 0.0
    n = 0

    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    use_amp = device.startswith("cuda")

    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        gt_letters = batch["gt_letters"]
        valids = batch["valids"]

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits

        idx = last_token_index(attn)
        next_logits = logits[torch.arange(logits.size(0), device=device), idx, :]

        logp_sum = 0.0
        for i in range(next_logits.size(0)):
            v = valids[i]
            vids = torch.tensor([letter2id[L] for L in v], device=device, dtype=torch.long)
            sub = next_logits[i, vids]
            p = F.softmax(sub, dim=0)
            gt = gt_letters[i]
            gt_pos = v.index(gt)
            pgt = p[gt_pos].clamp_min(EPS)
            logp_sum += float(torch.log(pgt).item())

        total += logp_sum
        n += next_logits.size(0)

    return total / max(n, 1)


# ----------------------------
# Training objective: push down p(gt)
# ----------------------------

def forget_loss_pushdown_gt(
    next_logits: torch.Tensor,
    gt_letters: List[str],
    valids: List[List[str]],
    letter2id: Dict[str, int],
) -> torch.Tensor:
    losses = []
    device = next_logits.device
    B = next_logits.size(0)
    for i in range(B):
        v = valids[i]
        vids = torch.tensor([letter2id[L] for L in v], device=device, dtype=torch.long)
        sub = next_logits[i, vids]
        p = F.softmax(sub, dim=0)
        gt = gt_letters[i]
        gt_pos = v.index(gt)
        pgt = p[gt_pos].clamp(EPS, 1.0 - EPS)
        losses.append(-torch.log(1.0 - pgt + EPS))
    return torch.stack(losses).mean()


# ----------------------------
# LoRA target discovery + freezing
# ----------------------------

def discover_target_modules(model, preferred: List[str]) -> List[str]:
    """
    PEFT needs target_modules that match leaf module names.
    We discover which of `preferred` actually exist in the model.
    """
    present: Set[str] = set()
    for name, _m in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in preferred:
            present.add(leaf)

    found = [m for m in preferred if m in present]
    return found


def set_lora_trainable_range(model, l_s: int, l_e: int, num_layers: int):
    """
    After get_peft_model, LoRA params typically appear in parameter names that include:
      "...layers.{i}....lora_A..." / "...lora_B..." (or lora_embedding_A/B etc)

    We:
      - set all params requires_grad=False
      - then enable ONLY LoRA params within [l_s,l_e]
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Enable LoRA params for layers in range
    enabled = 0
    for name, p in model.named_parameters():
        if "lora_" not in name:
            continue

        # Try to parse layer index from common patterns
        # e.g., "base_model.model.model.layers.12.self_attn.q_proj.lora_A.weight"
        # or    "model.layers.12....lora_A.weight"
        idx = None
        parts = name.split(".")
        for j in range(len(parts) - 1):
            if parts[j] == "layers":
                # next token should be an integer index
                try:
                    idx = int(parts[j + 1])
                except Exception:
                    idx = None
                break

        if idx is None:
            # If cannot parse, keep it frozen to be safe
            continue

        if 0 <= idx < num_layers and (l_s <= idx <= l_e):
            p.requires_grad = True
            enabled += p.numel()

    if enabled == 0:
        raise ValueError(
            f"Enabled LoRA params = 0 for range [{l_s},{l_e}]. "
            "This usually means layer index parsing failed or LoRA params aren't named as expected."
        )


def print_some_module_names(model, max_lines: int = 40):
    """
    Debug helper: prints some leaf module names to quickly see what target_modules should be.
    """
    c = 0
    seen = set()
    for name, _m in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in seen:
            continue
        if any(k in leaf for k in ["proj", "wq", "wk", "wv", "wo", "gate", "up", "down"]):
            print(f"  leaf={leaf} (example full={name})")
            seen.add(leaf)
            c += 1
            if c >= max_lines:
                break


# ----------------------------
# Model / LoRA building (robust)
# ----------------------------

def build_lora_model(
    base_model_name: str,
    device: str,
    dtype: str,
    l_s: int,
    l_e: int,
    num_layers: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    preferred_target_modules: List[str],
) -> Tuple[Any, Any, List[str]]:
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    model.to(device)

    # Discover which target module leaf names exist
    targets = discover_target_modules(model, preferred_target_modules)
    if len(targets) == 0:
        print("[DEBUG] No preferred target_modules found. Here are some candidate leaf module names:")
        print_some_module_names(model, max_lines=80)
        raise ValueError(
            "Could not find any of the preferred target module names in the model. "
            "You likely need to change --target_modules to match this model's naming."
        )

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
        # NOTE: do NOT use layers_to_transform/layers_pattern (robust across models)
    )

    model = get_peft_model(model, lora_cfg)

    # Freeze everything, then enable LoRA params only in [l_s,l_e]
    set_lora_trainable_range(model, l_s, l_e, num_layers)

    # Print trainable summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] using target_modules={targets}")
    print(f"[LoRA] trainable params: {trainable:,} / total: {total:,}")

    model.train()
    return model, tok, targets


# ----------------------------
# Candidate generation
# ----------------------------

def gen_candidates(num_layers: int, range_len_list: List[int], start_stride: int, min_start: int) -> List[Tuple[int, int]]:
    cands = []
    for L in range_len_list:
        for s in range(min_start, num_layers, start_stride):
            e = s + L - 1
            if e < num_layers:
                cands.append((s, e))
    return sorted(list(set(cands)))


# ----------------------------
# Main search loop
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--forget_train", type=str, required=True)
    ap.add_argument("--forget_eval", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--num_layers", type=int, default=32)
    ap.add_argument("--range_len_list", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--start_stride", type=int, default=2)
    ap.add_argument("--min_start", type=int, default=0, help="minimum start layer for candidates (e.g., 8)")

    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=512)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--dtype", type=str, choices=["bf16", "fp16"], default="bf16")
    ap.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="preferred leaf module names; script will auto-filter to ones that exist"
    )
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--train_subset", type=int, default=0)
    ap.add_argument("--eval_subset", type=int, default=0)

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    d_train = ForgetMCQDataset(args.forget_train)
    d_eval = ForgetMCQDataset(args.forget_eval)
    if args.train_subset and args.train_subset > 0:
        d_train.exs = d_train.exs[: args.train_subset]
    if args.eval_subset and args.eval_subset > 0:
        d_eval.exs = d_eval.exs[: args.eval_subset]

    # Tokenizer for collate + letter ids
    base_tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token
    letter2id = get_letter_token_ids(base_tok)

    train_dl = DataLoader(
        d_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(base_tok, args.max_len),
        drop_last=True,
    )
    eval_dl = DataLoader(
        d_eval,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate(base_tok, args.max_len),
        drop_last=False,
    )

    # Base metric once
    print("\n[BASE] computing mean log p(gt) on forget-eval ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=(torch.bfloat16 if args.dtype == "bf16" else torch.float16),
        device_map=None,
    ).to(device)
    base_model.eval()
    base_logp = mean_logp_gt(base_model, eval_dl, letter2id, device, args.dtype)
    print(f"[BASE] mean log p(gt) = {base_logp:.6f}")
    del base_model
    torch.cuda.empty_cache()

    candidates = gen_candidates(args.num_layers, args.range_len_list, args.start_stride, args.min_start)
    print(f"\nCandidates: {len(candidates)} intervals (min_start={args.min_start})")

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "l_s", "l_e", "steps", "train_loss_last",
            "eval_logp_gt", "delta_logp_base_minus_lora",
            "note"
        ])

    best_delta = -1e9
    best_interval = None

    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    use_amp = device.startswith("cuda")

    for (l_s, l_e) in candidates:
        print("\n" + "=" * 80)
        print(f"[CANDIDATE] layers [{l_s}, {l_e}]")

        model, _tok, targets = build_lora_model(
            base_model_name=args.model,
            device=device,
            dtype=args.dtype,
            l_s=l_s,
            l_e=l_e,
            num_layers=args.num_layers,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            preferred_target_modules=args.target_modules,
        )

        optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        model.train()
        step = 0
        last_loss = float("nan")
        it = iter(train_dl)

        while step < args.steps:
            optim.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for _ in range(args.grad_accum):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(train_dl)
                    batch = next(it)

                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                gt_letters = batch["gt_letters"]
                valids = batch["valids"]

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    out = model(input_ids=input_ids, attention_mask=attn)
                    logits = out.logits
                    idx = last_token_index(attn)
                    next_logits = logits[torch.arange(logits.size(0), device=device), idx, :]
                    loss = forget_loss_pushdown_gt(next_logits, gt_letters, valids, letter2id)

                (loss / args.grad_accum).backward()
                accum_loss += float(loss.item())

            optim.step()
            step += 1
            last_loss = accum_loss / args.grad_accum

            if step % args.eval_every == 0 or step == args.steps:
                eval_logp = mean_logp_gt(model, eval_dl, letter2id, device, args.dtype)
                delta = base_logp - eval_logp
                print(f"[EVAL] step={step:4d} train_loss={last_loss:.6f} "
                      f"eval_logp_gt={eval_logp:.6f} delta(base-lora)={delta:.6f}")

        eval_logp = mean_logp_gt(model, eval_dl, letter2id, device, args.dtype)
        delta = base_logp - eval_logp

        note = ""
        if delta > best_delta:
            best_delta = delta
            best_interval = (l_s, l_e)
            note = "BEST"
            best_path = out_dir / "best_adapter"
            best_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_path)
            base_tok.save_pretrained(best_path)
            print(f"[SAVE] new best adapter saved to: {best_path}")

        with open(csv_path, "a", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([l_s, l_e, args.steps, f"{last_loss:.6f}", f"{eval_logp:.6f}", f"{delta:.6f}", note])

        print(f"[DONE] layers [{l_s},{l_e}] eval_logp_gt={eval_logp:.6f} delta={delta:.6f} {note}")

        del model
        torch.cuda.empty_cache()

    print("\n" + "#" * 80)
    print(f"[RESULT] base mean log p(gt) = {base_logp:.6f}")
    print(f"[RESULT] best delta(base-lora) = {best_delta:.6f}")
    print(f"[RESULT] best interval = {best_interval}")
    print(f"[RESULT] summary.csv written to: {csv_path}")
    if best_interval is not None:
        print(f"[RESULT] best adapter at: {out_dir / 'best_adapter'}")


if __name__ == "__main__":
    main()
