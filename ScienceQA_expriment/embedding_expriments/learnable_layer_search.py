#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Layer search for unlearning (forget-only) using diagonal Fisher (layer-wise).

Adds:
  - --start_layer / --end_layer to restrict which layers to score
    (inclusive range). If end_layer is not set, it goes to last layer.
"""

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

LETTERS = ["A", "B", "C", "D"]
DEFAULT_PROMPT_SUFFIX = "\nThe answer is"


# ---------------------------
# Utilities: prompt + target
# ---------------------------

def norm_answer_to_letter(a: Any) -> str:
    if a is None:
        return ""
    s = str(a).strip().upper()
    # numeric handling (0-based or 1-based)
    if s.isdigit():
        i = int(s)
        if 0 <= i < 4:
            return LETTERTERS[i]  # typo guard below
        if 1 <= i <= 4:
            return LETTERS[i - 1]
    for ch in s:
        if ch in "ABCD":
            return ch
    return ""

# fix typo guard
LETTERTERS = LETTERS

def build_prompt(rec: Dict[str, Any], suffix: str = DEFAULT_PROMPT_SUFFIX) -> str:
    text = rec.get("text_input", rec.get("instruction", rec.get("question", "")))
    if text is None:
        text = ""
    text = str(text).strip()
    if "The answer is" in text:
        text = text.split("The answer is")[0].rstrip()
    return text + suffix

def valid_letters_for_sample(rec: Dict[str, Any], max_choices: int = 4) -> List[str]:
    n = rec.get("num_choices", None)
    if n is None:
        choices = rec.get("choices", None)
        if isinstance(choices, list):
            n = len(choices)
    if not isinstance(n, int):
        n = 4
    n = max(0, min(int(n), max_choices))
    return LETTERS[:n] if n > 0 else LETTERS  # fallback

def pick_target_letter(gt: str, valid: List[str], mode: str, rng: random.Random) -> str:
    gt = (gt or "").strip().upper()
    if gt not in valid:
        gt = valid[0] if len(valid) else "A"

    if mode == "gt":
        return gt

    wrongs = [x for x in valid if x != gt]
    if not wrongs:
        return gt

    if mode == "random_wrong":
        return rng.choice(wrongs)

    if mode == "wrong_fixed":
        return wrongs[0]

    raise ValueError(f"Unknown target_mode: {mode}")


# ---------------------------
# Dataset
# ---------------------------

@dataclass
class Item:
    prompt: str
    gt_letter: str
    valid_letters: List[str]

class ForgetDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], suffix: str):
        self.items: List[Item] = []
        for r in records:
            prompt = build_prompt(r, suffix=suffix)
            gt = norm_answer_to_letter(r.get("answer", r.get("label", r.get("gt", ""))))
            valid = valid_letters_for_sample(r, max_choices=4)
            self.items.append(Item(prompt=prompt, gt_letter=gt, valid_letters=valid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Item:
        return self.items[idx]

def collate_items(items: List[Item]) -> Dict[str, Any]:
    return {
        "prompts": [x.prompt for x in items],
        "gt_letters": [x.gt_letter for x in items],
        "valid_letters": [x.valid_letters for x in items],
    }


# ---------------------------
# Layer mapping helpers
# ---------------------------

def detect_layer_count(model) -> Optional[int]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    return None

def param_to_layer_id(param_name: str) -> Optional[int]:
    patterns = [
        r"\bmodel\.layers\.(\d+)\.",
        r"\blayers\.(\d+)\.",
        r"\btransformer\.h\.(\d+)\.",
        r"\bmodel\.decoder\.layers\.(\d+)\.",
    ]
    for pat in patterns:
        m = re.search(pat, param_name)
        if m:
            return int(m.group(1))
    return None

def is_lora_param(name: str) -> bool:
    return ("lora_" in name) or (".lora_A." in name) or (".lora_B." in name)

def filter_param_scope(name: str, scope: str) -> bool:
    if scope == "full":
        return True
    if scope == "lora":
        return is_lora_param(name)
    if scope == "attn":
        keys = ["q_proj", "k_proj", "v_proj", "o_proj", "attn", "self_attn"]
        return any(k in name for k in keys)
    if scope == "mlp":
        keys = ["gate_proj", "up_proj", "down_proj", "mlp", "feed_forward"]
        return any(k in name for k in keys)
    raise ValueError(f"Unknown param_scope: {scope}")


# ---------------------------
# Core: Fisher computation
# ---------------------------

def get_letter_token_ids(tokenizer, letters: List[str]) -> Dict[str, int]:
    out = {}
    for L in letters:
        ids = tokenizer.encode(" " + L, add_special_tokens=False)
        out[L] = ids[-1] if len(ids) != 1 else ids[0]
    return out

def last_prompt_index(attn_mask: torch.Tensor) -> torch.Tensor:
    return attn_mask.long().sum(dim=1) - 1

@torch.no_grad()
def compute_valid_letter_mask(
    batch_valid_letters: List[List[str]],
    device: torch.device,
) -> torch.Tensor:
    B = len(batch_valid_letters)
    mask = torch.zeros((B, len(LETTERS)), dtype=torch.bool, device=device)
    for i, valid in enumerate(batch_valid_letters):
        valid_set = set([v.strip().upper() for v in valid])
        for j, L in enumerate(LETTERS):
            if L in valid_set:
                mask[i, j] = True
    return mask

def fisher_layer_search(
    model,
    tokenizer,
    loader: DataLoader,
    *,
    device: torch.device,
    dtype: torch.dtype,
    target_mode: str,
    param_scope: str,
    seed: int,
    max_batches: Optional[int],
    restrict_to_valid_letters: bool,
    log_every: int,
    start_layer: int,
    end_layer: Optional[int],
) -> Dict[str, Any]:
    rng = random.Random(seed)
    model.train()

    letter_to_id = get_letter_token_ids(tokenizer, LETTERS)
    letter_ids_tensor = torch.tensor([letter_to_id[L] for L in LETTERS], device=device)

    n_layers = detect_layer_count(model) or 0

    # Resolve layer range
    if start_layer < 0:
        start_layer = 0
    if end_layer is None:
        end_layer = (n_layers - 1) if n_layers > 0 else None
    if end_layer is not None and end_layer < start_layer:
        raise ValueError(f"--end_layer({end_layer}) must be >= --start_layer({start_layer})")

    # Accumulators
    fisher_sum: Dict[int, float] = {}
    fisher_count: Dict[int, int] = {}
    n_seen = 0

    # Pre-collect parameters per layer (filtered by scope + layer range)
    layer_params: Dict[int, List[torch.nn.Parameter]] = {}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not filter_param_scope(name, param_scope):
            continue
        lid = param_to_layer_id(name)
        if lid is None:
            continue

        # Apply start/end filtering here
        if lid < start_layer:
            continue
        if end_layer is not None and lid > end_layer:
            continue

        layer_params.setdefault(lid, []).append(p)

    if len(layer_params) == 0:
        raise RuntimeError(
            "No parameters matched your selection (scope + layer range). "
            "Try widening --start_layer/--end_layer or changing --param_scope."
        )

    # Main loop
    for bidx, batch in enumerate(loader):
        if max_batches is not None and bidx >= max_batches:
            break

        prompts: List[str] = batch["prompts"]
        gt_letters: List[str] = batch["gt_letters"]
        valid_letters: List[List[str]] = batch["valid_letters"]

        targets = []
        for gt, valid in zip(gt_letters, valid_letters):
            targets.append(pick_target_letter(gt, valid, target_mode, rng))

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length if tokenizer.model_max_length and tokenizer.model_max_length > 0 else 2048,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        # Forward (autocast on cuda only)
        with torch.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=dtype,
            enabled=(device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)),
        ):
            out = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = out.logits  # (B, T, V)

            last_idx = last_prompt_index(attn_mask)  # (B,)
            B, T, V = logits.shape
            row = torch.arange(B, device=device)
            next_logits = logits[row, last_idx, :]  # (B, V)

            if restrict_to_valid_letters:
                letter_logits = next_logits[:, letter_ids_tensor]  # (B,4)
                valid_mask = compute_valid_letter_mask(valid_letters, device=device)  # (B,4)
                masked = letter_logits.masked_fill(~valid_mask, float("-inf"))
                tgt_idx = torch.tensor(
                    [LETTERS.index(t) if t in LETTERS else 0 for t in targets],
                    device=device,
                    dtype=torch.long,
                )
                loss = F.cross_entropy(masked, tgt_idx, reduction="mean")
            else:
                tgt_token_ids = torch.tensor(
                    [letter_to_id.get(t, letter_to_id["A"]) for t in targets],
                    device=device,
                    dtype=torch.long,
                )
                loss = F.cross_entropy(next_logits, tgt_token_ids, reduction="mean")

        model.zero_grad(set_to_none=True)
        loss.backward()

        for lid, params in layer_params.items():
            s = 0.0
            c = 0
            for p in params:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                s += float(g.float().pow(2).sum().item())
                c += p.numel()
            fisher_sum[lid] = fisher_sum.get(lid, 0.0) + s
            fisher_count[lid] = fisher_count.get(lid, 0) + c

        n_seen += 1

        if log_every > 0 and (bidx + 1) % log_every == 0:
            tmp = [(lid, fisher_sum[lid] / max(1, n_seen)) for lid in fisher_sum.keys()]
            tmp.sort(key=lambda x: x[1], reverse=True)
            print(f"[batch {bidx+1}] loss={loss.item():.4f} | top5 layers by avg fisher: {tmp[:5]}")

    # Finalize scores
    scores = {}
    for lid in fisher_sum.keys():
        avg = fisher_sum[lid] / max(1, n_seen)
        norm = avg / max(1, fisher_count.get(lid, 1))
        scores[lid] = {
            "fisher_sum_avg": avg,
            "fisher_sum_avg_per_param": norm,
            "param_count_accum": fisher_count.get(lid, 0),
        }

    ranked = sorted(scores.items(), key=lambda kv: kv[1]["fisher_sum_avg_per_param"], reverse=True)
    ranked_layers = [int(lid) for lid, _ in ranked]

    return {
        "n_batches_used": n_seen,
        "target_mode": target_mode,
        "param_scope": param_scope,
        "restrict_to_valid_letters": restrict_to_valid_letters,
        "start_layer": start_layer,
        "end_layer": end_layer,
        "ranked_layers": ranked_layers,
        "layer_scores": {str(lid): v for lid, v in scores.items()},
    }


# ---------------------------
# Main
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF model name or path")
    ap.add_argument("--forget_json", type=str, required=True, help="forget dataset json(list) or jsonl")
    ap.add_argument("--peft", type=str, default=None, help="optional PEFT adapter path (LoRA).")
    ap.add_argument("--out", type=str, required=True, help="output json path")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--max_batches", type=int, default=None)

    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--target_mode", type=str, default="gt", choices=["gt", "random_wrong", "wrong_fixed"])
    ap.add_argument("--prompt_suffix", type=str, default=DEFAULT_PROMPT_SUFFIX)
    ap.add_argument("--restrict_to_valid_letters", action="store_true")

    ap.add_argument("--param_scope", type=str, default="full", choices=["full", "lora", "attn", "mlp"])
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--log_every", type=int, default=20)

    # NEW: layer range
    ap.add_argument("--start_layer", type=int, default=0, help="start layer (inclusive)")
    ap.add_argument("--end_layer", type=int, default=None, help="end layer (inclusive). default: last layer")

    return ap.parse_args()

def main():
    args = parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Load data (JSON list or JSONL)
    records = []
    with open(args.forget_json, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            records = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

    if args.max_samples is not None and args.max_samples > 0:
        records = records[: args.max_samples]

    ds = ForgetDataset(records, suffix=args.prompt_suffix)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_items, drop_last=False)

    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
        device_map=None,
    ).to(device)

    if args.peft is not None:
        if PeftModel is None:
            raise RuntimeError("peft is not installed, cannot load --peft adapter.")
        model = PeftModel.from_pretrained(model, args.peft).to(device)

        if args.param_scope == "lora":
            for n, p in model.named_parameters():
                if is_lora_param(n):
                    p.requires_grad_(True)

    if args.param_scope == "full":
        for p in model.parameters():
            p.requires_grad_(True)

    result = fisher_layer_search(
        model,
        tokenizer,
        loader,
        device=device,
        dtype=dtype,
        target_mode=args.target_mode,
        param_scope=args.param_scope,
        seed=args.seed,
        max_batches=args.max_batches,
        restrict_to_valid_letters=args.restrict_to_valid_letters,
        log_every=args.log_every,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
    )

    ranked = result["ranked_layers"][: args.topk]
    result["topk"] = args.topk
    result["topk_layers"] = ranked

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[done] wrote: {args.out}")
    print(f"Layer range used: [{result['start_layer']}, {result['end_layer']}]")
    print(f"Top-{args.topk} layers (by fisher per-param): {ranked}")

if __name__ == "__main__":
    main()
