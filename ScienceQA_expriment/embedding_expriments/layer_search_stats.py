#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stats-based layer search (NO simulation) for ScienceQA biology split cache.

What it does
1) Load train_pred_cache.jsonl (produced by split_biology_by_pred.py).
2) For each sample:
   - Build prompt "... The answer is"
   - Append model-predicted letter (teacher forcing): "... The answer is <pred>"
   - Run model with output_hidden_states=True
   - Take hidden at the *letter token position* for every layer
   - Accumulate:
       (a) unit-vector resultant S_unit for intra/inter (cosine-based)
       (b) raw mean/var (diag) via Welford (optional, for later steering loss)
3) Compute per-layer score:
   - intra: average over classes of (1 - ||S_unit||/N)
   - inter: average pairwise cosine distance between class mean directions
   - score = inter / (intra + eps)
4) Print Top-K layers and save stats to .pt

Usage example
  python layer_search_stats.py \
    --jsonl data/scienceqa/splits_biology_pred/train_pred_cache.jsonl \
    --model meta-llama/Llama-3.1-8B \
    --out stats_biology_train.pt \
    --topk 10 \
    --min_count 50 \
    --use_margin --margin_thresh 1.0
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

LETTERS = ["A", "B", "C", "D"]
L2_EPS = 1e-12


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True, help="train_pred_cache.jsonl path")
    ap.add_argument("--model", type=str, required=True, help="HF model name or local path")
    ap.add_argument("--out", type=str, default="stats.pt", help="torch save path")
    ap.add_argument("--max_items", type=int, default=0, help="0 means all")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--min_count", type=int, default=50, help="min samples per class to include in a layer score")
    ap.add_argument("--eps", type=float, default=1e-6)

    # Prompt source
    ap.add_argument("--use_text_input", action="store_true", help="Prefer record['text_input'] as base prompt")
    ap.add_argument("--strip_existing_answer_is", action="store_true",
                    help="If text_input contains 'The answer is', truncate before it")

    # Confidence filter
    ap.add_argument("--use_margin", action="store_true", help="Filter samples by top1-top2 margin")
    ap.add_argument("--margin_thresh", type=float, default=1.0)

    return ap.parse_args()


def norm_answer_field(a) -> str:
    """Not used for layer search; kept for completeness."""
    if a is None:
        return ""
    s = str(a).strip().upper()
    if s.isdigit():
        i = int(s)
        if 0 <= i < 5:
            return "ABCDE"[i]
    for ch in s:
        if ch in "ABCDE":
            return ch
    return ""


def build_prompt_from_record(r: Dict, use_text_input: bool, strip_existing: bool) -> str:
    """
    Return a prompt that ends with 'The answer is' (no trailing letter).
    Priority:
      - text_input if requested and non-empty
      - else construct from instruction/question + choices
    """
    if use_text_input:
        ti = r.get("text_input", "")
        if isinstance(ti, str) and ti.strip():
            base = ti.strip()
            if strip_existing and "The answer is" in base:
                base = base.split("The answer is")[0].rstrip()
            # Ensure we end with "The answer is"
            if base.endswith("The answer is"):
                return base
            return base + "\nThe answer is"

    # Fallback formatting
    q = r.get("instruction") or r.get("question")
    choices = r.get("choices")
    if q is None or not isinstance(choices, list) or len(choices) < 2:
        raise KeyError(f"Missing question/choices. keys={list(r.keys())}")

    valid_letters = LETTERS[: len(choices)]
    s = f"Question: {q}\nChoices:\n"
    for i, c in enumerate(choices):
        s += f"({valid_letters[i]}) {c}\n"
    s += f"Answer with a single letter among {', '.join(valid_letters)}.\nThe answer is"
    return s


def get_valid_letters(r: Dict) -> List[str]:
    choices = r.get("choices", [])
    if not isinstance(choices, list):
        return []
    n = len(choices)
    if n < 2:
        return []
    if n > 4:
        # biology_all observed up to 4 in your scan
        n = 4
    return LETTERS[:n]


def margin_ok(r: Dict, pred: str, thresh: float) -> bool:
    scores = r.get("pred_scores", {})
    if not isinstance(scores, dict) or pred not in scores:
        return False
    vals = list(scores.values())
    if len(vals) < 2:
        return False
    vals_sorted = sorted(vals, reverse=True)
    margin = vals_sorted[0] - vals_sorted[1]
    return margin >= thresh


def load_model_and_tokenizer(model_name: str, device: str, dtype_str: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if dtype_str == "float16":
        dtype = torch.float16
    elif dtype_str == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # device_map="auto" when CUDA; else CPU load
    kwargs = {}
    if device.startswith("cuda") and torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["dtype"] = dtype
    else:
        kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def collect_layer_stats(
    model,
    tokenizer,
    jsonl_path: str,
    device: str,
    use_text_input: bool,
    strip_existing_answer_is: bool,
    use_margin: bool,
    margin_thresh: float,
    max_items: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Accumulate:
      S_unit[L,C,D], N[L,C]
      mean_raw[L,C,D], var_raw[L,C,D], N_raw[L,C]
    Using teacher forcing: prompt + " <pred_letter>" and take hidden at last token position.
    """
    # Model sizes
    n_layers = int(model.config.num_hidden_layers)
    d = int(model.config.hidden_size)
    C = len(LETTERS)

    # Accumulators on CPU float32 (stable)
    S_unit = torch.zeros(n_layers, C, d, dtype=torch.float32, device="cpu")
    N = torch.zeros(n_layers, C, dtype=torch.long, device="cpu")

    mean = torch.zeros(n_layers, C, d, dtype=torch.float32, device="cpu")
    M2 = torch.zeros(n_layers, C, d, dtype=torch.float32, device="cpu")
    N_raw = torch.zeros(n_layers, C, dtype=torch.long, device="cpu")

    # Precompute token ids for " A"/" B"/... for speed (single-token requirement)
    letter_token_ids = {}
    for L in LETTERS:
        ids = tokenizer(f" {L}", add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f'" {L}" is not a single token for this tokenizer: {ids}')
        letter_token_ids[L] = ids[0]

    jsonl = Path(jsonl_path)
    total = 0
    used = 0

    with jsonl.open("r") as f:
        for line in f:
            if max_items and total >= max_items:
                break
            total += 1

            r = json.loads(line)
            pred = r.get("pred", "")
            if pred not in LETTERS:
                continue

            valid_letters = get_valid_letters(r)
            if pred not in valid_letters:
                continue

            if use_margin and not margin_ok(r, pred, margin_thresh):
                continue

            prompt = build_prompt_from_record(r, use_text_input=use_text_input, strip_existing=strip_existing_answer_is)

            # Teacher forcing: append predicted letter as the next token
            # text = prompt + f" {pred}"   ## 수정!!!!!
            text = prompt

            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            # Move tensors
            if device.startswith("cuda") and torch.cuda.is_available():
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = {k: v.cpu() for k, v in inputs.items()}

            pos = inputs["input_ids"].shape[1] - 1  # letter token position
            out = model(**inputs, output_hidden_states=True)

            hs = out.hidden_states  # tuple: [emb] + each layer output
            # We use transformer block outputs: hs[1:] length = n_layers
            c = LETTERS.index(pred)

            for ell in range(n_layers):
                h = hs[ell + 1][0, pos, :].float().detach().cpu()  # [d] float32

                # Unit resultant
                u = h / (h.norm(p=2) + L2_EPS)
                S_unit[ell, c] += u
                N[ell, c] += 1

                # Raw Welford
                n_prev = int(N_raw[ell, c].item())
                n_new = n_prev + 1
                N_raw[ell, c] = n_new
                delta = h - mean[ell, c]
                mean[ell, c] += delta / n_new
                delta2 = h - mean[ell, c]
                M2[ell, c] += delta * delta2

            used += 1

    # diag variance
    denom = torch.clamp(N_raw.unsqueeze(-1) - 1, min=1)
    var = M2 / denom

    stats = {
        "S_unit": S_unit,     # [L,C,D]
        "N": N,               # [L,C]
        "mean_raw": mean,     # [L,C,D]
        "var_raw": var,       # [L,C,D]
        "N_raw": N_raw,       # [L,C]
        "meta": {
            "jsonl_path": str(jsonl_path),
            "total_lines_read": total,
            "used_samples": used,
            "num_layers": n_layers,
            "hidden_size": d,
            "letters": LETTERS,
            "use_text_input": use_text_input,
            "strip_existing_answer_is": strip_existing_answer_is,
            "use_margin": use_margin,
            "margin_thresh": margin_thresh,
        }
    }
    return stats


def compute_layer_scores(stats: Dict, min_count_per_class: int, eps: float) -> Tuple[List[float], List[Dict]]:
    """
    Per layer ell:
      - valid classes: N[ell,c] >= min_count_per_class
      - intra: avg_c (1 - ||S_unit|| / N)
      - inter: avg_{c<d} (1 - cos(mu_hat_c, mu_hat_d)), where mu_hat_c = normalize(S_unit)
      - score = inter / (intra + eps)
    Returns:
      scores: list length L
      details: list length L (dict with intra, inter, used_classes, counts)
    """
    S_unit = stats["S_unit"]
    N = stats["N"]
    L, C, D = S_unit.shape

    scores = []
    details = []

    for ell in range(L):
        valid_cls = [c for c in range(C) if int(N[ell, c].item()) >= min_count_per_class]
        if len(valid_cls) < 2:
            scores.append(float("-inf"))
            details.append({
                "layer": ell,
                "score": float("-inf"),
                "intra": None,
                "inter": None,
                "used_classes": [LETTERS[c] for c in valid_cls],
                "counts": {LETTERS[c]: int(N[ell, c].item()) for c in range(C)},
            })
            continue

        mu_hat = []
        intra_terms = []
        for c in valid_cls:
            s = S_unit[ell, c]
            n = int(N[ell, c].item())
            R = float(s.norm().item() / max(n, 1))  # mean resultant length
            intra_terms.append(1.0 - R)
            mu_hat.append(s / (s.norm() + 1e-12))

        intra = float(sum(intra_terms) / len(intra_terms))

        inter_sum = 0.0
        cnt = 0
        for i in range(len(mu_hat)):
            for j in range(i + 1, len(mu_hat)):
                cos = float(torch.dot(mu_hat[i], mu_hat[j]).item())
                inter_sum += (1.0 - cos)
                cnt += 1
        inter = float(inter_sum / max(cnt, 1))

        score = float(inter / (intra + eps))
        scores.append(score)
        details.append({
            "layer": ell,
            "score": score,
            "intra": intra,
            "inter": inter,
            "used_classes": [LETTERS[c] for c in valid_cls],
            "counts": {LETTERS[c]: int(N[ell, c].item()) for c in range(C)},
        })

    return scores, details


def main():
    args = parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model, tokenizer = load_model_and_tokenizer(args.model, device=device, dtype_str=args.dtype)

    stats = collect_layer_stats(
        model=model,
        tokenizer=tokenizer,
        jsonl_path=args.jsonl,
        device=device,
        use_text_input=args.use_text_input,
        strip_existing_answer_is=args.strip_existing_answer_is,
        use_margin=args.use_margin,
        margin_thresh=args.margin_thresh,
        max_items=args.max_items if args.max_items > 0 else 0,
    )

    scores, details = compute_layer_scores(stats, min_count_per_class=args.min_count, eps=args.eps)

    # Top-K
    L = len(scores)
    order = sorted(range(L), key=lambda i: scores[i], reverse=True)

    print("\n=== Layer Search Result (stats-based, NO simulation) ===")
    print(f"Used samples: {stats['meta']['used_samples']} / read: {stats['meta']['total_lines_read']}")
    print(f"min_count_per_class: {args.min_count}, use_margin: {args.use_margin}, margin_thresh: {args.margin_thresh}\n")

    shown = 0
    for ell in order:
        if not math.isfinite(scores[ell]):
            continue
        d = details[ell]
        print(
            f"layer={ell:02d}  score={d['score']:.4f}  inter={d['inter']:.4f}  intra={d['intra']:.4f}  "
            f"classes={d['used_classes']}  counts={d['counts']}"
        )
        shown += 1
        if shown >= args.topk:
            break

    # Save
    out_path = Path(args.out)
    payload = {
        "stats": {
            "S_unit": stats["S_unit"],
            "N": stats["N"],
            "mean_raw": stats["mean_raw"],
            "var_raw": stats["var_raw"],
            "N_raw": stats["N_raw"],
        },
        "scores": torch.tensor(scores, dtype=torch.float32),
        "details": details,
        "meta": stats["meta"],
    }
    torch.save(payload, out_path)
    print(f"\n[SAVED] {out_path.resolve()}")


if __name__ == "__main__":
    main()
