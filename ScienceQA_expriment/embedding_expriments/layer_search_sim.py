#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulation-based layer search (activation steering) for Llama 3.1-8B.

Inputs:
 - train_pred_cache.jsonl  (from split_biology_by_pred.py)
 - stats_full.pt           (from layer_search_stats.py; contains S_unit/N or mean_raw)

Method:
 - For each sample:
    * Build prompt ending with "The answer is"
    * Determine current pred (from cache) and target = 2nd-best letter by pred_scores
    * Run model with a hook at layer ell:
        h_ell(letter_pos) <- h_ell(letter_pos) + alpha * dir(pred->target)
    * Check if next-token argmax among valid letters becomes target
 - Rank layers by success rate

Usage:
  python layer_search_sim.py \
    --jsonl data/scienceqa/splits_biology_pred/train_pred_cache.jsonl \
    --model meta-llama/Llama-3.1-8B \
    --stats stats_full.pt \
    --alphas 0.5 1.0 1.5 \
    --layers 0 1 2 3 4 5 6 7 8 9 10 12 14 16 \
    --max_items 300 \
    --use_text_input --strip_existing_answer_is \
    --min_count 50
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

LETTERS = ["A", "B", "C", "D"]
EPS = 1e-12


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--stats", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    ap.add_argument("--layers", type=int, nargs="+", default=list(range(0, 32)))
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--min_count", type=int, default=50)

    ap.add_argument("--use_text_input", action="store_true")
    ap.add_argument("--strip_existing_answer_is", action="store_true")

    return ap.parse_args()


def build_prompt_from_record(r: Dict, use_text_input: bool, strip_existing: bool) -> str:
    if use_text_input:
        ti = r.get("text_input", "")
        if isinstance(ti, str) and ti.strip():
            base = ti.strip()
            if strip_existing and "The answer is" in base:
                base = base.split("The answer is")[0].rstrip()
            if base.endswith("The answer is"):
                return base
            return base + "\nThe answer is"

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
        n = 4
    return LETTERS[:n]


def pick_target_second_best(r: Dict, valid_letters: List[str]) -> str:
    """
    target = 2nd-best by pred_scores among valid letters.
    pred_scores in cache should only contain valid letters already.
    """
    scores = r.get("pred_scores", {})
    if not isinstance(scores, dict):
        return ""
    items = [(L, scores.get(L, -1e30)) for L in valid_letters]
    items.sort(key=lambda x: x[1], reverse=True)
    if len(items) < 2:
        return ""
    return items[1][0]


def load_model(model_name: str, device: str, dtype_str: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if dtype_str == "float16":
        dtype = torch.float16
    elif dtype_str == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    kwargs = {}
    if device.startswith("cuda") and torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["dtype"] = dtype
    else:
        kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tok


def load_prototypes(stats_path: str, min_count: int) -> torch.Tensor:
    """
    Return proto_unit: [L,C,D] normalized class prototypes.
    Use S_unit if available; otherwise mean_raw.
    """
    x = torch.load(stats_path, map_location="cpu")
    S_unit = x["stats"]["S_unit"]  # [L,C,D]
    N = x["stats"]["N"]            # [L,C]
    proto = S_unit.clone()

    # For classes with too few samples, mark as zero vector (will be excluded)
    for ell in range(proto.shape[0]):
        for c in range(proto.shape[1]):
            if int(N[ell, c].item()) < min_count:
                proto[ell, c].zero_()

    # Normalize
    proto = proto / (proto.norm(dim=-1, keepdim=True) + EPS)
    return proto


def token_ids_for_letters(tok, valid_letters: List[str]) -> Dict[str, int]:
    out = {}
    for L in valid_letters:
        ids = tok(f" {L}", add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f'" {L}" not single token: {ids}')
        out[L] = ids[0]
    return out


class LayerSteerer:
    """
    Forward hook that adds delta to the output hidden states at a specific token position
    for a specific transformer block layer.
    """
    def __init__(self, model, layer_idx: int, token_pos: int, delta: torch.Tensor):
        self.model = model
        self.layer_idx = layer_idx
        self.token_pos = token_pos
        self.delta = delta  # [D] on same device/dtype as activations
        self.handle = None

    def _hook(self, module, inputs, output):
        # output: [B,T,D]
        # We only use B=1
        out = output
        delta = self.delta.to(device=out.device, dtype=out.dtype)
        out[:, self.token_pos, :] = out[:, self.token_pos, :] + delta
        return out

    def __enter__(self):
        layer = self.model.model.layers[self.layer_idx]
        self.handle = layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def run_sim_for_layer_alpha(
    model, tok,
    jsonl_path: str,
    proto_unit: torch.Tensor,   # [L,C,D] on CPU float32
    layer_idx: int,
    alpha: float,
    device: str,
    max_items: int,
    use_text_input: bool,
    strip_existing: bool,
) -> Dict:
    total = 0
    attempted = 0
    success = 0

    jsonl = Path(jsonl_path)
    with jsonl.open("r") as f:
        for line in f:
            if max_items and total >= max_items:
                break
            total += 1
            r = json.loads(line)

            pred = r.get("pred", "")
            valid_letters = get_valid_letters(r)
            if pred not in valid_letters or len(valid_letters) < 2:
                continue

            target = pick_target_second_best(r, valid_letters)
            if target == "" or target == pred:
                continue

            # prototypes exist?
            c_pred = LETTERS.index(pred)
            c_tgt = LETTERS.index(target)
            p_pred = proto_unit[layer_idx, c_pred]
            p_tgt = proto_unit[layer_idx, c_tgt]
            if float(p_pred.norm()) < 1e-6 or float(p_tgt.norm()) < 1e-6:
                continue

            prompt = build_prompt_from_record(r, use_text_input, strip_existing)

            inputs = tok(prompt, return_tensors="pt", add_special_tokens=False)
            if device.startswith("cuda") and torch.cuda.is_available():
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = {k: v.cpu() for k, v in inputs.items()}

            # token_pos: last token position (prompt ends with "is")
            token_pos = inputs["input_ids"].shape[1] - 1

            # direction in float32 CPU -> move to device/dtype
            d = (p_tgt - p_pred)
            d = d / (d.norm() + EPS)
            d = (alpha * d).to(device=inputs["input_ids"].device, dtype=model.dtype)

            # run with hook
            with LayerSteerer(model, layer_idx=layer_idx, token_pos=token_pos, delta=d):
                out = model(**inputs)
            logits = out.logits[0, -1, :]  # next token

            ids = token_ids_for_letters(tok, valid_letters)
            cand = torch.tensor([ids[L] for L in valid_letters], device=logits.device)
            scores = logits[cand]
            new_pred = valid_letters[int(torch.argmax(scores).item())]

            attempted += 1
            if new_pred == target:
                success += 1

    return {
        "layer": layer_idx,
        "alpha": alpha,
        "attempted": attempted,
        "success": success,
        "success_rate": (success / attempted) if attempted > 0 else 0.0,
        "read_lines": total,
    }


def main():
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model, tok = load_model(args.model, device=device, dtype_str=args.dtype)
    proto_unit = load_prototypes(args.stats, min_count=args.min_count)  # CPU float32

    results = []
    for layer in args.layers:
        for alpha in args.alphas:
            r = run_sim_for_layer_alpha(
                model=model,
                tok=tok,
                jsonl_path=args.jsonl,
                proto_unit=proto_unit,
                layer_idx=layer,
                alpha=alpha,
                device=device,
                max_items=args.max_items,
                use_text_input=args.use_text_input,
                strip_existing=args.strip_existing_answer_is,
            )
            results.append(r)
            print(
                f"layer={layer:02d} alpha={alpha:.2f} "
                f"rate={r['success_rate']:.4f} ({r['success']}/{r['attempted']})"
            )

    # rank by success_rate
    results.sort(key=lambda x: x["success_rate"], reverse=True)
    print("\n=== Top results ===")
    for r in results[:20]:
        print(
            f"layer={r['layer']:02d} alpha={r['alpha']:.2f} "
            f"rate={r['success_rate']:.4f} ({r['success']}/{r['attempted']})"
        )


if __name__ == "__main__":
    main()
