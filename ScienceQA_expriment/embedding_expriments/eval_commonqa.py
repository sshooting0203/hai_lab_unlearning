#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CommonQA generation-based evaluation (BASE + LoRA) with tqdm progress bars.

Adds:
- tqdm progress per split (shows processed count, EM/F1 running avg)
- optional --log_every to print periodic status lines
"""

import argparse
import re
import string
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk
from tqdm import tqdm


# -------------------------
# Prompt building
# -------------------------
def build_prompt(ex, prompt_style="answer_colon"):
    ti = ex.get("text_input", "")
    if isinstance(ti, str) and ti.strip():
        base = ti.strip()

        if prompt_style == "the_answer_is":
            if "The answer is" in base:
                base = base.split("The answer is")[0].rstrip()
            return base + "\nThe answer is"
        else:
            for cue in ["\nAnswer:", "\nAnswer", "\nA:", "\nResponse:", "\nThe answer is"]:
                if cue in base:
                    base = base.split(cue)[0].rstrip()
                    break
            return base + "\nAnswer:"

    inst = ex.get("instruction", "")
    inp = ex.get("input", "")
    inst = inst.strip() if isinstance(inst, str) else ""
    inp = inp.strip() if isinstance(inp, str) else ""

    parts = []
    if inst:
        parts.append(inst)
    if inp:
        parts.append(inp)

    base = "\n\n".join(parts).strip()
    if not base:
        tf = ex.get("text_full", "")
        base = tf.strip() if isinstance(tf, str) else ""

    if prompt_style == "the_answer_is":
        return (base + "\nThe answer is").strip()
    else:
        return (base + "\nAnswer:").strip()


def get_ground_truth(ex):
    ans = ex.get("answer", "")
    if isinstance(ans, str) and ans.strip():
        return ans.strip()
    out = ex.get("output", "")
    if isinstance(out, str) and out.strip():
        return out.strip()
    return ""


# -------------------------
# Normalization + metrics
# -------------------------
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES.sub(" ", s)
    s = " ".join(s.split())
    return s

def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_text(pred).split()
    gold_toks = normalize_text(gold).split()
    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


# -------------------------
# Generation + parsing
# -------------------------
def maybe_set_pad_token(tok):
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

def parse_generation_as_answer(cont_text: str):
    if cont_text is None:
        return ""
    t = cont_text.strip()
    if "\n" in t:
        t = t.split("\n", 1)[0].strip()
    for prefix in ["Answer:", "answer:", "A:", "Response:", "response:", "The answer is", "the answer is"]:
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    return t

@torch.no_grad()
def generate_answer(model, tok, prompt: str, device: str, max_new_tokens: int, max_prompt_tokens: int):
    enc = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
        add_special_tokens=False,
    ).to(device)

    gen_ids = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )

    prompt_len = enc["input_ids"].shape[1]
    cont_ids = gen_ids[0, prompt_len:]
    cont_text = tok.decode(cont_ids, skip_special_tokens=True)
    return parse_generation_as_answer(cont_text)


# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def eval_split(
    model,
    tok,
    ds_split,
    device: str,
    split_name: str,
    max_items: int = 0,
    max_new_tokens: int = 32,
    max_prompt_tokens: int = 2048,
    prompt_style: str = "answer_colon",
    log_every: int = 0,
    print_samples: int = 0,
):
    em_sum = 0.0
    f1_sum = 0.0
    n = 0

    total = len(ds_split)
    limit = total if (max_items is None or max_items == 0) else min(total, max_items)

    pbar = tqdm(range(limit), desc=f"{split_name}", dynamic_ncols=True, leave=True)
    for i in pbar:
        ex = ds_split[i]
        gold = get_ground_truth(ex)
        if not gold:
            # skip, but update bar
            pbar.set_postfix_str(f"N={n} EM={em_sum/max(n,1):.3f} F1={f1_sum/max(n,1):.3f} (skip)")
            continue

        prompt = build_prompt(ex, prompt_style=prompt_style)
        pred = generate_answer(
            model,
            tok,
            prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            max_prompt_tokens=max_prompt_tokens,
        )

        em = exact_match(pred, gold)
        f1 = token_f1(pred, gold)

        em_sum += em
        f1_sum += f1
        n += 1

        # -------- 샘플 출력 --------
        if print_samples > 0 and n <= print_samples:
            tqdm.write("\n" + "=" * 80)
            tqdm.write(f"[{split_name}] SAMPLE {n}")
            tqdm.write("- PROMPT (tail) ----------------")
            tqdm.write(prompt[-500:])   # 너무 길면 뒤쪽만
            tqdm.write("- GOLD -------------------------")
            tqdm.write(gold)
            tqdm.write("- PRED -------------------------")
            tqdm.write(pred if pred else "<EMPTY>")
            tqdm.write("=" * 80)


        pbar.set_postfix_str(f"N={n} EM={em_sum/max(n,1):.3f} F1={f1_sum/max(n,1):.3f}")

        if log_every and n > 0 and (n % log_every == 0):
            tqdm.write(f"[{split_name}] step={n} EM={em_sum/max(n,1):.4f} F1={f1_sum/max(n,1):.4f}")

    return {"EM": em_sum / max(n, 1), "F1": f1_sum / max(n, 1), "N": n}


def parse_dtype(s: str):
    s = s.lower().strip()
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    raise ValueError(s)

def load_base_model(base_name_or_path, dtype, device):
    return AutoModelForCausalLM.from_pretrained(
        base_name_or_path,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", default="data/commonqa/commonqa_all.hf")
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    ap.add_argument("--max_items", type=int, default=0, help="0 means all items per split")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--prompt_style", default="answer_colon", choices=["answer_colon", "the_answer_is"])
    ap.add_argument("--log_every", type=int, default=0, help="If >0, prints status every N evaluated examples")
    ap.add_argument("--print_samples", type=int, default=0,
                help="If >0, print first K samples (prompt / gold / pred)")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = parse_dtype(args.dtype)

    ds = load_from_disk(args.dataset_path)

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    maybe_set_pad_token(tok)

    # -------- BASE --------
    base_model = load_base_model(args.base, dtype=dtype, device=device)
    base_model.eval()

    base_results = {}
    print("\n=== Evaluating BASE ===")
    for split in ["test"]:
        if split not in ds:
            continue
        base_results[split] = eval_split(
            base_model, tok, ds[split], device=device,
            split_name=f"BASE/{split}",
            max_items=args.max_items,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            prompt_style=args.prompt_style,
            log_every=args.log_every,
            print_samples=args.print_samples,
        )   

    print("\n=== BASE (generation EM/F1) ===")
    for split, m in base_results.items():
        print(f"[{split:10s}] EM={m['EM']:.4f}  F1={m['F1']:.4f}  N={m['N']}")

    # -------- LoRA (optional) --------
    if args.lora and args.lora.strip():
        lora_model = PeftModel.from_pretrained(base_model, args.lora)
        lora_model.eval()

        lora_results = {}
        print("\n=== Evaluating LoRA ===")
        for split in ["test"]:
            if split not in ds:
                continue
            lora_results[split] = eval_split(
                lora_model, tok, ds[split], device=device,
                split_name=f"LoRA/{split}",
                max_items=args.max_items,
                max_new_tokens=args.max_new_tokens,
                max_prompt_tokens=args.max_prompt_tokens,
                prompt_style=args.prompt_style,
                log_every=args.log_every,
                print_samples=args.print_samples,
            )

        print("\n=== LoRA (generation EM/F1) ===")
        for split, m in lora_results.items():
            print(f"[{split:10s}] EM={m['EM']:.4f}  F1={m['F1']:.4f}  N={m['N']}")

        print("\n=== DELTA (LoRA - BASE) ===")
        for split in lora_results.keys():
            b = base_results[split]
            l = lora_results[split]
            print(
                f"[{split:10s}] "
                f"ΔEM={l['EM'] - b['EM']:+.4f}  "
                f"ΔF1={l['F1'] - b['F1']:+.4f}  "
                f"(N_base={b['N']}, N_lora={l['N']})"
            )


if __name__ == "__main__":
    main()
