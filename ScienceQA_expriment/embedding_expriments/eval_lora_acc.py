#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

LETTERS = ["A", "B", "C", "D"]  # 4지선다 기준

def norm_answer_to_letter(a):
    if a is None:
        return ""
    s = str(a).strip().upper()

    # 숫자 레이블 방어적 처리: 0-based(0~3) 또는 1-based(1~4)
    if s.isdigit():
        i = int(s)
        if 0 <= i < 4:
            return LETTERS[i]
        if 1 <= i <= 4:
            return LETTERS[i - 1]

    # 문자열 내 첫 A/B/C/D
    for ch in s:
        if ch in "ABCD":
            return ch
    return ""

def build_prompt(r):
    """
    r["text_input"]을 우선 사용.
    이미 'The answer is'가 포함되어 있으면 그 앞까지만 사용.
    항상 마지막이 '\\nThe answer is'로 끝나도록 구성.
    """
    base = r.get("text_input", "")
    base = base.strip()
    if "The answer is" in base:
        base = base.split("The answer is")[0].rstrip()
    return base + "\nThe answer is"

def valid_letters(r):
    ch = r.get("choices", [])
    n = len(ch) if isinstance(ch, list) else 0
    n = max(0, min(n, len(LETTERS)))
    return LETTERS[:n]

def parse_generated_letter(text: str) -> str:
    """
    자유 생성 continuation에서 A/B/C/D 중 첫 번째로 등장하는 letter를 추출.
    """
    if text is None:
        return ""
    m = re.search(r"\b([ABCD])\b", text.upper())
    return m.group(1) if m else ""

@torch.no_grad()
def eval_file_generate(model, tok, jsonl, device="cuda", max_items=0, max_new_tokens=8, print_samples=0):
    """
    'The answer is' 이후를 제한 없이 generate()로 생성하고,
    생성된 텍스트에서 A/B/C/D를 파싱해 정답과 비교하는 accuracy.
    """
    correct = 0
    total = 0

    with open(jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_items and i >= max_items:
                break
            r = json.loads(line)

            v = valid_letters(r)
            if len(v) < 2:
                continue

            gt = norm_answer_to_letter(r.get("answer"))
            if gt not in v:
                continue

            prompt = build_prompt(r)
            inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)

            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy decoding (재현성)
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )

            prompt_len = inputs["input_ids"].shape[1]
            cont_ids = gen_ids[0, prompt_len:]
            cont_text = tok.decode(cont_ids, skip_special_tokens=True)

            pred = parse_generated_letter(cont_text)

            total += 1
            correct += int(pred == gt)
            
            # -------- 샘플 출력 --------
            if print_samples > 0 and total <= print_samples:
                print("\n" + "=" * 80)
                print(f"SAMPLE {total}")
                print("- PROMPT (tail) ----------------")
                print(prompt[-500:])   # 너무 길면 뒤쪽만
                print("- GOLD -------------------------")
                print(gt)
                print("- GENERATED CONTINUATION -------")
                print(repr(cont_text))
                print("- PARSED PRED ------------------")
                print(pred if pred else "<INVALID>")
                print("=" * 80)

    acc = correct / max(total, 1)
    return acc, total

def maybe_set_pad_token(tok):
    # LLaMA 계열에서 pad_token이 None인 경우가 있어 방어
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

def load_base_model(base_name_or_path, dtype, device):
    return AutoModelForCausalLM.from_pretrained(
        base_name_or_path,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype,
    )

def append_jsonl(path: str, record: dict):
    """
    JSONL(한 줄에 한 JSON)로 결과를 append 저장.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print_samples", type=int, default=0,
                    help="If >0, print first K samples (prompt / gold / generated / parsed pred)")
    ap.add_argument("--base", required=True, help="Base model name or path")
    ap.add_argument("--jsonl", required=True, help="Evaluation jsonl file")
    ap.add_argument("--lora", default="", help="LoRA adapter path (optional). If empty, only base is evaluated.")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--max_items", type=int, default=0, help="0 means all")
    ap.add_argument("--max_new_tokens", type=int, default=8, help="Tokens to generate after 'The answer is'")
    ap.add_argument("--out_jsonl", default="eval_results.jsonl",
                    help="Output JSONL path to append results (default: eval_results.jsonl)")
    args = ap.parse_args()

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    maybe_set_pad_token(tok)

    # 1) Base 로드 (LoRA 평가도 base 위에 얹어서 하므로 필수)
    base_model = load_base_model(args.base, dtype=dtype, device=device)
    base_model.eval()

    # 2) LoRA 평가(옵션)
    if args.lora and args.lora.strip():
        lora_model = PeftModel.from_pretrained(base_model, args.lora)
        lora_model.eval()

        lora_acc, lora_n = eval_file_generate(
            lora_model, tok, args.jsonl, device=device,
            max_items=args.max_items, max_new_tokens=args.max_new_tokens,
            print_samples=args.print_samples,
        )

        # 콘솔 출력
        print(f"[LoRA] ACC(generation, no restriction) = {lora_acc:.4f}  (N={lora_n})")

        # JSONL 저장 (방법 3)
        record = {
            "model": "lora",
            "base": args.base,
            "lora": args.lora,
            "jsonl": args.jsonl,
            "dtype": args.dtype,
            "max_items": args.max_items,
            "max_new_tokens": args.max_new_tokens,
            "acc": float(f"{lora_acc:.6f}"),
            "n": int(lora_n),
        }
        append_jsonl(args.out_jsonl, record)
        print(f"[SAVED] appended to {args.out_jsonl}")

    else:
        print("[INFO] --lora is empty. Nothing to evaluate (base eval is currently disabled in this script).")

if __name__ == "__main__":
    main()
