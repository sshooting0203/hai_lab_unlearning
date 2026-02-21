import argparse, json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

LETTERS = ["A","B","C","D"]

def norm_answer_to_letter(a):
    if a is None: return ""
    s = str(a).strip().upper()
    if s.isdigit():
        i = int(s)
        if 0 <= i < 4: return LETTERS[i]
        if 0 <= i < 5: return "ABCDE"[i]
    for ch in s:
        if ch in "ABCD": return ch
        if ch in "ABCDE": return ch
    return ""

def build_prompt(r):
    base = r.get("text_input","")
    base = base.strip()
    if "The answer is" in base:
        base = base.split("The answer is")[0].rstrip()
    return base + "\nThe answer is"

def valid_letters(r):
    ch = r.get("choices", [])
    n = len(ch) if isinstance(ch, list) else 0
    n = max(0, min(n, 4))
    return LETTERS[:n]

def letter_token_ids(tok, valid):
    ids=[]
    for L in valid:
        tid = tok(f" {L}", add_special_tokens=False).input_ids
        assert len(tid)==1
        ids.append(tid[0])
    return torch.tensor(ids, dtype=torch.long)

@torch.no_grad()
def eval_file(model, tok, jsonl, device="cuda", max_items=0):
    correct=0; total=0
    with open(jsonl) as f:
        for i,line in enumerate(f):
            if max_items and i>=max_items: break
            r=json.loads(line)
            v = valid_letters(r)
            if len(v) < 2: 
                continue
            gt = norm_answer_to_letter(r.get("answer"))
            if gt not in v:
                continue

            prompt = build_prompt(r)
            inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            out = model(**inputs)
            logits = out.logits[0, -1, :]

            cand = letter_token_ids(tok, v).to(device)
            scores = logits[cand]
            pred = v[int(torch.argmax(scores).item())]

            total += 1
            correct += int(pred == gt)
    return correct/ max(total,1), total

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--max_items", type=int, default=0)
    args=ap.parse_args()

    if args.dtype=="float16": dtype=torch.float16
    elif args.dtype=="bfloat16": dtype=torch.bfloat16
    else: dtype=torch.float32

    device="cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(args.base, device_map="auto" if device=="cuda" else None, dtype=dtype)
    model = PeftModel.from_pretrained(base, args.lora)
    model.eval()

    acc, n = eval_file(model, tok, args.jsonl, device=device, max_items=args.max_items)
    print(f"ACC(next-token, valid letters only) = {acc:.4f}  (N={n})")

if __name__=="__main__":
    main()
