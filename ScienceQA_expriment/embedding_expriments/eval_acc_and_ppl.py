#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Wiki PPL을 원하면 datasets가 필요합니다.
# - 로컬로 저장된 HF dataset을 쓰면: datasets.load_from_disk
# - 온라인으로 wikitext-2를 다운받으면: datasets.load_dataset (인터넷 필요)
try:
    from datasets import load_from_disk, load_dataset
except Exception:
    load_from_disk = None
    load_dataset = None

LETTERS = ["A", "B", "C", "D"]

# -----------------------
# ScienceQA next-token ACC
# -----------------------
def norm_answer_to_letter(a):
    if a is None:
        return ""
    s = str(a).strip().upper()
    if s.isdigit():
        i = int(s)
        # 0-based
        if 0 <= i < 4:
            return LETTERS[i]
        # (옵션) 5지 처리 흔적 유지
        if 0 <= i < 5:
            return "ABCDE"[i]
    for ch in s:
        if ch in "ABCD":
            return ch
        if ch in "ABCDE":
            return ch
    return ""

def build_prompt(r):
    base = r.get("text_input", "")
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
    ids = []
    for L in valid:
        tid = tok(f" {L}", add_special_tokens=False).input_ids
        if len(tid) != 1:
            raise ValueError(f'" {L}" is not a single token for this tokenizer: {tid}')
        ids.append(tid[0])
    return torch.tensor(ids, dtype=torch.long)

@torch.no_grad()
def eval_scienceqa_next_token_acc(model, tok, jsonl, device="cuda", max_items=0):
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
            out = model(**inputs)
            logits = out.logits[0, -1, :]  # next-token logits

            cand = letter_token_ids(tok, v).to(device)
            scores = logits[cand]
            pred = v[int(torch.argmax(scores).item())]

            total += 1
            correct += int(pred == gt)

    return correct / max(total, 1), total

# -----------------------
# Wiki perplexity (PPL)
# -----------------------
def _get_wiki_texts(wiki_source: str, wiki_split: str):
    """
    wiki_source:
      - "wikitext-2" (온라인 다운로드; 인터넷 필요)
      - "/path/to/dataset" (datasets.load_from_disk로 로컬 로드; 인터넷 불필요)
    반환: list[str] texts
    """
    if wiki_source == "wikitext-2":
        if load_dataset is None:
            raise RuntimeError("datasets가 설치되어 있지 않습니다. `pip install datasets` 후 재시도하세요.")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=wiki_split)
        return ds["text"]
    else:
        if load_from_disk is None:
            raise RuntimeError("datasets가 설치되어 있지 않습니다. `pip install datasets` 후 재시도하세요.")
        ds = load_from_disk(wiki_source)
        # load_from_disk 결과가 DatasetDict일 수도 있고 Dataset일 수도 있음
        if isinstance(ds, dict) or hasattr(ds, "keys"):
            ds = ds[wiki_split]
        # 일반적으로 'text' 컬럼
        if "text" not in ds.column_names:
            raise KeyError(f"Loaded dataset has no 'text' column. columns={ds.column_names}")
        return ds["text"]

@torch.no_grad()
def eval_wiki_ppl(
    model,
    tok,
    wiki_source="wikitext-2",
    wiki_split="test",
    device="cuda",
    max_samples=0,
    block_size=2048,
    stride=512,
):
    """
    표준 LM PPL 측정(슬라이딩 윈도우).
    - texts를 하나로 이어붙인 뒤 tokenization
    - 길이가 긴 경우 block_size 컨텍스트로 나눠서 loss 누적
    - stride로 겹치게 하면서 평가 (Transformer LM에서 흔히 쓰는 방식)
    """
    texts = _get_wiki_texts(wiki_source, wiki_split)

    # 빈 줄이 많으면 손해라서, 너무 짧은 라인은 제거/정리(원하면 주석 처리 가능)
    texts = [t for t in texts if isinstance(t, str) and len(t.strip()) > 0]

    if max_samples and max_samples > 0:
        texts = texts[:max_samples]

    joined = "\n\n".join(texts)
    enc = tok(joined, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"][0].to(device)

    # 모델 최대 길이보다 block_size가 크면 줄임
    # (config가 없거나 max_position_embeddings가 없는 경우 대비)
    try:
        max_ctx = int(getattr(model.config, "max_position_embeddings", block_size))
        block_size = min(block_size, max_ctx)
    except Exception:
        pass

    nll_sum = 0.0
    n_tokens = 0

    # 슬라이딩 윈도우
    seq_len = input_ids.numel()
    if seq_len < 2:
        return float("nan"), 0

    prev_end = 0
    for begin in range(0, seq_len - 1, stride):
        end = min(begin + block_size, seq_len)
        trg_len = end - prev_end  # 이번 스텝에서 새로 평가할 토큰 수(겹치는 부분 제외)
        if trg_len <= 0:
            break

        input_chunk = input_ids[begin:end].unsqueeze(0)  # [1, L]
        labels = input_chunk.clone()

        # 겹치는 prefix는 loss에서 제외
        labels[:, :-trg_len] = -100

        out = model(input_ids=input_chunk, labels=labels)
        # out.loss는 (마스킹 제외한) 평균 CE loss
        # 총 NLL = loss * trg_len
        nll_sum += float(out.loss.item()) * trg_len
        n_tokens += trg_len

        prev_end = end
        if end >= seq_len:
            break

    ppl = math.exp(nll_sum / max(n_tokens, 1))
    return ppl, n_tokens

# -----------------------
# Utils
# -----------------------
def maybe_set_pad_token(tok):
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

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

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()

    # Models
    ap.add_argument("--base", required=True, help="Base model name or path")
    ap.add_argument("--lora", default="", help="LoRA adapter path (optional). If empty, only base is evaluated.")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    # ScienceQA ACC
    ap.add_argument("--jsonl", default="", help="ScienceQA jsonl for next-token ACC (optional)")
    ap.add_argument("--max_items", type=int, default=0, help="ScienceQA: 0 means all")

    # Wiki PPL
    ap.add_argument("--do_wiki", action="store_true", help="If set, compute Wiki perplexity")
    ap.add_argument("--wiki_source", default="wikitext-2",
                    help='Either "wikitext-2" (downloads) or a local dataset path for load_from_disk')
    ap.add_argument("--wiki_split", default="test", help='Usually "test"')
    ap.add_argument("--wiki_max_samples", type=int, default=0, help="0 means use all texts")
    ap.add_argument("--wiki_block_size", type=int, default=2048)
    ap.add_argument("--wiki_stride", type=int, default=512)

    args = ap.parse_args()

    dtype = parse_dtype(args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    maybe_set_pad_token(tok)

    # 1) BASE
    base_model = load_base_model(args.base, dtype=dtype, device=device)
    base_model.eval()

    if args.jsonl:
        base_acc, base_n = eval_scienceqa_next_token_acc(
            base_model, tok, args.jsonl, device=device, max_items=args.max_items
        )
        print(f"[BASE] ScienceQA ACC(next-token, letters-only) = {base_acc:.4f}  (N={base_n})")

    if args.do_wiki:
        base_ppl, base_tok = eval_wiki_ppl(
            base_model, tok,
            wiki_source=args.wiki_source,
            wiki_split=args.wiki_split,
            device=device,
            max_samples=args.wiki_max_samples,
            block_size=args.wiki_block_size,
            stride=args.wiki_stride,
        )
        print(f"[BASE] Wiki PPL = {base_ppl:.4f}  (tokens={base_tok})")

    # 2) LoRA (optional)
    if args.lora and args.lora.strip():
        lora_model = PeftModel.from_pretrained(base_model, args.lora)
        lora_model.eval()

        if args.jsonl:
            lora_acc, lora_n = eval_scienceqa_next_token_acc(
                lora_model, tok, args.jsonl, device=device, max_items=args.max_items
            )
            print(f"[LoRA] ScienceQA ACC(next-token, letters-only) = {lora_acc:.4f}  (N={lora_n})")
            print(f"[DELTA] ACC(LoRA - BASE) = {lora_acc - base_acc:+.4f}")

        if args.do_wiki:
            lora_ppl, lora_tok = eval_wiki_ppl(
                lora_model, tok,
                wiki_source=args.wiki_source,
                wiki_split=args.wiki_split,
                device=device,
                max_samples=args.wiki_max_samples,
                block_size=args.wiki_block_size,
                stride=args.wiki_stride,
            )
            print(f"[LoRA] Wiki PPL = {lora_ppl:.4f}  (tokens={lora_tok})")
            print(f"[DELTA] PPL(LoRA - BASE) = {lora_ppl - base_ppl:+.4f}")

if __name__ == "__main__":
    main()
