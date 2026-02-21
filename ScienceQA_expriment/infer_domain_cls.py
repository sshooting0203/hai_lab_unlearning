#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
infer_domain_cls.py

Evaluate a trained domain classifier (10 topics + others) properly:
- Loads model/tokenizer from a checkpoint dir (e.g., ./domain_cls_checkpoints)
- Reads ScienceQA train json files from a directory (scienceqa_train_*.json)
  * Supports: single JSON, JSON list, JSONL, concatenated JSON objects
- Builds input text in the SAME way as training (question + choices + optional hint/lecture/meta)
- Uses TRUE label from filename topic mapping (10 topics + others)
- Reports:
  * overall accuracy
  * macro-F1 / weighted-F1
  * per-class precision/recall/F1/support
  * confusion matrix (printed + optional csv)
  * optional "no-hint" evaluation by disabling hint at inference

Usage examples:
  python infer_domain_cls.py \
    --ckpt_dir ./domain_cls_checkpoints \
    --data_dir ./new_data/train \
    --max_seq_length 256 \
    --include_hint

  # Evaluate without hint (sanity-check generalization):
  python infer_domain_cls.py \
    --ckpt_dir ./domain_cls_checkpoints \
    --data_dir ./new_data/train \
    --max_seq_length 256 \
    --no_hint

  # Save confusion matrix csv:
  python infer_domain_cls.py ... --save_confusion_csv ./cm.csv
"""

import argparse
import os
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# -----------------------------
# 1) Label mapping: MUST match training
# -----------------------------
TARGET_TOPICS = [
    "biology",
    "physics",
    "geography",
    "writing-strategies",
    "figurative-language",
    "chemistry",
    "earth-science",
    "science-and-engineering-practices",
    "units-and-measurement",
    "economics",
]
LABEL2ID = {name: i for i, name in enumerate(TARGET_TOPICS)}
OTHERS_ID = len(TARGET_TOPICS)  # 10
NUM_LABELS = len(TARGET_TOPICS) + 1  # 11


def topic_from_filename(fname: str) -> str:
    # scienceqa_train_biology.json -> biology
    base = os.path.basename(fname)
    if not base.startswith("scienceqa_train_") or not base.endswith(".json"):
        raise ValueError(f"Unexpected filename format: {fname}")
    return base.replace("scienceqa_train_", "").replace(".json", "")


# -----------------------------
# 2) Robust reader for JSON / JSONL / concatenated JSON objects
# -----------------------------
def read_json_any(path: str) -> List[Dict[str, Any]]:
    """
    Robust reader for:
      (a) a single JSON object
      (b) a JSON list
      (c) JSON Lines (one JSON object per line)
      (d) multiple JSON objects concatenated with whitespace/newlines: {...}\n{...}

    Returns: list of dict samples
    """
    with open(path, "r") as f:
        text = f.read().strip()

    if not text:
        return []

    # (a)/(b): try normal json first
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if "data" in obj and isinstance(obj["data"], list):
                return obj["data"]
            return [obj]
        return []
    except json.JSONDecodeError:
        pass

    # (c): JSONL
    samples = []
    ok = True
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            ok = False
            break
    if ok and len(samples) > 0:
        return samples

    # (d): concatenated JSON objects
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    samples = []
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        samples.append(obj)
        idx = end

    out: List[Dict[str, Any]] = []
    for obj in samples:
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            out.extend(obj["data"])
        elif isinstance(obj, dict):
            out.append(obj)
        elif isinstance(obj, list):
            out.extend(obj)
    return out


# -----------------------------
# 3) Text builder (must match training options)
# -----------------------------
def build_text(
    example: Dict[str, Any],
    include_hint: bool = True,
    include_lecture: bool = False,
    include_meta: bool = False,
) -> str:
    q = example.get("question", "")
    choices = example.get("choices", [])

    choices_block = ""
    if isinstance(choices, list) and len(choices) > 0:
        ch = "\n".join([f"({i}) {str(c)}" for i, c in enumerate(choices)])
        choices_block = f"Choices:\n{ch}"
    elif choices:
        choices_block = f"Choices:\n{str(choices)}"

    parts = []
    if q:
        parts.append(f"Question:\n{q}")
    if choices_block:
        parts.append(choices_block)

    if include_hint:
        hint = example.get("hint", "")
        if hint:
            parts.append(f"Hint:\n{hint}")

    if include_lecture:
        lecture = example.get("lecture", "")
        if lecture:
            parts.append(f"Lecture:\n{lecture}")

    if include_meta:
        # Meta can artificially inflate results if it's correlated with label.
        task = example.get("task", "")
        grade = example.get("grade", "")
        subject = example.get("subject", "")
        category = example.get("category", "")
        skill = example.get("skill", "")
        meta = []
        if task:
            meta.append(f"task={task}")
        if grade:
            meta.append(f"grade={grade}")
        if subject:
            meta.append(f"subject={subject}")
        if category:
            meta.append(f"category={category}")
        if skill:
            meta.append(f"skill={skill}")
        if meta:
            parts.append("Meta:\n" + ", ".join(meta))

    text = "\n\n".join(parts).strip()
    if not text:
        text = json.dumps(example, ensure_ascii=False)[:2000]
    return text


# -----------------------------
# 4) Batched inference
# -----------------------------
def predict_batch(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    max_seq_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      pred_ids: (B,)
      pred_probs: (B,) probability of predicted class
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits  # (B, C)
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)
        pred_probs = probs[torch.arange(probs.size(0)), pred_ids]

    return pred_ids.detach().cpu().numpy(), pred_probs.detach().cpu().numpy()


def save_confusion_csv(cm: np.ndarray, labels: List[str], out_path: str):
    # simple CSV writer without pandas dependency
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, row in enumerate(cm):
            f.write(labels[i] + "," + ",".join(map(str, row.tolist())) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, type=str, help="checkpoint dir saved by train_domain_cls.py")
    parser.add_argument("--data_dir", required=True, type=str, help="dir containing scienceqa_train_*.json")
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--batch_size", default=32, type=int)

    # match training options
    parser.add_argument("--include_hint", action="store_true", help="include hint in input text")
    parser.add_argument("--include_lecture", action="store_true", help="include lecture in input text")
    parser.add_argument("--include_meta", action="store_true", help="include meta fields in input text")

    # sanity checks
    parser.add_argument("--no_hint", action="store_true", help="force disable hint (overrides --include_hint)")
    parser.add_argument("--limit_per_file", default=0, type=int, help="debug: cap samples per file (0 = no cap)")
    parser.add_argument("--save_confusion_csv", default="", type=str, help="path to save confusion matrix csv")
    parser.add_argument("--save_errors_jsonl", default="", type=str, help="save misclassified samples (jsonl)")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load label_map.json if exists (for consistency / display)
    label_map_path = os.path.join(args.ckpt_dir, "label_map.json")
    if os.path.exists(label_map_path):
        with open(label_map_path, "r") as f:
            lm = json.load(f)
        # Prefer checkpoint mapping, but fall back to hard-coded if missing.
        ckpt_label2id = lm.get("LABEL2ID", None)
        ckpt_others = lm.get("OTHERS_ID", OTHERS_ID)
        if isinstance(ckpt_label2id, dict):
            # convert to int ids (json loads ints fine, but keys are strings)
            id2label = {int(v): k for k, v in ckpt_label2id.items()}
            id2label[int(ckpt_others)] = "others"
        else:
            id2label = {v: k for k, v in LABEL2ID.items()}
            id2label[OTHERS_ID] = "others"
    else:
        id2label = {v: k for k, v in LABEL2ID.items()}
        id2label[OTHERS_ID] = "others"

    labels_order = [id2label[i] for i in range(NUM_LABELS)]

    tokenizer = RobertaTokenizer.from_pretrained(args.ckpt_dir)
    model = RobertaForSequenceClassification.from_pretrained(args.ckpt_dir).to(device)
    model.eval()

    include_hint = args.include_hint and (not args.no_hint)

    # gather files
    files = sorted([f for f in os.listdir(args.data_dir) if f.startswith("scienceqa_train_") and f.endswith(".json")])
    if len(files) == 0:
        raise FileNotFoundError(f"No scienceqa_train_*.json found in {args.data_dir}")

    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[float] = []
    all_meta: List[Dict[str, Any]] = []  # for saving errors

    for fname in files:
        path = os.path.join(args.data_dir, fname)
        topic = topic_from_filename(fname)
        y_true = LABEL2ID[topic] if topic in LABEL2ID else OTHERS_ID

        samples = read_json_any(path)
        if args.limit_per_file > 0:
            samples = samples[: args.limit_per_file]

        # build texts
        texts = [
            build_text(
                ex,
                include_hint=include_hint,
                include_lecture=args.include_lecture,
                include_meta=args.include_meta,
            )
            for ex in samples
        ]

        # batched predict
        for i in tqdm(range(0, len(texts), args.batch_size), desc=f"Eval {fname}"):
            batch_texts = texts[i : i + args.batch_size]
            pred_ids, pred_probs = predict_batch(
                model=model,
                tokenizer=tokenizer,
                texts=batch_texts,
                device=device,
                max_seq_length=args.max_seq_length,
            )

            all_true.extend([y_true] * len(pred_ids))
            all_pred.extend(pred_ids.tolist())
            all_prob.extend(pred_probs.tolist())

            if args.save_errors_jsonl:
                # store minimal meta to reconstruct
                for j in range(len(pred_ids)):
                    ex = samples[i + j]
                    all_meta.append(
                        {
                            "source_file": fname,
                            "true_id": y_true,
                            "true_label": id2label[y_true],
                            "pred_id": int(pred_ids[j]),
                            "pred_label": id2label[int(pred_ids[j])],
                            "pred_prob": float(pred_probs[j]),
                            "question": ex.get("question", ""),
                            "choices": ex.get("choices", []),
                            "hint": ex.get("hint", ""),
                        }
                    )

    y_true_arr = np.array(all_true, dtype=np.int64)
    y_pred_arr = np.array(all_pred, dtype=np.int64)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    macro_f1 = f1_score(y_true_arr, y_pred_arr, average="macro")
    weighted_f1 = f1_score(y_true_arr, y_pred_arr, average="weighted")

    print("\n==================== Overall ====================")
    print(f"Samples: {len(y_true_arr)}")
    print(f"Accuracy   : {acc:.6f}")
    print(f"Macro-F1   : {macro_f1:.6f}")
    print(f"Weighted-F1: {weighted_f1:.6f}")
    print("Include hint   :", include_hint)
    print("Include lecture:", args.include_lecture)
    print("Include meta   :", args.include_meta)

    print("\n================ Per-class report ================")
    # ensure stable order by passing labels
    print(
        classification_report(
            y_true_arr,
            y_pred_arr,
            labels=list(range(NUM_LABELS)),
            target_names=labels_order,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(NUM_LABELS)))
    print("\n================ Confusion matrix ================")
    print("Rows=true, Cols=pred. Label order:", labels_order)
    print(cm)

    if args.save_confusion_csv:
        save_confusion_csv(cm, labels_order, args.save_confusion_csv)
        print(f"[Saved] confusion matrix csv -> {args.save_confusion_csv}")

    if args.save_errors_jsonl:
        # write only errors
        with open(args.save_errors_jsonl, "w", encoding="utf-8") as w:
            for m in all_meta:
                if m["true_id"] != m["pred_id"]:
                    w.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"[Saved] errors jsonl -> {args.save_errors_jsonl}")

    # Additional: confidence stats
    probs_arr = np.array(all_prob, dtype=np.float32)
    correct = (y_true_arr == y_pred_arr)
    if probs_arr.size > 0:
        print("\n================ Confidence stats ================")
        print(f"Mean pred prob (all)     : {probs_arr.mean():.4f}")
        print(f"Mean pred prob (correct) : {probs_arr[correct].mean():.4f}" if correct.any() else "N/A")
        print(f"Mean pred prob (wrong)   : {probs_arr[~correct].mean():.4f}" if (~correct).any() else "N/A")


if __name__ == "__main__":
    main()
