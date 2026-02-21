#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a ScienceQA domain classifier: 10 topics + others (total 11 classes)
- Reads ./new_data/train/scienceqa_train_*.json
- Maps the 10 target topics to class IDs [0..9], everything else -> others (10)
- Trains RoBERTa sequence classifier with CrossEntropy
- Stratified train/val split
- Optional class-weighting for imbalance
- Optional inclusion of hint / lecture / meta fields in the input text

Recommended:
  python train_domain_cls.py --train_dir ./new_data/train --output_dir ./domain_cls_checkpoints --use_class_weight
"""
import argparse, os, json, random, warnings, torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
warnings.filterwarnings("ignore")
torch.set_num_threads(10)

# -----------------------------
# 1) Label mapping: 10 topics + others = 11
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


# -----------------------------
# 2) Reproducibility
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# 3) Filename -> topic
# -----------------------------
def topic_from_filename(fname: str) -> str:
    # scienceqa_train_biology.json -> biology
    base = os.path.basename(fname)
    if not base.startswith("scienceqa_train_") or not base.endswith(".json"):
        raise ValueError(f"Unexpected filename format: {fname}")
    return base.replace("scienceqa_train_", "").replace(".json", "")


# -----------------------------
# 4) Text builder (맞춤형: 현재 제공한 JSON 스키마 기준)
# -----------------------------
def build_text(
    example: Dict[str, Any],
    include_hint: bool = True,
    include_lecture: bool = False,
    include_meta: bool = False,
) -> str:
    """
    Domain classification input text builder for ScienceQA-like samples.

    Default (recommended): question + choices + hint
    Exclude lecture/solution/answer to avoid leakage and overly strong topic cues.

    example keys observed:
      question, choices, answer, hint, task, grade, subject, topic, category, skill, lecture, solution
    """
    q = example.get("question", "")
    choices = example.get("choices", [])

    # choices formatting
    choices_block = ""
    if isinstance(choices, list) and len(choices) > 0:
        ch = "\n".join([f"({i}) {str(c)}" for i, c in enumerate(choices)])
        choices_block = f"Choices:\n{ch}"

    parts = []
    if q:
        parts.append(f"Question:\n{q}")
    if choices_block:
        parts.append(choices_block)

    if include_hint:
        hint = example.get("hint", "")
        if hint:
            parts.append(f"Hint:\n{hint}")

    # NOTE: lecture is often long and can dominate signals; keep off by default
    if include_lecture:
        lecture = example.get("lecture", "")
        if lecture:
            parts.append(f"Lecture:\n{lecture}")

    # Meta fields can inflate accuracy but may be unavailable at inference time.
    if include_meta:
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

    # DO NOT include: answer / solution / topic (label leakage)
    text = "\n\n".join(parts).strip()
    if not text:
        text = json.dumps(example, ensure_ascii=False)[:2000]
    return text


# -----------------------------
# 5) Dataset + Collator
# -----------------------------
class DomainJsonDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], tokenizer: RobertaTokenizer, max_len: int):
        """
        items: list of (text, label_id)
        """
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text, y = self.items[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,  # padding in collator
            return_tensors=None,
        )
        enc["labels"] = y
        return enc


@dataclass
class DataCollatorWithPadding:
    tokenizer: RobertaTokenizer
    max_length: int

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        for f in features:
            f.pop("labels")

        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch["labels"] = labels
        return batch



def read_json_any(path: str):
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
            # if wrapped
            if "data" in obj and isinstance(obj["data"], list):
                return obj["data"]
            return [obj]
        # fallback
        return []
    except json.JSONDecodeError:
        pass

    # (c): JSONL (line-delimited)
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

    # (d): concatenated JSON objects: {...}\n{...}
    # Use JSONDecoder raw_decode to consume sequential objects
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    samples = []
    while idx < n:
        # skip whitespace
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        samples.append(obj)
        idx = end
    # unwrap if needed
    out = []
    for obj in samples:
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            out.extend(obj["data"])
        elif isinstance(obj, dict):
            out.append(obj)
        elif isinstance(obj, list):
            out.extend(obj)
    return out



# -----------------------------
# 6) Load all jsons and label by filename topic
# -----------------------------
def load_domain_items(
    train_dir: str,
    include_hint: bool,
    include_lecture: bool,
    include_meta: bool,
) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"train_dir not found: {train_dir}")

    files = sorted([f for f in os.listdir(train_dir) if f.startswith("scienceqa_train_") and f.endswith(".json")])
    if len(files) == 0:
        raise FileNotFoundError(f"No scienceqa_train_*.json found under: {train_dir}")

    for fname in files:
        path = os.path.join(train_dir, fname)
        topic = topic_from_filename(fname)
        y = LABEL2ID[topic] if topic in LABEL2ID else OTHERS_ID

        data = read_json_any(path)  # <-- 핵심 변경

        for ex in data:
            text = build_text(
                ex,
                include_hint=include_hint,
                include_lecture=include_lecture,
                include_meta=include_meta,
            )
            items.append((text, y))

    return items



# -----------------------------
# 7) Split / imbalance helpers
# -----------------------------
def stratified_split(items: List[Tuple[str, int]], val_ratio: float, seed: int):
    """
    Simple stratified split by label.
    """
    rng = np.random.RandomState(seed)
    by_label: Dict[int, List[Tuple[str, int]]] = {}
    for text, y in items:
        by_label.setdefault(y, []).append((text, y))

    train_items, val_items = [], []
    for y, group in by_label.items():
        group = list(group)
        rng.shuffle(group)
        # keep val non-empty only if enough samples
        if len(group) >= 20:
            n_val = max(1, int(len(group) * val_ratio))
        else:
            n_val = max(0, int(len(group) * val_ratio))

        val_items.extend(group[:n_val])
        train_items.extend(group[n_val:])

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items


def compute_class_weights(train_items: List[Tuple[str, int]]) -> torch.Tensor:
    """
    class weights for imbalance; helpful if 'others' dominates.
    weight[c] = total / (num_classes * count[c])
    """
    counts = np.zeros(NUM_LABELS, dtype=np.int64)
    for _, y in train_items:
        counts[y] += 1

    total = counts.sum()
    weights = total / (NUM_LABELS * np.clip(counts, 1, None))
    return torch.tensor(weights, dtype=torch.float)


def label_stats(items: List[Tuple[str, int]]) -> Dict[int, int]:
    stats = {i: 0 for i in range(NUM_LABELS)}
    for _, y in items:
        stats[y] += 1
    return stats


# -----------------------------
# 8) Train / Eval
# -----------------------------
def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).float().mean().item()


def evaluate(args, model, dataloader) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            logits = out.logits
            acc = accuracy_from_logits(logits, batch["labels"])

            total_loss += loss.item()
            total_acc += acc
            n_steps += 1

    return {
        "loss": total_loss / max(1, n_steps),
        "acc": total_acc / max(1, n_steps),
    }


def train(args, model, train_dataset: DomainJsonDataset, val_dataset: DomainJsonDataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=DataCollatorWithPadding(tokenizer=args.tokenizer, max_length=args.max_seq_length),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=DataCollatorWithPadding(tokenizer=args.tokenizer, max_length=args.max_seq_length),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    total_steps = int(len(train_loader) * args.num_train_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Optional: class weights
    class_weights = None
    if args.use_class_weight:
        class_weights = compute_class_weights(train_dataset.items).to(args.device)
        print("[Info] Class-weight enabled. weights =", class_weights.detach().cpu().numpy())

    best_val_acc = -1.0
    global_step = 0

    for epoch in range(int(args.num_train_epochs)):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{int(args.num_train_epochs)}")
        for batch in pbar:
            batch = {k: v.to(args.device) for k, v in batch.items()}

            # forward
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask", None),
                labels=None,  # we compute CE ourselves to inject optional weights
            )
            logits = out.logits

            # loss
            if class_weights is None:
                loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
            else:
                loss = torch.nn.functional.cross_entropy(logits, batch["labels"], weight=class_weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            model.zero_grad(set_to_none=True)

            # metrics
            acc = accuracy_from_logits(logits.detach(), batch["labels"])
            epoch_loss += loss.item()
            epoch_acc += acc
            n_steps += 1
            global_step += 1

            if global_step % args.logging_steps == 0:
                pbar.set_postfix({"loss": epoch_loss / n_steps, "acc": epoch_acc / n_steps})

        # eval
        val_metrics = evaluate(args, model, val_loader)
        print(f"[Epoch {epoch+1}] val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}")

        # save best
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            os.makedirs(args.output_dir, exist_ok=True)

            model.save_pretrained(args.output_dir)
            args.tokenizer.save_pretrained(args.output_dir)

            with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
                json.dump(
                    {
                        "TARGET_TOPICS": TARGET_TOPICS,
                        "LABEL2ID": LABEL2ID,
                        "OTHERS_ID": OTHERS_ID,
                        "NUM_LABELS": NUM_LABELS,
                        "note": "others = any file topic not in TARGET_TOPICS (based on filename scienceqa_train_{topic}.json)",
                    },
                    f,
                    indent=2,
                )
            print(f"[Save] Best model saved to {args.output_dir} (best_val_acc={best_val_acc:.4f})")


# -----------------------------
# 9) Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    # data / model
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--train_dir", default="./new_data/train", type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)

    # train hyperparams
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    # infra
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--output_dir", default="./domain_cls_checkpoints", type=str)
    parser.add_argument("--logging_steps", default=50, type=int)

    # split / imbalance
    parser.add_argument("--val_ratio", default=0.1, type=float)
    parser.add_argument("--use_class_weight", action="store_true")

    # text options
    parser.add_argument("--include_hint", action="store_true")
    parser.add_argument("--include_lecture", action="store_true")
    parser.add_argument("--include_meta", action="store_true")

    args = parser.parse_args()

    # defaults: include_hint True for your data style
    # (argparse store_true defaults False, so we flip if user didn't pass anything)
    # If you want hint always-on, set default=True instead:
    #   parser.add_argument("--include_hint", action="store_true", default=True)
    # Here we keep it explicit: if you want hint, pass --include_hint.
    # Recommendation: pass --include_hint, keep lecture off.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    set_seed(args.seed)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    args.tokenizer = tokenizer

    # load
    items = load_domain_items(
        train_dir=args.train_dir,
        include_hint=args.include_hint,
        include_lecture=args.include_lecture,
        include_meta=args.include_meta,
    )

    train_items, val_items = stratified_split(items, val_ratio=args.val_ratio, seed=args.seed)

    print(f"[Data] Total={len(items)} Train={len(train_items)} Val={len(val_items)}")
    print("[Data] Train label stats:", label_stats(train_items))
    print("[Data] Val   label stats:", label_stats(val_items))
    print("[Labels] 0-9:", TARGET_TOPICS)
    print("[Labels] 10 : others")

    # datasets
    train_dataset = DomainJsonDataset(train_items, tokenizer, max_len=args.max_seq_length)
    val_dataset = DomainJsonDataset(val_items, tokenizer, max_len=args.max_seq_length)

    # model
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=NUM_LABELS,
    ).to(args.device)

    train(args, model, train_dataset, val_dataset)


if __name__ == "__main__":
    main()
