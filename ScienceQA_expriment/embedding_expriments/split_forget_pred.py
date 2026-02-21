import os, json, argparse
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

LETTERS = ["A", "B", "C", "D", "E"]  # E까지 확장


def build_prompt(ex):
    ti = ex.get("text_input", "")
    if isinstance(ti, str) and len(ti.strip()) > 0:
        base = ti.strip()
        if "The answer is" in base:
            base = base.split("The answer is")[0].rstrip()
        return base + "\nThe answer is"

    q = ex.get("instruction") or ex.get("question")
    choices = ex.get("choices")
    if q is None or choices is None:
        raise KeyError(f"Missing fields. keys={list(ex.keys())}")

    valid_letters = LETTERS[:len(choices)]
    s = f"Question: {q}\nChoices:\n"
    for i, c in enumerate(choices):
        s += f"({valid_letters[i]}) {c}\n"
    s += f"Answer with a single letter among {', '.join(valid_letters)}.\nThe answer is"
    return s


@torch.no_grad()
def predict_letter_logits(model, tokenizer, prompt, n_choices, device):
    valid_letters = LETTERS[:n_choices]

    tok_ids = {}
    for L in valid_letters:
        ids = tokenizer(f" {L}", add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f'" {L}" is not a single token: {ids}')
        tok_ids[L] = ids[0]

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    out = model(**inputs)
    next_logits = out.logits[0, -1]

    ids = torch.tensor([tok_ids[L] for L in valid_letters], device=next_logits.device)
    scores = next_logits[ids]
    pred_i = int(torch.argmax(scores).item())
    pred = valid_letters[pred_i]
    score_dict = {L: float(scores[i].item()) for i, L in enumerate(valid_letters)}
    return pred, score_dict, valid_letters


def main(args):
    dataset_path = f"../data/scienceqa/scienceqa_{args.subject}_all.hf"
    out_dir = f"../data/scienceqa/splits_{args.subject}_pred"
    model_name = args.model_name

    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_from_disk(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    for split_name in ds.keys():
        split_ds = ds[split_name]
        n = len(split_ds) if args.max_items_per_split is None else min(
            len(split_ds), args.max_items_per_split
        )

        cache_path = os.path.join(out_dir, f"{split_name}_pred_cache.jsonl")
        w_all = open(cache_path, "w")

        # split 내 최대 선택지 수 추정
        probe_n = min(n, 512)
        max_k = 0
        for i in range(probe_n):
            ch = split_ds[i].get("choices")
            if isinstance(ch, list):
                max_k = max(max_k, len(ch))
        max_k = max(2, min(max_k, len(LETTERS)))
        letters_in_split = LETTERS[:max_k]

        writers = {
            L: open(os.path.join(out_dir, f"{split_name}_pred_{L}.jsonl"), "w")
            for L in letters_in_split
        }

        count = 0
        for idx in range(n):
            ex = split_ds[idx]
            choices = ex.get("choices")
            if not isinstance(choices, list) or len(choices) < 2 or len(choices) > len(LETTERS):
                continue

            prompt = build_prompt(ex)
            pred, scores, valid_letters = predict_letter_logits(
                model, tokenizer, prompt, n_choices=len(choices), device=device
            )

            rec = dict(ex)
            rec["idx"] = idx
            rec["pred"] = pred
            rec["pred_scores"] = scores
            rec["valid_letters"] = valid_letters

            line = json.dumps(rec, ensure_ascii=False)
            w_all.write(line + "\n")
            writers[pred].write(line + "\n")
            count += 1

        w_all.close()
        for w in writers.values():
            w.close()

        print(f"[DONE] {args.subject} | {split_name}: processed={count}/{n} -> {cache_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="scienceqa subject name (e.g., chemistry, physics, biology)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
    )
    parser.add_argument(
        "--max_items_per_split",
        type=int,
        default=None,
        help="debug용 샘플 제한",
    )

    args = parser.parse_args()
    main(args)
