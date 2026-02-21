import os, json
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

LETTERS = ["A", "B", "C", "D"]  # biology_all에는 5지가 없으므로 D까지만

def build_prompt(ex):
    """
    text_input이 비어있지 않으면 그대로 사용(권장).
    아니면 instruction/question + choices로 포맷.
    prompt는 반드시 'The answer is'로 끝나게.
    """
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

    # 반드시 " {L}" (앞 공백 포함) 토큰 사용
    tok_ids = {}
    for L in valid_letters:
        ids = tokenizer(f" {L}", add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f'" {L}" is not a single token: {ids}')
        tok_ids[L] = ids[0]

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    out = model(**inputs)
    next_logits = out.logits[0, -1]  # 다음 토큰

    ids = torch.tensor([tok_ids[L] for L in valid_letters], device=next_logits.device)
    scores = next_logits[ids]
    pred_i = int(torch.argmax(scores).item())
    pred = valid_letters[pred_i]
    score_dict = {L: float(scores[i].item()) for i, L in enumerate(valid_letters)}
    return pred, score_dict, valid_letters

def main(
    dataset_path="data/scienceqa/scienceqa_biology_all.hf",
    model_name="meta-llama/Llama-3.1-8B",
    out_dir="data/scienceqa/splits_biology_pred",
    max_items_per_split=None,   # 디버그 시 숫자 넣기 (예: 200)
):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_from_disk(dataset_path)  # DatasetDict(train/validation/test)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    for split_name in ds.keys():
        split_ds = ds[split_name]
        n = len(split_ds) if max_items_per_split is None else min(len(split_ds), max_items_per_split)

        cache_path = os.path.join(out_dir, f"{split_name}_pred_cache.jsonl")
        writers = {L: open(os.path.join(out_dir, f"{split_name}_pred_{L}.jsonl"), "w") for L in LETTERS}
        w_all = open(cache_path, "w")

        count = 0
        for idx in range(n):
            ex = split_ds[idx]
            choices = ex.get("choices")
            if not isinstance(choices, list) or len(choices) < 2 or len(choices) > 4:
                continue  # 방어적

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

        print(f"[DONE] {split_name}: processed={count}/{n} -> {cache_path}")

if __name__ == "__main__":
    main()
