import os
import re
import json
import random
import torch
import tqdm
from typing import Dict, List, Any, Optional, Tuple

from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util


# -----------------------------
# Optional system prompt
# -----------------------------
def build_system_prompt(authors_json_path: Optional[str] = None) -> str:
    if authors_json_path is None:
        return ""
    with open(authors_json_path, "r", encoding="utf-8") as f:
        authors = json.load(f)
    names = authors.get("forget_author", [])
    if not names:
        return ""
    return (
        "<<SYS>>Please refrain from responding to the following authors' information: "
        + ", ".join(names)
        + "\n<</SYS>>"
    )


# -----------------------------
# JSON dataset loader
# -----------------------------
def load_json_dataset(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("data", "examples", "items"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            data = [data]

    if not isinstance(data, list):
        raise ValueError(f"Unsupported JSON format: {json_path}")

    normalized = []
    for ex in data:
        if not isinstance(ex, dict):
            continue

        normalized.append({
            "question": str(ex.get("question", "") or ""),
            "answer": str(ex.get("answer", "") or ""),
            "paraphrased_question": str(ex.get("paraphrased_question", "") or ""),
            "paraphrased_answer": str(ex.get("paraphrased_answer", "") or ""),
            "perturbed_answer": ex.get("perturbed_answer", []) or [],
            "author_idx": ex.get("author_idx", None),
            "forget_author": str(ex.get("forget_author", "") or ""),
        })

    return normalized

# -----------------------------
# Core helpers
# -----------------------------
def get_batch_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_fn(shift_logits.transpose(-1, -2), shift_labels).sum(dim=-1)
    return loss


def build_prompt_text(
    tokenizer,
    prompt: str,
    if_system: bool = False,
    sys_prompt: str = "",
) -> str:
    messages = []

    if if_system and sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def build_answer_prefix() -> str:
    return ""

def get_answer_probability(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    device: torch.device,
    if_llama: bool = False,
    if_system: bool = False,
    sys_prompt: str = "",
) -> float:
    question_text = build_prompt_text(
    tokenizer=tokenizer,
        prompt=prompt,
        if_system=if_system,
        sys_prompt=sys_prompt,
    )
    answer_prefix = build_answer_prefix()
    full_text = question_text + answer_prefix + answer

    tokenized_full = tokenizer(
        full_text,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    tokenized_question = tokenizer(
        question_text,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    input_ids = tokenized_full["input_ids"].to(device)
    attention_mask = tokenized_full["attention_mask"].to(device)

    labels = input_ids.clone()
    question_len = tokenized_question["input_ids"].shape[1]
    labels[:, :question_len] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    loss = get_batch_loss(outputs.logits, labels)
    answer_token_count = (labels != -100).sum(dim=-1).clamp(min=1)
    loss_per_token = loss / answer_token_count
    prob = torch.exp(-loss_per_token)

    return prob.item()


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    if_llama: bool = False,
    if_system: bool = False,
    sys_prompt: str = "",
    max_new_tokens: int = 128,
) -> str:
    question_text = build_prompt_text(
        tokenizer=tokenizer,
        prompt=prompt,
        if_system=if_system,
        sys_prompt=sys_prompt,
    )

    tokenized = tokenizer(question_text, return_tensors="pt", add_special_tokens=True)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text.replace("[pad]", "").replace("<pad>", "").strip()


def semantic_match_correct(
    generated_answer: str,
    true_answer: str,
    false_answers: List[str],
    sentence_model: SentenceTransformer,
) -> bool:
    gen_emb = sentence_model.encode(generated_answer, convert_to_tensor=True)
    true_emb = sentence_model.encode(true_answer, convert_to_tensor=True)

    scores = [util.pytorch_cos_sim(gen_emb, true_emb).item()]
    for fa in false_answers:
        fa_emb = sentence_model.encode(fa, convert_to_tensor=True)
        scores.append(util.pytorch_cos_sim(gen_emb, fa_emb).item())

    return max(scores) == scores[0]


def substring_match_correct(generated_answer: str, truth_answer: str) -> bool:
    pattern = re.compile(re.escape(truth_answer), re.IGNORECASE)
    return pattern.search(generated_answer) is not None


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_semantic_subset(
    model,
    tokenizer,
    dataset_split: List[Dict[str, Any]],
    device: torch.device,
    sentence_model: SentenceTransformer,
    if_llama: bool = False,
    if_system: bool = False,
    sys_prompt: str = "",
    desc: str = "evaluating semantic subset",
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    total = 0
    correct = 0
    truth_ratios = []
    generated_paraphrased_answers = []
    generated_original_answers = []

    mean_truth_ratio = 0.0
    mean_truth_prob = 0.0
    mean_rougeL_score = 0.0

    for example in tqdm.tqdm(dataset_split, desc=desc):
        total += 1

        paraphrased_question = example["paraphrased_question"]
        paraphrased_answer = example["paraphrased_answer"]
        perturbed_answers = example["perturbed_answer"]

        # 일부 JSON에 paraphrased 필드가 비어 있을 수 있으므로 fallback
        if not paraphrased_question:
            paraphrased_question = example["question"]
        if not paraphrased_answer:
            paraphrased_answer = example["answer"]

        paraphrased_answer_prob = get_answer_probability(
            model=model,
            tokenizer=tokenizer,
            prompt=paraphrased_question,
            answer=paraphrased_answer,
            device=device,
            if_llama=if_llama,
            if_system=if_system,
            sys_prompt=sys_prompt,
        )

        false_probs = []
        for false_answer in perturbed_answers:
            p = get_answer_probability(
                model=model,
                tokenizer=tokenizer,
                prompt=paraphrased_question,
                answer=false_answer,
                device=device,
                if_llama=if_llama,
                if_system=if_system,
                sys_prompt=sys_prompt,
            )
            false_probs.append(p)

        if len(false_probs) > 0:
            truth_ratio = (sum(false_probs) / len(false_probs)) / (paraphrased_answer_prob + 1e-12)
        else:
            truth_ratio = 0.0
        truth_ratios.append(truth_ratio)
        mean_truth_ratio += truth_ratio

        generated_para_answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=paraphrased_question,
            device=device,
            if_llama=if_llama,
            if_system=if_system,
            sys_prompt=sys_prompt,
            max_new_tokens=max_new_tokens,
        )
        generated_paraphrased_answers.append(generated_para_answer)

        if len(perturbed_answers) > 0:
            if semantic_match_correct(
                generated_answer=generated_para_answer,
                true_answer=paraphrased_answer,
                false_answers=perturbed_answers,
                sentence_model=sentence_model,
            ):
                correct += 1

        question = example["question"]
        answer = example["answer"]

        truth_answer_prob = get_answer_probability(
            model=model,
            tokenizer=tokenizer,
            prompt=question,
            answer=answer,
            device=device,
            if_llama=if_llama,
            if_system=if_system,
            sys_prompt=sys_prompt,
        )
        mean_truth_prob += truth_answer_prob

        generated_answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=question,
            device=device,
            if_llama=if_llama,
            if_system=if_system,
            sys_prompt=sys_prompt,
            max_new_tokens=max_new_tokens,
        )
        generated_original_answers.append(generated_answer)

        score = rouge.score(answer, generated_answer)
        mean_rougeL_score += score["rougeL"].recall

    n = max(len(dataset_split), 1)
    return {
        "truth_ratios": truth_ratios,
        "truth_ratio": mean_truth_ratio / n,
        "truth_prob": mean_truth_prob / n,
        "rougeL_score": mean_rougeL_score / n,
        "acc": correct / max(total, 1),
        "generated_answers": generated_paraphrased_answers,
        "original_answers": generated_original_answers,
    }


def evaluate_plain_subset(
    model,
    tokenizer,
    dataset_split: List[Dict[str, Any]],
    device: torch.device,
    if_llama: bool = False,
    if_system: bool = False,
    sys_prompt: str = "",
    desc: str = "evaluating plain subset",
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    total = 0
    correct = 0
    truth_ratios = []
    generated_answers = []

    mean_truth_ratio = 0.0
    mean_truth_prob = 0.0
    mean_rougeL_score = 0.0

    for example in tqdm.tqdm(dataset_split, desc=desc):
        total += 1

        question = example["question"]
        truth_answer = example["answer"]
        perturbed_answers = example["perturbed_answer"]

        truth_answer_prob = get_answer_probability(
            model=model,
            tokenizer=tokenizer,
            prompt=question,
            answer=truth_answer,
            device=device,
            if_llama=if_llama,
            if_system=if_system,
            sys_prompt=sys_prompt,
        )
        mean_truth_prob += truth_answer_prob

        generated_answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=question,
            device=device,
            if_llama=if_llama,
            if_system=if_system,
            sys_prompt=sys_prompt,
            max_new_tokens=max_new_tokens,
        )
        generated_answers.append(generated_answer)

        false_probs = []
        for false_answer in perturbed_answers:
            p = get_answer_probability(
                model=model,
                tokenizer=tokenizer,
                prompt=question,
                answer=false_answer,
                device=device,
                if_llama=if_llama,
                if_system=if_system,
                sys_prompt=sys_prompt,
            )
            false_probs.append(p)

        if len(false_probs) > 0:
            truth_ratio = (sum(false_probs) / len(false_probs)) / (truth_answer_prob + 1e-12)
        else:
            truth_ratio = 0.0

        truth_ratios.append(truth_ratio)
        mean_truth_ratio += truth_ratio

        if substring_match_correct(generated_answer, truth_answer):
            correct += 1

        score = rouge.score(truth_answer, generated_answer)
        mean_rougeL_score += score["rougeL"].recall

    n = max(len(dataset_split), 1)
    return {
        "truth_ratios": truth_ratios,
        "truth_ratio": mean_truth_ratio / n,
        "truth_prob": mean_truth_prob / n,
        "rougeL_score": mean_rougeL_score / n,
        "acc": correct / max(total, 1),
        "generated_answers": generated_answers,
    }


# -----------------------------
# Main runner
# -----------------------------
def run_tofu_evaluation_from_json(
    model,
    model_name_or_tokenizer_path: str,
    forget_json_path: str,
    retain_json_path: Optional[str] = None,
    real_author_json_path: Optional[str] = None,
    world_fact_json_path: Optional[str] = None,
    output_dir: str = ".",
    if_llama: bool = False,
    if_system: bool = False,
    authors_json_path: Optional[str] = "files/data/authors.json",
    forget_ratio: float = 0.8,
    dataset_seed: int = 1000,
    subset_name: Optional[str] = None,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_tokenizer_path, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    sys_prompt = build_system_prompt(authors_json_path) if if_system else ""
    sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # forget dataset
    forget_dataset = load_json_dataset(forget_json_path)
    # retain / real_author / world_fact
    retain_dataset = load_json_dataset(retain_json_path) if retain_json_path else []
    real_author_dataset = load_json_dataset(real_author_json_path) if real_author_json_path else []
    world_fact_dataset = load_json_dataset(world_fact_json_path) if world_fact_json_path else []

    result = {
        "meta": {
            "forget_json_path": forget_json_path,
            "retain_json_path": retain_json_path,
            "real_author_json_path": real_author_json_path,
            "world_fact_json_path": world_fact_json_path,
            "forget_ratio": forget_ratio,
            "dataset_seed": dataset_seed,
            "forget_used_size": len(forget_used),
            "forget_unused_size": len(forget_unused),
            "forget_index_list": forget_index_list,
            "retain_index_list": retain_index_list,
        }
    }

    result["forget_used"] = evaluate_semantic_subset(
        model=model,
        tokenizer=tokenizer,
        dataset_split=forget_used,
        device=device,
        sentence_model=sentence_model,
        if_llama=if_llama,
        if_system=if_system,
        sys_prompt=sys_prompt,
        desc="evaluating forget_used",
        max_new_tokens=max_new_tokens,
    )

    result["forget_unused"] = evaluate_semantic_subset(
        model=model,
        tokenizer=tokenizer,
        dataset_split=forget_unused,
        device=device,
        sentence_model=sentence_model,
        if_llama=if_llama,
        if_system=if_system,
        sys_prompt=sys_prompt,
        desc="evaluating forget_unused",
        max_new_tokens=max_new_tokens,
    )

    if len(retain_dataset) > 0:
        result["retain"] = evaluate_semantic_subset(
            model=model,
            tokenizer=tokenizer,
            dataset_split=retain_dataset,
            device=device,
            sentence_model=sentence_model,
            if_llama=if_llama,
            if_system=if_system,
            sys_prompt=sys_prompt,
            desc="evaluating retain",
            max_new_tokens=max_new_tokens,
        )

    if len(real_author_dataset) > 0:
        result["real_author"] = evaluate_plain_subset(
            model=model,
            tokenizer=tokenizer,
            dataset_split=real_author_dataset,
            device=device,
            if_llama=if_llama,
            if_system=if_system,
            sys_prompt=sys_prompt,
            desc="evaluating real_author",
            max_new_tokens=max_new_tokens,
        )

    if len(world_fact_dataset) > 0:
        result["world_fact"] = evaluate_plain_subset(
            model=model,
            tokenizer=tokenizer,
            dataset_split=world_fact_dataset,
            device=device,
            if_llama=if_llama,
            if_system=if_system,
            sys_prompt=sys_prompt,
            desc="evaluating world_fact",
            max_new_tokens=max_new_tokens,
        )

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "tofu_eval_from_json.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"[Saved] {save_path}")
    return result


# result = run_tofu_evaluation_from_json(
#     model=model,
#     model_name_or_tokenizer_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
#     forget_json_path="/path/to/forget01_perturbed.json",
#     retain_json_path="/path/to/retain_perturbed.json",
#     real_author_json_path="/path/to/real_authors_perturbed.json",
#     world_fact_json_path="/path/to/world_facts_perturbed.json",
#     output_dir="./eval_results",
#     if_llama=True,
#     if_system=False,
#     forget_ratio=0.8,
#     dataset_seed=1000,
#     subset_name="forget01",
# )