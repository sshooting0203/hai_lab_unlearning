import random
import torch
from sentence_transformers import SentenceTransformer, util
from typing import Any, Dict, List, Optional

_EVAL_EMBED_MODEL = None
_REFUSAL_CACHE = None


def get_embed_model():
    global _EVAL_EMBED_MODEL
    if _EVAL_EMBED_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _EVAL_EMBED_MODEL = SentenceTransformer("all-mpnet-base-v2").to(device)
    return _EVAL_EMBED_MODEL


def load_refusal_texts(
    refusal_path: str = "/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code/dataset/unlearning/refusal_response.txt",
    sample_k: Optional[int] = 5,
    seed: int = 42,
) -> List[str]:
    global _REFUSAL_CACHE

    if _REFUSAL_CACHE is None:
        with open(refusal_path, "r", encoding="utf-8") as f:
            _REFUSAL_CACHE = [line.strip() for line in f if line.strip()]

    refusal_texts = _REFUSAL_CACHE

    if sample_k is None or sample_k >= len(refusal_texts):
        return refusal_texts

    rng = random.Random(seed)
    return rng.sample(refusal_texts, sample_k)


import re

def _normalize_for_match(text: str) -> str:
    # Normalize text for string matching: lowercase and remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

def eval_choice_hit(
    gen_texts: List[str],
    gt_answers: List[str],
    perturbed_answers_list: Optional[List[List[str]]] = None,
    embed_model: Optional[Any] = None,
    refusal_path: str = "/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code/dataset/unlearning/refusal_response.txt",
    refusal_sample_k: int = 5,
    refusal_seed: int = 42,
    name: Optional[str] = None,
) -> Dict[str, float]:
    if not gen_texts:
        raise ValueError("gen_texts is empty!")

    if embed_model is None:
        embed_model = get_embed_model()

    refusal_texts = load_refusal_texts(
        refusal_path=refusal_path,
        sample_k=refusal_sample_k,
        seed=refusal_seed,
    )

    gen_embeddings = embed_model.encode(
        gen_texts,
        convert_to_tensor=True,
        show_progress_bar=False
    )

    gt_hit = 0
    perturbed_hit = 0
    refusal_hit = 0
    evaluated = 0
    eval_results = []

    for i, (gen_emb, gt) in enumerate(zip(gen_embeddings, gt_answers)):
        if not gt:
            continue
        if perturbed_answers_list is None or i >= len(perturbed_answers_list):
            continue

        perturbed = perturbed_answers_list[i]
        if perturbed is None:
            perturbed = []

        candidates = [gt] + perturbed + refusal_texts

        cand_embeddings = embed_model.encode(
            candidates,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        scores = util.cos_sim(gen_emb, cand_embeddings)[0]
        best_idx = torch.argmax(scores).item()

        # --- 포함되어 있으면 unlearn하지 않은 것으로 간주---
        gt_end = 1
        perturbed_end = 1 + len(perturbed)

        if name and ("real_authors" in name.lower() or "world_facts" in name.lower()):
            # GT가 1순위가 아니지만, 거절 답변군(Refusal)에 해당하지 않는 경우에만 포함 여부 체크
            if best_idx != 0 and best_idx < perturbed_end:
                norm_gen = _normalize_for_match(gen_texts[i])
                norm_gt = _normalize_for_match(gt)
                if norm_gt in norm_gen and len(norm_gt) > 0:
                    best_idx = 0
        # -----------------------------------------------------------------------

        evaluated += 1
        if best_idx == 0:
            gt_hit += 1
            eval_results.append(True)
        elif 1 <= best_idx < perturbed_end:
            perturbed_hit += 1
            eval_results.append(False)
        else:
            refusal_hit += 1
            eval_results.append(False)

    denom = evaluated if evaluated > 0 else 1

    return {
        "choice_accuracy": gt_hit / denom,   
        "choice_correct": float(gt_hit),   
        "perturbed_rate": perturbed_hit / denom,
        "perturbed_correct": float(perturbed_hit),
        "refusal_rate": refusal_hit / denom,
        "refusal_correct": float(refusal_hit),
        "eval_results": eval_results,
    }