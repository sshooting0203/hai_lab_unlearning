import re
from typing import Dict, List, Optional

def _extract_pred_intent(gen: str) -> Optional[str]:
    if not gen:
        return None
    g = gen.strip()

    # "Intent: xxx" / "Answer: xxx" / "Label: xxx" 라인 우선
    m = re.search(r"(?im)^\s*(intent|answer|label)\s*:\s*(.+?)\s*$", g)
    if m:
        cand = m.group(2).strip()
        return cand if cand else None

    # fallback: 마지막 줄에서 토큰(underscore 포함) 중 마지막을 라벨로 간주
    last = g.splitlines()[-1].strip()
    toks = re.findall(r"[A-Za-z0-9_]+", last)
    return toks[-1] if toks else None

def _contains_intent(gen: str, gt_intent: str) -> bool:
    if not gen:
        return False
    # intent 라벨은 substring 오탐 위험이 있어서 token match로만, underscore 포함 단어를 토큰처럼 취급
    toks = set(re.findall(r"[A-Za-z0-9_]+", gen))
    return gt_intent in toks

def eval_choice_hit(
    gen_texts: List[str],
    gt_answers: List[str],
    **kwargs,
) -> Dict[str, float]:
    # print("gen_texts:", gen_texts)
    # print("gt_answers:", gt_answers)
    evaluated = 0
    contains_ok = 0
    exact_ok = 0
    eval_results = []

    for gen, gt in zip(gen_texts, gt_answers):
        evaluated += 1
        if not gt:
            continue

        if _contains_intent(gen, gt):
            contains_ok += 1

        pred_intent = _extract_pred_intent(gen)
        if pred_intent is not None and pred_intent == gt:
            exact_ok += 1
            eval_results.append(True)
        else:
            eval_results.append(False)

    denom = evaluated if evaluated > 0 else 1
    return {
        "choice_accuracy": exact_ok / denom,      # 키 이름 유지
        "choice_correct": float(exact_ok),
        "label_contains_accuracy": contains_ok / denom,
        "label_contains_correct": float(contains_ok),
        "eval_results": eval_results,
    }