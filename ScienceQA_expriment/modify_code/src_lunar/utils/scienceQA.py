
import re
from typing import Dict, List, Optional

def _extract_pred_choice(gen: str) -> Optional[str]:
    if not gen:
        return None
    s = gen.upper().strip()

    # 1) "THE ANSWER IS B" / "ANSWER: THE ANSWER IS B."
    m = re.search(r"THE\s+ANSWER\s+IS\s+([A-E])\b", s)
    if m:
        return m.group(1)

    # 2) "ANSWER: B"
    m = re.search(r"ANSWER\s*:\s*([A-E])\b", s)
    if m:
        return m.group(1)
    return None

def _contains_choice(gen: str, choice: str) -> bool:
    if not gen or not choice:
        return False
    s = gen.upper()
    return bool(
        re.search(
            rf"(?:THE\s+ANSWER\s+IS\s+{choice}\b|ANSWER\s*:\s*{choice}\b|\({choice}\)|\b{choice}\b)",
            s,
        )
    )

def eval_choice_hit(
    gen_texts: List[str],
    gt_answers: List[str],
    **kwargs,
) -> Dict[str, float]:
    evaluated = 0
    contains_ok = 0
    exact_ok = 0

    eval_results = []
    for gen, gt in zip(gen_texts, gt_answers):
        evaluated += 1
        if not gt:
            raise ValueError("GT answer is empty")

        if _contains_choice(gen, gt):
            contains_ok += 1

        pred_choice = _extract_pred_choice(gen)
        if pred_choice is not None and pred_choice == gt:
            exact_ok += 1
            eval_results.append(True)
        else:
            eval_results.append(False)

    denom = evaluated if evaluated > 0 else 1
    return {
        "choice_accuracy": exact_ok / denom,
        "choice_correct": float(exact_ok),
        "label_contains_accuracy": contains_ok / denom,
        "label_contains_correct": float(contains_ok),
        "eval_results": eval_results,
    }


