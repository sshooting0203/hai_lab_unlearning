#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re
from collections import Counter

NUM_RE = re.compile(r"\d")

def has_digit(s: str) -> bool:
    return bool(s) and bool(NUM_RE.search(s))

def only_digits_or_punct(s: str) -> bool:
    if s is None:
        return False
    s = str(s).strip()
    if not s:
        return False
    # 숫자/구두점/공백만 남는지 (예: "1212", "10, 10", "1.", "(2)")
    return re.fullmatch(r"[0-9\s,.\-\(\):;]+", s) is not None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="train_pred_cache.jsonl etc.")
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--print_examples", type=int, default=20)
    args = ap.parse_args()

    total = 0
    cnt_text_input_digit = 0
    cnt_question_digit = 0
    cnt_choices_digit = 0
    cnt_answer_digit = 0
    cnt_answer_only_digits = 0

    # 무엇에 숫자가 많이 나타나는지 카운트
    digit_context_counter = Counter()

    examples = []

    with open(args.jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_items and i >= args.max_items:
                break
            r = json.loads(line)
            total += 1

            text_input = r.get("text_input", "")
            question = r.get("instruction") or r.get("question") or ""
            choices = r.get("choices", [])
            answer = r.get("answer", None)

            # text_input
            if isinstance(text_input, str) and has_digit(text_input):
                cnt_text_input_digit += 1
                digit_context_counter["text_input"] += 1

            # question/instruction
            if isinstance(question, str) and has_digit(question):
                cnt_question_digit += 1
                digit_context_counter["question_or_instruction"] += 1

            # choices
            if isinstance(choices, list):
                if any(isinstance(c, str) and has_digit(c) for c in choices):
                    cnt_choices_digit += 1
                    digit_context_counter["choices"] += 1

            # answer
            if answer is not None:
                ans_s = str(answer)
                if has_digit(ans_s):
                    cnt_answer_digit += 1
                    digit_context_counter["answer"] += 1
                if only_digits_or_punct(ans_s):
                    cnt_answer_only_digits += 1
                    digit_context_counter["answer_only_digits_or_punct"] += 1

            # 예시 저장 (숫자가 들어간 케이스 위주)
            if len(examples) < args.print_examples:
                flagged = (
                    (isinstance(text_input, str) and has_digit(text_input))
                    or (isinstance(question, str) and has_digit(question))
                    or (isinstance(choices, list) and any(isinstance(c, str) and has_digit(c) for c in choices))
                    or (answer is not None and (has_digit(str(answer)) or only_digits_or_punct(str(answer))))
                )
                if flagged:
                    examples.append({
                        "idx": r.get("idx", i),
                        "answer": answer,
                        "choices": choices if isinstance(choices, list) else None,
                        "text_input_tail": (text_input[-300:] if isinstance(text_input, str) else ""),
                        "question": question[:200] if isinstance(question, str) else "",
                    })

    def pct(x): 
        return 100.0 * x / max(total, 1)

    print(f"Total records: {total}")
    print(f"text_input contains digits: {cnt_text_input_digit} ({pct(cnt_text_input_digit):.2f}%)")
    print(f"question/instruction contains digits: {cnt_question_digit} ({pct(cnt_question_digit):.2f}%)")
    print(f"any choice contains digits: {cnt_choices_digit} ({pct(cnt_choices_digit):.2f}%)")
    print(f"answer contains digits: {cnt_answer_digit} ({pct(cnt_answer_digit):.2f}%)")
    print(f"answer is only digits/punct: {cnt_answer_only_digits} ({pct(cnt_answer_only_digits):.2f}%)")

    print("\nDigit context counts:", dict(digit_context_counter))

    if examples:
        print("\n--- Examples (flagged) ---")
        for ex in examples:
            print("=" * 80)
            print(f"idx={ex['idx']} answer={ex['answer']}")
            if ex["choices"] is not None:
                print("choices:", ex["choices"])
            if ex["question"]:
                print("question:", ex["question"])
            print("text_input_tail:")
            print(ex["text_input_tail"])

if __name__ == "__main__":
    main()
