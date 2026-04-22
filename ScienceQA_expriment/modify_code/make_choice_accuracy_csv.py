#!/usr/bin/env python3
import os
import json
import csv
import argparse
from pathlib import Path

TARGET_METRIC = "choice_accuracy"
ROW_ORDER = ["forget", "retain", "mrpc", "rte"]

# JSON 안에서 사용할 alias들
TARGET_SPLITS = {
    "forget": ["forget"],
    "retain": ["retain", "retained"],
    "mrpc": ["mrpc"],
    "rte": ["rte"],
}


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


def extract_choice_accuracy_from_json(data):
    """
    JSON 내부에서 forget/retain/mrpc/rte의 results.choice_accuracy를 찾는다.
    반환 예시:
    {
        "forget": 0.0,
        "retain": 0.97,
        "mrpc": 0.84,
        "rte": 0.76,
    }
    """
    found = {}

    if not isinstance(data, dict):
        return found

    # 1) 최상위에서 바로 찾기
    for canonical_name, aliases in TARGET_SPLITS.items():
        for alias in aliases:
            if alias in data and isinstance(data[alias], dict):
                results = data[alias].get("results", {})
                if isinstance(results, dict) and TARGET_METRIC in results:
                    found[canonical_name] = results[TARGET_METRIC]
                    break

    # 2) 재귀 탐색
    def recursive_search(obj):
        if not isinstance(obj, dict):
            return

        for canonical_name, aliases in TARGET_SPLITS.items():
            if canonical_name in found:
                continue
            for alias in aliases:
                if alias in obj and isinstance(obj[alias], dict):
                    results = obj[alias].get("results", {})
                    if isinstance(results, dict) and TARGET_METRIC in results:
                        found[canonical_name] = results[TARGET_METRIC]
                        break

        for v in obj.values():
            if isinstance(v, dict):
                recursive_search(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        recursive_search(item)

    recursive_search(data)
    return found


def guess_task_dirs(base_dir: Path):
    """
    base_dir 바로 아래의 하위 디렉토리들을 task 디렉토리로 간주한다.
    """
    return sorted([p for p in base_dir.iterdir() if p.is_dir()])


def collect_metrics_for_task(task_dir: Path):
    """
    task 디렉토리 아래의 모든 json 파일을 뒤져서
    forget / retain / mrpc / rte 의 choice_accuracy를 수집한다.
    """
    metrics = {k: None for k in ROW_ORDER}

    json_files = list(task_dir.rglob("*.json"))
    if not json_files:
        print(f"[WARN] No JSON files found in {task_dir}")
        return metrics

    for jf in json_files:
        data = load_json(jf)
        if data is None:
            continue

        found = extract_choice_accuracy_from_json(data)
        for k, v in found.items():
            if metrics[k] is None:
                metrics[k] = v

    return metrics


def build_table(base_dir: Path):
    """
    최종 형태:
    row = forget/retain/mrpc/rte
    col = task 이름
    value = choice_accuracy
    """
    task_dirs = guess_task_dirs(base_dir)
    if not task_dirs:
        raise RuntimeError(f"No task directories found under: {base_dir}")

    table = {row_name: {} for row_name in ROW_ORDER}

    for task_dir in task_dirs:
        task_name = task_dir.name
        metrics = collect_metrics_for_task(task_dir)

        for row_name in ROW_ORDER:
            table[row_name][task_name] = metrics[row_name]

        print(f"[INFO] Collected: {task_name}")

    return table, [td.name for td in task_dirs]


def save_csv(table, task_names, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["split"] + task_names

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row_name in ROW_ORDER:
            row = {"split": row_name}
            row.update(table[row_name])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Collect choice_accuracy for forget/retain/mrpc/rte across task directories and save as CSV."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="예: /dataset/.../clincqa_moe_second_intent_selectedCE"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="choice_accuracy_by_task.csv",
        help="저장할 csv 경로"
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    table, task_names = build_table(base_dir)
    output_csv = Path(args.output_csv)
    save_csv(table, task_names, output_csv)

    print(f"[DONE] Saved CSV to: {output_csv}")
    print(f"[DONE] Tasks: {len(task_names)}")
    print(f"[DONE] Rows: {len(ROW_ORDER)} ({', '.join(ROW_ORDER)})")


if __name__ == "__main__":
    main()
