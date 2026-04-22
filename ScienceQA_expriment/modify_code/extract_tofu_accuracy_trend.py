import os
import json
import pandas as pd
import glob
import re
from openpyxl import load_workbook

DECIMAL_PLACES = 6

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s

def apply_excel_number_format(excel_path, decimal_places=DECIMAL_PLACES):
    wb = load_workbook(excel_path)
    fmt = "0." + ("0" * decimal_places)

    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = fmt

    wb.save(excel_path)

def extract_tofu_accuracy(base_dir):
    seeds = [0, 1, 2]
    tasks = [f"t{i}" for i in range(30)]  # TOFU: t0 ~ t29
    all_data = []

    for seed in seeds:
        seed_dir_name = f"tofu_moe_fixed_layer16_{seed}"
        seed_path = os.path.join(base_dir, seed_dir_name)

        if not os.path.exists(seed_path):
            print(f"Warning: Seed directory not found: {seed_path}")
            continue

        for i, task in enumerate(tasks):
            domain_id = f"AG{i+1:02d}"
            task_folder_pattern = f"lunar_llama3.1-8b-instruct_eval_base_tofu_{task}_{domain_id}"
            task_folder_path = os.path.join(seed_path, task_folder_pattern)

            if not os.path.exists(task_folder_path):
                folders = glob.glob(os.path.join(seed_path, f"*_{task}_{domain_id}"))
                if folders:
                    task_folder_path = folders[0]
                else:
                    print(f"Warning: Task folder not found for {task} (seed {seed}): {task_folder_path}")
                    continue

            json_file = os.path.join(task_folder_path, f"forget_tofu_{task}_{domain_id}.json")
            if not os.path.exists(json_file):
                print(f"Warning: JSON file not found: {json_file}")
                continue

            with open(json_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
                    continue

            if "forget" not in data or "results" not in data["forget"]:
                continue

            generated_text = data["forget"]["results"].get("generated_text", [])

            domain_counts = {}
            domain_correct = {}

            for entry in generated_text:
                if len(entry) < 4:
                    continue

                # entry: [Question, Gen, GT, Domain, Metadata]
                gen_text = normalize_text(entry[1])
                gt_label = normalize_text(entry[2])
                domain = entry[3]

                # exact match after normalization
                is_correct = (gen_text == gt_label)

                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                domain_correct[domain] = domain_correct.get(domain, 0) + (1 if is_correct else 0)

            for domain in domain_counts:
                acc = domain_correct[domain] / domain_counts[domain]
                all_data.append({
                    "Seed": seed,
                    "TaskStep": task,
                    "Domain": domain,
                    "Accuracy": acc
                })

    if not all_data:
        print("No data extracted!")
        return

    df = pd.DataFrame(all_data)

    # seed 평균 전 개별 seed 결과
    df["Accuracy"] = df["Accuracy"].round(DECIMAL_PLACES)

    # seed 평균
    summary_df = df.groupby(["TaskStep", "Domain"])["Accuracy"].mean().reset_index()
    summary_df["Accuracy"] = summary_df["Accuracy"].round(DECIMAL_PLACES)

    def task_sort_key(task_str):
        return int(task_str[1:])

    pivot_df = summary_df.pivot(index="Domain", columns="TaskStep", values="Accuracy")

    sorted_cols = sorted(pivot_df.columns, key=task_sort_key)
    pivot_df = pivot_df[sorted_cols]

    pivot_df = pivot_df.sort_index()
    pivot_df = pivot_df.round(DECIMAL_PLACES)

    pd.set_option("display.float_format", lambda x: f"{x:.{DECIMAL_PLACES}f}")

    print("\nTOFU Accuracy Trend:")
    print(pivot_df)

    output_file = "tofu_accuracy_trend.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        pivot_df.to_excel(writer, sheet_name="Accuracy Trend")
        summary_df.to_excel(writer, sheet_name="Detailed Trend", index=False)
        df.to_excel(writer, sheet_name="Per Seed Accuracy", index=False)

    # 엑셀 셀 표시 형식까지 소수점 9자리로 고정
    apply_excel_number_format(output_file, decimal_places=DECIMAL_PLACES)

    print(f"Results saved to {output_file}")
    return pivot_df

if __name__ == "__main__":
    base_results_dir = "/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code/run_results/eval_basemoe_error/llama3.1-8b-instruct/tofu_moe_fixed"
    extract_tofu_accuracy(base_results_dir)