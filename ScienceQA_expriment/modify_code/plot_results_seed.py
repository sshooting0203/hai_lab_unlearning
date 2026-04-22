import os
import json
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="eval 결과 json 디렉토리 루트",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="plot, excel 저장 디렉토리",
    )
    parser.add_argument(
        "--excel_filename",
        type=str,
        default="task_dataset_metrics.xlsx",
        help="저장할 excel 파일명",
    )

    parser.add_argument(
        "--keys_to_extract",
        nargs="+",
        required=True,
        help="json 내부에서 뽑을 상위 key들. 예: forget retained mrpc rte",
    )

    parser.add_argument(
        "--row_names",
        nargs="+",
        default=["forget", "retain", "utility1", "utility2"],
        help="keys_to_extract에 대응하는 표시 이름. 예: forget retain utility1 utility2",
    )

    parser.add_argument(
        "--metric_columns",
        nargs="+",
        default=[
            "choice_accuracy",
            "choice_correct",
            "gt_loss",
            "gt_loss_per_token",
            "hit_rate",
            "mrr",
            "label_contains_accuracy",
            "label_contains_correct",
            "perplexity",
            "exact_match",
            "token_f1",
            "rouge_l_recall",
            "best_exact_match",
            "best_token_f1",
            "best_rouge_l_recall",
            "avg_prob",
            "best_prob",
            "truth_ratio",
            "embedding_similarity",
            "best_embedding_similarity",
        ],
        help="엑셀 컬럼으로 저장할 metric들",
    )
    parser.add_argument(
        "--excluded_metrics",
        nargs="*",
        default=["generated_text", "probs"],
        help="metric_columns 중 제외할 metric들",
    )
    parser.add_argument(
        "--plot_metrics",
        nargs="+",
        default=[
            "choice_accuracy",
            "es_score",
            "hit_rate",
            "mrr",
            "perplexity",
            "gt_loss",
            "gt_loss_per_token",
            "label_contains_accuracy",
            "exact_match",
            "token_f1",
            "rouge_l_recall",
            "best_exact_match",
            "best_token_f1",
            "best_rouge_l_recall",
            "avg_prob",
            "best_prob",
            "truth_ratio",
            "embedding_similarity",
            "best_embedding_similarity",
        ],
        help="plot 생성할 metric들",
    )

    parser.add_argument(
        "--task_regex",
        type=str,
        default=r"_t(\d+)(?:_|$)",
        help="task 번호 추출용 regex (예: _t22_)",
    )
    parser.add_argument(
        "--layer_regex",
        type=str,
        default=r"(?:layer|L)(\d+)(?:_|$)",
        help="layer 번호 추출용 regex (예: layer16, L16)",
    )
    parser.add_argument(
        "--seed_regex",
        type=str,
        default=r"(?:compare_|seed_|_|layer\d+_)(\d+)(?:_|$)",
        help="seed 번호 추출용 regex (예: compare_0, seed_1, layer16_0)",
    )

    parser.add_argument(
        "--json_pick_mode",
        type=str,
        default="all",
        choices=["all", "first"],
        help="task 디렉토리 내 json 처리 방식",
    )

    parser.add_argument(
        "--min_task_num",
        type=int,
        default=None,
        help="이 번호 이상 task만 포함",
    )
    parser.add_argument(
        "--max_task_num",
        type=int,
        default=None,
        help="이 번호 이하 task만 포함",
    )

    parser.add_argument(
        "--min_layer_num",
        type=int,
        default=None,
        help="이 번호 이상 layer만 포함",
    )
    parser.add_argument(
        "--max_layer_num",
        type=int,
        default=None,
        help="이 번호 이하 layer만 포함",
    )

    parser.add_argument(
        "--x_axis_mode",
        type=str,
        default="task",
        choices=["task", "layer"],
        help="plot x축 기준: task 또는 layer",
    )

    parser.add_argument(
        "--group_by",
        type=str,
        default="dataset",
        choices=["dataset"],
        help="현재는 dataset별 선 그래프만 지원",
    )

    return parser.parse_args()


def get_number_from_regex(text, pattern):
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))
    return None


def build_row_name_map(keys_to_extract, row_names):
    if row_names is None:
        return {k: k for k in keys_to_extract}
    if len(keys_to_extract) != len(row_names):
        raise ValueError(
            f"len(keys_to_extract)={len(keys_to_extract)} != len(row_names)={len(row_names)}"
        )
    return dict(zip(keys_to_extract, row_names))


def apply_excel_float_format(writer):
    for sheet_name in writer.sheets:
        ws = writer.sheets[sheet_name]
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "0.00000"


def find_eval_json_entries(base_dir, task_regex, layer_regex, seed_regex, json_pick_mode="all"):
    """
    재귀적으로 json을 찾고,
    - task_num, layer_num, seed_num: 경로 상의 모든 폴더명(parts)에서 탐색하여 추출
    """
    entries = []

    for root, _, files in os.walk(base_dir):
        json_files = sorted([f for f in files if f.endswith(".json")])
        if not json_files:
            continue

        rel_root = os.path.relpath(root, base_dir)
        # rel_root 전체에서도 한번 찾고, 폴더별로도 찾음
        task_num = get_number_from_regex(rel_root, task_regex)
        
        path_parts = rel_root.split(os.sep) if rel_root != "." else []
        layer_num = None
        seed_num = None

        for part in path_parts:
            # task_num이 rel_root에서 안 잡혔으면 각 part에서도 재확인
            if task_num is None:
                task_num = get_number_from_regex(part, task_regex)
            if layer_num is None:
                layer_num = get_number_from_regex(part, layer_regex)
            if seed_num is None:
                seed_num = get_number_from_regex(part, seed_regex)

        # task_num은 적어도 하나 있어야 함 (t0, t1...)
        if task_num is None:
            continue

        picked_files = json_files[:1] if json_pick_mode == "first" else json_files

        for jf in picked_files:
            entries.append({
                "root": root,
                "rel_root": rel_root,
                "json_file": jf,
                "json_path": os.path.join(root, jf),
                "task_num": task_num,
                "layer_num": layer_num,
                "seed_num": seed_num,
            })

    return entries


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    row_name_map = build_row_name_map(args.keys_to_extract, args.row_names)

    metric_columns = [m for m in args.metric_columns if m not in set(args.excluded_metrics)]
    plot_metrics = [m for m in args.plot_metrics if m in metric_columns]
    dataset_order = list(row_name_map.values())

    # 1. json 엔트리 재귀 수집
    entries = find_eval_json_entries(
        base_dir=args.base_dir,
        task_regex=args.task_regex,
        layer_regex=args.layer_regex,
        seed_regex=args.seed_regex,
        json_pick_mode=args.json_pick_mode,
    )

    filtered_entries = []
    for entry in entries:
        task_num = entry["task_num"]
        layer_num = entry["layer_num"]

        if args.min_task_num is not None and task_num < args.min_task_num:
            continue
        if args.max_task_num is not None and task_num > args.max_task_num:
            continue

        if args.x_axis_mode == "layer":
            if layer_num is None:
                continue
            if args.min_layer_num is not None and layer_num < args.min_layer_num:
                continue
            if args.max_layer_num is not None and layer_num > args.max_layer_num:
                continue

        filtered_entries.append(entry)

    filtered_entries.sort(
        key=lambda x: (
            x["task_num"],
            -1 if x["layer_num"] is None else x["layer_num"],
            -1 if x["seed_num"] is None else x["seed_num"],
            x["rel_root"],
            x["json_file"],
        )
    )

    # 2. records 생성
    records = []

    for entry in filtered_entries:
        json_path = entry["json_path"]

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read json: {json_path} ({e})")
            continue

        task_num = entry["task_num"]
        layer_num = entry["layer_num"]
        seed_num = entry["seed_num"]

        task_name = f"t{task_num}"
        layer_name = f"layer{layer_num}" if layer_num is not None else None

        for dataset_key in args.keys_to_extract:
            if dataset_key not in data:
                print(f"[WARN] key '{dataset_key}' not found in {json_path}")
                continue

            row_dataset_name = row_name_map.get(dataset_key, dataset_key)
            results = data[dataset_key].get("results", {})

            row = {
                "task_num": task_num,
                "task": task_name,
                "layer_num": layer_num,
                "layer": layer_name,
                "seed": seed_num,
                "dataset": row_dataset_name,
                "source_dataset_key": dataset_key,
                "dir_name": entry["rel_root"],
                "json_file": entry["json_file"],
                "json_path": json_path,
            }

            for metric in metric_columns:
                row[metric] = results.get(metric, None)

            records.append(row)

    if not records:
        print("[INFO] No records found. Nothing to save.")
        if args.x_axis_mode == "layer":
            print("[TIP] If your directory doesn't have 'layerN' names, try using '--x_axis_mode task' instead.")
        return

    # 3. raw DataFrame
    df = pd.DataFrame(records)
    df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order, ordered=True)

    for col in metric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sort_cols = ["task_num"]
    if "layer_num" in df.columns:
        sort_cols.append("layer_num")
    if "seed" in df.columns:
        sort_cols.append("seed")
    sort_cols.append("dataset")

    df = df.sort_values(sort_cols).reset_index(drop=True)

    # 4. seed 평균 테이블 생성
    group_cols = ["task_num", "task", "layer_num", "layer", "dataset", "source_dataset_key"]
    numeric_metric_columns = [col for col in metric_columns if df[col].notna().any()]

    agg_dict = {col: "mean" for col in numeric_metric_columns}
    grouped_df = (
        df.groupby(group_cols, dropna=False, observed=True)
        .agg(agg_dict)
        .reset_index()
    )

    seed_count_df = (
        df.groupby(group_cols, dropna=False, observed=True)["seed"]
        .nunique(dropna=True)
        .reset_index()
        .rename(columns={"seed": "num_seeds"})
    )

    grouped_df = grouped_df.merge(seed_count_df, on=group_cols, how="left")
    grouped_df["num_seeds"] = grouped_df["num_seeds"].fillna(0).astype(int)

    grouped_df["dataset"] = pd.Categorical(
        grouped_df["dataset"], categories=dataset_order, ordered=True
    )

    grouped_df = grouped_df.sort_values(
        ["task_num", "layer_num", "dataset"]
    ).reset_index(drop=True)

    if args.x_axis_mode == "layer":
        grouped_df["x_value"] = grouped_df["layer_num"]
        grouped_df["x_label"] = grouped_df["layer"].fillna("")
    else:
        grouped_df["x_value"] = grouped_df["task_num"]
        grouped_df["x_label"] = grouped_df["task"]

    # 5. Excel 테이블
    output_columns = ["task", "layer", "dataset", "num_seeds"] + numeric_metric_columns
    output_columns = [c for c in output_columns if c in grouped_df.columns]
    final_df = grouped_df[output_columns].copy()

    raw_output_columns = ["task", "layer", "seed", "dataset", "json_path"] + numeric_metric_columns
    raw_output_columns = [c for c in raw_output_columns if c in df.columns]
    raw_df = df[raw_output_columns].copy()

    # 6. dataset별 통계
    dataset_stats_df = pd.DataFrame()
    if numeric_metric_columns:
        dataset_stats_df = (
            grouped_df.groupby("dataset", observed=True)[numeric_metric_columns]
            .agg(["count", "mean", "std", "min", "median", "max"])
        )
        dataset_stats_df.columns = [
            f"{metric}_{stat}" for metric, stat in dataset_stats_df.columns
        ]
        dataset_stats_df = dataset_stats_df.reset_index()

    # 7. 전체 통계
    overall_stats_rows = []
    for col in numeric_metric_columns:
        s = pd.to_numeric(grouped_df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue

        overall_stats_rows.append({
            "metric": col,
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std()) if s.count() > 1 else None,
            "min": float(s.min()),
            "median": float(s.median()),
            "max": float(s.max()),
        })

    overall_stats_df = pd.DataFrame(overall_stats_rows)

    # 8. Excel 저장
    excel_path = os.path.join(args.output_dir, args.excel_filename)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        final_df = final_df.round(5)
        raw_df = raw_df.round(5)

        if not dataset_stats_df.empty:
            dataset_stats_df = dataset_stats_df.round(5)
        if not overall_stats_df.empty:
            overall_stats_df = overall_stats_df.round(5)

        final_df.to_excel(writer, sheet_name="seed_avg_table", index=False)
        raw_df.to_excel(writer, sheet_name="seed_raw_table", index=False)

        if not dataset_stats_df.empty:
            dataset_stats_df.to_excel(writer, sheet_name="dataset_stats", index=False)

        if not overall_stats_df.empty:
            overall_stats_df.to_excel(writer, sheet_name="overall_metric_stats", index=False)

        apply_excel_float_format(writer)

    print(f"[INFO] Saved Excel: {excel_path}")

    # 9. Plot 생성
    for metric in plot_metrics:
        if metric not in grouped_df.columns:
            print(f"[SKIP] plot metric '{metric}' not found in grouped_df")
            continue

        metric_numeric = pd.to_numeric(grouped_df[metric], errors="coerce")
        if metric_numeric.notna().sum() == 0:
            print(f"[SKIP] plot metric '{metric}' has no numeric values")
            continue

        plt.figure(figsize=(10, 6))

        plotted_any = False
        for ds in dataset_order:
            sub = grouped_df[grouped_df["dataset"] == ds].copy()
            if sub.empty:
                continue

            y = pd.to_numeric(sub[metric], errors="coerce")
            if y.notna().sum() == 0:
                continue

            sub = sub.assign(metric_numeric=y)
            sub = sub.dropna(subset=["x_value", "metric_numeric"]).sort_values("x_value")

            if sub.empty:
                continue

            plt.plot(
                sub["x_value"],
                sub["metric_numeric"],
                marker="o",
                label=ds
            )
            plotted_any = True

        if not plotted_any:
            plt.close()
            print(f"[SKIP] no valid dataset series for metric '{metric}'")
            continue

        if args.x_axis_mode == "layer":
            plt.title(f"{metric} across Layers (seed-averaged)")
            plt.xlabel("Layer")
        else:
            plt.title(f"{metric} across Tasks (seed-averaged)")
            plt.xlabel("Task")

        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        unique_x = sorted(grouped_df["x_value"].dropna().unique())
        if args.x_axis_mode == "task":
            plt.xticks(unique_x, [f"t{int(x)}" for x in unique_x])
        else:
            plt.xticks(unique_x)

        plt.tight_layout()

        plot_path = os.path.join(
            args.output_dir,
            f"{metric}_comparison_by_{args.x_axis_mode}_seed_avg.png"
        )
        plt.savefig(plot_path)
        plt.close()
        print(f"[INFO] Generated plot: {plot_path}")


if __name__ == "__main__":
    main()


# python plot_results_seed.py \
#   --base_dir "/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code/run_results/eval_basemoe_error/llama3.1-8b-instruct/clinc_moe_nn_fixed" \
#   --output_dir "./run_plots/clinc_moe_nn_fixed" \
#   --excel_filename "seed_avg_metrics.xlsx" \
#   --keys_to_extract forget retained mrpc rte \
#   --row_names forget retain utility1 utility2 \
#   --metric_columns choice_accuracy choice_correct gt_loss gt_loss_per_token hit_rate mrr perplexity \
#   --plot_metrics choice_accuracy hit_rate mrr perplexity \
#   --x_axis_mode task

# python plot_results_seed.py \
#   --base_dir "/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code/run_results/eval_basemoe_error/llama3.1-8b-instruct/tofu_moe_nn_fixed" \
#   --output_dir "./tofu_moe_nn_4_20" \
#   --excel_filename "seed_avg_metrics.xlsx" \
#   --keys_to_extract forget retained real_authors world_facts \
#   --row_names forget retain utility1 utility2 \
#   --metric_columns choice_accuracy choice_correct gt_loss gt_loss_per_token hit_rate mrr perplexity \
#   --plot_metrics choice_accuracy hit_rate mrr perplexity \
#   --x_axis_mode task
