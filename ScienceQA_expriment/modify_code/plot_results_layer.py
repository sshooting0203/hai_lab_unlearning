import os
import json
import re
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="eval 결과 json 디렉토리 루트. 예: base_dir/layer8/<task_dirs>/...",
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
        ],
        help="plot 생성할 metric들",
    )

    parser.add_argument(
        "--task_regex",
        type=str,
        default=r"_t(\d+)_",
        help="task 번호 추출용 regex",
    )
    parser.add_argument(
        "--layer_regex",
        type=str,
        default=r"layer(\d+)$",
        help="layer 번호 추출용 regex. 부모 폴더명에서 추출",
    )

    parser.add_argument(
        "--json_pick_mode",
        type=str,
        default="first",
        choices=["first"],
        help="현재는 각 task 디렉토리의 첫 번째 json만 사용",
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
        choices=["dataset", "task_dataset"],
        help=(
            "dataset: dataset별 선 그래프 "
            "| task_dataset: task별로 metric plot을 따로 만들고 각 plot에서 dataset별 선 그래프"
        ),
    )

    parser.add_argument(
        "--recursive_base",
        action="store_true",
        help=(
            "base_dir 바로 아래가 task 디렉토리인 flat 구조가 아니라 "
            "base_dir/layer8/<task_dir> 같은 2단계 구조일 때 사용. "
            "현재 기본 동작도 layer 하위 task 구조를 우선 지원함."
        ),
    )

    return parser.parse_args()


def get_number_from_regex(text, pattern):
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))
    return None


def is_numeric_value(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)


def build_row_name_map(keys_to_extract, row_names):
    if row_names is None:
        return {k: k for k in keys_to_extract}
    if len(keys_to_extract) != len(row_names):
        raise ValueError(
            f"len(keys_to_extract)={len(keys_to_extract)} != len(row_names)={len(row_names)}"
        )
    return dict(zip(keys_to_extract, row_names))


def collect_run_dirs_nested(args):
    """
    기대 구조:
      base_dir/
        layer8/
          lunar_..._t1_xxx/
          lunar_..._t2_xxx/
        layer9/
          ...

    반환:
      [(task_num, layer_num, layer_dir_name, task_dir_name), ...]
    """
    run_dirs = []

    if not os.path.isdir(args.base_dir):
        raise FileNotFoundError(f"base_dir not found: {args.base_dir}")

    layer_dirs = sorted([
        d for d in os.listdir(args.base_dir)
        if os.path.isdir(os.path.join(args.base_dir, d))
    ])

    for layer_dir in layer_dirs:
        layer_num = get_number_from_regex(layer_dir, args.layer_regex)
        if layer_num is None:
            continue

        if args.min_layer_num is not None and layer_num < args.min_layer_num:
            continue
        if args.max_layer_num is not None and layer_num > args.max_layer_num:
            continue

        layer_path = os.path.join(args.base_dir, layer_dir)
        task_dirs = sorted([
            d for d in os.listdir(layer_path)
            if os.path.isdir(os.path.join(layer_path, d))
        ])

        for task_dir in task_dirs:
            task_num = get_number_from_regex(task_dir, args.task_regex)
            if task_num is None:
                continue

            if args.min_task_num is not None and task_num < args.min_task_num:
                continue
            if args.max_task_num is not None and task_num > args.max_task_num:
                continue

            run_dirs.append((task_num, layer_num, layer_dir, task_dir))

    run_dirs.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return run_dirs


def collect_run_dirs_flat(args):
    """
    기존 flat 구조 fallback:
      base_dir/
        lunar_..._t1_xxx_layer8/
        lunar_..._t2_xxx_layer8/
        ...

    반환:
      [(task_num, layer_num, None, task_dir_name), ...]
    """
    run_dirs = []

    if not os.path.isdir(args.base_dir):
        raise FileNotFoundError(f"base_dir not found: {args.base_dir}")

    subdirs = sorted([
        d for d in os.listdir(args.base_dir)
        if os.path.isdir(os.path.join(args.base_dir, d))
    ])

    for d in subdirs:
        task_num = get_number_from_regex(d, args.task_regex)
        layer_num = get_number_from_regex(d, args.layer_regex)

        if task_num is None:
            continue

        if args.min_task_num is not None and task_num < args.min_task_num:
            continue
        if args.max_task_num is not None and task_num > args.max_task_num:
            continue

        if layer_num is not None:
            if args.min_layer_num is not None and layer_num < args.min_layer_num:
                continue
            if args.max_layer_num is not None and layer_num > args.max_layer_num:
                continue

        run_dirs.append((task_num, -1 if layer_num is None else layer_num, None, d))

    run_dirs.sort(key=lambda x: (x[0], x[1], x[3]))
    return run_dirs


def collect_run_dirs(args):
    """
    우선 nested 구조를 시도하고, 결과가 없으면 flat 구조를 시도.
    """
    nested = collect_run_dirs_nested(args)
    if nested:
        print(f"[INFO] Detected nested layer/task structure under: {args.base_dir}")
        return nested

    flat = collect_run_dirs_flat(args)
    if flat:
        print(f"[INFO] Detected flat task directory structure under: {args.base_dir}")
        return flat

    return []


def pick_json_file(dir_path, json_pick_mode="first"):
    json_files = sorted([f for f in os.listdir(dir_path) if f.endswith(".json")])
    if not json_files:
        return None, []

    if json_pick_mode == "first":
        return os.path.join(dir_path, json_files[0]), json_files

    return os.path.join(dir_path, json_files[0]), json_files


def sanitize_filename(text):
    text = str(text)
    text = text.replace("/", "_")
    text = re.sub(r"[^\w\-.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    row_name_map = build_row_name_map(args.keys_to_extract, args.row_names)

    metric_columns = [m for m in args.metric_columns if m not in set(args.excluded_metrics)]
    plot_metrics = [m for m in args.plot_metrics if m in metric_columns]
    dataset_order = list(row_name_map.values())

    # 1. 디렉토리 수집
    run_dirs = collect_run_dirs(args)

    if not run_dirs:
        print("[INFO] No run directories found. Check base_dir and regex settings.")
        return

    # 2. records 생성
    records = []

    for task_num, layer_num, layer_dir, task_dir in run_dirs:
        if layer_dir is None:
            dir_path = os.path.join(args.base_dir, task_dir)
        else:
            dir_path = os.path.join(args.base_dir, layer_dir, task_dir)

        if not os.path.isdir(dir_path):
            print(f"[WARN] Not a directory: {dir_path}")
            continue

        json_path, json_files = pick_json_file(dir_path, args.json_pick_mode)
        if json_path is None:
            print(f"[WARN] No json in {dir_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        task_name = f"t{task_num}"
        layer_name = f"layer{layer_num}" if layer_num >= 0 else None

        for dataset_key in args.keys_to_extract:
            if dataset_key not in data:
                print(f"[WARN] key '{dataset_key}' not found in {json_path}")
                continue

            row_dataset_name = row_name_map.get(dataset_key, dataset_key)
            results = data[dataset_key].get("results", {})

            row = {
                "task_num": task_num,
                "task": task_name,
                "layer_num": None if layer_num < 0 else layer_num,
                "layer": layer_name,
                "dataset": row_dataset_name,
                "source_dataset_key": dataset_key,
                "layer_dir": layer_dir,
                "dir_name": task_dir,
                "json_file": os.path.basename(json_path),
            }

            for metric in metric_columns:
                row[metric] = results.get(metric, None)

            records.append(row)

    if not records:
        print("[INFO] No records found. Nothing to save.")
        return

    # 3. DataFrame 생성
    df = pd.DataFrame(records)

    df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order, ordered=True)

    sort_cols = ["task_num"]
    if "layer_num" in df.columns:
        sort_cols.append("layer_num")
    sort_cols.append("dataset")

    df = df.sort_values(sort_cols).reset_index(drop=True)

    # x축용 컬럼 생성
    if args.x_axis_mode == "layer":
        df["x_value"] = df["layer_num"]
        df["x_label"] = df["layer"].fillna("")
    else:
        df["x_value"] = df["task_num"]
        df["x_label"] = df["task"]

    # 4. Excel 메인 테이블
    output_columns = [
        "task",
        "layer",
        "dataset",
        "source_dataset_key",
        "dir_name",
        "json_file",
    ] + metric_columns
    output_columns = [c for c in output_columns if c in df.columns]
    final_df = df[output_columns].copy()

    # 5. numeric metric 탐색
    numeric_metric_columns = []
    for col in metric_columns:
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        if numeric_series.notna().any():
            numeric_metric_columns.append(col)

    # dataset별 통계
    dataset_stats_df = pd.DataFrame()
    if numeric_metric_columns:
        dataset_stats_df = (
            df.groupby("dataset", observed=False)[numeric_metric_columns]
            .agg(["count", "mean", "std", "min", "median", "max"])
        )
        dataset_stats_df.columns = [
            f"{metric}_{stat}" for metric, stat in dataset_stats_df.columns
        ]
        dataset_stats_df = dataset_stats_df.reset_index()

    # task + dataset별 통계
    task_dataset_stats_df = pd.DataFrame()
    if numeric_metric_columns:
        task_dataset_stats_df = (
            df.groupby(["task", "dataset"], observed=False)[numeric_metric_columns]
            .agg(["count", "mean", "std", "min", "median", "max"])
        )
        task_dataset_stats_df.columns = [
            f"{metric}_{stat}" for metric, stat in task_dataset_stats_df.columns
        ]
        task_dataset_stats_df = task_dataset_stats_df.reset_index()

    # 전체 metric 통계
    overall_stats_rows = []
    for col in numeric_metric_columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
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

    # pivot 테이블도 같이 저장
    pivot_df = pd.DataFrame()
    if numeric_metric_columns:
        pivot_candidates = ["choice_accuracy", "hit_rate", "mrr", "perplexity", "gt_loss", "gt_loss_per_token"]
        pivot_metrics = [m for m in pivot_candidates if m in numeric_metric_columns]
        if pivot_metrics:
            pivot_frames = []
            for metric in pivot_metrics:
                tmp = df.pivot_table(
                    index=["task", "dataset"],
                    columns="layer",
                    values=metric,
                    aggfunc="first",
                ).reset_index()
                tmp.insert(0, "metric", metric)
                pivot_frames.append(tmp)
            if pivot_frames:
                pivot_df = pd.concat(pivot_frames, axis=0, ignore_index=True)

    # 6. Excel 저장
    excel_path = os.path.join(args.output_dir, args.excel_filename)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="task_dataset_table", index=False)

        if not dataset_stats_df.empty:
            dataset_stats_df.to_excel(writer, sheet_name="dataset_stats", index=False)

        if not task_dataset_stats_df.empty:
            task_dataset_stats_df.to_excel(writer, sheet_name="task_dataset_stats", index=False)

        if not overall_stats_df.empty:
            overall_stats_df.to_excel(writer, sheet_name="overall_metric_stats", index=False)

        if not pivot_df.empty:
            pivot_df.to_excel(writer, sheet_name="task_dataset_by_layer", index=False)

    print(f"[INFO] Saved Excel: {excel_path}")

    # 7. Plot 생성
    for metric in plot_metrics:
        metric_numeric = pd.to_numeric(df[metric], errors="coerce")
        if metric_numeric.notna().sum() == 0:
            print(f"[SKIP] plot metric '{metric}' has no numeric values")
            continue

        # ---------------------------------------------------------
        # A. dataset별 1개 plot
        # ---------------------------------------------------------
        if args.group_by == "dataset":
            plt.figure(figsize=(10, 6))

            plotted_any = False
            for ds in dataset_order:
                sub = df[df["dataset"] == ds].copy()
                if sub.empty:
                    continue

                y = pd.to_numeric(sub[metric], errors="coerce")
                if y.notna().sum() == 0:
                    continue

                sub = sub.assign(metric_numeric=y)
                sub = sub.dropna(subset=["x_value", "metric_numeric"]).sort_values("x_value")

                if sub.empty:
                    continue

                # 같은 x값이 여러 개일 수 있으니 평균으로 집계
                sub = (
                    sub.groupby(["x_value", "x_label"], as_index=False)["metric_numeric"]
                    .mean()
                    .sort_values("x_value")
                )

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
                plt.title(f"{metric} across Layers")
                plt.xlabel("Layer")
            else:
                plt.title(f"{metric} across Tasks")
                plt.xlabel("Task")

            plt.ylabel(metric)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)

            unique_x = sorted(df["x_value"].dropna().unique())
            if args.x_axis_mode == "task":
                plt.xticks(unique_x, [f"t{int(x)}" for x in unique_x])
            else:
                plt.xticks(unique_x)

            plt.tight_layout()

            plot_path = os.path.join(
                args.output_dir,
                f"{sanitize_filename(metric)}_comparison_by_{args.x_axis_mode}.png"
            )
            plt.savefig(plot_path)
            plt.close()
            print(f"[INFO] Generated plot: {plot_path}")

        # ---------------------------------------------------------
        # B. task별로 plot 따로 생성, 각 plot 안에는 dataset별 선
        #    특히 x_axis_mode=layer일 때 유용
        # ---------------------------------------------------------
        elif args.group_by == "task_dataset":
            unique_tasks = sorted(df["task_num"].dropna().unique())

            for task_num in unique_tasks:
                task_name = f"t{int(task_num)}"
                task_df = df[df["task_num"] == task_num].copy()
                if task_df.empty:
                    continue

                plt.figure(figsize=(10, 6))
                plotted_any = False

                for ds in dataset_order:
                    sub = task_df[task_df["dataset"] == ds].copy()
                    if sub.empty:
                        continue

                    y = pd.to_numeric(sub[metric], errors="coerce")
                    if y.notna().sum() == 0:
                        continue

                    sub = sub.assign(metric_numeric=y)
                    sub = sub.dropna(subset=["x_value", "metric_numeric"]).sort_values("x_value")

                    if sub.empty:
                        continue

                    sub = (
                        sub.groupby(["x_value", "x_label"], as_index=False)["metric_numeric"]
                        .mean()
                        .sort_values("x_value")
                    )

                    plt.plot(
                        sub["x_value"],
                        sub["metric_numeric"],
                        marker="o",
                        label=ds
                    )
                    plotted_any = True

                if not plotted_any:
                    plt.close()
                    print(f"[SKIP] no valid dataset series for metric '{metric}' in {task_name}")
                    continue

                if args.x_axis_mode == "layer":
                    plt.title(f"{metric} across Layers ({task_name})")
                    plt.xlabel("Layer")
                else:
                    plt.title(f"{metric} across Tasks ({task_name})")
                    plt.xlabel("Task")

                plt.ylabel(metric)
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.7)

                unique_x = sorted(task_df["x_value"].dropna().unique())
                if args.x_axis_mode == "task":
                    plt.xticks(unique_x, [f"t{int(x)}" for x in unique_x])
                else:
                    plt.xticks(unique_x)

                plt.tight_layout()

                plot_path = os.path.join(
                    args.output_dir,
                    f"{sanitize_filename(metric)}_{task_name}_by_{args.x_axis_mode}.png"
                )
                plt.savefig(plot_path)
                plt.close()
                print(f"[INFO] Generated plot: {plot_path}")


if __name__ == "__main__":
    main()

# python plot_results_layer.py \
#   --base_dir "/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code/run_results/eval_basemoe_error/llama3.1-8b-instruct/scienceqa_lunar_base_layer_search" \
#   --output_dir "./run_plots/scienceqa_by_layer" \
#   --excel_filename "task_dataset_metrics.xlsx" \
#   --keys_to_extract forget retained commonqa openbookqa \
#   --row_names forget retain utility1 utility2 \
#   --metric_columns \
#     choice_accuracy \
#     choice_correct \
#     gt_loss \
#     gt_loss_per_token \
#     hit_rate \
#     mrr \
#     label_contains_accuracy \
#     label_contains_correct \
#     perplexity \
#   --plot_metrics \
#     choice_accuracy \
#     hit_rate \
#     mrr \
#     perplexity \
#     gt_loss \
#     gt_loss_per_token \
#     label_contains_accuracy \
#   --x_axis_mode layer \
#   --group_by task_dataset