import os, json, re
from datetime import datetime

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def setting_key(token_pool: str, module: str, last_k: int, whole_reduce=None, whole_topk=None) -> str:
    if token_pool == "whole":
        return f"pool={token_pool}|mod={module}|k={last_k}|reduce={whole_reduce}|topk={whole_topk}"
    return f"pool={token_pool}|mod={module}|k={last_k}"

def fmt_row_short(r: dict) -> str:
    return (
        f"layer={r['layer']:2d} d={r['cohens_d']:+.3f} auc={r['auroc_forget_vs_retain']:.3f} "
        f"ks={r['ks']:.3f} gap={r['mean_gap']:+.3f} "
        f"tau@{r['tau_quantile']:.2f}={r['tau_from_forget']:.3f} fpr~{r['retain_fpr_proxy_at_tau']:.3f}"
    )

def parse_ts_from_run_dirname(name: str):
    m = re.search(r"(\d{8}_\d{6})$", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    except Exception:
        return None


def export_domain_auc_ge_to_txt(
    root_dir: str,
    out_txt_path: str,
    domains=None,
    pick_latest: bool = True,
    auc_thr: float = 0.90,
):
    """
    각 도메인 최신 run_dir/single/results_sorted.json 에서
    auroc_forget_vs_retain >= auc_thr 인 row를 모두 모아 txt로 저장.
    """
    if domains is None:
        domains = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    lines = []
    lines.append(f"[ROOT] {root_dir}")
    lines.append(f"[FILTER] auroc_forget_vs_retain >= {auc_thr:.2f}")
    lines.append("")

    total_kept = 0
    total_rows = 0

    for dom in domains:
        domain_dir = os.path.join(root_dir, dom)
        if not os.path.isdir(domain_dir):
            continue

        run_dir = find_latest_run_dir(domain_dir) if pick_latest else domain_dir
        if run_dir is None:
            lines.append(f"==================== DOMAIN: {dom} ====================")
            lines.append("[SKIP] no run dirs found")
            lines.append("")
            continue

        path = os.path.join(run_dir, "single", "results_sorted.json")
        if not os.path.exists(path):
            lines.append(f"==================== DOMAIN: {dom} ====================")
            lines.append(f"[RUN] {run_dir}")
            lines.append(f"[SKIP] missing {path}")
            lines.append("")
            continue

        obj = load_json(path)
        rows = obj.get("results", [])
        if not isinstance(rows, list):
            lines.append(f"==================== DOMAIN: {dom} ====================")
            lines.append(f"[RUN] {run_dir}")
            lines.append("[SKIP] bad format (results is not a list)")
            lines.append("")
            continue

        kept = []
        for r in rows:
            auc = r.get("auroc_forget_vs_retain", None)
            if isinstance(auc, (int, float)) and auc >= auc_thr:
                kept.append(r)

        def auc_sort_key(r):
            auc = r.get("auroc_forget_vs_retain", float("-inf"))
            ad  = r.get("abs_cohens_d", float("-inf"))
            ks  = r.get("ks", float("-inf"))
            amg = r.get("abs_mean_gap", float("-inf"))
            return (auc, ad, ks, amg)

        kept.sort(key=auc_sort_key, reverse=True)

        total_kept += len(kept)
        total_rows += len(rows)

        lines.append(f"==================== DOMAIN: {dom} ====================")
        lines.append(f"[RUN] {run_dir}")
        lines.append(f"[COUNT] kept={len(kept)}/{len(rows)}")
        lines.append("")

        if not kept:
            lines.append("(none)")
            lines.append("")
            continue

        for i, r in enumerate(kept):
            extra = ""
            if r.get("token_pool") == "whole":
                extra = f"| reduce={r.get('whole_reduce')} topk={r.get('whole_topk')}"
            lines.append(
                f"[{i:03d}] "
                f"{setting_key(r['token_pool'], r['module'], r['last_k'], r.get('whole_reduce'), r.get('whole_topk'))} "
                f"| {fmt_row_short(r)} {extra}"
            )
        lines.append("")

    lines.append(f"[TOTAL] kept={total_kept}/{total_rows}")

    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[SAVED] {out_txt_path}")

def find_latest_run_dir(domain_dir: str):
    cands = []
    for d in os.listdir(domain_dir):
        p = os.path.join(domain_dir, d)
        if not os.path.isdir(p):
            continue
        ts = parse_ts_from_run_dirname(d)
        if ts is None:
            continue
        cands.append((ts, p))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return cands[-1][1]

def print_domain_auc_ge(root_dir: str, domains=None, pick_latest=True, auc_thr=0.90):
    """
    root_dir 예:
      /.../forget_analyze_runs/llama3.1-8b-instruct/seed0

    각 도메인 최신 run_dir/single/results_sorted.json 에서
    auroc_forget_vs_retain >= auc_thr 인 row를 모두 출력.
    """
    if domains is None:
        domains = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    for dom in domains:
        domain_dir = os.path.join(root_dir, dom)
        if not os.path.isdir(domain_dir):
            continue

        run_dir = find_latest_run_dir(domain_dir) if pick_latest else domain_dir
        if run_dir is None:
            print(f"\n[SKIP] {dom}: no run dirs found")
            continue

        path = os.path.join(run_dir, "single", "results_sorted.json")
        if not os.path.exists(path):
            print(f"\n[SKIP] {dom}: missing {path}")
            continue

        obj = load_json(path)
        rows = obj.get("results", [])
        if not isinstance(rows, list):
            print(f"\n[SKIP] {dom}: bad format (results is not a list)")
            continue

        # results_sorted.json은 이미 정렬된 상태라고 가정 (너 코드가 그렇게 저장함)
        kept = []
        for r in rows:
            auc = r.get("auroc_forget_vs_retain", None)
            if isinstance(auc, (int, float)) and auc >= auc_thr:
                kept.append(r)

        print(f"\n==================== DOMAIN: {dom} ====================")
        print(f"[RUN] {run_dir}")
        print(f"[FILTER] auroc_forget_vs_retain >= {auc_thr:.2f} | kept={len(kept)}/{len(rows)}")
        print("==================== TOP SEPARATION (filtered) ====================")

        if not kept:
            print("(none)")
            continue

        for i, r in enumerate(kept):
            extra = ""
            if r.get("token_pool") == "whole":
                extra = f"| reduce={r.get('whole_reduce')} topk={r.get('whole_topk')}"
            print(
                f"[{i:03d}] "
                f"{setting_key(r['token_pool'], r['module'], r['last_k'], r.get('whole_reduce'), r.get('whole_topk'))} "
                f"| {fmt_row_short(r)} {extra}"
            )

# 사용 예시:
root = "/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code/forget_analyze_runs/llama3.1-8b-instruct/seed0"

domains = [
  "biology",
  "writing-strategies",
  "figurative-language",
  "units-and-measurement",
  "physics",
  "reference-skills",
  "chemistry",
  "punctuation",
  "economics",
  "grammar",
  "earth-science",
]

# print_domain_auc_ge(root_dir=root, domains=domains, pick_latest=True, auc_thr=0.90)

out_txt = os.path.join(root, "domain_auc_ge_0.90.txt")
export_domain_auc_ge_to_txt(root_dir=root, out_txt_path=out_txt, domains=domains, pick_latest=True, auc_thr=0.90)
