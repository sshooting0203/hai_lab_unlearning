import os
from matplotlib import pyplot as plt
import numpy as np
import torch, einops, re
from torch import Tensor
from jaxtyping import Int, Float

TID2TOPIC_PAT = re.compile(r"(?:^|_)t\d+_([A-Za-z0-9\-]+)(?:$|_)")
def tid_to_topic(tid: str) -> str:  # scienceqa_t1_biology -> biology
    if not tid:
        return ""
    m = TID2TOPIC_PAT.search(tid)
    return (m.group(1) if m else tid).lower()

# (A) 도메인별 tau_by_layer 라인 플롯: holdout / retain 각각
def _plot_tau_by_layer(
    tau_by_layer_map,
    title: str,
    save_name: str,
    *,
    domains_to_plot,
    agg_domains=None,
    agg_label="HOLDOUT-AGG(median)",
    agg="median",
    layer_idxs, 
    plot_dir
):
    plt.figure()
    # 1) 도메인별 라인 (얇게)
    for dom in domains_to_plot:
        xs, ys = [], []
        for l in layer_idxs:
            v = tau_by_layer_map.get(dom, {}).get(l, None)
            if v is None:
                continue
            xs.append(l)
            ys.append(v)
        if len(xs) > 0:
            plt.plot(xs, ys, marker="o", linewidth=1.0, alpha=0.7, label=dom)
    # 2) 대표선 (굵게)
    if agg_domains is not None and len(agg_domains) > 0:
        agg_line = _aggregate_domain_line(
            tau_by_layer_map, layer_idxs=layer_idxs,
            domains_subset=agg_domains, agg=agg
        )
        xs, ys = [], []
        for l in layer_idxs:
            v = agg_line.get(l, None)
            if v is None:
                continue
            xs.append(l)
            ys.append(v)
        if len(xs) > 0:
            plt.plot(xs, ys, marker="o", linewidth=3.0, alpha=1.0, label=agg_label)
    plt.xlabel("Layer")
    plt.ylabel(f"tau_by_layer")
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, save_name), dpi=200)
    plt.close()

def _aggregate_domain_line(tau_by_layer_map, layer_idxs, domains_subset, agg="median"):
    out = {}
    for l in layer_idxs:
        vals = []
        for dom in domains_subset:
            v = tau_by_layer_map.get(dom, {}).get(l, None)
            if v is not None:
                vals.append(float(v))
        if len(vals) == 0:
            out[l] = None
        else:
            if agg == "mean":
                out[l] = float(np.mean(vals))
            else:  # default median
                out[l] = float(np.median(vals))
    return out

def _plot_smax_all_in_one(
    *,
    S_hold_by_dom: dict,
    S_ret_by_dom: dict,
    hold_domains,
    ret_domains,
    plot_dir: str,
    save_name: str,
    q: float,
    rep_line="holdout_tau",   # "holdout_tau" | "holdout_mean" | "holdout_median"
    bins: int = 60,
):
    """
    - 도메인별 S_max 분포를 한 plot에 모두 올림
    - holdout 도메인들은 같은 스타일(예: 실선), retain 도메인들은 다른 스타일(예: 점선)
    - 대표선은 holdout 전체를 합친 값으로 하나의 수평선 추가
    """
    import numpy as np
    import torch
    import os
    from matplotlib import pyplot as plt

    def _cat(domains, S_by_dom):
        xs = []
        for d in domains:
            v = S_by_dom.get(d)
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                if v.numel() == 0:
                    continue
                xs.append(v.detach().float().cpu())
            else:
                # 혹시 numpy/list로 들어오면
                vv = torch.tensor(v, dtype=torch.float32)
                if vv.numel() == 0:
                    continue
                xs.append(vv)
        if len(xs) == 0:
            return torch.empty(0)
        return torch.cat(xs, dim=0)

    # ---- 합친 분포(holdout 전체 / retain 전체)
    S_hold_all = _cat(hold_domains, S_hold_by_dom)
    S_ret_all  = _cat(ret_domains,  S_ret_by_dom)

    if S_hold_all.numel() == 0 and S_ret_all.numel() == 0:
        print("[PLOT] skip _plot_smax_all_in_one: both empty")
        return

    # ---- 대표선 값 계산
    rep_y = None
    if rep_line == "holdout_tau":
        if S_hold_all.numel() > 0:
            rep_y = float(torch.quantile(S_hold_all, q).item())
    elif rep_line == "holdout_mean":
        if S_hold_all.numel() > 0:
            rep_y = float(S_hold_all.mean().item())
    elif rep_line == "holdout_median":
        if S_hold_all.numel() > 0:
            rep_y = float(S_hold_all.median().item())
    else:
        raise ValueError(f"Unknown rep_line={rep_line}")

    # ---- bins 범위 고정(둘 다 합쳐서)
    all_np = []
    if S_hold_all.numel() > 0:
        all_np.append(S_hold_all.numpy())
    if S_ret_all.numel() > 0:
        all_np.append(S_ret_all.numpy())
    all_np = np.concatenate(all_np, axis=0) if len(all_np) else np.array([])
    if all_np.size > 0:
        xmin, xmax = float(all_np.min()), float(all_np.max())
        # 너무 딱 붙으면 보기 안 좋아서 약간 padding
        pad = (xmax - xmin) * 0.03 if xmax > xmin else 0.01
        rng = (xmin - pad, xmax + pad)
    else:
        rng = None

    plt.figure()

    # ---- 도메인별 hist 겹쳐 그리기 (투명도 낮게)
    # holdout 도메인들
    for dom in sorted(hold_domains):
        v = S_hold_by_dom.get(dom, None)
        if v is None or (isinstance(v, torch.Tensor) and v.numel() == 0):
            continue
        arr = v.detach().float().cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v, dtype=np.float32)
        plt.hist(arr, bins=bins, range=rng, density=True, alpha=0.25, label=f"H:{dom}")

    # retain 도메인들
    for dom in sorted(ret_domains):
        v = S_ret_by_dom.get(dom, None)
        if v is None or (isinstance(v, torch.Tensor) and v.numel() == 0):
            continue
        arr = v.detach().float().cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v, dtype=np.float32)
        plt.hist(arr, bins=bins, range=rng, density=True, alpha=0.25, label=f"R:{dom}")

    # ---- 전체(holdout / retain)도 굵게 한 번 더
    if S_hold_all.numel() > 0:
        plt.hist(S_hold_all.numpy(), bins=bins, range=rng, density=True, alpha=0.35, label="H:ALL", histtype="step", linewidth=2.0)
    if S_ret_all.numel() > 0:
        plt.hist(S_ret_all.numpy(),  bins=bins, range=rng, density=True, alpha=0.35, label="R:ALL", histtype="step", linewidth=2.0)

    # ---- 대표선(수평선)
    if rep_y is not None:
        plt.axvline(rep_y, linewidth=3.0, alpha=1.0, label=f"REP({rep_line})={rep_y:.4f}")

    plt.xlabel("S_max = max_l S_l (vs representative dists)")
    plt.ylabel("density")
    plt.title(f"S_max distribution (holdout+retain) + representative line, q={q}")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, save_name), dpi=200)
    plt.close()

def _plot_smax_id_ood(
    *,
    S_hold_by_dom: dict,
    S_ret_by_dom: dict,
    hold_domains,
    ret_domains,
    plot_dir: str,
    save_name: str,
    q: float,
    rep_line="holdout_tau",   # "holdout_tau" | "holdout_mean" | "holdout_median"
    bins: int = 80,
    show_per_domain: bool = False,  # True면 도메인별도 얇게 깔기
):
    import numpy as np
    import torch
    import os
    from matplotlib import pyplot as plt

    def _cat(domains, S_by_dom):
        xs = []
        for d in domains:
            v = S_by_dom.get(d)
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                if v.numel() == 0:
                    continue
                xs.append(v.detach().float().cpu())
            else:
                vv = torch.tensor(v, dtype=torch.float32)
                if vv.numel() == 0:
                    continue
                xs.append(vv)
        if len(xs) == 0:
            return torch.empty(0)
        return torch.cat(xs, dim=0)

    # ---- 그룹 단위: ID=holdout, OOD=retain
    S_id  = _cat(hold_domains, S_hold_by_dom)
    S_ood = _cat(ret_domains,  S_ret_by_dom)

    if S_id.numel() == 0 and S_ood.numel() == 0:
        print("[PLOT] skip _plot_smax_id_ood: both empty")
        return

    # ---- 대표선(holdout 기반)
    rep_y = None
    if rep_line == "holdout_tau":
        if S_id.numel() > 0:
            rep_y = float(torch.quantile(S_id, q).item())
    elif rep_line == "holdout_mean":
        if S_id.numel() > 0:
            rep_y = float(S_id.mean().item())
    elif rep_line == "holdout_median":
        if S_id.numel() > 0:
            rep_y = float(S_id.median().item())
    else:
        raise ValueError(f"Unknown rep_line={rep_line}")

    # ---- x range 고정(두 그룹 합쳐서)
    all_np = []
    if S_id.numel() > 0:
        all_np.append(S_id.numpy())
    if S_ood.numel() > 0:
        all_np.append(S_ood.numpy())
    all_np = np.concatenate(all_np, axis=0) if len(all_np) else np.array([])
    rng = None
    if all_np.size > 0:
        xmin, xmax = float(all_np.min()), float(all_np.max())
        pad = (xmax - xmin) * 0.03 if xmax > xmin else 0.01
        rng = (xmin - pad, xmax + pad)

    plt.figure()

    # ---- (옵션) 도메인별 얇게 깔기
    if show_per_domain:
        for dom in sorted(hold_domains):
            v = S_hold_by_dom.get(dom, None)
            if v is None or (isinstance(v, torch.Tensor) and v.numel() == 0):
                continue
            arr = v.detach().float().cpu().numpy()
            plt.hist(arr, bins=bins, range=rng, density=True, alpha=0.08, label=None)

        for dom in sorted(ret_domains):
            v = S_ret_by_dom.get(dom, None)
            if v is None or (isinstance(v, torch.Tensor) and v.numel() == 0):
                continue
            arr = v.detach().float().cpu().numpy()
            plt.hist(arr, bins=bins, range=rng, density=True, alpha=0.08, label=None)

    # ---- 그룹 2개만 굵게 (ID vs OOD)
    if S_id.numel() > 0:
        plt.hist(S_id.numpy(), bins=bins, range=rng, density=True,
                 alpha=0.35, label="ID (holdout)", histtype="stepfilled")
        plt.hist(S_id.numpy(), bins=bins, range=rng, density=True,
                 alpha=0.9, label="ID (holdout) outline", histtype="step", linewidth=2.0)

    if S_ood.numel() > 0:
        plt.hist(S_ood.numpy(), bins=bins, range=rng, density=True,
                 alpha=0.35, label="OOD (retain)", histtype="stepfilled")
        plt.hist(S_ood.numpy(), bins=bins, range=rng, density=True,
                 alpha=0.9, label="OOD (retain) outline", histtype="step", linewidth=2.0)

    # ---- 대표선(세로선)
    if rep_y is not None:
        plt.axvline(rep_y, linewidth=3.0, alpha=1.0, label=f"REP({rep_line})={rep_y:.4f}")

    plt.xlabel("S_max = max_l S_l (vs representative dists)")
    plt.ylabel("density")
    plt.title(f"ID vs OOD S_max distribution + representative line, q={q}")
    plt.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, save_name), dpi=200)
    plt.close()
