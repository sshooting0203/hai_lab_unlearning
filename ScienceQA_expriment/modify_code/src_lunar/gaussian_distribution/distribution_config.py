# save_one_task_distributions_qv( # 통계량 계산 
#     cfg,
#     model_base,
#     forget_dataset, # 통계량에 forget 데이터셋 사용(->분할)
#     cfg.middle_distribution_layer, # 통계량을 계산할 layer
#     cfg.save_distribution_path, # 통계량을 저장할 path
#     cfg.task_id,
#     cfg.dist_batch_size,
#     retain_dataset
# )
# print(f"[SAVE] Distribution QV : {len(forget_dataset)}")
import os, torch, pickle, re, math
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import random_split
from src_lunar.gaussian_distribution.gaussian_plot import _plot_smax_all_in_one, _plot_smax_id_ood, _plot_tau_by_layer, tid_to_topic
from src_lunar.gaussian_distribution.gaussian import GaussianDistribution, to_device_dtype
from src_lunar.gaussian_distribution.distribution_utils import (
    cosine_sim, 
    safe_normalize,
    reduce_token_scores,
    collect_qv_feats_with_hooks,
    compute_similarity_scores
)


SHARED_META = {"dist_token_pool", "dist_last_k"}
ALLOW_META = { "tau_by_layer", "tau_debug", "dist_token_pool", "dist_last_k", "task_id" }

def calculate_distances_unified(
    cfg, 
    features: Union[torch.Tensor, Dict[str, torch.Tensor]],   # pooled: [B,D], whole: {"x":[B,T,D], "mask":[B,T]}
    distributions: List[GaussianDistribution],
    distance_type: str = "cosine",
    temperature: float = 1.0,
    whole_reduce: str = "mean",
    whole_topk: int = 8,
):
    with torch.no_grad():
        is_whole = isinstance(features, dict)  # is_whole ->  {"x": [B,T,D], "mask": [B,T]}
        if is_whole:
            print("[DEBUG] is_whole : ", is_whole)
            x = features["x"]          # [B,T,D] : 토큰 단위 전체 시퀀스
            mask = features["mask"]    # [B,T]
        else:
            x = features              # [B,D] : 이미 샘플당 1벡터로 풀링된 표현
            mask = None

        distances = []
        for gaussian in distributions:
            mean = gaussian.mean.to(device=x.device, dtype=x.dtype)
            var = gaussian.var.to(device=x.device, dtype=x.dtype)
            if not is_whole: # [B,D] : 이미 샘플당 1벡터로 풀링된 표현 -> [B]
                if distance_type == "L2":
                    dist = torch.norm(x - mean, dim=1)
                elif distance_type == "gaussian":
                    var2 = var + 1e-6
                    log_prob = -0.5 * (torch.log(2 * torch.pi * var2) + ((x - mean) ** 2) / var2)
                    dist = log_prob.sum(dim=1)
                elif distance_type == "attention":
                    dim = x.size(-1)
                    dist = torch.matmul(x, mean) / math.sqrt(float(dim))
                elif distance_type == "cosine":
                    dist = cosine_sim(x, mean, eps=1e-8)
                else:
                    raise ValueError(f"Unsupported distance type: {distance_type}")
            else: # [B,T,D] : 토큰 단위 전체 시퀀스 -> [B,T] -> -> [B]
                if distance_type == "L2":
                    tok = torch.norm(x - mean.view(1, 1, -1), dim=-1)  # [B,T]
                    dist = reduce_token_scores(tok, mask, reduce=whole_reduce, topk=whole_topk)
                elif distance_type == "gaussian":
                    var2 = var + 1e-6
                    diff2 = (x - mean.view(1, 1, -1)) ** 2
                    tok_lp = -0.5 * (torch.log(2 * torch.pi * var2).view(1, 1, -1) + diff2 / var2.view(1, 1, -1))
                    tok = tok_lp.sum(dim=-1)  # [B,T]
                    dist = reduce_token_scores(tok, mask, reduce=whole_reduce, topk=whole_topk)
                elif distance_type == "attention":
                    dim = x.size(-1)
                    tok = torch.matmul(x, mean) / math.sqrt(float(dim))  # [B,T]
                    dist = reduce_token_scores(tok, mask, reduce=whole_reduce, topk=whole_topk)
                elif distance_type == "cosine":
                    x_n = safe_normalize(x, dim=-1, eps=1e-8)  # [B,T,D]
                    mu_n = safe_normalize(mean, dim=0, eps=1e-8)  # [D]
                    tok = (x_n * mu_n.view(1, 1, -1)).sum(dim=-1)  # [B,T]
                    dist = reduce_token_scores(tok, mask, reduce=whole_reduce, topk=whole_topk)
                else:
                    raise ValueError(f"Unsupported distance type: {distance_type}")
            distances.append(dist)
        raw = torch.stack(distances, dim=1)  # [B,N]
        if distance_type in ["gaussian", "attention", "cosine"]:
            dis_weights = F.softmax(raw / float(temperature), dim=1)
        elif distance_type == "L2":
            dis_weights = F.softmax(-raw / float(temperature), dim=1)
        else:
            raise ValueError(f"Unsupported distance type: {distance_type}")
        return (dis_weights, raw)

def build_task_distributions_qv(
    cfg,
    model_base,
    dataset,  # task distribution용 데이터셋
    layer_idxs: List[int],
    batch_size: int = 8,
    tau_dataset=None, # threshold 계산용 데이터셋 
    retain_dataset=None,  # threshold 계산용 데이터셋 
):
    model_base.model.eval()
    dists: Dict[int, Dict[str, GaussianDistribution]] = {}
    token_pool = cfg.dist_cal_type
    tau_quantile = float(cfg.tau_quantile)
    for l in layer_idxs:
        layer_mod = model_base.model.layers[l] if hasattr(model_base.model, "layers") else model_base.model_block_modules[l]
        q_dim = layer_mod.self_attn.q_proj.out_features
        v_dim = layer_mod.self_attn.v_proj.out_features
        dists[l] = {"q": GaussianDistribution(dim=q_dim), "v": GaussianDistribution(dim=v_dim)}

    ###
    print(f"[BUILD_DIST] token_pool={token_pool}, layers={layer_idxs}")
    feats, handles = collect_qv_feats_with_hooks(
        cfg, 
        model_base=model_base,
        layer_idxs=layer_idxs,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    with torch.no_grad():
        for batch in loader:
            texts = batch["text_input"] # topics = batch["topic"]
            tok = model_base.tokenize_instructions_fn(instructions=texts)

            model_base._cur_attn_mask_2d = tok.attention_mask.to(model_base.model.device)
            _ = model_base.model(
                input_ids=tok.input_ids.to(model_base.model.device),
                attention_mask=model_base._cur_attn_mask_2d,
                use_cache=False,
            )
            for l in layer_idxs:
                if token_pool == "whole": # 유효 토큰 전체를 모아서 분포를 토큰 단위로 학습
                    q = feats[l]["q"]       # [B,T,D]
                    v = feats[l]["v"]       # [B,T,D]
                    m = feats[l]["mask"]    # [B,T]
                    valid = (m.reshape(-1) > 0.5)
                    q_feat = q.reshape(-1, q.size(-1))[valid].detach().float().cpu()
                    v_feat = v.reshape(-1, v.size(-1))[valid].detach().float().cpu()
                else: # 샘플당 하나의 pooled 벡터로 분포를 학습”
                    q_feat = feats[l]["q"].detach().float().cpu()  # [B,D]
                    v_feat = feats[l]["v"].detach().float().cpu()
                dists[l]["q"].update(q_feat)
                dists[l]["v"].update(v_feat)
    for h in handles:
        h.remove()
    handles.clear()

    if tau_dataset is None:
        raise AttributeError("Tau dataset (forget holdout) is required for tau calculation.")
    if retain_dataset is None:
        raise AttributeError("retain_dataset is required for retain similarity plotting.")
    
    print(f"[CAL_SIM] holdout & retain similarity, layers={layer_idxs}")
    s_hold_by_dom, S_hold_by_dom = compute_similarity_scores(
        cfg=cfg, model_base=model_base, layer_idxs=layer_idxs, dists=dists,
        dataset=tau_dataset, batch_size=batch_size
    )
    s_ret_by_dom, S_ret_by_dom = compute_similarity_scores(
        cfg=cfg, model_base=model_base, layer_idxs=layer_idxs, dists=dists,
        dataset=retain_dataset, batch_size=batch_size
    )

    hold_domains = set(s_hold_by_dom.keys()) # 도메인 리스트 (holdout/retain에 등장한 도메인 합집합)
    ret_domains  = set(s_ret_by_dom.keys())
    print(f"[DOMAINS] holdout({len(hold_domains)}): {sorted(hold_domains)}")
    print(f"[DOMAINS] retain ({len(ret_domains)}): {sorted(ret_domains)}")
    domains = sorted(hold_domains | ret_domains)

    tau_by_layer_hold = {} # tau_by_layer_hold[dom][l], tau_global_hold[dom]
    tau_global_hold = {}
    tau_by_layer_retain = {}
    tau_global_retain = {}
    q = float(tau_quantile) 

    for dom in domains:
        # holdout
        if dom in s_hold_by_dom:
            tau_by_layer_hold[dom] = {l: float(torch.quantile(s_hold_by_dom[dom][l], q).item()) for l in layer_idxs}
            tau_global_hold[dom] = float(torch.quantile(S_hold_by_dom[dom], q).item()) if S_hold_by_dom[dom].numel() > 0 else None
        else:
            tau_by_layer_hold[dom] = {l: None for l in layer_idxs}
            tau_global_hold[dom] = None

        # retain
        if dom in s_ret_by_dom:
            tau_by_layer_retain[dom] = {l: float(torch.quantile(s_ret_by_dom[dom][l], q).item()) for l in layer_idxs}
            tau_global_retain[dom] = float(torch.quantile(S_ret_by_dom[dom], q).item()) if S_ret_by_dom[dom].numel() > 0 else None
        else:
            tau_by_layer_retain[dom] = {l: None for l in layer_idxs}
            tau_global_retain[dom] = None

    plot_dir = os.path.join(str(getattr(cfg, "save_distribution_path", ".")), "sim_plots")
    os.makedirs(plot_dir, exist_ok=True)
    hold_domains_sorted = sorted(hold_domains)
    _plot_tau_by_layer(
        tau_by_layer_hold,
        title=f"Holdout tau_by_layer (q={q})",
        save_name=f"tau_by_layer_holdout_q{q}.png",
        domains_to_plot=hold_domains_sorted,
        agg_domains=hold_domains_sorted,
        agg_label="HOLDOUT-AGG(median)",
        agg="median",
        layer_idxs=layer_idxs,
        plot_dir=plot_dir,
    )
    ret_domains_sorted = sorted(ret_domains)
    _plot_tau_by_layer(
        tau_by_layer_retain,
        title=f"Retain tau_by_layer (q={q}) + Holdout-AGG",
        save_name=f"tau_by_layer_retain_q{q}_with_holdout_agg.png",
        domains_to_plot=ret_domains_sorted,
        agg_domains=hold_domains_sorted,          
        agg_label="HOLDOUT-AGG(median)",
        agg="median",
        layer_idxs=layer_idxs,
        plot_dir=plot_dir,
    )
    _plot_smax_all_in_one(
        S_hold_by_dom=S_hold_by_dom,
        S_ret_by_dom=S_ret_by_dom,
        hold_domains=hold_domains,
        ret_domains=ret_domains,
        plot_dir=plot_dir,
        save_name=f"Smax_all_domains_with_rep_q{q}.png",
        q=q,
        rep_line="holdout_tau",  # 대표선: holdout 전체 tau_global (quantile)
        bins=60,
    )
    _plot_smax_id_ood(
        S_hold_by_dom=S_hold_by_dom,
        S_ret_by_dom=S_ret_by_dom,
        hold_domains=hold_domains,
        ret_domains=ret_domains,
        plot_dir=plot_dir,
        save_name=f"Smax_ID_vs_OOD_q{q}.png",
        q=q,
        rep_line="holdout_tau",
        bins=80,
        show_per_domain=False,  # True로 하면 도메인별 faint overlay도 깔림
    )
    out_obj = { #  결과를 pkl로 같이 저장 (나중에 분석용)
        "quantile": q,
        "domains": domains,
        "layer_idxs": list(layer_idxs),
        "tau_by_layer_hold": tau_by_layer_hold,
        "tau_global_hold": tau_global_hold,
        "tau_by_layer_retain": tau_by_layer_retain,
        "tau_global_retain": tau_global_retain,
        # 분포 자체까지 저장하고 싶으면 아래도 가능(용량 커짐)
        # "S_hold_by_dom": {k: v.numpy() for k,v in S_hold_by_dom.items()},
        # "S_ret_by_dom":  {k: v.numpy() for k,v in S_ret_by_dom.items()},
    }
    with open(os.path.join(plot_dir, "domain_tau_summary.pkl"), "wb") as f:
        pickle.dump(out_obj, f)
    print(f"[SAVE] plots -> {plot_dir}")
    print(f"[SAVE] summary -> {os.path.join(plot_dir, 'domain_tau_summary.pkl')}")


def split_forget(cfg, dataset):
    n = len(dataset)
    n_hold = int(round(n * float(cfg.dist_holdout_ratio)))
    n_hold = max(1, min(n_hold, n - 1))
    n_train = n - n_hold
    g = torch.Generator().manual_seed(int(cfg.seed))
    train_ds, hold_ds = random_split(dataset, [n_train, n_hold], generator=g)
    return train_ds, hold_ds


def save_one_task_distributions_qv(
    cfg,
    model_base,
    dataset,
    layer_idxs: List[int], #
    save_dir: str, #
    task_id: str, #
    batch_size: int = 8,
    retain_dataset=None
):
    topic_id = tid_to_topic(task_id) 
    task_dir = os.path.join(save_dir, topic_id)
    os.makedirs(task_dir, exist_ok=True)

    # task_dir = os.path.join(save_dir, str(task_id))
    # os.makedirs(task_dir, exist_ok=True)
    train_ds, hold_ds = split_forget(cfg, dataset)

    dists, tau_by_layer, tau_debug = build_task_distributions_qv(
        cfg=cfg,
        model_base=model_base,
        dataset=train_ds,
        layer_idxs=layer_idxs,
        batch_size=batch_size,
        tau_dataset=hold_ds,
        retain_dataset=retain_dataset 
    )

    out: Dict[str, Any] = {}
    for l in layer_idxs:
        out[f"layers.{l}.task.{task_id}.q"] = dists[l]["q"].copy()
        out[f"layers.{l}.task.{task_id}.v"] = dists[l]["v"].copy()

    out["tau_by_layer"] = {int(k): float(v) for k, v in tau_by_layer.items()}
    out["tau_debug"] = tau_debug
    out["dist_token_pool"] = str(cfg.dist_cal_type) # 동일 방식 재현을 위해 반드시 저장
    out["dist_last_k"] = int(cfg.dist_last_k)
    out["task_id"] = str(task_id)

    if tau_by_layer is None:        
        raise ValueError(f"Invalid tau_by_layer")
    save_path = os.path.join(task_dir, "qv_distribution.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(out, f)
    return out


def load_task_distributions_qv(
    load_dir: str,
    task_id: str,
    device: Union[str, torch.device, None] = "cpu",
    dtype: Optional[torch.dtype] = None,
):
    pkl_path = os.path.join(load_dir, task_id, "qv_distribution.pkl")
    if not os.path.exists(pkl_path):
        alt = os.path.join(load_dir, "qv_distribution.pkl")
        if os.path.exists(alt):
            pkl_path = alt
        else:
            raise FileNotFoundError(f"Distribution file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    for k, v in list(obj.items()):
        if isinstance(v, GaussianDistribution):
            obj[k] = to_device_dtype(v, device=device, dtype=dtype)
    return obj


def load_many_task_distributions_qv(
    load_dir: str,
    task_ids: List[str],
    device: Union[str, torch.device, None] = "cpu",
    dtype: Optional[torch.dtype] = None,
):
    merged = {}
    meta_ref = None  # 첫 task의 meta를 기준으로 저장/검증
    tau_by_task = {}

    print("[LOAD] task ids:", task_ids, "from", load_dir)
    for tid in task_ids:
        d = load_task_distributions_qv(
            load_dir=load_dir, task_id=tid, device=device, dtype=dtype,
        )
        print("tau_by_layer type:", type(d.get("tau_by_layer")))
        print("tau_by_layer value:", d.get("tau_by_layer"))

        meta = {k: d.get(k, None) for k in (ALLOW_META | SHARED_META)}
        payload = {k: v for k, v in d.items() if k not in (ALLOW_META | SHARED_META)}

        tau_by_task[tid] = {
            "tau_by_layer": {int(k): float(v) for k, v in meta["tau_by_layer"].items()} if meta.get("tau_by_layer") is not None else (_ for _ in ()).throw(KeyError(f"[tau] missing tau_by_layer for task={tid}")),
            "tau_debug": meta.get("tau_debug", None),
        }
        if meta_ref is None:
            meta_ref = meta
            for k in SHARED_META:
                if meta_ref.get(k) is None:
                    raise KeyError(f"[META] missing shared '{k}' in first task={tid}")
                merged[k] = meta_ref[k]
        else:
            for k in SHARED_META:
                if meta_ref.get(k) != meta.get(k):
                    raise ValueError(f"[META MISMATCH] key={k} ref={meta_ref.get(k)} vs tid={tid} val={meta.get(k)}")

        overlap = set(merged.keys()) & set(payload.keys())
        if overlap:
            raise KeyError(f"Overlapping keys detected (examples): {list(overlap)[:3]}")
        merged.update(payload)
    merged["tau_by_task"] = tau_by_task 
    return merged


def align_distributions(
    dist_dict: dict,
    layer_idxs: Optional[List[int]],
    which: Union[str, Tuple[str, ...]] = ("q", "v"),
):
    if isinstance(which, str):
        which = (which,)
    for w in which:
        if w not in ("q", "v"):
            raise ValueError(f"`which` must be 'q' or 'v' (or tuple), got: {w}")

    layer_set = set(layer_idxs) if layer_idxs is not None else None
    t_pat = re.compile(r"_t(\d+)_")
    def tid_sort_key(tid: str):
        m = t_pat.search(tid)
        t = int(m.group(1)) if m else 10**9
        return (t, tid)
    def parse_key(k: str):
        parts = k.split(".")
        if len(parts) < 5 or parts[0] != "layers" or parts[2] != "task":
            return None
        try:
            layer = int(parts[1])
        except Exception:
            return None
        tid = parts[3]
        head = parts[4]
        return tid, layer, head

    out = {}
    meta_ref = None
    for w in which:
        items = []
        for k, dist in dist_dict.items():
            if not isinstance(dist, GaussianDistribution):
                continue
            parsed = parse_key(k)
            if parsed is None:
                continue
            tid, layer, head = parsed
            if head != w:
                continue
            if layer_set is not None and layer not in layer_set:
                continue
            items.append((tid, layer, dist))
        items.sort(key=lambda x: (tid_sort_key(x[0]), x[1]))
        out[w] = [d for (_, _, d) in items]
        meta = [(tid, layer) for (tid, layer, _) in items]
        if meta_ref is None:
            meta_ref = meta
        elif meta != meta_ref:
            raise ValueError(f"meta mismatch between heads. which={w}")
    return out, meta_ref


def compute_gate_vec_from_hidden(
    cfg,
    q_feats: Dict[int, Dict[str, torch.Tensor]],   # layer -> {"q":..., "v":..., ("mask":...)}
    v_feats: Dict[int, Dict[str, torch.Tensor]],   # layer -> {"q":..., "v":..., ("mask":...)}
    qv_dists: Dict[str, List[GaussianDistribution]],  # {"q":[...], "v":[...]}
    meta_dists: List[Tuple[str, int]],                
    topics: Optional[List[str]] = None, 
    tau_by_layer: Optional[Dict[str, Dict[str, Any]]] = None,
):
    if tau_by_layer is None:
        # raise ValueError("tau_by_layer is None. Did you forget to load/pass tau_by_layer from qv_distribution.pkl?")
        print("tau_by_layer is None.")

    def _build_index(meta_dists):
        task_ids = []
        layer_to_tid_cols = {}
        for j, (tid, layer) in enumerate(meta_dists):
            if tid not in task_ids:
                task_ids.append(tid)
            layer_to_tid_cols.setdefault(int(layer), {}).setdefault(tid, []).append(j)
        return task_ids, layer_to_tid_cols

    ### START
    task_ids, layer_to_tid_cols = _build_index(meta_dists)
    layers_sorted = sorted(int(l) for l in layer_to_tid_cols.keys())
    used_layers = [l for l in layers_sorted if (l in q_feats and l in v_feats)]
    if not used_layers:
        raise ValueError("No layers available for gating.")

    # 1) 레이어별 task_phi / alpha / best_phi를 텐서로 바로 쌓기 위한 리스트
    task_phi_list = []   # each: [B, num_tasks]
    alpha_list    = []   # each: [B, num_tasks]
    best_phi_list = []   # each: [B]

    with torch.no_grad():
        for layer in used_layers:
            tid_cols = layer_to_tid_cols[layer]
            if cfg.dist_cal_type == "whole":
                q_pack = {"x": q_feats[layer]["q"], "mask": q_feats[layer]["mask"]}
                v_pack = {"x": v_feats[layer]["v"], "mask": v_feats[layer]["mask"]}
            else:
                q_pack = q_feats[layer]["q"]   # [B,D]
                v_pack = v_feats[layer]["v"]   # [B,D]

            _, raw_q = calculate_distances_unified(cfg, features=q_pack, distributions=qv_dists["q"])
            _, raw_v = calculate_distances_unified(cfg, features=v_pack, distributions=qv_dists["v"])
            phi = 0.5 * (raw_q + raw_v)  # [B, N_slots]

            per_task = []
            for tid in task_ids:
                cols = tid_cols.get(tid)
                if not cols:
                    raise ValueError(f"No columns for task id {tid} at layer {layer}")
                per_task.append(phi[:, cols].max(dim=1).values) # task별 max over columns -> [B, num_tasks]
            task_phi = torch.stack(per_task, dim=1)   # [B, num_tasks]

            temp = float(getattr(cfg, "temperature", 1.0))
            alpha = F.softmax(task_phi / temp, dim=1) # [B, num_tasks]
            best_phi = task_phi.max(dim=1).values     # [B]

            task_phi_list.append(task_phi)
            alpha_list.append(alpha)
            best_phi_list.append(best_phi)

        # 2) 레이어 축 stack: [L,B,*]
        task_phi_stack = torch.stack(task_phi_list, dim=0)  # [L,B,num_tasks]
        alpha_stack    = torch.stack(alpha_list,    dim=0)  # [L,B,num_tasks]
        best_phi_stack = torch.stack(best_phi_list, dim=0)  # [L,B]
        B = best_phi_stack.size(1)

        # 3) best layer 선택: [B]
        best_layer_pos = best_phi_stack.argmax(dim=0)  # [B]
        tau_vec = torch.tensor([float(tau_by_layer[l]) for l in used_layers],
                            device=best_phi_stack.device, dtype=best_phi_stack.dtype)  # [L]

        pass_mat = best_phi_stack >= tau_vec[:, None]    # [L,B]
        num_pass = pass_mat.sum(dim=0)                   # [B]

        # 4) gate_min_layers 기준으로 ID/OOD 결정 
        gate_min_layers = int(getattr(cfg, "gate_min_layers", 1))
        gate_vec = (num_pass >= gate_min_layers).to(best_phi_stack.dtype)  # [B]

        # 5) best layer의 alpha/phi를 gather로 뽑기 (깔끔)
        alpha_best = alpha_stack.gather(
            dim=0,
            index=best_layer_pos.view(1, B, 1).expand(1, B, alpha_stack.size(-1))
        ).squeeze(0)

        task_phi_best = task_phi_stack.gather(
            dim=0,
            index=best_layer_pos.view(1, B, 1).expand(1, B, task_phi_stack.size(-1))
        ).squeeze(0)

        # 6) OOD는 LoRA 선택 0 처리
        alpha_best = alpha_best * gate_vec[:, None]
    
    pred_idx = task_phi_best.argmax(dim=1)  # [B]
    pred_tid = [task_ids[i] for i in pred_idx.detach().cpu().tolist()]

    # best layer에서의 best_phi (스칼라)
    best_phi = best_phi_stack.gather(
        dim=0, index=best_layer_pos.view(1, B)
    ).squeeze(0)  # [B]

    return {
        "gate_vec": gate_vec,                 # [B]
        "alpha_best": alpha_best,             # [B, num_tasks]
        "task_phi_best": task_phi_best,       # [B, num_tasks]
        "pred_tid": pred_tid,                 # list[str], len=B
        "best_layer_pos": best_layer_pos,     # [B]
        "best_phi": best_phi,                 # [B]
        "num_pass": num_pass,                 # [B]
    }
