import os, json, copy, random, time, socket, platform, torch, hydra 
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader, random_split
from src_lunar.data_loader import convert_raw_data_to_model_qa, custom_qa_collator
from src_lunar.eval_util import get_dataloader
from src_lunar.model_utils.model_loader import load_model
from src_lunar.dataset_utils import (
    split_raw_dataset_for_forget,
)
from src_lunar.gaussian_distribution.distribution_utils import collect_qv_feats_with_hooks

# =========================================================
# 0) Seed
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# =========================================================
# 0.1) Small utils
# =========================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def safe_json_dump(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_jsonl(path: str, obj: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def to_py(x: Any) -> Any:
    """Convert numpy/torch scalars to python types for JSON."""
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return x.item()
    return x

def cfg_to_dict(cfg) -> Dict[str, Any]:
    """Hydra cfg -> plain dict safely (best-effort)."""
    try:
        # OmegaConf is typically available with Hydra; we keep it best-effort
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        # fallback: shallow object __dict__
        d = {}
        for k in dir(cfg):
            if k.startswith("_"):
                continue
            try:
                v = getattr(cfg, k)
                if callable(v):
                    continue
                d[k] = v
            except Exception:
                pass
        return d

def runtime_meta(seed: int, device: str) -> Dict[str, Any]:
    m = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "seed": int(seed),
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        try:
            m["cuda_device_count"] = int(torch.cuda.device_count())
            cur = torch.cuda.current_device()
            m["cuda_current_device"] = int(cur)
            m["cuda_device_name"] = torch.cuda.get_device_name(cur)
        except Exception:
            pass
    return m

def to_batch_dict(batch, device="cuda") -> Dict[str, torch.Tensor]:
    if isinstance(batch, dict):
        bd = dict(batch)

        # normalize label key
        if "labels" not in bd and "label" in bd:
            bd["labels"] = bd.pop("label")

        # drop non-forward keys
        for k in ["topic", "real_answer", "answer", "question"]:
            if k in bd:
                bd.pop(k)

    else:
        if len(batch) >= 3:
            input_ids = batch[0]
            labels = batch[1]
            attention_mask = batch[2]
            bd = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        else:
            raise ValueError(f"Unsupported batch format. got len={len(batch)}")

    for k in list(bd.keys()):
        if isinstance(bd[k], torch.Tensor):
            bd[k] = bd[k].to(device)
    return bd

from torch.utils.data import Dataset
class QATextInputDataset(torch.utils.data.Dataset):
    """
    item = {"text_input": str, "topic": str?} 형태를
    convert_raw_data_to_model_qa()에 넣어서 (input_ids, labels, attention_mask)로 변환
    """
    def __init__(self, data, tokenizer, cfg, max_length=512, text_key="text_input"):
        self.data = data
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.max_length = max_length
        self.text_key = text_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.text_key not in item:
            raise KeyError(f"Expected key '{self.text_key}', got keys={list(item.keys())}")

        # text_input을 instruction으로 보고, answer는 빈 문자열로 넣거나
        # convert_raw_data_to_model_qa가 instruction-only 모드를 지원하면 그걸 쓰기
        instruction = item[self.text_key]
        answer = ""  # 또는 cfg에 맞는 dummy

        input_ids, labels, attention_mask = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length, instruction, answer, self.cfg
        )
        return {
            "input_ids": input_ids,
            "label": labels,
            "attention_mask": attention_mask,
            "topic": item.get("topic", "")
        }

def build_loader_from_raw_dataset(
    cfg,
    model_base,
    raw_dataset,
    shuffle: bool,
    is_eval: bool = False,
    eval_target: str = None,  
):
    # -------------------------
    # (A) eval: path 기반이면 get_dataloader로 통일
    # -------------------------
    if is_eval:
        assert isinstance(raw_dataset, str), "eval loader expects a path string"
        assert eval_target is not None, "eval loader requires eval_target (forget_edge/retained_edge/...)"
        return get_dataloader(cfg=cfg, data_path=raw_dataset, tokenizer=model_base.tokenizer, eval_target=eval_target)

    # -------------------------
    # (B) mu: raw list/subset이면 별도 Dataset (text_input 기반)
    # -------------------------
    ds = QATextInputDataset(
        data=raw_dataset,  # list or Subset
        tokenizer=model_base.tokenizer,
        cfg=cfg,
        max_length=512,
        text_key="text_input",
    )
    loader = DataLoader(
        ds,
        batch_size=int(getattr(cfg, "dist_batch_size", 8)),
        shuffle=shuffle,
        num_workers=int(getattr(cfg, "num_workers", 0)),
        drop_last=False,
        collate_fn=custom_qa_collator,
    )
    return loader

# =========================================================
# 2) Hook-based feature extraction
# =========================================================
def extract_qv_feats(
    cfg,
    model_base,
    batch_dict: Dict[str, torch.Tensor],
    layer_idxs: List[int],
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, Dict[str, torch.Tensor]]]:
    feats_dict, handles = collect_qv_feats_with_hooks(
        cfg=cfg,
        model_base=model_base,
        layer_idxs=layer_idxs,
    )
    try:
        if "attention_mask" not in batch_dict:
            raise ValueError("batch_dict must contain 'attention_mask' for _cur_attn_mask_2d")
        model_base._cur_attn_mask_2d = batch_dict["attention_mask"]

        _ = model_base.model(
            **batch_dict,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
    finally:
        for h in handles:
            h.remove()

    q_feats_by_layer = {}
    v_feats_by_layer = {}
    for L in layer_idxs:
        if L not in feats_dict:
            raise ValueError(f"Layer {L} missing in feats_dict. Check hooks/layer_idxs.")
        q_feats_by_layer[L] = {"q": feats_dict[L]["q"], "mask": feats_dict[L].get("mask", None)}
        v_feats_by_layer[L] = {"v": feats_dict[L]["v"], "mask": feats_dict[L].get("mask", None)}
    return q_feats_by_layer, v_feats_by_layer


# =========================================================
# 3) Similarity helpers
# =========================================================
def safe_norm(x: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def cos_sim_batch_mean(x: torch.Tensor, mu: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x: [B,D], mu: [D] or [1,D] -> [B]
    """
    if mu.dim() == 1:
        mu = mu.unsqueeze(0)
    x_n = F.normalize(x, dim=1, eps=eps)
    mu_n = F.normalize(mu, dim=1, eps=eps)
    return (x_n * mu_n).sum(dim=1)

def reduce_token_scores(
    scores: torch.Tensor,          # [B,T]
    mask: torch.Tensor,            # [B,T] float {0,1}
    reduce: str = "mean",          # "mean" | "max" | "topk_mean"
    topk: int = 8,
) -> torch.Tensor:
    if reduce == "mean":
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (scores * mask).sum(dim=1) / denom
    if reduce == "max":
        neg_inf = torch.finfo(scores.dtype).min
        masked = scores.masked_fill(mask <= 0.5, neg_inf)
        return masked.max(dim=1).values
    if reduce == "topk_mean":
        K = int(topk)
        neg_inf = torch.finfo(scores.dtype).min
        masked = scores.masked_fill(mask <= 0.5, neg_inf)
        vals, _ = masked.topk(k=min(K, masked.size(1)), dim=1)
        valid_cnt = (mask > 0.5).sum(dim=1).clamp_min(1)
        bad = valid_cnt < vals.size(1)
        out = vals.mean(dim=1)
        if bad.any():
            out[bad] = reduce_token_scores(scores[bad], mask[bad], reduce="mean")
        return out
    raise ValueError(f"Unsupported reduce: {reduce}")

def cosine_tokenwise_reduce(
    x_tok: torch.Tensor,
    mu: torch.Tensor,
    mask: torch.Tensor,
    reduce: str,
    topk: int
) -> torch.Tensor:
    """
    x_tok: [B,T,D], mu: [D], mask: [B,T] float{0,1} -> [B]
    """
    x_n = safe_norm(x_tok, dim=-1, eps=1e-8)                 # [B,T,D]
    mu_n = safe_norm(mu, dim=0, eps=1e-8).view(1, 1, -1)     # [1,1,D]
    tok = (x_n * mu_n).sum(dim=-1)                           # [B,T]
    return reduce_token_scores(tok, mask, reduce=reduce, topk=topk)

def masked_global_mean(x_tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x_tok: [B,T,D], mask:[B,T] -> mu:[D]
    """
    m = mask.unsqueeze(-1).to(x_tok.dtype)  # [B,T,1]
    denom = m.sum(dim=(0, 1)).clamp_min(1.0)
    s = (x_tok * m).sum(dim=(0, 1))         # [D]
    return s / denom


# =========================================================
# 4) Metrics: AUROC / KS / Cohen's d
# =========================================================
def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = np.sort(a)
    b = np.sort(b)
    allv = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, allv, side="right") / max(len(a), 1)
    cdf_b = np.searchsorted(b, allv, side="right") / max(len(b), 1)
    return float(np.max(np.abs(cdf_a - cdf_b)))

def _average_ranks(x: np.ndarray) -> np.ndarray:
    """
    평균 rank (ties 평균 처리). 1..N
    """
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)

    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks

def auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """
    Mann-Whitney U 기반 AUROC (pos=forget, neg=retain).
    """
    scores_pos = np.asarray(scores_pos, dtype=np.float64)
    scores_neg = np.asarray(scores_neg, dtype=np.float64)
    n1 = len(scores_pos)
    n0 = len(scores_neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    x = np.concatenate([scores_pos, scores_neg], axis=0)
    r = _average_ranks(x)
    r1 = r[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * n0))

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max((len(a) + len(b) - 2), 1))
    if sp < 1e-12:
        return float("nan")
    return float((ma - mb) / sp)


# =========================================================
# 6) Distribution summaries / snapshots
# =========================================================
def summarize_scores(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {k: float("nan") for k in ["mean", "std", "median", "mad", "q05", "q25", "q75", "q95", "min", "max"]}
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=0)),
        "median": median,
        "mad": mad,
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }

def hist_snapshot(x: np.ndarray, bins: int = 80, range_: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"bins": bins, "edges": [], "counts": []}
    if range_ is None:
        lo = float(np.quantile(x, 0.001))
        hi = float(np.quantile(x, 0.999))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(x.min()), float(x.max())
            if lo == hi:
                lo, hi = lo - 1e-6, hi + 1e-6
        range_ = (lo, hi)
    counts, edges = np.histogram(x, bins=bins, range=range_)
    return {
        "bins": int(bins),
        "range": [float(range_[0]), float(range_[1])],
        "edges": [float(v) for v in edges.tolist()],
        "counts": [int(v) for v in counts.tolist()],
    }

def save_mu_vectors(task_dir: str, mu_by_layer: Dict[int, Dict[str, torch.Tensor]], modules: List[str]):
    """
    Save mu vectors as float16 npy per layer/module for best config.
    """
    mu_dir = os.path.join(task_dir, "mu")
    ensure_dir(mu_dir)
    for l, md in mu_by_layer.items():
        for m in modules:
            if m not in md:
                continue
            arr = md[m].detach().float().cpu().numpy().astype(np.float16)
            np.save(os.path.join(mu_dir, f"mu_layer{int(l):02d}_{m}.npy"), arr)


# =========================================================
# 7) Core: compute mu + similarity distributions
# =========================================================
@dataclass
class SimConfig:
    token_pool: str              # "mean" | "last" | "last_k" | "whole"
    module: str                  # "q" | "v" | "qv"
    whole_reduce: str = "mean"   # for token_pool="whole"
    whole_topk: int = 8
    last_k: int = 8

def compute_mu_from_forget(
    cfg,
    model_base,
    loader_forget_mu,
    layers: List[int],
    simcfg: SimConfig,
    device: str,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    returns mu_by_layer[l] = {"q": [Dq], "v": [Dv], "qv":[Dq+Dv]}
    - token_pool != "whole": mu는 pooled feature의 평균
    - token_pool == "whole": mu는 (mask로 가중한 전체 토큰 평균) [D]
    """
    mu_by_layer: Dict[int, Dict[str, torch.Tensor]] = {l: {} for l in layers}

    sums_q = {l: None for l in layers}
    sums_v = {l: None for l in layers}
    cnt_q  = {l: 0.0 for l in layers}
    cnt_v  = {l: 0.0 for l in layers}

    model_base.model.eval()
    with torch.no_grad():
        for batch in loader_forget_mu:
            bd = to_batch_dict(batch, device=device)
            q_feats, v_feats = extract_qv_feats(cfg, model_base, bd, layers)

            for l in layers:
                q = q_feats[l]["q"].float()
                v = v_feats[l]["v"].float()

                if simcfg.token_pool == "whole":
                    m = q_feats[l]["mask"]
                    if m is None:
                        raise ValueError("token_pool='whole' requires mask in feats.")
                    m = m.float()

                    mu_q = masked_global_mean(q, m)
                    mu_v = masked_global_mean(v, m)

                    if sums_q[l] is None:
                        sums_q[l] = mu_q.clone()
                        sums_v[l] = mu_v.clone()
                    else:
                        sums_q[l] += mu_q
                        sums_v[l] += mu_v
                    cnt_q[l] += 1.0
                    cnt_v[l] += 1.0
                else:
                    if q.dim() != 2 or v.dim() != 2:
                        raise ValueError(f"pooled mode expects [B,D], got q={tuple(q.shape)} v={tuple(v.shape)}")
                    if sums_q[l] is None:
                        sums_q[l] = q.sum(dim=0)
                        sums_v[l] = v.sum(dim=0)
                    else:
                        sums_q[l] += q.sum(dim=0)
                        sums_v[l] += v.sum(dim=0)
                    cnt_q[l] += float(q.size(0))
                    cnt_v[l] += float(v.size(0))

    for l in layers:
        mu_q = (sums_q[l] / max(cnt_q[l], 1.0)).detach()
        mu_v = (sums_v[l] / max(cnt_v[l], 1.0)).detach()
        mu_by_layer[l]["q"] = mu_q
        mu_by_layer[l]["v"] = mu_v
        mu_by_layer[l]["qv"] = torch.cat([mu_q, mu_v], dim=0)
    return mu_by_layer

def compute_similarity_scores(
    cfg,
    model_base,
    loader,
    layers: List[int],
    simcfg: SimConfig,
    mu_by_layer: Dict[int, Dict[str, torch.Tensor]],
    device: str,
) -> Dict[int, np.ndarray]:
    """
    returns scores_by_layer[l] = np.ndarray[N]
    score = cosine similarity (higher means closer)
    """
    scores_acc: Dict[int, List[float]] = {l: [] for l in layers}

    model_base.model.eval()
    with torch.no_grad():
        for batch in loader:
            bd = to_batch_dict(batch, device=device)
            q_feats, v_feats = extract_qv_feats(cfg, model_base, bd, layers)

            for l in layers:
                mu = mu_by_layer[l][simcfg.module].to(device=device, dtype=torch.float32)

                if simcfg.token_pool == "whole":
                    q = q_feats[l]["q"].float()  # [B,T,Dq]
                    v = v_feats[l]["v"].float()
                    m = q_feats[l]["mask"]
                    if m is None:
                        raise ValueError("token_pool='whole' requires mask in feats.")
                    m = m.float()

                    if simcfg.module == "q":
                        s = cosine_tokenwise_reduce(q, mu, m, reduce=simcfg.whole_reduce, topk=simcfg.whole_topk)
                    elif simcfg.module == "v":
                        s = cosine_tokenwise_reduce(v, mu, m, reduce=simcfg.whole_reduce, topk=simcfg.whole_topk)
                    elif simcfg.module == "qv":
                        x = torch.cat([q, v], dim=-1)  # [B,T,Dq+Dv]
                        s = cosine_tokenwise_reduce(x, mu, m, reduce=simcfg.whole_reduce, topk=simcfg.whole_topk)
                    else:
                        raise ValueError(f"bad module: {simcfg.module}")

                else:
                    q = q_feats[l]["q"].float()   # [B,Dq]
                    v = v_feats[l]["v"].float()   # [B,Dv]
                    if simcfg.module == "q":
                        x = q
                    elif simcfg.module == "v":
                        x = v
                    elif simcfg.module == "qv":
                        x = torch.cat([q, v], dim=-1)
                    else:
                        raise ValueError(f"bad module: {simcfg.module}")

                    s = cos_sim_batch_mean(x, mu)  # [B]

                scores_acc[l].extend(s.detach().cpu().tolist())

    return {l: np.asarray(scores_acc[l], dtype=np.float64) for l in layers}


# =========================================================
# 8) Grid search: find max separation between forget/retain
# =========================================================
# def analyze_forget_retain_separation(
#     cfg,
#     model_base,
#     forget_mu_loader,
#     forget_eval_loader,
#     retain_eval_loader,
#     layers: List[int],
#     device: str,
#     token_pools: List[str],
#     modules: List[str],
#     whole_reduces: List[str],
#     whole_topks: List[int],
#     last_ks: List[int],
#     topk_print: int = 20,
#     out_json_path: Optional[str] = None,
#     out_jsonl_path: Optional[str] = None,
#     meta_row_base: Optional[Dict[str, Any]] = None,
#     save_best_score_snapshot: bool = True,
#     task_dir_for_snapshots: Optional[str] = None,
# ) -> List[Dict[str, Any]]:
#     """
#     Returns results_sorted (list of dict).

#     Extra outputs:
#       - out_json_path: full sorted JSON
#       - out_jsonl_path: JSONL (all rows)
#       - If save_best_score_snapshot: save histogram snapshot for best row (forget/retain) per layer (best layer only)
#     """
#     results: List[Dict[str, Any]] = []

#     base_cfg = cfg
#     tau_q = float(getattr(cfg, "tau_quantile", 0.90))  # used to propose tau on forget

#     for token_pool in token_pools:
#         for module in modules:
#             if token_pool != "whole":
#                 ks = last_ks if token_pool == "last_k" else [int(getattr(cfg, "dist_last_k", 8))]
#                 reduce_list = ["mean"]
#                 topk_list = [int(getattr(cfg, "whole_topk", 8))]
#             else:
#                 ks = [int(getattr(cfg, "dist_last_k", 8))]
#                 reduce_list = whole_reduces
#                 topk_list = whole_topks

#             for last_k in ks:
#                 for whole_reduce in reduce_list:
#                     for whole_topk in topk_list:
#                         cfg_local = copy.deepcopy(base_cfg)
#                         cfg_local.dist_cal_type = token_pool
#                         cfg_local.dist_last_k = int(last_k)
#                         cfg_local.whole_reduce = str(whole_reduce)
#                         cfg_local.whole_topk = int(whole_topk)

#                         simcfg = SimConfig(
#                             token_pool=token_pool,
#                             module=module,
#                             whole_reduce=whole_reduce,
#                             whole_topk=whole_topk,
#                             last_k=last_k,
#                         )

#                         # 1) mu (forget 기반)
#                         mu_by_layer = compute_mu_from_forget(
#                             cfg_local, model_base, forget_mu_loader, layers, simcfg, device
#                         )

#                         # 2) scores distributions
#                         s_forget = compute_similarity_scores(
#                             cfg_local, model_base, forget_eval_loader, layers, simcfg, mu_by_layer, device
#                         )
#                         s_retain = compute_similarity_scores(
#                             cfg_local, model_base, retain_eval_loader, layers, simcfg, mu_by_layer, device
#                         )

#                         # 3) per-layer separation metrics + rich stats
#                         for l in layers:
#                             a = s_forget[l]
#                             b = s_retain[l]

#                             ma, mb = float(a.mean()) if len(a) else float("nan"), float(b.mean()) if len(b) else float("nan")
#                             gap = ma - mb if np.isfinite(ma) and np.isfinite(mb) else float("nan")

#                             d = cohens_d(a, b)
#                             aucv = auroc(a, b)   # pos=forget, neg=retain
#                             ksv = ks_statistic(a, b)

#                             a_stats = summarize_scores(a)
#                             b_stats = summarize_scores(b)

#                             # tau proposal from forget + retain intrusion proxy
#                             tau_val = float(np.quantile(a, tau_q)) if len(a) else float("nan")
#                             # proxy: fraction of retain above forget's tau (like FPR at tau)
#                             fpr_proxy = float(np.mean(b >= tau_val)) if len(b) and np.isfinite(tau_val) else float("nan")

#                             row = {
#                                 "layer": int(l),
#                                 "token_pool": token_pool,
#                                 "module": module,
#                                 "last_k": int(last_k),
#                                 "whole_reduce": whole_reduce if token_pool == "whole" else None,
#                                 "whole_topk": int(whole_topk) if token_pool == "whole" else None,

#                                 "mean_forget": ma,
#                                 "mean_retain": mb,
#                                 "mean_gap": gap,
#                                 "abs_mean_gap": abs(gap) if np.isfinite(gap) else float("nan"),

#                                 "cohens_d": d,
#                                 "abs_cohens_d": abs(d) if not np.isnan(d) else float("nan"),
#                                 "auroc_forget_vs_retain": aucv,
#                                 "ks": ksv,

#                                 "n_forget": int(len(a)),
#                                 "n_retain": int(len(b)),

#                                 "forget_stats": a_stats,
#                                 "retain_stats": b_stats,

#                                 "tau_quantile": tau_q,
#                                 "tau_from_forget": tau_val,
#                                 "retain_fpr_proxy_at_tau": fpr_proxy,
#                             }

#                             if meta_row_base:
#                                 row.update(meta_row_base)

#                             results.append(row)

#                             if out_jsonl_path is not None:
#                                 append_jsonl(out_jsonl_path, row)

#     # 정렬: abs_cohens_d > auroc > ks > abs_mean_gap
#     def key_fn(r):
#         ad = r.get("abs_cohens_d", float("nan"))
#         if ad != ad:  # nan
#             ad = -1e9
#         aucv = r.get("auroc_forget_vs_retain", float("nan"))
#         if aucv != aucv:
#             aucv = -1e9
#         ksv = r.get("ks", float("nan"))
#         if ksv != ksv:
#             ksv = -1e9
#         amg = r.get("abs_mean_gap", float("nan"))
#         if amg != amg:
#             amg = -1e9
#         return (ad, aucv, ksv, amg)

#     results_sorted = sorted(results, key=key_fn, reverse=True)

#     print("\n==================== TOP SEPARATION (forget vs retain) ====================")
#     for i, r in enumerate(results_sorted[:topk_print]):
#         print(
#             f"[{i:02d}] layer={r['layer']:2d} | pool={r['token_pool']:6s} | mod={r['module']:2s} "
#             f"| d={r['cohens_d']:+.3f} | auc={r['auroc_forget_vs_retain']:.3f} | ks={r['ks']:.3f} "
#             f"| gap={r['mean_gap']:+.3f} (f={r['mean_forget']:.3f}, r={r['mean_retain']:.3f}) "
#             f"| tau@{r['tau_quantile']:.2f}={r['tau_from_forget']:.3f} fpr~{r['retain_fpr_proxy_at_tau']:.3f} "
#             f"| last_k={r['last_k']} "
#             f"{'| reduce='+str(r['whole_reduce'])+' topk='+str(r['whole_topk']) if r['token_pool']=='whole' else ''}"
#         )

#     if out_json_path is not None:
#         safe_json_dump(out_json_path, {"results": results_sorted})
#         print(f"\n[SAVED] {out_json_path}")

#     # Save snapshot for best row (best layer only) if requested.
#     if save_best_score_snapshot and task_dir_for_snapshots and len(results_sorted) > 0:
#         best = results_sorted[0]
#         snap_dir = os.path.join(task_dir_for_snapshots, "scores")
#         ensure_dir(snap_dir)

#         # Recompute best config scores once to snapshot (avoid storing for all rows)
#         cfg_best = copy.deepcopy(cfg)
#         cfg_best.dist_cal_type = best["token_pool"]
#         cfg_best.dist_last_k = int(best["last_k"])
#         if best["token_pool"] == "whole":
#             cfg_best.whole_reduce = best["whole_reduce"]
#             cfg_best.whole_topk = int(best["whole_topk"])
#         simcfg_best = SimConfig(
#             token_pool=best["token_pool"],
#             module=best["module"],
#             whole_reduce=best["whole_reduce"] if best["token_pool"] == "whole" else "mean",
#             whole_topk=int(best["whole_topk"]) if best["token_pool"] == "whole" else int(getattr(cfg, "whole_topk", 8)),
#             last_k=int(best["last_k"]),
#         )

#         mu_best = compute_mu_from_forget(cfg_best, model_base, forget_mu_loader, layers, simcfg_best, device)
#         sf_best = compute_similarity_scores(cfg_best, model_base, forget_eval_loader, layers, simcfg_best, mu_best, device)
#         sr_best = compute_similarity_scores(cfg_best, model_base, retain_eval_loader, layers, simcfg_best, mu_best, device)

#         L = int(best["layer"])
#         a = sf_best[L]
#         b = sr_best[L]

#         rng = None
#         if len(a) and len(b):
#             lo = float(min(np.quantile(a, 0.001), np.quantile(b, 0.001)))
#             hi = float(max(np.quantile(a, 0.999), np.quantile(b, 0.999)))
#             if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
#                 rng = (lo, hi)

#         snapshot = {
#             "best_setting": best,
#             "forget_stats": summarize_scores(a),
#             "retain_stats": summarize_scores(b),
#             "forget_hist": hist_snapshot(a, bins=int(getattr(cfg, "score_hist_bins", 80)), range_=rng),
#             "retain_hist": hist_snapshot(b, bins=int(getattr(cfg, "score_hist_bins", 80)), range_=rng),
#         }
#         safe_json_dump(os.path.join(snap_dir, "best_layer_hist.json"), snapshot)

#         # Optionally save mu vectors for best config
#         if bool(getattr(cfg, "save_mu_vectors", True)):
#             save_mu_vectors(task_dir_for_snapshots, mu_best, modules=["q", "v", "qv"])

#     return results_sorted
def analyze_forget_retain_separation(
    cfg,
    model_base,
    forget_mu_loader,
    forget_eval_loader,
    retain_eval_loader,
    layers: List[int],
    device: str,
    token_pools: List[str],
    modules: List[str],
    whole_reduces: List[str],
    whole_topks: List[int],
    last_ks: List[int],
    topk_print: int = 20,
    out_json_path: Optional[str] = None,
    out_jsonl_path: Optional[str] = None,
    meta_row_base: Optional[Dict[str, Any]] = None,
    save_best_score_snapshot: bool = True,
    task_dir_for_snapshots: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Grid-search token_pool/module/(last_k)/(whole_reduce/topk) and compute separation
    between forget vs retain similarity distributions per layer.

    Outputs:
      - results_sorted: all rows (one row per (setting, layer))
      - summary_by_setting.json: best layer per setting
      - summary_topK.json: topK settings (by best layer metric)
      - best-layer histogram snapshot: scores/best_layer_hist.json (best row only)

    Notes:
      - Uses your existing functions:
        compute_mu_from_forget, compute_similarity_scores, cohens_d, auroc, ks_statistic,
        summarize_scores, append_jsonl, safe_json_dump, ensure_dir, hist_snapshot, save_mu_vectors
    """

    # -----------------------------
    # local helpers (self-contained)
    # -----------------------------
    def setting_key(token_pool: str, module: str, last_k: int, whole_reduce=None, whole_topk=None) -> str:
        if token_pool == "whole":
            return f"pool={token_pool}|mod={module}|k={last_k}|reduce={whole_reduce}|topk={whole_topk}"
        return f"pool={token_pool}|mod={module}|k={last_k}"

    def fmt_row_short(r: Dict[str, Any]) -> str:
        return (
            f"layer={r['layer']:2d} d={r['cohens_d']:+.3f} auc={r['auroc_forget_vs_retain']:.3f} "
            f"ks={r['ks']:.3f} gap={r['mean_gap']:+.3f} "
            f"tau@{r['tau_quantile']:.2f}={r['tau_from_forget']:.3f} fpr~{r['retain_fpr_proxy_at_tau']:.3f}"
        )

    def key_fn(r: Dict[str, Any]):
        # sort priority: abs_cohens_d > auroc > ks > abs_mean_gap
        ad = r.get("abs_cohens_d", float("nan"))
        if ad != ad:
            ad = -1e9
        aucv = r.get("auroc_forget_vs_retain", float("nan"))
        if aucv != aucv:
            aucv = -1e9
        ksv = r.get("ks", float("nan"))
        if ksv != ksv:
            ksv = -1e9
        amg = r.get("abs_mean_gap", float("nan"))
        if amg != amg:
            amg = -1e9
        return (ad, aucv, ksv, amg)

    def best_row_among_layers(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return sorted(rows, key=key_fn, reverse=True)[0]

    def count_total_settings(token_pools, modules, whole_reduces, whole_topks, last_ks, cfg) -> int:
        total = 0
        for token_pool in token_pools:
            for _module in modules:
                if token_pool != "whole":
                    ks = last_ks if token_pool == "last_k" else [int(getattr(cfg, "dist_last_k", 8))]
                    total += len(ks) * 1 * 1
                else:
                    total += 1 * len(whole_reduces) * len(whole_topks)
        return total

    # -----------------------------
    # run
    # -----------------------------
    results: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    global_best: Optional[Dict[str, Any]] = None

    base_cfg = cfg
    tau_q = float(getattr(cfg, "tau_quantile", 0.90))
    hist_bins = int(getattr(cfg, "score_hist_bins", 80))
    save_mu = bool(getattr(cfg, "save_mu_vectors", True))
    topk_setting_summary = int(getattr(cfg, "topk_setting_summary", 10))

    total_settings = count_total_settings(token_pools, modules, whole_reduces, whole_topks, last_ks, cfg)
    setting_idx = 0

    print(
        f"[GRID] total_settings={total_settings} | layers={len(layers)} "
        f"| expected_rows≈{total_settings * max(len(layers), 1)} | tau_quantile={tau_q}"
    )

    for token_pool in token_pools:
        for module in modules:
            if token_pool != "whole":
                ks = last_ks if token_pool == "last_k" else [int(getattr(cfg, "dist_last_k", 8))]
                reduce_list = ["mean"]
                topk_list = [int(getattr(cfg, "whole_topk", 8))]
            else:
                ks = [int(getattr(cfg, "dist_last_k", 8))]
                reduce_list = whole_reduces
                topk_list = whole_topks

            for last_k in ks:
                for whole_reduce in reduce_list:
                    for whole_topk in topk_list:
                        setting_idx += 1
                        skey = setting_key(
                            token_pool, module, int(last_k),
                            whole_reduce if token_pool == "whole" else None,
                            int(whole_topk) if token_pool == "whole" else None,
                        )
                        print(f"\n[SETTING {setting_idx:03d}/{total_settings}] {skey}")

                        # cfg patch per setting
                        cfg_local = copy.deepcopy(base_cfg)
                        cfg_local.dist_cal_type = token_pool
                        cfg_local.dist_last_k = int(last_k)
                        cfg_local.whole_reduce = str(whole_reduce)
                        cfg_local.whole_topk = int(whole_topk)

                        simcfg = SimConfig(
                            token_pool=token_pool,
                            module=module,
                            whole_reduce=whole_reduce,
                            whole_topk=int(whole_topk),
                            last_k=int(last_k),
                        )

                        # 1) mu (from forget_mu_loader)
                        mu_by_layer = compute_mu_from_forget(
                            cfg_local, model_base, forget_mu_loader, layers, simcfg, device
                        )

                        # 2) scores distributions (forget vs retain)
                        s_forget = compute_similarity_scores(
                            cfg_local, model_base, forget_eval_loader, layers, simcfg, mu_by_layer, device
                        )
                        s_retain = compute_similarity_scores(
                            cfg_local, model_base, retain_eval_loader, layers, simcfg, mu_by_layer, device
                        )

                        # 3) per-layer metrics
                        rows_this_setting: List[Dict[str, Any]] = []

                        for l in layers:
                            a = s_forget[l]
                            b = s_retain[l]

                            ma = float(a.mean()) if len(a) else float("nan")
                            mb = float(b.mean()) if len(b) else float("nan")
                            gap = (ma - mb) if (np.isfinite(ma) and np.isfinite(mb)) else float("nan")

                            d = cohens_d(a, b)
                            aucv = auroc(a, b)  # pos=forget, neg=retain
                            ksv = ks_statistic(a, b)

                            a_stats = summarize_scores(a)
                            b_stats = summarize_scores(b)

                            # tau from forget distribution at tau_q
                            tau_val = float(np.quantile(a, tau_q)) if len(a) else float("nan")
                            # retain intrusion proxy = fraction of retain above tau_forget
                            fpr_proxy = float(np.mean(b >= tau_val)) if (len(b) and np.isfinite(tau_val)) else float("nan")

                            row = {
                                "layer": int(l),
                                "token_pool": token_pool,
                                "module": module,
                                "last_k": int(last_k),
                                "whole_reduce": whole_reduce if token_pool == "whole" else None,
                                "whole_topk": int(whole_topk) if token_pool == "whole" else None,

                                "mean_forget": ma,
                                "mean_retain": mb,
                                "mean_gap": gap,
                                "abs_mean_gap": abs(gap) if np.isfinite(gap) else float("nan"),

                                "cohens_d": d,
                                "abs_cohens_d": abs(d) if not np.isnan(d) else float("nan"),
                                "auroc_forget_vs_retain": aucv,
                                "ks": ksv,

                                "n_forget": int(len(a)),
                                "n_retain": int(len(b)),

                                "forget_stats": a_stats,
                                "retain_stats": b_stats,

                                "tau_quantile": tau_q,
                                "tau_from_forget": tau_val,
                                "retain_fpr_proxy_at_tau": fpr_proxy,
                            }

                            if meta_row_base:
                                row.update(meta_row_base)

                            results.append(row)
                            rows_this_setting.append(row)

                            if out_jsonl_path is not None:
                                append_jsonl(out_jsonl_path, row)

                        # ---- summarize this setting (best layer) ----
                        if len(rows_this_setting) == 0:
                            print("[SETTING BEST] (no rows)")
                            continue

                        best_setting_row = best_row_among_layers(rows_this_setting)
                        summary_rows.append(best_setting_row)

                        print(f"[SETTING BEST] {fmt_row_short(best_setting_row)}")

                        # ---- update global best live ----
                        if global_best is None:
                            global_best = best_setting_row
                            print(f"[GLOBAL BEST] init -> {fmt_row_short(global_best)}")
                        else:
                            if key_fn(best_setting_row) > key_fn(global_best):
                                global_best = best_setting_row
                                print(f"[GLOBAL BEST] updated -> {fmt_row_short(global_best)}")

    # -----------------------------
    # final sort & print
    # -----------------------------
    results_sorted = sorted(results, key=key_fn, reverse=True)

    print("\n==================== TOP SEPARATION (forget vs retain) ====================")
    for i, r in enumerate(results_sorted[:topk_print]):
        extra = ""
        if r["token_pool"] == "whole":
            extra = f"| reduce={r['whole_reduce']} topk={r['whole_topk']}"
        print(
            f"[{i:02d}] "
            f"{setting_key(r['token_pool'], r['module'], r['last_k'], r.get('whole_reduce'), r.get('whole_topk'))} "
            f"| {fmt_row_short(r)} {extra}"
        )

    # -----------------------------
    # save outputs
    # -----------------------------
    if out_json_path is not None:
        safe_json_dump(out_json_path, {"results": results_sorted})
        print(f"\n[SAVED] {out_json_path}")

    # per-setting summary (best layer per setting)
    if task_dir_for_snapshots:
        ensure_dir(task_dir_for_snapshots)
        safe_json_dump(
            os.path.join(task_dir_for_snapshots, "summary_by_setting.json"),
            {"best_per_setting": sorted(summary_rows, key=key_fn, reverse=True)}
        )
        safe_json_dump(
            os.path.join(task_dir_for_snapshots, f"summary_top{topk_setting_summary}.json"),
            {"topk": sorted(summary_rows, key=key_fn, reverse=True)[:topk_setting_summary]}
        )

    # -----------------------------
    # Save snapshot for best row (best layer only)
    # -----------------------------
    if save_best_score_snapshot and task_dir_for_snapshots and len(results_sorted) > 0:
        best = results_sorted[0]
        snap_dir = os.path.join(task_dir_for_snapshots, "scores")
        ensure_dir(snap_dir)

        # Recompute best config scores once (avoid storing all)
        cfg_best = copy.deepcopy(cfg)
        cfg_best.dist_cal_type = best["token_pool"]
        cfg_best.dist_last_k = int(best["last_k"])
        if best["token_pool"] == "whole":
            cfg_best.whole_reduce = best["whole_reduce"]
            cfg_best.whole_topk = int(best["whole_topk"])

        simcfg_best = SimConfig(
            token_pool=best["token_pool"],
            module=best["module"],
            whole_reduce=best["whole_reduce"] if best["token_pool"] == "whole" else "mean",
            whole_topk=int(best["whole_topk"]) if best["token_pool"] == "whole" else int(getattr(cfg, "whole_topk", 8)),
            last_k=int(best["last_k"]),
        )

        mu_best = compute_mu_from_forget(cfg_best, model_base, forget_mu_loader, layers, simcfg_best, device)
        sf_best = compute_similarity_scores(cfg_best, model_base, forget_eval_loader, layers, simcfg_best, mu_best, device)
        sr_best = compute_similarity_scores(cfg_best, model_base, retain_eval_loader, layers, simcfg_best, mu_best, device)

        L = int(best["layer"])
        a = sf_best[L]
        b = sr_best[L]

        rng = None
        if len(a) and len(b):
            lo = float(min(np.quantile(a, 0.001), np.quantile(b, 0.001)))
            hi = float(max(np.quantile(a, 0.999), np.quantile(b, 0.999)))
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                rng = (lo, hi)

        snapshot = {
            "best_setting": best,
            "forget_stats": summarize_scores(a),
            "retain_stats": summarize_scores(b),
            "forget_hist": hist_snapshot(a, bins=hist_bins, range_=rng),
            "retain_hist": hist_snapshot(b, bins=hist_bins, range_=rng),
        }
        safe_json_dump(os.path.join(snap_dir, "best_layer_hist.json"), snapshot)

        if save_mu:
            save_mu_vectors(task_dir_for_snapshots, mu_best, modules=["q", "v", "qv"])

        print(f"[SNAPSHOT] saved under {snap_dir}")

    return results_sorted


# =========================================================
# 9) Continual runner
# =========================================================
def compute_best_delta(prev_best: Optional[Dict[str, Any]], cur_best: Dict[str, Any]) -> Dict[str, Any]:
    if not prev_best:
        return {"has_prev": False}
    keys = ["abs_cohens_d", "auroc_forget_vs_retain", "ks", "abs_mean_gap", "cohens_d", "mean_gap"]
    delta = {}
    for k in keys:
        pv = prev_best.get(k, float("nan"))
        cv = cur_best.get(k, float("nan"))
        if isinstance(pv, (int, float)) and isinstance(cv, (int, float)) and np.isfinite(pv) and np.isfinite(cv):
            delta[k] = float(cv - pv)
        else:
            delta[k] = float("nan")
    # stability: same config?
    cfg_keys = ["layer", "token_pool", "module", "last_k", "whole_reduce", "whole_topk"]
    delta["same_setting"] = all(prev_best.get(k) == cur_best.get(k) for k in cfg_keys)
    delta["has_prev"] = True
    delta["prev_best_setting_keys"] = {k: prev_best.get(k) for k in cfg_keys}
    return delta

def resolve_tasks_from_cfg(cfg) -> List[str]:
    """
    1) If cfg.task_ids exists and non-empty -> use it
    2) Else fallback to single task ["single"]
    """
    tids = getattr(cfg, "task_ids", None)
    if tids is None:
        return ["single"]
    try:
        tids = list(tids)
    except Exception:
        return ["single"]
    if len(tids) == 0:
        return ["single"]
    return [str(t) for t in tids]

def get_task_paths(cfg, task_id: str) -> Tuple[Any, Any]:
    """
    Supports:
      - cfg.task_to_forget_path[task_id], cfg.task_to_retain_path[task_id]
      - or fallback: cfg.dist_forget_path / cfg.dist_retain_path
    """
    t2f = getattr(cfg, "task_to_forget_path", None)
    t2r = getattr(cfg, "task_to_retain_path", None)

    if t2f is not None and t2r is not None:
        try:
            f = t2f[task_id]
            r = t2r[task_id]
            return f, r
        except Exception:
            pass

    return getattr(cfg, "dist_forget_path", None), getattr(cfg, "dist_retain_path", None)


@hydra.main(version_base=None, config_path="config", config_name="forget_analyze")
def main(cfg):
    print(cfg)
    set_seed(int(cfg.seed))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LOAD] loading model from {cfg.model_family} at {cfg.model_path}")
    model_base = load_model(cfg.model_family, cfg.model_path, torch.device(device))
    model_base.model.to(device)
    model_base.model.eval()

    # run_id and save root
    run_id = str(getattr(cfg, "run_id", "")) or f"forget_analyze_{now_ts()}"
    save_root = str(getattr(cfg, "svdd_save_dir", "./svdd_artifacts"))
    save_root = os.path.join(save_root, run_id)
    ensure_dir(save_root)

    # record run meta once
    run_meta = {
        "run_id": run_id,
        "model_family": str(getattr(cfg, "model_family", "")),
        "model_path": str(getattr(cfg, "model_path", "")),
        "cfg_resolved": cfg_to_dict(cfg),
        "runtime": runtime_meta(seed=int(cfg.seed), device=device),
    }
    safe_json_dump(os.path.join(save_root, "run_meta.json"), run_meta)

    # layers
    layers = [int(x) for x in list(getattr(cfg, "middle_distribution_layer", []))]
    if not layers:
        raise ValueError("cfg.middle_distribution_layer is empty. Provide layers to analyze.")

    hold_ratio = float(getattr(cfg, "dist_holdout_ratio", 0.2))

    # Search space
    token_pools = list(getattr(cfg, "token_pools", ["mean", "last", "last_k", "whole"]))
    modules = list(getattr(cfg, "modules", ["q", "v", "qv"]))
    whole_reduces = list(getattr(cfg, "whole_reduces", ["mean", "max", "topk_mean"]))
    whole_topks = list(getattr(cfg, "whole_topks", [4, 8, 16]))
    last_ks = list(getattr(cfg, "last_ks", [4, 8, 16]))

    topk_print = int(getattr(cfg, "topk_print", 25))
    topk_save = int(getattr(cfg, "topk_save", 50))

    prev_best: Optional[Dict[str, Any]] = None
    all_task_summaries: List[Dict[str, Any]] = []

    task_ids = resolve_tasks_from_cfg(cfg)

    for ti, task_id in enumerate(task_ids):
        print(f"\n==================== [TASK {ti+1}/{len(task_ids)}] {task_id} ====================")

        # task-local cfg override for eval paths
        cfg_task = copy.deepcopy(cfg)
        dist_forget_path, dist_retain_path = get_task_paths(cfg_task, task_id)
        cfg_task.dist_forget_path = dist_forget_path
        cfg_task.dist_retain_path = dist_retain_path

        task_dir = os.path.join(save_root, str(task_id))
        ensure_dir(task_dir)

        # 1) create forget/retain datasets for mu-split
        forget_dataset, retain_dataset = split_raw_dataset_for_forget(
            cfg_task,
            model_base,
            instructions_only=True,
            torch_reformat=False,
        )
        print(f"[DATASET] forget_dataset={len(forget_dataset)} | retain_dataset={len(retain_dataset)}")

        # 2) split forget into train/hold for mu fitting
        n = len(forget_dataset)
        n_hold = max(1, int(n * hold_ratio))
        n_train = max(1, n - n_hold)
        forget_train_raw, forget_hold_raw = random_split(
            forget_dataset,
            [n_train, n_hold],
            generator=torch.Generator().manual_seed(int(cfg_task.seed)),
        )
        print(f"[SPLIT] forget_train_raw={len(forget_train_raw)} | forget_hold_raw={len(forget_hold_raw)}")

        # 3) loaders
        forget_mu_loader = build_loader_from_raw_dataset(cfg_task, model_base, forget_train_raw, shuffle=False, is_eval=False)
        print("forget_mu_loader : ", len(forget_mu_loader))

        if cfg_task.dist_forget_path is None or cfg_task.dist_retain_path is None:
            raise ValueError(
                f"dist_forget_path/dist_retain_path are None for task={task_id}. "
                "Provide cfg.dist_forget_path & cfg.dist_retain_path (single-task) OR "
                "cfg.task_to_forget_path/task_to_retain_path (multi-task)."
            )
        
        forget_eval_loader = build_loader_from_raw_dataset(
            cfg_task, model_base, cfg_task.dist_forget_path,
            shuffle=False, is_eval=True, eval_target="forget_edge"
        )
        retain_eval_loader = build_loader_from_raw_dataset(
            cfg_task, model_base, cfg_task.dist_retain_path,
            shuffle=False, is_eval=True, eval_target="retained_edge"
        )

        # meta for each row (goes into jsonl)
        meta_row_base = {
            "run_id": run_id,
            "task_id": str(task_id),
            "model_family": str(getattr(cfg_task, "model_family", "")),
            "model_path": str(getattr(cfg_task, "model_path", "")),
            "dist_forget_path": str(cfg_task.dist_forget_path),
            "dist_retain_path": str(cfg_task.dist_retain_path),
            "seed": int(cfg_task.seed),
            "layers": [int(x) for x in layers],
            "dist_batch_size": int(getattr(cfg_task, "dist_batch_size", 8)),
            "dist_holdout_ratio": float(getattr(cfg_task, "dist_holdout_ratio", 0.2)),
        }

        # output files
        out_sorted_json = os.path.join(task_dir, "results_sorted.json")
        out_jsonl = os.path.join(task_dir, "results.jsonl")
        out_topk = os.path.join(task_dir, "topk.json")
        out_summary = os.path.join(task_dir, "summary.json")

        # fresh jsonl per task: remove if exists
        if os.path.exists(out_jsonl):
            os.remove(out_jsonl)

        # 4) analyze
        results_sorted = analyze_forget_retain_separation(
            cfg=cfg_task,
            model_base=model_base,
            forget_mu_loader=forget_mu_loader,
            forget_eval_loader=forget_eval_loader,
            retain_eval_loader=retain_eval_loader,
            layers=layers,
            device=device,
            token_pools=token_pools,
            modules=modules,
            whole_reduces=whole_reduces,
            whole_topks=whole_topks,
            last_ks=last_ks,
            topk_print=topk_print,
            out_json_path=out_sorted_json,
            out_jsonl_path=out_jsonl,
            meta_row_base=meta_row_base,
            save_best_score_snapshot=True,
            task_dir_for_snapshots=task_dir,
        )

        # 5) save topk
        safe_json_dump(out_topk, {"topk": results_sorted[:topk_save]})

        # 6) best + delta
        best = results_sorted[0] if len(results_sorted) else {}
        delta = compute_best_delta(prev_best, best) if best else {"has_prev": False}

        summary = {
            "run_id": run_id,
            "task_id": str(task_id),
            "task_index": int(ti),
            "dist_forget_path": str(cfg_task.dist_forget_path),
            "dist_retain_path": str(cfg_task.dist_retain_path),
            "runtime": runtime_meta(seed=int(cfg_task.seed), device=device),
            "best": best,
            "delta_vs_prev_best": delta,
        }
        safe_json_dump(out_summary, summary)
        all_task_summaries.append(summary)

        prev_best = best

    # run-level summary
    safe_json_dump(os.path.join(save_root, "all_tasks_summary.json"), {"tasks": all_task_summaries})
    print(f"\n[DONE] saved under: {save_root}")
    return all_task_summaries


if __name__ == "__main__":
    main()
