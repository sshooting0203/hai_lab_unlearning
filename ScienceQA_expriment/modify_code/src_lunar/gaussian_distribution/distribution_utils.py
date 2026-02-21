from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src_lunar.gaussian_distribution.gaussian import dist_mean_cpu


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    # F.normalize already does x / (norm + eps)
    return F.normalize(x, dim=dim, eps=eps)


def cosine_sim(x: torch.Tensor, mu: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Cosine similarity between x and mu.
    - If x is [B,D] and mu is [D], returns [B]
    - If x is [B,T,D] and mu is [D], returns [B,T]
    """
    x_n = _normalize(x, dim=-1, eps=eps)
    mu_n = _normalize(mu, dim=-1, eps=eps)
    # broadcast mu over leading dims of x
    while mu_n.dim() < x_n.dim():
        mu_n = mu_n.unsqueeze(0)
    return (x_n * mu_n).sum(dim=-1)


def pool_tokens(
    x: torch.Tensor,           # [B,T,D]
    mask: torch.Tensor,        # [B,T] float/bool (1=valid)
    mode: str,
    last_k: int = 8,
) -> torch.Tensor:
    """
    Returns pooled features per sample: [B,D]
    mode: "mean" | "last" | "last_k"
    """
    if x.dim() != 3:
        raise ValueError(f"pool_tokens expects [B,T,D], got {tuple(x.shape)}")

    B, T, D = x.shape
    device = x.device

    mask_f = mask.to(dtype=x.dtype)
    valid = mask_f > 0.5
    lengths = valid.sum(dim=1).long()  # [B]

    if mode == "mean":
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
        return (x * mask_f.unsqueeze(-1)).sum(dim=1) / denom

    if mode == "last":
        # last valid token index for each sample
        idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)   # [B,T]
        masked_idx = idx.masked_fill(~valid, -1)
        last_idx = masked_idx.max(dim=1).values.clamp_min(0)  # [B]
        return x[torch.arange(B, device=device), last_idx, :]

    if mode == "last_k":
        K = int(last_k)
        # window [start, lengths)
        start = (lengths - K).clamp_min(0)  # [B]
        t = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B,T]
        win = (t >= start.unsqueeze(1)) & (t < lengths.unsqueeze(1))  # [B,T]
        win_f = win.to(dtype=x.dtype)
        denom = win_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * win_f.unsqueeze(-1)).sum(dim=1) / denom

    raise ValueError(f"Unsupported pool mode: {mode}")


def pool_qv(
    q: torch.Tensor,            # [B,T,Dq]
    v: torch.Tensor,            # [B,T,Dv]
    pad_mask_2d: torch.Tensor,  # [B,T]
    token_pool: str,
    last_k: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns per-sample pooled q and v:
      q_s: [B,Dq], v_s: [B,Dv]
    """
    q_s = pool_tokens(q, pad_mask_2d, mode=token_pool, last_k=last_k)
    v_s = pool_tokens(v, pad_mask_2d, mode=token_pool, last_k=last_k)
    return q_s, v_s


# -----------------------------
# Tokenwise score reduction
# -----------------------------

def reduce_token_scores(
    scores: torch.Tensor,   # [B,T]
    mask: torch.Tensor,     # [B,T] float/bool (1=valid)
    reduce: str = "mean",
    topk: int = 8,
) -> torch.Tensor:
    """
    Reduce per-token scores to per-sample scores: [B]
    reduce: "mean" | "max" | "topk_mean"
    """
    mask_f = mask.to(dtype=scores.dtype)
    valid = mask_f > 0.5

    if reduce == "mean":
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (scores * mask_f).sum(dim=1) / denom

    neg_inf = torch.finfo(scores.dtype).min
    masked = scores.masked_fill(~valid, neg_inf)

    if reduce == "max":
        return masked.max(dim=1).values

    if reduce == "topk_mean":
        K = min(int(topk), masked.size(1))
        vals = masked.topk(k=K, dim=1).values  # [B,K]
        # If a row has <K valid tokens, masked will include neg_inf in topk,
        # which can drag the mean down. We'll compute mean over valid-only topk:
        # simple fix: clamp neg_inf to 0 contribution and divide by actual count.
        is_finite = torch.isfinite(vals)
        cnt = is_finite.sum(dim=1).clamp_min(1)
        vals2 = torch.where(is_finite, vals, torch.zeros_like(vals))
        return vals2.sum(dim=1) / cnt

    raise ValueError(f"Unsupported reduce: {reduce}")


def cosine_tokenwise_reduce(
    x_tok: torch.Tensor,    # [B,T,D]
    mu: torch.Tensor,       # [D]
    mask: torch.Tensor,     # [B,T]
    reduce: str,
    topk: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute cosine per token then reduce to [B].
    """
    tok_scores = cosine_sim(x_tok, mu, eps=eps)  # [B,T]
    return reduce_token_scores(tok_scores, mask, reduce=reduce, topk=topk)


def collect_qv_feats_with_hooks(
    cfg,
    model_base,
    layer_idxs: List[int],
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], List[Any]]:
    """
    Registers forward-pre-hooks on specified transformer blocks to collect q/v features.

    Stores in feats[layer]:
      - if cfg.dist_cal_type == "whole":
          {"q": [B,T,Dq], "v": [B,T,Dv], "mask": [B,T]}
      - else:
          {"q": [B,Dq], "v": [B,Dv]}  (pooled already)
    """
    layers = model_base.model.model.layers
    token_pool = str(getattr(cfg, "dist_cal_type", "mean"))
    last_k = int(getattr(cfg, "dist_last_k", 8))

    feats: Dict[int, Dict[str, torch.Tensor]] = {}

    def _get_pad_mask_2d(B: int, T: int, device, dtype) -> torch.Tensor:
        pad_mask_2d = getattr(model_base, "_cur_attn_mask_2d", None)
        if pad_mask_2d is None:
            raise ValueError("model_base._cur_attn_mask_2d is None; set it before forward.")
        if pad_mask_2d.dim() != 2 or pad_mask_2d.size(0) != B or pad_mask_2d.size(1) != T:
            got = tuple(pad_mask_2d.shape)
            raise ValueError(f"pad_mask_2d wrong shape: got={got}, expected={(B,T)}")
        return pad_mask_2d.to(device=device, dtype=dtype)

    def _register_pre_hook(module, fn):
        # HF/PT versions differ on with_kwargs; support both
        try:
            return module.register_forward_pre_hook(fn, with_kwargs=True)
        except TypeError:
            return module.register_forward_pre_hook(lambda m, inputs: fn(m, inputs, {}))

    def make_pre_hook(layer_idx: int):
        def pre_hook(module, inputs, kwargs):
            h = inputs[0]  # [B,T,D]
            B, T, _ = h.shape
            q = module.self_attn.q_proj(h)  # [B,T,Dq]
            v = module.self_attn.v_proj(h)  # [B,T,Dv]
            mask = _get_pad_mask_2d(B, T, q.device, q.dtype)

            if token_pool == "whole":
                feats[layer_idx] = {"q": q, "v": v, "mask": mask}
            else:
                q_s, v_s = pool_qv(q, v, mask, token_pool=token_pool, last_k=last_k)
                feats[layer_idx] = {"q": q_s, "v": v_s}
        return pre_hook

    handles: List[Any] = []
    for l in layer_idxs:
        handles.append(_register_pre_hook(layers[l], make_pre_hook(l)))
    return feats, handles


def _get_text_and_topic_from_batch(batch) -> Tuple[List[str], List[Any]]:
    """
    Supports either:
      - batch is dict: {"text_input": [...], "topic": [...]}
      - batch is list[dict]: [{"text_input":..., "topic":...}, ...]
    """
    if isinstance(batch, dict):
        return list(batch["text_input"]), list(batch["topic"])
    # list[dict]
    return [x["text_input"] for x in batch], [x["topic"] for x in batch]


def _group_indices_by_topic(topics: Sequence[Any]) -> Dict[str, torch.Tensor]:
    """
    Returns a dict: topic_str -> LongTensor[indices]
    Batch-level grouping to avoid per-sample loops.
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, t in enumerate(topics):
        groups[str(t)].append(i)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in groups.items()}


def compute_similarity_scores(
    cfg,
    model_base,
    layer_idxs: List[int],
    dists: Dict[int, Dict[str, Any]],   # {layer: {"q": GaussianDistribution, "v": ...}}
    dataset: List[Dict[str, Any]],
    batch_size: int,
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, torch.Tensor]]:
    """
    Returns:
      s_by_layer[domain][layer] = Tensor[N_domain]   (S_l per sample)
      S_by_domain[domain]       = Tensor[N_domain]   (S_max per sample)
    """
    device = model_base.model.device
    token_pool = str(getattr(cfg, "dist_cal_type", "mean"))  # "mean" | "last_k" | "whole" ...
    whole_reduce = str(getattr(cfg, "whole_reduce", "mean"))
    whole_topk = int(getattr(cfg, "whole_topk", 8))

    # Precompute mu vectors (CPU -> move to device per layer when needed)
    mu_q_cpu = {l: dist_mean_cpu(dists[l]["q"]).detach().cpu() for l in layer_idxs}
    mu_v_cpu = {l: dist_mean_cpu(dists[l]["v"]).detach().cpu() for l in layer_idxs}

    # Accumulators (store CPU tensors, concat at end)
    s_acc: Dict[str, Dict[int, List[torch.Tensor]]] = defaultdict(lambda: {l: [] for l in layer_idxs})
    S_acc: Dict[str, List[torch.Tensor]] = defaultdict(list)

    feats, handles = collect_qv_feats_with_hooks(cfg, model_base=model_base, layer_idxs=layer_idxs)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    try:
        with torch.no_grad():
            for batch in loader:
                texts, topics = _get_text_and_topic_from_batch(batch)
                groups = _group_indices_by_topic(topics)  # topic -> indices tensor (CPU)

                tok = model_base.tokenize_instructions_fn(instructions=texts)
                attn_mask = tok.attention_mask.to(device)
                model_base._cur_attn_mask_2d = attn_mask

                # Forward once to populate feats via hooks
                _ = model_base.model(
                    input_ids=tok.input_ids.to(device),
                    attention_mask=attn_mask,
                    use_cache=False,
                )

                B = len(topics)
                # Collect per-layer per-sample scores on device
                layer_scores = torch.empty((len(layer_idxs), B), device=device, dtype=torch.float32)

                for li, l in enumerate(layer_idxs):
                    mu_q = mu_q_cpu[l].to(device=device, dtype=torch.float32)
                    mu_v = mu_v_cpu[l].to(device=device, dtype=torch.float32)

                    if token_pool == "whole":
                        q_tok = feats[l]["q"].to(dtype=torch.float32)      # [B,T,Dq]
                        v_tok = feats[l]["v"].to(dtype=torch.float32)      # [B,T,Dv]
                        m = feats[l]["mask"].to(dtype=torch.float32)       # [B,T]

                        phi_q = cosine_tokenwise_reduce(q_tok, mu_q, m, reduce=whole_reduce, topk=whole_topk)
                        phi_v = cosine_tokenwise_reduce(v_tok, mu_v, m, reduce=whole_reduce, topk=whole_topk)
                        S_l = 0.5 * (phi_q + phi_v)  # [B]
                    else:
                        q_s = feats[l]["q"].to(dtype=torch.float32)        # [B,Dq]
                        v_s = feats[l]["v"].to(dtype=torch.float32)        # [B,Dv]
                        phi_q = cosine_sim(q_s, mu_q)                      # [B]
                        phi_v = cosine_sim(v_s, mu_v)                      # [B]
                        S_l = 0.5 * (phi_q + phi_v)

                    layer_scores[li] = S_l

                # S_max per sample
                S_max = layer_scores.max(dim=0).values  # [B]

                # Accumulate per domain at batch-level (no per-sample loops)
                layer_scores_cpu = layer_scores.detach().cpu()  # [L,B]
                S_max_cpu = S_max.detach().cpu()                # [B]

                for dom, idx_cpu in groups.items():
                    idx = idx_cpu  # CPU LongTensor
                    # per layer: [N_dom_batch]
                    for li, l in enumerate(layer_idxs):
                        s_acc[dom][l].append(layer_scores_cpu[li, idx])
                    S_acc[dom].append(S_max_cpu[idx])

    finally:
        # Remove hooks even on error
        for h in handles:
            h.remove()
        handles.clear()

    # Concat to final tensors
    s_by_layer: Dict[str, Dict[int, torch.Tensor]] = {}
    S_by_domain: Dict[str, torch.Tensor] = {}

    for dom in s_acc.keys():
        s_by_layer[dom] = {}
        for l in layer_idxs:
            parts = s_acc[dom][l]
            s_by_layer[dom][l] = torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=torch.float32)
        parts_S = S_acc[dom]
        S_by_domain[dom] = torch.cat(parts_S, dim=0) if parts_S else torch.empty(0, dtype=torch.float32)

    return s_by_layer, S_by_domain
