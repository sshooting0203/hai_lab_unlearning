import math, torch
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.cache_utils import Cache, DynamicCache

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "LlamaConfig"
DEFAULT_LORA_TARGETS = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"] # @ 수정(layer처리)

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
# Make causal mask used for bi-directional self-attention
def _make_causal_mask(
    input_ids_shape: torch.Size, 
    dtype: torch.dtype, 
    device: torch.device, 
    past_key_values_length: int = 0
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

# Copied from transformers.models.bart.modeling_bart._expand_mask
# Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

# LlamaRMSNorm == T5LayerNorm
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached: # x: [bs, num_attention_heads, seq_len, head_size]
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

# LlamaRotaryEmbedding + linear scaling 
class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

# LlamaRotaryEmbedding + Dynamic NTK scaling
class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

class Llama3RotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        old_context_len=8192,
    ):
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        inv_freq = self.inv_freq
        low_freq_wavelen = self.old_context_len / self.low_freq_factor
        high_freq_wavelen = self.old_context_len / self.high_freq_factor
        wavelen = 2 * math.pi / inv_freq

        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / self.scaling_factor, inv_freq)
        smooth_condition = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        smooth_factor = (self.old_context_len / wavelen - self.low_freq_factor) / (
            self.high_freq_factor - self.low_freq_factor
        )
        inv_freq_llama = torch.where(
            smooth_condition,
            (1 - smooth_factor) * (inv_freq / self.scaling_factor) + smooth_factor * inv_freq,
            inv_freq_llama,
        )

        freqs = torch.einsum("i,j->ij", t, inv_freq_llama)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


# Rotates half the hidden dims of the input
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1) # 마지막 차원(dim)을 반으로 쪼개서(앞 절반 = x1, 뒤 절반 = x2) [x1, x2] -> [-x2, x1]

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]

    if torch.any(position_ids < 0) or torch.any(position_ids >= cos.size(0)):
        print("DEBUG rotary OOB")
        print("  cos.shape:", cos.shape)
        print("  sin.shape:", sin.shape)
        print("  position_ids.shape:", position_ids.shape)
        print("  position_ids.min():", position_ids.min().item())
        print("  position_ids.max():", position_ids.max().item())
        raise RuntimeError("position_ids out of bounds for rotary cache")

    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pretraining_tp = config.pretraining_tp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.lora_target_modules = getattr(config, "lora_target_modules", DEFAULT_LORA_TARGETS) # self.lora_target_modules = config.lora_target_modules  # @ 수정 
    
    ##modified
    def forward(
        self, 
        x, 
        active_adapters_d, 
        ood_weight, 
        o_lora_layer, 
        lora_start_layer 
    ):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)
            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)
            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)]
            down_proj = sum(down_proj)
        else:
            ##newmodified
            o_loss = 0  
            def is_lora_wrapped(m: nn.Module) -> bool:
                return hasattr(m, "base_layer") 
            def call_proj(proj: nn.Module, x: torch.Tensor, key: str):
                zero = x.new_zeros(())
                if not is_lora_wrapped(proj):
                    return proj(x), zero
                if o_lora_layer is None:
                    return proj.base_layer(x), zero
                if len(o_lora_layer) == 0:
                    out, o_l = proj(x, active_adapters_d, ood_weight, o_lora_layer)
                    return out, o_l
                if isinstance(o_lora_layer, dict):
                    ref = o_lora_layer.get(key, None)
                    if ref is None:
                        return proj(x), zero
                    out, o_l = proj(x, active_adapters_d, ood_weight, ref)
                    return out, o_l
                return proj(x), zero

            use_olora = ( ("down_proj" in self.lora_target_modules) and ("gate_proj" in self.lora_target_modules) and ("up_proj" in self.lora_target_modules) )
            if use_olora:
                up,   o_l1 = call_proj(self.up_proj, x, "mlp.up_proj")
                gate, o_l2 = call_proj(self.gate_proj, x, "mlp.gate_proj")
                h = self.act_fn(gate) * up
                down_proj, o_l3 = call_proj(self.down_proj, h, "mlp.down_proj")
                o_loss = o_loss + o_l1 + o_l2 + o_l3
                return down_proj, o_loss
            else:
                down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                return down_proj

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# Multi-headed attention from 'Attention Is All You Need' paper
class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.lora_target_modules = getattr(config, "lora_target_modules", DEFAULT_LORA_TARGETS) # self.lora_target_modules = config.lora_target_modules  # @ 수정 
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False) # o_proj로 다시 (B, T, hidden_size)로 되돌림
        self._init_rope()
    
    def _init_rope(self):
        base = getattr(self.config, "rope_theta", getattr(self.config, "base", 10000.0))
        rope_cfg = getattr(self.config, "rope_scaling", None)
        if rope_cfg is None:
            rope_cfg = getattr(self.config, "rope_parameters", None)

        if rope_cfg is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=base,
            )
            return

        scaling_type = rope_cfg.get("type", rope_cfg.get("rope_type"))
        scaling_factor = rope_cfg.get("factor", 1.0)

        if scaling_type == "linear":
            self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=base,
                scaling_factor=scaling_factor,
            )
        elif scaling_type == "dynamic":
            self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=base,
                scaling_factor=scaling_factor,
            )
        elif scaling_type == "llama3":
            low_freq_factor = rope_cfg.get("low_freq_factor", 1.0)
            high_freq_factor = rope_cfg.get("high_freq_factor", 4.0)
            old_context_len = rope_cfg.get("original_max_position_embeddings", 8192)
            self.rotary_emb = Llama3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=base,
                scaling_factor=scaling_factor,
                low_freq_factor=low_freq_factor,
                high_freq_factor=high_freq_factor,
                old_context_len=old_context_len,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
        
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        active_adapters_d=None,
        ood_weight=None,
        o_lora_layer=None,
        use_cache: bool = False,
        lora_start_layer: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        ########################################################################################
        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

            o_loss = hidden_states.new_zeros(())
        else:
            o_loss = hidden_states.new_zeros(())

            def is_lora_wrapped(m: nn.Module) -> bool:
                return hasattr(m, "base_layer")

            def call_proj(proj: nn.Module, x: torch.Tensor, key: str):
                zero = x.new_zeros(())
                if not is_lora_wrapped(proj):
                    return proj(x), zero
                if o_lora_layer is None:
                    return proj.base_layer(x), zero
                if len(o_lora_layer) == 0:
                    out, o_l = proj(x, active_adapters_d, ood_weight, o_lora_layer)
                    return out, o_l
                if isinstance(o_lora_layer, dict):
                    ref = o_lora_layer.get(key, None)
                    if ref is None:
                        return proj(x), zero
                    out, o_l = proj(x, active_adapters_d, ood_weight, ref)
                    return out, o_l
                return proj(x), zero

            zero = hidden_states.new_zeros(())
            query_states, o_l1 = (
                call_proj(self.q_proj, hidden_states, "self_attn.q_proj")
                if "q_proj" in self.lora_target_modules
                else (self.q_proj(hidden_states), zero)
            )
            key_states, o_l2 = (
                call_proj(self.k_proj, hidden_states, "self_attn.k_proj")
                if "k_proj" in self.lora_target_modules
                else (self.k_proj(hidden_states), zero)
            )
            value_states, o_l3 = (
                call_proj(self.v_proj, hidden_states, "self_attn.v_proj")
                if "v_proj" in self.lora_target_modules
                else (self.v_proj(hidden_states), zero)
            )
            o_loss = o_loss + o_l1 + o_l2 + o_l3
        ########################################################################################

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # -----------------------------
        # usable past cache 검사
        # -----------------------------
        has_usable_past = (
            past_key_value is not None
            and isinstance(past_key_value, (list, tuple))
            and len(past_key_value) == 2
            and past_key_value[0] is not None
            and past_key_value[1] is not None
        )

        if past_key_value is not None and not isinstance(past_key_value, (list, tuple)):
            raise TypeError(f"Unsupported past_key_value type: {type(past_key_value)}")

        if has_usable_past:
            # shape sanity check
            if not torch.is_tensor(past_key_value[0]) or not torch.is_tensor(past_key_value[1]):
                raise TypeError("past_key_value must contain tensors when not None")
            # expected: [bsz, num_kv_heads, past_len, head_dim]
            if past_key_value[0].dim() != 4 or past_key_value[1].dim() != 4:
                raise ValueError(
                    f"Invalid past_key_value dims: key={past_key_value[0].shape}, value={past_key_value[1].shape}"
                )

        kv_seq_len = key_states.shape[-2]
        if has_usable_past:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if has_usable_past:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, "
                f"but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                    f"but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, "
                f"but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        ########################################################################################
        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            if "o_proj" in self.lora_target_modules:
                attn_output, o_l = call_proj(self.o_proj, attn_output, "self_attn.o_proj")
                o_loss = o_loss + o_l
            else:
                attn_output = self.o_proj(attn_output)
        ########################################################################################

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present_key_value, o_loss

class LlamaDecoderLayer(nn.Module): # LlamaDecoderLayer = self-attention + MLP + residual + layernorm
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.lora_target_modules = getattr(config, "lora_target_modules", DEFAULT_LORA_TARGETS) # self.lora_target_modules = config.lora_target_modules  # @ 수정 
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    ##newmodified
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        active_adapters_d = None,
        ood_weight = None,
        o_lora_layer = None,
        use_cache: Optional[bool] = False,
        lora_start_layer = None 
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states # pre-LN 구조
        hidden_states = self.input_layernorm(hidden_states) # RSMNorm1

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, o_loss = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            active_adapters_d=active_adapters_d,
            ood_weight=ood_weight,
            o_lora_layer=o_lora_layer, 
            use_cache=use_cache,
            lora_start_layer=lora_start_layer # @ 수정 -> LlamaAttention
        )

        hidden_states = residual + hidden_states 
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states) # RSMNorm2

        # MLP
        if 'down_proj' in self.lora_target_modules and 'gate_proj' in self.lora_target_modules and 'up_proj' in self.lora_target_modules:
            hidden_states, o_m = self.mlp(
                hidden_states, 
                active_adapters_d, 
                ood_weight, 
                o_lora_layer, 
                lora_start_layer # @ 수정 -> LlamaMLP
            )
            o_loss += o_m
            # hidden_states = hidden_states[0]
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states # +
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        outputs += (o_loss,)  ##modified
        return outputs # [torch.Size([8, 256, 4096]), o_loss]

class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value

class LlamaModel(LlamaPreTrainedModel):
    # Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.orthogonal_loss = getattr(config, "orthogonal_loss", False) # self.orthogonal_loss = config.orthogonal_loss # @ 수정
        self.post_init() # Initialize weights and apply final processing

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # LLaMA 디코더에서 self-attention에 들어갈 "최종 attention mask = causal mask + padding mask"를 만들어주는 헬퍼 함수
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask( # create causal mask
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask( # create padding mask 
                attention_mask, 
                inputs_embeds.dtype, 
                tgt_len=input_shape[-1]).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask
    
    def _get_olora_for_layer(self, olora, idx): # @ 수정(helper: olora에서 idx번째 레이어를 안전하게 가져오기
        if olora is None:
            return None
        if isinstance(olora, (list, tuple)):
            return olora[idx] if idx < len(olora) else None
        if isinstance(olora, dict):
            return olora.get(idx, None)
        return None

    def forward(
        self,
        input_ids: torch.LongTensor = None, # 토큰(id)를 담은 정수 텐서 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None, # 토큰(id)를 embedding한 실수 텐서
        active_adapters_d = None, # @ 추가
        ood_weight = None, # @ 추가
        lora_start_layer = None, # @ 수정 
        olora = None, # @ 추가
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None, # softmax 이후의 attention
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing during training. Setting `use_cache=False`."
            )
            use_cache = False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None: 
            batch_size, seq_length = input_ids.shape # (batch_size, seq_length)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape # (batch_size, seq_length, hidden_size)
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length # 현재 입력으로 들어온 토큰의 길이 
        past_key_values_length = 0

        legacy_past_key_values = past_key_values

        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                legacy_past_key_values = past_key_values.to_legacy_cache()
            elif not isinstance(past_key_values, (list, tuple)):
                raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")

            first_past = legacy_past_key_values[0] if len(legacy_past_key_values) > 0 else None
            has_usable_past = (
                first_past is not None
                and isinstance(first_past, (list, tuple))
                and len(first_past) == 2
                and first_past[0] is not None
                and first_past[1] is not None
            )

            if has_usable_past:
                past_key_values_length = first_past[0].shape[2]
                seq_length_with_past = seq_length + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None: # (batch, seq_len, hidden_size) 임베딩 생성 
            inputs_embeds = self.embed_tokens(input_ids) 
        
        if attention_mask is None: # attention mask가 없다면 기본값 1로 (batch, seq_length_with_past) 생성 
            attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)

        attention_mask = self._prepare_decoder_attention_mask( # attention mask 생성 
            attention_mask, 
            (batch_size, seq_length), 
            inputs_embeds, 
            past_key_values_length
        ) 
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        o_loss_all_layers = []
        layer_number = 0
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # past_key_value = past_key_values[idx] if past_key_values is not None else None
            if legacy_past_key_values is None:
                past_key_value = None
            else:
                candidate = legacy_past_key_values[idx]
                if (
                    candidate is None
                    or not isinstance(candidate, (list, tuple))
                    or len(candidate) != 2
                    or candidate[0] is None
                    or candidate[1] is None
                ):
                    past_key_value = None
                else:
                    past_key_value = candidate

            if self.orthogonal_loss:
                olora_input = self._get_olora_for_layer(olora, idx)
            else:
                olora_input = olora

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            *inputs,
                            output_attentions=output_attentions,
                            active_adapters_d=active_adapters_d,
                            ood_weight=ood_weight,
                            o_lora_layer=olora_input,
                            use_cache=False,
                            lora_start_layer=lora_start_layer,
                        )
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    active_adapters_d=active_adapters_d,
                    ood_weight=ood_weight,
                    o_lora_layer=olora_input,
                    use_cache=use_cache,
                    lora_start_layer=lora_start_layer,
                )

            o_loss = layer_outputs[-1]
            layer_outputs_wo_oloss = layer_outputs[:-1]

            device = hidden_states.device
            dtype = hidden_states.dtype
            if torch.is_tensor(o_loss):
                o_loss = o_loss.to(device)
            else:
                o_loss = torch.tensor(o_loss, device=device, dtype=dtype)

            o_loss_all_layers.append(o_loss)
            layer_number += 1

            hidden_states = layer_outputs_wo_oloss[0]

            offset = 1
            if output_attentions:
                all_self_attns += (layer_outputs_wo_oloss[offset],)
                offset += 1

            if use_cache:
                next_decoder_cache += (layer_outputs_wo_oloss[offset],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        o_loss_all_layers = torch.sum(torch.stack(o_loss_all_layers, dim=0)) / layer_number

        if not return_dict:
            outputs = tuple(
                v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None
            )
            return outputs, o_loss_all_layers

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ), o_loss_all_layers

# LlamaForCausalLM : 다음 토큰 예측
# self.lm_head : vocab_size가 들어감 
class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lora_target_modules = getattr(config, "lora_target_modules", DEFAULT_LORA_TARGETS) # self.lora_target_modules = config.lora_target_modules @ 수정
        # print("NEW PARAMETERS", self.lora_target_modules)
        self.post_init() # Initialize weights and apply final processing

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs, _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state

        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)

        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1:].contiguous()

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

class LlamaForCausalLM_ood(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # self.scale = config.orthogonal_loss_weight # @ 수정 
        # self.lora_target_modules = config.lora_target_modules  # @ 수정 
        self.scale = getattr(config, "orthogonal_loss_weight", 0.1)
        self.lora_target_modules = getattr(config, "lora_target_modules", DEFAULT_LORA_TARGETS)
        self.orthogonal_loss = getattr(config, "orthogonal_loss", False)
        # print("NEW PARAMETERS", self.lora_target_modules)
        self.post_init() # Initialize weights and apply final processing
        self.lora_start_layer = 0 # @ 수정

    def init_olora(self, orthogonal_loss, olora_weights):
        self.olora_weights = {}
        # only lora_a
        if len(olora_weights) == 0:
            self.orthogonal_loss = False
        else:
            self.orthogonal_loss = orthogonal_loss
            for i in range(len(olora_weights)):
                for j in olora_weights[i]:
                    if "lora_A" not in j:
                        continue
                    layer = j.split('layers.')[1].split('.')[0]
                    type = j.split(layer + '.')[1].split('.lora_A')[0]
                    if int(layer) not in self.olora_weights:
                        self.olora_weights[int(layer)] = {}
                    if type not in self.olora_weights[int(layer)]:
                        self.olora_weights[int(layer)][type] = [olora_weights[i][j]] # self.olora_weights[0]["q_proj"] = [A0_q, A1_q]
                    else:
                        self.olora_weights[int(layer)][type].append(olora_weights[i][j])
        print("Orthogonal: ", self.orthogonal_loss, "Selector initialized")

    def init_oodweight(self, ood_weight):
        self.ood_weight = ood_weight

    def init_active_adapters_d(self, active_adapters_d):
        self.active_adapters_d = active_adapters_d

    def init_lora_start_layer(self, lora_start_layer):
        self.lora_start_layer = lora_start_layer

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        olora_weights = getattr(self, "olora_weights", {})

        outputs, o_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            active_adapters_d=getattr(self, "active_adapters_d", None),
            ood_weight=getattr(self, "ood_weight", None),
            lora_start_layer=self.lora_start_layer,
            olora=olora_weights,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state

        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)

        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if self.orthogonal_loss:
            o_loss = o_loss.to(logits.device)
            if loss is None:
                loss = self.scale * o_loss
            else:
                loss = loss.to(logits.device) + (self.scale * o_loss)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        **kwargs
    ):
        def _has_usable_past(past_key_values):
            if past_key_values is None:
                return False

            if isinstance(past_key_values, Cache):
                past_key_values = past_key_values.to_legacy_cache()

            if not isinstance(past_key_values, (list, tuple)) or len(past_key_values) == 0:
                return False

            first = past_key_values[0]
            if first is None or not isinstance(first, (list, tuple)) or len(first) != 2:
                return False

            return first[0] is not None and first[1] is not None

        has_past = _has_usable_past(past_key_values)

        if has_past:
            input_ids = input_ids[:, -1:]

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if has_past:
                position_ids = position_ids[:, -1:].contiguous()

        if inputs_embeds is not None and not has_past:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past



