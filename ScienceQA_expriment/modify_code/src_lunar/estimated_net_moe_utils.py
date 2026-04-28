# estimated_net_moe_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src_lunar.estimated_net_utils import _get_down_proj, _set_down_proj

# experts + gate
# gate의 역할 : 입력 activation x를 보고 expert K개 중 
# 어떤 expert를 쓸지 점수(logit)를 만든 다음 softmax로 확률로 바꿈
# token단위의 gating? VS sequence단위의 gating?
# token단위 gating이 더 세밀하게 전문가 선택 가능, sequence단위 gating은 전체 시퀀스에 대해 하나의 전문가 선택 

#==============MoE with LoRA================
class LUNAR_LoRA_MoE_net(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_experts: int = 4,
        top_k: int = 1,
        gate_hidden: int = 0,   # 현재 add_experts는 gate_hidden=0 (Linear gate) 기준으로 구현
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        pretrained_weight=None,
    ):
        super(LUNAR_LoRA_MoE_net, self).__init__()

        dtype = pretrained_weight.dtype if pretrained_weight is not None else torch.float32
        device = pretrained_weight.device if pretrained_weight is not None else "cuda"

        # frozen base linear(14336 -> 4096)
        self.linear = nn.Linear(input_dim, output_dim, bias=False, device=device, dtype=dtype)

        if pretrained_weight is not None:
            with torch.no_grad():
                self.linear.weight.copy_(pretrained_weight)

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.n_experts = int(n_experts)
        self.top_k = min(int(top_k), self.n_experts)

        self.lora_r = int(lora_rank)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        self.scaling = self.lora_alpha / self.lora_r
        self.gate_hidden = int(gate_hidden)

        # experts -> A[e]: (r, in), B[e]: (out, r)
        self.lora_A = nn.Parameter(
            torch.randn(self.n_experts, self.lora_r, input_dim, device=device, dtype=dtype) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.randn(self.n_experts, output_dim, self.lora_r, device=device, dtype=dtype) * 0.01
        )

        with torch.no_grad(): # expert 0번만 zero로 설정
            self.lora_A[0].zero_()
            self.lora_B[0].zero_()

        # gate
        # self.gate.weight : shape (n_experts, input_dim)
        # self.gate.bias : shape (n_experts,)
        # 즉 expert 수가 4면, 각 입력 벡터를 보고 4개 expert 점수를 내는 선형 classifier로 동작함
        if self.gate_hidden > 0:
            self.gate = nn.Sequential(
                nn.Linear(input_dim, self.gate_hidden, device=device, dtype=dtype),
                nn.GELU(),
                nn.Linear(self.gate_hidden, self.n_experts, device=device, dtype=dtype),
            )
        else:
            self.gate = nn.Linear(input_dim, self.n_experts, device=device, dtype=dtype)

    def _ensure_dtype(self, x):
        if x.dtype != self.linear.weight.dtype:
            x = x.to(self.linear.weight.dtype)
        return x

    def add_experts(
        self,
        num_new_experts: int,
        init_std: float = 0.01,
        update_top_k: bool = False,
        new_top_k: int | None = None,
    ):
        num_new_experts = int(num_new_experts)
        if num_new_experts <= 0:
            return []

        device = self.lora_A.device
        dtype = self.lora_A.dtype

        old_n = self.n_experts
        new_n = old_n + num_new_experts

        input_dim = self.lora_A.shape[-1]
        output_dim = self.lora_B.shape[1]
        rank = self.lora_A.shape[1]

        # 1) LoRA expert 확장
        new_A = torch.randn(num_new_experts, rank, input_dim, device=device, dtype=dtype) * init_std
        new_B = torch.randn(num_new_experts, output_dim, rank, device=device, dtype=dtype) * init_std

        with torch.no_grad():
            cat_A = torch.cat([self.lora_A.data, new_A], dim=0)
            cat_B = torch.cat([self.lora_B.data, new_B], dim=0)

        self.lora_A = nn.Parameter(cat_A)
        self.lora_B = nn.Parameter(cat_B)

        # 2) gate 확장
        if self.gate_hidden > 0:
            if not isinstance(self.gate, nn.Sequential):
                raise TypeError("self.gate must be nn.Sequential when gate_hidden > 0.")
            old_gate_out = self.gate[-1]
            new_gate_out = nn.Linear(old_gate_out.in_features, new_n, bias=(old_gate_out.bias is not None), device=device, dtype=dtype)
            with torch.no_grad():
                new_gate_out.weight[:old_n].copy_(old_gate_out.weight.data)
                new_gate_out.weight[old_n:].normal_(mean=0.0, std=init_std)
                if old_gate_out.bias is not None:
                    new_gate_out.bias[:old_n].copy_(old_gate_out.bias.data)
                    new_gate_out.bias[old_n:].zero_()
            self.gate[-1] = new_gate_out
        else:
            if not isinstance(self.gate, nn.Linear):
                raise TypeError("self.gate must be nn.Linear when gate_hidden == 0.")
            old_gate = self.gate
            new_gate = nn.Linear(old_gate.in_features, new_n, bias=(old_gate.bias is not None), device=device, dtype=dtype)
            with torch.no_grad():
                new_gate.weight[:old_n].copy_(old_gate.weight.data)
                new_gate.weight[old_n:].normal_(mean=0.0, std=init_std)
                if old_gate.bias is not None:
                    new_gate.bias[:old_n].copy_(old_gate.bias.data)
                    new_gate.bias[old_n:].zero_()
            self.gate = new_gate
        self.n_experts = new_n

        if update_top_k:
            if new_top_k is not None:
                self.top_k = min(int(new_top_k), self.n_experts)
            else:
                self.top_k = min(self.top_k, self.n_experts)
        else:
            self.top_k = min(self.top_k, self.n_experts)
        return list(range(old_n, new_n))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_gate: bool = False,
        return_router_logits: bool = False,
        return_all_experts: bool = False,
        forced_expert_idx: int | None = None,
    ):
        x = self._ensure_dtype(x) # [B, 14336]
        base = self.linear(x) # down proj : [B, 14336] -> [B, 4096]
        logits = self.gate(x)  # router logits 계산 : [B, 14336] -> [B, E]

        probs = F.softmax(logits, dim=-1)  # router logit으로 softmax 확률 계산 
        top1 = probs.argmax(dim=-1)

        if self.top_k < self.n_experts: # 현재 코드는 top_k=1로 fixed, 사실 상 hard_routing으로 동작
            _, topi = torch.topk(probs, k=self.top_k, dim=-1)
            
            # 라우터가 선택한 expert 번호를 직접 기록
            if getattr(self, "record_expert", False):
                if not hasattr(self, "expert_history"):
                    self.expert_history = []
                self.expert_history.append(topi.detach().cpu())
                
            mask = torch.zeros_like(probs)
            mask.scatter_(-1, topi, 1.0)
            probs = probs * mask
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9) # probs 정규화([0,1])

        x_d = self.lora_dropout(x)
        Ax = torch.einsum("...h,erh->...er", x_d, self.lora_A)  
        BAx = torch.einsum("...er,eor->...eo", Ax, self.lora_B)  
        update = torch.einsum("...e,...eo->...o", probs, BAx)   

        out = base + self.scaling * update
        res = [out]
        
        if return_router_logits:
            res.append(logits)
        if return_gate:
            res.append(probs)
        if return_all_experts:
            res.append(BAx)
            res.append(base)
        if len(res) == 1:
            return res[0]
        return tuple(res)

def save_lora_moe_only(
    model,
    save_path,
    layer_idxs=None,
):
    payload = {}

    if layer_idxs is None:
        raise ValueError("layer_idxs must be provided for save_lora_moe_only().")

    for l in layer_idxs:
        module = _get_down_proj(model, l)

        if not isinstance(module, LUNAR_LoRA_MoE_net):
            raise TypeError(
                f"Layer {l} down_proj is not LUNAR_LoRA_MoE_net. "
                f"Got: {type(module)}"
            )

        payload_l = {
            "type": "LUNAR_LoRA_MoE_net",
            "input_dim": int(module.linear.weight.shape[1]),
            "output_dim": int(module.linear.weight.shape[0]),
            "n_experts": int(module.n_experts),
            "top_k": int(module.top_k),
            "gate_hidden": int(module.gate_hidden),
            "lora_r": int(module.lora_r),
            "lora_alpha": int(module.lora_alpha),
            "lora_dropout": float(module.lora_dropout.p),
            "pretrained_weight": module.linear.weight.detach().cpu(),

            "lora_A": module.lora_A.detach().cpu(),
            "lora_B": module.lora_B.detach().cpu(),
        }
        if module.gate_hidden > 0:
            payload_l["gate_weight_0"] = module.gate[0].weight.detach().cpu()
            payload_l["gate_bias_0"] = module.gate[0].bias.detach().cpu() if module.gate[0].bias is not None else None
            payload_l["gate_weight_2"] = module.gate[2].weight.detach().cpu()
            payload_l["gate_bias_2"] = module.gate[2].bias.detach().cpu() if module.gate[2].bias is not None else None
        else:
            payload_l["gate_weight"] = module.gate.weight.detach().cpu()
            payload_l["gate_bias"] = module.gate.bias.detach().cpu() if module.gate.bias is not None else None
        
        payload[l] = payload_l

    torch.save(payload, save_path)
    print(f"[SAVE] LUNAR MoE LoRA-only saved to {save_path}")


def load_lora_moe_only(
    model,
    load_path,
    layer_idxs=None,
    device="cuda",
    dtype=torch.bfloat16,
):
    payload = torch.load(load_path, map_location="cpu")
    use_layers = layer_idxs if layer_idxs is not None else list(payload.keys())

    for l in use_layers:
        if l not in payload:
            raise KeyError(f"Layer {l} not found in saved payload.")
        obj = payload[l]
        if obj.get("type", None) != "LUNAR_LoRA_MoE_net":
            raise ValueError(
                f"Layer {l} payload type mismatch. "
                f"Expected 'LUNAR_LoRA_MoE_net', got {obj.get('type', None)}"
            )

        pretrained_weight = obj["pretrained_weight"].to(device=device, dtype=dtype)

        lunar_moe = LUNAR_LoRA_MoE_net(
            input_dim=int(obj["input_dim"]),
            output_dim=int(obj["output_dim"]),
            n_experts=int(obj["n_experts"]),
            top_k=int(obj["top_k"]),
            gate_hidden=int(obj.get("gate_hidden", 0)),
            lora_rank=int(obj["lora_r"]),
            lora_alpha=int(obj["lora_alpha"]),
            lora_dropout=float(obj["lora_dropout"]),
            pretrained_weight=pretrained_weight,
        ).to(device=device, dtype=dtype)

        with torch.no_grad():
            lunar_moe.lora_A.copy_(obj["lora_A"].to(device=device, dtype=dtype))
            lunar_moe.lora_B.copy_(obj["lora_B"].to(device=device, dtype=dtype))
            if lunar_moe.gate_hidden > 0:
                lunar_moe.gate[0].weight.copy_(obj["gate_weight_0"].to(device=device, dtype=dtype))
                if obj.get("gate_bias_0") is not None:
                    lunar_moe.gate[0].bias.copy_(obj["gate_bias_0"].to(device=device, dtype=dtype))
                lunar_moe.gate[2].weight.copy_(obj["gate_weight_2"].to(device=device, dtype=dtype))
                if obj.get("gate_bias_2") is not None:
                    lunar_moe.gate[2].bias.copy_(obj["gate_bias_2"].to(device=device, dtype=dtype))
            else:
                lunar_moe.gate.weight.copy_(obj["gate_weight"].to(device=device, dtype=dtype))
                if obj.get("gate_bias") is not None:
                    lunar_moe.gate.bias.copy_(obj["gate_bias"].to(device=device, dtype=dtype))

        _set_down_proj(model, l, lunar_moe)

    print(f"[LOAD] LUNAR MoE LoRA-only loaded from {load_path}")
    return model


def expand_lora_moe_experts(
    model,
    layer_idxs,
    num_new_experts=1,
    init_std=0.01,
):
    estimated_net_list = [] # 빈 리스트에서 시작 
    new_expert_map = {}

    for l in layer_idxs:
        module = _get_down_proj(model, l)
        if not isinstance(module, LUNAR_LoRA_MoE_net):
            raise TypeError(
                f"Layer {l} down_proj is not LUNAR_LoRA_MoE_net. Got: {type(module)}"
            )

        new_ids = module.add_experts(
            num_new_experts=num_new_experts,
            init_std=init_std,
            update_top_k=False,
        )

        with torch.no_grad():
            module.lora_A.data[0].zero_() # retain용 0번 expert의 파라미터를 0으로 초기화 
            module.lora_B.data[0].zero_() # gate row 0은 유지(re-init 하지 않음)

        new_expert_map[l] = new_ids[-1] if len(new_ids) > 0 else None
        estimated_net_list.append(module)  
        # 업데이트된 expert 모듈을 종합하여 estimated_net_list를 업데이트
        print(f"[EXPAND] layer={l}, re-initialized retain expert 0, added forget expert={new_ids}, total_experts={module.n_experts}")
    return estimated_net_list, new_expert_map


#==============MoE with LoRA================
class LUNAR_NN_MoE_net(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_experts: int = 4,
        top_k: int = 1,
        gate_hidden: int = 0,
        pretrained_weight=None,
    ):
        super(LUNAR_NN_MoE_net, self).__init__()

        dtype = pretrained_weight.dtype if pretrained_weight is not None else torch.float32
        device = pretrained_weight.device if pretrained_weight is not None else "cuda"

        # frozen base linear
        self.linear = nn.Linear(input_dim, output_dim, bias=False, device=device, dtype=dtype)
        
        if pretrained_weight is not None:
            with torch.no_grad():
                self.linear.weight.copy_(pretrained_weight)

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.n_experts = int(n_experts)
        self.top_k = min(int(top_k), self.n_experts)

        self.scaling = 1.0 
        self.gate_hidden = int(gate_hidden)

        # experts -> W: (n_experts, out, in)
        self.expert_W = nn.Parameter(
            torch.randn(self.n_experts, output_dim, input_dim, device=device, dtype=dtype) * 0.01
        )

        with torch.no_grad(): # expert 0번만 zero로 설정
            self.expert_W[0].zero_()

        # gate
        if self.gate_hidden > 0:
            self.gate = nn.Sequential(
                nn.Linear(input_dim, self.gate_hidden, device=device, dtype=dtype),
                nn.GELU(),
                nn.Linear(self.gate_hidden, self.n_experts, device=device, dtype=dtype),
            )
        else:
            self.gate = nn.Linear(input_dim, self.n_experts, device=device, dtype=dtype)

    def _ensure_dtype(self, x):
        if x.dtype != self.linear.weight.dtype:
            x = x.to(self.linear.weight.dtype)
        return x

    def add_experts(
        self,
        num_new_experts: int,
        init_std: float = 0.01,
        update_top_k: bool = False,
        new_top_k: int | None = None,
    ):
        num_new_experts = int(num_new_experts)
        if num_new_experts <= 0:
            return []

        device = self.expert_W.device
        dtype = self.expert_W.dtype

        old_n = self.n_experts
        new_n = old_n + num_new_experts

        input_dim = self.expert_W.shape[-1]
        output_dim = self.expert_W.shape[1]

        # 1) NN expert 확장
        new_W = torch.randn(num_new_experts, output_dim, input_dim, device=device, dtype=dtype) * init_std

        with torch.no_grad():
            cat_W = torch.cat([self.expert_W.data, new_W], dim=0)

        self.expert_W = nn.Parameter(cat_W)

        # 2) gate 확장
        if self.gate_hidden > 0:
            if not isinstance(self.gate, nn.Sequential):
                raise TypeError("self.gate must be nn.Sequential when gate_hidden > 0.")
            old_gate_out = self.gate[-1]
            new_gate_out = nn.Linear(
                old_gate_out.in_features, new_n, bias=(old_gate_out.bias is not None),
                device=device, dtype=dtype
            )
            with torch.no_grad():
                new_gate_out.weight[:old_n].copy_(old_gate_out.weight.data)
                new_gate_out.weight[old_n:].normal_(mean=0.0, std=init_std)
                if old_gate_out.bias is not None:
                    new_gate_out.bias[:old_n].copy_(old_gate_out.bias.data)
                    new_gate_out.bias[old_n:].zero_()
            self.gate[-1] = new_gate_out
        else:
            if not isinstance(self.gate, nn.Linear):
                raise TypeError("self.gate must be nn.Linear when gate_hidden == 0.")
            old_gate = self.gate
            new_gate = nn.Linear(
                old_gate.in_features, new_n, bias=(old_gate.bias is not None),
                device=device, dtype=dtype,
            )
            with torch.no_grad():
                new_gate.weight[:old_n].copy_(old_gate.weight.data)
                new_gate.weight[old_n:].normal_(mean=0.0, std=init_std)
                if old_gate.bias is not None:
                    new_gate.bias[:old_n].copy_(old_gate.bias.data)
                    new_gate.bias[old_n:].zero_()
            self.gate = new_gate

        self.n_experts = new_n

        if update_top_k:
            if new_top_k is not None:
                self.top_k = min(int(new_top_k), self.n_experts)
            else:
                self.top_k = min(self.top_k, self.n_experts)
        else:
            self.top_k = min(self.top_k, self.n_experts)
        return list(range(old_n, new_n))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_gate: bool = False,
        return_router_logits: bool = False,
        return_all_experts: bool = False,
        forced_expert_idx: int | None = None,
    ):
        x = self._ensure_dtype(x)
        base = self.linear(x)
        logits = self.gate(x) 

        if forced_expert_idx is not None:
            if forced_expert_idx < 0 or forced_expert_idx >= self.n_experts:
                raise ValueError(f"forced_expert_idx={forced_expert_idx} out of range [0, {self.n_experts})")
            
            Wx_e = torch.einsum("...h,oh->...o", x, self.expert_W[forced_expert_idx])
            out = base + self.scaling * Wx_e
            
            if return_router_logits:
                return out, logits
            if return_gate:
                return out, F.softmax(logits, dim=-1)
            return out

        probs = F.softmax(logits, dim=-1)
        
        if self.top_k < self.n_experts:
            _, topi = torch.topk(probs, k=self.top_k, dim=-1)
            
            # 라우터가 선택한 expert 번호를 직접 기록
            if getattr(self, "record_expert", False):
                if not hasattr(self, "expert_history"):
                    self.expert_history = []
                self.expert_history.append(topi.detach().cpu())
                
            mask = torch.zeros_like(probs)
            mask.scatter_(-1, topi, 1.0)
            probs = probs * mask
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)

        Wx = torch.einsum("...h,eoh->...eo", x, self.expert_W)
        update = torch.einsum("...e,...eo->...o", probs, Wx)

        out = base + self.scaling * update

        res = [out]
        if return_router_logits:
            res.append(logits)
        if return_gate:
            res.append(probs)
        if return_all_experts:
            res.append(Wx)
            res.append(base)
        
        if len(res) == 1:
            return res[0]
        return tuple(res)


def save_nn_moe_only(
    model, 
    save_path, 
    layer_idxs=None
):
    payload = {}
    if layer_idxs is None:
        raise ValueError("layer_idxs must be provided")

    for l in layer_idxs:
        module = _get_down_proj(model, l)
        if not isinstance(module, LUNAR_NN_MoE_net):
            raise TypeError(f"Layer {l} is not LUNAR_NN_MoE_net.")

        payload_l = {
            "type": "LUNAR_NN_MoE_net",
            "input_dim": int(module.linear.weight.shape[1]),
            "output_dim": int(module.linear.weight.shape[0]),
            "n_experts": int(module.n_experts),
            "top_k": int(module.top_k),
            "gate_hidden": int(module.gate_hidden),
            "pretrained_weight": module.linear.weight.detach().cpu(),
            "expert_W": module.expert_W.detach().cpu(),
        }
        if module.gate_hidden > 0:
            payload_l["gate_weight_0"] = module.gate[0].weight.detach().cpu()
            payload_l["gate_bias_0"] = module.gate[0].bias.detach().cpu() if module.gate[0].bias is not None else None
            payload_l["gate_weight_2"] = module.gate[2].weight.detach().cpu()
            payload_l["gate_bias_2"] = module.gate[2].bias.detach().cpu() if module.gate[2].bias is not None else None
        else:
            payload_l["gate_weight"] = module.gate.weight.detach().cpu()
            payload_l["gate_bias"] = module.gate.bias.detach().cpu() if module.gate.bias is not None else None
        
        payload[l] = payload_l

    torch.save(payload, save_path)
    print(f"[SAVE] LUNAR MoE NN-only saved to {save_path}")


def load_nn_moe_only(
    model, 
    load_path, 
    layer_idxs=None, 
    device="cuda", 
    dtype=torch.bfloat16
):
    payload = torch.load(load_path, map_location="cpu")
    use_layers = layer_idxs if layer_idxs is not None else list(payload.keys())

    for l in use_layers:
        if l not in payload:
            raise KeyError(f"Layer {l} not found in saved payload.")
        obj = payload[l]
        if obj.get("type", None) != "LUNAR_NN_MoE_net":
            raise ValueError(f"Expected 'LUNAR_NN_MoE_net', got {obj.get('type')}")

        pretrained_weight = obj["pretrained_weight"].to(device=device, dtype=dtype)

        lunar_moe = LUNAR_NN_MoE_net(
            input_dim=int(obj["input_dim"]),
            output_dim=int(obj["output_dim"]),
            n_experts=int(obj["n_experts"]),
            top_k=int(obj["top_k"]),
            gate_hidden=int(obj.get("gate_hidden", 0)),
            pretrained_weight=pretrained_weight,
        ).to(device=device, dtype=dtype)

        with torch.no_grad():
            lunar_moe.expert_W.copy_(obj["expert_W"].to(device=device, dtype=dtype))
            if lunar_moe.gate_hidden > 0:
                lunar_moe.gate[0].weight.copy_(obj["gate_weight_0"].to(device=device, dtype=dtype))
                if obj.get("gate_bias_0") is not None:
                    lunar_moe.gate[0].bias.copy_(obj["gate_bias_0"].to(device=device, dtype=dtype))
                lunar_moe.gate[2].weight.copy_(obj["gate_weight_2"].to(device=device, dtype=dtype))
                if obj.get("gate_bias_2") is not None:
                    lunar_moe.gate[2].bias.copy_(obj["gate_bias_2"].to(device=device, dtype=dtype))
            else:
                lunar_moe.gate.weight.copy_(obj["gate_weight"].to(device=device, dtype=dtype))
                if obj.get("gate_bias") is not None:
                    lunar_moe.gate.bias.copy_(obj["gate_bias"].to(device=device, dtype=dtype))

        _set_down_proj(model, l, lunar_moe)

    print(f"[LOAD] LUNAR MoE NN-only loaded from {load_path}")
    return model

def expand_nn_moe_experts(
    model, 
    layer_idxs, 
    num_new_experts=1, 
    init_std=0.01
):
    estimated_net_list = []
    new_expert_map = {}

    for l in layer_idxs:
        module = _get_down_proj(model, l)
        if not isinstance(module, LUNAR_NN_MoE_net):
            raise TypeError(f"Layer {l} down_proj is not LUNAR_NN_MoE_net. Got: {type(module)}")

        new_ids = module.add_experts(
            num_new_experts=num_new_experts,
            init_std=init_std,
            update_top_k=False,
        )

        with torch.no_grad():
            module.expert_W.data[0].zero_()

        new_expert_map[l] = new_ids[-1] if len(new_ids) > 0 else None
        estimated_net_list.append(module)  
        print(f"[EXPAND] layer={l}, re-initialized retain expert 0, added forget expert={new_ids}, total_experts={module.n_experts}")
    return estimated_net_list, new_expert_map