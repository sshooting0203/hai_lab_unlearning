import torch
import torch.nn as nn
import torch.nn.functional as F
from src_lunar.estimated_net_utils import _get_down_proj, _set_down_proj

# experts + gate
# gate의 역할 : 입력 activation x를 보고 expert K개 중 
# 어떤 expert를 쓸지 점수(logit)를 만든 다음 softmax로 확률로 바꿈
# token단위의 gating? VS sequence단위의 gating?
# token단위 gating이 더 세밀하게 전문가 선택 가능, sequence단위 gating은 전체 시퀀스에 대해 하나의 전문가 선택 
# 지금은 sequence단위 gating을 선택

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

        self.lora_r = int(lora_rank)
        self.lora_alpha = int(lora_alpha)
        self.scaling = self.lora_alpha / self.lora_r
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.gate_hidden = int(gate_hidden)

        # experts
        # A[e]: (r, in), B[e]: (out, r)
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

        if self.gate_hidden > 0:
            raise NotImplementedError(
                "add_experts currently supports only linear gate (gate_hidden=0)."
            )
        if not isinstance(self.gate, nn.Linear):
            raise TypeError("self.gate must be nn.Linear when using add_experts().")

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
        old_gate = self.gate
        new_gate = nn.Linear(
            old_gate.in_features,
            new_n,
            bias=(old_gate.bias is not None),
            device=device,
            dtype=dtype,
        )

        with torch.no_grad():
            # 기존 expert 부분 복사
            new_gate.weight[:old_n].copy_(old_gate.weight.data)
            # 새 expert 부분 초기화
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
        forced_expert_idx: int | None = None,
    ):
        x = self._ensure_dtype(x)
        base = self.linear(x)
        logits = self.gate(x)  # 1) router logits 계산

        if forced_expert_idx is not None: # train시에 특정 expert로 강제 force forward
            if forced_expert_idx < 0 or forced_expert_idx >= self.n_experts:
                raise ValueError(
                    f"forced_expert_idx={forced_expert_idx} out of range "
                    f"[0, {self.n_experts})"
                )
            x_d = self.lora_dropout(x)
            Ax_e = torch.einsum(
                "...h,rh->...r", x_d, self.lora_A[forced_expert_idx]
            )
            BAx_e = torch.einsum(
                "...r,or->...o", Ax_e, self.lora_B[forced_expert_idx]
            )
            out = base + self.scaling * BAx_e
            if return_router_logits:
                return out, logits
            if return_gate:
                return out, F.softmax(logits, dim=-1)
            return out

        probs = F.softmax(logits, dim=-1)  # 2) router softmax 확률 계산 
        top1 = probs.argmax(dim=-1)

        if self.top_k < self.n_experts:
            # print("[DEBUG] Applying top-k gating: top_k =", self.top_k)
            _, topi = torch.topk(probs, k=self.top_k, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(-1, topi, 1.0)
            probs = probs * mask
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9) # probs 정규화([0,1])
            # print("[DEBUG] probs : ", probs)

        x_d = self.lora_dropout(x)
        Ax = torch.einsum("...h,erh->...er", x_d, self.lora_A)  
        BAx = torch.einsum("...er,eor->...eo", Ax, self.lora_B)  
        update = torch.einsum("...e,...eo->...o", probs, BAx)   
        # fully differentiable -> activate sparse expert top-k

        out = base + self.scaling * update
        if return_router_logits:
            return out, logits
        if return_gate:
            return out, probs
        return out
    

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

        if not isinstance(module.gate, nn.Linear):
            raise NotImplementedError(
                "save_lora_moe_only currently supports only linear gate."
            )

        payload[l] = {
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

            "gate_weight": module.gate.weight.detach().cpu(),
            "gate_bias": None if module.gate.bias is None else module.gate.bias.detach().cpu(),
        }

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
            lunar_moe.gate.weight.copy_(obj["gate_weight"].to(device=device, dtype=dtype))

            if obj["gate_bias"] is not None:
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

        # Re-initialize the shared retain expert (index 0)
        with torch.no_grad():
            module.lora_A.data[0].zero_()
            module.lora_B.data[0].zero_()
        #  gate row 0은 유지: re-init 하지 않음

        new_expert_map[l] = new_ids[-1] if len(new_ids) > 0 else None
        estimated_net_list.append(module)  # 업데이트된 expert 모듈을 종합하여 estimated_net_list를 업데이트
        print(f"[EXPAND] layer={l}, re-initialized retain expert 0, added forget expert={new_ids}, total_experts={module.n_experts}")
    return estimated_net_list, new_expert_map


# 첫 번째:
# 모든 expert 출력을 계산한 뒤, gate 확률로 섞는 dense mixture형 LoRA-MoE

# 두 번째:
# top-k로 선택된 expert에 대해서만 토큰별로 실제 연산하는 sparse routed MoE형 LoRA

    # def forward(self, x, return_gate=False):
    #     x = self._ensure_dtype(x)
    #     base = self.linear(x)

    #     # gate logits computed per token/row
    #     logits = self.gate(x)  # (..., n_experts)
    #     probs = F.softmax(logits, dim=-1)  # (..., n_experts)

    #     # top-k masking (keeps differentiability on selected experts)
    #     if self.top_k < self.n_experts:
    #         topv, topi = torch.topk(probs, k=self.top_k, dim=-1)
    #         mask = torch.zeros_like(probs)
    #         mask.scatter_(-1, topi, 1.0)
    #         probs = probs * mask
    #         probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)

    #     x_d = self.lora_dropout(x)  # (..., in)

    #     # compute each expert update: (..., out)
    #     # Ax: for each expert e => (..., r)
    #     # Use einsum to vectorize: 
    #     #   x_d (..., in)
    #     #   A (E, r, in) -> Ax (..., E, r)
    #     Ax = torch.einsum("...i,eri->...er", x_d, self.lora_A)  # (..., E, r)
    #     #   B (E, out, r) -> B(Ax): (..., E, out)
    #     BAx = torch.einsum("...er,eor->...eo", Ax, self.lora_B) # (..., E, out)

    #     # mixture: sum_e probs_e * update_e
    #     update = torch.einsum("...e,...eo->...o", probs, BAx)   # (..., out)

    #     out = base + self.scaling * update
    #     if return_gate:
    #         return out, probs
    #     return out

    # def forward(self, x: torch.Tensor):
        # previous_dtype = x.dtype

        # ##newmodified
        # ### modified2 elif self.r[self.active_adapter] > 0 and not self.merged:

        # result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        # batch_size, sequence_length, hidden_dim = x.shape
        # x = x.view(-1, hidden_dim)
        # # router_logits = self.router['router'](x)
        # x_router = x.to(self.router['router'].weight.dtype) # @ 수정
        # router_logits = self.router['router'](x_router)
        # # router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        # routing_weights_before = F.softmax(router_logits, dim=1, dtype=torch.float)
        # routing_weights, selected_experts = torch.topk(routing_weights_before, self.top_k, dim=-1)
        # routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # routing_weights = routing_weights.to(x.dtype)

        # expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.number_experts).permute(2, 1, 0)

        # final_hidden_states = torch.zeros(
        #     (batch_size * sequence_length, self.out_features), dtype=x.dtype, device=x.device
        # )

        # for expert_idx in range(self.number_experts):
        #     adapter_name_moe = self.active_adapter + '_' + str(expert_idx)
        #     idx, top_x = torch.where(expert_mask[expert_idx])

        #     if top_x.shape[0] == 0:
        #         continue

        #     # in torch it is faster to index using lists than torch tensors
        #     top_x_list = top_x.tolist()
        #     idx_list = idx.tolist()

        #     # Index the correct hidden states and compute the expert hidden state for
        #     # the current expert. We need to make sure to multiply the output hidden
        #     # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        #     current_state = x[None, top_x_list].reshape(-1, hidden_dim)
        #     expert_output = (
        #             self.lora_B[adapter_name_moe](
        #                 self.lora_A[adapter_name_moe](self.lora_dropout[self.active_adapter](current_state))
        #             )
        #             * self.scaling[self.active_adapter]
        #     )
        #     current_hidden_states = expert_output * routing_weights[top_x_list, idx_list, None] # @ 수식 : 여기서 expert output과 routing network 결과 합쳐서 도출 

        #     # However `index_add_` only support torch tensors for indexing so we'll use
        #     # the `top_x` tensor here.
        #     final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))

        #     # x = x.to(self.lora_A[adapter_name_moe].weight.dtype)

        #     # expert_Gating = router_probabilities[:,:,i].unsqueeze(-1)
        #     # expert_o = expert_Gating * expert_output
        #     # result += router_probabilities[:,:,i].unsqueeze(-1) * expert_output

        # final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, self.out_features)
        # result += final_hidden_states
        # result = result.to(previous_dtype)
        # return result, router_logits # @ 수정