"""
MoE 구조에서 expert 1개 추가 시 파라미터 증가량 계산 스크립트

두 가지 MoE variant를 모두 분석:
  1. LUNAR_LoRA_MoE_net  – LoRA 기반 expert (lora_A, lora_B)
  2. LUNAR_NN_MoE_net    – Full-NN expert (expert_W)

gate 타입별 (linear / dense) 증가분도 함께 계산합니다.
"""

import argparse


def fmt(n: int) -> str:
    """천 단위 콤마 포맷"""
    return f"{n:,}"


# ─────────────────────────── 공식 ───────────────────────────
#
# [LoRA MoE]
#   expert 파라미터:
#     lora_A[e]: (r, input_dim)  →  r × input_dim
#     lora_B[e]: (output_dim, r) →  output_dim × r
#     총: r × input_dim + output_dim × r = r × (input_dim + output_dim)
#
#   gate 파라미터 (expert 1개 추가 → gate output이 n→n+1):
#     linear gate:
#       weight 행 1줄 추가: input_dim
#       bias   1개 추가:    1
#       총: input_dim + 1
#     dense gate (hidden_dim=H):
#       첫 번째 Linear(input_dim→H)는 변화 없음
#       두 번째 Linear(H→n_experts)에 행 1줄 + bias 1 추가:
#       총: H + 1
#
# [NN MoE]
#   expert 파라미터:
#     expert_W[e]: (output_dim, input_dim) → output_dim × input_dim
#
#   gate 증가분은 LoRA MoE와 동일
# ────────────────────────────────────────────────────────────


def calc_lora_moe(input_dim: int, output_dim: int, lora_rank: int,
                  gate_hidden: int, n_layers: int):
    expert_params = lora_rank * (input_dim + output_dim)
    if gate_hidden > 0:
        gate_params = gate_hidden + 1        # Linear(H→n+1) 행 1줄 + bias
    else:
        gate_params = input_dim + 1          # Linear(in→n+1) 행 1줄 + bias
    per_layer = expert_params + gate_params
    total = per_layer * n_layers

    print("=" * 60)
    print("  LUNAR_LoRA_MoE_net  –  Expert 1개 추가 시 파라미터 증가량")
    print("=" * 60)
    print(f"  input_dim   = {fmt(input_dim)}")
    print(f"  output_dim  = {fmt(output_dim)}")
    print(f"  lora_rank   = {lora_rank}")
    print(f"  gate_hidden = {gate_hidden}  ({'Dense gate' if gate_hidden > 0 else 'Linear gate'})")
    print(f"  n_layers    = {n_layers}")
    print("-" * 60)
    print(f"  LoRA expert params (per layer):")
    print(f"    lora_A : {lora_rank} × {fmt(input_dim)} = {fmt(lora_rank * input_dim)}")
    print(f"    lora_B : {fmt(output_dim)} × {lora_rank} = {fmt(output_dim * lora_rank)}")
    print(f"    소계   : {fmt(expert_params)}")
    print(f"  Gate 증가분 (per layer):")
    if gate_hidden > 0:
        print(f"    Linear(H→n+1) weight 행 1줄 : {fmt(gate_hidden)}")
        print(f"    Linear(H→n+1) bias 1개      : 1")
    else:
        print(f"    Linear(in→n+1) weight 행 1줄 : {fmt(input_dim)}")
        print(f"    Linear(in→n+1) bias 1개      : 1")
    print(f"    소계   : {fmt(gate_params)}")
    print("-" * 60)
    print(f"  Per-layer 총 증가 : {fmt(per_layer)}")
    print(f"  전체 ({n_layers} layers)   : {fmt(total)}")
    print("=" * 60)
    return total


def calc_nn_moe(input_dim: int, output_dim: int,
                gate_hidden: int, n_layers: int):
    expert_params = output_dim * input_dim
    if gate_hidden > 0:
        gate_params = gate_hidden + 1
    else:
        gate_params = input_dim + 1
    per_layer = expert_params + gate_params
    total = per_layer * n_layers

    print()
    print("=" * 60)
    print("  LUNAR_NN_MoE_net  –  Expert 1개 추가 시 파라미터 증가량")
    print("=" * 60)
    print(f"  input_dim   = {fmt(input_dim)}")
    print(f"  output_dim  = {fmt(output_dim)}")
    print(f"  gate_hidden = {gate_hidden}  ({'Dense gate' if gate_hidden > 0 else 'Linear gate'})")
    print(f"  n_layers    = {n_layers}")
    print("-" * 60)
    print(f"  NN expert params (per layer):")
    print(f"    expert_W : {fmt(output_dim)} × {fmt(input_dim)} = {fmt(expert_params)}")
    print(f"  Gate 증가분 (per layer):")
    if gate_hidden > 0:
        print(f"    Linear(H→n+1) weight 행 1줄 : {fmt(gate_hidden)}")
        print(f"    Linear(H→n+1) bias 1개      : 1")
    else:
        print(f"    Linear(in→n+1) weight 행 1줄 : {fmt(input_dim)}")
        print(f"    Linear(in→n+1) bias 1개      : 1")
    print(f"    소계   : {fmt(gate_params)}")
    print("-" * 60)
    print(f"  Per-layer 총 증가 : {fmt(per_layer)}")
    print(f"  전체 ({n_layers} layers)   : {fmt(total)}")
    print("=" * 60)
    return total


def main():
    parser = argparse.ArgumentParser(
        description="MoE expert 1개 추가 시 파라미터 증가량 계산"
    )
    parser.add_argument("--input_dim", type=int, default=4096,
                        help="down_proj input dim (= intermediate_size). "
                             "Llama-3.1-8B: 14336, Qwen2-7B: 18944")
    parser.add_argument("--output_dim", type=int, default=4096,
                        help="down_proj output dim (= hidden_size). "
                             "Llama-3.1-8B: 4096, Qwen2-7B: 3584")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank (LoRA MoE 전용)")
    parser.add_argument("--gate_hidden", type=int, default=0,
                        help="Dense gate hidden dim (0이면 Linear gate)")
    parser.add_argument("--n_layers", type=int, default=1,
                        help="MoE를 적용한 레이어 수")
    parser.add_argument("--model", type=str, default=None,
                        choices=["llama3-8b", "qwen2-7b"],
                        help="모델 프리셋 (input_dim/output_dim 자동 설정)")
    args = parser.parse_args()

    # 모델 프리셋
    if args.model == "llama3-8b":
        args.input_dim = 14336
        args.output_dim = 4096
    elif args.model == "qwen2-7b":
        args.input_dim = 18944
        args.output_dim = 3584

    calc_lora_moe(args.input_dim, args.output_dim, args.lora_rank,
                  args.gate_hidden, args.n_layers)
    calc_nn_moe(args.input_dim, args.output_dim,
                args.gate_hidden, args.n_layers)

    # ── 비교표: gate_hidden 별 ──
    print("\n")
    print("=" * 70)
    print("  [비교표] gate_hidden 별 expert 1개 추가 시 파라미터 증가량")
    print(f"  (input_dim={fmt(args.input_dim)}, output_dim={fmt(args.output_dim)}, "
          f"lora_rank={args.lora_rank}, n_layers={args.n_layers})")
    print("=" * 70)
    header = f"  {'gate_hidden':>12} | {'LoRA expert':>14} | {'Gate Δ':>12} | {'LoRA 총Δ':>14} | {'NN expert':>14} | {'NN 총Δ':>14}"
    print(header)
    print("  " + "-" * len(header.strip()))

    for gh in [0, 64, 128, 256, 512]:
        lora_expert = args.lora_rank * (args.input_dim + args.output_dim)
        nn_expert = args.output_dim * args.input_dim
        gate_delta = (gh + 1) if gh > 0 else (args.input_dim + 1)
        lora_total = (lora_expert + gate_delta) * args.n_layers
        nn_total = (nn_expert + gate_delta) * args.n_layers
        print(f"  {gh:>12} | {fmt(lora_expert):>14} | {fmt(gate_delta):>12} | "
              f"{fmt(lora_total):>14} | {fmt(nn_expert):>14} | {fmt(nn_total):>14}")

    print("=" * 70)


if __name__ == "__main__":
    main()
