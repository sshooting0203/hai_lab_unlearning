#!/usr/bin/env bash
set -e  # 중간 에러 시 즉시 종료

# -----------------------
# 설정
# -----------------------
MODEL="meta-llama/Llama-3.1-8B"
DATA_ROOT="../data/scienceqa"
STATS="../stats_pre_full.pt"

SUBJECTS=(
  biology
  chemistry
  earth-science
  # figurative-language
  # grammar
  physics
  punctuation
  # reference-skills
  units-and-measurement
  # writing-strategies
)

WINDOWS=(
  "8 15"
  # "12 15"
  "16 23"
  # "20 23"
  "24 31"
  # "28 31"
)

# -----------------------
# 반복 실행
# -----------------------
for SUBJECT in "${SUBJECTS[@]}"; do
  for W in "${WINDOWS[@]}"; do
    read WINDOW_S WINDOW_E <<< "$W"

    echo "=============================================="
    echo "SUBJECT=${SUBJECT}, WINDOW=(${WINDOW_S}, ${WINDOW_E})"
    echo "=============================================="

    OUT_DIR="./out_lora_${SUBJECT}_${WINDOW_S}_${WINDOW_E}"

    # -----------------------
    # Train
    # -----------------------
    python train_lora_random.py \
      --model ${MODEL} \
      --train_jsonl ${DATA_ROOT}/splits_${SUBJECT}_pred/train_pred_cache.jsonl \
      --stats ${STATS} \
      --out_dir ${OUT_DIR} \
      --lora_layer_start ${WINDOW_S} \
      --lora_layer_end ${WINDOW_E} \
      --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
      --lam_repel 0.0 \
      --lora_r 64 \
      --lora_alpha 32 \
      --lora_dropout 0.05 \
      --lr 1e-4 \
      --batch_size 4 \
      --grad_accum 4 \
      --max_steps 2000 \
      --device cuda \
      --dtype float16

    # -----------------------
    # Eval
    # -----------------------
    python eval_lora_acc.py \
      --base ${MODEL} \
      --lora ${OUT_DIR}/lora_adapter \
      --jsonl ${DATA_ROOT}/splits_${SUBJECT}_pred/test_pred_cache.jsonl \
      --out_jsonl ./eval_${SUBJECT}_${WINDOW_S}_${WINDOW_E}.jsonl \
      --print_samples 5

  done
done
