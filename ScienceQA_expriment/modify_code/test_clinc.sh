#!/usr/bin/env bash
set -euo pipefail

# Config
PY="${PY:-python}"
SCRIPT="${SCRIPT:-analyze.py}"   # 네 분석 코드 파일명으로 맞춰
CFG_NAME="${CFG_NAME:-forget_analyze}"  # hydra config_name

# base model
MODEL_FAMILY="llama3.1-8b-instruct"
BASE_MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"

# output root (task별/seed별로 하위 폴더 생성)
OUT_ROOT="${OUT_ROOT:-/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code/forget_analyze_runs_clinc/llama3.1-8b-instruct}"

# dataset roots
BASE_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data/clincqa"
SD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data/clincqa_SD_only_10"
RD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data/clincqa_RD_only_10"

# 11 domains
DOMAINS=(
  "work"
  "travel"
  "home"
  "banking"
  "credit_cards"
  "kitchen_and_dining"
  "auto_and_commute"
  "utility"
  "small_tack"
  "meta"
)

mkdir -p "${OUT_ROOT}"

# Run
for SEED in 0 1 2; do
  echo "================ SEED=${SEED} ================"

  for domain in "${DOMAINS[@]}"; do
    # ---------------------------
    # train files (mu split용: split_raw_dataset_for_forget)
    # ---------------------------
    data_name="${BASE_DIR}/clincqa_${domain}_train.json"

    # ---------------------------
    # eval files (separation 계산용)
    # ---------------------------
    dist_forget_path="${SD_DIR}/clincqa_${domain}_test_SD.json"
    dist_retain_path="${RD_DIR}/clincqa_not_${domain}_test_RD.json"

    # output dir
    run_dir="${OUT_ROOT}/seed${SEED}/${domain}"
    mkdir -p "${run_dir}"

    echo "---- domain=${domain}"
    echo " data_name        : ${data_name}"
    echo " dist_forget_path : ${dist_forget_path}"
    echo " dist_retain_path : ${dist_retain_path}"
    echo " out              : ${run_dir}"
    echo

    ${PY} ${SCRIPT} \
      --config-name "${CFG_NAME}" \
      model_family="${MODEL_FAMILY}" \
      base_model_path="${BASE_MODEL_PATH}" \
      model_path="${BASE_MODEL_PATH}" \
      seed="${SEED}" \
      data_name="${data_name}" \
      dist_forget_path="${dist_forget_path}" \
      dist_retain_path="${dist_retain_path}" \
      svdd_save_dir="${run_dir}" \
      dist_batch_size=8 \
      dist_holdout_ratio=0.3 \
      num_workers=0
    echo
  done
done

echo "[DONE] All runs finished. Results under: ${OUT_ROOT}"
