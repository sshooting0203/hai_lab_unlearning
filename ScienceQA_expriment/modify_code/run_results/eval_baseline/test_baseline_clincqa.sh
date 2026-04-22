#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python}"
# SCRIPT="${SCRIPT:-eval_only.py}"
SCRIPT="${SCRIPT:-lunar_eval_only.py}"
CFG_NAME="${CFG_NAME:-forget.yaml}"  

MODEL_FAMILY="llama3.1-8b-instruct"
BASE_MODEL_PATH="./train31Llama/trained_model/O3_LLAMA31_ClincQA"
# dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/O3_LLAMA31_ClincQA

SD_DIR=/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/clincqa_SD 
RD_DIR=/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/clincqa_RD 
UTILITY1="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/mrpcqa/mrpc_test.json"
UTILITY2="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/rteqa/rte_validation.json"

DOMAINS=(
  "work"
  "travel"
  "home"
  "banking"
  "credit_cards"
  "kitchen_and_dining"
  "auto_and_commute"
  "utility"
  "small_talk"
  # "meta"
)

# seed loop (필요한 것만)
SEEDS=(0)

for SEED in "${SEEDS[@]}"; do
  echo "================ SEED=${SEED} ================"

  ACC=()   # 누적 도메인
  t=0

  for domain in "${DOMAINS[@]}"; do
    ACC+=("${domain}")
    t=$((t+1))

    joined="$(IFS=_; echo "${ACC[*]}")"         
    task_id="clincqa_t${t}_${domain}"

    #############
    data_path1="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/clincqa_SD/clincqa_${joined}_train_SD.json" 
    data_path2="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/clincqa_RD/clincqa_not_${joined}_train_RD.json" 
    #############

    sampled_data_path="${SD_DIR}/clincqa_${joined}_test_SD.json" 
    retain_data_path="${RD_DIR}/clincqa_not_${joined}_test_RD.json" 

    echo "---- step=${t}"
    echo " domains           : ${joined}"
    echo " sampled_data_path  : ${sampled_data_path}"
    echo " retain_data_path   : ${retain_data_path}"
    echo

    ${PY} ${SCRIPT} \
      --config-name "${CFG_NAME}" \
      data_name="clincqa" \
      model_family="${MODEL_FAMILY}" \
      base_model_path="${BASE_MODEL_PATH}" \
      model_path="${BASE_MODEL_PATH}" \
      seed="${SEED}" \
      task_id="${task_id}" \
      # data_path="${data_path1}" \
      # r_data_path="${data_path2}" \
      sampled_data_path="${sampled_data_path}" \
      retain_data_path="${retain_data_path}" \
      utiliy_data_path1="${UTILITY1}" \
      utiliy_data_path2="${UTILITY2}" \
      eval_generation_max_length=512 \
      eval_generation_max_new_tokens=512 \
      save_folder="clincqa/lunar_${MODEL_FAMILY}_eval_base_${task_id}" 
    echo
  done
done

echo "[DONE] All cumulative eval runs finished."