#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python}"
# SCRIPT="${SCRIPT:-eval_only.py}"
SCRIPT="${SCRIPT:-lunar_eval_only.py}"
CFG_NAME="${CFG_NAME:-forget.yaml}"  

# fixed model
MODEL_FAMILY="llama3.1-8b-instruct"
BASE_MODEL_PATH="./train31Llama/trained_model/O3_LLAMA31_ScienceQA"

SD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/scienceqa_SD_10"
RD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/scienceqa_RD_10"
UTILITY1="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/commonqa/commonqa_test.json"
UTILITY2="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/openbookqa/openbookqa_test.json"

# 누적에 사용할 도메인 순서 (원하는 순서대로)
DOMAINS=(
  "biology"
  "figurative-language"
  "units-and-measurement"
  "physics"
  "reference-skills"
  "chemistry"
  "punctuation"
  "economics"
  "grammar"
  "earth-science"
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

    joined="$(IFS=_; echo "${ACC[*]}")"          # biology_figurative-language_...
    task_id="scienceqa_t${t}_${domain}"

    #############
    data_path1="${SD_DIR}/scienceqa_${joined}_train_SD.json"
    data_path2="${RD_DIR}/scienceqa_not_${joined}_train_RD.json"
    #############
  
    sampled_data_path="${SD_DIR}/scienceqa_${joined}_test_SD.json"
    retain_data_path="${RD_DIR}/scienceqa_not_${joined}_test_RD.json"

    echo "---- step=${t}"
    echo " domains           : ${joined}"
    echo " sampled_data_path  : ${sampled_data_path}"
    echo " retain_data_path   : ${retain_data_path}"
    echo

    ${PY} ${SCRIPT} \
      --config-name "${CFG_NAME}" \
      data_name="scienceqa" \
      model_family="${MODEL_FAMILY}" \
      base_model_path="${BASE_MODEL_PATH}" \
      model_path="${BASE_MODEL_PATH}" \
      seed="${SEED}" \
      task_id="${task_id}" \
      data_path="${data_path1}" \
      r_data_path="${data_path2}" \
      sampled_data_path="${sampled_data_path}" \
      retain_data_path="${retain_data_path}" \
      utiliy_data_path1="${UTILITY1}" \
      utiliy_data_path2="${UTILITY2}" \
      save_folder="lunar_${MODEL_FAMILY}_eval_base_${task_id}" 
    echo
  done
done

echo "[DONE] All cumulative eval runs finished."