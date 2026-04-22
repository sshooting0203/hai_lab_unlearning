#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1 nohup bash test_lunar_tofu.sh > run_logs/test_lunar_tofu.log 2>&1 &

set -euo pipefail

PY="${PY:-python}"
# SCRIPT="${SCRIPT:-eval_only.py}"
SCRIPT="${SCRIPT:-lunar_eval_only_tofu.py}"
CFG_NAME="${CFG_NAME:-forget_tofu}"  

MODEL_FAMILY="llama3.1-8b-instruct"
BASE_MODEL_PATH="./train31Llama/trained_model/O3_LLAMA31_TOFU"

SD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ntofu_SD_only_20"
RD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ntofu_RD_only_20"
# main_retain_path="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ntofu_SD_only_20/tofu_AG20_SD.json"
# 최소한의 retain 데이터셋(A20 그룹에 해당하는 저자들의 정보, context를 유지시키기 위한 목적)
UTILITY1="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/tofu_real_authors/real_authors_test.json"
UTILITY2="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/tofu_world_facts/world_facts_test.json"

DOMAINS=(
  "AG01" "AG02" "AG03" "AG04" "AG05"
  "AG06" "AG07" "AG08" "AG09" "AG10"
  "AG11" "AG12" "AG13" "AG14" "AG15"
  "AG16" "AG17" "AG18" "AG19" 
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
    data_path1="${SD_DIR}/tofu_${domain}_SD.json"
    data_path2="${RD_DIR}/tofu_not_${domain}_RD.json"
    #############

    sampled_data_path="${SD_DIR}/tofu_${domain}_SD.json" # @@
    retain_data_path="${RD_DIR}/tofu_not_${domain}_RD.json" # @@

    echo "---- step=${t}"
    echo " domains           : ${domain}"
    echo " sampled_data_path  : ${sampled_data_path}"
    echo " retain_data_path   : ${retain_data_path}"
    echo

    ${PY} ${SCRIPT} \
      --config-name "${CFG_NAME}" \
      data_name="tofu" \
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
      eval_generation_max_length=512 \
      eval_generation_max_new_tokens=512 \
      save_folder="test_lunar_tofu_3/lunar_${MODEL_FAMILY}_eval_base_${task_id}" 
    echo
  done
done

echo "[DONE] All cumulative eval runs finished."