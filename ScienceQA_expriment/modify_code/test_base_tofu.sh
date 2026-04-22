#!/usr/bin/env bash
# <TOFU Exception>
# tofu는 utility dataset을 마지막 1개의 그룹의 저자로 구성(우선적으로 19개의 그룹에 대해서 제거)
# 20줄 x 1그룹(10명) = 200 samples
# CUDA_VISIBLE_DEVICES=0 nohup bash test_base_tofu.sh > run_logs/test_base_tofu.log 2>&1 &

set -euo pipefail

PY="${PY:-python}"
# SCRIPT="${SCRIPT:-eval_only.py}"
SCRIPT="${SCRIPT:-lunar_eval_only_tofu2.py}"
CFG_NAME="${CFG_NAME:-forget}"  

MODEL_FAMILY="llama3.1-8b-instruct"
BASE_MODEL_PATH="./train31Llama/trained_model/O3_LLAMA31_TOFU"

SD_DIR_ONLY="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ptofu_SD_only"
SD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ptofu_SD"
RD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ptofu_RD"

UTILITY1="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/tofu_real_authors/real_authors_test.json"
UTILITY2="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/tofu_world_facts/world_facts_test.json"
SAVE_FOLDER="tofu_base_3_28"

DOMAINS=(
  "AG01" "AG02" "AG03" "AG04" "AG05" "AG06" "AG07" "AG08" "AG09" "AG10"
  "AG11" "AG12" "AG13" "AG14" "AG15" "AG16" "AG17" "AG18" "AG19" "AG20"
  "AG21" "AG22" "AG23" "AG24" "AG25" "AG26"
)

SEEDS=(0)

for SEED in "${SEEDS[@]}"; do
  echo "================ SEED=${SEED} ================"

  ACC=()   # 누적 도메인
  step_idx=0
  for domain in "${DOMAINS[@]}"; do
    ACC+=("${domain}")

    # 3-digit padded step number (001, 002, ...)
    step_num=$(printf "%03d" $((step_idx + 1)))
    task_id="scienceqa_t${step_idx}_${domain}"

    # 현재 step 데이터 (Train)
    data_path="${SD_DIR_ONLY}/tofu_${domain}_SD.json"
    
    # r_data_path: cumulative retain data for training
    r_data_path="${RD_DIR}/tofu_step${step_num}_RD.json"

    # Evaluation용 데이터
    sampled_data_path="${SD_DIR}/tofu_step${step_num}_SD.json"
    retain_data_path="${RD_DIR}/tofu_step${step_num}_RD.json"

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
      data_path="${data_path}" \
      r_data_path="${r_data_path}" \
      sampled_data_path="${sampled_data_path}" \
      retain_data_path="${retain_data_path}" \
      utiliy_data_path1="${UTILITY1}" \
      utiliy_data_path2="${UTILITY2}" \
      eval_generation_max_length=512 \
      eval_generation_max_new_tokens=512 \
      save_folder="${SAVE_FOLDER}/lunar_${MODEL_FAMILY}_eval_base_${task_id}" 
    echo
    step_idx=$((step_idx + 1))
  done
done

echo "[DONE] All cumulative eval runs finished."