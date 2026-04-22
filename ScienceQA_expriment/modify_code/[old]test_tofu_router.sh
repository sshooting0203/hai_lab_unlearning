#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=2 nohup bash test_tofu_router.sh > run_logs/test_tofu_router.log 2>&1 &

set -euo pipefail

PY="${PY:-python}"
SCRIPT="${SCRIPT:-mmain_tofu_fixed_only.py}"
PLOT_SCRIPT="${PLOT_SCRIPT:-plot_results_tofu.py}"
CFG_NAME="${CFG_NAME:-forget}"

# fixed model
MODEL_FAMILY="llama3.1-8b-instruct"
BASE_MODEL_PATH="./train31Llama/trained_model/O3_LLAMA31_TOFU"

SD_DIR_ONLY="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ntofu_SD_only_20"
RD_DIR_ONLY="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ntofu_RD_20"
SD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ntofu_SD_20"
RD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ntofu_RD_20"

UTILITY1="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/tofu_real_authors/real_authors_test.json"
UTILITY2="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/tofu_world_facts/world_facts_test.json"
SAVE_FOLDER_BASE="tofu_moe_fixed"

# MoE 저장 루트
CKPT_ROOT="./outputs/tofu_moe_ckpt_origin"

DOMAINS=(
  "AG01" "AG02" "AG03" "AG04" "AG05"
  "AG06" "AG07" "AG08" "AG09" "AG10"
  "AG11" "AG12" "AG13" "AG14" "AG15"
  "AG16" "AG17" "AG18" "AG19"
)

# 원하는 순서대로
ROUTER_WEIGHTS=(0.01 0.1 0.5)

ACC=()
step_idx=0

for domain in "${DOMAINS[@]}"; do
  ACC+=("${domain}")
  joined="$(IFS=_; echo "${ACC[*]}")"

  echo "======================================================"
  echo "DOMAIN=${domain} / step=${step_idx}"
  echo "======================================================"

  # 현재 step 데이터 (Train)
  data_path="${SD_DIR_ONLY}/tofu_${domain}_SD.json"
  r_data_path="${RD_DIR_ONLY}/tofu_not_${joined}_RD.json"

  # Evaluation용 데이터
  sampled_data_path="${SD_DIR}/tofu_${joined}_SD.json"
  retain_data_path="${RD_DIR}/tofu_not_${joined}_RD.json"

  for ROUTER_WEIGHT in "${ROUTER_WEIGHTS[@]}"; do
    RW_TAG="${ROUTER_WEIGHT//./p}"
    SAVE_FOLDER="${SAVE_FOLDER_BASE}_rw_${RW_TAG}"
    task_id="tofu_rw${RW_TAG}_t${step_idx}_${domain}"

    # 저장/로드 경로
    save_unlearned_model_path="${CKPT_ROOT}/router_weight_${RW_TAG}/t${step_idx}_${domain}/lora_moe.pt"

    if [[ "${step_idx}" -eq 0 ]]; then
      load_unlearned_model="false"
      load_unlearned_model_path=""

      echo "---- router=${ROUTER_WEIGHT} / step=${step_idx} (initial task)"
      echo " load_unlearned_model : false"
    else
      prev_step=$((step_idx - 1))
      prev_domain="${DOMAINS[$prev_step]}"
      load_unlearned_model="true"
      load_unlearned_model_path="${CKPT_ROOT}/router_weight_${RW_TAG}/t${prev_step}_${prev_domain}/lora_moe.pt"

      echo "---- router=${ROUTER_WEIGHT} / step=${step_idx} (cumulative task)"
      echo " load_unlearned_model : true"
      echo " load path            : ${load_unlearned_model_path}"
    fi

    echo " router_weight        : ${ROUTER_WEIGHT}"
    echo " domain               : ${domain}"
    echo " train SD             : ${data_path}"
    echo " train RD             : ${r_data_path}"
    echo " test SD              : ${sampled_data_path}"
    echo " test RD              : ${retain_data_path}"
    echo " save path            : ${save_unlearned_model_path}"
    echo

    mkdir -p "$(dirname "${save_unlearned_model_path}")"

    ${PY} ${SCRIPT} \
      --config-name "${CFG_NAME}" \
      data_name="tofu" \
      model_family="${MODEL_FAMILY}" \
      model_path="${BASE_MODEL_PATH}" \
      moe.router_weight="${ROUTER_WEIGHT}" \
      task_id="${task_id}" \
      data_path="${data_path}" \
      r_data_path="${r_data_path}" \
      sampled_data_path="${sampled_data_path}" \
      retain_data_path="${retain_data_path}" \
      utiliy_data_path1="${UTILITY1}" \
      utiliy_data_path2="${UTILITY2}" \
      load_unlearned_model="${load_unlearned_model}" \
      load_unlearned_model_path="${load_unlearned_model_path}" \
      save_unlearned_model=true \
      save_unlearned_model_path="${save_unlearned_model_path}" \
      save_folder="${SAVE_FOLDER}/lunar_${MODEL_FAMILY}_eval_base_${task_id}"

    echo
    echo "[DONE] domain=${domain}, router_weight=${ROUTER_WEIGHT}"
    echo
  done

  step_idx=$((step_idx + 1))
done

echo "[DONE] All runs finished."