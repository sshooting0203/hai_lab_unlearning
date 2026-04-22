#!/usr/bin/env bash
set -euo pipefail

# Config
PY="${PY:-python}"
SCRIPT="${SCRIPT:-mmain_tofu_fixed.py}"
PLOT_SCRIPT="${PLOT_SCRIPT:-plot_results_tofu.py}"
CFG_NAME="${CFG_NAME:-forget_tofu}"

# fixed model
MODEL_FAMILY="llama3.1-8b-instruct"
BASE_MODEL_PATH="./train31Llama/trained_model/O3_LLAMA31_TOFU"

SD_DIR_ONLY="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ntofu_SD_only_20"
RD_DIR_ONLY="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ntofu_RD_20"
SD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ntofu_SD_20"
RD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ntofu_RD_20"

UTILITY1="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/tofu_real_authors/real_authors_test.json"
UTILITY2="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/tofu_world_facts/world_facts_test.json"
SAVE_FOLDER="tofu_moe_fixed"

# MoE 저장 루트
CKPT_ROOT="./outputs/tofu_moe_ckpt_origin"

DOMAINS=(
  "AG01" "AG02" "AG03" "AG04" "AG05"
  "AG06" "AG07" "AG08" "AG09" "AG10"
  "AG11" "AG12" "AG13" "AG14" "AG15"
  "AG16" "AG17" "AG18" "AG19" 
)

SEEDS=(0)

for SEED in "${SEEDS[@]}"; do
  echo "================ SEED=${SEED} ================"

  ACC=()
  step_idx=0

  for domain in "${DOMAINS[@]}"; do
    ACC+=("${domain}")

    joined="$(IFS=_; echo "${ACC[*]}")"
    task_id="tofu_t${step_idx}_${domain}"

    # 현재 step 데이터 (Train)
    data_path="${SD_DIR_ONLY}/tofu_${domain}_SD.json"
    r_data_path="${RD_DIR_ONLY}/tofu_not_${joined}_RD.json"

    # Evaluation용 데이터
    sampled_data_path="${SD_DIR}/tofu_${joined}_SD.json"
    retain_data_path="${RD_DIR}/tofu_not_${joined}_RD.json"

    # 저장/로드 경로
    save_unlearned_model_path="${CKPT_ROOT}/seed_${SEED}/t${step_idx}_${domain}/lora_moe.pt"

    if [[ "${step_idx}" -eq 0 ]]; then
      load_unlearned_model="false"
      load_unlearned_model_path=""

      echo "---- step=${step_idx} (initial task)"
      echo " load_unlearned_model : false"
    else
      prev_step=$((step_idx - 1))
      prev_domain="${DOMAINS[$prev_step]}"
      load_unlearned_model="true"
      load_unlearned_model_path="${CKPT_ROOT}/seed_${SEED}/t${prev_step}_${prev_domain}/lora_moe.pt"

      echo "---- step=${step_idx} (cumulative task)"
      echo " load_unlearned_model : true"
      echo " load path            : ${load_unlearned_model_path}"
    fi

    echo " domains              : ${domain} (step ${step_idx})"
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
      seed="${SEED}" \
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
    step_idx=$((step_idx + 1))
  done
done

echo "[DONE] All cumulative eval runs finished."


################################################################################
RESULT_BASE_DIR="./run_results/eval_basemoe_error/llama3.1-8b-instruct/${SAVE_FOLDER}"
PLOT_OUTPUT_DIR="./run_plots/${SAVE_FOLDER}"

echo "[INFO] Start summarizing eval results..."

${PY} "${PLOT_SCRIPT}" \
  --base_dir "${RESULT_BASE_DIR}" \
  --output_dir "${PLOT_OUTPUT_DIR}" \
  --excel_filename "task_dataset_metrics.xlsx" \
  --keys_to_extract forget retained real_authors world_facts \
  --row_names forget retain utility1 utility2 \
  --metric_columns \
    exact_match \
    token_f1 \
    rouge_l_recall \
    best_exact_match \
    best_token_f1 \
    best_rouge_l_recall \
    avg_prob \
    best_prob \
    truth_ratio \
    embedding_similarity \
    best_embedding_similarity \
  --excluded_metrics \
    probs \
  --plot_metrics \
    exact_match \
    token_f1 \
    rouge_l_recall \
    best_exact_match \
    best_token_f1 \
    best_rouge_l_recall \
    avg_prob \
    best_prob \
    truth_ratio \
    embedding_similarity \
    best_embedding_similarity

echo "[DONE] Plot and Excel summary finished."
