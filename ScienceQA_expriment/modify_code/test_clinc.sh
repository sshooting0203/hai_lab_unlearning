#!/usr/bin/env bash
set -euo pipefail
# CUDA_VISIBLE_DEVICES=2 nohup bash test_clinc.sh > run_logs/test_clinc_4_23.log 2>&1 &

# Config
PY="${PY:-python}"
SCRIPT="${SCRIPT:-mmain_fixed.py}"
PLOT_SCRIPT="${PLOT_SCRIPT:-plot_results.py}"
CFG_NAME="${CFG_NAME:-forget}"

# fixed model
MODEL_FAMILY="llama3.1-8b-instruct"
BASE_MODEL_PATH="./train31Llama/trained_model/O3_LLAMA31_ClincQA"

SD_DIR_ONLY="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/clincqa_SD_only_200"
RD_DIR_ONLY="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/clincqa_RD_only_200_c" # retain 데이터셋이 200개씩 줄어든다고 보면 됨 
SD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/clincqa_SD_10"
RD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/clincqa_RD_10"
UTILITY1="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/mrpcqa/mrpc_test.json"
UTILITY2="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/rteqa/rte_validation.json"
SAVE_FOLDER="clincqa_moe_fixed_layer16_4_23"

# MoE 저장 루트
CKPT_ROOT="./outputs/clincqa_moe_ckpt_4_23"
LAYER_MODIFIED="[16]"

DOMAINS=(
  "work" # 1번 expert
  "travel" # 2번 expert
  "home"   # 3번 expert
  "banking"
  "credit_cards"
  "kitchen_and_dining"
  "auto_and_commute"
  "utility"
  "small_talk"
)

SEEDS=(0)

for SEED in "${SEEDS[@]}"; do
  echo "================ SEED=${SEED} ================"

  ACC=()
  step_idx=0
  
  for domain in "${DOMAINS[@]}"; do
    ACC+=("${domain}")

    joined="$(IFS=_; echo "${ACC[*]}")"
    task_id="clincqa_t${step_idx}_${domain}"

    # 현재 step 데이터
    data_path="${SD_DIR_ONLY}/clincqa_${domain}_train_SD.json"
    r_data_path="${RD_DIR_ONLY}/clincqa_not_${joined}_train_RD.json"
    sampled_data_path="${SD_DIR}/clincqa_${joined}_test_SD.json"
    retain_data_path="${RD_DIR}/clincqa_not_${joined}_test_RD.json"

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

    echo " domains              : ${joined}"
    echo " train SD             : ${data_path}"
    echo " train RD             : ${r_data_path}"
    echo " test SD              : ${sampled_data_path}"
    echo " test RD              : ${retain_data_path}"
    echo " save path            : ${save_unlearned_model_path}"
    echo

    mkdir -p "$(dirname "${save_unlearned_model_path}")"

    ${PY} ${SCRIPT} \
      --config-name "${CFG_NAME}" \
      data_name="clincqa" \
      model_family="${MODEL_FAMILY}" \
      model_path="${BASE_MODEL_PATH}" \
      seed="${SEED}" \
      task_id="${task_id}" \
      layer_modified="${LAYER_MODIFIED}" \
      data_path="${data_path}" \
      r_data_path="${r_data_path}" \
      use_unverified=false \
      use_harmful=true \
      sampled_data_path="${sampled_data_path}" \
      retain_data_path="${retain_data_path}" \
      utiliy_data_path1="${UTILITY1}" \
      utiliy_data_path2="${UTILITY2}" \
      load_unlearned_model="${load_unlearned_model}" \
      load_unlearned_model_path="${load_unlearned_model_path}" \
      save_unlearned_model=true \
      save_unlearned_model_path="${save_unlearned_model_path}" \
      save_folder="${SAVE_FOLDER}_${SEED}/lunar_${MODEL_FAMILY}_eval_base_${task_id}"

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
  --keys_to_extract forget retained rte mrpc \
  --row_names forget retain utility1 utility2 \
  --metric_columns \
    choice_accuracy \
    choice_correct \
    gt_loss \
    gt_loss_per_token \
    hit_rate \
    label_contains_accuracy \
    label_contains_correct \
    mrr \
    perplexity \
    probs \
  --excluded_metrics \
    probs \
  --plot_metrics \
    choice_accuracy \
    hit_rate \
    mrr \
    perplexity \
    gt_loss \
    gt_loss_per_token \
    label_contains_accuracy

echo "[DONE] Plot and Excel summary finished."
