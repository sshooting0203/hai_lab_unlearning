#!/usr/bin/env bash
set -euo pipefail
# CUDA_VISIBLE_DEVICES=1 nohup bash test_clinc_sampling_fixed.sh > run_logs/test_clinc_sampling_fixed.log 2>&1 &

PY="${PY:-python}"
SCRIPT="${SCRIPT:-mmain_fixed.py}"
PLOT_SCRIPT="${PLOT_SCRIPT:-plot_results.py}"
CFG_NAME="${CFG_NAME:-forget}"

# fixed model
MODEL_FAMILY="llama3.1-8b-instruct"
BASE_MODEL_PATH="./train31Llama/trained_model/O3_LLAMA31_ClincQA"

UTILITY1="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/mrpcqa/mrpc_test.json"
UTILITY2="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/rteqa/rte_validation.json"
SD_EVAL_BASE="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/clincqa_SD_10"
RD_EVAL_BASE="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/clincqa_RD_10"
LAYER_MODIFIED="[16]"

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
)

SEEDS=(0)
RATES=(100 50 10 1)

TMP_DIR="./tmp_sampled_data"
mkdir -p "${TMP_DIR}"

for RATE in "${RATES[@]}"; do
  echo "===================================================="
  echo ">>> SPECIFIC SAMPLE RATE: ${RATE} <<<"
  echo "===================================================="

  SAVE_FOLDER="clincqa_moe_fixed_layer16_329_rate${RATE}_retained"
  CKPT_ROOT="./outputs/clincqa_moe_ckpt_329_rate${RATE}_retained"

  # SD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/329clincqa_SD_only_${RATE}"
  SD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/clincqa_SD_only_200"
  RD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/329clincqa_RD_only_${RATE}_c"

  for SEED in "${SEEDS[@]}"; do
    echo "================ SEED=${SEED} (RATE=${RATE}) ================"
    ACC=()
    step_idx=0
    
    for domain in "${DOMAINS[@]}"; do
      ACC+=("${domain}")

      joined="$(IFS=_; echo "${ACC[*]}")"
      task_id="clincqa_t${step_idx}_${domain}_rate${RATE}"

      data_path="${SD_DIR}/clincqa_${domain}_train_SD.json"
      r_data_path="${RD_DIR}/clincqa_not_${joined}_train_RD.json"
      orig_sampled_data_path="${SD_EVAL_BASE}/clincqa_${joined}_test_SD.json"
      orig_retain_data_path="${RD_EVAL_BASE}/clincqa_not_${joined}_test_RD.json"

      save_unlearned_model_path="${CKPT_ROOT}/seed_${SEED}/t${step_idx}_${domain}/lora_moe.pt"

      if [[ "${step_idx}" -eq 0 ]]; then
        load_unlearned_model="false"
        load_unlearned_model_path=""
      else
        prev_step=$((step_idx - 1))
        prev_domain="${DOMAINS[$prev_step]}"
        load_unlearned_model="true"
        load_unlearned_model_path="${CKPT_ROOT}/seed_${SEED}/t${prev_step}_${prev_domain}/lora_moe.pt"
      fi

      echo "---- step=${step_idx} (${domain}) Rate=${RATE}"
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
        sampled_data_path="${orig_sampled_data_path}" \
        retain_data_path="${orig_retain_data_path}" \
        utiliy_data_path1="${UTILITY1}" \
        utiliy_data_path2="${UTILITY2}" \
        load_unlearned_model="${load_unlearned_model}" \
        load_unlearned_model_path="${load_unlearned_model_path}" \
        save_unlearned_model=true \
        save_unlearned_model_path="${save_unlearned_model_path}" \
        save_folder="${SAVE_FOLDER}_${SEED}/lunar_${MODEL_FAMILY}_eval_base_${task_id}"

      step_idx=$((step_idx + 1))
    done
  done
done