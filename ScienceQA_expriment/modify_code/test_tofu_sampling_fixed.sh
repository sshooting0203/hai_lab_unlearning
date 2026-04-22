#!/usr/bin/env bash
set -euo pipefail
# CUDA_VISIBLE_DEVICES=3 nohup bash test_tofu_sampling_fixed.sh > run_logs/test_tofu_sampling_fixed.log 2>&1 &

PY="${PY:-python}"
SCRIPT="${SCRIPT:-mmain_tofu_fixed.py}"
PLOT_SCRIPT="${PLOT_SCRIPT:-plot_results_tofu.py}"
CFG_NAME="${CFG_NAME:-forget}"

# fixed model
MODEL_FAMILY="llama3.1-8b-instruct"
BASE_MODEL_PATH="./train31Llama/trained_model/O3_LLAMA31_TOFU"

UTILITY1="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/tofu_real_authors/real_authors_test.json"
UTILITY2="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/tofu_world_facts/world_facts_test.json"
SD_EVAL_BASE="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ptofu_SD"
RD_EVAL_BASE="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ptofu_RD"
LAYER_MODIFIED="[21]"

DOMAINS=(
  "AG01" "AG02" "AG03" "AG04" "AG05" "AG06" "AG07" "AG08" "AG09" "AG10"
  "AG11" "AG12" "AG13" "AG14" "AG15" "AG16" "AG17" "AG18" "AG19" "AG20"
  # "AG21" "AG22" "AG23" "AG24" "AG25" "AG26" "AG27" "AG28" "AG29" "AG30"
)

SEEDS=(0)
RATES=(100 40 10 1)

TMP_DIR="./tmp_sampled_data"
mkdir -p "${TMP_DIR}"

for RATE in "${RATES[@]}"; do
  echo "===================================================="
  echo ">>> ON-THE-FLY TOFU SAMPLING RATE: ${RATE} <<<"
  echo "===================================================="

  SAVE_FOLDER="tofu_moe_fixed_layer23_329_rate${RATE}_retained"
  CKPT_ROOT="./outputs/tofu_moe_ckpt_329_rate${RATE}_retained"

  SD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ptofu_SD_only"
  RD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/329tofu_RD_only_${RATE}_c"

  for SEED in "${SEEDS[@]}"; do
    echo "================ SEED=${SEED} (RATE=${RATE}) ================"
    ACC=()
    step_idx=0
    
    for domain in "${DOMAINS[@]}"; do
      ACC+=("${domain}")

      joined="$(IFS=_; echo "${ACC[*]}")"
      task_id="tofu_t${step_idx}_${domain}_rate${RATE}"

      # data_path: sampled data (fixed 200/max)
      data_path="${SD_DIR}/tofu_${domain}_SD.json"
      # r_data_path: retain data (sampled rate)
      r_data_path="${RD_DIR}/tofu_not_${joined}_RD.json"

      # Evaluation sets are still from the fixed base (ptofu_SD / ptofu_RD)
      step_num=$(printf "%03d" $((step_idx + 1)))
      orig_sampled_data_path="${SD_EVAL_BASE}/tofu_step${step_num}_SD.json"
      orig_retain_data_path="${RD_EVAL_BASE}/tofu_step${step_num}_RD.json"

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
        data_name="tofu" \
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