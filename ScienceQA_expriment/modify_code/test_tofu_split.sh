#!/usr/bin/env bash
set -euo pipefail
# CUDA_VISIBLE_DEVICES=3 nohup bash test_tofu_o3.sh > run_logs/test_tofu_o3.log 2>&1 &
# Config
PY="${PY:-python}"
SCRIPT="${SCRIPT:-mmain_fixed2.py}"
PLOT_SCRIPT="${PLOT_SCRIPT:-plot_results_tofu.py}"
CFG_NAME="${CFG_NAME:-forget}"

# fixed model
MODEL_FAMILY="llama3.1-8b-instruct"
BASE_MODEL_PATH="./train31Llama/trained_model/O3_LLAMA31_TOFU"

SD_DIR_ONLY="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ptofu_SD_only"
RD_DIR_ONLY="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ptofu_RD_only"
SD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ptofu_SD"
RD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/ptofu_RD"

UTILITY1="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/tofu_real_authors/real_authors_test.json"
UTILITY2="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/tofu_world_facts/world_facts_test.json"
SAVE_FOLDER="tofu_moe_fixed_4_23_split"

# MoE 저장 루트
CKPT_ROOT="./outputs/tofu_moe_ckpt_4_23_split"
LAYER_MODIFIED="[16]"

DOMAINS=(
  "AG01" "AG02"
  "AG03" "AG04" "AG05" "AG06" "AG07" "AG08" "AG09" "AG10"
  "AG11" "AG12" "AG13" "AG14" "AG15" "AG16" "AG17" "AG18" "AG19" "AG20"
  "AG21" "AG22" "AG23" "AG24" "AG25" "AG26" "AG27" "AG28" "AG29" "AG30"
  # "AG31" "AG32" "AG33" "AG34" "AG35" "AG36" "AG37" "AG38" "AG39" "AG40"
  # "AG41" "AG42" "AG43" "AG44" "AG45" "AG46" "AG47" "AG48" "AG49" "AG50"
  # "AG51" "AG52" "AG53" "AG54" "AG55" "AG56" "AG57" "AG58" "AG59" "AG60"
  # "AG61" "AG62" "AG63" "AG64" "AG65" "AG66" "AG67" "AG68" "AG69" "AG70"
  # "AG71" "AG72" "AG73" "AG74" "AG75" "AG76" "AG77" "AG78" "AG79" "AG80"
  # "AG81" "AG82" "AG83" "AG84" "AG85" "AG86" "AG87" "AG88" "AG89" "AG90"
  # "AG91" "AG92" "AG93" "AG94" "AG95" "AG96" "AG97" "AG98" "AG99" "AG100"
)

SEEDS=(0 1 2)

for SEED in "${SEEDS[@]}"; do
  echo "================ SEED=${SEED} ================"

  ACC=()
  step_idx=0

  for domain in "${DOMAINS[@]}"; do
    ACC+=("${domain}")

    joined="$(IFS=_; echo "${ACC[*]}")"
    task_id="tofu_t${step_idx}_${domain}"
    step_num=$(printf "%03d" $((step_idx + 1)))

    data_path="${SD_DIR_ONLY}/tofu_${domain}_SD.json"
    r_data_path="${RD_DIR}/tofu_step${step_num}_RD.json"
    sampled_data_path="${SD_DIR}/tofu_step${step_num}_SD.json"
    retain_data_path="${RD_DIR}/tofu_step${step_num}_RD.json"

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
      layer_modified="${LAYER_MODIFIED}" \
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
