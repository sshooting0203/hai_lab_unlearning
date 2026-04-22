#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python}"
# SCRIPT="${SCRIPT:-eval_only.py}"
SCRIPT="${SCRIPT:-lunar_eval_only.py}"
CFG_NAME="${CFG_NAME:-forget.yaml}"  

MODEL_FAMILY="llama3.1-8b-instruct"
BASE_MODEL_PATH="./train31Llama/trained_model/O3_LLAMA31_ClincQA"

# SD_DIR=/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/clincqa_SD # @@
# RD_DIR=/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/clincqa_RD # @@
SD_DIR=/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/clincqa_SD_only_200 # @@
RD_DIR=/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/clincqa_RD_only_200 # @@
UTILITY1="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/mrpcqa/mrpc_test.json"
UTILITY2="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/rteqa/rte_validation.json"

DOMAINS=(
  "utility"
)

SEEDS=(0)

for SEED in "${SEEDS[@]}"; do
  echo "================ SEED=${SEED} ================"

  ACC=()
  t=0

  for domain in "${DOMAINS[@]}"; do
    ACC+=("${domain}")
    t=$((t+1))

    joined="$(IFS=_; echo "${ACC[*]}")"
    base_task_id="clincqa_t${t}_${domain}"

    # 현재 설정: domain 단독 파일 사용
    data_path1="${SD_DIR}/clincqa_${domain}_train_SD.json"
    data_path2="${RD_DIR}/clincqa_not_${domain}_train_RD.json"

    sampled_data_path="${SD_DIR}/clincqa_${domain}_test_SD.json"
    retain_data_path="${RD_DIR}/clincqa_not_${domain}_test_RD.json"

    echo "---- step=${t}"
    echo " domain             : ${domain}"
    echo " domains(acc)       : ${joined}"
    echo " sampled_data_path  : ${sampled_data_path}"
    echo " retain_data_path   : ${retain_data_path}"
    echo

    # layer 0 ~ 30 각각 단독 실험
    for LAYER in $(seq 0 30); do
      layer_modified="[${LAYER}]"
      task_id="${base_task_id}_layer${LAYER}"

      echo "  >>> layer=${LAYER}"
      echo "      task_id        : ${task_id}"
      echo "      layer_modified : ${layer_modified}"
      echo

      ${PY} ${SCRIPT} \
        --config-name "${CFG_NAME}" \
        data_name="clincqa" \
        model_family="${MODEL_FAMILY}" \
        base_model_path="${BASE_MODEL_PATH}" \
        model_path="${BASE_MODEL_PATH}" \
        seed="${SEED}" \
        task_id="${task_id}" \
        layer_modified="${layer_modified}" \
        data_path="${data_path1}" \
        r_data_path="${data_path2}" \
        sampled_data_path="${sampled_data_path}" \
        retain_data_path="${retain_data_path}" \
        utiliy_data_path1="${UTILITY1}" \
        utiliy_data_path2="${UTILITY2}" \
        eval_generation_max_length=512 \
        eval_generation_max_new_tokens=512 \
        save_folder="clincqa_lunar_base_t/lunar_${MODEL_FAMILY}_eval_base_${task_id}"

      echo
    done
  done
done

echo "[DONE] All cumulative eval runs finished."