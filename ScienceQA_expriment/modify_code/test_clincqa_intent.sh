# #!/usr/bin/env bash
# # CUDA_VISIBLE_DEVICES=3 nohup bash test_clincqa_intent.sh > run_logs/test_clincqa_intent.log 2>&1 &
# set -euo pipefail

# # Config
# PY="${PY:-python}"
# SCRIPT="${SCRIPT:-mmain_fixed.py}"
# CFG_NAME="${CFG_NAME:-forget_old}"

# # fixed model
# MODEL_FAMILY="llama3.1-8b-instruct"
# BASE_MODEL_PATH="./train31Llama/trained_model/O3_LLAMA31_ClincQA"

# # train SD: LOO 버전
# # train RD: cumulative 버전
# # test SD/RD: cumulative 버전
# SD_DIR_ONLY="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/intent_clincqa_SD_only_100"
# RD_DIR_ONLY="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/intent_clincqa_RD_10_c"

# SD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/intent_clincqa_SD_100_c"
# RD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/intent_clincqa_RD_100_c"

# UTILITY1="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/mrpcqa/mrpc_test.json"
# UTILITY2="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/rteqa/rte_validation.json"

# # MoE 저장 루트
# CKPT_ROOT="./outputs/clincqa_intent_moe_ckpt"

# SEEDS=(0)

# # ---------------------------------------------------------
# # manifest에서 ordered_units 읽기
# # ---------------------------------------------------------
# MANIFEST_TRAIN="$(find "${SD_DIR_ONLY}" -maxdepth 1 -type f -name 'manifest_*_train.json' | head -n 1)"

# if [[ -z "${MANIFEST_TRAIN}" ]]; then
#   echo "[ERROR] train manifest를 찾지 못했습니다: ${SD_DIR_ONLY}"
#   exit 1
# fi

# echo "[INFO] using manifest: ${MANIFEST_TRAIN}"

# mapfile -t UNIT_KEYS < <("${PY}" - <<PY
# import json
# from pathlib import Path

# manifest_path = Path(r"""${MANIFEST_TRAIN}""")
# with manifest_path.open("r", encoding="utf-8") as f:
#     data = json.load(f)

# for u in data["ordered_units"]:
#     print(u)
# PY
# )

# if [[ "${#UNIT_KEYS[@]}" -eq 0 ]]; then
#   echo "[ERROR] ordered_units가 비어 있습니다."
#   exit 1
# fi

# echo "[INFO] total intent units: ${#UNIT_KEYS[@]}"
# echo

# # ---------------------------------------------------------
# # helper
# # ---------------------------------------------------------
# get_domain_from_unit() {
#   local unit_key="$1"
#   echo "${unit_key%%__*}"
# }

# check_file() {
#   local path="$1"
#   if [[ ! -f "${path}" ]]; then
#     echo "[ERROR] file not found: ${path}"
#     exit 1
#   fi
# }

# # ---------------------------------------------------------
# # main loop
# # ---------------------------------------------------------
# for SEED in "${SEEDS[@]}"; do
#   echo "================ SEED=${SEED} ================"

#   step_idx=0

#   for unit_key in "${UNIT_KEYS[@]}"; do
#     domain="$(get_domain_from_unit "${unit_key}")"
#     step_tag="$(printf "t%03d" "${step_idx}")"
#     task_id="clincqa_${step_tag}_${unit_key}"

#     # 현재 step 데이터
#     # train SD: loo 버전
#     data_path="${SD_DIR_ONLY}/${domain}/clincqa_${unit_key}_train_SD.json"

#     # train RD: cumulative 버전 (파일명에 step_tag 포함)
#     r_data_path="${RD_DIR_ONLY}/${domain}/clincqa_not_${step_tag}_${unit_key}_train_RD.json"

#     # eval: cumulative 버전 사용
#     sampled_data_path="${SD_DIR}/${domain}/clincqa_${step_tag}_${unit_key}_test_SD.json"
#     retain_data_path="${RD_DIR}/${domain}/clincqa_not_${step_tag}_${unit_key}_test_RD.json"

#     # 저장/로드 경로
#     save_unlearned_model_path="${CKPT_ROOT}/seed_${SEED}/${step_tag}_${unit_key}/lora_moe.pt"

#     if [[ "${step_idx}" -eq 0 ]]; then
#       load_unlearned_model="false"
#       load_unlearned_model_path=""

#       echo "---- step=${step_idx} (initial task)"
#       echo " load_unlearned_model : false"
#     else
#       prev_step=$((step_idx - 1))
#       prev_step_tag="$(printf "t%03d" "${prev_step}")"
#       prev_unit_key="${UNIT_KEYS[$prev_step]}"

#       load_unlearned_model="true"
#       load_unlearned_model_path="${CKPT_ROOT}/seed_${SEED}/${prev_step_tag}_${prev_unit_key}/lora_moe.pt"

#       echo "---- step=${step_idx} (cumulative task)"
#       echo " load_unlearned_model : true"
#       echo " load path            : ${load_unlearned_model_path}"
#     fi

#     echo " unit                 : ${unit_key}"
#     echo " domain               : ${domain}"
#     echo " step_tag             : ${step_tag}"
#     echo " train SD             : ${data_path}"
#     echo " train RD             : ${r_data_path}"
#     echo " test SD              : ${sampled_data_path}"
#     echo " test RD              : ${retain_data_path}"
#     echo " save path            : ${save_unlearned_model_path}"
#     echo

#     # 파일 존재 확인
#     check_file "${data_path}"
#     check_file "${r_data_path}"
#     check_file "${sampled_data_path}"
#     check_file "${retain_data_path}"
#     check_file "${UTILITY1}"
#     check_file "${UTILITY2}"

#     if [[ "${load_unlearned_model}" == "true" ]]; then
#       check_file "${load_unlearned_model_path}"
#     fi

#     mkdir -p "$(dirname "${save_unlearned_model_path}")"

#     ${PY} ${SCRIPT} \
#       --config-name "${CFG_NAME}" \
#       data_name="clincqa" \
#       model_family="${MODEL_FAMILY}" \
#       model_path="${BASE_MODEL_PATH}" \
#       seed="${SEED}" \
#       task_id="${task_id}" \
#       data_path="${data_path}" \
#       r_data_path="${r_data_path}" \
#       sampled_data_path="${sampled_data_path}" \
#       retain_data_path="${retain_data_path}" \
#       utiliy_data_path1="${UTILITY1}" \
#       utiliy_data_path2="${UTILITY2}" \
#       load_unlearned_model="${load_unlearned_model}" \
#       load_unlearned_model_path="${load_unlearned_model_path}" \
#       save_unlearned_model=true \
#       save_unlearned_model_path="${save_unlearned_model_path}" \
#       save_folder="clincqa_moe_second_intent_selectedCE/lunar_${MODEL_FAMILY}_eval_base_${task_id}"

#     echo
#     step_idx=$((step_idx + 1))
#   done
# done

# echo "[DONE] All cumulative eval runs finished."

#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=3 nohup bash test_clincqa_intent.sh > run_logs/test_clincqa_intent.log 2>&1 &
set -euo pipefail

PY="${PY:-python}"
SCRIPT="${SCRIPT:-mmain_fixed.py}"
CFG_NAME="${CFG_NAME:-forget_old}"

MODEL_FAMILY="llama3.1-8b-instruct"
BASE_MODEL_PATH="./train31Llama/trained_model/O3_LLAMA31_ClincQA"

SD_DIR_ONLY="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/intent_clincqa_SD_only_100"
RD_DIR_ONLY="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/intent_clincqa_RD_10_c"

SD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/intent_clincqa_SD_100_c"
RD_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/intent_clincqa_RD_100_c"

UTILITY1="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/mrpcqa/mrpc_test.json"
UTILITY2="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/rteqa/rte_validation.json"

CKPT_ROOT="./outputs/clincqa_intent_moe_ckpt"
SEEDS=(0)

START_STEP="67"

MANIFEST_TRAIN="$(find "${SD_DIR_ONLY}" -maxdepth 1 -type f -name 'manifest_*_train.json' | head -n 1)"

if [[ -z "${MANIFEST_TRAIN}" ]]; then
  echo "[ERROR] train manifest를 찾지 못했습니다: ${SD_DIR_ONLY}"
  exit 1
fi

echo "[INFO] using manifest: ${MANIFEST_TRAIN}"

mapfile -t UNIT_KEYS < <("${PY}" - <<PY
import json
from pathlib import Path

manifest_path = Path(r"""${MANIFEST_TRAIN}""")
with manifest_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

for u in data["ordered_units"]:
    print(u)
PY
)

if [[ "${#UNIT_KEYS[@]}" -eq 0 ]]; then
  echo "[ERROR] ordered_units가 비어 있습니다."
  exit 1
fi

echo "[INFO] total intent units: ${#UNIT_KEYS[@]}"
echo "[INFO] START_STEP=${START_STEP}"
echo

get_domain_from_unit() {
  local unit_key="$1"
  echo "${unit_key%%__*}"
}

check_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] file not found: ${path}"
    exit 1
  fi
}

for SEED in "${SEEDS[@]}"; do
  echo "================ SEED=${SEED} ================"

  step_idx=0

  for unit_key in "${UNIT_KEYS[@]}"; do
    if (( step_idx < START_STEP )); then
      echo "[SKIP] step=${step_idx}, unit=${unit_key}"
      step_idx=$((step_idx + 1))
      continue
    fi

    domain="$(get_domain_from_unit "${unit_key}")"
    step_tag="$(printf "t%03d" "${step_idx}")"
    task_id="clincqa_${step_tag}_${unit_key}"

    data_path="${SD_DIR_ONLY}/${domain}/clincqa_${unit_key}_train_SD.json"
    r_data_path="${RD_DIR_ONLY}/${domain}/clincqa_not_${step_tag}_${unit_key}_train_RD.json"
    sampled_data_path="${SD_DIR}/${domain}/clincqa_${step_tag}_${unit_key}_test_SD.json"
    retain_data_path="${RD_DIR}/${domain}/clincqa_not_${step_tag}_${unit_key}_test_RD.json"

    save_unlearned_model_path="${CKPT_ROOT}/seed_${SEED}/${step_tag}_${unit_key}/lora_moe.pt"

    if [[ "${step_idx}" -eq 0 ]]; then
      load_unlearned_model="false"
      load_unlearned_model_path=""

      echo "---- step=${step_idx} (initial task)"
      echo " load_unlearned_model : false"
    else
      prev_step=$((step_idx - 1))
      prev_step_tag="$(printf "t%03d" "${prev_step}")"
      prev_unit_key="${UNIT_KEYS[$prev_step]}"

      load_unlearned_model="true"
      load_unlearned_model_path="${CKPT_ROOT}/seed_${SEED}/${prev_step_tag}_${prev_unit_key}/lora_moe.pt"

      echo "---- step=${step_idx} (resume/cumulative task)"
      echo " load_unlearned_model : true"
      echo " load path            : ${load_unlearned_model_path}"
    fi

    echo " unit                 : ${unit_key}"
    echo " domain               : ${domain}"
    echo " step_tag             : ${step_tag}"
    echo " train SD             : ${data_path}"
    echo " train RD             : ${r_data_path}"
    echo " test SD              : ${sampled_data_path}"
    echo " test RD              : ${retain_data_path}"
    echo " save path            : ${save_unlearned_model_path}"
    echo

    check_file "${data_path}"
    check_file "${r_data_path}"
    check_file "${sampled_data_path}"
    check_file "${retain_data_path}"
    check_file "${UTILITY1}"
    check_file "${UTILITY2}"

    if [[ "${load_unlearned_model}" == "true" ]]; then
      check_file "${load_unlearned_model_path}"
    fi

    mkdir -p "$(dirname "${save_unlearned_model_path}")"

    ${PY} ${SCRIPT} \
      --config-name "${CFG_NAME}" \
      data_name="clincqa" \
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
      save_folder="clincqa_moe_second_intent_selectedCE/lunar_${MODEL_FAMILY}_eval_base_${task_id}"

    echo
    step_idx=$((step_idx + 1))
  done
done

echo "[DONE] All cumulative eval runs finished."