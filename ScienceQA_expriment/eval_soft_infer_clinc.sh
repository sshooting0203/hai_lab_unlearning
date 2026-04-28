# #!/usr/bin/env bash
# set -euo pipefail

# DOMAINS=(
#   "work"
#   "travel"
#   "home"
#   "banking"
#   "credit_cards"
#   "kitchen_and_dining"
#   "auto_and_commute"
#   "utility"
#   "small_talk"
# )
# # BASE_MODEL="sshl505/O3_LLAMA2_ClincQA"
# BASE_MODEL="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code/train31Llama/trained_model/O3_LLAMA31_ClincQA"
# OOD_SETTING="C"
# OOD_SETTING_NAME="clincqa"

# for SCALE in 0.1
# do
#   for SEED in 0 1 2
#   do
#     for LABEL_K in "force"
#     do
#       OUTPUT_1="./SCALE_${SCALE}_seed_${SEED}_o_unlearn_lora_${LABEL_K}_checkpoints_clinc9_origin/lora_${LABEL_K}_random"
#       TYPE=""
#       for UNLEAN_D in "${DOMAINS[@]}"
#       do

#         OUTPUT_1+="_${UNLEAN_D}_${LABEL_K}"
#         TYPE+="_${UNLEAN_D}"

#         OOD_PATH="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment"

#         TESTPATH_1="./data_new/clincqa_RD_10/clincqa_not${TYPE}_test_RD.json"
#         python eval_o3_clinc.py \
#           --unlearn_topic "${UNLEAN_D}_testRD" \
#           --test_dataset ${TESTPATH_1} \
#           --base_model ${BASE_MODEL} \
#           --seed ${SEED} \
#           --lora_weights ${OUTPUT_1} \
#           --ood_type ${TYPE} \
#           --ood_setting ${OOD_SETTING} \
#           --ood_setting_name ${OOD_SETTING_NAME} \
#           --ood_weights "${OOD_PATH}/ood_checkpoints_clincqa_${SEED}/"

#         TESTPATH_1="./data_new/clincqa_SD_10/clincqa${TYPE}_train_SD.json"
#         python eval_o3_clinc.py \
#           --unlearn_topic "${UNLEAN_D}_trainSD" \
#           --test_dataset ${TESTPATH_1} \
#           --base_model ${BASE_MODEL} \
#           --seed ${SEED} \
#           --lora_weights ${OUTPUT_1} \
#           --ood_type ${TYPE} \
#           --ood_setting ${OOD_SETTING} \
#           --ood_setting_name ${OOD_SETTING_NAME} \
#           --ood_weights "${OOD_PATH}/ood_checkpoints_clincqa_${SEED}/"

#         TESTPATH_1="./data_new/clincqa_SD_10/clincqa${TYPE}_test_SD.json"
#         python eval_o3_clinc.py \
#           --unlearn_topic "${UNLEAN_D}_testSD" \
#           --test_dataset ${TESTPATH_1} \
#           --base_model ${BASE_MODEL} \
#           --seed ${SEED} \
#           --lora_weights ${OUTPUT_1} \
#           --ood_type ${TYPE} \
#           --ood_setting ${OOD_SETTING} \
#           --ood_setting_name ${OOD_SETTING_NAME} \
#           --ood_weights "${OOD_PATH}/ood_checkpoints_clincqa_${SEED}/"

#         TESTPATH_1="./data_new/mrpcqa/mrpc_test.json"
#         python eval_o3_clinc.py \
#           --unlearn_topic "${UNLEAN_D}_mrpc" \
#           --test_dataset ${TESTPATH_1} \
#           --base_model ${BASE_MODEL} \
#           --seed ${SEED} \
#           --lora_weights ${OUTPUT_1} \
#           --ood_type ${TYPE} \
#           --ood_setting ${OOD_SETTING} \
#           --ood_setting_name ${OOD_SETTING_NAME} \
#           --ood_weights "${OOD_PATH}/ood_checkpoints_clincqa_${SEED}/"

#         TESTPATH_1="./data_new/rteqa/rte_validation.json" # @ 수정 : rte test.json label없음
#         python eval_o3_clinc.py \
#           --unlearn_topic "${UNLEAN_D}_rte" \
#           --test_dataset ${TESTPATH_1} \
#           --base_model ${BASE_MODEL} \
#           --seed ${SEED} \
#           --lora_weights ${OUTPUT_1} \
#           --ood_type ${TYPE} \
#           --ood_setting ${OOD_SETTING} \
#           --ood_setting_name ${OOD_SETTING_NAME} \
#           --ood_weights "${OOD_PATH}/ood_checkpoints_clincqa_${SEED}/"
#       done
#     done
#   done
# done

#!/usr/bin/env bash
set -euo pipefail

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

BASE_MODEL="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code/train31Llama/trained_model/O3_LLAMA31_ClincQA"
OOD_SETTING="C"
OOD_SETTING_NAME="clincqa"
OOD_PATH="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment"

for SCALE in 0.1
do
  for SEED in 0 1 2
  do
    for LABEL_K in force
    do
      OUTPUT_1="./SCALE_${SCALE}_seed_${SEED}_o_unlearn_lora_${LABEL_K}_checkpoints_clinc9_origin/lora_${LABEL_K}_random"
      TYPE=""

      for UNLEAN_D in "${DOMAINS[@]}"
      do
        # 누적 LoRA 경로
        OUTPUT_1+="_${UNLEAN_D}_${LABEL_K}"

        # 누적 dataset type 문자열
        if [ -z "${TYPE}" ]; then
          TYPE="${UNLEAN_D}"
        else
          TYPE="${TYPE}_${UNLEAN_D}"
        fi

        RESULT_DIR="$(dirname "${OUTPUT_1}")"

        echo "=================================================="
        echo "[INFO] SCALE=${SCALE} SEED=${SEED} LABEL_K=${LABEL_K}"
        echo "[INFO] UNLEAN_D=${UNLEAN_D}"
        echo "[INFO] TYPE=${TYPE}"
        echo "[INFO] OUTPUT_1=${OUTPUT_1}"
        echo "[INFO] OOD_TYPE=${UNLEAN_D}"
        echo "=================================================="

        # 1) test RD
        TESTPATH_1="./data_new/clincqa_RD_10/clincqa_not_${TYPE}_test_RD.json"
        RESULT_FILE="${RESULT_DIR}/test_seed${SEED}_${UNLEAN_D}_testRD.json"
        if [ -f "${RESULT_FILE}" ]; then
          echo "[SKIP] ${RESULT_FILE}"
        else
          python eval_o3_clinc.py \
            --unlearn_topic "${UNLEAN_D}_testRD" \
            --test_dataset "${TESTPATH_1}" \
            --base_model "${BASE_MODEL}" \
            --seed "${SEED}" \
            --lora_weights "${OUTPUT_1}" \
            --ood_type "${UNLEAN_D}" \
            --ood_setting "${OOD_SETTING}" \
            --ood_setting_name "${OOD_SETTING_NAME}" \
            --ood_weights "${OOD_PATH}/ood_checkpoints_clincqa_${SEED}/"
        fi

        # 2) train SD
        TESTPATH_1="./data_new/clincqa_SD_10/clincqa_${TYPE}_train_SD.json"
        RESULT_FILE="${RESULT_DIR}/test_seed${SEED}_${UNLEAN_D}_trainSD.json"
        if [ -f "${RESULT_FILE}" ]; then
          echo "[SKIP] ${RESULT_FILE}"
        else
          python eval_o3_clinc.py \
            --unlearn_topic "${UNLEAN_D}_trainSD" \
            --test_dataset "${TESTPATH_1}" \
            --base_model "${BASE_MODEL}" \
            --seed "${SEED}" \
            --lora_weights "${OUTPUT_1}" \
            --ood_type "${UNLEAN_D}" \
            --ood_setting "${OOD_SETTING}" \
            --ood_setting_name "${OOD_SETTING_NAME}" \
            --ood_weights "${OOD_PATH}/ood_checkpoints_clincqa_${SEED}/"
        fi

        # 3) test SD
        TESTPATH_1="./data_new/clincqa_SD_10/clincqa_${TYPE}_test_SD.json"
        RESULT_FILE="${RESULT_DIR}/test_seed${SEED}_${UNLEAN_D}_testSD.json"
        if [ -f "${RESULT_FILE}" ]; then
          echo "[SKIP] ${RESULT_FILE}"
        else
          python eval_o3_clinc.py \
            --unlearn_topic "${UNLEAN_D}_testSD" \
            --test_dataset "${TESTPATH_1}" \
            --base_model "${BASE_MODEL}" \
            --seed "${SEED}" \
            --lora_weights "${OUTPUT_1}" \
            --ood_type "${UNLEAN_D}" \
            --ood_setting "${OOD_SETTING}" \
            --ood_setting_name "${OOD_SETTING_NAME}" \
            --ood_weights "${OOD_PATH}/ood_checkpoints_clincqa_${SEED}/"
        fi

        # 4) MRPC
        TESTPATH_1="./data_new/mrpcqa/mrpc_test.json"
        RESULT_FILE="${RESULT_DIR}/test_seed${SEED}_${UNLEAN_D}_mrpc.json"
        if [ -f "${RESULT_FILE}" ]; then
          echo "[SKIP] ${RESULT_FILE}"
        else
          python eval_o3_clinc.py \
            --unlearn_topic "${UNLEAN_D}_mrpc" \
            --test_dataset "${TESTPATH_1}" \
            --base_model "${BASE_MODEL}" \
            --seed "${SEED}" \
            --lora_weights "${OUTPUT_1}" \
            --ood_type "${UNLEAN_D}" \
            --ood_setting "${OOD_SETTING}" \
            --ood_setting_name "${OOD_SETTING_NAME}" \
            --ood_weights "${OOD_PATH}/ood_checkpoints_clincqa_${SEED}/"
        fi

        # 5) RTE
        TESTPATH_1="./data_new/rteqa/rte_validation.json"
        RESULT_FILE="${RESULT_DIR}/test_seed${SEED}_${UNLEAN_D}_rte.json"
        if [ -f "${RESULT_FILE}" ]; then
          echo "[SKIP] ${RESULT_FILE}"
        else
          python eval_o3_clinc.py \
            --unlearn_topic "${UNLEAN_D}_rte" \
            --test_dataset "${TESTPATH_1}" \
            --base_model "${BASE_MODEL}" \
            --seed "${SEED}" \
            --lora_weights "${OUTPUT_1}" \
            --ood_type "${UNLEAN_D}" \
            --ood_setting "${OOD_SETTING}" \
            --ood_setting_name "${OOD_SETTING_NAME}" \
            --ood_weights "${OOD_PATH}/ood_checkpoints_clincqa_${SEED}/"
        fi
      done
    done
  done
done