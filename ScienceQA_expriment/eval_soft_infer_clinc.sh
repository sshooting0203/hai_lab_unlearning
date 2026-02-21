BASE_MODEL="sshl505/O3_LLAMA2_ClincQA"
OOD_SETTING="C"
OOD_SETTING_NAME="clincqa"
for SCALE in 0.1
do
  for SEED in 0 1 2
  do
    for LABEL_K in "force"
    do
      OUTPUT_1="./SCALE_${SCALE}_seed_${SEED}_o_unlearn_lora_${LABEL_K}_checkpoints_clinc/lora_${LABEL_K}_random"
      TYPE=""
      for UNLEAN_D in "work" "travel" "home"
      do

        OUTPUT_1+="_${UNLEAN_D}_${LABEL_K}"
        TYPE+="_${UNLEAN_D}"

        OOD_PATH="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/ood_checkpoints_clincqa"

        TESTPATH_1="./data/clincqa_RD/clinc_qa_not${TYPE}_test_RD.json"
        python eval_o3_clinc.py \
          --unlearn_topic "${UNLEAN_D}_testRD" \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --seed ${SEED} \
          --lora_weights ${OUTPUT_1} \
          --ood_type ${TYPE} \
          --ood_setting ${OOD_SETTING} \
          --ood_setting_name ${OOD_SETTING_NAME} \
          --ood_weights "${OOD_PATH}/ood_checkpoints_clincqa_${SEED}/"

        TESTPATH_1="./data/clincqa_SD/clinc_qa${TYPE}_train_SD.json"
        python eval_o3_clinc.py \
          --unlearn_topic "${UNLEAN_D}_trainSD" \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --seed ${SEED} \
          --lora_weights ${OUTPUT_1} \
          --ood_type ${TYPE} \
          --ood_setting ${OOD_SETTING} \
          --ood_setting_name ${OOD_SETTING_NAME} \
          --ood_weights "${OOD_PATH}/ood_checkpoints_clincqa_${SEED}/"

        TESTPATH_1="./data/clincqa_SD/clinc_qa${TYPE}_test_SD.json"
        python eval_o3_clinc.py \
          --unlearn_topic "${UNLEAN_D}_testSD" \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --seed ${SEED} \
          --lora_weights ${OUTPUT_1} \
          --ood_type ${TYPE} \
          --ood_setting ${OOD_SETTING} \
          --ood_setting_name ${OOD_SETTING_NAME} \
          --ood_weights "${OOD_PATH}/ood_checkpoints_clincqa_${SEED}/"

        TESTPATH_1="./data/mrpcqa/mrpc_test.json"
        python eval_o3_clinc.py \
          --unlearn_topic "${UNLEAN_D}_mrpc" \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --seed ${SEED} \
          --lora_weights ${OUTPUT_1} \
          --ood_type ${TYPE} \
          --ood_setting ${OOD_SETTING} \
          --ood_setting_name ${OOD_SETTING_NAME} \
          --ood_weights "${OOD_PATH}/ood_checkpoints_clincqa_${SEED}/"

        TESTPATH_1="./data/rteqa/rte_validation.json" # @ 수정 : rte test.json label없음
        python eval_o3_clinc.py \
          --unlearn_topic "${UNLEAN_D}_rte" \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --seed ${SEED} \
          --lora_weights ${OUTPUT_1} \
          --ood_type ${TYPE} \
          --ood_setting ${OOD_SETTING} \
          --ood_setting_name ${OOD_SETTING_NAME} \
          --ood_weights "${OOD_PATH}/ood_checkpoints_clincqa_${SEED}/"
      done
    done
  done
done

