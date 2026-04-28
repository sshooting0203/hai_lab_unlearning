export PYTHONNOUSERSITE=1
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
# BASE_MODEL="sshl505/O3_LLAMA2_ClincQA"
BASE_MODEL="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code/train31Llama/trained_model/O3_LLAMA31_ClincQA"
START_LAYER=0

for SCALE in 0.1
do
 for SEED in 0 1 2
 do
   for LABEL_K in "force"
   do
     OUTPUT_1="./SCALE_${SCALE}_seed_${SEED}_o_unlearn_lora_${LABEL_K}_checkpoints_clinc9_origin/lora_${LABEL_K}_random"
     LORA_W=""
     ADAPTER=""
     TYPE=""
     for UNLEAN_D in "${DOMAINS[@]}"
     do
    # ./SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_10/lora_force_random   _biology_force  _physics_force  _geography_force  _writing-strategies_force  _figurative-language_force  _chemistry_force'
       DATAPATH_1="./data_new/clincqa_random_${LABEL_K}/clincqa_${UNLEAN_D}_train_random_${LABEL_K}.json"
       OUTPUT_1+="_${UNLEAN_D}_${LABEL_K}"
       python train_unlearn_lora_o.py \
            --base_model ${BASE_MODEL} \
            --data_path ${DATAPATH_1} \
            --output_dir ${OUTPUT_1} \
            --seed ${SEED} \
            --batch_size 128 \
            --micro_batch_size 8 \
            --num_epochs 20 \
            --learning_rate 3e-4 \
            --cutoff_len 256 \
            --val_set_size 1 \
            --lora_r 8 \
            --lora_alpha 16 \
            --lora_dropout 0.05 \
            --lora_weights ${LORA_W} \
            --ood_weight "1,1" \
            --orthogonal_loss_weight ${SCALE} \
            --train_on_inputs \
            --group_by_length \
            --add_eos_token \
            --resume_from_checkpoint ${ADAPTER} \
            --lora_start_layer ${START_LAYER} 

       ADAPTER=${OUTPUT_1}
       TYPE+="_${UNLEAN_D}"

       TESTPATH_1="./data_new/mrpcqa/mrpc_test.json"
       python eval_lora_clinc.py \
           --unlearn_topic "${UNLEAN_D}_mrpc" \
           --test_dataset ${TESTPATH_1} \
           --base_model ${BASE_MODEL} \
           --seed ${SEED} \
           --lora_weights ${OUTPUT_1} \
           --lora_start_layer ${START_LAYER}

       TESTPATH_1="./data_new/rteqa/rte_validation.json" # @ 수정 : rte test.json label없음
       python eval_lora_clinc.py \
           --unlearn_topic "${UNLEAN_D}_rte" \
           --test_dataset ${TESTPATH_1} \
           --base_model ${BASE_MODEL} \
           --seed ${SEED} \
           --lora_weights ${OUTPUT_1} \
           --lora_start_layer ${START_LAYER}

       TESTPATH_1="./data_new/clincqa_RD_10/clincqa_not${TYPE}_test_RD.json"
       python eval_lora_clinc.py \
           --unlearn_topic "${UNLEAN_D}_testRD" \
           --test_dataset ${TESTPATH_1} \
           --base_model ${BASE_MODEL} \
           --seed ${SEED} \
           --lora_weights ${OUTPUT_1} \
           --lora_start_layer ${START_LAYER}

       TESTPATH_1="./data_new/clincqa_SD_10/clincqa${TYPE}_train_SD.json"
       python eval_lora_clinc.py \
           --unlearn_topic "${UNLEAN_D}_trainSD" \
           --test_dataset ${TESTPATH_1} \
           --base_model ${BASE_MODEL} \
           --seed ${SEED} \
           --lora_weights ${OUTPUT_1} \
           --lora_start_layer ${START_LAYER}

       TESTPATH_1="./data_new/clincqa_SD_10/clincqa${TYPE}_test_SD.json"
       python eval_lora_clinc.py \
           --unlearn_topic "${UNLEAN_D}_testSD" \
           --test_dataset ${TESTPATH_1} \
           --base_model ${BASE_MODEL} \
           --seed ${SEED} \
           --lora_weights ${OUTPUT_1} \
           --lora_start_layer ${START_LAYER}
           
       LORA_W+="${OUTPUT_1},"
     done
   done
 done
done