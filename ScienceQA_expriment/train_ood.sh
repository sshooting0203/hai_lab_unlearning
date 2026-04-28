export PYTHONNOUSERSITE=1
# ## base
# for SEED in 0 1 2
# do
#   TYPE=""
#   for UNLEAN_D in "biology" "physics" "chemistry"
#   do
#       TYPE+="_${UNLEAN_D}"
#       OODPATH_1="./data/scienceqa_RD_5/scienceqa_not${TYPE}"
#       python train_ood.py \
#            --unlearn_dataset "scienceqa_${UNLEAN_D}" \
#            --ood_dataset "ood_scienceqa" \
#            --base_unlearn_path "./data/scienceqa/" \
#            --base_ood_path ${OODPATH_1} \
#            --seed ${SEED}
#   done
# done

### 3개의 ocsvm 생성하기(separate)
# for SEED in 0 1 2
# do
#   TYPE=""
#   for UNLEAN_D in "biology" "physics" "chemistry"
#   do
#       TYPE+="_${UNLEAN_D}"
#       SDPATH_1="./data/scienceqa_SD_only_10/scienceqa_${UNLEAN_D}"
#       OODPATH_1="./data/scienceqa_RD_only_10/scienceqa_not_${UNLEAN_D}"
#       python train_ood.py \
#            --unlearn_dataset "scienceqa_${UNLEAN_D}" \
#            --ood_dataset "ood_scienceqa" \
#            --base_unlearn_path ${SDPATH_1} \
#            --base_ood_path ${OODPATH_1} \
#            --seed ${SEED}
#   done
# done

### 11개의 ocsvm 생성하기(separate)
# for SEED in 0 1 2
# do
#   TYPE=""
#   for UNLEAN_D in "biology" "writing-strategies" "figurative-language" "units-and-measurement" "physics" "reference-skills" "chemistry" "punctuation" "economics" "grammar" "earth-science"
#   do
#       TYPE+="_${UNLEAN_D}"
#       SDPATH_1="./data/scienceqa_SD_only_10/scienceqa_${UNLEAN_D}"
#       OODPATH_1="./data/scienceqa_RD_only_10/scienceqa_not_${UNLEAN_D}"
#       python train_ood.py \
#            --unlearn_dataset "scienceqa_${UNLEAN_D}" \
#            --ood_dataset "ood_scienceqa" \
#            --base_unlearn_path ${SDPATH_1} \
#            --base_ood_path ${OODPATH_1} \
#            --seed ${SEED}
#   done
# done

### 1개의 ocsvm 생성하기(cumulative)
# for SEED in 0 1 2
# do
#   TYPE=""
#   for UNLEAN_D in "biology" "writing-strategies" "figurative-language" "units-and-measurement" "physics" "reference-skills" "chemistry" "punctuation" "economics" "grammar" "earth-science"
#   do
#       TYPE+="_${UNLEAN_D}"
#   done
#   SDPATH_1="./data/scienceqa_SD_10/scienceqa${TYPE}"
#   OODPATH_1="./data/scienceqa_RD_10/scienceqa_not${TYPE}"
#   python train_ood.py \
#        --unlearn_dataset "scienceqa" \
#        --ood_dataset "ood_scienceqa" \
#        --base_unlearn_path ${SDPATH_1} \
#        --base_ood_path ${OODPATH_1} \
#        --seed ${SEED}
# done

#================sciencQA====================
# biology                              1192
# writing-strategies                    888
# figurative-language                   739
# physics                               595
# units-and-measurement                 484
# reference-skills                      441
# chemistry                             403
# punctuation                           322
# grammar                               240
# economics                             237
# earth-science                         191
#================sciencQA=====================

#================clincQA=====================
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
#================clincQA=====================


## clinc ocsvm 생성하기(base, seperate)
# for SEED in 0 1 2
# do
#   TYPE=""
#   for domain in "${DOMAINS[@]}"
#   do
#       TYPE+="_${domain}"
#       OODPATH_1="./data/clincqa_RD_only_10/clincqa_not_${domain}"
#       python train_ood.py \
#            --unlearn_dataset "clincqa_${domain}" \
#            --ood_dataset "ood_clincqa" \
#            --base_unlearn_path "./data/clincqa/PROCESSED/" \
#            --base_ood_path ${OODPATH_1} \
#            --seed ${SEED}
#   done
# done


for SEED in 0 1 2
do
  TYPE=""
  for domain in "${DOMAINS[@]}"
  do
      TYPE+="_${domain}"
      OODPATH_1="./data_new/clincqa_RD_only_10/clincqa_not_${domain}"
      python train_ood.py \
           --unlearn_dataset "clincqa_${domain}" \
           --ood_dataset "ood_clincqa" \
           --base_unlearn_path "./data_new/clincqa_SD_only_10" \
           --base_ood_path ${OODPATH_1} \
           --seed ${SEED}
  done
done

# /dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new/clincqa_SD_only_10