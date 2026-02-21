# 1. t개의 도메인 t개의 LoRA, no gate(무조건 LoRA 적용)
set -euo pipefail

PY=python
SCRIPT=dc.py
EXP_ROOT=/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code/distribution/llama3.1-8b-instruct
BASE_DATA_DIR="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data/scienceqa_random_force_10"
TASK_PREFIX="scienceqa"
DOMAINS=(
  "biology"
  "writing-strategies"
  "figurative-language"
  "units-and-measurement"
  "physics"
  "reference-skills"
  "chemistry"
  "punctuation"
  "economics"
  "grammar"
  "earth-science"
)

for SEED in 0 1 2; do
  echo "============= SEED=${SEED} ============="

  for i in "${!DOMAINS[@]}"; do
    domain="${DOMAINS[$i]}"
    t=$((i+1))

    # task_id="${TASK_PREFIX}_t${t}_${domain}"   # seed까지 붙여서 충돌 방지 추천
    task_id="${domain}"   # seed까지 붙여서 충돌 방지 추천
    data_name="${BASE_DATA_DIR}/scienceqa_${domain}_train_random_force.json"

    echo "==== seed=${SEED} t=${t} domain=${domain}"
    echo "task_id=${task_id}"
    echo "data_name=${data_name}"
    echo

    ${PY} ${SCRIPT} \
      seed=${SEED} \
      task_id=${task_id} \
      data_name=${data_name} \
      save_distribution_path="${EXP_ROOT}_mean/seed${SEED}" \
      load_distribution_path="${EXP_ROOT}_mean/seed${SEED}"

  done
done
