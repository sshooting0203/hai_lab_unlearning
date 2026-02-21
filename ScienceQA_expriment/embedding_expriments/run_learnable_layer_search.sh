SUBJECTS=(
  biology
  chemistry
  earth-science
  # figurative-language
  # grammar
  physics
  punctuation
  # reference-skills
  units-and-measurement
  # writing-strategies
)

for SUBJECT in "${SUBJECTS[@]}"; do
  python learnable_layer_search.py \
    --model meta-llama/Llama-3.1-8B \
    --forget_json ../data/scienceqa/splits_${SUBJECT}_pred/train_pred_cache.jsonl \
    --out layer_biology.json \
    --batch_size 20 \
    --max_samples 200 \
    --dtype bf16 \
    --target_mode gt \
    --param_scope full \
    --start_layer 8 \
    --end_layer 31 \
    --restrict_to_valid_letters \
    --topk 20
done