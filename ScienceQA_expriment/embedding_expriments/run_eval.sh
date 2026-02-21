CUDA_VISIBLE_DEVICES=0
python eval_lora_acc.py \
  --base meta-llama/Llama-3.1-8B \
  --lora ./out_lora_random/lora_adapter \
  --jsonl ../data/scienceqa/splits_chemistry_pred/test_pred_cache.jsonl \
  --print_samples 5
