python eval_commonqa.py \
  --base meta-llama/Llama-3.1-8B \
  --lora ./out_lora_random/lora_adapter \
  --dataset_path ../data/commonqa/commonqa_all.hf \
  --dtype float16 \
  --max_items 200 \
  --max_new_tokens 32 \
  --max_prompt_tokens 2048 \
  --prompt_style answer_colon
