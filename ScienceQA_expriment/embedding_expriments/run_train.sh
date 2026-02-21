CUDA_VISIBLE_DEVICES=0
python train_lora_random.py \
  --model meta-llama/Llama-3.1-8B \
  --train_jsonl ../data/scienceqa/splits_chemistry_pred/train_pred_cache.jsonl \
  --stats ../stats_pre_full.pt  \
  --out_dir ./out_lora_random \
  --lora_layer_start 10 \
  --lora_layer_end 11 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lam_repel 0.0 \
  --lora_r 64 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lr 1e-4 \
  --batch_size 4 \
  --grad_accum 4 \
  --max_steps 2000 \
  --device cuda \
  --dtype float16


