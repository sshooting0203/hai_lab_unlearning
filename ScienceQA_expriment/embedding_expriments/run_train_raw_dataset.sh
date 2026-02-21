CUDA_VISIBLE_DEVICES=0,1,2
python train_lora_raw_dataset.py \
  --dataset_path ../data/scienceqa/scienceqa_biology_all.hf \
  --split train \
  --stats ../stats_pre_full.pt  \
  --out_dir out_raw_train \
  --lam_repel 0.0 \
  --batch_size 4 --grad_accum 4 --max_steps 2000 \
  --target_modules q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_r 64 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lr 1e-4 \
  --lora_layer_start 10 --lora_layer_end 31


