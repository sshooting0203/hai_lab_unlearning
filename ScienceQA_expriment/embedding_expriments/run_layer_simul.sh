# 1. Stats 기반 layer search
python layer_search_stats.py \
  --jsonl ../data/scienceqa/splits_biology_pred/train_pred_cache.jsonl \
  --model meta-llama/Llama-3.1-8B \
  --out stats_biology_train.pt \
  --min_count 50 \
  --use_text_input \
  --strip_existing_answer_is \
  --use_margin \
  --margin_thresh 1.0

# 2. (선택) Simulation으로 상위 layer 검증
python layer_search_sim.py \
  --jsonl ../data/scienceqa/splits_biology_pred/train_pred_cache.jsonl \
  --model meta-llama/Llama-3.1-8B \
  --stats stats_biology_train.pt \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31\
  --alphas 3