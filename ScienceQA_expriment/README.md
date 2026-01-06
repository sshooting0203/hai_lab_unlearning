The experiment on ScienceQA dataset.

## 1) Installation
You can install the required dependencies using the following command:
```
conda create -n o3 python=3.10
conda activate o3
pip install -r requiremets.txt
```

## 2) Preparing data
Generate the ScienceQA, CommonQA and OpenbookQA data with the following command:
```
python preprocess_SQ.py
```
Generating the unlearned data:
```
python preprocess_scienceqa.py
python preprocess_scienceqa_RD.py
python preprocess_scienceqa_SD.py
python preprocess_random_labeling.py
```
The generated data will saved at ``./data``.


## 3) Training the base model
Train the base model using the QA data from ScienceQA, CommonQA and OpenbookQA for following up unlearning:
```
torchrun --nproc_per_node=3 --nnodes=1 train_basemodel.py \
    --model_name_or_path NousResearch/Llama-2-7b-hf \
    --data_path ./data/qa_all/qaall_train.json \
    --bf16 True \
    --output_dir O3_LLAMA2_ScienceQA \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --tf32 True
```

CUDA_VISIBLE_DEVICES=0,1 nohup torchrun --nproc_per_node=2 --nnodes=1 train_basemodel.py \
  --model_name_or_path NousResearch/Llama-2-7b-hf \
  --data_path ./data/qa_all_clinc150_mrpc_rte_alpaca/qaall_train.json \
  --bf16 True \
  --output_dir O3_LLAMA2_ClincQA \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 2000 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --deepspeed "./configs/default_offload_opt_param.json" \
  --tf32 True \
  > O3_LLAMA2_ClincQA.nohup.log 2>&1 &

We have provided the model checkpoint at Huggingface: [gcyzsl/O3_LLAMA2_ScienceQA](https://huggingface.co/gcyzsl/O3_LLAMA2_ScienceQA).


## 4) Training OOD module
To train the OOD module, you can use the following command:
```
bash ./train_ood.sh
CUDA_VISIBLE_DEVICES=3 nohup bash train_ood_clinc.sh > ood_base_clinc.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup bash train_ood.sh > ood_11_all.log 2>&1 &
```

## 5) Training orthogonal-regularized LoRA
To unlearn with orthogonal-regularized LoRA, you can use the following command:
```
bash ./train_unlearn_lora.sh
CUDA_VISIBLE_DEVICES=3 nohup bash ./train_unlearn_lora.sh > unlearn_lora10.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup bash ./train_unlearn_lora2.sh > unlearn_lora10_r32.log 2>&1 &
```

## 6) Soft-weighted inference
To run the experiments, you can use the following command:
```
bash eval_soft_infer.sh
CUDA_VISIBLE_DEVICES=3 nohup bash ./eval_soft_infer.sh > eval_soft_infer10.log 2>&1 &
```
To gather the final results:
```
python read_results_oodlora.py
```

