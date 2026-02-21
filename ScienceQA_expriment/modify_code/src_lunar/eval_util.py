# Copyright (c) Meta Platforms, Inc. and affiliates.
from tqdm import tqdm
import torch, math, os, json, evaluate, re, json
from rouge_score import rouge_scorer
from typing import List, Sequence, Dict, Any, Iterable
from torch.utils.data import DataLoader
from collections import Counter, defaultdict
from src_lunar.estimated_net_utils import LUNAR_LoRA_net
from src_lunar.gaussian_distribution.distribution_config import (
    align_distributions,
    collect_qv_feats_with_hooks, 
    compute_gate_vec_from_hidden, 
    load_many_task_distributions_qv
)
from src_lunar.data_loader import (
    get_batch_loss,
    custom_qa_collator,
    QAForgetEdgeDataset,
    QARetainedEdgeDataset,
    QAFactualDataset,
)
from src_lunar.utils.hook_utils import add_hooks, build_gate_pre_hooks

# <LUNAR에서 사용한 평가지표 6가지>
# Deviation Score
# ROUGE1
# Control score
# MRR(Mean reciprocal rank)
# THR(Top Hit Rate)
# ES(Extraction-Strength) Score


def eval_choice_hit_simple(
    gen_texts: List[str], # 생성 결과 
    gt_answers: List[str], # 각 샘플의 정답 
) -> Dict[str, float]:
    total = len(gen_texts)
    if total == 0:
        return {"choice_hit_rate": 0.0, "choice_hit": 0.0, "choice_total": 0.0}
    hits = 0
    for gen, gt in zip(gen_texts, gt_answers):
        gt = (gt or "").strip().upper() # gt에서 A~E 하나만 뽑음(가장 먼저 나오는 것)
        m = re.search(r"\b([A-E])\b", gt)
        if not m:
            continue
        ans = m.group(1)  # 'A'..'E', gen에 ans가 포함되면 hit (단독 토큰 기준)
        if re.search(rf"\b{re.escape(ans)}\b", (gen or "").upper()):
            hits += 1
    return {
        "choice_hit_rate": hits / total,
        "choice_hit": float(hits),
        "choice_total": float(total),
    }


# @ 수정 : gate_vec를 모두 remove
def clear_all_gates(module):
    for m in module.modules():
        if hasattr(m, "_gate_vec"):
            m._gate_vec = None


def eval_accuracy(logits, labels):
    preds = logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = shifted_labels != -100
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()
    return {"eval accuracy": acc.item()}


def eval_rouge_recall(gen_outputs, ground_truths):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_recall = []
    rougeL_recall = []
    for gen, gt in zip(gen_outputs, ground_truths):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall.append(rouge_scores["rouge1"].recall)
        rougeL_recall.append(rouge_scores["rougeL"].recall)
    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rougeL_recall}


def get_all_evals(
    cfg, 
    name,
    model, 
    tokenizer, 
    eval_dataloader, 
    q_dists=None, meta_dists=None,
    fwd_pre_hooks=[], fwd_hooks=[], 
    output_es_score=False,
    tau_by_layer=None, #
):
    eval_logs = {} # 최종 평가 지표 
    gen_outputs, ground_truths, input_strings = [], [], []
    num_token_gt_list, mrr_list, hit_rate_list, perplexity_list, es_score_list = [], [], [], [], []

    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask, topics = batch
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
    
        for k, v in batch.items(): # send to device
            batch[k] = v.to("cuda")

        with torch.no_grad():
            ###################################################
            # feats_dict, handles = collect_qv_feats_with_hooks(
            #     cfg, 
            #     model_base=model,                    
            #     layer_idxs=cfg.middle_distribution_layer,
            # )

            model._cur_attn_mask_2d = batch["attention_mask"]
            outputs = model.model( # @ TODO : 현재 코드는 평가 시에 forward 1회 무조건 수행 -> 실제 연산량 개선하려면 detect layer에 등록된 hook까지만 forward하고 stop하는 로직 따로 필요 
                **batch,
                output_hidden_states=False,  
                return_dict=True,
            ) # logit : torch.Size([16, 512, 32000]), hidden state : torch.Size([33, 16, 512, 4096])

            # for h in handles:
            #     h.remove()
            # layer_idxs = list(cfg.middle_distribution_layer) # [L, B, D]
            # q_feats_by_layer = {L: {"q": feats_dict[L]["q"], "mask": feats_dict[L].get("mask", None)}for L in layer_idxs}
            # v_feats_by_layer = {L: {"v": feats_dict[L]["v"], "mask": feats_dict[L].get("mask", None)}for L in layer_idxs}
            ###################################################

            input_string, gen_output, gt, scores, perplexity, es_score = run_generation( 
                cfg,
                batch,
                model,
                tokenizer=tokenizer,
                fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks,
                output_es_score=output_es_score,
                topics=topics, # 
            )
            mrr_per_batch, hit_rate_per_batch = compute_MRR(scores, gt, tokenizer)
            mrr_list.extend(mrr_per_batch)
            hit_rate_list.extend(hit_rate_per_batch)

            input_strings.extend(input_string)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            perplexity_list.append(perplexity)

            if output_es_score:
                es_score_list.extend(es_score)

        gt_loss = get_batch_loss(outputs.logits, batch["labels"]) 
        probabilities = torch.softmax(outputs.logits, dim=-1) 
        max_probs, _ = torch.max(probabilities, dim=-1)

        num_token_gt = (batch["labels"] != -100).sum(-1)
        num_token_gt_list.extend(num_token_gt)

        probs = [sum(max_probs[idx, :v]).item() for idx, v in enumerate(num_token_gt)]
        probs = [p / v.item() for p, v in zip(probs, num_token_gt)]

        eval_logs["gt_loss_per_token"] = (eval_logs.get("gt_loss_per_token", []) + (gt_loss / num_token_gt).float().cpu().numpy().tolist())
        eval_logs["gt_loss"] = eval_logs.get("gt_loss", []) + gt_loss.tolist()
        eval_logs["probs"] = eval_logs.get("probs", []) + probs

    clear_all_gates(model.model)  # @ 수정 : gate clear : model.model : LlamaForCausalLM
    eval_logs["num_token_gt"] = (eval_logs.get("num_token_gt", []) + num_token_gt.tolist())
    eval_logs["mrr"] = eval_logs.get("mrr", []) + mrr_list
    eval_logs["hit_rate"] = eval_logs.get("hit_rate", []) + hit_rate_list
    eval_logs["perplexity"] = eval_logs.get("perplexity", []) + perplexity_list

    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths)) # @ 수정
    eval_logs.update(eval_choice_hit_simple(gen_outputs, ground_truths)) # @ 수정 : hit rate 계산 추가 

    if output_es_score:
        eval_logs["es_score"] = eval_logs.get("es_score", []) + es_score_list
    else:
         eval_logs["es_score"] = eval_logs.get("es_score", [])
    eval_logs["generated_text"] = list(zip(input_strings, gen_outputs, ground_truths)) # 입력, 출력, 정답 
    return eval_logs

def run_generation(
    cfg, 
    batch, 
    model, 
    tokenizer, 
    fwd_pre_hooks=[], fwd_hooks=[], 
    output_es_score=False,
    topics=None, #
):
    input_ids = batch["input_ids"] # torch.Size([16, 512])
    input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    if cfg.model_family == "llama3-8b-instruct":
        input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=False)  # skip special token was TRUE for llama2b
        split_symbol = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif cfg.model_family == "llama3.1-8b-instruct":
        input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=False) # skip special token was TRUE for llama2b
        split_symbol = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    elif cfg.model_family == "Qwen2-7B-Instruct" or cfg.model_family == "Qwen2.5-7B-Instruct":
        input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        split_symbol = "<|im_end|>\n<|im_start|>assistant\n"
    elif cfg.model_family == "gemma-7b-it":
        input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        split_symbol = "<end_of_turn>\n<start_of_turn>model\n"
    elif cfg.model_family == "zephyr-7b":
        input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        split_symbol = "<|assistant|>\n"
    else:
        input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        split_symbol = " [/INST]"

    # print("inp strings : ", input_strings[0])
    
    ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]
    input_strings = [s + split_symbol for s in input_strings]

    if cfg.model_family == "llama3-8b-instruct" or cfg.model_family == "llama3.1-8b-instruct": # 여기에서 eot_id 제거
        ground_truth = [ re.sub(r"(<\|eot_id\|>)+$", "", re.sub(r"\n\n", "", text)) for text in ground_truth ]
    elif cfg.model_family == "Qwen2-7B-Instruct" or cfg.model_family == "Qwen2.5-7B-Instruct":
        ground_truth = [ re.sub(r"(<\|im_end\|>)+$", "", re.sub(r"\n<\|im_start\|>assistant", "", text)) for text in ground_truth ]
    elif cfg.model_family == "gemma-7b-it":
        ground_truth = [ re.sub(r"(<eos>)+$", "", re.sub(r"\n<start_of_turn>model", "", text)) for text in ground_truth ]
    
    # print("ground truth : ", ground_truth) # Answer: The answer is B.

    print("topics : ", topics)
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = "left" 
    left_pad_tokenizer.padding_size = "longest"
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id
    inputs = left_pad_tokenizer.batch_encode_plus(
        input_strings, add_special_tokens=True, return_tensors="pt", padding=True
    ).to("cuda")

    # @ 수정 : 배치에 해당하는 gate 생성하고, forward에 (x,gate,alpha) 형태로 호출 
    # merged_pre_hooks = list(fwd_pre_hooks) 
    # gate_out= compute_gate_vec_from_hidden(
    #     cfg=cfg,
    #     q_feats=q_feats, # 특정 레이어 L에서 v feature vector를 뽑음 
    #     v_feats=v_feats, # 특정 레이어 L에서 q feature vector를 뽑음
    #     qv_dists=q_dists, # task t별 가우시안 분포
    #     meta_dists=meta_dists, 
    #     topics=topics,
    #     tau_by_layer=tau_by_layer,
    # )

    # START similarity evaluation
    # gate_vec = gate_out["gate_vec"]
    # alpha_best = gate_out["alpha_best"]
    # gate_pre_hooks = build_gate_pre_hooks(cfg, model.model, gate_vec, alpha_best)
    # merged_pre_hooks = merged_pre_hooks + gate_pre_hooks 

    # generate
    with add_hooks(
        module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks
    ):
        out = model._generate( 
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=cfg.eval_generation_max_length,  # @ 수정 : 입력 포함 길이라는데
            max_new_tokens=cfg.eval_generation_max_new_tokens,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            use_cache=True,
            pad_token_id=left_pad_tokenizer.eos_token_id,
            output_scores=True,  # return logits
            return_dict_in_generate=True,
        )
        output = model.model(**inputs, labels=inputs["input_ids"])
        strs = left_pad_tokenizer.batch_decode(
            out.sequences[:, inputs.input_ids.shape[-1] :], skip_special_tokens=True
        )
        scores = (
            out.scores
        )
        loss = output.loss
        perplexity = math.exp(loss.item())

    # compute the ES score
    if output_es_score:
        es_scores = compute_ES(cfg, model, tokenizer, inputs, ground_truth)
    else:
        es_scores = []
    return input_strings, strs, ground_truth, scores, perplexity, es_scores

def compute_ES( # @ 수정 : ES score(Batched Extraction-Strength) 계산(es_scores를 리턴 -> list[float] length B, each in [0, 1])
    cfg,
    model,
    tokenizer,
    inputs,
    targets,
    max_new_tokens: int = 256,
) -> List[float]:
    pad_id   = tokenizer.pad_token_id
    eos_id   = tokenizer.eos_token_id
    B        = inputs.input_ids.size(0)
    es_scores = []

    for i in range(B): # loop over each item in the batch
        x_ids: List[int] = inputs.input_ids[i].tolist()

        while x_ids and x_ids[0] == pad_id:
            x_ids.pop(0)

        y_ids: List[int] = tokenizer(
            targets[i],
            add_special_tokens=False
        ).input_ids
        L = len(y_ids)

        if L == 0:
            es_scores.append(0.0)
            continue

        found = False
        # try prefixes y[:k]  for k = 0 … L
        for k in range(L + 1):
            ctx_ids = x_ids + y_ids[:k]
            ctx = torch.tensor([ctx_ids], device='cuda')

            with torch.no_grad():
                out = model._generate(
                    input_ids=ctx,
                    attention_mask=None,
                    max_length=cfg.eval_generation_max_length,  # @ 수정 : 입력 포함 길이라는데
                    max_new_tokens=cfg.eval_generation_max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    num_return_sequences=1,
                    use_cache=True,
                    pad_token_id=eos_id,
                    output_scores=False,  # return logits
                    return_dict_in_generate=False,
                )
            seq_ids = out[0][len(ctx_ids): len(ctx_ids) + (L - k)]
            strs = tokenizer.decode(seq_ids.tolist(), skip_special_tokens=True)
            gen_ids = out[0][len(ctx_ids): len(ctx_ids) + (L - k)].tolist()
            if tokenizer.decode(gen_ids, skip_special_tokens=True) == tokenizer.decode(y_ids[k:], skip_special_tokens=True):
                es_scores.append(1.0 - k / L)
                found = True
                break
        if not found:                         # model never reproduced yk
            es_scores.append(0.0)
    return es_scores

def compute_MRR(scores, gt, tokenizer):
    ## gt is a list with length of batch size
    MRR_res = []
    hit_rate = []
    # Convert scores as tuple to torch tensors
    # Initialize an empty tensor of the desired shape, filled with zeros
    score_size = scores[0].shape[0]
    vocab_size = scores[0].shape[1]

    T = len(scores) # @ 수정 : 512 -> T
    logits = torch.zeros(score_size, T, vocab_size, device="cuda")

    # Iterate over the tuple of tensors and assign each to the correct position in the combined tensor
    for i, score_tensor in enumerate(scores):
        # print(f"Tensor {i}: shape {score_tensor.shape}, device {score_tensor.device}")
        logits[:, i, :] = score_tensor
    probabilities = torch.nn.functional.softmax(
        logits, dim=-1
    )  # torch.Size([16, 512, 32000])
    
    for i in range(len(gt)):
        probs_per_gt = probabilities[i]
        reciprocal_ranks = []
        hit_check = []

        # Tokenize the ground truth
        gt_indices = tokenizer.encode(gt[i], add_special_tokens=False)

        # @ 수정
        L = min(len(gt_indices), probs_per_gt.size(0))
        if L == 0:
            MRR_res.append(0.0)
            hit_rate.append(0.0)
            continue
        for j in range(L):
            gt_index = gt_indices[j]
            probs = probs_per_gt[j]                
            sorted_indices = probs.argsort(descending=True)
            positions = (sorted_indices == gt_index).nonzero(as_tuple=False)
            if positions.numel() == 0:
                rank = vocab_size + 1
            else:
                rank = positions[0].item() + 1
            reciprocal_ranks.append(1.0 / rank)
            hit_check.append(1 if rank <= 100 else 0)

        # for j, gt_index in enumerate(gt_indices):
        #     probs = probs_per_gt[j]  # len = 32000, # torch.Size([1, 32000])
        #     sorted_indices = probs.argsort(descending=True)
        #     positions = (sorted_indices == gt_index).nonzero()
        #     rank = positions[0].item() + 1
        #     reciprocal_rank = 1.0 / rank
        #     reciprocal_ranks.append(reciprocal_rank)
        #     if rank <= 100:
        #         hit_check.append(1)
        #     else:
        #         hit_check.append(0)

        MRR_res.append(sum(reciprocal_ranks) / len(reciprocal_ranks))
        hit_rate.append(sum(hit_check) / len(hit_check))
    return MRR_res, hit_rate


#==================load eval dataset=================
def get_dataloader(cfg, data_path, tokenizer, eval_target):
    if eval_target == "forget_edge":
        torch_format_dataset = QAForgetEdgeDataset( # dataset_format_converstion
            data_path,
            tokenizer=tokenizer,
            configs=cfg,
            max_length=512,
        )
    elif eval_target == "retained_edge":
        torch_format_dataset = QARetainedEdgeDataset( # dataset_format_converstion
            data_path,
            tokenizer=tokenizer,
            configs=cfg,
            max_length=512,
        )
    elif eval_target == "factual_edge":
        torch_format_dataset = QAFactualDataset( # dataset_format_converstion
            data_path,
            tokenizer=tokenizer,
            configs=cfg,
            max_length=512,
        )
    else:
        raise ValueError(f"Invalid eval_target")
    
    torch_format_dataset.data = torch_format_dataset.data.select(
        range(min(200, len(torch_format_dataset.data))) # @@@ : sample 200개만 뽑아서 평가
    )
    eval_dataloader = DataLoader(
        torch_format_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=custom_qa_collator,
    )
    print("[LODA EVAL DATA] from ", eval_target, "...", len(eval_dataloader.dataset)) 
    return eval_dataloader


def custom_evaluate(
    cfg, 
    name, 
    data_path, 
    tokenizer, 
    model, 
    eval_target, fwd_pre_hooks=[], fwd_hooks=[], 
    output_es_score=False
):
    eval_dataloader = get_dataloader( # @@@ : 평가 데이터 로드
        cfg=cfg, data_path=data_path, tokenizer=tokenizer, eval_target=eval_target
    )
    model._to("cuda")
    model._eval()
    
    q_dists = None
    meta_dists = None

    # dist_dict = load_many_task_distributions_qv( # 일단 이미 있는 distribution을 로드 
    #     load_dir=cfg.load_distribution_path,
    #     task_ids=cfg.gate_task_ids,  
    #     device="cuda",
    #     dtype=torch.float32,
    # ) 
    # q_dists, meta_dists = align_distributions( # q,v의 분포가 2개 x task는 11개
    #     dist_dict,
    #     layer_idxs=cfg.middle_distribution_layer,
    #     which=cfg.gate_which,        
    # )
    # tau_by_layer = dist_dict.get("tau_by_layer", None)
    eval_logs = get_all_evals(
        cfg=cfg,
        name=name,
        model=model,
        tokenizer=tokenizer,
        eval_dataloader=eval_dataloader,
        fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks,
        output_es_score=output_es_score,
    )
    eval_logs = summarize_eval_logs(eval_logs) # @ 수정
    return eval_logs

###########################################
def summarize_value(v):
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return "N/A"
        if all(isinstance(x, (int, float, bool)) for x in v):
            return float(sum(v)) / len(v) # 리스트지만 숫자 리스트가 아니면 유지 (예: per_sample, per_topic 등)
        return v
    return v

def summarize_eval_logs(eval_logs: dict):
    skip_keys = {"generated_text"}
    for k, v in list(eval_logs.items()):
        if k in skip_keys:
            continue
        eval_logs[k] = summarize_value(v)
    return eval_logs

# def generate_and_save_completions_for_dataset(
#     cfg, model_base, fwd_pre_hooks, fwd_hooks, dataset=None, save_file=None
# ):
#     for d in dataset:
#         d["instruction"] = d.pop("question")

#     completions = model_base.generate_completions(
#         dataset,
#         fwd_pre_hooks=fwd_pre_hooks,
#         fwd_hooks=fwd_hooks,
#         max_new_tokens=cfg.max_new_tokens,
#     )
#     if not os.path.exists(os.path.join("unlearning", "completions")):
#         os.makedirs(os.path.join("unlearning", "completions"))
#     print(f"Saving completions to {save_file}")
#     with open(save_file, "w") as f:
#         json.dump(completions, f, indent=4)
