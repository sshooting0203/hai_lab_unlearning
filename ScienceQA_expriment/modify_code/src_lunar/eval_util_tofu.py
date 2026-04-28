# Copyright (c) Meta Platforms, Inc. and affiliates.
from tqdm import tqdm
import torch, math, os, json, evaluate, re
from rouge_score import rouge_scorer
from typing import List, Dict, Optional, Any, Iterable
from torch.utils.data import DataLoader
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from src_lunar.data_loader import (
    get_batch_loss,
    custom_tofu_collator,
    TofuForgetEdgeDataset,
    TofuRetainedEdgeDataset,
    TofuFactualDataset,
)
from src_lunar.utils.hook_utils import add_hooks
from src_lunar.utils.model_utils import extract_instruction_from_model 
from src_lunar.model_utils.llama31model import build_prompt_from_instruction, LLAMA3_CHAT_TEMPLATE
from src_lunar.estimated_net_moe_utils import LUNAR_LoRA_MoE_net, LUNAR_NN_MoE_net

# TOFU 데이터셋 평가 시, 필요한 key들
# QA_dict["question"].append(item["question"]) 
# QA_dict["answer"].append(item["answer"])
# QA_dict["topic"].append(item["topic"]) -> domain

# QA_dict["paraphrased_answer"].append(item["paraphrased_answer"])
# QA_dict["perturbed_answer"].append(item["perturbed_answer"])
# QA_dict["paraphrased_question"].append(item["paraphrased_question"])

# QA_dict["author_idx"].append(item["author_idx"])
# QA_dict["forget_author"].append(item["forget_author"])

def _normalize_text(text: str) -> str:
    # Remove any unwanted characters and normalize whitespace
    return " ".join(text.split()).lower()

def _normalize_for_match(text: str) -> str:
    # More aggressive normalization for exact/contains match
    text = _normalize_text(text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def _strip_assistant_prefix(text: str) -> str:
    # Llama 3 specific: remove assistant header if present
    assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    if text.startswith(assistant_header):
        return text[len(assistant_header):]
    return text

def safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s

def get_eval_choice_hit(cfg):
    task = str(cfg.data_name).lower()
    if task == "scienceqa":
        from src_lunar.utils.scienceQA import eval_choice_hit
    elif task == "clincqa":
        from src_lunar.utils.clincQA import eval_choice_hit
    elif task == "tofu":
        from src_lunar.utils.tofu import eval_choice_hit
    else:
        raise ValueError(f"Unknown task={cfg.task}")
    return eval_choice_hit

# @ 수정 : gate_vec를 모두 remove
def clear_all_gates(module):
    for m in module.modules():
        if hasattr(m, "_gate_vec"):
            m._gate_vec = None


def eval_accuracy(logits, labels):
    # Answer: The answer is B.
    preds = logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = shifted_labels != -100
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()
    return {"eval accuracy": acc.item()}


def eval_rouge_recall(gen_outputs, ground_truths):
    # Answer: The answer is B.
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_recall = []
    rougeL_recall = []
    for gen, gt in zip(gen_outputs, ground_truths):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall.append(rouge_scores["rouge1"].recall)
        rougeL_recall.append(rouge_scores["rougeL"].recall)
    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rougeL_recall}

# TOFU Specific Metrics
def exact_match(prediction: str, target: str) -> float:
    return 1.0 if _normalize_for_match(prediction) == _normalize_for_match(target) else 0.0

def contains_match(prediction: str, target: str) -> float:
    return 1.0 if _normalize_for_match(target) in _normalize_for_match(prediction) else 0.0

def token_f1(prediction: str, target: str) -> float:
    prediction_tokens = _normalize_text(prediction).split()
    target_tokens = _normalize_text(target).split()
    if not prediction_tokens or not target_tokens:
        return 0.0
    
    common = Counter(prediction_tokens) & Counter(target_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(target_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def rouge_l_recall(prediction: str, target: str, scorer: Optional[rouge_scorer.RougeScorer] = None) -> float:
    if scorer is None:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(target, prediction)
    return scores['rougeL'].recall

def compute_embedding_scores(predictions: List[str], targets: List[str], embed_model: SentenceTransformer) -> List[float]:
    if not predictions:
        return []
    pred_embeds = embed_model.encode(predictions, convert_to_tensor=True, show_progress_bar=False)
    target_embeds = embed_model.encode(targets, convert_to_tensor=True, show_progress_bar=False)
    cosine_scores = util.cos_sim(pred_embeds, target_embeds)
    return cosine_scores.diag().tolist()

def best_metric_against_list(prediction: str, target_list: List[str], metric_fn) -> float:
    if not target_list:
        return 0.0
    return max(metric_fn(prediction, target) for target in target_list)

def compute_answer_prob(model, tokenizer, instruction: str, answer: str) -> float:
    # print("compute_answer_prob model is : ", model)
    prompt = build_prompt_from_instruction(tokenizer, instruction)
    full_text = prompt + answer
    
    inputs = tokenizer(full_text, return_tensors="pt")
    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = prompt_inputs.input_ids.shape[1]

    base_model = model.model if hasattr(model, "model") else model
    device = base_model.get_input_embeddings().weight.device
    inputs = tokenizer(full_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = base_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits
        input_ids = inputs["input_ids"]
    
    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., prompt_len-1:-1, :].contiguous()
    shift_labels = input_ids[..., prompt_len:].contiguous()
    
    # Calculate log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather log probs for the actual tokens
    token_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Return average probability across answer tokens
    return torch.exp(token_log_probs.mean()).item()

def mean_prob_against_list(model, tokenizer, instruction: str, answer_list: List[str]) -> float:
    if not answer_list:
        return 0.0
    probs = [compute_answer_prob(model, tokenizer, instruction, ans) for ans in answer_list]
    return safe_mean(probs)

def best_prob_against_list(model, tokenizer, instruction: str, answer_list: List[str]) -> float:
    if not answer_list:
        return 0.0
    probs = [compute_answer_prob(model, tokenizer, instruction, ans) for ans in answer_list]
    return max(probs)

#########################################
def build_prompt_only_batch(batch, tokenizer):
    input_ids = batch["input_ids"]          # (B, S)
    labels = batch["labels"]                # (B, S)
    attention_mask = batch["attention_mask"]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    prompt_input_ids_list = []
    prompt_attention_mask_list = []

    B = input_ids.size(0)
    for i in range(B):
        ids_i = input_ids[i]
        labels_i = labels[i]
        attn_i = attention_mask[i]

        valid_len = int(attn_i.sum().item())
        ids_i = ids_i[:valid_len]
        labels_i = labels_i[:valid_len]

        prompt_mask = labels_i.eq(-100)
        if not prompt_mask.any():
            raise ValueError(f"[build_prompt_only_batch] sample {i}: no prompt tokens found")

        prompt_ids = ids_i[prompt_mask]
        prompt_input_ids_list.append(prompt_ids)
        prompt_attention_mask_list.append(torch.ones_like(prompt_ids))

    max_prompt_len = max(x.size(0) for x in prompt_input_ids_list)

    padded_input_ids = []
    padded_attention_mask = []

    for ids_i, mask_i in zip(prompt_input_ids_list, prompt_attention_mask_list):
        pad_len = max_prompt_len - ids_i.size(0)
        padded_input_ids.append(
            torch.cat([
                torch.full((pad_len,), pad_id, dtype=torch.long, device=ids_i.device),
                ids_i
            ], dim=0)
        )
        padded_attention_mask.append(
            torch.cat([
                torch.zeros(pad_len, dtype=torch.long, device=mask_i.device),
                mask_i
            ], dim=0)
        )
    prompt_input_ids = torch.stack(padded_input_ids, dim=0).to(input_ids.device)
    prompt_attention_mask = torch.stack(padded_attention_mask, dim=0).to(input_ids.device)
    return {
        "input_ids": prompt_input_ids,
        "attention_mask": prompt_attention_mask,
    }


def _majority_vote_int(values):
    if len(values) == 0:
        return None
    return Counter(values).most_common(1)[0][0]


def get_layer_input_cache(cache_list):
    def hook_fn(module, input):
        # input[0]: [B, S, D]
        cache_list.append(input[0].detach())
    return hook_fn


def extract_eval_activations_BD(
    model,
    prompt_batch,
    moe_layers,
):
    cache_per_layer = [[] for _ in moe_layers]

    hooks = []
    for i, m in enumerate(moe_layers):
        hooks.append((m, get_layer_input_cache(cache_per_layer[i])))

    with torch.no_grad():
        with add_hooks(module_forward_pre_hooks=hooks, module_forward_hooks=[]):
            _ = model.model(
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                output_hidden_states=False,
                return_dict=True,
            )

    activations_per_layer = []
    attn = prompt_batch["attention_mask"]  # [B, S]

    for i in range(len(moe_layers)):
        x = cache_per_layer[i][0]  # [B, S, D]
        B, S, D = x.shape
        
        # 각 sample의 마지막 유효 prompt token 인덱스
        last_idx = attn.sum(dim=1) - 1
        
        # 마지막 토큰 위치의 activation만 추출: [B, D]
        x_last = x[torch.arange(B, device=x.device), last_idx]
        activations_per_layer.append(x_last)
    return activations_per_layer


def compute_router_accuracy_BD(
    name,
    model,
    tokenizer,
    batch_gpu,
    topics,
    layer_idx_list,
    forget_label_idx=1,
):
    """
    Computes router accuracy by extracting activations [B, D] at the last token
    and feeding them directly into the MoE gate.
    """
    prompt_batch = build_prompt_only_batch(batch_gpu, tokenizer)
    moe_layers = [model.model_block_modules[i].mlp.down_proj for i in layer_idx_list]
    activations_per_layer = extract_eval_activations_BD(model, prompt_batch, moe_layers)

    B = prompt_batch["input_ids"].size(0)
    per_sample_acc = []
    per_sample_info = []
    is_forget = "forget" in name.lower()

    # layer별 logits/예측
    all_layer_preds = []
    for moe_layer, x_bd in zip(moe_layers, activations_per_layer):
        with torch.no_grad():
            _, logits = moe_layer(x_bd, return_router_logits=True)  # logits: [B, E]
        preds = logits.argmax(dim=-1).tolist()
        all_layer_preds.append(preds)

    for b in range(B):
        per_layer_preds = [layer_preds[b] for layer_preds in all_layer_preds]
        seq_pred = _majority_vote_int(per_layer_preds)

        if not is_forget:
            target_expert = 0
        else:
            # 도메인별 타겟 전문가 결정 로직 (TOFU용)
            topic_str = str(topics[b])
            target_expert = None
            if topic_str.startswith('AG'):
                try:
                    target_expert = int(topic_str[2:])
                except (ValueError, TypeError):
                    pass
    
        is_correct = float(seq_pred == target_expert)
        per_sample_acc.append(is_correct)
        per_sample_info.append({
            "router_pred_expert": seq_pred,
            "router_target_expert": target_expert,
            "router_layer_preds": per_layer_preds,
        })

    return per_sample_acc, per_sample_info


def get_all_evals(
    cfg, 
    name,
    model, 
    tokenizer, 
    eval_dataloader, 
    layer_idx_list,
    forget_label_idx=1,
    fwd_pre_hooks=[], fwd_hooks=[], 
    output_es_score=False,
):
    eval_logs = {} # 최종 평가 지표 
    gen_outputs, ground_truths, input_strings = [], [], []
    num_token_gt_list, mrr_list, hit_rate_list, perplexity_list, es_score_list = [], [], [], [], []
    topics_list, expert_probs_list, all_perturbed_answers = [], [], []
    eval_hit = get_eval_choice_hit(cfg)

    #=====================MoE Prob Hook Setup==============
    moe_gates = []
    for m in model.model.modules():
        if isinstance(m, (LUNAR_LoRA_MoE_net, LUNAR_NN_MoE_net)):
            moe_gates.append(m.gate)
    
    current_batch_logits = []
    def moe_hook(mod, inp, out):
        current_batch_logits.append(out.detach())
    handles = [g.register_forward_hook(moe_hook) for g in moe_gates]
    #=====================MoE Prob Hook Setup==============
    
    #=====================TOFU metrics=====================
    tofu_metrics = {
        "exact_match": [], "token_f1": [], "rouge_l_recall": [],
        "best_exact_match": [], "best_token_f1": [], "best_rouge_l_recall": [],
        "avg_prob": [], "best_prob": [], "truth_ratio": []
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer("all-mpnet-base-v2").to(device)
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    #=====================TOFU metrics=====================
    with tqdm(total=len(eval_dataloader), desc=f"Eval {name}") as pbar:
        for batch in eval_dataloader:
            input_ids, labels, attention_mask, topics, gts, paraphrased_answers, perturbed_answers, paraphrased_questions, author_idxs, forget_authors = batch
            topics_list.extend(topics)

            batch_gpu = {
                "input_ids": input_ids.to("cuda"),
                "labels": labels.to("cuda"),
                "attention_mask": attention_mask.to("cuda"),
            }

            with torch.no_grad():
                # ================== Measure router accuracy ==================
                router_acc_batch, router_info_batch = compute_router_accuracy_BD(
                    name=name,
                    model=model,
                    tokenizer=tokenizer,
                    batch_gpu=batch_gpu,
                    topics=topics,
                    layer_idx_list=layer_idx_list,
                    forget_label_idx=forget_label_idx,
                )
                eval_logs["router_accuracy"] = eval_logs.get("router_accuracy", []) + router_acc_batch
                if "router_info" not in eval_logs:
                    eval_logs["router_info"] = []
                eval_logs["router_info"].extend(router_info_batch)
                # =============================================================

                current_batch_logits.clear()

                model._cur_attn_mask_2d = batch_gpu["attention_mask"]
                outputs = model.model(
                    **batch_gpu,
                    output_hidden_states=False,
                    return_dict=True,
                )
                
                # ================== Measure expert probs ==================
                # if current_batch_logits:
                #     all_logits = torch.stack(current_batch_logits)   # (L, B, S, E) : torch.Size([1, 16, 512, 2])
                #     all_probs = torch.softmax(all_logits, dim=-1)    # (L, B, S, E) : torch.Size([1, 16, 512, 2])

                #     # layer 평균
                #     mean_layer_probs = all_probs.mean(dim=0)         # (B, S, E)

                #     # mask 생성 (padding 제외)
                #     attn_mask = batch_gpu["attention_mask"].unsqueeze(-1) # (B, S, 1)
                #     full_mask = attn_mask.float()
                #     prompt_mask = ((batch_gpu["labels"] == -100).unsqueeze(-1) & (attn_mask == 1)).float()
                #     answer_mask = ((batch_gpu["labels"] != -100).unsqueeze(-1) & (attn_mask == 1)).float()

                #     def masked_avg_probs(probs, mask):
                #         sum_probs = (probs * mask).sum(dim=1)        # (B, E)
                #         counts = mask.sum(dim=1)                     # (B, 1)
                #         # print("[DEBUG] token counts : ", counts)
                #         avg_probs = sum_probs / (counts + 1e-9)      # (B, E)
                #         return avg_probs

                #     full_avg = masked_avg_probs(mean_layer_probs, full_mask).cpu()
                #     prompt_avg = masked_avg_probs(mean_layer_probs, prompt_mask).cpu()
                #     answer_avg = masked_avg_probs(mean_layer_probs, answer_mask).cpu()

                #     # 각 token에서 가장 높은 expert index 
                #     token_top1 = mean_layer_probs.argmax(dim=-1).cpu()   # (B, S)

                #     B = full_avg.size(0)
                #     for i in range(B):
                #         full_list = full_avg[i].tolist()
                #         prompt_list = prompt_avg[i].tolist()
                #         answer_list = answer_avg[i].tolist()

                #         full_top = int(full_avg[i].argmax().item())
                #         prompt_top = int(prompt_avg[i].argmax().item())
                #         answer_top = int(answer_avg[i].argmax().item())

                #         expert_probs_list.append({
                #             "full_avg_probs": full_list,
                #             "full_top_expert": full_top,
                #             "prompt_avg_probs": prompt_list,
                #             "prompt_top_expert": prompt_top,
                #             "answer_avg_probs": answer_list,
                #             "answer_top_expert": answer_top,
                #         })
                # ================== Measure expert probs ==================
                moe_layers = []
                for m in model.model.modules():
                    if isinstance(m, (LUNAR_LoRA_MoE_net, LUNAR_NN_MoE_net)):
                        m.record_expert = True
                        m.expert_history = []
                        moe_layers.append(m)
                # print("moe_layers len : ", len(moe_layers))
                # print("moe_layer element : ", moe_layers[0])

                #========== Measure expert props for this batch==========
                input_string, gen_output, _, _, es_score = run_generation( 
                    cfg,
                    batch_gpu,
                    model,
                    tokenizer=tokenizer,
                    fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks,
                    output_es_score=output_es_score,
                    topics=topics, # 
                )
                # print("input_strings : ", input_string)  # <|begin_of_text|><|start_header_id|>user<|end_header_id|>\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nContext: Read the description of a trait.\nKenneth has pale skin.\nQuestion: What information supports the conclusion that Kenneth inherited this trait?\nOptions: (A) Kenneth and his father both have dark hair. (B) Kenneth's parents have pale skin. They passed down this trait to Kenneth.\n\n\n### Response:\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n
                # print("gen_output : ", gen_output, ", gen_output.shape : ", gen_output.shape) # (B) Kenneth's parents have pale skin. They passed down this trait to Kenneth
                # print("gt : ",gt) # Answer: The answer is B
                # print("scores : ",len(scores))
                #========== Measure expert props for this batch==========

                
                #==================generation moe expert==========================
                B_curr = input_ids.size(0)
                if moe_layers and len(moe_layers[0].expert_history) > 1:
                    L_layers = len(moe_layers)
                    for b in range(B_curr):
                        sample_expert_seqs = []
                        for l in range(L_layers):
                            layer_history = moe_layers[l].expert_history[1:]
                            chosen_experts = [int(h[b].item()) for h in layer_history if h.size(1) == 1]
                            sample_expert_seqs.append(chosen_experts)
                            
                #         expert_probs_list.append({
                #             "gen_expert_sequence": sample_expert_seqs
                #         })
                # else:
                #     for _ in range(B_curr):
                #         expert_probs_list.append({"gen_expert_sequence": []})

                # 다음 배치를 위해 비활성화 및 초기화
                for m in moe_layers:
                    m.record_expert = False
                    m.expert_history = []
                #==================generation moe expert==========================

                perplexity = math.exp(outputs.loss.item())
                mrr_per_batch, hit_rate_per_batch = compute_MRR(outputs.logits, batch_gpu["labels"]) # @ 수정 : model.generate() -> model.model()
                mrr_list.extend(mrr_per_batch)
                hit_rate_list.extend(hit_rate_per_batch)

                input_string = [extract_instruction_from_model(x, model) for x in input_string]
                input_strings.extend(input_string)
                gen_outputs.extend(gen_output)
                ground_truths.extend(gts)
                all_perturbed_answers.extend(perturbed_answers)
                perplexity_list.append(perplexity)

                if output_es_score:
                    es_score_list.extend(es_score)
            ###############################################################
                #=====================TOFU metrics=====================
                for i in range(len(gen_output)):
                    pred = _strip_assistant_prefix(gen_output[i])
                    gt_main = gts[i]
                    paraphrased_ans = paraphrased_answers[i]
                    perturbed_ans_list = perturbed_answers[i]
                    inst = input_string[i]
                    
                    # Candidates for "best" metrics
                    gold_candidates = [gt_main]
                    if paraphrased_ans:
                        gold_candidates.append(paraphrased_ans)
                    
                    # 1. String Metrics against main GT
                    tofu_metrics["exact_match"].append(exact_match(pred, gt_main)) # 생성된 답변과 GT가 완전히 같은지 
                    tofu_metrics["token_f1"].append(token_f1(pred, gt_main)) # 생성된 답변과 GT 사이의 토큰 단위 겹쳐지는 F1 score
                    tofu_metrics["rouge_l_recall"].append(rouge_l_recall(pred, gt_main, scorer=rouge_scorer_obj)) # 생성된 답변이 GT를 얼마나 회수했는지(Rouge-L recall)
                    
                    # 2. Best String Metrics against gold candidates
                    tofu_metrics["best_exact_match"].append(best_metric_against_list(pred, gold_candidates, exact_match)) # GT + paraphrased_ans 중 가장 잘 맞는 후보 기준 exact match 
                    tofu_metrics["best_token_f1"].append(best_metric_against_list(pred, gold_candidates, token_f1))  # GT + paraphrased_ans 중 가장 잘 맞는 후보 기준 token f1
                    tofu_metrics["best_rouge_l_recall"].append(best_metric_against_list(pred, gold_candidates, lambda p, t: rouge_l_recall(p, t, scorer=rouge_scorer_obj))) # GT + paraphrased_ans 중 가장 잘 맞는 후보 기준 rouge l recall
                    
                    # 3. Probability Metrics
                    avg_gold_prob = mean_prob_against_list(model, tokenizer, inst, gold_candidates)
                    best_perturbed_prob = best_prob_against_list(model, tokenizer, inst, perturbed_ans_list)
                    
                    tofu_metrics["avg_prob"].append(avg_gold_prob) # 질문 inst가 주어졌을 때 GT + paraphrased_ans에 대해 모델이 주는 확률의 평균 
                    tofu_metrics["best_prob"].append(best_prob_against_list(model, tokenizer, inst, gold_candidates)) # GT + paraphrased_ans 중 가장 확률이 높은 후보 기준 
                    
                    # Truth Ratio: avg_gold_prob / (avg_gold_prob + best_perturbed_prob)
                    truth_ratio = avg_gold_prob / (avg_gold_prob + best_perturbed_prob + 1e-9) # 확률 비교
                    tofu_metrics["truth_ratio"].append(truth_ratio)

            # --- Embedding Similarity Score ---
            if gen_output:
                # Strip prefixes for embedding
                clean_preds = [_strip_assistant_prefix(p) for p in gen_output]
                # Compute similarity against main GT
                sim_scores = compute_embedding_scores(clean_preds, gts, embed_model)
                tofu_metrics["embedding_similarity"] = tofu_metrics.get("embedding_similarity", []) + sim_scores # 생성된 답변과 GT 사이의 임베딩 유사도 
                
                # Compute best similarity against gold candidates
                best_sims = []
                for i, pred in enumerate(clean_preds):
                    candidates = [gts[i]]
                    if paraphrased_answers[i]:
                        candidates.append(paraphrased_answers[i])
                    best_sim = max(compute_embedding_scores([pred]*len(candidates), candidates, embed_model))
                    best_sims.append(best_sim)
                tofu_metrics["best_embedding_similarity"] = tofu_metrics.get("best_embedding_similarity", []) + best_sims # GT + paraphrased_ans 중 가장 임베딩 유사도가 높은 후보 기준 
                
            #=====================TOFU metrics=====================
            gt_loss = get_batch_loss(outputs.logits, batch_gpu["labels"]) # model.model()
            probabilities = torch.softmax(outputs.logits, dim=-1) 
            max_probs, _ = torch.max(probabilities, dim=-1) # max_probs : [16, 512]

            num_token_gt = (batch_gpu["labels"] != -100).sum(-1) # 8개의 gt token
            num_token_gt_cpu = num_token_gt.detach().cpu().tolist()
            num_token_gt_list.extend(num_token_gt_cpu)

            # probs = [sum(max_probs[idx, :v]).item() for idx, v in enumerate(num_token_gt)]  # @ 수정 : 뒤에서 v 잘라내면 gt token인 것을 보장할 수 없음
            probs = []
            for i in range(max_probs.size(0)):
                m = (batch_gpu["labels"][i] != -100)          # (S,)
                if m.any():
                    probs.append(max_probs[i][m].mean().item())
                else:
                    raise ValueError(f"m is empty")

            eval_logs["gt_loss_per_token"] = (eval_logs.get("gt_loss_per_token", []) + (gt_loss / num_token_gt).float().cpu().numpy().tolist())
            eval_logs["gt_loss"] = eval_logs.get("gt_loss", []) + gt_loss.tolist()
            eval_logs["probs"] = eval_logs.get("probs", []) + probs
            pbar.update(1)

    for h in handles:
        h.remove()
    
    for k, v in tofu_metrics.items():
        eval_logs[k] = eval_logs.get(k, []) + v

    eval_logs["num_token_gt"] = (eval_logs.get("num_token_gt", []) + num_token_gt_list)
    eval_logs["mrr"] = eval_logs.get("mrr", []) + mrr_list
    eval_logs["hit_rate"] = eval_logs.get("hit_rate", []) + hit_rate_list
    eval_logs["perplexity"] = eval_logs.get("perplexity", []) + perplexity_list
    
    # if topics_list:
    #     eval_logs["topic"] = topics_list
    # if expert_probs_list:
    #     eval_logs["expert_probs"] = expert_probs_list

    # eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths))
    # print("gen_outputs : ", gen_outputs, ",ground_truths : ", ground_truths)
    hit_logs = eval_hit(gen_outputs, ground_truths, perturbed_answers_list=all_perturbed_answers, embed_model=embed_model, name=name)
    eval_logs.update(hit_logs)

    if output_es_score:
        eval_logs["es_score"] = eval_logs.get("es_score", []) + es_score_list
    else:
        eval_logs["es_score"] = eval_logs.get("es_score", [])

    if topics_list:
        # router_info도 함께 zip에 포함
        router_info_list = eval_logs.get("router_info", [None] * len(input_strings))
        eval_logs["generated_text"] = list(zip(input_strings, gen_outputs, ground_truths, topics_list, router_info_list))
    else:
        eval_logs["generated_text"] = list(zip(input_strings, gen_outputs, ground_truths))

    # failure case collect
    eval_results = hit_logs.get("eval_results", [])
    if eval_results and "generated_text" in eval_logs:
        failure_samples = []
        is_forget = "forget" in name.lower()
        for i, is_hit in enumerate(eval_results):
            if (is_forget and is_hit) or (not is_forget and not is_hit):
                failure_samples.append(eval_logs["generated_text"][i])
        eval_logs["failure_text"] = failure_samples
        eval_logs["num_failure"] = len(failure_samples)
        eval_logs["total_generated"] = len(eval_results)

    # results 항목에서 개별 리스트인 router_info 등은 제거 (generated_text에 포함되어 있으므로)
    for k in ["router_info", "expert_probs", "topic"]:
        if k in eval_logs:
            del eval_logs[k]

    return eval_logs


def run_generation(
    cfg, 
    batch, 
    model, 
    tokenizer, 
    fwd_pre_hooks=[], fwd_hooks=[], 
    output_es_score=False,
    topics=None,
):
    input_ids = batch["input_ids"]          # (B, S)
    labels = batch["labels"]                # (B, S)
    attention_mask = batch["attention_mask"]

    prompt_input_ids_list = []
    prompt_attention_mask_list = []
    ground_truth = []

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    B = input_ids.size(0)

    for i in range(B):
        ids_i = input_ids[i]
        labels_i = labels[i]
        attn_i = attention_mask[i]

        # 실제 토큰 길이
        valid_len = int(attn_i.sum().item())
        ids_i = ids_i[:valid_len]
        labels_i = labels_i[:valid_len]

        # prompt 구간: labels == -100
        prompt_mask = labels_i.eq(-100)
        answer_mask = labels_i.ne(-100)

        if not prompt_mask.any():
            raise ValueError(f"[run_generation] sample {i}: no prompt tokens found")
        if not answer_mask.any():
            raise ValueError(f"[run_generation] sample {i}: no answer tokens found")

        prompt_ids = ids_i[prompt_mask]
        gt_ids = ids_i[answer_mask]

        prompt_input_ids_list.append(prompt_ids)
        prompt_attention_mask_list.append(torch.ones_like(prompt_ids))
        ground_truth.append(
            normalize_text(tokenizer.decode(gt_ids.tolist(), skip_special_tokens=True))
        )

    # left padding for generation
    max_prompt_len = max(x.size(0) for x in prompt_input_ids_list)

    padded_input_ids = []
    padded_attention_mask = []

    for ids_i, mask_i in zip(prompt_input_ids_list, prompt_attention_mask_list):
        pad_len = max_prompt_len - ids_i.size(0)
        padded_input_ids.append(
            torch.cat([
                torch.full((pad_len,), pad_id, dtype=torch.long, device=ids_i.device),
                ids_i
            ], dim=0)
        )
        padded_attention_mask.append(
            torch.cat([
                torch.zeros(pad_len, dtype=torch.long, device=mask_i.device),
                mask_i
            ], dim=0)
        )

    gen_input_ids = torch.stack(padded_input_ids, dim=0).to(input_ids.device)
    gen_attention_mask = torch.stack(padded_attention_mask, dim=0).to(input_ids.device)

    input_strings = tokenizer.batch_decode(gen_input_ids, skip_special_tokens=False)

    with add_hooks(
        module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks
    ):
        out = model._generate(
            input_ids=gen_input_ids,
            attention_mask=gen_attention_mask,
            max_length=cfg.eval_generation_max_length,
            max_new_tokens=cfg.eval_generation_max_new_tokens,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            use_cache=True,
            pad_token_id=pad_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

        gen_ids = out.sequences[:, gen_input_ids.shape[-1]:]
        strs = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        strs = [normalize_text(x) for x in strs]
        scores = out.scores

    if output_es_score:
        es_scores = compute_ES(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            prompt_input_ids=gen_input_ids,
            prompt_attention_mask=gen_attention_mask,
            targets=ground_truth,
        )
    else:
        es_scores = []

    return input_strings, strs, ground_truth, scores, es_scores

def compute_ES(
    cfg,
    model,
    tokenizer,
    prompt_input_ids,
    prompt_attention_mask,
    targets,
) -> List[float]:
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    B = prompt_input_ids.size(0)
    es_scores = []

    for i in range(B):
        x_ids = prompt_input_ids[i].tolist()
        while x_ids and x_ids[0] == pad_id:
            x_ids.pop(0)

        y_ids = tokenizer(
            targets[i],
            add_special_tokens=False
        )["input_ids"]
        L = len(y_ids)

        if L == 0:
            es_scores.append(0.0)
            continue

        found = False
        for k in range(L + 1):
            ctx_ids = x_ids + y_ids[:k]
            ctx = torch.tensor([ctx_ids], device=prompt_input_ids.device, dtype=torch.long)
            ctx_attn = torch.ones_like(ctx)

            with torch.no_grad():
                out = model._generate(
                    input_ids=ctx,
                    attention_mask=ctx_attn,
                    max_length=cfg.eval_generation_max_length,
                    max_new_tokens=cfg.eval_generation_max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    num_return_sequences=1,
                    use_cache=True,
                    pad_token_id=eos_id,
                    output_scores=False,
                    return_dict_in_generate=False,
                )

            gen_ids = out[0][len(ctx_ids): len(ctx_ids) + (L - k)].tolist()

            if tokenizer.decode(gen_ids, skip_special_tokens=True) == tokenizer.decode(y_ids[k:], skip_special_tokens=True):
                es_scores.append(1.0 - k / L)
                found = True
                break

        if not found:
            es_scores.append(0.0)
    return es_scores

def compute_MRR(logits, labels, hit_k=100):
    B, S, V = logits.shape # logits.shape : [16, 512, 128256], labels.shape : [16, 512]

    # align next-token prediction
    logits = logits[:, :-1, :]      # (B, S-1, V) : [16, 511, 128256]
    labels = labels[:, 1:]          # (B, S-1) : [16, 511]
    mask = labels != -100           # mask.shsape : [16, 511]

    mrr_list = []
    hit_list = []
    for i in range(B):
        pos = mask[i].nonzero(as_tuple=False).squeeze(-1)  # 정답 토큰만 선택, 8개의 토큰
        if pos.numel() == 0:
            raise ValueError(f"pos.numel() == 0")
        rr = []
        hit = []
        for t in pos.tolist():
            gt_id = labels[i, t].item() # i번째 샘플에서 t번째 정답 토큰 id를 꺼내기
            scores = logits[i, t]   # logits에서 i번째 샘플에서 t번째 모든 vocab에 대한 점수를 꺼내기

            gt_score = scores[gt_id] # 정답 토큰이 받은 점수
            rank = int((scores > gt_score).sum().item()) + 1 # 정답 토큰보다 점수가 더 큰 토큰들만 True

            rr.append(1.0 / rank)
            hit.append(1 if rank <= hit_k else 0)

        mrr_list.append(sum(rr) / len(rr))
        hit_list.append(sum(hit) / len(hit))

    return mrr_list, hit_list


#==================load eval dataset=================
def get_dataloader(cfg, data_path, tokenizer, eval_target):
    if eval_target == "forget_edge":
        torch_format_dataset = TofuForgetEdgeDataset( # dataset_format_converstion
            data_path,
            tokenizer=tokenizer,
            configs=cfg,
            max_length=cfg.eval_dataset_max_length,
        )
    elif eval_target == "retained_edge":
        torch_format_dataset = TofuRetainedEdgeDataset( # dataset_format_converstion
            data_path,
            tokenizer=tokenizer,
            configs=cfg,
            max_length=cfg.eval_dataset_max_length,
        )
    elif eval_target == "factual_edge":
        torch_format_dataset = TofuFactualDataset( # dataset_format_converstion
            data_path,
            tokenizer=tokenizer,
            configs=cfg,
            max_length=cfg.eval_dataset_max_length,
        )
    else:
        raise ValueError(f"Invalid eval_target")
    print("torch_format_dataset.data : ", torch_format_dataset.data) # @@@ : dataset의 내용 확인
    eval_dataloader = DataLoader(
        torch_format_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=custom_tofu_collator,
    )
    print("[LODA EVAL DATA] from ", eval_target, "...", len(eval_dataloader.dataset)) 
    return eval_dataloader


def custom_evaluate_tofu(
    cfg, 
    name, 
    data_path, 
    tokenizer, 
    model, 
    eval_target="factual_edge",
    forget_label_idx=1,
    fwd_pre_hooks=[], 
    fwd_hooks=[], 
    output_es_score=False
):
    eval_dataloader = get_dataloader( # @@@ : 평가 데이터 로드
        cfg=cfg, data_path=data_path, tokenizer=tokenizer, eval_target=eval_target
    )
    model._to("cuda")
    model._eval()

    eval_logs = get_all_evals(
        cfg=cfg,
        name=name,
        model=model,
        tokenizer=tokenizer,
        eval_dataloader=eval_dataloader,
        fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks,
        output_es_score=output_es_score,
        layer_idx_list=cfg.layer_modified,
        forget_label_idx=forget_label_idx,
    )
    eval_logs = summarize_eval_logs(eval_logs) # @ 수정
    return eval_logs

#==================summarize eval result=================
def summarize_value(v):
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return "N/A" # 빈 리스트일 경우
        if all(isinstance(x, (int, float, bool)) for x in v):
            return float(sum(v)) / len(v) # 산술평균을 내서 스칼라로 변환
        return v
    return v

def summarize_eval_logs(eval_logs: dict):
    skip_keys = {"generated_text", "topic", "expert_probs", "failure_text", "router_info"} # 평균 요약에서 제외할 key들
    for k, v in list(eval_logs.items()):
        # print("[Debug] Result key is : ", k)
        if k in skip_keys:
            continue
        eval_logs[k] = summarize_value(v)
    return eval_logs