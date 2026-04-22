# Copyright (c) Meta Platforms, Inc. and affiliates.
from tqdm import tqdm
import torch, math, os, json, evaluate, re, json
from rouge_score import rouge_scorer
from typing import List,  Dict, Optional, Any, Iterable
from torch.utils.data import DataLoader
from src_lunar.data_loader import (
    get_batch_loss,
    custom_qa_collator,
    QAForgetEdgeDataset,
    QARetainedEdgeDataset,
    QAFactualDataset,
)
# from src_lunar.utils.clincQA import eval_choice_hit # @@@ 평가
from src_lunar.utils.hook_utils import add_hooks
from src_lunar.utils.model_utils import extract_instruction_from_model 
from src_lunar.model_utils.llama31model import build_prompt_from_instruction, LLAMA3_CHAT_TEMPLATE
from src_lunar.model_utils.qwen_model_old import QWEN_CHAT_TEMPLATE

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

    
def get_all_evals(
    cfg, 
    name,
    model, 
    tokenizer, 
    eval_dataloader, 
    fwd_pre_hooks=[], fwd_hooks=[], 
    output_es_score=False,
):
    from src_lunar.estimated_net_moe_utils import LUNAR_LoRA_MoE_net, LUNAR_NN_MoE_net

    # prompt = build_prompt_from_instruction(tokenizer, "What is 2+2?", system=None)
    # print("build_prompt_from_inst : ",repr(prompt))
    # old_prompt = LLAMA3_CHAT_TEMPLATE.format(instruction="What is 2+2?")
    # print("old_prompt : ",repr(old_prompt))

    eval_logs = {} # 최종 평가 지표 
    gen_outputs, ground_truths, input_strings = [], [], []
    num_token_gt_list, mrr_list, hit_rate_list, perplexity_list, es_score_list = [], [], [], [], []
    topics_list, expert_probs_list = [], []
    eval_hit = get_eval_choice_hit(cfg) # @ 수정 : dataset에 따라 평가 함수 다르게 호출

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

    with tqdm(total=len(eval_dataloader), desc=f"Eval {name}") as pbar:
        for batch in eval_dataloader:
            input_ids, labels, attention_mask, topics, gt = batch
            topics_list.extend(topics)

            batch_gpu = {
                "input_ids": input_ids.to("cuda"),
                "labels": labels.to("cuda"),
                "attention_mask": attention_mask.to("cuda"),
            }

            with torch.no_grad():
                current_batch_logits = []

                model._cur_attn_mask_2d = batch_gpu["attention_mask"]
                outputs = model.model(
                    **batch_gpu,
                    output_hidden_states=False,
                    return_dict=True,
                )

                # ================== Measure expert probs ==================
                # if current_batch_logits:
                #     all_logits = torch.stack(current_batch_logits)   # (L, B, S, E)
                #     all_probs = torch.softmax(all_logits, dim=-1)    # (L, B, S, E)

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
                #             # "token_top1_experts": token_top1[i].tolist(),
                #         })

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
                            
                        expert_probs_list.append({
                            "gen_expert_sequence": sample_expert_seqs
                        })
                else:
                    for _ in range(B_curr):
                        expert_probs_list.append({"gen_expert_sequence": []})

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
                ground_truths.extend(gt)
                perplexity_list.append(perplexity)

                if output_es_score:
                    es_score_list.extend(es_score)
            ###############################################################
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
    hit_logs = eval_hit(gen_outputs, ground_truths)
    eval_logs.update(hit_logs)

    if output_es_score:
        eval_logs["es_score"] = eval_logs.get("es_score", []) + es_score_list
    else:
        eval_logs["es_score"] = eval_logs.get("es_score", [])

    if topics_list and expert_probs_list:
        eval_logs["generated_text"] = list(zip(input_strings, gen_outputs, ground_truths, topics_list, expert_probs_list)) # 입력, 출력, 정답, 도메인, expert확률 
    else:
        eval_logs["generated_text"] = list(zip(input_strings, gen_outputs, ground_truths)) # 입력, 출력, 정답 

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
        torch_format_dataset = QAForgetEdgeDataset( # dataset_format_converstion
            data_path,
            tokenizer=tokenizer,
            configs=cfg,
            max_length=cfg.eval_dataset_max_length,
        )
    elif eval_target == "retained_edge":
        torch_format_dataset = QARetainedEdgeDataset( # dataset_format_converstion
            data_path,
            tokenizer=tokenizer,
            configs=cfg,
            max_length=cfg.eval_dataset_max_length,
        )
    elif eval_target == "factual_edge":
        torch_format_dataset = QAFactualDataset( # dataset_format_converstion
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
    skip_keys = {"generated_text", "topic", "expert_probs", "failure_text"} # 평균 요약에서 제외할 key들
    for k, v in list(eval_logs.items()):
        # print("[Debug] Result key is : ", k)
        if k in skip_keys:
            continue
        eval_logs[k] = summarize_value(v)
    return eval_logs

