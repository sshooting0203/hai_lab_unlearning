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

def get_eval_choice_hit(cfg):
    task = str(cfg.data_name).lower()
    if task == "scienceqa":
        from src_lunar.utils.scienceQA import eval_choice_hit
    elif task == "clincqa":
        from src_lunar.utils.clincQA import eval_choice_hit
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
    from src_lunar.estimated_net_moe_utils import LUNAR_LoRA_MoE_net

    eval_logs = {} # 최종 평가 지표 
    gen_outputs, ground_truths, input_strings = [], [], []
    num_token_gt_list, mrr_list, hit_rate_list, perplexity_list, es_score_list = [], [], [], [], []
    eval_hit = get_eval_choice_hit(cfg) # @ 수정 : dataset에 따라 평가 함수 다르게 호출
    topics_list, expert_probs_list = [], []

    #=====================MoE Prob Hook Setup==============
    moe_gates = []
    for m in model.model.modules():
        if isinstance(m, LUNAR_LoRA_MoE_net):
            moe_gates.append(m.gate)
    
    current_batch_logits = []
    def moe_hook(mod, inp, out):
        current_batch_logits.append(out.detach())
    handles = [g.register_forward_hook(moe_hook) for g in moe_gates]
    #=====================MoE Prob Hook Setup==============


    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask, topics, gt = batch
        topics_list.extend(topics)
        batch_gpu = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        for k, v in batch_gpu.items():
            batch_gpu[k] = v.to("cuda")

        with torch.no_grad():
            current_batch_logits = [] # clear for this forward pass
            model._cur_attn_mask_2d = batch_gpu["attention_mask"]
            outputs = model.model( # @ TODO : 현재 코드는 평가 시에 forward 1회 무조건 수행 -> 실제 연산량 개선하려면 detect layer에 등록된 hook까지만 forward하고 stop하는 로직 따로 필요 
                **batch_gpu,
                output_hidden_states=False, # outputs.output_hidden_states 리턴
                return_dict=True, # output.logits : 전 구간 logits, output.loss : -100 maksing 적용되어 있어 labels 기준 loss
            ) 
            # logit : torch.Size([16, 512, 32000]), hidden state : torch.Size([33, 16, 512, 4096])

            #========== Measure expert props for this batch==========
            if current_batch_logits:
                all_logits = torch.stack(current_batch_logits) # (L, B, S, E)
                all_probs = torch.softmax(all_logits, dim=-1) # (L, B, S, E)
                mean_layer_probs = all_probs.mean(dim=0) # (B, S, E)
                
                # attention_mask: (B, S)
                mask = batch_gpu["attention_mask"].unsqueeze(-1).float() # (B, S, 1)
                sum_probs = (mean_layer_probs * mask).sum(dim=1)
                counts = mask.sum(dim=1)
                avg_probs = (sum_probs / (counts + 1e-9)).cpu().tolist()
                expert_probs_list.extend(avg_probs)
            
            current_batch_logits = [] # clear again to avoid RunGeneration results contaminating
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
            # print("gen_output : ", gen_output) # (B) Kenneth's parents have pale skin. They passed down this trait to Kenneth
            # print("gt : ",gt) # Answer: The answer is B
            # print("scores : ",len(scores))

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
        probs = [p / v.item() for p, v in zip(probs, num_token_gt)]

        eval_logs["gt_loss_per_token"] = (eval_logs.get("gt_loss_per_token", []) + (gt_loss / num_token_gt).float().cpu().numpy().tolist())
        eval_logs["gt_loss"] = eval_logs.get("gt_loss", []) + gt_loss.tolist()
        eval_logs["probs"] = eval_logs.get("probs", []) + probs


    for h in handles:
        h.remove()
    # clear_all_gates(model.model)  # @ 수정 : gate clear : model.model : LlamaForCausalLM
    
    eval_logs["num_token_gt"] = (eval_logs.get("num_token_gt", []) + num_token_gt_list)
    eval_logs["mrr"] = eval_logs.get("mrr", []) + mrr_list
    eval_logs["hit_rate"] = eval_logs.get("hit_rate", []) + hit_rate_list
    eval_logs["perplexity"] = eval_logs.get("perplexity", []) + perplexity_list
    
    if topics_list:
        eval_logs["topic"] = topics_list
    if expert_probs_list:
        eval_logs["expert_probs"] = expert_probs_list

    # eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths))
    # print("gen_outputs : ", gen_outputs, ",ground_truths : ", ground_truths)
    eval_logs.update(eval_hit(gen_outputs, ground_truths)) # @ 수정 : hit rate 계산 추가 

    if output_es_score:
        eval_logs["es_score"] = eval_logs.get("es_score", []) + es_score_list
    else:
         eval_logs["es_score"] = eval_logs.get("es_score", [])
    
    if topics_list and expert_probs_list:
        eval_logs["generated_text"] = list(zip(input_strings, gen_outputs, ground_truths, topics_list, expert_probs_list)) # 입력, 출력, 정답, 도메인, expert확률 
    else:
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

    ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]
    input_strings = [s + split_symbol for s in input_strings]

    if cfg.model_family == "llama3-8b-instruct" or cfg.model_family == "llama3.1-8b-instruct": # 여기에서 eot_id 제거
        ground_truth = [ re.sub(r"(<\|eot_id\|>)+$", "", re.sub(r"\n\n", "", text)) for text in ground_truth ]
    elif cfg.model_family == "Qwen2-7B-Instruct" or cfg.model_family == "Qwen2.5-7B-Instruct":
        ground_truth = [ re.sub(r"(<\|im_end\|>)+$", "", re.sub(r"\n<\|im_start\|>assistant", "", text)) for text in ground_truth ]
    elif cfg.model_family == "gemma-7b-it":
        ground_truth = [ re.sub(r"(<eos>)+$", "", re.sub(r"\n<start_of_turn>model", "", text)) for text in ground_truth ]
    
    print("ground truth : ", ground_truth) # Answer: The answer is B.
    print("input string : ", input_strings) # <|begin_of_text|><|start_header_id|>user<|end_header_id|>\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nQuestion: A tree is not the habitat of a\nOptions: (A) squirrel (B) woodpecker (C) monkey (D) lion\n\n\n### Response:\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n
    print("topics : ", topics)
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = "left" 
    left_pad_tokenizer.padding_size = "longest"
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id

    inputs = left_pad_tokenizer.batch_encode_plus( # 다시 tokenize
        input_strings, add_special_tokens=True, return_tensors="pt", padding=True
    ).to("cuda")

    with add_hooks(
        module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks
    ):
        out = model._generate(  # Llama31Model
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
        # output = model.model(**inputs, labels=inputs["input_ids"])
        # loss = output.loss
        # perplexity = math.exp(loss.item())
        strs = left_pad_tokenizer.batch_decode(
            out.sequences[:, inputs.input_ids.shape[-1] :], skip_special_tokens=True
        )
        print("decoded output : ",strs)
        scores = (
            out.scores
        )

    # compute the ES score
    if output_es_score:
        es_scores = compute_ES(cfg, model, tokenizer, inputs, ground_truth) 
        # tokenized된 inputs로 입력됨, ground_truth는 The answer is B
    else:
        es_scores = []
    return input_strings, strs, ground_truth, scores, es_scores

def compute_ES( # @ 수정 : ES score(Batched Extraction-Strength) 계산(es_scores를 리턴 -> list[float] length B, each in [0, 1])
    cfg,
    model,
    tokenizer,
    inputs,
    targets,
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


# def compute_MRR(scores, gt, tokenizer): #  MRR(Mean reciprocal rank), scores는 model.generate에서 옴 
#     print("scores lens: ", len(scores))
#     MRR_res = []
#     hit_rate = []
    
#     score_size = scores[0].shape[0] # Convert scores as tuple to torch tensors
#     vocab_size = scores[0].shape[1]
#     print("score_size : ", score_size, ", vocab_size : ", vocab_size)

#     T = len(scores) # @ 수정 : 512을 고정하지 않고 생성된 scores의 길이로 지정
#     logits = torch.zeros(score_size, T, vocab_size, device="cuda") # Initialize an empty tensor of the desired shape, filled with zeros
#     print("T logits : ",logits.shape)

#     # Iterate over the tuple of tensors and assign each to the correct position in the combined tensor
#     for i, score_tensor in enumerate(scores):
#         print(f"Tensor {i}: shape {score_tensor.shape}, device {score_tensor.device}")
#         logits[:, i, :] = score_tensor

#     probabilities = torch.nn.functional.softmax( # logits : [16, 54, 128256], probabilities : [16, 54, 128256]
#         logits, dim=-1
#     )  # torch.Size([16, 54, 128256]) : 배치는 16개인데, 생성된 토큰의 개수는 54개, 해당 생성된 토큰의 step에서 vocab이 등장할 확률이 들어있음
    
#     for i in range(len(gt)):
#         probs_per_gt = probabilities[i]
#         reciprocal_ranks = []
#         hit_check = []
#         gt_indices = tokenizer.encode(gt[i], add_special_tokens=False) # Tokenize the ground truth

#         L = min(len(gt_indices), probs_per_gt.size(0))
#         if L == 0:
#             MRR_res.append(0.0)
#             hit_rate.append(0.0)
#             continue

#         for j in range(L):
#             gt_index = gt_indices[j]
#             probs = probs_per_gt[j]                
#             sorted_indices = probs.argsort(descending=True)
#             positions = (sorted_indices == gt_index).nonzero(as_tuple=False) # 해당 step에서 생성된 토큰이 gt와 일치하는지를 세기
#             if positions.numel() == 0: # @ 수정 : 0일때
#                 # rank = vocab_size + 1
#                 raise ValueError(f"positions.numel() == 0")
#             else:
#                 rank = positions[0].item() + 1
#                 reciprocal_rank = 1.0 / rank
#                 reciprocal_ranks.append(reciprocal_rank)
#                 if rank <= 100: # 100위 안에 들어옴
#                     hit_check.append(1)
#                 else:
#                     hit_check.append(0)
#         MRR_res.append(sum(reciprocal_ranks) / len(reciprocal_ranks))
#         hit_rate.append(sum(hit_check) / len(hit_check))
#     return MRR_res, hit_rate


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
            # mrr_list.append(0.0)
            # hit_list.append(0.0)
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
    # torch_format_dataset.data = torch_format_dataset.data.select(
    #     range(min(200, len(torch_format_dataset.data))) # @@@ : sample 200개만 뽑아서 평가
    # )
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
    skip_keys = {"generated_text", "topic", "expert_probs"} # 평균 요약에서 제외할 key들
    for k, v in list(eval_logs.items()):
        # print("[Debug] Result key is : ", k)
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
