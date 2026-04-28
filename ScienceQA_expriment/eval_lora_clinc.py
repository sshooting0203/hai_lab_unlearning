import os, re, json, argparse, random, torch, sys
from tqdm import tqdm
from typing import Union
from collections import defaultdict
import numpy as np
from src.mapping_hacked_o import debug_forward_effect, debug_lora_layers, debug_peft_basic
from transformers import GenerationConfig, AutoTokenizer, AutoConfig
from transformers import LlamaForCausalLM
from peft import PeftModel
from utils.prompter import build_chat_prompt

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--test_dataset', type=str)
    parser.add_argument('--base_model', type=str)
    parser.add_argument('--lora_weights', type=str)
    parser.add_argument('--max_batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--unlearn_topic', type=str)
    parser.add_argument('--lora_start_layer', type=int, default=0)
    parser.add_argument('--task_type', type=str, default="clinc", choices=["clinc", "mrpc", "rte"])

    # Parsing arguments
    args = parser.parse_args()
    set_seed(args.seed)
    data_a = json.load(open(args.test_dataset))
    base_model = args.base_model
    lora_weights = args.lora_weights
    max_batch_size = args.max_batch_size

    load_8bit = False

    path = "/".join(lora_weights.split("/")[:-1])
    if not os.path.exists(path):
        os.mkdir(path)
    result_file = os.path.join(path, f"seed{args.seed}_{args.unlearn_topic}.json")
    print(result_file)

    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side='left')

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer.eos_token is None; cannot set pad_token safely.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.bfloat16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    # if not load_8bit:
    #     model.half()

    model.eval()
    ################################################
    max_new_tokens = 128
    save_every = 200

    contains_correct = 0
    exact_correct = 0

    results = []

    domain_total = defaultdict(int)
    domain_contains_correct = defaultdict(int)
    domain_exact_correct = defaultdict(int)

    for start_idx in tqdm(range(0, len(data_a), max_batch_size)):
        end_idx = min(start_idx + max_batch_size, len(data_a))
        batch = data_a[start_idx:end_idx]

        intents = [str(example["intent"]) for example in batch]
        # debug('intent is : ', intents)
        domains = [str(example.get("domain", "unknown")) for example in batch]

        prompts = [build_chat_prompt(tokenizer, instruction=example['instruction'], system=None) for example in batch]
        ###########################################
        inputs = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        sequences = generation_output.sequences
        gen_only_ids = sequences[:, input_ids.shape[1]:]
        output = tokenizer.batch_decode(gen_only_ids, skip_special_tokens=True)
        debug("GEN ONLY OUTPUT :", output)

        for r_i, raw_text in enumerate(output):
            d = domains[r_i]
            domain_total[d] += 1

            resp = raw_text.strip()
            intent = resp.splitlines()[0].strip() if resp else "FAILED"
            intent = intent.replace("</s>", "").replace("<|eot_id|>", "").strip()
            if not intent:
                intent = "FAILED"

            pred_text = str(intent).strip().lower()
            gt_text = str(intents[r_i]).strip().lower()

            is_contains_correct = gt_text in pred_text
            is_exact_correct = gt_text == pred_text

            if is_contains_correct:
                contains_correct += 1
                domain_contains_correct[d] += 1

            if is_exact_correct:
                exact_correct += 1
                domain_exact_correct[d] += 1

            debug("intent : ", intent)
            debug("gt : ", intents[r_i])

            if is_exact_correct:
                print("exact correct:", pred_text, gt_text, flush=True)
            elif is_contains_correct:
                print("contains correct:", pred_text, gt_text, flush=True)
            else:
                print("pred/gt:", pred_text, gt_text, flush=True)

            results.append({
                "domain": d,
                "prompt": prompts[r_i],
                "gt": intents[r_i],
                "pred": intent,
                "raw_output": resp,
                "contains_correct": is_contains_correct,
                "exact_correct": is_exact_correct,
            })

        #########################################
        contains_acc = contains_correct / len(results) * 100
        exact_acc = exact_correct / len(results) * 100

        if end_idx % save_every == 0 or end_idx == len(data_a):
            per_topic = {}
            for t in domain_total:
                tot = int(domain_total[t])
                contains_cor = int(domain_contains_correct[t])
                exact_cor = int(domain_exact_correct[t])

                per_topic[t] = {
                    "total": tot,
                    "contains_correct": contains_cor,
                    "exact_correct": exact_cor,
                    "contains_acc": (contains_cor / tot * 100) if tot > 0 else 0.0,
                    "exact_acc": (exact_cor / tot * 100) if tot > 0 else 0.0,
                }

            per_domain_sorted = dict(
                sorted(per_topic.items(), key=lambda kv: kv[1]["total"], reverse=True)
            )

            print(f"{len(results)}/{len(data_a)}, contains_correct: {contains_correct}, contains_acc: {round(contains_acc, 2)}%, exact_correct: {exact_correct}, exact_acc: {round(exact_acc, 2)}%, saving to {result_file}")
            
            data = {}
            data["contains_acc"] = contains_acc
            data["exact_acc"] = exact_acc
            data["contains_correct"] = contains_correct
            data["exact_correct"] = exact_correct
            data["len"] = len(results)
            data["results"] = results
            data["per_domain"] = per_domain_sorted

            with open(result_file, 'w') as f:
                json.dump(data, f, indent=2, separators=(',', ': '))


def debug(*args):
    print(*args, file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()