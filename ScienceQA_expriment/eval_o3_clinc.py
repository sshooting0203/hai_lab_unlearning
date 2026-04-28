import os, re, json, argparse, torch, random, pickle, math, sys 
from tqdm import tqdm
from collections import defaultdict
from src.ood_model_selector import RobertaForSelector, RobertaForSelector_inference
from transformers import RobertaConfig, RobertaTokenizer, BertConfig, BertTokenizer
from transformers import GenerationConfig, AutoTokenizer, AutoConfig
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np
from typing import Union
from sklearn.mixture import GaussianMixture as GMM
from src.peft_model_hacked_o import PeftModel
from src.modeling_llama_hacked_31 import LlamaForCausalLM_ood
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

def weighting_func_gmm(train_in_score, test_in_score):
    # 1. fit two gaussians
    mean1, std1 = norm.fit(train_in_score)
    mean2, std2 = norm.fit(test_in_score)

    # 2. build the gaussian mixture model
    gmm = GMM(n_components=2)
    gmm.means_ = np.array([[mean1], [mean2]])
    # gmm.covariances_ = np.array([[[std1**2]], [[std2**2]]])
    gmm.covariances_ = np.array([[[std2 ** 2]], [[std2 ** 2]]])
    gmm.weights_ = np.array([0.5, 0.5])
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))

    # center: x0
    # x0 = minimize(lambda x: (gmm_cdf(x, gmm) - 0.5)**2, x0=0).x[0]
    x0 = (mean1 + mean2) / 2
    return gmm, x0

# cumulative prob func
def gmm_cdf(x, gmm):
    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    cdf_vals = [w * norm.cdf(x, mean, std) for w, mean, std in zip(weights, means, stds)]
    return np.sum(cdf_vals)

# the cumulative prob for a point
def cumulative_probability(x, gmm):
    return gmm_cdf(x, gmm)

# the cumulative prob for the symmtric point
def symmetric_cumulative_probability(x, x0, gmm):
    symmetric_x = 2 * x0 - x
    return gmm_cdf(symmetric_x, gmm)

all_w_res = []
all_w_res_dic = {}

def obtain_weights(input_x, gmm, x0):
    cp_x = cumulative_probability(input_x, gmm)
    cp_symmetric_x = symmetric_cumulative_probability(input_x, x0, gmm)

    cp_sum = 1 - max(cp_x, cp_symmetric_x) + min(cp_x, cp_symmetric_x)
    scaling_factor = 10
    cp_sum *= scaling_factor
    range_th = 2 # 2 0.5
    w_res = math.exp(cp_sum - range_th) / (1 + math.exp(cp_sum - range_th))

    # if w_res > 0.4:
    #     w_res = 1.2
    # elif w_res <= 0.4 and w_res > 0.3:
    #     w_res = w_res * 1.2 # 2
    # else:
    #     w_res=0
    if w_res > 0.9: # not_test_RD 0.997 0.995 0.98 # test_SD 0.999 0.990 0.9845
        w_res = 1.2
    elif w_res <= 0.4 and w_res > 0.3:
        w_res = w_res # 2
    else:
        w_res=0
    return w_res

def main():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--test_dataset', type=str, help='test_dataset')
    parser.add_argument('--base_model', type=str, help='base_model')
    parser.add_argument('--ood_base_model', type=str, default="roberta-large", help='base_model')
    parser.add_argument('--lora_weights', type=str, help='lora_model')
    parser.add_argument('--ood_weights', type=str, help='ood_model')
    parser.add_argument('--ood_type', type=str, help='ood type')
    parser.add_argument('--ood_setting', type=str, help='ood setting')
    parser.add_argument('--ood_setting_name', type=str, help='ood setting name')
    parser.add_argument('--seed', type=int, default=1, help='base_model')
    parser.add_argument('--unlearn_topic', type=str, help='file_name') # @ 수정 

    # Parsing arguments
    args = parser.parse_args()
    set_seed(args.seed)
    data_a = json.load(open(args.test_dataset))
    base_model = args.base_model
    lora_weight = args.lora_weights
    max_batch_size = 1
    
    # types = args.ood_type.split("_")
    # ood_types = []
    # for i in types:
    #     if len(i) > 0:
    #         ood_types.append(i)
    ood_types = [args.ood_type]
    ood_setting = args.ood_setting
    ood_setting_names = args.ood_setting_name
    load_8bit = False
    ood_weights = []
    ood_type = "ocsvm"

    for i in ood_types:  # "biology", "physics", "chemistry"
        o_p = args.ood_weights + f"{ood_setting_names}_{i}_ood_{ood_setting_names}"
        ood_weights.append(o_p)

    path = "/".join(lora_weight.split("/")[:-1])
    if not os.path.exists(path):
        os.mkdir(path)
    result_file = os.path.join(path, f"test_seed{args.seed}_{args.unlearn_topic}.json")
    print(result_file)

    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side='left')

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer.eos_token is None; cannot set pad_token safely.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    lora_target_modules = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj"
    ]
    config = AutoConfig.from_pretrained(base_model)
    config.lora_target_modules = lora_target_modules

    orthogonal_loss = False
    olora_weights = {}
    config.orthogonal_loss = orthogonal_loss
    config.orthogonal_loss_weight = 0.1

    model = LlamaForCausalLM_ood.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weight,
        torch_dtype=torch.bfloat16,
    )

    model.init_olora(orthogonal_loss=orthogonal_loss, olora_weights=olora_weights)
    model.init_active_adapters_d(active_adapters_d=['default'])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # if not load_8bit:
    #     model.half()  # seems to fix bugs for some users.

    model.eval()
    ##############################################
    ood_base_model = args.ood_base_model
    ood_tokenizer = RobertaTokenizer.from_pretrained(ood_base_model)
    ood_models = []
    ood_clrs = []
    ood_thresholds = []
    ood_x0 = []
    ood_mean_lists = []
    ood_precision_lists = []
    ood_fea_lists = []
    ood_gmm_w_cls = []

    for i in ood_weights:
        roberta_path = i + f"_roberta_{ood_type}"
        ocsvm_path = i + f"_{ood_type}.pkl"
        threshold_path = i + f"_threshold_{ood_type}.json"
        mean_list_path = i + f"_mean_list_{ood_type}.pt"
        precision_list_path = i + f"_precision_list_{ood_type}.pt"
        fea_list_path = i + f"_fea_list_{ood_type}.pt"
        gmm_w_path = i + f"_gmm_w_{ood_type}.pkl"

        ood_models.append(RobertaForSelector_inference(ood_base_model, lora_path=roberta_path, projection_dim=100).to(device))
        with open(ocsvm_path, "rb") as input_file:
            c_lr = pickle.load(input_file)
        ood_clrs.append(c_lr)
        with open(gmm_w_path , "rb") as input_file:
            gmm_w = pickle.load(input_file)
        ood_gmm_w_cls.append(gmm_w)
        with open(threshold_path) as f:
            threshold = json.load(f)
        ood_thresholds.append(threshold[1])
        ood_x0.append(threshold[0])

        ood_mean_lists.append(torch.load(mean_list_path, map_location=torch.device(device)))
        ood_precision_lists.append(torch.load(precision_list_path, map_location=torch.device(device)))
        ood_fea_lists.append(torch.load(fea_list_path, map_location=torch.device(device)))
    ##############################################
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

        llm_prompts = []
        ood_texts = []

        for example in batch:
            instruction = str(example.get("instruction", ""))
            input_text = str(example.get("input", ""))
            # OOD는 학습 때 text_input plain text를 봤으므로 raw text 유지
            if input_text:
                raw_text = f"{instruction}\n\n{input_text}"
            else:
                raw_text = instruction
            ood_texts.append(raw_text)
            # LLM은 chat template 사용
            llm_prompts.append(
                build_chat_prompt(
                    tokenizer,
                    instruction=raw_text,
                    system=None,
                )
            )

        ##############################################
        ood_input = ood_tokenizer(
            ood_texts, 
            padding='max_length', 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ) # 256
        max_ood = 0

        for i in range((len(ood_weights))):
            mah_score = ood_models[i].get_unsup_Mah_score_s(ood_input, 
                    ood_mean_lists[i], ood_precision_lists[i], ood_fea_lists[i])[:, 1:]
            test_score = ood_clrs[i].score_samples(mah_score)
            w_ood = obtain_weights(test_score, ood_gmm_w_cls[i], ood_x0[i])
            if w_ood > max_ood:
                max_ood = w_ood

        all_w_res.append(max_ood)
        dic_key = str(max_ood)[:5]
        if dic_key in all_w_res_dic:
            all_w_res_dic[dic_key] += 1
        else:
            all_w_res_dic[dic_key] = 1

        print("ood_weight: ", [1, max_ood])
        model.init_oodweight(ood_weight=[1, max_ood])
        ##############################################
        inputs = tokenizer(
            llm_prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device) # @ 수정 : llama3.1-instruct

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
                "prompt": llm_prompts[r_i],
                "gt": intents[r_i],
                "pred": intent,
                "raw_output": resp,
                "contains_correct": is_contains_correct,
                "exact_correct": is_exact_correct,
                "ood_text": ood_texts[r_i],
                "ood_weight": float(max_ood),
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

    print(np.mean(all_w_res), min(all_w_res), max(all_w_res))

def debug(*args):
    print(*args, file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()