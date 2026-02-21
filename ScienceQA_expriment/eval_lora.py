import os, re, json, argparse, random, torch, sys 
from tqdm import tqdm
from typing import Union
from collections import defaultdict
import numpy as np
from src.mapping_hacked_o import  debug_forward_effect, debug_lora_layers, debug_peft_basic
from transformers import GenerationConfig, LlamaTokenizer, AutoConfig
from transformers import LlamaForCausalLM
from peft import PeftModel 

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def set_seed(seed: int):
    """Fix PRNG seed for reproducable experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

class Prompter(object):
    __slots__ = ("template", "_verbose")
    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
            "response_split": "### Response:"
        }
        if self._verbose:
            print(f"Using prompt template {template_name}: {self.template['description']}")

    def generate_prompt(self, instruction: str, input: Union[None, str] = None, label: Union[None, str] = None) -> str:
        if input:
            res = self.template["prompt_input"].format(instruction=instruction, input=input)
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

def main():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--test_dataset', type=str, default="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data/sceinceqa_test_sample.json", help='test_dataset')
    parser.add_argument('--base_model', type=str, default="gcyzsl/O3_LLAMA2_ScienceQA", help='base_model')
    parser.add_argument('--lora_weights', type=str, default="/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_10/lora_force_random_biology_force", help='lora_model')
    parser.add_argument('--max_batch_size', type=int, default=8, help='base_model')
    parser.add_argument('--seed', type=int, default=8, help='base_model')
    parser.add_argument('--unlearn_topic', type=str, help='file_name') # @ 수정 
    parser.add_argument('--lora_start_layer', type=int, default=0)

    # Parsing arguments
    args = parser.parse_args()
    set_seed(args.seed)
    data_a = json.load(open(args.test_dataset))
    base_model = args.base_model
    lora_weights = args.lora_weights
    max_batch_size = args.max_batch_size

    print(args.test_dataset)
    print(args.base_model)
    print(args.lora_weights)
    load_8bit = False

    path = "/".join(lora_weights.split("/")[:-1])
    if not os.path.exists(path):
        os.mkdir(path)
    result_file = os.path.join(path, f"seed{args.seed}_{args.unlearn_topic}.json")
    # result_file = path + "/seed{}_{}_{}".format(str(args.seed), lora_weights.split("/")[-1], args.test_dataset.split("/")[-1])
    print(result_file)

    tokenizer = LlamaTokenizer.from_pretrained(base_model, padding_side='left')
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
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

    # print(model.config.pad_token_id, tokenizer.pad_token_id)
    # print(model.config.bos_token_id, tokenizer.bos_token_id)
    # print(model.config.eos_token_id, tokenizer.eos_token_id)
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    ##############################################
    prompter = Prompter(template_name="alpaca")
    max_new_tokens = 128
    save_every = 200
    correct = 0
    results, outputs, gt = [], [], []
    topic_total = defaultdict(int) # @ 수정
    topic_correct = defaultdict(int) # @ 수정
    topics_list = [] # @ 수정

    for start_idx in tqdm(range(0, len(data_a), max_batch_size)):
        end_idx = min(start_idx + max_batch_size, len(data_a))
        batch = data_a[start_idx:end_idx]

        answers = [str(example["answer"]) for example in batch]
        # print('answer is : ', answers)
        topics  = [str(ex.get("topic", "unknown")) for ex in batch] # @ 수정 

        prompts = [prompter.generate_prompt(example['instruction'], example['input']) for example in batch]
        # print('prompt is : ', prompts)

        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True, # 생성 과정에서 각 스텝마다의 점수(logits)를 저장해서 같이 반환
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        output = tokenizer.batch_decode(s) # 토큰 → 텍스트로 디코딩
        output = [prompter.get_response(otp) for otp in output] # "### Response:" 같은 구분자 이후만 잘라내기

        # extract the answer
        # print('output is : ', output)
        # pattern = re.compile(r'The anwser to the question is (\d+):*')
        pattern = re.compile(r'The answer is ([A-Z]).')
        res = [pattern.findall(otp) for otp in output]
        # print('res is', res)
        pred = []
        
        for r_i in range(len(res)):
            t = topics[r_i] # @ 수정
            topic_total[t] += 1 # @ 수정

            if len(res[r_i]) == 1:
                answer = res[r_i][0]  # 'A', 'B', ...
            else:
                answer = "FAILED"
                
            pred.append(answer)
            results.append(res[r_i])
            outputs.append(output[r_i])
            gt.append(answers[r_i])
            topics_list.append(t) # @ 수정

            if str(answer) == str(answers[r_i]):
                correct += 1
                topic_correct[t] += 1 # @ 수정
                print('correct:', str(answer), str(answers[r_i]))
            else:
                print('gt-ans:', str(answer), str(answers[r_i]))

        acc = correct / len(results) * 100
        if end_idx % save_every == 0 or end_idx == len(data_a):
            per_topic = {}
            for t in topic_total:
                tot = int(topic_total[t])
                cor = int(topic_correct[t])
                per_topic[t] = {
                    "total": tot,
                    "correct": cor,
                    "acc": (cor / tot * 100) if tot > 0 else 0.0
                } # @ 수정 
            per_topic_sorted = dict(sorted(per_topic.items(), key=lambda kv: kv[1]["total"], reverse=True))

            print(f"{len(results)}/{len(data_a)}, correct: {correct}, acc: {round(acc, 2)}%, saving to {result_file}")
            data = {}
            data['acc'] = acc
            data['correct'] = correct
            data['len'] = len(results)
            data['results'] = results
            data['outputs'] = outputs
            data['per_topic'] = per_topic_sorted # @ 수정
            with open(result_file, 'w') as f:
                json.dump(data, f, indent=2, separators=(',', ': '))

if __name__ == "__main__":
    main()