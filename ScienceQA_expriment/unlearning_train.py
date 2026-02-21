import os, re, json, argparse, random, sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)

# =========================
# 0) 네가 이미 갖고 있는 것들을 import 한다고 가정
# =========================
# 예: src/unlearn_methods/__init__.py 에서 get_unlearn_method를 export 해두었다고 치자.
# from src.unlearn_methods import get_unlearn_method
#
# 또는 네가 보여준 코드가 있는 파일을 그대로 import:
# from src.unlearn_methods.router import get_unlearn_method

# ---- 여기서는 "외부에서 가져온다"는 전제만 둔다.
get_unlearn_method = None  # 이 줄은 반드시 너 코드로 교체/삭제


# =========================
# 1) Reproducibility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# =========================
# 2) Prompting (ScienceQA eval에 사용)
# =========================
class Prompter(object):
    __slots__ = ("template", "_verbose")
    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n"
            ),
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
        parts = output.split(self.template["response_split"])
        return parts[1].strip() if len(parts) > 1 else output.strip()


# =========================
# 3) (중요) forget/retain을 함께 내보내는 dataset + collator
#    네 compute_loss는 inputs["forget"], inputs["retain"]을 기대함.
# =========================
class PairForgetRetainDataset(Dataset):
    """
    __getitem__ 이 {"forget": forget_item, "retain": retain_item} 반환
    forget_item/retain_item 의 내부 구조는 네 기존 코드(CL.compute_loss)가 기대하는 포맷에 맞춰야 함.
    """
    def __init__(self, forget_items: List[Any], retain_items: List[Any]):
        assert len(forget_items) == len(retain_items), "forget/retain 길이가 같아야 함(간단한 1:1 pairing 가정)"
        self.forget_items = forget_items
        self.retain_items = retain_items

    def __len__(self):
        return len(self.forget_items)

    def __getitem__(self, idx):
        return {"forget": self.forget_items[idx], "retain": self.retain_items[idx]}


@dataclass
class IdentityPairCollator:
    """
    네 데이터가 이미 tensor pack 형태(예: forget_data = (input_ids, attn_mask, ..., labels, positions))로
    만들어져 있다면 collator는 "그냥 배치로 묶기"만 하면 된다.

    BUT: torch default stack이 안 되는 구조면 여기에서 직접 stack/pad 해야 함.
    지금은 '이미 같은 shape tensor로 들어온다'는 전제의 최소 구현.
    """
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # features: [{"forget": X, "retain": Y}, ...]
        forget_batch = [f["forget"] for f in features]
        retain_batch = [f["retain"] for f in features]

        # 만약 forget/retain item이 tuple(list) of tensors 라면, 항목별로 stack
        # 예: forget_item = (input_ids, attn_mask, ..., labels, positions)
        def stack_like(items):
            if isinstance(items[0], (list, tuple)):
                out = []
                for k in range(len(items[0])):
                    elems = [it[k] for it in items]
                    if torch.is_tensor(elems[0]):
                        out.append(torch.stack(elems, dim=0))
                    else:
                        # positions 같은 게 python list/int면 tensor로 변환
                        out.append(torch.tensor(elems))
                return tuple(out)
            else:
                # dict/tensor/기타
                return items

        return {"forget": stack_like(forget_batch), "retain": stack_like(retain_batch)}


# =========================
# 4) 데이터 로더 연결부 (너 환경에 맞게 여기만 채우면 됨)
# =========================
def build_forget_retain_dataset(args, tokenizer):
    """
    TODO: 여기만 너 코드로 연결.
    - forget train set
    - retain train set
    둘을 같은 길이로 pairing 해서 PairForgetRetainDataset으로 반환.

    네가 이미 쓰는 QuestionsDataset / custom loader가 있으면,
    그 loader의 __getitem__이 반환하는 포맷을 CL.compute_loss가 기대하는 포맷으로 맞춰서
    forget_items, retain_items 리스트로 만들어 넣으면 된다.
    """
    raise NotImplementedError(
        "build_forget_retain_dataset()를 너 프로젝트 데이터로더로 연결해야 함.\n"
        "핵심: 각 item은 CL.compute_loss가 기대하는 구조(튜플 인덱싱 [0],[1],[3],[4] 가능)여야 함."
    )


# =========================
# 5) LoRA 모델 준비
# =========================
def build_lora_model(base_model: str, lora_r: int, lora_alpha: int, lora_dropout: float,
                     target_modules: List[str], bf16: bool, device_map: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left", use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    return model, tokenizer


# =========================
# 6) ScienceQA 평가 (너가 준 eval 코드 거의 그대로 함수화)
# =========================
@torch.no_grad()
def eval_scienceqa(
    base_model: str,
    lora_weights: str,
    test_dataset_path: str,
    result_file: str,
    max_batch_size: int,
    seed: int,
    unlearn_topic: str,
    max_new_tokens: int = 128,
):
    set_seed(seed)
    data_a = json.load(open(test_dataset_path, "r"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompter = Prompter(template_name="alpaca")

    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left", use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # base + lora load
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device == "cuda" else {"": device},
    )
    model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=model.dtype, device_map="auto" if device == "cuda" else {"": device})
    model.eval()

    # torch.compile optional
    if torch.__version__ >= "2" and sys.platform != "win32" and device == "cuda":
        model = torch.compile(model)

    pattern = re.compile(r"The answer is ([A-Z]).")

    correct = 0
    results, outputs, gt = [], [], []
    topic_total = defaultdict(int)
    topic_correct = defaultdict(int)

    save_every = 200

    for start_idx in range(0, len(data_a), max_batch_size):
        end_idx = min(start_idx + max_batch_size, len(data_a))
        batch = data_a[start_idx:end_idx]

        answers = [str(ex["answer"]) for ex in batch]
        topics  = [str(ex.get("topic", "unknown")) for ex in batch]
        prompts = [prompter.generate_prompt(ex["instruction"], ex.get("input", "")) for ex in batch]

        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        gen = model.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
        s = gen.sequences
        decoded = tokenizer.batch_decode(s, skip_special_tokens=False)
        decoded = [prompter.get_response(x) for x in decoded]

        found = [pattern.findall(x) for x in decoded]

        for i in range(len(batch)):
            t = topics[i]
            topic_total[t] += 1

            if len(found[i]) == 1:
                pred = found[i][0]
            else:
                pred = "FAILED"

            results.append(found[i])
            outputs.append(decoded[i])
            gt.append(answers[i])

            if str(pred) == str(answers[i]):
                correct += 1
                topic_correct[t] += 1

        acc = correct / len(results) * 100

        if end_idx % save_every == 0 or end_idx == len(data_a):
            per_topic = {}
            for t in topic_total:
                tot = int(topic_total[t])
                cor = int(topic_correct[t])
                per_topic[t] = {"total": tot, "correct": cor, "acc": (cor / tot * 100) if tot > 0 else 0.0}
            per_topic_sorted = dict(sorted(per_topic.items(), key=lambda kv: kv[1]["total"], reverse=True))

            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            with open(result_file, "w") as f:
                json.dump(
                    {
                        "unlearn_topic": unlearn_topic,
                        "acc": acc,
                        "correct": correct,
                        "len": len(results),
                        "results": results,
                        "outputs": outputs,
                        "per_topic": per_topic_sorted,
                    },
                    f,
                    indent=2,
                    separators=(",", ": "),
                )

    return acc


# =========================
# 7) Main: unlearn train -> save -> eval
# =========================
def main():
    parser = argparse.ArgumentParser()

    # ---- base / method
    parser.add_argument("--base_model", type=str, required=True)  # ex) meta-llama/Llama-3.1-8B or your HF repo
    parser.add_argument("--method", type=str, required=True)      # FT, GA, GA+FT, CL+FT, DPO, KL_FT, ...

    # ---- LoRA
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # ---- train args
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device_map", type=str, default="auto")

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)

    # ---- eval (ScienceQA)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--scienceqa_test_dataset", type=str, default="")
    parser.add_argument("--unlearn_topic", type=str, default="unknown")
    parser.add_argument("--eval_batch_size", type=int, default=8)

    args = parser.parse_args()
    set_seed(args.seed)

    if get_unlearn_method is None:
        raise RuntimeError(
            "get_unlearn_method가 None임. 위 import 구문을 너 프로젝트에 맞게 연결해줘."
        )

    # 1) build model/tokenizer + LoRA
    target_modules = [x.strip() for x in args.lora_targets.split(",") if x.strip()]
    model, tokenizer = build_lora_model(
        base_model=args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bf16=args.bf16,
        device_map=args.device_map,
    )

    # 2) build dataset (너 코드로 채우는 부분)
    train_dataset = build_forget_retain_dataset(args, tokenizer)
    data_collator = IdentityPairCollator()

    # 3) training args
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=(not args.bf16) and torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,  # 중요: {"forget","retain"} 그대로 넘기기
    )

    # 4) pick unlearner (compute_loss override trainer)
    #    여기서 *args, **kwargs는 네 BaseTrainer가 요구하는 것들에 맞춰서 넣어야 함.
    #    최소로는 (model, args, train_dataset, data_collator, tokenizer 등) 필요할 가능성이 큼.
    #
    #    예시: 네 BaseTrainer가 transformers.Trainer를 감싼 구조라면 아래처럼 들어가면 됨.
    unlearner = get_unlearn_method(
        args.method,
        model=model,
        args=targs,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 5) train
    unlearner.train()

    # 6) save LoRA adapter
    os.makedirs(args.output_dir, exist_ok=True)
    unlearner.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 7) eval (ScienceQA)
    if args.do_eval:
        assert args.scienceqa_test_dataset, "--do_eval이면 --scienceqa_test_dataset 필요"
        result_file = os.path.join(args.output_dir, f"seed{args.seed}_{args.unlearn_topic}.json")
        acc = eval_scienceqa(
            base_model=args.base_model,
            lora_weights=args.output_dir,
            test_dataset_path=args.scienceqa_test_dataset,
            result_file=result_file,
            max_batch_size=args.eval_batch_size,
            seed=args.seed,
            unlearn_topic=args.unlearn_topic,
        )
        print(f"[DONE] ScienceQA acc={acc:.2f} saved to {result_file}")


if __name__ == "__main__":
    main()
