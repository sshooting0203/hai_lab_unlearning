import json
import os.path as osp
from typing import Union

class Prompter(object):
    def __init__(
            self, 
            tokenizer,
            template_name: str = "", 
            train_on_inputs = True, 
            add_eos_token = True, 
            cutoff_len = 256,
            verbose: bool = False
    ):
        self.tokenizer = tokenizer
        self.train_on_inputs = train_on_inputs
        self.add_eos_token = add_eos_token
        self.cutoff_len = cutoff_len
    
        self._verbose = verbose
        if not template_name:
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(f"Using prompt template {template_name}: {self.template['description']}")

    def generate_prompt( # returns the full prompt from instruction and optional input
            self,
            instruction: str,
            input: Union[None, str] = None,
            label: Union[None, str] = None,
    ) -> str:
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
    
    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt)
        if not self.train_on_inputs: # false이면 promp까지 masking
            user_prompt = self.generate_prompt(data_point["instruction"], data_point["input"])
            tokenized_user_prompt = self.tokenize(user_prompt)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            if self.add_eos_token:
                user_prompt_len -= 1
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:] 
        return tokenized_full_prompt
    
    def tokenize(self, prompt):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.cutoff_len
                and self.add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result