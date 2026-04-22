# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch, functools
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
from torch import Tensor
from jaxtyping import Float
from src_lunar.utils.utils import get_orthogonalized_matrix
from src_lunar.model_utils.model_base import ModelBase

SAMPLE_SYSTEM_PROMPT = """You are a helpful assistant."""

QWEN_REFUSAL_TOKS = [40, 2121]  # ['I', 'As']


def build_prompt_from_instruction_qwen(tokenizer: AutoTokenizer, instruction, system: str = None):
# Utterance: ai, how long have you been alive
# [DEBUG] type(system) = <class 'NoneType'>
# [DEBUG] type(instruction) = <class 'str'>
# [DEBUG] instruction = Classify the user's intent from the following utterance. Return only the intent label.
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": instruction})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def tokenize_instructions_qwen_chat(
    tokenizer, 
    instructions, 
    system=None
):
    processed_instructions = []
    for inst in instructions:
        if isinstance(inst, dict):
            processed_instructions.append(inst.get("text_input", inst.get("instruction", inst.get("question", str(inst)))))
        else:
            processed_instructions.append(inst)

    prompts = [
        build_prompt_from_instruction_qwen(tokenizer=tokenizer, instruction=instruction, system=system)
        for instruction in processed_instructions
    ]
    return tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    )

def orthogonalize_qwen_weights(model, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
        model.model.embed_tokens.weight.data, direction
    )

    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
            block.self_attn.o_proj.weight.data.T, direction
        ).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
            block.mlp.down_proj.weight.data.T, direction
        ).T


def act_add_qwen_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.model.layers[layer - 1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer - 1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    if (
        not hasattr(model.model.layers[layer - 1].mlp.down_proj, "bias")
        or model.model.layers[layer - 1].mlp.down_proj.bias is None
    ):
        model.model.layers[layer - 1].mlp.down_proj.bias = torch.nn.Parameter(
            torch.zeros(model.config.hidden_size, dtype=dtype, device=device)
        )

    model.model.layers[layer - 1].mlp.down_proj.bias.data.add_(bias)


class Qwen2Model(ModelBase):
    def _load_model(self, model_path, dtype=torch.bfloat16):
        device="cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).eval()
        model = model.to(device).eval()   
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token = "<|endoftext|>"
                tokenizer.pad_token_id = 151643
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_qwen_chat,
            tokenizer=self.tokenizer,
            system=None,
        )

    def _get_eoi_toks(self):
        dummy_instruction = "DUMMY_INSTRUCTION_FOR_EOI"

        messages = [{"role": "user", "content": dummy_instruction}]

        prompt_without_gen = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_with_gen = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        ids_without = self.tokenizer(
            prompt_without_gen,
            add_special_tokens=False,
        )["input_ids"]
        ids_with = self.tokenizer(
            prompt_with_gen,
            add_special_tokens=False,
        )["input_ids"]

        return ids_with[len(ids_without):]

    def _get_refusal_toks(self):
        return QWEN_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList(
            [block_module.self_attn for block_module in self.model_block_modules]
        )

    def _get_mlp_modules(self):
        return torch.nn.ModuleList(
            [block_module.mlp for block_module in self.model_block_modules]
        )

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_qwen_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(
            act_add_qwen_weights, direction=direction, coeff=coeff, layer=layer
        )

    def _to(self, device=None, dtype=None):
        if device is not None and dtype is not None:
            self.model = self.model.to(device=device, dtype=dtype)
        elif device is not None:
            self.model = self.model.to(device=device)
        elif dtype is not None:
            self.model = self.model.to(dtype=dtype)
        return self

    def _eval(self):
        self.model.eval()
        return self

    def _forward(self, batch):
        outputs = self.model(**batch)
        return outputs

    def _device(self):
        return next(self.model.parameters()).device.type

    def _generate(
        self,
        input_ids,
        attention_mask,
        max_length,
        max_new_tokens,
        do_sample,
        num_beams,
        num_return_sequences,
        use_cache,
        pad_token_id,
        output_scores,
        return_dict_in_generate,
    ):
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
        )
        return output