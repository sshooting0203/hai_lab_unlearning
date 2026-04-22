# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch, copy
from torch import nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
from torch import Tensor
from jaxtyping import Int, Float
from src_lunar.utils.hook_utils import add_hooks

class ModelBase(ABC):
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.model: AutoModelForCausalLM = self._load_model(model_name_or_path)
        self.tokenizer: AutoTokenizer = self._load_tokenizer(model_name_or_path)

        self.tokenize_instructions_fn = self._get_tokenize_instructions_fn() # tokenizer로 배치 텐서(input_ids, attention_mask)로 바꿔주는 함수
        self.eoi_toks = self._get_eoi_toks() 
        # @ 수정 : 프롬프트에서 instruction이 끝나는 지점에 붙는 토큰의 ID 리스트 반환
        self.refusal_toks = self._get_refusal_toks() 
        # 사용 x, 거절(refusal)을 나타내는 대표 토큰의 ID 리스트 반환

        self.model_block_modules = self._get_model_block_modules() # decoder layer 가져오기
        self.model_attn_modules = self._get_attn_modules()
        self.model_mlp_modules = self._get_mlp_modules()

    def del_model(self):
        if hasattr(self, "model") and self.model is not None:
            del self.model

    @abstractmethod
    def _load_model(self, model_name_or_path: str) -> AutoModelForCausalLM:
        pass
    @abstractmethod
    def _load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        pass
    @abstractmethod
    def _get_tokenize_instructions_fn(self):
        pass
    @abstractmethod
    def _get_eoi_toks(self):
        pass
    @abstractmethod
    def _get_refusal_toks(self):
        pass
    @abstractmethod
    def _get_model_block_modules(self):
        pass
    @abstractmethod
    def _get_attn_modules(self):
        pass
    @abstractmethod
    def _get_mlp_modules(self):
        pass
    @abstractmethod
    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        pass
    @abstractmethod
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff: float, layer: int):
        pass

    def generate_completions( # eval할 때 -> generate_and_save_completions_for_dataset 함수에서 호출(사용 x)
        self, dataset, fwd_pre_hooks=[], fwd_hooks=[], batch_size=8, max_new_tokens=64
    ):
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            use_cache=True,
        )

        generation_config.pad_token_id = self.tokenizer.pad_token_id
        completions = []
        instructions = [
            x["instruction"] if "instruction" in x else x["question"]
            for x in dataset
            if "instruction" in x or "question" in x
        ]

        if "category" in dataset[0]:
            categories = [x["category"] for x in dataset]
        else:
            categories = [x["edge"] for x in dataset]

        if "answer" in dataset[0]:
            answer = [x["answer"] for x in dataset]
        else:
            answer = [None for x in dataset]

        for i in tqdm(range(0, len(dataset), batch_size)):
            tokenized_instructions = self.tokenize_instructions_fn(
                instructions=instructions[i : i + batch_size]
            )

            with add_hooks(
                module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks
            ):
                generation_toks = self.model.generate(
                    input_ids=tokenized_instructions.input_ids.to(self.model.device),
                    attention_mask=tokenized_instructions.attention_mask.to(
                        self.model.device
                    ),
                    generation_config=generation_config,
                )

                generation_toks = generation_toks[
                    :, tokenized_instructions.input_ids.shape[-1] :
                ]

                for generation_idx, generation in enumerate(generation_toks):
                    completions.append(
                        {
                            "category": categories[i + generation_idx],
                            "prompt": instructions[i + generation_idx],
                            "response": self.tokenizer.decode(
                                generation, skip_special_tokens=True
                            ).strip(),
                            "original_answer": answer[i + generation_idx],
                        }
                    )
        return completions

    # @ 수정 : inference -> 들어오는 샘플과 이미 저장된 distibution들끼리의 거리 계산
    def calculate_distances(self, features, distributions, distance_type="L2", temperature=1.0):
        with torch.no_grad():
            distances = []
            for gaussian in distributions:
                mean = gaussian.mean
                var = gaussian.var
                if distance_type == 'L2':
                    dist = torch.norm(features - mean, dim=1)
                elif distance_type == 'Gaussian':
                    var = var + 1e-6 
                    log_prob = -0.5 * (torch.log(2 * torch.pi * var) + ((features - mean) ** 2) / var)
                    log_prob_sum = log_prob.sum(dim=1)  # [B]
                    dist = log_prob_sum
                elif distance_type == 'Attention':
                    dim = features.size(-1)
                    ty=features.dtype
                    attn_score = torch.matmul(features, mean.type(ty)) / torch.sqrt(torch.tensor(dim))
                    dist = attn_score  # [B]
                elif distance_type == 'Cosine': # distibutions를 모아놓은 것의 mean과 feature의 유사도 계산 
                    features_norm = F.normalize(features, dim=1)
                    mean_norm = F.normalize(mean.reshape(1, -1), dim=1)
                    dist = torch.matmul(features_norm, mean_norm.t())  # [B, 1]
                    dist=dist.reshape(-1)
                else:
                    raise ValueError(f"Unsupported distance type: {distance_type}")
                distances.append(dist)
            
            if distance_type == 'Gaussian':
                dis_weights = torch.stack(distances, dim=1)  # [B, num_distributions]
                dis_weights = F.softmax(dis_weights / temperature, dim=1)
            elif distance_type == 'Attention':
                dis_weights = torch.stack(distances, dim=1)  # [B, num_distributions]
                dis_weights = F.softmax(dis_weights / temperature, dim=1)
            elif distance_type == 'L2':
                dis_weights = torch.stack(distances, dim=1)  # [B, num_distributions]
                dis_weights = F.softmax(-dis_weights / temperature, dim=1)
            elif distance_type == 'Cosine':
                dis_weights = torch.stack(distances, dim=1)
                dis_weights = F.softmax(dis_weights / temperature, dim=1)
            else:
                raise ValueError(f"Unsupported distance type: {distance_type}")
            
            return dis_weights.unsqueeze(-1)  # [B, num_distributions, 1]

    
