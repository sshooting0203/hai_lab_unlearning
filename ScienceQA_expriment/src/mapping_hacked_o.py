# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
from typing import TYPE_CHECKING, Any
import torch, re
from peft.config import PeftConfig
from peft.mixed_model import PeftMixedModel
from src.lora_model_hacked_o import LoraModel
from peft.utils import _prepare_prompt_learning_config
from src.peft_model_hacked_o import (
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForFeatureExtraction,
    PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
)
from peft.tuners import (
    AdaLoraConfig,
    AdaLoraModel,
    AdaptionPromptConfig,
    IA3Config,
    IA3Model,
    LoHaConfig,
    LoHaModel,
    LoKrConfig,
    LoKrModel,
    LoraConfig,
    MultitaskPromptTuningConfig,
    OFTConfig,
    OFTModel,
    PolyConfig,
    PolyModel,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel

MODEL_TYPE_TO_PEFT_MODEL_MAPPING: dict[str, PeftModel] = {
    "SEQ_CLS": PeftModelForSequenceClassification,
    "SEQ_2_SEQ_LM": PeftModelForSeq2SeqLM,
    "CAUSAL_LM": PeftModelForCausalLM,
    "TOKEN_CLS": PeftModelForTokenClassification,
    "QUESTION_ANS": PeftModelForQuestionAnswering,
    "FEATURE_EXTRACTION": PeftModelForFeatureExtraction,
}

PEFT_TYPE_TO_CONFIG_MAPPING: dict[str, PeftConfig] = {
    "ADAPTION_PROMPT": AdaptionPromptConfig,
    "PROMPT_TUNING": PromptTuningConfig,
    "PREFIX_TUNING": PrefixTuningConfig,
    "P_TUNING": PromptEncoderConfig,
    "LORA": LoraConfig,
    "LOHA": LoHaConfig,
    "LOKR": LoKrConfig,
    "ADALORA": AdaLoraConfig,
    "IA3": IA3Config,
    "MULTITASK_PROMPT_TUNING": MultitaskPromptTuningConfig,
    "OFT": OFTConfig,
    "POLY": PolyConfig,
}

PEFT_TYPE_TO_TUNER_MAPPING = {
    "LORA": LoraModel,
    "LOHA": LoHaModel,
    "LOKR": LoKrModel,
    "ADALORA": AdaLoraModel,
    "IA3": IA3Model,
    "OFT": OFTModel,
    "POLY": PolyModel,
}

def get_peft_config(config_dict: dict[str, Any]) -> PeftConfig:
    """
    Returns a Peft config object from a dictionary.
    Args:
        config_dict (`Dict[str, Any]`): Dictionary containing the configuration parameters.
    """
    return PEFT_TYPE_TO_CONFIG_MAPPING[config_dict["peft_type"]](**config_dict)

def get_peft_model(
    model: PreTrainedModel, 
    peft_config: PeftConfig, 
    adapter_name: str = "default", 
    mixed: bool = False, 
    lora_start_layer: int = -1,
) -> PeftModel | PeftMixedModel:
    """
    Returns a Peft model object from a model and a config.
    Args:
        model ([`transformers.PreTrainedModel`]):
            Model to be wrapped.
        peft_config ([`PeftConfig`]):
            Configuration object containing the parameters of the Peft model.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
        mixed (`bool`, `optional`, defaults to `False`):
            Whether to allow mixing different (compatible) adapter types.
    """
    model_config = getattr(model, "config", {"model_type": "custom"})
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()

    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
    setattr(peft_config, "lora_start_layer", lora_start_layer) # @ 수정 
    
    if mixed:
        return PeftMixedModel(model, peft_config, adapter_name=adapter_name)

    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and not peft_config.is_prompt_learning:
        return PeftModel(model, peft_config, adapter_name=adapter_name)

    if peft_config.is_prompt_learning:
        peft_config = _prepare_prompt_learning_config(peft_config, model_config)

    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](model, peft_config, adapter_name=adapter_name)

def debug_peft_basic(peft_model, adapter_name="default"):
    print("active_adapter:", getattr(peft_model, "active_adapter", None))
    lora_params = [(n, p) for n, p in peft_model.named_parameters() if "lora_" in n]
    print("num lora params:", len(lora_params))
    if len(lora_params) == 0:
        print("LoRA 파라미터가 하나도 없음")
        return
    # for n, p in lora_params[:8]:
        # print(" ", n, tuple(p.shape), p.dtype, p.device)
    # with torch.no_grad():
    #     s = 0.0
    #     for _, p in lora_params[:50]:  # 너무 많이 돌면 느리니 일부만
    #         s += float(p.abs().sum().item())
    #     print("abs-sum(first 50 lora params):", s)
    #     if s == 0.0:
    #         print("LoRA 값이 전부 0일 수 있음(로드 실패/초기화 상태 가능)")
    #     else:
    #         print("LoRA 값이 0이 아님(로드된 흔적 있음)")

def debug_lora_layers(peft_model, start_layer: int):
    layer_re = re.compile(r"\.layers\.(\d+)\.")
    found = {}
    for n, _ in peft_model.named_parameters():
        if "lora_" not in n:
            continue
        m = layer_re.search(n)
        if m:
            idx = int(m.group(1))
            found[idx] = found.get(idx, 0) + 1

    if not found:
        print("레이어별 LoRA 파라미터를 못 찾음")
        return

    min_layer, max_layer = min(found), max(found)
    print(f"LoRA layers present: min={min_layer}, max={max_layer}")
    early = [k for k in sorted(found) if k < start_layer]
    if early:
        print("start_layer보다 앞 레이어에도 LoRA가 존재:", early[:20])
    else:
        print("start_layer 이전 레이어에는 LoRA가 없음(주입 스킵 적용)")

def debug_forward_effect(peft_model, tokenizer, text="hello", adapter_name="default", device=None):
    peft_model.eval()
    if device is None:
        device = next(peft_model.parameters()).device

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        if hasattr(peft_model, "set_adapter"):
            peft_model.set_adapter(adapter_name)
        out_on = peft_model(**inputs).logits
        try:
            with peft_model.disable_adapter():
                out_off = peft_model(**inputs).logits
        except Exception:
            print("disable_adapter() 없음/실패 -> 대체 방법 필요(아래 참고)")
            return

    diff = (out_on - out_off).abs().mean().item()
    print("mean |logits_on - logits_off| =", diff)
    if diff == 0.0:
        print("차이가 0: LoRA가 적용 안 되었거나(혹은 아주 특이하게 영향이 0)일 수 있음")
    else:
        print("LoRA ON/OFF 차이 존재: forward에 영향 있음")


def inject_adapter_in_model(
    peft_config: PeftConfig, model: torch.nn.Module, adapter_name: str = "default"
) -> torch.nn.Module:
    r"""
    A simple API to create and inject adapter in-place into a model. Currently the API does not support prompt learning
    methods and adaption prompt. Make sure to have the correct `target_names` set in the `peft_config` object. The API
    calls `get_peft_model` under the hood but would be restricted only to non-prompt learning methods.

    Args:
        peft_config (`PeftConfig`):
            Configuration object containing the parameters of the Peft model.
        model (`torch.nn.Module`):
            The input model where the adapter will be injected.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
    """
    if peft_config.is_prompt_learning or peft_config.is_adaption_prompt:
        raise ValueError("`create_and_replace` does not support prompt learning and adaption prompt yet.")

    if peft_config.peft_type not in PEFT_TYPE_TO_TUNER_MAPPING.keys():
        raise ValueError(
            f"`inject_adapter_in_model` does not support {peft_config.peft_type} yet. Please use `get_peft_model`."
        )

    tuner_cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type] # By instantiating a peft model we are injecting randomly initialized LoRA layers into the model's modules.
    peft_model = tuner_cls(model, peft_config, adapter_name=adapter_name)
    return peft_model.model
