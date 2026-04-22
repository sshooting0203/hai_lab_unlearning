# Copyright (c) Meta Platforms, Inc. and affiliates.
from src_lunar.model_utils.llama2_model import Llama2Model
from src_lunar.model_utils.llama3_model import Llama3Model
from src_lunar.model_utils.llama31model import Llama31Model
from src_lunar.model_utils.mistral_model import MistralModel
from src_lunar.model_utils.gemma_model import GemmaModel
# from src_lunar.model_utils.qwen_model import QwenModel
from src_lunar.model_utils.qwen25model import Qwen25Model
from src_lunar.model_utils.qwen2_model import Qwen2Model

def load_model(model_family, model_path, device):
    if model_family == "llama2-7b-chat":
        model_base = Llama2Model(model_path)
    elif model_family == "mistral-7b-instruct":
        model_base = MistralModel(model_path)
    elif model_family == "llama3-8b-instruct":
        model_base = Llama3Model(model_path)
    elif model_family == "llama3.1-8b-instruct":
        model_base = Llama31Model(model_path)
    elif model_family == "gemma-7b-it":
        model_base = GemmaModel(model_path)
    elif model_family == "Qwen2-7B-Instruct":
        model_base = Qwen2Model(model_path)
    elif model_family == "Qwen2.5-7B-Instruct":
        model_base = Qwen25Model(model_path)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
    model_base = model_base._to(device)
    return model_base
