from src_lunar.model_utils.llama2_model import Llama2Model
from src_lunar.model_utils.llama3_model import Llama3Model
from src_lunar.model_utils.llama31model import Llama31Model
from src_lunar.model_utils.mistral_model import MistralModel
from src_lunar.model_utils.gemma_model import GemmaModel
# from src_lunar.model_utils.qwen_model import QwenModel
from src_lunar.model_utils.qwen2_model import Qwen2Model
from src_lunar.model_utils.qwen25model import Qwen25Model

PREFIX_BY_MODEL = {
    Llama2Model: {
        "begin": "",
        "user": "[INST]",
        "assistant": " [/INST]",
    },
    MistralModel: {
        "begin": "",
        "user": "[INST]",
        "assistant": " [/INST]",  
    },
    Llama3Model: {
        "begin": "<|begin_of_text|>",
        "user": "<|start_header_id|>user<|end_header_id|>",
        "assistant": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    },
    Llama31Model: {
        "begin": "<|begin_of_text|>",
        "user": "<|start_header_id|>user<|end_header_id|>",
        "assistant": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    },
    # QwenModel: {
    #     "begin": "",  # Qwen은 보통 begin 토큰 없음
    #     "user": "<|im_start|>user\n",
    #     "assistant": "<|im_end|>\n<|im_start|>assistant\n",
    # },
    Qwen2Model: {
        "begin": "",  # Qwen은 보통 begin 토큰 없음
        "user": "<|im_start|>user\n",
        "assistant": "<|im_end|>\n<|im_start|>assistant\n",
    },
    Qwen25Model: {
        "begin": "",  # Qwen은 보통 begin 토큰 없음
        "user": "<|im_start|>user\n",
        "assistant": "<|im_end|>\n<|im_start|>assistant\n",
    },
    GemmaModel: {
        "begin": "",
        "user": "<start_of_turn>user\n",
        "assistant": "<end_of_turn>\n<start_of_turn>model\n",
    },
}

def resolve_prefix_cfg(model_base):
    # isinstance로 매칭되는 첫 키를 사용
    for cls, cfg in PREFIX_BY_MODEL.items():
        if isinstance(model_base, cls):
            return cfg
    raise ValueError(f"Unsupported model type: {type(model_base)}")

def extract_instruction_from_model(s: str, model_base) -> str:
    cfg = resolve_prefix_cfg(model_base)

    begin = cfg.get("begin", "")
    if begin:
        s = s.replace(begin, "")

    user_prefix = cfg["user"]
    if user_prefix in s:
        s = s.split(user_prefix, 1)[1]
    s = s.lstrip("\n")

    assist_prefix = cfg["assistant"]
    if assist_prefix in s:
        s = s.split(assist_prefix, 1)[0] # 대부분은 assistant prefix 이전이 instruction

    return s.strip()