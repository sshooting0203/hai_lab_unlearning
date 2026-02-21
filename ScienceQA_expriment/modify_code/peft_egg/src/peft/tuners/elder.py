# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math, re, warnings, torch 
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft_egg.src.peft.tuners.base import Gate, GraceLayer, LoraLayer
from ..import_utils import is_bnb_4bit_available, is_bnb_available
from ..utils import (
    COMMON_LAYERS_PATTERN,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)


NO_LORA = -100
LORA_BLOCK_MAPPING = []
BINARY_CODE = []

# 1. ELDERModel -> ELDERLinear(experts + Gating Network) 
# -> ElderLoraLayer(expert) -> self.lora_A = nn.ModuleDict({})
# -> Gate(Gating Network) -> self.gate = nn.Linear(d_model, self.num_expert, bias=gate_bias)

# 2. ELDERModel -> ELDERGraceLinear(experts + Gating Network) 
# -> GraceLayer -> self.VecDB = VecDB(grace_config)

#=====================elder model/config=====================
class ElderModel(torch.nn.Module): # @ ELDERMODEL 로드 
    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.device = model.device

        self.add_my_adapter(adapter_name, self.peft_config[adapter_name])
        self.add_grace(adapter_name, self.peft_config[adapter_name]) # 用GRACE记录并固定每个edit的表示向量
        # @ 1. adapter 추가 2. grace 추가

    def add_my_adapter(self, adapter_name, config=None): # @ Adapter Add 1
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_lora_config(config, model_config) # 检查config是否指定了修改模型哪个位置
            self.peft_config[adapter_name] = config
        self.gate_linear_layers = self._find_and_replace_my_adapter(adapter_name) # # @ Adapter Add 2
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)


    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "LoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def add_grace(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_melo_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace_grace(adapter_name) # @ Grace Add 1
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)

    def _check_quantization_dependency(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit) and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

    def _check_target_module_exists(self, lora_config, key):
        if isinstance(lora_config.target_modules, str):
            target_module_found = re.fullmatch(lora_config.target_modules, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            is_using_layer_indexes = getattr(lora_config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(lora_config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(lora_config.layers_to_transform, int):
                            target_module_found = layer_index == lora_config.layers_to_transform
                        else:
                            target_module_found = layer_index in lora_config.layers_to_transform

                        break
                    else:
                        target_module_found = False
        return target_module_found

    def _check_grace_layer_exists(self, grace_layer:str, key):
        target_module_found = re.fullmatch(grace_layer, key)

        return target_module_found

    def _find_and_replace_my_adapter(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency() # 没啥用
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        gate_linear_layers = [] 

        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue
            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            new_module = self._create_new_module(lora_config, adapter_name, target)  # @ Adapter Add 3 : ElderLinear을 생성(예를 들어 어떤 linear ffn layer -> MoE layer)
            self._replace_module(parent, target_name, new_module, target)
            gate_linear_layers.append(new_module.gate.gate)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )
        
        return gate_linear_layers

    def _create_new_grace_module(self, config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "fan_in_fan_out": config.fan_in_fan_out,
            "top_k": 2,
            "num_experts": config.num_experts,
        }

        if isinstance(target, torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = config.fan_in_fan_out = False
        elif isinstance(target, Conv1D):
            in_features, out_features = (
                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
            )
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target grace module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` and 'torch.nn.Conv1D' are supported."
            )
        new_module = ElderGraceLinear(
            adapter_name, 
            in_features, 
            out_features, 
            config.grace_config, 
            gate_linears=self.gate_linear_layers, **kwargs) # @ Grace Add 3
        return new_module

    def _find_and_replace_grace(self, adapter_name): # @ grace 추가
        config = self.peft_config[adapter_name]
        is_target_module_in_base_model = False
        grace_layer = config.grace_layer
        key_list = [key for key,_ in self.model.named_modules()]

        for key in key_list:
            if not self._check_grace_layer_exists(grace_layer,key):
                continue
            print(f"[Debug] Target Grace Layer is found: {key}")
            is_target_module_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                raise ValueError("Cannot set LoraLayer as GraceLayer")
            new_module = self._create_new_grace_module(config, adapter_name, target) # @ Grace Add 2
            self._replace_module(parent, target_name, new_module, target)
        if not is_target_module_in_base_model:
            raise ValueError(
                f"Target grace modules {config.model.grace_layer} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _create_new_module(self, lora_config, adapter_name, target):
        """ create new Linear layer """
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "num_experts": lora_config.num_experts,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "is_redundant_experts": lora_config.is_redundant_experts,
        }

        if isinstance(target, torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        elif isinstance(target, Conv1D):
            in_features, out_features = (
                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
            )
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
            )
        new_module = ElderLinear(adapter_name, in_features, out_features, bias=bias, **kwargs) # @ ElderLinear을 생성
        return new_module


    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)
            if "ranknum" in name:
                module.to(old_module.weight.device)


    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, ElderLinear):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def disable_grace_layer(self):
        for module in self.model.modules():
            if isinstance(module, GraceLayer):
                module.disable_grace = True

    def enable_grace_layer(self):
        for module in self.model.modules():
            if isinstance(module, GraceLayer):
                module.disable_grace = False


    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.merge()

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.unmerge()

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config
    @staticmethod
    def _prepare_melo_config(peft_config, model_config):
        if peft_config.grace_layer is None:
                raise ValueError("Please specify `grace_layer` in `peft_config`")
        return peft_config

    def merge_and_unload(self):
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.
        """
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging LORA layers")

        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                else:
                    bias = target.bias is not None
                    new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def add_weighted_adapter(self, adapters, weights, adapter_name):
        if len({self.peft_config[adapter].r for adapter in adapters}) != 1:
            raise ValueError("All adapters must have the same r value")
        self.peft_config[adapter_name] = self.peft_config[adapters[0]]
        self.peft_config[adapter_name].lora_alpha = self.peft_config[adapters[0]].r
        self._find_and_replace(adapter_name)
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        _freeze_adapter(self.model, adapter_name)
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                if adapter_name in target.lora_A:
                    target.lora_A[adapter_name].weight.data = target.lora_A[adapter_name].weight.data * 0.0
                    target.lora_B[adapter_name].weight.data = target.lora_B[adapter_name].weight.data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_A:
                            continue
                        target.lora_A[adapter_name].weight.data += (
                            target.lora_A[adapter].weight.data * weight * target.scaling[adapter]
                        )
                        target.lora_B[adapter_name].weight.data += target.lora_B[adapter].weight.data * weight

                elif adapter_name in target.lora_embedding_A:
                    target.lora_embedding_A[adapter_name].data = target.lora_embedding_A[adapter_name].data * 0.0
                    target.lora_embedding_B[adapter_name].data = target.lora_embedding_B[adapter_name].data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_embedding_A:
                            continue
                        target.lora_embedding_A[adapter_name].data += (
                            target.lora_embedding_A[adapter].data * weight * target.scaling[adapter]
                        )
                        target.lora_embedding_B[adapter_name].data += target.lora_embedding_B[adapter].data * weight

@dataclass
class ElderConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of ELDER model 
    """
    num_iters: int = field(default=None, metadata={"help": "number of training iterations"})

    num_experts: int = field(default=8, metadata={"help": "Num of experts in LoRA-MoE"})

    is_redundant_experts: bool = field(default=False) # 중복되는 expert를 허용 

    ### LoRA Parm
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})

    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )

    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})

    fan_in_fan_out: bool = field( # target layer의 weight 저장된 형태가 linear와 다를 때 사용 
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )

    ### MELO Parm
    grace_layer: str = field(
        default = None,
        metadata = {
            "help":"Module name as a grace layer"
        }
    )

    grace_config: dict = field(
        default = None,
        metadata={
            "help": "Default settings of the grace layer"
        }
    )

    bias: str = field( # weight뿐만 아니라 bias도 학습할지 여부
        default="none", 
        metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"}
    )
    
    modules_to_save: Optional[List[str]] = field( # LoRA layer외에 추가로 학습/저장할 layer를 지정
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    init_lora_weights: bool = field( # LoRA weight AB를 초기화할지 여부
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora layers."},
    )

    layers_to_transform: Optional[Union[List, int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )

    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.ELDER # PEFT module: ELDER


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        # if "lora_" not in n:
        if "lora_" not in n and "gate" not in n:
            p.requires_grad = False
        if "gate_proj" in n: # 处理Llama2里的mlp.gate_proj
            p.requires_grad = False
    # for name, m in model.named_modules():
    #     if isinstance(m, Gate):
    #         for n, p in m.named_parameters():
    #             p.requires_grad = True

    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class ElderLoraLayer(nn.Module): # @ experts : 1 expert == ElderLoraLayer
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super(ElderLoraLayer, self).__init__()
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})

        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # lora
        if r > 0:
            self.lora_A = nn.ParameterDict({adapter_name: nn.Parameter(torch.zeros((self.in_features, r)))})
            self.lora_B = nn.ParameterDict({adapter_name: nn.Parameter(torch.zeros((r, self.out_features)))})
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name])


#===========================elder layer config===========================
class ElderLinear(nn.Linear): # @@ MoE(ElderLoraLayer x expert) + Gating Network(Gate)
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        num_experts: int = 4,
        top_k: int = 2,
        gate_bias: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
        is_redundant_experts: bool = False,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        nn.Linear.__init__(self, in_features=in_features, out_features=out_features, **kwargs)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.active_adapter = adapter_name
        self.disable_adapters = False
        self.key_idx = -1

        """ Add parameters for ELDER adapter """
        def one_expert(d_model):
            expert_func = ElderLoraLayer(d_model, out_features) # @ experts : 1 expert == ElderLoraLayer
            expert_func.update_layer(self.active_adapter, r, lora_alpha, lora_dropout, init_lora_weights)
            return expert_func
        
        expert = one_expert
        self.num_experts = num_experts
        self.gate = Gate(in_features, num_experts, top_k, gate_bias=gate_bias) # @ gating network : Gate
        self.experts = nn.ModuleList([expert(in_features) for _ in range(num_experts)]) # @ experts : 1 expert == ElderLoraLayer

    def merge(self):
        """ 
        Merge LoRA Weights
        """
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    def forward(self, x: torch.Tensor):
        # print("ELDERLINEAR FORWARD HIT", self.__class__.__name__, "at", self)
        previous_dtype = x.dtype
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        """ 
        First computes gate output, 
        then conduct MoE forward according to the gate 
        """
        if self.disable_adapters:
            return result

        original_shape = x.shape # [B, L, H]
        batch_size = original_shape[0]
        seq_length = original_shape[1]

        """ take representations from GRACE layer for routing """
        batch_query = SEQ_REPR # @ 수정
        
        gate_top_k_idx, gate_score = self.gate(batch_query)
        full_gate_weights = torch.zeros((batch_size, self.num_experts), device=x.device) # [B, n-experts]
        full_gate_weights.scatter_(dim=1, index=gate_top_k_idx, src=gate_score)

        expert_outputs = torch.stack([
            (e.lora_dropout[self.active_adapter](x) @ e.lora_A[self.active_adapter] @ e.lora_B[self.active_adapter])\
                 * e.scaling[self.active_adapter] for e in self.experts
        ], dim=1)
        lora_result = torch.einsum('belh, be -> blh', expert_outputs, full_gate_weights)
        # print("full gate_weights : ", full_gate_weights)

        """ Make sure deferral scope """
        flag = IN_EDIT_SCOPE # condition list
        print("flag list : ", flag)
        condition_tensor = torch.tensor(flag).view(len(flag), 1, 1).to(result.device)
        result_tensor = torch.where(condition_tensor, result + lora_result, result) # apply lora only to date in the edit scope

        self.gate_loss = self.gate.loss
        result_tensor = result_tensor.to(previous_dtype)
        return result_tensor


#===========================elder grace layer config===========================
class ElderGraceLinear(nn.Linear, GraceLayer): # @@ GRACE(Anchor Routing)
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        grace_config: dict,
        gate_linears: list,
        top_k: int,
        num_experts: int,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features,**kwargs)
        GraceLayer.__init__(self,grace_config=grace_config, in_features=in_features, out_features=out_features)
        self.fan_in_fan_out = fan_in_fan_out
        self.gate_linear_layers = gate_linears
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        self.threshold = grace_config["threshold"] # @ 수정
        print("[Debug] num_experts : ", self.num_experts, ", self.top_k : ", self.top_k, ", self.threshold : ", self.threshold)


    def get_bin_code(self, batch_query): # @ 수식 : 실제로 test sample의 allocate pattern을 얻기
        """ when this is the last_iter, record bin_code """
        with torch.no_grad():
            # print("\n[DEBUG] ===== get_bin_code START =====")
            k_hot_outputs = []

            for linear in self.gate_linear_layers:
                gate = linear(batch_query).detach()
                # print(f"[DEBUG] Layer gate output shape:", gate.shape)
                _, gate_top_k_idx = torch.topk(
                    gate, k=self.top_k, dim=-1, largest=True, sorted=False
                )  # [.. x top_k]
                # print(f"[DEBUG] Layer gate output (first sample):")
                # print(gate[0])
                k_hot_outputs.append(torch.zeros((gate_top_k_idx.shape[0], self.num_experts), dtype=torch.int32, device=gate.device).scatter_(1, gate_top_k_idx, 1))
                k_hot = torch.zeros(
                    (gate_top_k_idx.shape[0], self.num_experts),
                    dtype=torch.int32,
                    device=gate.device
                ).scatter_(1, gate_top_k_idx, 1)
                # print(f"[DEBUG] Layer k-hot (first sample):")
                # print(k_hot[0])

            gate_output_concat = torch.cat(k_hot_outputs, dim=1) # [B, L*E] L:n-layers
            self.binary_code = gate_output_concat
            # print("\n[DEBUG] ===== get_bin_code END =====")

    
    def discriminate(self, threshold): # @ 수식 : test sample과 이미 계산된 training edit의 allocate pattern을 서로 비교
        # print("\n[DEBUG] ===== Discriminate Start =====")
        binary_code_dataset_bool = torch.cat(BINARY_CODE, dim=0).to(torch.bool)
        expanded_self_binary_code_bool = self.binary_code.to(torch.bool).unsqueeze(1)
        
        # print("[DEBUG] self.binary_code shape:", self.binary_code.shape)
        # print("[DEBUG] dataset binary_code shape:", binary_code_dataset_bool.shape)
        # print("[DEBUG] self.binary_code (first sample):")
        print(self.binary_code[0])

        xor_result = expanded_self_binary_code_bool ^ binary_code_dataset_bool
        hamming_distance = xor_result.sum(dim=2)
        distances_min = hamming_distance.min(dim=1)[0].tolist()
        # print("[DEBUG] threshold:", threshold)
        # print("[DEBUG] min hamming distance per sample:", distances_min)
        # print("\n[DEBUG] ===== Discriminate END=====")
        flag = [x < threshold for x in distances_min]
        return flag

    def set_discrimination_anchor(self, hash_initialization):
        binary_code_dataset_bool = torch.stack([torch.cat(value, dim=0) for value in hash_initialization.values()]) # [num-class, length-of-code]
        self.discriminate_anchor = binary_code_dataset_bool.to(self.weight.device)
        self.hash_init_as_anchor = True


    def forward(self, x: torch.Tensor):
        # print("GRACE FORWARD HIT", self.__class__.__name__, "at", self)
        global SEQ_REPR
        global IN_EDIT_SCOPE

        layer_out = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_grace:
            return layer_out

        if isinstance(self.key_id, torch.Tensor):
            batch_query = torch.gather(x, 1, self.key_id.unsqueeze(1).unsqueeze(1).expand(-1, -1, x.shape[2])).squeeze(1).detach()
        elif self.key_id == -1: 
            batch_query = x[:, self.key_id].detach()
        else:
            raise NotImplementedError
        SEQ_REPR = batch_query

        if hasattr(self, 'is_last_iter'):
            # print("last iter is set")
            if self.is_last_iter: # last iteration에서 BINARY CODE를 채우기
                print("last iter is true")
                self.get_bin_code(batch_query) 
                BINARY_CODE.append(self.binary_code)
                self.is_last_iter = False # @ 수정

        """ Inference time. Deferral Mechanism """
        if hasattr(self, 'editing'): # @ 수식 : deferral mechanism
            if not self.editing: # inference
                print("inference")
                self.get_bin_code(batch_query)
                IN_EDIT_SCOPE = self.discriminate(threshold=self.threshold)
            else: # editing
                # print("editing")
                IN_EDIT_SCOPE = [True for _ in range(x.shape[0])]
        
        return layer_out

