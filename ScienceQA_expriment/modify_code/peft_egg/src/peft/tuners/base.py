# coding=utf-8
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
import math, re, warnings, torch 
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

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
if is_bnb_available():
    import bitsandbytes as bnb


class LoraLayer:
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        self.nd_lora_A = dynamic()
        self.nd_lora_B = dynamic()
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, num_rank_per_block):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            """ 
            创建两个LoRA参数lora_A和lora_B。这些参数是通过nn.ParameterDict存储的，每个参数都使用nn.Parameter封装。参数的形状分别为(self.in_features, r)和(r, self.out_features)，其中r是低秩矩阵的秩，代表这些矩阵的列数（对于lora_A）和行数（对于lora_B）。初始状态下，这些参数被设置为零矩阵，使用self.weight.new_zeros方法确保它们和模型权重在同一个设备上。
            """
            self.lora_A = nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((self.in_features, r)))})
            self.lora_B = nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((r, self.out_features)))})
            self.nd_lora_A.update_dynamic(r, num_rank_per_block) # 设置maximun_rank，maximum_block等参数
            self.nd_lora_B.update_dynamic(r,num_rank_per_block)
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_conv2d(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            kernel_size = self.kwargs["kernel_size"]
            stride = self.kwargs["stride"]
            padding = self.kwargs["padding"]
            self.lora_A.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)})
            )
            self.lora_B.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)})
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_embedding_A.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((r, self.in_features)))})
            )
            self.lora_embedding_B.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((self.out_features, r)))})
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name])
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

class RedundantLoraLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, **kwargs) -> None:
        super().__init__()


#===========================general model layer config===========================
class Linear(nn.Linear, LoraLayer):
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
        num_rank_per_block: int = 1,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, num_rank_per_block)
        self.lora_block_mapping = LORA_BLOCK_MAPPING
        self.active_adapter = adapter_name

    def merge(self):
        """ 
        Merge LoRA weights with Linear weights. Not applicable to Mix-LoRA
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
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            lora_A = self.nd_lora_A(self.lora_A[self.active_adapter].T).mT
            lora_B = self.nd_lora_B(self.lora_B[self.active_adapter])
            result += (self.lora_dropout[self.active_adapter](x) @ lora_A @ lora_B) \
                      * self.scaling[self.active_adapter]

        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result

class Embedding(nn.Embedding, LoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        adapter_name: str,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoraLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim)

        self.weight.requires_grad = False

        nn.Embedding.reset_parameters(self)
        self.update_layer_embedding(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def unmerge(self, mode: bool = True):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_embedding_B[self.active_adapter] @ self.lora_embedding_A[self.active_adapter], True
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.lora_embedding_B[self.active_adapter] @ self.lora_embedding_A[self.active_adapter], True
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r[self.active.adapter] > 0 and self.merged:
                self.weight.data -= (
                    transpose(
                        self.lora_embedding_B[self.active_adapter].weight
                        @ self.lora_embedding_A[self.active_adapter].weight,
                        True,
                    )
                    * self.scaling[self.active_adapter]
                )
                self.merged = False
            return nn.Embedding.forward(self, x)

        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r[self.active_adapter] > 0:
                after_A = F.embedding(
                    x,
                    self.lora_embedding_A[self.active_adapter].T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lora_embedding_B[self.active_adapter].T) * self.scaling[self.active_adapter]
            return result
        else:
            return nn.Embedding.forward(self, x)


class Conv2d(nn.Conv2d, LoraLayer):
    # Lora implemented in a conv2d layer
    def __init__(
        self,
        adapter_name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding)
        LoraLayer.__init__(
            self,
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        nn.Conv2d.reset_parameters(self)
        self.update_layer_conv2d(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
            if self.weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                self.weight.data += (
                    self.lora_B[self.active_adapter].weight.squeeze(3).squeeze(2)
                    @ self.lora_A[self.active_adapter].weight.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3) * self.scaling[self.active_adapter]
            else:
                # conv2d 3x3
                self.weight.data += (
                    F.conv2d(
                        self.lora_A[self.active_adapter].weight.permute(1, 0, 2, 3),
                        self.lora_B[self.active_adapter].weight,
                    ).permute(1, 0, 2, 3)
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
            if self.weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                self.weight.data -= (
                    self.lora_B[self.active_adapter].weight.squeeze(3).squeeze(2)
                    @ self.lora_A[self.active_adapter].weight.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3) * self.scaling[self.active_adapter]
            else:
                # conv2d 3x3
                self.weight.data += (
                    F.conv2d(
                        self.lora_A[self.active_adapter].weight.permute(1, 0, 2, 3),
                        self.lora_B[self.active_adapter].weight,
                    ).permute(1, 0, 2, 3)
                    * self.scaling[self.active_adapter]
                )
            self.merged = False

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.lora_A.keys():
            return F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            x = x.to(self.lora_A[self.active_adapter].weight.dtype)

            result += (
                self.lora_B[self.active_adapter](
                    self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                )
                * self.scaling[self.active_adapter]
            )
        else:
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        result = result.to(previous_dtype)

        return result

#=====================================================
class mem_point:
    def __init__(self, key, value):
        self.key = key
        self.value = value
    def get_key(self):
        return self.key

    def get_value(self):
        return self.value

    def get_lora_id(self):
        return self.value


class VecDB:
    def __init__(self, grace_config):
        self.config = grace_config
        self.table = []
        self.forget_num = 0
        self.conflict_num = 0
        self.forget_keys = []

    def __len__(self):
        return len(self.table)

    def __getitem__(self, item):
        return self.table[item]

    def add_cluster(self, new_key, new_value, new_edit_label):
        new_row = {'cluster_center':None, 'radius': None, 'key_label': None, 'points':[]}

        new_row['cluster_center'] = new_key.detach()
        new_row['radius'] = torch.tensor(self.config['init_radius'], device = new_key.device).view(1)
        new_row['key_label'] = new_edit_label
        new_row['points'].append(mem_point(new_key.detach(),new_value))

        self.table.append(new_row)

    def update_cluster(self, index, new_key, new_value):
        self.table[index]['points'].append(mem_point(new_key,new_value))
        key_list = [x.get_key() for x in self.table[index]['points']]
        new_cluster_center = sum(key_list)/len(key_list)
        self.table[index]['cluster_center'] = new_cluster_center

        dists = self.euc(key_list, new_cluster_center).view(-1,1)
        largest_distance, _ = dists.max(0)
        self.table[index]['radius'] = max(largest_distance, torch.tensor(self.config['init_radius'], device = new_key.device).view(1))


    def label_match(self, edit_label, key_label):
        edit_label = edit_label.masked_fill(edit_label == -100, 0)
        key_label = key_label.masked_fill(key_label == -100, 0)
        return torch.sum(edit_label) == torch.sum(key_label)

    def split_cluster_radii_in_half(self, nearest_cluster, smallest_distance):
        self.table[nearest_cluster]['radius'] = (smallest_distance / 2) - 1e-5
        self.table[-1]['radius'] = (smallest_distance / 2) + 1e-5

        cluster_radius = self.table[nearest_cluster]['radius']
        key_list = [x.get_key() for x in self.table[nearest_cluster]['points']]
        key_list = torch.stack(key_list,dim=0)
        cluster_center = self.table[nearest_cluster]['cluster_center']
        dists = self.euc(key_list,cluster_center).view(-1,1)
        filtered_key_list = []
        for index, dist in enumerate(dists):
            if dist <= cluster_radius:
                filtered_key_list.append(self.table[nearest_cluster]['points'][index])
            else:
                self.forget_keys.append(key_list[index])
        if len(filtered_key_list) == 0:
            filtered_key_list.append(mem_point(cluster_center,NO_LORA))
        self.table[nearest_cluster]['points'] = filtered_key_list

        self.forget_num += len(key_list) - len(filtered_key_list)
        self.conflict_num += 1




    def euc(self, batch_query, key):
        if isinstance(batch_query, list):
            batch_query = torch.stack(batch_query,dim=0)
        # Euclidean distance
        if len(key.shape) < 2:
            key = key.view(1, -1)
        return torch.cdist(batch_query,key, p=2, compute_mode='donot_use_mm_for_euclid_dist')

    def search_database(self, batch_query):
        dists = []
        for x in self.table:
            dists.append(self.euc(batch_query,x['cluster_center']).view(-1,1))
        dists = torch.stack(dists).view(-1, len(batch_query))
        smallest_distance_list, nearest_cluster_list = dists.min(0)
        return smallest_distance_list,nearest_cluster_list

    def search_cluster(self, batch_query, smallest_distance_list,  nearest_cluster_list):
        lora_mapping_block = []
        for query, smallest_distance, nearest_cluster in zip(batch_query, smallest_distance_list,nearest_cluster_list):
            try:
                if smallest_distance > self.table[nearest_cluster]['radius']:
                    lora_mapping_block.append(NO_LORA)
                    continue
                # Valid Cluster
                key_list = [x.get_key() for x in self.table[nearest_cluster]['points']]
                key_list = torch.stack(key_list, dim=0)
                dists = self.euc(key_list,query).view(-1,1)
                _, nearest_key = dists.min(0)
                lora_mapping_block.append(self.table[nearest_cluster]['points'][nearest_key].get_value())
            except Exception as e:
                print(e)
                print(f'[smallest_distace]: {smallest_distance}')
                print(f"[nearest_cluster]: {self.table[nearest_cluster]['radius']}")

        return lora_mapping_block


#===========grace layer===========
class GraceLayer:
    """ Layer of Grace in "Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors" """
    def __init__(self, grace_config: dict, in_features: int, out_features: int, **kwargs):
        self.grace_config = grace_config
        self.batch_iter = None
        for k, v in grace_config.items():
            setattr(self, k, v)
        self.VecDB = VecDB(grace_config)
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs
        self.lora_block_mapping = []
        self.non_overlap_edit = 0
        self.disable_grace = False
        self.block_id = 0

    def search(self, batch_query):
        smallest_distance_list, nearest_cluster_list = self.VecDB.search_database(batch_query)
        lora_block_mapping = self.VecDB.search_cluster(batch_query, smallest_distance_list, nearest_cluster_list)
        return smallest_distance_list, nearest_cluster_list, lora_block_mapping

    def current_block(self):
        return self.block_id
            
    def init_key_value(self, batch_query):
        lora_block_mapping = []
        for index, query in enumerate(batch_query):
            new_key = query.detach()
            new_eidt_label = self.edit_label[index]
            new_value = self.current_block()
            self.VecDB.add_cluster(new_key= new_key, new_value=new_value, new_edit_label= new_eidt_label)
            lora_block_mapping.append(new_value)
            # self.block_id += 1
        self.block_id += 1
        return lora_block_mapping

    def add_cluster(self, query, label_index):
        new_key = query.detach()
        new_value = self.current_block()
        new_edit_label = self.edit_label[label_index]
        self.VecDB.add_cluster(new_key = new_key, new_value = new_value, new_edit_label=new_edit_label)
        # self.block_id += 1
    
    def update_cluster(self, index, query, value):
         self.VecDB.update_cluster(index,query.detach(), value)


class GraceLinear(nn.Linear, GraceLayer):
    def __init__(
            self,
            adapter_name: str,
            in_features: int,
            out_features: int,
            grace_config: dict,
            fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features,**kwargs)
        GraceLayer.__init__(self,grace_config=grace_config, in_features=in_features, out_features=out_features)
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def forward(self, x: torch.Tensor):
        global LORA_BLOCK_MAPPING
        layer_out = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_grace:
            return layer_out

        '''Search Vector Database
        '''
        key_id = min(self.key_id, layer_out.shape[1] - 1)
        batch_query = layer_out[:, key_id]
        smallest_distance_list, nearest_cluster_list, lora_block_mapping = None, None, [NO_LORA] * layer_out.shape[0]
        if len(self.VecDB) != 0:
            smallest_distance_list, nearest_cluster_list, lora_block_mapping = self.search(batch_query)

        if not self.training:
            self.lora_block_mapping = LORA_BLOCK_MAPPING = lora_block_mapping
            return layer_out



        '''Only update the vector database once per batch
        '''
        if len(self.VecDB) == 0:
            self.lora_block_mapping = self.init_key_value(batch_query)
        elif self.batch_iter == 0:
            self.lora_block_mapping = [self.block_id] * layer_out.shape[0]
            for index, query in enumerate(batch_query):
                row = self.VecDB[nearest_cluster_list[index]]
                if smallest_distance_list[index] > row['radius'] + self.init_radius:
                    self.add_cluster(query,label_index=index)
                elif self.VecDB.label_match(self.edit_label[index], row['key_label']):
                    print(f'The {index}th query is close to a previous edit, the labels are the same')
                    self.update_cluster(nearest_cluster_list[index],query,self.block_id)
                else:
                    print(f'The {index}th query is close to a previous edit, but the labels are different')
                    self.add_cluster(query,label_index = index)
                    self.VecDB.split_cluster_radii_in_half(nearest_cluster_list[index], smallest_distance_list[index])
            self.block_id += 1
        else:
            pass

        LORA_BLOCK_MAPPING = self.lora_block_mapping
        return layer_out


class dynamic(nn.Module):
    def __init__(
            self,
            maximum_rank: int = 1,
            num_rank_per_block: int = 1
    ):
        assert maximum_rank % num_rank_per_block == 0, \
            "Maximum_rank % num_rank_per_block == 0 should be True"
        super(dynamic, self).__init__()
        self.maximum_rank = maximum_rank
        self.num_rank_per_block = num_rank_per_block
        self.maximum_block = maximum_rank // num_rank_per_block
        self.current_block = 0
    def get_block_dimension(self):
        return self.maximum_block

    def get_block(self):
        return self.current_block

    def set_block(self, block):
        self.current_block = max(0, min(block, self.get_block()))

    def block_rank_mapping(self, block_id):
        start = block_id * self.num_rank_per_block
        end = start + self.num_rank_per_block
        return start, end

    def update_dynamic(self, maximum_rank, num_rank_per_block):
        assert maximum_rank % num_rank_per_block == 0, \
            "Maximum_rank % num_rank_per_block == 0 should be True"
        self.maximum_rank = maximum_rank
        self.num_rank_per_block = num_rank_per_block
        self.maximum_block = maximum_rank // num_rank_per_block
        self.current_block = 0

    def forward(self, inputs):
        block_list = []
        assert len(LORA_BLOCK_MAPPING) != 0, "No element in LORA_BLOCK_MAPPING"
        for block_id in LORA_BLOCK_MAPPING:
            if block_id == NO_LORA:
                zero_tensor = torch.zeros((self.num_rank_per_block,inputs.shape[1]),device=inputs.device)
                block_list.append(zero_tensor)
            else:
                start, end = self.block_rank_mapping(block_id)
                block_list.append(inputs[start:end])
        result = torch.stack(block_list,0)

        return result * math.sqrt(self.maximum_rank / self.num_rank_per_block)


class Gate(nn.Module):
    """ https://github.com/laekov/fastmoe/blob/master/fmoe/gates/naive_gate.py#L11 """
    def __init__(self, d_model, num_expert, top_k=2, gate_bias=True) -> None:
        super().__init__()
        self.num_expert = num_expert
        self.loss = None

        self.gate = nn.Linear(d_model, self.num_expert, bias=gate_bias)
        self.top_k = min(top_k, num_expert)
        self.preset_target = False # use preset LoRA allocation
        self.target_set_mode = None
        self.editing = False

    def forward(self, inp, return_all_scores=False):
        """
        Router for Mixture-of-LoRA
        """
        gate = self.gate(inp)
        gate = F.softmax(gate, dim=-1)  # 각 expert의 확률
         
        if self.preset_target and self.target_set_mode == 'hard':
            gate_top_k_idx = self.preset_gate_top_k_idx
            gate_top_k_val = torch.gather(input=gate, index=gate_top_k_idx, dim=-1)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        if self.preset_target and self.target_set_mode == 'soft':
            target_top_k_idx = self.preset_gate_top_k_idx # 미리 지정된 expert 인덱스 집합(= allocation)

        gate_score = gate_top_k_val 

        """ Compute Guided Loss """
        if self.editing:
            init_expert_probs = torch.gather(input=gate, index=target_top_k_idx, dim=-1) # 각 expert에 대해 확률 분포를 뽑고, 그중 미리 지정된 target expert의 확률만 선택
            guided_loss = -torch.log(init_expert_probs).sum() / gate.size(0) # @ 수식 : guided_loss 사용
            self.set_loss(guided_loss)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def set_target(self, target, set_mode):
        self.preset_target = True
        self.target_set_mode = set_mode
        self.preset_gate_top_k_idx = torch.stack([torch.where(t)[0] for t in target]).to(self.gate.weight.device) # [batch-size, top-k]
        
    def set_loss(self, loss):
        self.loss = loss

    def get_loss(self, clear=True):
        loss = self.loss
        if clear:
            self.loss = None
        return loss

    @property
    def has_loss(self):
        return self.loss is not None

    def set_result(self, gate_top_k_idx):
        k_hot_output = torch.zeros((gate_top_k_idx.shape[0], self.num_expert), dtype=torch.int32)
        k_hot_output.scatter_(1, gate_top_k_idx, 1)
        self.gate_score = k_hot_output


class _LoraExpert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = nn.Linear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = nn.Linear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x


#===========================bnb config===========================
if is_bnb_available():
    class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                return result
            elif self.r[self.active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                        self.lora_B[self.active_adapter](
                            self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                        ).to(expected_dtype)
                        * self.scaling[self.active_adapter]
                    )
                else:
                    output = (
                        self.lora_B[self.active_adapter](
                            self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                        )
                        * self.scaling[self.active_adapter]
                    )
                result += output
            return result

    if is_bnb_4bit_available():

        class Linear4bit(bnb.nn.Linear4bit, LoraLayer):
            # Lora implemented in a dense layer
            def __init__(
                self,
                adapter_name,
                in_features,
                out_features,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.0,
                **kwargs,
            ):
                bnb.nn.Linear4bit.__init__(
                    self,
                    in_features,
                    out_features,
                    bias=kwargs.get("bias", True),
                    compute_dtype=kwargs.get("compute_dtype", torch.float32),
                    compress_statistics=kwargs.get("compress_statistics", True),
                    quant_type=kwargs.get("quant_type", "nf4"),
                )
                LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False

                init_lora_weights = kwargs.pop("init_lora_weights", True)
                self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
                self.active_adapter = adapter_name

            def forward(self, x: torch.Tensor):
                result = super().forward(x)

                if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                    return result
                elif self.r[self.active_adapter] > 0:
                    result = result.clone()
                    if not torch.is_autocast_enabled():
                        expected_dtype = result.dtype
                        x = x.to(self.lora_A[self.active_adapter].weight.dtype)
                        output = (
                            self.lora_B[self.active_adapter](
                                self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                            ).to(expected_dtype)
                            * self.scaling[self.active_adapter]
                        )
                    else:
                        output = (
                            self.lora_B[self.active_adapter](
                                self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                            )
                            * self.scaling[self.active_adapter]
                        )
                    result += output
                return result


