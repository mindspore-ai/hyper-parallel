# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""simple mlp"""
import torch
from torch import nn
import torch.distributed as dist
# pylint: disable=W0611
import torch_npu  # 昇腾NPU核心适配
from hyper_parallel import DTensor


class MLP(nn.Module):
    """MLP net."""
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(in_size, out_size, dtype=torch.float32).npu())
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        if isinstance(x, DTensor):
            x = x.reduce_partial()
        x = self.relu(x)
        return x


class SimpleMLP(nn.Module):
    """Simple MLP net."""
    def __init__(self, num_layers, in_size, out_size):
        super().__init__()
        self.mlp_layers = nn.ModuleDict()
        for mlp_id in range(num_layers):
            self.mlp_layers[str(mlp_id)] = MLP(in_size, out_size)

    def forward(self, x):
        for mlp in self.mlp_layers.values():
            x = mlp(x)
        return x


def model_split_manual(model, stage_index, num_layers):
    """pipeline parallel split."""
    for i in range(num_layers):
        if i == stage_index:
            continue
        del model.mlp_layers[str(i)]


def run_standalone(dim, model):
    """run standalone."""
    steps = 1
    data = torch.ones(dim, 16, dtype=torch.float32).npu()
    for _ in range(steps):
        loss = model(data)
        sens = torch.ones(loss.shape, dtype=loss.dtype, device="npu")
        loss.backward(gradient=sens)
    return loss


def init_hccl():
    """init hccl and set device"""
    dist.init_process_group("hccl")
    rank_id = dist.get_rank()
    torch.npu.set_device(rank_id)


def get_stage_index(stage_num):
    """get stage index"""
    rank_id = dist.get_rank()
    device_num = dist.get_world_size()
    device_num_per_stage = device_num // stage_num
    stage_index = rank_id // device_num_per_stage
    return stage_index


def get_rank_list(stage_num):
    """get rank list"""
    rank_id = dist.get_rank()
    device_num = dist.get_world_size()
    device_num_per_stage = device_num // stage_num
    stage_index = rank_id // device_num_per_stage
    rank_list = [device_num_per_stage * stage_index + i for i in range(device_num_per_stage)]
    return rank_list
