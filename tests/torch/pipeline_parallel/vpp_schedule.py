# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test vpp"""
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
# pylint: disable=W0611
import torch_npu  # 昇腾NPU核心适配
from hyper_parallel.platform.torch.pipeline_parallel.stage import PipelineStage
from hyper_parallel.platform.torch.pipeline_parallel.schedule import ScheduleInterleaved1F1B


class MLP(nn.Module):
    """MLP net."""
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(in_size, out_size, dtype=torch.float32).npu())
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.matmul(x, self.weight)
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


def run_standalone(dim):
    """run standalone."""
    model = SimpleMLP(8, 16, 16).npu()

    steps = 1
    data = torch.ones(dim, 16, dtype=torch.float32).npu()
    for _ in range(steps):
        loss = model(data)
        sens = torch.ones(loss.shape, dtype=loss.dtype, device="npu")
        loss.backward(gradient=sens)
    return loss, model


def model_split_manual(model, stage_index, num_layers):
    """pipeline parallel split."""
    for i in range(num_layers):
        if i == stage_index:
            continue
        del model.mlp_layers[str(i)]


def run_parallel(micro_batch_num):
    """
    Feature: PipelineParallel.
    Description: Test simple mlp net.
    Expectation: Run success.
    """
    dist.init_process_group("hccl")
    rank_id = dist.get_rank()
    torch.npu.set_device(rank_id)

    # pp config
    num_stages = 4
    device_num = dist.get_world_size()
    device_num_per_stage = device_num // num_stages
    stage_index = rank_id // device_num_per_stage
    local_batch_size = micro_batch_num
    local_hidden_size = 16

    model0 = SimpleMLP(8, 16, 16)
    model1 = SimpleMLP(8, 16, 16)
    model_split_manual(model0, stage_index, 8)
    model_split_manual(model1, stage_index + 4, 8)

    # pp stage
    device = torch.device("npu")
    pipeline_stage0 = PipelineStage(model0, stage_index , num_stages + 4, device)
    pipeline_stage1 = PipelineStage(model1, stage_index + 4, num_stages + 4, device)
    schedule = ScheduleInterleaved1F1B([pipeline_stage0, pipeline_stage1], micro_batch_num)

    # input
    local_batch_size = 8
    local_hidden_size = 16
    x = torch.ones(local_batch_size, local_hidden_size, dtype=torch.float32).npu()

    # train config
    epochs = 1
    for _ in range(epochs):
        if stage_index == 0:
            loss = schedule.run(x)
        else:
            loss = schedule.run()
    return loss, (model0, model1)


def test_vpp():
    """
    Feature: VPP.
    Description: Test simple mlp net + vpp.
    Expectation: Run success.
    """
    micro = 8
    standalone_loss, standalone_model = run_standalone(micro)
    pp_loss, pp_model = run_parallel(micro)
    num_stages = 4
    rank_id = dist.get_rank()
    device_num = dist.get_world_size()
    device_num_per_stage = device_num // num_stages
    stage_index = rank_id // device_num_per_stage
    if stage_index == 3:
        assert np.allclose(standalone_loss.cpu().detach().numpy(), pp_loss[0].cpu().detach().numpy())
    np.allclose(standalone_model.mlp_layers[str(stage_index)].weight.cpu().detach().numpy(),
               pp_model[0].mlp_layers[str(stage_index)].weight.cpu().detach().numpy())
    np.allclose(standalone_model.mlp_layers[str(stage_index+4)].weight.cpu().detach().numpy(),
               pp_model[1].mlp_layers[str(stage_index+4)].weight.cpu().detach().numpy())


if __name__ == "__main__":
    test_vpp()
