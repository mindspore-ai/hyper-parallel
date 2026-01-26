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
"""vpp schedule"""

import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, mint
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common.initializer import initializer
from hyper_parallel import PipelineStage, ScheduleInterleaved1F1B
from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.hsdp import hsdp
from hyper_parallel import shard
from hyper_parallel.core.placement_types import Shard, Replicate

class MLP(nn.Cell):
    """MLP net."""
    def __init__(self, in_size, out_size, device_mesh=None, w_placements=None):
        super().__init__()
        if device_mesh is None or w_placements is None:
            self.weight = ms.Parameter(
                initializer("ones", [in_size, out_size],ms.float32),
                name="weight")
        else:
            self.weight = ms.Parameter(
                DTensor.from_local(initializer("ones", [in_size, out_size],ms.float32),
                                   device_mesh, w_placements), name="weight")
        self.relu = mint.nn.ReLU()

    def construct(self, x):
        x = mint.matmul(x, self.weight)
        x = self.relu(x)
        return x

class SimpleMLP(nn.Cell):
    """Simple MLP net."""

    def __init__(self, num_layers, in_size, out_size, device_mesh=None, w_placements=None):
        super().__init__()
        self.mlp_layers = nn.CellDict()
        for mlp_id in range(num_layers):
            self.mlp_layers[str(mlp_id)] = MLP(in_size, out_size, device_mesh=device_mesh, w_placements=w_placements)

    def construct(self, x):
        for mlp in self.mlp_layers.values():
            x = mlp(x)
        return x

def run_standalone(dim):
    """run standalone."""
    model = SimpleMLP(8, 16, 16)

    def forward_fn(data):
        logits = model(data)
        return logits

    grad_fn = ms.value_and_grad(forward_fn, None, model.trainable_params(), has_aux=False)

    steps = 1
    ret_loss = None
    ret_grads = None
    data = Tensor(np.ones((dim, 16)), dtype=ms.float32)
    for _ in range(steps):
        ret_loss, ret_grads = grad_fn(data)
    return ret_loss, ret_grads


def model0_split_manual(model, stage_index, num_layers):
    """pipeline parallel split."""
    for i in range(num_layers):
        if i == stage_index:
            continue
        del model.mlp_layers[str(i)]

def model1_split_manual(model, stage_index, num_layers):
    """pipeline parallel split."""
    for i in range(num_layers):
        if i == stage_index:
            continue
        del model.mlp_layers[str(i)]

def run_parallel(micro_batch_num):
    """
    Feature: HSDP + SHARD + PP.
    Description: Test simple mlp net.
    Expectation: Run success.
    """
    ms.set_seed(0)
    init("hccl")

    # pp config
    num_stages = 4
    rank_id = get_rank()
    device_num = get_group_size()
    device_num_per_stage = device_num // num_stages
    stage_index = rank_id // device_num_per_stage
    local_batch_size = micro_batch_num
    local_hidden_size = 16

    model0 = SimpleMLP(8, 16, 16)
    model1 = SimpleMLP(8, 16, 16)
    model0_split_manual(model0, stage_index, 8)
    model1_split_manual(model1, stage_index + 4, 8)
    # shard config
    dp = 2
    mp = 1
    mesh = init_device_mesh(mesh_shape=(dp, mp), alias_name=("dp", "mp"))

    # Define placements using Placement format
    in_placements = (Shard(0), Replicate())
    w_placements = (Replicate(), Replicate())
    out_placements = (Shard(0), Replicate())

    model_stra = {"forward": {"input": in_placements, "output": out_placements},
                  "parameter": {"weight": w_placements}}
    shard(model0, device_mesh=mesh, sharding_plan=model_stra)
    shard(model1, device_mesh=mesh, sharding_plan=model_stra)

    model0 = hsdp(model0, 2, 0, "level1", True)
    model1 = hsdp(model1, 2, 0, "level1", True)

    # pp stage
    pipeline_stage0 = PipelineStage(model0, stage_index , num_stages + 4)
    pipeline_stage1 = PipelineStage(model1,stage_index + 4,num_stages + 4)
    schedule = ScheduleInterleaved1F1B([pipeline_stage0, pipeline_stage1], micro_batch_num)

    # input
    x_placements = (Shard(0), Shard(1))
    local_batch_size = 8
    local_hidden_size = 16
    x = DTensor.from_local(Tensor(np.ones((local_batch_size, local_hidden_size)), dtype=ms.float32),
                           mesh, x_placements)

    # train config
    epochs = 1
    for _ in range(epochs):
        model0.zero_grads()
        model1.zero_grads()
        if stage_index == 0:
            loss = schedule.run(x)
        else:
            loss = schedule.run()
    return loss, model0


def vpp(micro):
    """
    Feature: HSDP + SHARD + PP.
    Description: Test simple mlp net.
    Expectation: Run success.
    """
    standalone_loss, standalone_grads = run_standalone(micro*2)
    pp_loss, model0 = run_parallel(micro)
    num_stages = 4
    rank_id = get_rank()
    device_num = get_group_size()
    device_num_per_stage = device_num // num_stages
    stage_index = rank_id // device_num_per_stage
    if stage_index == 3:
        assert np.allclose(standalone_loss[:1, :].asnumpy(), pp_loss[0].asnumpy())
    expect_grad = None
    if stage_index in [1, 2]:
        expect_grad = standalone_grads[stage_index].asnumpy()
    else:
        expect_grad = standalone_grads[0].asnumpy()
    assert np.allclose(expect_grad[:8, :], model0.trainable_params()[0].grad.asnumpy())


def test():
    micro = 8
    vpp(micro)
