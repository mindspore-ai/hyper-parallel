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
"""test st for parallel transpose"""
import time
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor
from hyper_parallel import hsdp, shard_module, DTensor, init_device_mesh
from hyper_parallel.core.placement_types import Shard
from hyper_parallel.core.shard.sharding_plan import ShardingPlan

learning_rate = 0.01
epochs = 2


class SimpleModel(nn.Cell):
    def __init__(self):
        super().__init__()
        self.transpose = ms.ops.Transpose()

    def construct(self, x):
        x = self.transpose(x, (1, 0, 2))
        return x


def create_dtensor(data, device_mesh, placements):
    """create_dtensor"""
    return DTensor.from_local(Tensor(data), device_mesh, placements)


def create_tensor(data):
    return Tensor(data, dtype=ms.float32)


def run_standalone(x):
    """run SimpleModel with input x in standalone mode."""
    model = SimpleModel()

    def forward_fn(data):
        logits = model(data)
        return logits

    x = create_tensor(x)

    ret_output = None
    for epoch in range(epochs):
        start = time.time()
        output = forward_fn(x)
        end = time.time()
        ret_output = output
        print(f"[standalone] Epoch: {epoch + 1}/{epochs}, Time: {end - start}")

    return ret_output


def run_parallel(local_x, device_mesh, x_placements):
    """run SimpleModel with input x in parallel mode."""
    model = SimpleModel()

    transpose_stra = ShardingPlan(
        input_plan={"input": x_placements},
    )
    shard_module(model.transpose, device_mesh=device_mesh, sharding_plan=transpose_stra)

    model = hsdp(model, shard_size=1, threshold=0)

    def forward_fn(data):
        logits = model(data)
        return logits

    x = create_dtensor(local_x, device_mesh, x_placements)

    ret_output = None
    for epoch in range(epochs):
        start = time.time()
        output = forward_fn(x)
        end = time.time()
        ret_output = output
        print(f"[parallel] Epoch: {epoch + 1}/{epochs}, Time: {end - start}")

    return ret_output


def base_case(dp, mp):
    """run case."""
    D.init()

    # standalone
    batch_size = 4
    seq_length = 8
    hidden_size = 16

    x = np.ones([batch_size, seq_length, hidden_size]).astype(np.float32)
    standalone_output = run_standalone(x)

    # parallel
    local_batch_size = batch_size // dp
    local_seq_length = seq_length // mp
    local_x = np.ones([local_batch_size, local_seq_length, hidden_size]).astype(np.float32)

    # Create DeviceMesh
    mesh = init_device_mesh(device_type="npu", mesh_shape=(dp, mp), mesh_dim_names=("dp", "mp"))

    # Define placements using Placement format
    x_placements = (Shard(0), Shard(1))
    parallel_output = run_parallel(local_x, mesh, x_placements)

    # compare output
    assert np.allclose(standalone_output.asnumpy()[::dp, ::mp, :],
                       parallel_output.asnumpy(), 0.001, 0.001)


def test_loss_repeat_mean_0():
    '''
    Feature: Transpose parallel op.
    Description: Test Transpose parallel op.
    Expectation: Run success.
    '''
    base_case(dp=2, mp=2)
