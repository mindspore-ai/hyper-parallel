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
"""base shard with Alias-style alias placements"""

import time
import numpy as np
import mindspore as ms
from mindspore._c_expression import NoFallbackGuard
import mindspore.communication.management as D
from mindspore import nn, Tensor
from mindspore.nn.utils import no_init_parameters
from mindspore.common.initializer import initializer
from hyper_parallel import init_device_mesh, hsdp, init_parameters, shard_module, parallelize_value_and_grad
from hyper_parallel.core.shard.sharding_plan import ShardingPlan
from tests.mindspore.st.shard.utils import create_dtensor

D.init()

learning_rate = 0.01
epochs = 2


class SimpleModel(nn.Cell):
    """simple model"""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = ms.Parameter(initializer("ones", [input_size, output_size], ms.float32), name='weight')
        self.relu = ms.mint.nn.ReLU()

    def construct(self, x):
        x = ms.mint.matmul(x, self.weight)
        x = self.relu(x)
        x = ms.mint.sum(x)
        return x


def run_model(x, model, parallel=False):
    """run model"""
    def forward_fn(data):
        logits = model(data)
        return logits

    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)
    if parallel is False:
        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)
    else:
        grad_fn = parallelize_value_and_grad(forward_fn, optimizer.parameters)

    ret_loss = None
    ret_grads = None
    for epoch in range(epochs):
        start = time.time()
        (loss_value, grads) = grad_fn(x)
        with NoFallbackGuard():
            optimizer(grads)
        end = time.time()
        ret_loss = loss_value
        ret_grads = grads
        print(f"[standalone] Epoch: {epoch+1}/{epochs}, Loss: {loss_value}, Time: {end - start}")

    return ret_loss, ret_grads


def base_case_single_axis(dp, mp, hsdp_shard_size):
    """
    Test Alias-style single-axis alias placements (e.g., ["dp", "None"]).
    Equivalent to Placement-style [Shard(0), Replicate()].
    """
    # standalone
    input_size = 32
    output_size = 2
    batch_size = 4

    standalone_x = Tensor(np.ones([batch_size, input_size]).astype(np.float32), dtype=ms.float32)
    standalone_model = SimpleModel(input_size, output_size)
    standalone_loss, standalone_grads = run_model(standalone_x, standalone_model)

    # parallel
    local_batch_size = batch_size // dp
    local_input_size = input_size // mp
    local_output_size = output_size
    local_x = np.ones([local_batch_size, local_input_size]).astype(np.float32)

    # Create DeviceMesh
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(dp, mp),
        mesh_dim_names=("dp", "tp")
    )

    # Define placements using Alias-style single-axis alias
    x_placements = ("dp", "tp")
    w_placements = ("tp", "None")
    out_placements = ("None", "None")
    relu_input_placements = ("dp", "None")
    relu_output_placements = ("dp", "None")

    # step 1: define network with no init parameters
    with no_init_parameters():
        model = SimpleModel(input_size, output_size)

    # step 2: shard
    model_sharding_plan = ShardingPlan(
        plan={"weight": w_placements},
        input_plan={"input": x_placements},
        output_plan={"output": out_placements},
    )
    shard_module(model, device_mesh=mesh, sharding_plan=model_sharding_plan)

    model_relu_sharding_plan = ShardingPlan(
        input_plan={"input": relu_input_placements},
        output_plan={"output": relu_output_placements}
    )
    shard_module(model.relu, device_mesh=mesh, sharding_plan=model_relu_sharding_plan)

    # step 3: hsdp
    model = hsdp(model, shard_size=hsdp_shard_size, threshold=0)

    # step 4: init parameters
    model = init_parameters(model)

    x = create_dtensor(local_x, mesh, x_placements)
    parallel_loss, parallel_grads = run_model(x, model, parallel=True)

    # compare loss
    assert np.allclose(standalone_loss.asnumpy(), parallel_loss.asnumpy(), 0.001, 0.001)

    # compare grad
    if hsdp_shard_size < 0:
        hsdp_shard_size = dp

    standalone_grad = standalone_grads[0].asnumpy()
    standalone_grad_slice = standalone_grad[:local_input_size // hsdp_shard_size, :local_output_size]
    parallel_grad = parallel_grads[0].asnumpy()
    assert np.allclose(standalone_grad_slice, parallel_grad, 0.001, 0.001)


def base_case_multi_axis(dp, mp, hsdp_shard_size):
    """
    Test Alias-style multi-axis alias placements (e.g., [("dp", "tp"), "None"]).
    One tensor dimension is sharded across multiple mesh axes.
    """
    # standalone
    input_size = 32
    output_size = 2
    batch_size = dp * mp  # must be divisible by total shards (dp * mp)

    standalone_x = Tensor(np.ones([batch_size, input_size]).astype(np.float32), dtype=ms.float32)
    standalone_model = SimpleModel(input_size, output_size)
    standalone_loss, standalone_grads = run_model(standalone_x, standalone_model)

    # parallel with multi-axis: tensor dim 0 sharded across both dp and tp
    total_shards_dim0 = dp * mp  # dim 0 is sharded across dp and tp
    local_batch_size = batch_size // total_shards_dim0
    local_x = np.ones([local_batch_size, input_size]).astype(np.float32)

    # Create DeviceMesh
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(dp, mp),
        mesh_dim_names=("dp", "tp")
    )

    # Define placements using Alias-style multi-axis alias
    # x: dim 0 sharded across ("dp", "tp"), dim 1 replicated
    x_placements = (("dp", "tp"), "None")
    # weight: dim 0 replicated, dim 1 replicated
    w_placements = ("None", "None")
    out_placements = ("None",)
    relu_input_placements = (("dp", "tp"), "None")
    relu_output_placements = (("dp", "tp"), "None")

    # step 1: define network with no init parameters
    with no_init_parameters():
        model = SimpleModel(input_size, output_size)

    # step 2: shard
    model_sharding_plan = ShardingPlan(
        plan={"weight": w_placements},
        input_plan={"input": x_placements},
        output_plan={"output": out_placements},
    )
    shard_module(model, device_mesh=mesh, sharding_plan=model_sharding_plan)

    model_relu_sharding_plan = ShardingPlan(
        input_plan={"input": relu_input_placements},
        output_plan={"output": relu_output_placements}
    )
    shard_module(model.relu, device_mesh=mesh, sharding_plan=model_relu_sharding_plan)

    # step 3: hsdp
    model = hsdp(model, shard_size=hsdp_shard_size, threshold=0)

    # step 4: init parameters
    model = init_parameters(model)

    x = create_dtensor(local_x, mesh, x_placements)
    parallel_loss, parallel_grads = run_model(x, model, parallel=True)

    # compare loss
    assert np.allclose(standalone_loss.asnumpy(), parallel_loss.asnumpy(), 0.001, 0.001)

    # compare grad
    if hsdp_shard_size < 0:
        hsdp_shard_size = dp

    # weight is fully replicated ("None", "None"), HSDP shards it by hsdp_shard_size
    standalone_grad = standalone_grads[0].asnumpy()
    standalone_grad_slice = standalone_grad[:input_size // hsdp_shard_size, :output_size]
    parallel_grad = parallel_grads[0].asnumpy()
    assert np.allclose(standalone_grad_slice, parallel_grad, 0.001, 0.001)


def test_base_shard_single_axis_alias():
    '''
    Feature: Alias-style single-axis alias placements with shard + hsdp + init param.
    Description: Test that single-axis alias placements ("dp", "tp", "None") produce
        the same results as equivalent Placement-style placements.
    Expectation: Run success, loss and grad match standalone.
    '''
    base_case_single_axis(dp=4, mp=2, hsdp_shard_size=4)


def test_base_shard_multi_axis_alias():
    '''
    Feature: Alias-style multi-axis alias placements with shard + hsdp + init param.
    Description: Test that multi-axis alias placements (("dp", "tp"), "None") correctly
        shard a tensor dimension across multiple mesh axes.
    Expectation: Run success, loss and grad match standalone.
    '''
    base_case_multi_axis(dp=4, mp=2, hsdp_shard_size=4)
