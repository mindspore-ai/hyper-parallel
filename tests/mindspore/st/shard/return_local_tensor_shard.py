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
"""return local tensor shard"""

import time
import numpy as np
import mindspore as ms
from mindspore._c_expression import NoFallbackGuard
import mindspore.communication.management as D
from mindspore import nn, Tensor
from mindspore.nn.utils import no_init_parameters
from mindspore.common.initializer import initializer
from hyper_parallel import hsdp, init_parameters, shard_module, parallelize_value_and_grad, DTensor, init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.sharding_plan import ShardingPlan
from tests.mindspore.st.shard.utils import create_dtensor

learning_rate = 0.01
epochs = 2


class SimpleModelCheck(nn.Cell):
    """simple model with check"""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = ms.Parameter(initializer("ones", [input_size, output_size], ms.float32), name='weight')
        self.relu = ms.mint.nn.ReLU()

    def construct(self, x):
        x = ms.mint.matmul(x, self.weight)
        x = self.relu(x)
        assert not isinstance(x, DTensor), "Output of relu should be a local Tensor, but got DTensor"
        assert isinstance(x, Tensor), f"Output of relu should be a local Tensor, but got type {type(x)}"
        x = ms.mint.sum(x)
        return x


def run_model(x, model, parallel=False):
    """rum model"""

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
        print(f"[standalone] Epoch: {epoch + 1}/{epochs}, Loss: {loss_value}, Time: {end - start}")

    return ret_loss, ret_grads


def base_case(dp, mp, hsdp_shard_size):
    """base case"""
    D.init()

    input_size = 32
    output_size = 2
    batch_size = 4

    local_batch_size = batch_size // dp
    local_input_size = input_size // mp
    local_x = np.ones([local_batch_size, local_input_size]).astype(np.float32)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(dp, mp),
        mesh_dim_names=("dp", "tp")
    )

    x_placements = (Shard(0), Shard(1))
    w_placements = (Replicate(), Shard(0))
    out_placements = (Replicate(), Replicate())
    relu_input_placements = (Shard(0), Replicate())
    relu_output_placements = (Shard(0), Replicate())

    # step 1: define network with no init parameters
    with no_init_parameters():
        model = SimpleModelCheck(input_size, output_size)

    # step 2: shard
    # Add return_local_tensor=["relu"] to the plan
    # step 2: shard
    model_sharding_plan = ShardingPlan(
        plan={"weight": w_placements},
        input_plan={"input": x_placements},
        output_plan={"output": out_placements},
        return_local_tensor=["relu"]
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
    run_model(x, model, parallel=True)


def test_return_local_tensor_shard():
    '''
    Feature: return_local_tensor functionality.
    Description: Test return_local_tensor with SimpleModelCheck.
    Expectation: Run success.
    '''
    base_case(dp=4, mp=2, hsdp_shard_size=4)
