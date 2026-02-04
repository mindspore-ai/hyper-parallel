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
"""base shard"""

import os
import time

import mindspore as ms
from mindspore._c_expression import NoFallbackGuard
import mindspore.communication.management as D
from mindspore import nn
from mindspore.nn.utils import no_init_parameters
from mindspore.common.initializer import initializer
from mindspore.communication import get_rank

from hyper_parallel import shard_module, parallelize_value_and_grad, init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.sharding_plan import ShardingPlan
from hyper_parallel.core.checkpoint.layout import get_current_layout, save_layout, load_layout, get_global_layout
from hyper_parallel.core.checkpoint.loader import load_checkpoint
from hyper_parallel.core.checkpoint.saver import save_checkpoint

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
        print(f"[standalone] Epoch: {epoch+1}/{epochs}, Loss: {loss_value}, Time: {end - start}")

    return ret_loss, ret_grads


def base_case(dp, mp):
    """base case"""
    D.init()

    # standalone
    input_size = 32
    output_size = 2

    # Create DeviceMesh
    mesh = init_device_mesh(mesh_shape=(dp, mp), alias_name=("dp", "mp"))

    # Define placements using Placement format
    x_placements = (Shard(0), Shard(1))
    w_placements = (Replicate(), Shard(0))
    out_placements = (Replicate(), Replicate())
    relu_input_placements = (Shard(0), Replicate())
    relu_output_placements = (Shard(0), Replicate())

    # step 1: define network with no init parameters
    with no_init_parameters():
        model = SimpleModel(input_size, output_size)

    # step 2: shard
    model_stra = ShardingPlan(
        plan = {"weight": w_placements},
        input_plan={"input": x_placements},
        output_plan={"output": out_placements},
    )
    shard_module(model, device_mesh=mesh, sharding_plan=model_stra)

    model_relu_stra = ShardingPlan(
        input_plan={"input": relu_input_placements},
        output_plan={"output": relu_output_placements},
    )
    shard_module(model.relu, device_mesh=mesh, sharding_plan=model_relu_stra)

    # step 3: save layout
    layout_dict = get_current_layout(model)
    rank_id = get_rank()
    file_name = f"test_{rank_id}.layout"
    save_layout(layout_dict, file_name)
    assert os.path.isfile(file_name)
    layout_dict = load_layout(file_name)
    assert isinstance(layout_dict, dict)


def save_load_checkpoint(dp: int, mp: int) -> None:
    """
    Test using saver to save checkpoint to safetensors file, and then using loader to load checkpoint from this
    safetensors file.

    Args:
        dp, mp: Mesh shape coordinate.
    """
    D.init()

    # standalone
    input_size = 32
    output_size = 2

    # Create DeviceMesh
    mesh = init_device_mesh(mesh_shape=(dp, mp), alias_name=("dp", "mp"))

    # Define placements using Placement format
    x_placements = (Shard(0), Shard(1))
    w_placements = (Replicate(), Shard(0))
    out_placements = (Replicate(), Replicate())
    relu_input_placements = (Shard(0), Replicate())
    relu_output_placements = (Shard(0), Replicate())

    # step 1: define network with no init parameters
    with no_init_parameters():
        model = SimpleModel(input_size, output_size)

    # step 2: shard
    model_stra = ShardingPlan(
        plan = {"weight": w_placements},
        input_plan={"input": x_placements},
        output_plan={"output": out_placements},
    )
    shard_module(model, device_mesh=mesh, sharding_plan=model_stra)

    model_relu_stra = ShardingPlan(
        input_plan={"input": relu_input_placements},
        output_plan={"output": relu_output_placements},
    )
    shard_module(model.relu, device_mesh=mesh, sharding_plan=model_relu_stra)

    # step 3: save checkpoint
    file_path = "tmp1.safetensors"
    save_checkpoint(model, file_path)

    # step 4: load checkpoint
    param_dict = load_checkpoint(file_path)
    os.remove(file_path)
    assert isinstance(param_dict, dict)


def test_base_layout():
    """
    Feature: test layout save and load.
    Description: Test base layout save.
    Expectation: Run success.
    """
    base_case(dp=4, mp=2)


def test_saver_loader():
    """
    Feature: Test checkpoint saver and loader.
    Description: Test when a simple model sharded by dp and mp, use saver and loader to save checkpoint to safetensors
    file, and use loader to load checkpoint from this safetensors file.
    Expectation: Run success.
    """
    save_load_checkpoint(dp=4, mp=2)


def base_global_layout(dp: int, mp: int):
    """ base global layout """
    D.init()

    # standalone
    input_size = 32
    output_size = 4

    # Create DeviceMesh
    mesh = init_device_mesh(mesh_shape=(dp, mp), alias_name=("dp", "mp"))

    # Define placements using Placement format
    x_placements = (Shard(0), Shard(1))
    w_placements = (Replicate(), Shard(0))
    out_placements = (Replicate(), Replicate())
    relu_input_placements = (Shard(0), Replicate())
    relu_output_placements = (Shard(0), Replicate())

    # step 1: define network with no init parameters
    with no_init_parameters():
        model = SimpleModel(input_size, output_size)

    # step 2: shard
    model_stra = ShardingPlan(
        plan = {"weight": w_placements},
        input_plan={"input": x_placements},
        output_plan={"output": out_placements},
    )
    shard_module(model, device_mesh=mesh, sharding_plan=model_stra)

    model_relu_stra = ShardingPlan(
        input_plan={"input": relu_input_placements},
        output_plan={"output": relu_output_placements},
    )
    shard_module(model.relu, device_mesh=mesh, sharding_plan=model_relu_stra)

    # step 3: get global layout
    global_layout = get_global_layout(model)
    assert isinstance(global_layout, dict)


def test_get_global_layout():
    """
    Feature: Test get global layout on all ranks.
    Description: Test when a simple model sharded by dp and mp, gather global layout on all ranks.
    Expectation: Run success.
    """
    base_global_layout(dp=4, mp=2)
