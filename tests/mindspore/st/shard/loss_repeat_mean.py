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
"""loss repeat mean"""

import time
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor
from hyper_parallel import init_device_mesh, hsdp, shard, DTensor, parallelize_value_and_grad
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.mindspore.st.shard.utils import create_dtensor

learning_rate = 0.01
epochs = 2


class SimpleModel(nn.Cell):
    """simple model"""

    def __init__(self, input_size, output_size, device_mesh=None, w_placements=None):
        super().__init__()
        if device_mesh is None or w_placements is None:
            self.weight = ms.Parameter(
                Tensor(np.ones([input_size, output_size]).astype(np.float32)),
                name='weight'
            )
        else:
            self.weight = ms.Parameter(
                DTensor.from_local(
                    Tensor(np.ones([input_size, output_size]).astype(np.float32)),
                    device_mesh,
                    w_placements
                ),
                name='weight'
            )
        self.relu = ms.mint.nn.ReLU()

    def construct(self, x):
        x = ms.mint.matmul(x, self.weight)
        x = self.relu(x)
        return x


def create_tensor(data):
    """create tensor """
    return Tensor(data, dtype=ms.float32)


def run_standalone(x, input_size, output_size):
    """run standalone"""
    model = SimpleModel(input_size, output_size)

    def forward_fn(data):
        logits = model(data)
        return logits

    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    x = create_tensor(x)

    ret_loss = None
    ret_grads = None
    for epoch in range(epochs):
        start = time.time()
        (loss_value, grads) = grad_fn(x)
        optimizer(grads)
        end = time.time()
        ret_loss = loss_value
        ret_grads = grads
        print(f"[standalone] Epoch: {epoch + 1}/{epochs}, Loss: {loss_value}, Time: {end - start}")

    return ret_loss, ret_grads


def run_parallel(local_x, local_input_size, local_output_size, device_mesh, x_placements, w_placements,
                 relu_input_placements, relu_output_placements, hsdp_shard_size):
    """run parallel"""
    model = SimpleModel(local_input_size, local_output_size, device_mesh, w_placements)

    model_sharding_plan = {"forward": {"input": (x_placements,)}}
    shard(model, device_mesh=device_mesh, sharding_plan=model_sharding_plan)

    model_relu_sharding_plan = {"forward": {"input": (relu_input_placements,), "output": (relu_output_placements,)}}
    shard(model.relu, device_mesh=device_mesh, sharding_plan=model_relu_sharding_plan)

    model = hsdp(model, shard_size=hsdp_shard_size, threshold=0)

    def forward_fn(data):
        logits = model(data)
        return logits

    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)
    grad_fn = parallelize_value_and_grad(forward_fn, optimizer.parameters)

    x = create_dtensor(local_x, device_mesh, x_placements, ms.float32)

    ret_loss = None
    ret_grads = None
    for epoch in range(epochs):
        start = time.time()
        (loss_value, grads) = grad_fn(x)
        optimizer(grads)
        end = time.time()
        ret_loss = loss_value
        ret_grads = grads
        print(f"[parallel] Epoch: {epoch + 1}/{epochs}, Loss: {loss_value}, Time: {end - start}")

    return ret_loss, ret_grads


def base_case(dp, mp, hsdp_shard_size):
    """base case"""
    D.init()

    # standalone
    batch_size = 4
    input_size = 32
    output_size = 2

    x = np.ones([batch_size, input_size]).astype(np.float32)
    standalone_loss, standalone_grads = run_standalone(x, input_size, output_size)

    # parallel
    local_batch_size = batch_size // dp
    local_input_size = input_size // mp
    local_output_size = output_size
    local_x = np.ones([local_batch_size, local_input_size]).astype(np.float32)

    # Create DeviceMesh
    mesh = init_device_mesh(
        mesh_shape=(dp, mp),
        alias_name=("dp", "mp")
    )

    # Define placements using Placement format
    x_placements = (Shard(0), Shard(1))
    w_placements = (Replicate(), Shard(0))
    relu_input_placements = (Shard(0), Replicate())
    relu_output_placements = (Shard(0), Replicate())

    parallel_loss, parallel_grads = run_parallel(local_x, local_input_size, local_output_size, mesh, x_placements,
                                                 w_placements, relu_input_placements, relu_output_placements,
                                                 hsdp_shard_size)

    # compare loss
    assert np.allclose(standalone_loss.asnumpy(), parallel_loss.asnumpy(), 0.001, 0.001)

    # compare grad
    if hsdp_shard_size < 0:
        hsdp_shard_size = dp

    standalone_grad = standalone_grads[0].asnumpy()
    # note: this way of obtaining grad slice is a simplified way, not a strict way
    standalone_grad_slice = standalone_grad[:local_input_size // hsdp_shard_size, :local_output_size]
    parallel_grad = parallel_grads[0].asnumpy()
    assert np.allclose(standalone_grad_slice, parallel_grad, 0.001, 0.001)


def test_loss_repeat_mean_0():
    '''
    Feature: loss repeat mean + partial + hsdp fully shard.
    Description: Test loss repeat mean.
    Expectation: Run success.
    '''
    base_case(dp=4, mp=2, hsdp_shard_size=4)


def test_loss_repeat_mean_1():
    '''
    Feature: loss repeat mean + partial + hsdp no fully shard.
    Description: Test loss repeat mean.
    Expectation: Run success.
    '''
    base_case(dp=4, mp=2, hsdp_shard_size=2)


def test_loss_repeat_mean_2():
    '''
    Feature: loss repeat mean + partial + hsdp no shard.
    Description: Test loss repeat mean.
    Expectation: Run success.
    '''
    base_case(dp=4, mp=2, hsdp_shard_size=1)


def test_loss_repeat_mean_3():
    '''
    Feature: loss repeat mean + partial + hsdp shard size is -1.
    Description: Test loss repeat mean.
    Expectation: Run success.
    '''
    base_case(dp=4, mp=2, hsdp_shard_size=-1)
