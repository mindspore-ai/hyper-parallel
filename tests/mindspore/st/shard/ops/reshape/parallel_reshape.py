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
"""parallel reshape"""

import time
import numpy as np
import mindspore as ms
from mindspore._c_expression import NoFallbackGuard
import mindspore.communication.management as D
from mindspore import nn, Tensor
from hyper_parallel import Layout, hsdp, DTensor, parallelize_value_and_grad, shard
from tests.mindspore.st.shard.utils import create_dtensor

learning_rate = 0.01
epochs = 2


class SimpleModel(nn.Cell):
    """simple model"""

    def __init__(self, input_size, output_size, w_layout=None):
        super().__init__()
        if not w_layout:
            self.weight = ms.Parameter(
                Tensor(np.ones([input_size, output_size]).astype(np.float32)),
                name='weight'
            )
        else:
            self.weight = ms.Parameter(
                DTensor.from_local(Tensor(np.ones([input_size, output_size]).astype(np.float32)), w_layout),
                name='weight'
            )

        self.relu = ms.mint.nn.ReLU()

    def construct(self, x):
        target_shape = x.shape[:-2] + (self.weight.shape[0],)
        x = ms.ops.reshape(x, target_shape)
        x = ms.mint.matmul(x, self.weight)
        x = self.relu(x)
        return x


def create_tensor(data):
    """create tensor"""
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


def run_parallel(local_x, local_input_size, local_output_size, x_layout, w_layout, relu_strategy, hsdp_shard_size):
    """run parallel"""
    model = SimpleModel(local_input_size, local_output_size, w_layout)

    model_stra = {
        "forward": {
            "input": (x_layout,)
        }
    }
    shard(model, model_stra)

    model_relu_stra = {
        "forward": {
            "input": relu_strategy[0],
            "output": relu_strategy[1]
        }
    }
    shard(model.relu, model_relu_stra)

    model = hsdp(model, shard_size=hsdp_shard_size)

    def forward_fn(data):
        logits = model(data)
        return logits

    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)
    grad_fn = parallelize_value_and_grad(forward_fn, optimizer.parameters)

    x = create_dtensor(local_x, x_layout)

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
        print(f"[parallel] Epoch: {epoch + 1}/{epochs}, Loss: {loss_value}, Time: {end - start}")

    return ret_loss, ret_grads


def base_case(dp, mp):
    """base case"""
    D.init()

    # standalone
    input_size = 32
    output_size = 2
    batch_size = 4
    x_last_dim = 4

    x = np.ones([batch_size, input_size // x_last_dim, x_last_dim]).astype(np.float32)
    standalone_loss, standalone_grads = run_standalone(x, input_size, output_size)

    # parallel
    hsdp_shard_size = dp
    local_batch_size = batch_size // dp
    local_input_size = input_size // mp
    local_output_size = output_size
    local_x = np.ones([local_batch_size, local_input_size // x_last_dim, x_last_dim]).astype(np.float32)
    layout = Layout((dp, mp), ("dp", "mp"))
    x_layout = layout("dp", "mp", "None")
    w_layout = layout("mp", "None")
    relu_strategy = ((layout("dp", "None"),), (layout("dp", "None"),))
    parallel_loss, parallel_grads = run_parallel(local_x, local_input_size, local_output_size, x_layout, w_layout,
                                                 relu_strategy, hsdp_shard_size)

    # compare loss
    assert np.allclose(standalone_loss.asnumpy(), parallel_loss.asnumpy(), 0.001, 0.001)

    standalone_grad = standalone_grads[0].asnumpy()
    # note: this way of obtaining grad slice is a simplified way, not a strict way
    standalone_grad_slice = standalone_grad[:local_input_size, :local_output_size]
    parallel_grad = parallel_grads[0].asnumpy()
    assert np.allclose(standalone_grad_slice, parallel_grad, 0.001, 0.001)


def test_parallel_reshape_0():
    '''
    Feature: Reshape parallel op.
    Description: Test Reshape parallel op.
    Expectation: Run success.
    '''
    base_case(dp=4, mp=2)
