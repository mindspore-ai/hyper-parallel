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
"""parallelize value and grad"""

import time
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn
from mindspore.nn.utils import no_init_parameters
from mindspore.common.initializer import initializer
from hyper_parallel import Layout, hsdp, init_parameters, shard, parallelize_value_and_grad
from tests.mindspore.st.tensor_parallel.utils import create_dtensor

learning_rate = 0.01
epochs = 2


class SimpleModel(nn.Cell):
    """simple model"""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = ms.Parameter(initializer("ones", [input_size, output_size], ms.float32), name='weight')
        self.relu = ms.mint.nn.ReLU()
        self.num = 0

    def construct(self, x):
        x = ms.mint.matmul(x, self.weight)
        x = self.relu(x)
        self.num = self.num + 1
        x = ms.mint.sum(x)
        return x


def run_model(x, model, use_cell=False):
    """rum model"""
    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)

    def forward_fn(data):
        """forward fn"""
        logits = model(data)
        return logits

    if use_cell is True:
        fn = model
    else:
        fn = forward_fn

    grad_fn = parallelize_value_and_grad(fn, optimizer.parameters)

    ret_loss = None
    ret_grads = None
    for epoch in range(epochs):
        start = time.time()
        (loss_value, grads) = grad_fn(x)
        optimizer(grads)
        end = time.time()
        ret_loss = loss_value
        ret_grads = grads
        print(f"use_cell: {use_cell}, Epoch: {epoch + 1}/{epochs}, Loss: {loss_value}, Time: {end - start}")

    return ret_loss, ret_grads


def base_case(dp, mp, hsdp_shard_size):
    """base case"""
    D.init()

    input_size = 32
    output_size = 2
    batch_size = 4

    # parallel
    local_batch_size = batch_size // dp
    local_input_size = input_size // mp
    local_x = np.ones([local_batch_size, local_input_size]).astype(np.float32)
    layout = Layout((dp, mp), ("dp", "mp"))
    x_layout = layout("dp", "mp")
    w_layout = layout("mp", "None")
    out_layout = layout()
    relu_strategy = ((layout("dp", "None"),), (layout("dp", "None"),))

    # input is fn
    with no_init_parameters():
        model = SimpleModel(input_size, output_size)

    stra = {"forward": {"input": (x_layout,), "output": (out_layout,)}, "parameter": {"weight": w_layout}}
    shard(model, stra)

    relu_stra = {"forward": {"input": relu_strategy[0], "output": relu_strategy[1]}}
    shard(model.relu, relu_stra)

    model = hsdp(model, shard_size=hsdp_shard_size, threshold=0)

    model = init_parameters(model)

    x = create_dtensor(local_x, x_layout)
    parallel_loss, parallel_grads = run_model(x, model, use_cell=False)

    # input is cell
    with no_init_parameters():
        model_1 = SimpleModel(input_size, output_size)

    shard(model_1, stra)
    shard(model_1.relu, relu_stra)

    model_1 = hsdp(model_1, shard_size=hsdp_shard_size, threshold=0)

    model_1 = init_parameters(model_1)

    parallel_loss_1, parallel_grads_1 = run_model(x, model_1, use_cell=True)

    # validate
    assert np.allclose(parallel_loss_1.asnumpy(), parallel_loss.asnumpy(), 0.001, 0.001)
    grad = parallel_grads[0].asnumpy()
    grad_1 = parallel_grads_1[0].asnumpy()
    assert np.allclose(grad, grad_1, 0.001, 0.001)
    assert model.num == epochs
    assert model_1.num == epochs


def test_parallelize_value_and_grad():
    '''
    Feature: the input is fn or cell, it only run once in one step
    Description: Test base case.
    Expectation: Run success.
    '''
    base_case(dp=4, mp=2, hsdp_shard_size=4)
