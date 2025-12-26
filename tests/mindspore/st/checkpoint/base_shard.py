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

from hyper_parallel import Layout, shard, parallelize_value_and_grad
from hyper_parallel.core.checkpoint.layout import get_current_layout, save_layout, load_layout


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
    layout = Layout((dp, mp), ("dp", "mp"))
    x_layout = layout("dp", "mp")
    w_layout = layout("mp", "None")
    out_layout = layout()
    relu_strategy = ((layout("dp", "None"),), (layout("dp", "None"),))

    # step 1: define network with no init parameters
    with no_init_parameters():
        model = SimpleModel(input_size, output_size)

    # step 2: shard
    model_stra = {"forward": {"input": (x_layout,), "output": (out_layout,)},
                  "parameter": {"weight": w_layout}}
    shard(model, model_stra)

    model_relu_stra = {"forward": {"input": relu_strategy[0], "output": relu_strategy[1]}}
    shard(model.relu, model_relu_stra)

    # step 3: save layout
    layout_dict = get_current_layout(model)
    if get_rank() == 0:
        file_name = "test.layout"
        save_layout(layout_dict, file_name)
        assert os.path.isfile(file_name)
        layout_dict = load_layout(file_name)
        assert isinstance(layout_dict, dict)
        os.remove(file_name)


def test_base_layout():
    """
    Feature: test layout save and load.
    Description: Test base layout save.
    Expectation: Run success.
    """
    base_case(dp=4, mp=2)
