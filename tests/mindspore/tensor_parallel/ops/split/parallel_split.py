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
"""parallel split"""

import time
import numpy as np
import mindspore as ms
from mindspore._c_expression import NoFallbackGuard
import mindspore.communication.management as D
from mindspore import nn, Tensor
from hyper_parallel import Layout, hsdp, init_parameters, parallelize_value_and_grad, shard
from mindspore.nn.utils import no_init_parameters
from mindspore.common.initializer import initializer
from tests.mindspore.tensor_parallel.utils import create_dtensor

learning_rate = 0.01
epochs = 2


class SimpleModel(nn.Cell):
    """simple model"""

    def __init__(self, batch_size, input_size, output_size):
        super().__init__()
        self.batch_size = batch_size
        self.weight = ms.Parameter(initializer("ones", [input_size, output_size], ms.float32), name='weight')
        self.relu = ms.mint.nn.ReLU()

    def construct(self, x):
        x, _, _ = ms.mint.split(x, self.batch_size, 0)
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
        print(f"[standalone] Epoch: {epoch + 1}/{epochs}, Loss: {loss_value}, Time: {end - start}")

    return ret_loss, ret_grads


def base_case(dp, mp, hsdp_shard_size):
    """base case"""
    D.init()

    # standalone
    input_size = 32
    output_size = 2
    batch_size = 4

    standalone_x = Tensor(np.ones([batch_size * 3, input_size]).astype(np.float32), dtype=ms.float32)
    standalone_model = SimpleModel(batch_size, input_size, output_size)
    standalone_loss, standalone_grads = run_model(standalone_x, standalone_model)

    # parallel
    local_batch_size = batch_size
    local_input_size = input_size // mp
    local_output_size = output_size
    local_x = np.ones([local_batch_size * 3, local_input_size]).astype(np.float32)
    layout = Layout((dp, mp), ("dp", "mp"))
    x_layout = layout("None", "mp")
    w_layout = layout("mp", "None")
    out_layout = layout()
    relu_strategy = ((layout("dp", "None"),), (layout("dp", "None"),))

    # step 1: define network with no init parameters
    with no_init_parameters():
        model = SimpleModel(batch_size, input_size, output_size)

    # step 2: shard
    model_stra = {
        "forward": {
            "input": (x_layout,),
            "output": (out_layout,)
        },
        "parameter": {
            "weight": w_layout
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

    # step 3: hsdp
    model = hsdp(model, shard_size=hsdp_shard_size, threshold=0)

    # step 4: init parameters
    model = init_parameters(model)

    x = create_dtensor(local_x, x_layout)
    parallel_loss, parallel_grads = run_model(x, model, parallel=True)

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


def test_parallel_split():
    '''
    Feature: with no_init_parameters + cell shard + hsdp + init param + loss repeat + partial.
    Description: Test base shard.
    Expectation: Run success.
    '''
    base_case(dp=4, mp=2, hsdp_shard_size=4)
