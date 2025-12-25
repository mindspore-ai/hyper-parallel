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
"""enable set hsdp comm reduce type"""
import os
import numpy as np
import mindspore as ms
from mindspore._c_expression import NoFallbackGuard
from mindspore import nn, Tensor
from mindspore.communication.management import init
from hyper_parallel import hsdp, parallelize_value_and_grad
from tests.common.mark_utils import arg_mark

os.environ["MS_SIMULATION_LEVEL"] = "1"
os.environ["RANK_SIZE"] = "32"
os.environ["RANK_ID"] = "32"
init()

loss_fn = nn.MSELoss()

def get_forward_fn(net):
    def forward_fn(data, label):
        logits = net(data)
        loss = loss_fn(logits, label)
        return loss, logits
    return forward_fn

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_hsdp_reduce_dtype():
    """
    Feature: hsdp reduce dtype
    Description: set hsdp gradient reduce dtype
    Expectation: run success
    """
    hidden_size = 64
    batch_size = 4
    net = nn.Dense(hidden_size, hidden_size, dtype=ms.float16)
    data = Tensor(np.random.randn(batch_size, hidden_size), ms.float16)
    label = Tensor(np.random.randn(batch_size, hidden_size), ms.float16)
    shard_size = 4
    threshold = 0
    hsdp(net, shard_size, threshold, reduce_dtype=ms.float32)
    grad_fn = parallelize_value_and_grad(get_forward_fn(net), net.trainable_params())
    train_steps = 2
    optimizer = nn.Adam(net.trainable_params(), 0.01)
    for _ in range(train_steps):
        _, grads = grad_fn(data, label)
        with NoFallbackGuard():
            optimizer(grads)
