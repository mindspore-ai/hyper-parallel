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
"""test base shard"""
import numpy as np
import torch
from torch import nn
from torch import optim
# pylint: disable=W0611
import torch_npu  # 昇腾NPU核心适配
from hyper_parallel import DTensor, Layout, SkipDTensorDispatch
from hyper_parallel.core.shard.api import shard
from tests.torch.utils import init_dist


class SimpleModel(nn.Module):
    """simple model"""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(8, 8))

    def forward(self, x):
        """forward"""
        return torch.matmul(x, self.weight)


def mse_loss_sum(y_pred, y_true):
    """
    mse loss sum
    """
    error = torch.sub(y_pred, y_true)
    square_error = torch.square(error)
    mse = torch.sum(square_error)
    return mse


def test_base_shard():
    '''
    Feature: shard function.
    Description: Test shard function with simple model.
    Expectation: Run success and result matches standalone.
    '''
    rank, _ = init_dist()
    step = 2
    torch.manual_seed(1)
    np.random.seed(0)

    # -----------------------------------standalone----------------------------------
    # Ensure same seed for initialization
    torch.manual_seed(1)
    standalone_model = SimpleModel().npu()
    standalone_optimizer = optim.SGD(standalone_model.parameters(), lr=0.01)

    # Fixed input and label for standalone
    torch.manual_seed(2)
    standalone_x = torch.randn(32, 8).npu()
    torch.manual_seed(3)
    standalone_y = torch.randn(32, 8).npu()

    for _ in range(step):
        standalone_loss = mse_loss_sum(standalone_model(standalone_x), standalone_y)
        standalone_loss.backward()
        standalone_grad = standalone_model.weight.grad.data.clone()
        standalone_optimizer.step()
        standalone_optimizer.zero_grad()

    # --------------------------------------dist-------------------------------------
    # Ensure same seed for initialization
    torch.manual_seed(1)
    dist_model = SimpleModel().npu()

    # Define sharding plan
    layout = Layout((1, 8), ("dp", "tp"))
    sharding_plan = {
        "parameter": {
            "weight": layout("None", "tp")
        },
        "forward": {
            "input": [layout("None", "None")],
            "output": [layout("None", "tp")]
        }
    }

    # Apply shard
    dist_model = shard(dist_model, sharding_plan)

    dist_optimizer = optim.SGD(dist_model.parameters(), lr=0.01)

    torch.manual_seed(2)
    dist_x_local = torch.randn(32, 8).npu()
    dist_x = DTensor.from_local(dist_x_local, layout("None", "None"))

    torch.manual_seed(3)
    dist_y_local = torch.randn(32, 8).npu()
    dist_y = DTensor.from_local(dist_y_local, layout("None", "None"))

    for _ in range(step):
        y_pred = dist_model(dist_x)

        y_shard = dist_y.redistribute(y_pred.layout)

        dist_loss = mse_loss_sum(y_pred, y_shard)
        dist_loss = dist_loss.reduce_partial()

        repeat_num = dist_loss.layout.repeat_num()
        backward_input = torch.tensor(1.0 / repeat_num).npu()
        dist_loss.backward(backward_input)

        dist_grad = dist_model.weight.grad.data

        with SkipDTensorDispatch():
            dist_optimizer.step()
            dist_optimizer.zero_grad()

    assert np.allclose(standalone_loss.cpu().detach().numpy(),
                       dist_loss.to_local().cpu().detach().numpy(),
                       rtol=1e-3, atol=1e-3)

    start = rank
    end = rank + 1
    standalone_grad_slice = standalone_grad[:, start:end]

    assert np.allclose(standalone_grad_slice.cpu().detach().numpy(),
                       dist_grad.cpu().detach().numpy(),
                       rtol=1e-3, atol=1e-3)
