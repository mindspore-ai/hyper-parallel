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
"""test torch dtensor"""
import numpy as np
import torch
from torch import nn
from torch import optim
# pylint: disable=W0611
import torch_npu  # 昇腾NPU核心适配
from hyper_parallel import DTensor, Layout, SkipDTensorDispatch
from tests.torch.utils import init_dist


class SimpleModel(nn.Module):
    """simple model"""
    def __init__(self, dist=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8, 8).npu())
        self.dist = dist
        if self.dist is True:
            layout = Layout((1, 8), ("dp", "tp"))
            self.layout1 = layout("None", "None")
            self.layout2 = layout("dp", "tp")
            self.layout3 = layout("tp", "dp")

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        if self.dist is True:
            x = x.redistribute(self.layout1)
        x = torch.relu(x)
        if self.dist is True:
            x = x.redistribute(self.layout2)
            x = x.redistribute(self.layout3)
        x = torch.sum(x)
        return x


def test_base_dtensor():
    '''
    Feature: dtensor infer layout and redistribute.
    Description:
    Expectation: Run success.
    '''
    init_dist()
    step = 2

    # -----------------------------------standalone----------------------------------
    standalone_model = SimpleModel().npu()
    standalone_optimizer = optim.SGD(standalone_model.parameters(), lr=0.01)
    standalone_x = torch.ones(8, 8).npu()
    for _ in range (step):
        standalone_loss = standalone_model(standalone_x)
        standalone_loss.backward()
        standalone_grad = standalone_model.weight.grad.data  # using for validation
        standalone_optimizer.step()
        standalone_optimizer.zero_grad()

    # --------------------------------------dist-------------------------------------
    dist_model = SimpleModel(dist=True).npu()

    layout = Layout((1, 8), ("dp", "tp"))
    x_layout = layout("None", "None")
    w_layout = layout("None", "tp")
    dist_x = DTensor.from_local(torch.ones(8, 8).npu(), x_layout)
    local_w = torch.ones(8, 1).npu()
    # pylint: disable=W0212
    for key, param in dist_model._parameters.items():
        if param is not None and not isinstance(param, DTensor):
            dist_model.register_parameter(
                key,
                nn.Parameter(DTensor.from_local(local_w, w_layout)),
            )

    dist_optimizer = optim.SGD(dist_model.parameters(), lr=0.01)

    for _ in range (step):
        dist_loss = dist_model(dist_x)
        dist_loss = dist_loss.reduce_partial()  # handle partial state

        # handle backward input
        repeat_num = dist_loss.layout.repeat_num()
        backward_input = torch.tensor(1.0 / repeat_num)
        dist_loss.backward(backward_input)

        dist_grad = dist_model.weight.grad.data

        with SkipDTensorDispatch():
            dist_optimizer.step()
            dist_optimizer.zero_grad()

    assert np.allclose(standalone_loss.cpu().detach().numpy(),
                       dist_loss.to_local().cpu().detach().numpy(),  # use to_local()
                       0.001, 0.001)

    assert np.allclose(standalone_grad.cpu().detach().numpy(),
                       dist_grad.cpu().detach().numpy(),
                       0.001, 0.001)
