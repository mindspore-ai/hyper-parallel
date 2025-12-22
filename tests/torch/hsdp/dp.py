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
"""test data parallel"""
import numpy as np
import torch
from torch import nn
from torch import optim
# pylint: disable=W0611
import torch_npu
from hyper_parallel import hsdp, DTensor, Layout, SkipDTensorDispatch
from tests.torch.common_net import SimpleModel
from tests.torch.utils import init_dist


def test_data_parallel():
    """test data parallel"""
    rank, _ = init_dist()
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
    dist_model = SimpleModel().npu()

    layout = Layout((2, 4), ("dp", "tp"))
    x_layout = layout("None", "None")
    w_layout = layout("None", "tp")
    dist_x = DTensor.from_local(torch.ones(8, 8).npu(), x_layout)
    local_w = torch.ones(8, 2).npu()
    # pylint: disable=W0212
    for key, param in dist_model._parameters.items():
        if param is not None and not isinstance(param, DTensor):
            dist_model.register_parameter(
                key,
                nn.Parameter(DTensor.from_local(local_w, w_layout)),
            )
    dist_model = hsdp(dist_model, shard_size=1)
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
    offset = rank % 4 * 2
    assert np.allclose(standalone_grad.cpu().detach().numpy()[:, offset: offset + 2],
                       dist_grad.cpu().detach().numpy(),
                       0.001, 0.001)
