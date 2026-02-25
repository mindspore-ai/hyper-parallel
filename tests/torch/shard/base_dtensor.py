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
from hyper_parallel import DTensor, SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist


class SimpleModel(nn.Module):
    """simple model"""
    def __init__(self, dist=False, mesh=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8, 8).npu())
        self.dist = dist
        self.mesh = mesh

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        if self.dist is True:
            x = x.redistribute(self.mesh, (Replicate(), Replicate()))
        x = torch.relu(x)
        if self.dist is True:
            x = x.redistribute(self.mesh, (Shard(0), Shard(1)))
            x = x.redistribute(self.mesh, (Shard(1), Shard(0)))
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
    for _ in range(step):
        standalone_loss = standalone_model(standalone_x)
        standalone_loss.backward()
        standalone_grad = standalone_model.weight.grad.data  # using for validation
        standalone_optimizer.step()
        standalone_optimizer.zero_grad()

    # --------------------------------------dist-------------------------------------
    # Create DeviceMesh
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(1, 8),
        mesh_dim_names=("dp", "tp")
    )

    dist_model = SimpleModel(dist=True, mesh=mesh).npu()

    # Define placements using Placement format
    x_placements = (Replicate(), Replicate())
    w_placements = (Replicate(), Shard(1))

    dist_x = DTensor.from_local(torch.ones(8, 8).npu(), mesh, x_placements)
    local_w = torch.ones(8, 1).npu()

    # pylint: disable=W0212
    for key, param in dist_model._parameters.items():
        if param is not None and not isinstance(param, DTensor):
            dist_model.register_parameter(
                key,
                nn.Parameter(DTensor.from_local(local_w, mesh, w_placements)),
            )

    dist_optimizer = optim.SGD(dist_model.parameters(), lr=0.01)

    for _ in range(step):
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


def test_dtensor_float():
    '''
    Feature: DTensor.float() method.
    Description: Test converting DTensor to float dtype
    Expectation: Run success and DTensor is converted to float dtype.
    '''
    init_dist()
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(1, 8),
        mesh_dim_names=("dp", "tp")
    )
    local_tensor = torch.ones(8, 8).npu().half()
    local_float_tensor = local_tensor.clone().detach().float()
    dtensor = DTensor.from_local(local_tensor, mesh, (Replicate(), Shard(0)))
    float_dtensor = dtensor.float()
    assert float_dtensor._local_tensor.dtype == torch.float32 #pylint: disable=W0212
    assert (float_dtensor._local_tensor.cpu().detach().numpy() == local_float_tensor.cpu().detach().numpy()).all() #pylint: disable=W0212
