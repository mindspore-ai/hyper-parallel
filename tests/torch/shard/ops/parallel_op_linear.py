# Copyright 2026 Huawei Technologies Co., Ltd
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
"""test torch dtensor with distributed linear"""

import numpy as np
import torch
from torch import nn
from torch import optim
from hyper_parallel import DTensor, Layout, SkipDTensorDispatch, hsdp
from tests.torch.utils import init_dist
from tests.torch.shard.utils import global_to_local

# Generate input data using numpy at file header
np.random.seed(42)
standalone_x_np = np.random.randn(16, 8).astype(np.float32)

class SimpleModel(nn.Module):
    """simple model"""
    def __init__(self, dist=False, use_bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 8).npu())   # shape: (out_features, in_features)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(4).npu())    # shape: (out_features)
        else:
            self.bias = None
        self.dist = dist

    def forward(self, x):
        """
        x: shape (batch, in_features)
        weight: (out_features, in_features) – DTensor
        bias:   (out_features) – DTensor or local
        """
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.dist and isinstance(out, DTensor):
            out = out.reduce_partial()
        out = torch.relu(out)
        out = torch.sum(out)
        return out


def test_distributed_linear_with_bias():
    """
    Feature: dtensor + torch.nn.functional.linear with bias
    Description:
        - Train a simple model: Linear → ReLU → Sum
        - Compare loss between single-machine and distributed training
    Expectation: Run successfully.
    """
    init_dist()
    step = 10

    standalone_model = SimpleModel().npu()
    standalone_optimizer = optim.SGD(standalone_model.parameters(), lr=0.01)
    standalone_x = torch.from_numpy(standalone_x_np).npu()

    for _ in range(step):
        standalone_loss = standalone_model(standalone_x)
        standalone_loss.backward()
        standalone_optimizer.step()
        standalone_optimizer.zero_grad()

    dist_model = SimpleModel(dist=True).npu()

    # weight:  (8,4)  -> split on second dim
    local_w = torch.ones(4, 4).npu()
    # bias:   (4,)   -> broadcast (no split)
    local_b = torch.zeros(4).npu()

    layout = Layout((4, 2), ("dp", "tp"))
    w_layout = layout("None", "tp")
    b_layout = layout("None",)
    x_layout = layout("dp", "tp")

    dist_model.register_parameter(
        "weight",
        nn.Parameter(DTensor.from_local(local_w, w_layout.mesh, w_layout.placements)),
    )
    dist_model.register_parameter(
        "bias",
        nn.Parameter(DTensor.from_local(local_b, b_layout.mesh, b_layout.placements)),
    )

    dist_x = global_to_local(standalone_x, x_layout)

    dist_model = hsdp(dist_model)
    dist_optimizer = optim.SGD(dist_model.parameters(), lr=0.01)

    for _ in range(step):
        dist_loss = dist_model(dist_x)
        dist_loss = dist_loss.reduce_partial()

        repeat_num = dist_loss.layout.repeat_num()
        backward_input = torch.tensor(1.0 / repeat_num)
        dist_loss.backward(backward_input)

        with SkipDTensorDispatch():
            dist_optimizer.step()
            dist_optimizer.zero_grad()

    # loss
    assert np.allclose(
        standalone_loss.cpu().detach().numpy(),
        dist_loss.to_local().cpu().detach().numpy(),
        atol=1e-3,
        rtol=1e-3,
    ), "loss mismatch between standalone and distributed"


def test_distributed_linear_without_bias():
    """
    Feature: dtensor + torch.nn.functional.linear without bias
    Description:
        - Train a simple model: Linear → ReLU → Sum
        - Compare loss between single-machine and distributed training
        - Bias is None, no bias parameter
    Expectation: Run successfully.
    """
    init_dist()
    step = 10

    standalone_model = SimpleModel(use_bias=False).npu()
    standalone_optimizer = optim.SGD(standalone_model.parameters(), lr=0.01)
    standalone_x = torch.from_numpy(standalone_x_np).npu()

    for _ in range(step):
        standalone_loss = standalone_model(standalone_x)
        standalone_loss.backward()
        standalone_optimizer.step()
        standalone_optimizer.zero_grad()

    dist_model = SimpleModel(dist=True, use_bias=False).npu()

    # weight:  (8,4)  -> split on second dim
    local_w = torch.ones(4, 4).npu()

    layout = Layout((4, 2), ("dp", "tp"))
    w_layout = layout("None", "tp")
    x_layout = layout("dp", "tp")

    dist_model.register_parameter(
        "weight",
        nn.Parameter(DTensor.from_local(local_w, w_layout.mesh, w_layout.placements)),
    )

    dist_x = global_to_local(standalone_x, x_layout)

    dist_model = hsdp(dist_model)
    dist_optimizer = optim.SGD(dist_model.parameters(), lr=0.01)

    for _ in range(step):
        dist_loss = dist_model(dist_x)
        dist_loss = dist_loss.reduce_partial()

        repeat_num = dist_loss.layout.repeat_num()
        backward_input = torch.tensor(1.0 / repeat_num)
        dist_loss.backward(backward_input)

        with SkipDTensorDispatch():
            dist_optimizer.step()
            dist_optimizer.zero_grad()

    # loss
    assert np.allclose(
        standalone_loss.cpu().detach().numpy(),
        dist_loss.to_local().cpu().detach().numpy(),
        atol=1e-3,
        rtol=1e-3,
    ), "loss mismatch between standalone and distributed"


def test_distributed_linear_with_bias_shard():
    """
    Feature: dtensor + torch.nn.functional.linear with bias sharding
    Description:
        - Train a simple model: Linear → ReLU → Sum
        - Compare loss between single-machine and distributed training
        - Bias is not None and needs sharding
    Expectation: Run successfully.
    """
    init_dist()
    step = 10

    standalone_model = SimpleModel().npu()
    standalone_optimizer = optim.SGD(standalone_model.parameters(), lr=0.01)
    standalone_x = torch.from_numpy(standalone_x_np).npu()

    for _ in range(step):
        standalone_loss = standalone_model(standalone_x)
        standalone_loss.backward()
        standalone_optimizer.step()
        standalone_optimizer.zero_grad()

    dist_model = SimpleModel(dist=True).npu()

    # weight:  (8,4)  -> split on second dim
    local_w = torch.ones(2, 4).npu()
    # bias:   (4,)   -> split on first dim
    local_b = torch.zeros(2).npu()

    layout = Layout((2, 2, 2), ("dp", "tp1", "tp2"))
    x_layout = layout("dp", "tp1")
    w_layout = layout("tp2", "tp1")
    b_layout = layout("tp2")

    dist_model.register_parameter(
        "weight",
        nn.Parameter(DTensor.from_local(local_w, w_layout.mesh, w_layout.placements)),
    )
    dist_model.register_parameter(
        "bias",
        nn.Parameter(DTensor.from_local(local_b, b_layout.mesh, b_layout.placements)),
    )

    dist_x = global_to_local(standalone_x, x_layout)

    dist_model = hsdp(dist_model)
    dist_optimizer = optim.SGD(dist_model.parameters(), lr=0.01)

    for _ in range(step):
        dist_loss = dist_model(dist_x)
        dist_loss = dist_loss.reduce_partial()

        repeat_num = dist_loss.layout.repeat_num()
        backward_input = torch.tensor(1.0 / repeat_num)
        dist_loss.backward(backward_input)

        with SkipDTensorDispatch():
            dist_optimizer.step()
            dist_optimizer.zero_grad()

    # loss
    assert np.allclose(
        standalone_loss.cpu().detach().numpy(),
        dist_loss.to_local().cpu().detach().numpy(),
        atol=1e-3,
        rtol=1e-3,
    ), "loss mismatch between standalone and distributed"
