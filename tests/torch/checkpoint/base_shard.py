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
"""base shard"""
import numpy as np
import torch
from torch import optim
from hyper_parallel import shard, init_device_mesh
from hyper_parallel.core.checkpoint.layout import get_global_layout
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.common_net import SimpleModel
from tests.torch.shard.base_shard import mse_loss_sum

from tests.torch.utils import init_dist


def base_global_layout():
    """ base global layout """
    init_dist()
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
        standalone_optimizer.step()
        standalone_optimizer.zero_grad()

    # --------------------------------------dist-------------------------------------
    # Ensure same seed for initialization
    torch.manual_seed(1)
    dist_model = SimpleModel().npu()

    # Create DeviceMesh
    mesh = init_device_mesh(mesh_shape=(1, 8), alias_name=("dp", "tp"))

    # Define placements using Placement format
    w_placements = (Replicate(), Shard(1))  # tp shard on dim 1
    in_placements = (Replicate(), Replicate())
    out_placements = (Replicate(), Shard(1))  # tp shard on dim 1

    sharding_plan = {
        "parameter": {
            "weight": w_placements
        },
        "forward": {
            "input": in_placements,
            "output": out_placements
        }
    }

    # Apply shard
    dist_model = shard(dist_model, device_mesh=mesh, sharding_plan=sharding_plan)

    # step 3: get global layout
    global_layout = get_global_layout(dist_model)
    assert isinstance(global_layout, dict)


def test_get_global_layout():
    """
    Feature: Test get global layout on all ranks.
    Description: Test when a simple model sharded by dp and tp, gather global layout on all ranks.
    Expectation: Run success.
    """
    base_global_layout()
