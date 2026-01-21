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
from hyper_parallel import Layout, shard
from hyper_parallel.core.checkpoint.layout import get_global_layout
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
