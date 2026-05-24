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
"""test torch dtensor with distributed isnan"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


# Generate mock data with NaN values
np.random.seed(42)

# 2D mock data
input_2d_np = np.random.randn(8, 8).astype(np.float32)
input_2d_np[1, 2] = np.nan
input_2d_np[5, 4] = np.nan
input_2d_np[7, 0] = np.nan

# 3D mock data
input_3d_np = np.random.randn(4, 8, 6).astype(np.float32)
input_3d_np[0, 1, 1] = np.nan
input_3d_np[3, 5, 2] = np.nan


def test_isnan_basic() -> None:
    """Test torch.isnan basic element-wise check."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(input_2d_np), _DEVICE_TYPE)
    standalone_output = torch.isnan(standalone_input)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.isnan(dist_input)

    expected_layout = _build_layout(mesh, x_placements, 2)
    assert dist_output.layout == expected_layout, \
        f"isnan output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "isnan output mismatch between standalone and distributed execution"


def test_isnan_replicate() -> None:
    """Test torch.isnan on replicated tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(input_2d_np), _DEVICE_TYPE)
    standalone_output = torch.isnan(standalone_input)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.isnan(dist_input)

    expected_layout = _build_layout(mesh, x_placements, 2)
    assert dist_output.layout == expected_layout, \
        f"isnan replicate layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "isnan replicate output mismatch"


def test_isnan_3d() -> None:
    """Test torch.isnan on 3D tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(input_3d_np), _DEVICE_TYPE)
    standalone_output = torch.isnan(standalone_input)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_output = dist_input.isnan()

    expected_layout = _build_layout(mesh, x_placements, 3)
    assert dist_output.layout == expected_layout, \
        f"isnan 3D layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "isnan 3D output mismatch"
