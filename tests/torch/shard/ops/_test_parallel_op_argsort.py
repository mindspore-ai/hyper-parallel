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
"""test torch dtensor with distributed argsort"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


# Use distinct random values to avoid ambiguous sort orders
np.random.seed(42)
standalone_input_2d_np = np.random.rand(8, 4).astype(np.float32)
standalone_input_3d_np = np.random.rand(4, 8, 6).astype(np.float32)


def test_argsort_basic_unsharded() -> None:
    """Test torch.argsort on unsharded dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = torch.argsort(standalone_input, dim=-1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.argsort(dist_input, dim=-1)

    expected_layout = _build_layout(mesh, x_placements, 2)
    assert dist_output.layout == expected_layout, (
        f"Argsort output layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Argsort output mismatch between standalone and distributed execution"


def test_argsort_specific_dim() -> None:
    """Test torch.argsort on a specific dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = torch.argsort(standalone_input, dim=0)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.argsort(dist_input, dim=0)

    expected_layout = _build_layout(mesh, x_placements, 2)
    assert dist_output.layout == expected_layout, (
        f"Argsort layout mismatch on specific dim: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Argsort output mismatch on specific dim"


def test_argsort_descending() -> None:
    """Test torch.argsort with descending flag."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    standalone_output = torch.argsort(standalone_input, dim=1, descending=True)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.argsort(dist_input, dim=1, descending=True)

    expected_layout = _build_layout(mesh, x_placements, 3)
    assert dist_output.layout == expected_layout, (
        f"Argsort descending layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Argsort descending output mismatch"
