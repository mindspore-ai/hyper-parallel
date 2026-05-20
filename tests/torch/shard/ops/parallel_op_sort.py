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
"""test torch dtensor with distributed sort"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import init_backend, to_device
from tests.torch.shard.utils import local_to_global

try:
    import torch_npu  # pylint: disable=W0611
    _DEVICE_TYPE = "npu"
except ImportError:
    _DEVICE_TYPE = "cpu"

np.random.seed(42)
torch.manual_seed(42)

standalone_input_2d_np = np.random.randn(8, 16).astype(np.float32)
standalone_input_3d_np = np.random.randn(4, 8, 6).astype(np.float32)


def test_sort_basic() -> None:
    """Test sort along an unsharded dimension (last dim)."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    s_values, s_indices = torch.sort(standalone_input, dim=1, descending=False)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    d_values, d_indices = dist_input.sort(dim=1, descending=False)

    expected_layout = _build_layout(mesh, x_placements, 2)
    assert d_values.layout == expected_layout, \
        f"Values layout mismatch: expected {expected_layout}, got {d_values.layout}"
    assert d_indices.layout == expected_layout, \
        f"Indices layout mismatch: expected {expected_layout}, got {d_indices.layout}"

    g_values = local_to_global(d_values)
    g_indices = local_to_global(d_indices)

    assert torch.equal(s_values, g_values), "Sort values mismatch"
    assert torch.equal(s_indices, g_indices), "Sort indices mismatch"


def test_sort_descending() -> None:
    """Test sort along unsharded dimension with descending=True."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    s_values, s_indices = torch.sort(standalone_input, dim=1, descending=True)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    d_values, d_indices = dist_input.sort(dim=1, descending=True)

    g_values = local_to_global(d_values)
    g_indices = local_to_global(d_indices)

    assert torch.equal(s_values, g_values), "Sort descending values mismatch"
    assert torch.equal(s_indices, g_indices), "Sort descending indices mismatch"


def test_sort_middle_dim() -> None:
    """Test sort on middle dimension with 3D tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    s_values, s_indices = torch.sort(standalone_input, dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    d_values, d_indices = dist_input.sort(dim=1)

    expected_layout = _build_layout(mesh, x_placements, 3)
    assert d_values.layout == expected_layout
    assert d_indices.layout == expected_layout

    g_values = local_to_global(d_values)
    g_indices = local_to_global(d_indices)

    assert torch.equal(s_values, g_values), "3D Sort values mismatch"
    assert torch.equal(s_indices, g_indices), "3D Sort indices mismatch"


def test_sort_negative_dim() -> None:
    """Test sort with negative dimension index."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    s_values, s_indices = torch.sort(standalone_input, dim=-1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    d_values, d_indices = dist_input.sort(dim=-1)

    g_values = local_to_global(d_values)
    g_indices = local_to_global(d_indices)

    assert torch.equal(s_values, g_values), "Sort negative dim values mismatch"
    assert torch.equal(s_indices, g_indices), "Sort negative dim indices mismatch"
