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
"""test torch dtensor with distributed tensor_split"""

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
standalone_input_2d_np = np.random.randn(8, 6).astype(np.float32)


def test_tensor_split_by_sections_unsharded() -> None:
    """Test tensor_split by sections (int) on unsharded dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_outputs = standalone_input.tensor_split(3, dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split(3, dim=1)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert len(dist_outputs) == 3, f"Expected 3 outputs, got {len(dist_outputs)}"

    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout, \
            f"Output {i} layout mismatch: expected {expected_layout}, got {out.layout}"

        gathered_output = local_to_global(out)
        assert torch.equal(
            standalone_outputs[i], gathered_output
        ), f"Split output {i} mismatch between standalone and distributed execution"


def test_tensor_split_by_indices_unsharded() -> None:
    """Test tensor_split by indices (tuple/list) on unsharded dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_outputs = standalone_input.tensor_split((1, 4), dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split((1, 4), dim=1)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert len(dist_outputs) == 3, f"Expected 3 outputs (indices len + 1), got {len(dist_outputs)}"

    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout, \
            f"Output {i} layout mismatch: expected {expected_layout}, got {out.layout}"

        gathered_output = local_to_global(out)
        assert torch.equal(
            standalone_outputs[i], gathered_output
        ), f"Indices split output {i} mismatch"


def test_tensor_split_default_dim() -> None:
    """Test tensor_split default dim (0)."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_outputs = standalone_input.tensor_split(2)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(4, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split(2)

    expected_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout
        gathered_output = local_to_global(out)
        assert torch.equal(standalone_outputs[i], gathered_output)


def test_tensor_split_negative_dim() -> None:
    """Test tensor_split with negative dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_outputs = standalone_input.tensor_split(2, dim=-1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split(2, dim=-1)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout
        gathered_output = local_to_global(out)
        assert torch.equal(standalone_outputs[i], gathered_output)


def test_tensor_split_3d_sections() -> None:
    """Test tensor_split on 3D tensor."""
    init_backend(_DEVICE_TYPE)
    input_np = np.random.randn(8, 6, 8).astype(np.float32)
    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_outputs = standalone_input.tensor_split(2, dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(2))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split(2, dim=1)

    expected_layout = _build_layout(mesh, x_placements, 3)
    assert len(dist_outputs) == 2

    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout
        gathered_output = local_to_global(out)
        assert torch.equal(standalone_outputs[i], gathered_output)


def test_tensor_split_1d_tensor_indices() -> None:
    """Test tensor_split by 1D tensor indices."""
    init_backend(_DEVICE_TYPE)
    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    split_indices = torch.tensor([1, 4])
    standalone_outputs = standalone_input.tensor_split(split_indices, dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split(split_indices, dim=1)

    expected_layout = _build_layout(mesh, x_placements, 2)
    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout
        gathered_output = local_to_global(out)
        assert torch.equal(standalone_outputs[i], gathered_output)


def test_tensor_split_uneven_sections() -> None:
    """Test tensor_split with uneven sections."""
    init_backend(_DEVICE_TYPE)
    input_np = np.random.randn(8, 7).astype(np.float32)
    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_outputs = standalone_input.tensor_split(3, dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split(3, dim=1)

    expected_layout = _build_layout(mesh, x_placements, 2)
    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout
        gathered_output = local_to_global(out)
        assert torch.equal(standalone_outputs[i], gathered_output)


def test_tensor_split_out_of_bounds_indices() -> None:
    """Test tensor_split with out-of-bounds indices."""
    init_backend(_DEVICE_TYPE)
    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_outputs = standalone_input.tensor_split((2, 10), dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split((2, 10), dim=1)

    expected_layout = _build_layout(mesh, x_placements, 2)
    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout
        gathered_output = local_to_global(out)
        assert torch.equal(standalone_outputs[i], gathered_output)


def test_tensor_split_4d_multi_shard() -> None:
    """Test tensor_split on 4D tensor with multi-sharding."""
    init_backend(_DEVICE_TYPE)
    input_np = np.random.randn(8, 4, 6, 8).astype(np.float32)
    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_outputs = standalone_input.tensor_split(2, dim=2)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(3))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split(2, dim=2)

    expected_layout = _build_layout(mesh, x_placements, 4)
    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout
        gathered_output = local_to_global(out)
        assert torch.equal(standalone_outputs[i], gathered_output)


def test_tensor_split_list_indices() -> None:
    """Test tensor_split by list of indices."""
    init_backend(_DEVICE_TYPE)
    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    indices_list = [2, 5]
    standalone_outputs = standalone_input.tensor_split(indices_list, dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split(indices_list, dim=1)

    expected_layout = _build_layout(mesh, x_placements, 2)
    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout
        gathered_output = local_to_global(out)
        assert torch.equal(standalone_outputs[i], gathered_output)


def test_tensor_split_replicated() -> None:
    """Test tensor_split on fully replicated tensor."""
    init_backend(_DEVICE_TYPE)
    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_outputs = standalone_input.tensor_split(4, dim=0)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split(4, dim=0)

    expected_layout = _build_layout(mesh, x_placements, 2)
    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout
        gathered_output = local_to_global(out)
        assert torch.equal(standalone_outputs[i], gathered_output)
