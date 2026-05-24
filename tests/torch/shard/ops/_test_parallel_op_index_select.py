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
"""test torch dtensor with distributed index_select"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


np.random.seed(42)
standalone_input_2d_np = np.random.randn(8, 4).astype(np.float32)
standalone_input_3d_np = np.random.randn(4, 6, 8).astype(np.float32)
index_1d_np = np.array([1, 3], dtype=np.int64)


def test_index_select_basic() -> None:
    """Test index_select basic selection on unsharded dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_index = to_device(torch.from_numpy(index_1d_np), _DEVICE_TYPE)
    standalone_output = torch.index_select(standalone_input, 1, standalone_index)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    idx_placements = (Replicate(), Replicate())
    dist_index = distribute_tensor(standalone_index, mesh, idx_placements)

    dist_output = torch.index_select(dist_input, 1, dist_index)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, (
        f"IndexSelect output layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "IndexSelect output mismatch between standalone and distributed execution"


def test_index_select_3d() -> None:
    """Test index_select on 3D tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    standalone_index = to_device(torch.from_numpy(index_1d_np), _DEVICE_TYPE)
    standalone_output = torch.index_select(standalone_input, 1, standalone_index)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Shard(1))
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    idx_placements = (Replicate(), Replicate())
    dist_index = distribute_tensor(standalone_index, mesh, idx_placements)

    dist_output = torch.index_select(dist_input, 1, dist_index)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 3)
    assert dist_output.layout == expected_layout, (
        f"IndexSelect 3D layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "IndexSelect 3D output mismatch"


def test_index_select_negative_dim() -> None:
    """Test index_select with negative dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_index = to_device(torch.from_numpy(index_1d_np), _DEVICE_TYPE)
    standalone_output = torch.index_select(standalone_input, -1, standalone_index)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    idx_placements = (Replicate(), Replicate())
    dist_index = distribute_tensor(standalone_index, mesh, idx_placements)

    dist_output = torch.index_select(dist_input, -1, dist_index)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, (
        f"IndexSelect negative dim layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "IndexSelect negative dim output mismatch between standalone and distributed execution"


def test_index_select_2d_dim0() -> None:
    """Test index_select on dim 0 with 2D input."""
    init_backend(_DEVICE_TYPE)
    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_index = to_device(torch.from_numpy(index_1d_np), _DEVICE_TYPE)
    standalone_output = torch.index_select(standalone_input, 0, standalone_index)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Replicate(), Shard(1)))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 0, dist_index)

    expected_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
    assert dist_output.layout == expected_layout, "Layout mismatch for 2D dim 0 selection"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for 2D dim 0"


def test_index_select_2d_dim1() -> None:
    """Test index_select on dim 1 with 2D input."""
    init_backend(_DEVICE_TYPE)
    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_index = to_device(torch.from_numpy(index_1d_np), _DEVICE_TYPE)
    standalone_output = torch.index_select(standalone_input, 1, standalone_index)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Shard(0), Replicate()))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 1, dist_index)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, "Layout mismatch for 2D dim 1 selection"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for 2D dim 1"


def test_index_select_3d_dim1() -> None:
    """Test index_select on 3D tensor dim 1."""
    init_backend(_DEVICE_TYPE)
    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    standalone_index = to_device(torch.from_numpy(index_1d_np), _DEVICE_TYPE)
    standalone_output = torch.index_select(standalone_input, 1, standalone_index)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Shard(0), Replicate(), Shard(1)))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 1, dist_index)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 3)
    assert dist_output.layout == expected_layout, "Layout mismatch for 3D tensor"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for 3D tensor"


def test_index_select_single_element() -> None:
    """Test index_select with a single-element index."""
    init_backend(_DEVICE_TYPE)
    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    single_index_np = np.array([2], dtype=np.int64)
    standalone_index = to_device(torch.from_numpy(single_index_np), _DEVICE_TYPE)
    standalone_output = torch.index_select(standalone_input, 0, standalone_index)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Replicate(), Shard(1)))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 0, dist_index)

    expected_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
    assert dist_output.layout == expected_layout, "Layout mismatch for single-element index"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for single element"


def test_index_select_sharded_dim0_2d() -> None:
    """Test index_select on a sharded dimension (dim 0)."""
    init_backend(_DEVICE_TYPE)
    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_index = to_device(torch.from_numpy(index_1d_np), _DEVICE_TYPE)
    standalone_output = torch.index_select(standalone_input, 0, standalone_index)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Shard(0), Replicate()))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 0, dist_index)

    expected_layout = _build_layout(mesh, (Partial(), Replicate()), 2)
    assert dist_output.layout == expected_layout, "Layout mismatch for 2D sharded dim 0 selection"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for sharded dim 0"


def test_index_select_sharded_dim1_2d() -> None:
    """Test index_select on a sharded dimension (dim 1)."""
    init_backend(_DEVICE_TYPE)
    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_index = to_device(torch.from_numpy(np.array([1, 2], dtype=np.int64)), _DEVICE_TYPE)
    standalone_output = torch.index_select(standalone_input, 1, standalone_index)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Replicate(), Shard(1)))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 1, dist_index)

    expected_layout = _build_layout(mesh, (Replicate(), Partial()), 2)
    assert dist_output.layout == expected_layout, "Layout mismatch for 2D sharded dim 1 selection"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for sharded dim 1"


def test_index_select_sharded_dim2_3d() -> None:
    """Test index_select on a sharded 3D tensor dim 2."""
    init_backend(_DEVICE_TYPE)
    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    standalone_index = to_device(torch.from_numpy(index_1d_np), _DEVICE_TYPE)
    standalone_output = torch.index_select(standalone_input, 2, standalone_index)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Shard(0), Replicate(), Shard(1)))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 2, dist_index)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
    assert dist_output.layout == expected_layout, "Layout mismatch for 3D sharded dim 2 selection"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for 3D sharded dim 2"


def test_index_select_duplicate_indices_sharded() -> None:
    """Test index_select with duplicate indices on sharded dimension."""
    init_backend(_DEVICE_TYPE)
    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    dup_index = np.array([1, 1, 3, 2, 1], dtype=np.int64)
    standalone_index = to_device(torch.from_numpy(dup_index), _DEVICE_TYPE)
    standalone_output = torch.index_select(standalone_input, 0, standalone_index)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Shard(0), Replicate()))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 0, dist_index)

    expected_layout = _build_layout(mesh, (Partial(), Replicate()), 2)
    assert dist_output.layout == expected_layout, "Layout mismatch for duplicate indices"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for duplicate indices"


def test_index_select_out_of_order_sharded() -> None:
    """Test index_select with out-of-order indices on sharded dimension."""
    init_backend(_DEVICE_TYPE)
    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    ooo_index = np.array([6, 1, 7, 0, 3], dtype=np.int64)
    standalone_index = to_device(torch.from_numpy(ooo_index), _DEVICE_TYPE)
    standalone_output = torch.index_select(standalone_input, 0, standalone_index)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Shard(0), Replicate()))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 0, dist_index)

    expected_layout = _build_layout(mesh, (Partial(), Replicate()), 2)
    assert dist_output.layout == expected_layout, "Layout mismatch for out-of-order indices"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for out-of-order indices"


def test_index_select_fully_replicated() -> None:
    """Test index_select on fully replicated tensor."""
    init_backend(_DEVICE_TYPE)
    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_index = to_device(torch.from_numpy(index_1d_np), _DEVICE_TYPE)
    standalone_output = torch.index_select(standalone_input, 0, standalone_index)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Replicate(), Replicate()))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 0, dist_index)

    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    assert dist_output.layout == expected_layout, "Layout mismatch for fully replicated selection"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for replicated selection"


def test_index_select_negative_sharded_dim() -> None:
    """Test index_select with negative dim on sharded axis."""
    init_backend(_DEVICE_TYPE)
    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_index = to_device(torch.from_numpy(np.array([1, 2], dtype=np.int64)), _DEVICE_TYPE)
    standalone_output = torch.index_select(standalone_input, -1, standalone_index)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Replicate(), Shard(1)))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, -1, dist_index)

    expected_layout = _build_layout(mesh, (Replicate(), Partial()), 2)
    assert dist_output.layout == expected_layout, "Layout mismatch for negative dim sharded selection"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), (
        "Numerical mismatch for negative dim sharded selection"
    )
