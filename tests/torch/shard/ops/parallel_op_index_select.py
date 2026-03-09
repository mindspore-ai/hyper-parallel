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
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
standalone_input_2d_np = np.random.randn(8, 4).astype(np.float32)
standalone_input_3d_np = np.random.randn(4, 6, 8).astype(np.float32)
index_1d_np = np.array([1, 3], dtype=np.int64)


def test_distributed_index_select_basic():
    """
    Feature: dtensor + torch.index_select basic selection
    Description:
        - Perform index_select on an unsharded dimension while preserving sharding on other dimensions.
        - Input: shape (8, 4) sharded on dim=0, index_select on dim=1.
        - Index: shape (2,) fully replicated.
        - Output layout should preserve sharding on dim=0, unsharded on selected dim=1.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_index = torch.from_numpy(index_1d_np).npu()
    standalone_output = torch.index_select(standalone_input, 1, standalone_index)

    # Distributed setup: shard dim=0 ("dp"), keep dim=1 unsharded (dimension to select)
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    # Index tensor must be fully replicated
    idx_placements = (Replicate(), Replicate())
    dist_index = distribute_tensor(standalone_index, mesh, idx_placements)

    dist_output = torch.index_select(dist_input, 1, dist_index)

    # Layout validation: dim0 preserved (sharded), dim1 selected (unsharded)
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"IndexSelect output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation via gathering (local back to global)
    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "IndexSelect output mismatch between standalone and distributed execution"


def test_distributed_index_select_3d():
    """
    Feature: dtensor + torch.index_select 3D tensor
    Description:
        - Perform index_select on a 3D tensor where multiple dimensions are sharded.
        - Input: shape (4, 6, 8) sharded on dim=0 and dim=2, index_select on unsharded dim=1.
        - Output should preserve sharding on dim=0 and dim=2.
    Expectation: Success with correct layout propagation.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_3d_np).npu()
    standalone_index = torch.from_numpy(index_1d_np).npu()
    standalone_output = torch.index_select(standalone_input, 1, standalone_index)

    # Shard dim=0 ("dp") and dim=2 ("tp"), keep dim=1 unsharded
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Shard(1))
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    idx_placements = (Replicate(), Replicate())
    dist_index = distribute_tensor(standalone_index, mesh, idx_placements)

    dist_output = torch.index_select(dist_input, 1, dist_index)

    # Layout validation
    expected_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 3)
    assert dist_output.layout == expected_layout, \
        f"IndexSelect 3D layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "IndexSelect 3D output mismatch"


def test_distributed_index_select_negative_dim():
    """
    Feature: dtensor + torch.index_select with negative dimension
    Description:
        - Perform index_select using a negative dimension index (e.g., dim=-1).
        - Input: shape (8, 4) sharded on dim=0, index_select on dim=-1 (which maps to dim=1).
        - Index: shape (2,) fully replicated.
        - Output should preserve sharding on dim=0 and correctly resolve the negative dimension.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference using dim=-1
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_index = torch.from_numpy(index_1d_np).npu()
    standalone_output = torch.index_select(standalone_input, -1, standalone_index)

    # Distributed setup
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    idx_placements = (Replicate(), Replicate())
    dist_index = distribute_tensor(standalone_index, mesh, idx_placements)

    # Perform distributed index_select with dim=-1
    dist_output = torch.index_select(dist_input, -1, dist_index)

    # Layout validation: dim0 preserved, dim1 selected (resolved from -1)
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"IndexSelect negative dim layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation via gathering
    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "IndexSelect negative dim output mismatch between standalone and distributed execution"



def test_distributed_index_select_2d_dim0():
    """
    Feature: dtensor + torch.index_select on dim 0
    Description:
        - Input: shape (8, 4) sharded on dim=1.
        - Index_select on dim=0 (unsharded dimension).
        - Index: fully replicated.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_index = torch.from_numpy(index_1d_np).npu()
    standalone_output = torch.index_select(standalone_input, 0, standalone_index)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Replicate(), Shard(1)))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 0, dist_index)

    expected_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
    assert dist_output.layout == expected_layout, "Layout mismatch for 2D dim 0 selection"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for 2D dim 0"

def test_distributed_index_select_2d_dim1():
    """
    Feature: dtensor + torch.index_select on dim 1
    Description:
        - Input: shape (8, 4) sharded on dim=0.
        - Index_select on dim=1 (unsharded dimension).
        - Index: fully replicated.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_index = torch.from_numpy(index_1d_np).npu()
    standalone_output = torch.index_select(standalone_input, 1, standalone_index)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Shard(0), Replicate()))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 1, dist_index)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, "Layout mismatch for 2D dim 1 selection"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for 2D dim 1"

def test_distributed_index_select_3d_dim1():
    """
    Feature: dtensor + torch.index_select on 3D tensor
    Description:
        - Input: shape (4, 6, 8) sharded on dim=0 and dim=2.
        - Index_select on dim=1 (unsharded dimension).
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_3d_np).npu()
    standalone_index = torch.from_numpy(index_1d_np).npu()
    standalone_output = torch.index_select(standalone_input, 1, standalone_index)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    # Shard dim 0 on 'dp' and dim 2 on 'tp'
    dist_input = distribute_tensor(standalone_input, mesh, (Shard(0), Replicate(), Shard(1)))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 1, dist_index)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 3)
    assert dist_output.layout == expected_layout, "Layout mismatch for 3D tensor"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for 3D tensor"


def test_distributed_index_select_single_element():
    """
    Feature: dtensor + torch.index_select with a single-element index
    Description:
        - Input: shape (8, 4) sharded on dim=1.
        - Index: length 1.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    single_index_np = np.array([2], dtype=np.int64)
    standalone_index = torch.from_numpy(single_index_np).npu()
    standalone_output = torch.index_select(standalone_input, 0, standalone_index)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Replicate(), Shard(1)))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 0, dist_index)

    expected_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
    assert dist_output.layout == expected_layout, "Layout mismatch for single-element index"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for single element"

def test_distributed_index_select_sharded_dim0_2d():
    """
    Feature: dtensor + torch.index_select on a sharded dimension
    Description:
        - Input: shape (8, 4) sharded on dim=0.
        - Index_select on dim=0 (which is the sharded dimension).
        - Index: fully replicated.
        - Output layout should replace the sharded dimension with a replicated one.
    Expectation: Success with correct layout and numerical equivalence after AllReduce.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_index = torch.from_numpy(index_1d_np).npu()
    standalone_output = torch.index_select(standalone_input, 0, standalone_index)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Shard(0), Replicate()))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 0, dist_index)

    # Sharding on dim 0 is eliminated because we selected across the sharded axis
    expected_layout = _build_layout(mesh, (Partial(), Replicate()), 2)
    assert dist_output.layout == expected_layout, "Layout mismatch for 2D sharded dim 0 selection"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for sharded dim 0"


def test_distributed_index_select_sharded_dim1_2d():
    """
    Feature: dtensor + torch.index_select on a sharded dimension
    Description:
        - Input: shape (8, 4) sharded on dim=1.
        - Index_select on dim=1 (which is the sharded dimension).
        - Index: fully replicated.
    Expectation: Success with correct layout and numerical equivalence after AllReduce.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_index = torch.from_numpy(np.array([1, 2], dtype=np.int64)).npu()
    standalone_output = torch.index_select(standalone_input, 1, standalone_index)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Replicate(), Shard(1)))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 1, dist_index)

    # Sharding on dim 1 is eliminated
    expected_layout = _build_layout(mesh, (Replicate(), Partial()), 2)
    assert dist_output.layout == expected_layout, "Layout mismatch for 2D sharded dim 1 selection"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for sharded dim 1"


def test_distributed_index_select_sharded_dim2_3d():
    """
    Feature: dtensor + torch.index_select on a sharded 3D tensor
    Description:
        - Input: shape (4, 6, 8) sharded on dim=0 and dim=2.
        - Index_select on dim=2 (sharded dimension).
        - Output should keep dim=0 sharded but replicate dim=2.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_3d_np).npu()
    standalone_index = torch.from_numpy(index_1d_np).npu()
    standalone_output = torch.index_select(standalone_input, 2, standalone_index)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Shard(0), Replicate(), Shard(1)))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 2, dist_index)

    # dim 0 keeps Shard(0), dim 2 loses Shard(1)
    expected_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
    assert dist_output.layout == expected_layout, "Layout mismatch for 3D sharded dim 2 selection"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for 3D sharded dim 2"


def test_distributed_index_select_duplicate_indices_sharded():
    """
    Feature: dtensor + torch.index_select with duplicate indices
    Description:
        - Input: shape (8, 4) sharded on dim=0.
        - Index_select on dim=0 using indices that repeat (e.g., [1, 1, 3, 2, 1]).
        - Tests if the masking and all-reduce logic correctly aggregates repeated indices.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    dup_index = np.array([1, 1, 3, 2, 1], dtype=np.int64)
    standalone_index = torch.from_numpy(dup_index).npu()
    standalone_output = torch.index_select(standalone_input, 0, standalone_index)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Shard(0), Replicate()))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 0, dist_index)

    expected_layout = _build_layout(mesh, (Partial(), Replicate()), 2)
    assert dist_output.layout == expected_layout, "Layout mismatch for duplicate indices"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for duplicate indices"


def test_distributed_index_select_out_of_order_sharded():
    """
    Feature: dtensor + torch.index_select with out-of-order indices
    Description:
        - Input: shape (8, 4) sharded on dim=0.
        - Index_select on dim=0 using non-monotonic indices across device boundaries.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    # Indices that cross device boundaries (e.g., 6 is on rank 1, 1 is on rank 0)
    ooo_index = np.array([6, 1, 7, 0, 3], dtype=np.int64)
    standalone_index = torch.from_numpy(ooo_index).npu()
    standalone_output = torch.index_select(standalone_input, 0, standalone_index)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Shard(0), Replicate()))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 0, dist_index)

    expected_layout = _build_layout(mesh, (Partial(), Replicate()), 2)
    assert dist_output.layout == expected_layout, "Layout mismatch for out-of-order indices"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for out-of-order indices"


def test_distributed_index_select_fully_replicated():
    """
    Feature: dtensor + torch.index_select on fully replicated tensor
    Description:
        - Input: shape (8, 4) fully replicated across all devices.
        - Triggers the fallback logic where `shard_mesh_dim_name == "None"`.
    Expectation: Success with correct layout and numerical equivalence without communication.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_index = torch.from_numpy(index_1d_np).npu()
    standalone_output = torch.index_select(standalone_input, 0, standalone_index)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Replicate(), Replicate()))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, 0, dist_index)

    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    assert dist_output.layout == expected_layout, "Layout mismatch for fully replicated selection"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for replicated selection"


def test_distributed_index_select_negative_sharded_dim():
    """
    Feature: dtensor + torch.index_select with negative dim on sharded axis
    Description:
        - Input: shape (8, 4) sharded on dim=1.
        - Index_select using negative dim=-1 (which maps to the sharded dim=1).
    Expectation: Success with negative dim resolving to the correct sharded axis.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_index = torch.from_numpy(np.array([1, 2], dtype=np.int64)).npu()
    standalone_output = torch.index_select(standalone_input, -1, standalone_index)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    dist_input = distribute_tensor(standalone_input, mesh, (Replicate(), Shard(1)))
    dist_index = distribute_tensor(standalone_index, mesh, (Replicate(), Replicate()))

    dist_output = torch.index_select(dist_input, -1, dist_index)

    expected_layout = _build_layout(mesh, (Replicate(), Partial()), 2)
    assert dist_output.layout == expected_layout, "Layout mismatch for negative dim sharded selection"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch for negative dim sharded selection"
