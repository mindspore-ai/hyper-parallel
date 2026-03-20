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
"""test torch dtensor with distributed cat"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

# Set seed for reproducibility
np.random.seed(42)
t1_np = np.random.randn(8, 16).astype(np.float32)
t2_np = np.random.randn(8, 16).astype(np.float32)
t3_np = np.random.randn(4, 8, 16).astype(np.float32)
t4_np = np.random.randn(8, 16).astype(np.float32)
t5_np = np.random.randn(8, 16).astype(np.float32)
t6_np = np.random.randn(8, 16).astype(np.float32)
t_diff1_np = np.random.randn(4, 8).astype(np.float32)
t_diff2_np = np.random.randn(4, 16).astype(np.float32)
t_empty_np = np.zeros((8, 0), dtype=np.float32)
t4d_1_np = np.random.randn(4, 4, 8, 8).astype(np.float32)
t4d_2_np = np.random.randn(4, 4, 16, 8).astype(np.float32)
t_2d_sharded_1_np = np.random.randn(8, 16).astype(np.float32)
t_2d_sharded_2_np = np.random.randn(8, 16).astype(np.float32)
t_5d_1_np = np.random.randn(2, 4, 8, 16, 32).astype(np.float32)
t_5d_2_np = np.random.randn(2, 4, 8, 16, 32).astype(np.float32)
t_single_1_np = np.random.randn(8, 1).astype(np.float32)
t_single_2_np = np.random.randn(8, 1).astype(np.float32)
t_1d_1_np = np.random.randn(16).astype(np.float32)
t_1d_2_np = np.random.randn(16).astype(np.float32)

def test_distributed_cat_basic():
    """
    Feature: dtensor + torch.cat basic alignment
    Description:
        - Concatenate two tensors sharded on dim=0 ("dp").
        - Input: two tensors of shape (8, 16) sharded on dim=0.
        - Concatenate along dim=1 (unsharded dimension).
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference
    t1 = torch.from_numpy(t1_np).npu()
    t2 = torch.from_numpy(t2_np).npu()
    standalone_output = torch.cat([t1, t2], dim=1)  # Expected shape (8, 32)

    # Distributed setup: 8-card mesh (2x4)
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_t1 = distribute_tensor(t1, mesh, placements)
    dist_t2 = distribute_tensor(t2, mesh, placements)

    # Perform distributed cat
    dist_output = torch.cat([dist_t1, dist_t2], dim=1)

    # Layout validation
    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout

    # Numerical validation
    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch in basic cat"



def test_distributed_cat_3d_complex():
    """
    Feature: dtensor + torch.cat 3D complex mesh
    Description:
        - Input: (4, 8, 16) sharded on dim=0 ("dp") and dim=1 ("tp").
        - Concatenate along dim=2 (replicated dimension).
    Expectation: Success with multi-axis sharding preserved.
    """
    init_dist()

    t3 = torch.from_numpy(t3_np).npu()
    standalone_output = torch.cat([t3, t3], dim=2)

    # Mesh 2x2x2
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "tp", "mp"))
    # tensor_map = (2, 1, -1) -> Sharded on dp and tp
    placements = (Shard(0), Shard(1), Replicate())

    dist_t3 = distribute_tensor(t3, mesh, placements)
    dist_output = torch.cat([dist_t3, dist_t3], dim=2)

    expected_layout = _build_layout(mesh, placements, 3)
    assert dist_output.layout == expected_layout

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch in 3D complex cat"


def test_distributed_cat_multiple_tensors():
    """
    Feature: dtensor + torch.cat with multiple (>2) tensors.
    Description:
        - Input: three tensors of shape (8, 16).
        - Concatenate along dim=1.
    Expectation: Success with correct output layout and numerical results.
    """
    init_dist()

    t1 = torch.from_numpy(t4_np).npu()
    t2 = torch.from_numpy(t5_np).npu()
    t3 = torch.from_numpy(t6_np).npu()
    standalone_output = torch.cat([t1, t2, t3], dim=1)

    # 8-card mesh
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    placements = (Shard(0),)

    dist_t1 = distribute_tensor(t1, mesh, placements)
    dist_t2 = distribute_tensor(t2, mesh, placements)
    dist_t3 = distribute_tensor(t3, mesh, placements)

    dist_output = torch.cat([dist_t1, dist_t2, dist_t3], dim=1)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch in multiple tensor cat"


def test_distributed_cat_mismatched_shapes():
    """
    Feature: dtensor + torch.cat with tensors of different shapes in the concat dimension.
    Description:
        - Input 1: (4, 8), Input 2: (4, 16).
        - Concatenate along dim=1.
        - Sharded along dim=0.
    Expectation: Success.
    """
    init_dist()

    t1 = torch.from_numpy(t_diff1_np).npu()
    t2 = torch.from_numpy(t_diff2_np).npu()
    standalone_output = torch.cat([t1, t2], dim=1)

    # 4-card mesh (2x2)
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_t1 = distribute_tensor(t1, mesh, placements)
    dist_t2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.cat([dist_t1, dist_t2], dim=1)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch in mismatched shape cat"



def test_distributed_cat_with_empty():
    """
    Feature: dtensor + torch.cat with an empty tensor.
    Description:
        - Input 1: (8, 16), Input 2: (8, 0).
        - Concatenate along dim=1.
        - Sharded on dim=0.
    Expectation: Success.
    """
    init_dist()

    t1 = torch.from_numpy(t4_np).npu()
    t2 = torch.from_numpy(t_empty_np).npu()
    standalone_output = torch.cat([t1, t2], dim=1)

    # 4-card mesh
    mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("dp",))
    placements = (Shard(0),)

    dist_t1 = distribute_tensor(t1, mesh, placements)
    dist_t2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.cat([dist_t1, dist_t2], dim=1)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch in empty tensor cat"


def test_distributed_cat_4d_tensor():
    """
    Feature: dtensor + torch.cat for 4D tensors.
    Description:
        - Input: 4D tensors sharded on dim=0 and dim=3.
        - Concatenate along dim=2.
    Expectation: Success.
    """
    init_dist()

    t1 = torch.from_numpy(t4d_1_np).npu()
    t2 = torch.from_numpy(t4d_2_np).npu()
    standalone_output = torch.cat([t1, t2], dim=2)

    # 8-card mesh (2x4) -> 2D mesh
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # placements length must equal mesh.ndim (which is 2 here).
    # Mesh dim 0 ("dp") shards tensor dim 0 -> Shard(0)
    # Mesh dim 1 ("tp") shards tensor dim 3 -> Shard(3)
    placements = (Shard(0), Shard(3))

    dist_t1 = distribute_tensor(t1, mesh, placements)
    dist_t2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.cat([dist_t1, dist_t2], dim=2)

    # _build_layout takes tensor_dim=4 as the third parameter
    expected_layout = _build_layout(mesh, placements, 4)
    assert dist_output.layout == expected_layout

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch in 4D tensor cat"



def test_distributed_cat_5d_mixed_placements():
    """
    Feature: dtensor + torch.cat for 5D tensors with mixed placements.
    Description:
        - Input: 5D tensors of shape (2, 4, 8, 16, 32).
        - Mesh: 2x4, placements: Replicate() on mesh dim 0, Shard(3) on mesh dim 1.
        - Concatenate along dim=4.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    t1 = torch.from_numpy(t_5d_1_np).npu()
    t2 = torch.from_numpy(t_5d_2_np).npu()
    standalone_output = torch.cat([t1, t2], dim=4)

    # 8-card mesh (2x4)
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Shard(3))

    dist_t1 = distribute_tensor(t1, mesh, placements)
    dist_t2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.cat([dist_t1, dist_t2], dim=4)

    expected_layout = _build_layout(mesh, placements, 5)
    assert dist_output.layout == expected_layout

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch in 5D mixed placements cat"


def test_distributed_cat_shard_last_cat_first():
    """
    Feature: dtensor + torch.cat sharding the last dimension but concatenating the first.
    Description:
        - Input: shape (8, 16).
        - Sharded on dim=1.
        - Concatenate along dim=0.
    Expectation: Success.
    """
    init_dist()

    t1 = torch.from_numpy(t1_np).npu()
    t2 = torch.from_numpy(t2_np).npu()
    standalone_output = torch.cat([t1, t2], dim=0)

    # 8-card mesh (8,)
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    placements = (Shard(1),)

    dist_t1 = distribute_tensor(t1, mesh, placements)
    dist_t2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.cat([dist_t1, dist_t2], dim=0)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch in shard last cat first"


def test_distributed_cat_singleton_dimension():
    """
    Feature: dtensor + torch.cat on singleton dimensions.
    Description:
        - Input: shape (8, 1).
        - Sharded on dim=0.
        - Concatenate along dim=1.
    Expectation: Success.
    """
    init_dist()

    t1 = torch.from_numpy(t_single_1_np).npu()
    t2 = torch.from_numpy(t_single_2_np).npu()
    standalone_output = torch.cat([t1, t2], dim=1)

    # 4-card mesh (4,)
    mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("dp",))
    placements = (Shard(0),)

    dist_t1 = distribute_tensor(t1, mesh, placements)
    dist_t2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.cat([dist_t1, dist_t2], dim=1)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), "Numerical mismatch in singleton dimension cat"
