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
from tests.torch.utils import init_backend, to_device
from tests.torch.shard.utils import local_to_global

try:
    import torch_npu  # pylint: disable=W0611
    _DEVICE_TYPE = "npu"
except ImportError:
    _DEVICE_TYPE = "cpu"

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


def test_cat_basic() -> None:
    """Test torch.cat basic alignment."""
    init_backend(_DEVICE_TYPE)

    t1 = to_device(torch.from_numpy(t1_np), _DEVICE_TYPE)
    t2 = to_device(torch.from_numpy(t2_np), _DEVICE_TYPE)
    standalone_output = torch.cat([t1, t2], dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_t1 = distribute_tensor(t1, mesh, placements)
    dist_t2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.cat([dist_t1, dist_t2], dim=1)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"Cat basic layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), (
        "Numerical mismatch in basic cat"
    )


def test_cat_3d_complex() -> None:
    """Test torch.cat with 3D complex mesh."""
    init_backend(_DEVICE_TYPE)

    t3 = to_device(torch.from_numpy(t3_np), _DEVICE_TYPE)
    standalone_output = torch.cat([t3, t3], dim=2)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "tp", "mp"))
    placements = (Shard(0), Shard(1), Replicate())

    dist_t3 = distribute_tensor(t3, mesh, placements)
    dist_output = torch.cat([dist_t3, dist_t3], dim=2)

    expected_layout = _build_layout(mesh, placements, 3)
    assert dist_output.layout == expected_layout, (
        f"Cat 3D layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), (
        "Numerical mismatch in 3D complex cat"
    )


def test_cat_multiple_tensors() -> None:
    """Test torch.cat with multiple tensors."""
    init_backend(_DEVICE_TYPE)

    t1 = to_device(torch.from_numpy(t4_np), _DEVICE_TYPE)
    t2 = to_device(torch.from_numpy(t5_np), _DEVICE_TYPE)
    t3 = to_device(torch.from_numpy(t6_np), _DEVICE_TYPE)
    standalone_output = torch.cat([t1, t2, t3], dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(8,), mesh_dim_names=("dp",))
    placements = (Shard(0),)

    dist_t1 = distribute_tensor(t1, mesh, placements)
    dist_t2 = distribute_tensor(t2, mesh, placements)
    dist_t3 = distribute_tensor(t3, mesh, placements)

    dist_output = torch.cat([dist_t1, dist_t2, dist_t3], dim=1)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"Cat multiple layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), (
        "Numerical mismatch in multiple tensor cat"
    )


def test_cat_mismatched_shapes() -> None:
    """Test torch.cat with mismatched shapes."""
    init_backend(_DEVICE_TYPE)

    t1 = to_device(torch.from_numpy(t_diff1_np), _DEVICE_TYPE)
    t2 = to_device(torch.from_numpy(t_diff2_np), _DEVICE_TYPE)
    standalone_output = torch.cat([t1, t2], dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_t1 = distribute_tensor(t1, mesh, placements)
    dist_t2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.cat([dist_t1, dist_t2], dim=1)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"Cat mismatched layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), (
        "Numerical mismatch in mismatched shape cat"
    )


def test_cat_with_empty() -> None:
    """Test torch.cat with empty tensor."""
    init_backend(_DEVICE_TYPE)

    t1 = to_device(torch.from_numpy(t4_np), _DEVICE_TYPE)
    t2 = to_device(torch.from_numpy(t_empty_np), _DEVICE_TYPE)
    standalone_output = torch.cat([t1, t2], dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(4,), mesh_dim_names=("dp",))
    placements = (Shard(0),)

    dist_t1 = distribute_tensor(t1, mesh, placements)
    dist_t2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.cat([dist_t1, dist_t2], dim=1)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"Cat empty layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), (
        "Numerical mismatch in empty tensor cat"
    )


def test_cat_4d_tensor() -> None:
    """Test torch.cat with 4D tensors."""
    init_backend(_DEVICE_TYPE)

    t1 = to_device(torch.from_numpy(t4d_1_np), _DEVICE_TYPE)
    t2 = to_device(torch.from_numpy(t4d_2_np), _DEVICE_TYPE)
    standalone_output = torch.cat([t1, t2], dim=2)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    placements = (Shard(0), Shard(3))

    dist_t1 = distribute_tensor(t1, mesh, placements)
    dist_t2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.cat([dist_t1, dist_t2], dim=2)

    expected_layout = _build_layout(mesh, placements, 4)
    assert dist_output.layout == expected_layout, (
        f"Cat 4D layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), (
        "Numerical mismatch in 4D tensor cat"
    )


def test_cat_5d_mixed_placements() -> None:
    """Test torch.cat with 5D mixed placements."""
    init_backend(_DEVICE_TYPE)

    t1 = to_device(torch.from_numpy(t_5d_1_np), _DEVICE_TYPE)
    t2 = to_device(torch.from_numpy(t_5d_2_np), _DEVICE_TYPE)
    standalone_output = torch.cat([t1, t2], dim=4)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Shard(3))

    dist_t1 = distribute_tensor(t1, mesh, placements)
    dist_t2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.cat([dist_t1, dist_t2], dim=4)

    expected_layout = _build_layout(mesh, placements, 5)
    assert dist_output.layout == expected_layout, (
        f"Cat 5D layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), (
        "Numerical mismatch in 5D mixed placements cat"
    )


def test_cat_shard_last_cat_first() -> None:
    """Test torch.cat sharding last dim, concatenating first."""
    init_backend(_DEVICE_TYPE)

    t1 = to_device(torch.from_numpy(t1_np), _DEVICE_TYPE)
    t2 = to_device(torch.from_numpy(t2_np), _DEVICE_TYPE)
    standalone_output = torch.cat([t1, t2], dim=0)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(8,), mesh_dim_names=("dp",))
    placements = (Shard(1),)

    dist_t1 = distribute_tensor(t1, mesh, placements)
    dist_t2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.cat([dist_t1, dist_t2], dim=0)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"Cat shard-last layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), (
        "Numerical mismatch in shard last cat first"
    )


def test_cat_singleton_dimension() -> None:
    """Test torch.cat on singleton dimensions."""
    init_backend(_DEVICE_TYPE)

    t1 = to_device(torch.from_numpy(t_single_1_np), _DEVICE_TYPE)
    t2 = to_device(torch.from_numpy(t_single_2_np), _DEVICE_TYPE)
    standalone_output = torch.cat([t1, t2], dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(4,), mesh_dim_names=("dp",))
    placements = (Shard(0),)

    dist_t1 = distribute_tensor(t1, mesh, placements)
    dist_t2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.cat([dist_t1, dist_t2], dim=1)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"Cat singleton layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), (
        "Numerical mismatch in singleton dimension cat"
    )
