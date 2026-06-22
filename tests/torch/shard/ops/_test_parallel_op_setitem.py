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
"""test torch dtensor with distributed __setitem__"""
import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


np.random.seed(42)
_standalone_2d = np.random.randn(8, 5).astype(np.float32)
_standalone_2d_shardable = np.random.randn(8, 6).astype(np.float32)


def test_setitem_scalar() -> None:
    """Test __setitem__ with scalar value on replicated tensor."""
    init_backend(_DEVICE_TYPE)

    # Standalone: x[1:3] = 0.0
    standalone_input = to_device(torch.from_numpy(_standalone_2d.copy()), _DEVICE_TYPE)
    standalone_ref = standalone_input.clone()
    standalone_ref[1:3] = 0.0

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    # __setitem__ is in-place; the return value is None
    dist_input[1:3] = 0.0

    # Gather the modified distributed tensor and compare
    gathered_output = local_to_global(dist_input)
    assert torch.equal(standalone_ref, gathered_output), (
        f"Setitem scalar numerical mismatch: standalone={standalone_ref}, parallel={gathered_output}"
    )


def test_setitem_tensor_replicated() -> None:
    """Test __setitem__ with tensor value on replicated tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d.copy()), _DEVICE_TYPE)
    value = to_device(torch.ones(2, 5), _DEVICE_TYPE)

    standalone_ref = standalone_input.clone()
    standalone_ref[1:3] = value

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_input[1:3] = value

    gathered_output = local_to_global(dist_input)
    assert torch.equal(standalone_ref, gathered_output), (
        f"Setitem tensor numerical mismatch: standalone={standalone_ref}, parallel={gathered_output}"
    )


def test_setitem_shard_kept_dim() -> None:
    """Test __setitem__ on slice that doesn't touch the sharded dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d.copy()), _DEVICE_TYPE)
    value = to_device(torch.zeros(8, 2), _DEVICE_TYPE)

    standalone_ref = standalone_input.clone()
    standalone_ref[:, 1:3] = value

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    # Shard dim0, keep dim1 replicate; index dim1 (not the sharded dim)
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_input[:, 1:3] = value

    gathered_output = local_to_global(dist_input)
    assert torch.equal(standalone_ref, gathered_output), (
        f"Setitem shard kept dim numerical mismatch: standalone={standalone_ref}, parallel={gathered_output}"
    )


def test_setitem_advanced() -> None:
    """Test __setitem__ with advanced indexing on replicated tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d.copy()), _DEVICE_TYPE)
    value = to_device(torch.ones(2, 5), _DEVICE_TYPE)

    standalone_ref = standalone_input.clone()
    standalone_ref[[0, 2]] = value

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_input[[0, 2]] = value

    gathered_output = local_to_global(dist_input)
    assert torch.equal(standalone_ref, gathered_output), (
        f"Setitem advanced numerical mismatch: standalone={standalone_ref}, parallel={gathered_output}"
    )


def test_setitem_view_inplace_zero_() -> None:
    """Test that basic-indexing view in-place mutation propagates to source tensor (replicated)."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d.copy()), _DEVICE_TYPE)
    # Standalone: y = x[1:3]; y.zero_()
    standalone_ref = standalone_input.clone()
    y_ref = standalone_ref[1:3]
    y_ref.zero_()

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    y_dist = dist_input[1:3]
    y_dist.zero_()

    gathered_output = local_to_global(dist_input)
    assert torch.equal(standalone_ref, gathered_output), (
        f"View inplace zero_ numerical mismatch: standalone={standalone_ref}, parallel={gathered_output}"
    )


def test_setitem_view_inplace_add_() -> None:
    """Test that basic-indexing view in-place add propagates to source tensor (shard dim1)."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d_shardable.copy()), _DEVICE_TYPE)
    standalone_ref = standalone_input.clone()
    standalone_ref[1:3].add_(100)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    # Shard dim1; index dim0 (not sharded) → view keeps sharding on dim1
    x_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_input[1:3].add_(100)

    gathered_output = local_to_global(dist_input)
    assert torch.equal(standalone_ref, gathered_output), (
        f"View inplace add_ numerical mismatch: standalone={standalone_ref}, parallel={gathered_output}"
    )


def test_setitem_int_view_inplace_zero_() -> None:
    """Test that int-indexing view in-place mutation propagates to source tensor (replicated)."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d.copy()), _DEVICE_TYPE)
    standalone_ref = standalone_input.clone()
    standalone_ref[2].zero_()

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_input[2].zero_()

    gathered_output = local_to_global(dist_input)
    assert torch.equal(standalone_ref, gathered_output), (
        f"Int view inplace zero_ numerical mismatch: standalone={standalone_ref}, parallel={gathered_output}"
    )


def test_setitem_int_view_inplace_add_() -> None:
    """Test that int-indexing view in-place add propagates to source tensor (shard dim1)."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d_shardable.copy()), _DEVICE_TYPE)
    standalone_ref = standalone_input.clone()
    standalone_ref[2].add_(100)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    # Shard dim1; index dim0 (not sharded) → view keeps sharding on dim1
    x_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_input[2].add_(100)

    gathered_output = local_to_global(dist_input)
    assert torch.equal(standalone_ref, gathered_output), (
        f"Int view inplace add_ numerical mismatch: standalone={standalone_ref}, parallel={gathered_output}"
    )


def test_setitem_broadcast_tensor_value() -> None:
    """Test __setitem__ with tensor value that needs broadcasting (plain Tensor)."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d.copy()), _DEVICE_TYPE)
    # value shape (1, 5) broadcasts to LHS shape (2, 5)
    value = to_device(torch.ones(1, 5), _DEVICE_TYPE)

    standalone_ref = standalone_input.clone()
    standalone_ref[1:3] = value

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_input[1:3] = value

    gathered_output = local_to_global(dist_input)
    assert torch.equal(standalone_ref, gathered_output), (
        f"Broadcast tensor value numerical mismatch: standalone={standalone_ref}, parallel={gathered_output}"
    )


def test_setitem_broadcast_tensor_shard_kept() -> None:
    """Test __setitem__ with broadcast tensor value when LHS has sharded dim."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d.copy()), _DEVICE_TYPE)
    # value shape (8, 1) broadcasts to LHS shape (8, 2), dim0 is sharded
    value = to_device(torch.zeros(8, 1), _DEVICE_TYPE)

    standalone_ref = standalone_input.clone()
    standalone_ref[:, 1:3] = value

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_input[:, 1:3] = value

    gathered_output = local_to_global(dist_input)
    assert torch.equal(standalone_ref, gathered_output), (
        f"Broadcast tensor shard kept numerical mismatch: standalone={standalone_ref}, parallel={gathered_output}"
    )


def test_setitem_global_tensor_dim0_shard() -> None:
    """Test __setitem__ with global tensor value on dim0-sharded self.

    self: (8, 5) shard dim0, key: [:, 1:3], value: (8, 2) global.
    """
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d.copy()), _DEVICE_TYPE)
    value = to_device(torch.ones(8, 2), _DEVICE_TYPE)

    standalone_ref = standalone_input.clone()
    standalone_ref[:, 1:3] = value

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_input[:, 1:3] = value

    gathered_output = local_to_global(dist_input)
    assert torch.equal(standalone_ref, gathered_output), (
        f"Global tensor dim0 shard numerical mismatch: standalone={standalone_ref}, parallel={gathered_output}"
    )


def test_setitem_broadcast_tensor_1d_dim0_shard() -> None:
    """Test __setitem__ with 1-d broadcast tensor value on dim0-sharded self.

    self: (8, 5) shard dim0, key: [:, 1:3], value: (2,) broadcasts to (8, 2).
    """
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d.copy()), _DEVICE_TYPE)
    value = to_device(torch.ones(2), _DEVICE_TYPE)

    standalone_ref = standalone_input.clone()
    standalone_ref[:, 1:3] = value

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_input[:, 1:3] = value

    gathered_output = local_to_global(dist_input)
    assert torch.equal(standalone_ref, gathered_output), (
        f"Broadcast 1d dim0 shard numerical mismatch: standalone={standalone_ref}, parallel={gathered_output}"
    )


def test_setitem_broadcast_tensor_2d_dim0_shard() -> None:
    """Test __setitem__ with (1, 2) broadcast tensor value on dim0-sharded self.

    self: (8, 5) shard dim0, key: [:, 1:3], value: (1, 2) broadcasts to (8, 2).
    """
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d.copy()), _DEVICE_TYPE)
    value = to_device(torch.ones(1, 2), _DEVICE_TYPE)

    standalone_ref = standalone_input.clone()
    standalone_ref[:, 1:3] = value

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_input[:, 1:3] = value

    gathered_output = local_to_global(dist_input)
    assert torch.equal(standalone_ref, gathered_output), (
        f"Broadcast 2d dim0 shard numerical mismatch: standalone={standalone_ref}, parallel={gathered_output}"
    )


def test_setitem_global_tensor_dim1_shard() -> None:
    """Test __setitem__ with global tensor value on dim1-sharded self.

    self: (8, 6) shard dim1, key: [1:3, :], value: (2, 6) global.
    """
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d_shardable.copy()), _DEVICE_TYPE)
    value = to_device(torch.ones(2, 6), _DEVICE_TYPE)

    standalone_ref = standalone_input.clone()
    standalone_ref[1:3, :] = value

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_input[1:3, :] = value

    gathered_output = local_to_global(dist_input)
    assert torch.equal(standalone_ref, gathered_output), (
        f"Global tensor dim1 shard numerical mismatch: standalone={standalone_ref}, parallel={gathered_output}"
    )


def test_setitem_broadcast_tensor_dim1_shard() -> None:
    """Test __setitem__ with broadcast tensor value on dim1-sharded self.

    self: (8, 6) shard dim1, key: [1:3, :], value: (2, 1) broadcasts to (2, 6).
    """
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d_shardable.copy()), _DEVICE_TYPE)
    value = to_device(torch.ones(2, 1), _DEVICE_TYPE)

    standalone_ref = standalone_input.clone()
    standalone_ref[1:3, :] = value

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_input[1:3, :] = value

    gathered_output = local_to_global(dist_input)
    assert torch.equal(standalone_ref, gathered_output), (
        f"Broadcast dim1 shard numerical mismatch: standalone={standalone_ref}, parallel={gathered_output}"
    )
