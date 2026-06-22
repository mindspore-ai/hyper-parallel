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
"""test torch dtensor with distributed __getitem__"""
import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


np.random.seed(42)
_standalone_2d = np.random.randn(8, 5).astype(np.float32)
_standalone_3d = np.random.randn(8, 5, 4).astype(np.float32)
_standalone_2d_shardable = np.random.randn(8, 6).astype(np.float32)


def test_getitem_basic_int_replicated() -> None:
    """Test __getitem__ with int index on fully replicated tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d), _DEVICE_TYPE)
    standalone_output = standalone_input[2]

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input[2]

    expected_layout = _build_layout(mesh, (Replicate(),), 1)
    assert dist_output.layout == expected_layout, (
        f"Int index layout mismatch: expected {expected_layout}, got {dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"Int index numerical mismatch: standalone={standalone_output}, parallel={gathered_output}"
    )


def test_getitem_basic_slice_keep_dim() -> None:
    """Test __getitem__ with slice keeping sharded dim."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d), _DEVICE_TYPE)
    standalone_output = standalone_input[:, 1:3]

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input[:, 1:3]

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, (
        f"Slice keep dim layout mismatch: expected {expected_layout}, got {dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"Slice keep dim numerical mismatch: standalone={standalone_output}, parallel={gathered_output}"
    )


def test_getitem_basic_newaxis() -> None:
    """Test __getitem__ with None (newaxis) insertion."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d), _DEVICE_TYPE)
    standalone_output = standalone_input[:, None, :]

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input[:, None, :]

    expected_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
    assert dist_output.layout == expected_layout, (
        f"Newaxis layout mismatch: expected {expected_layout}, got {dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"Newaxis numerical mismatch: standalone={standalone_output}, parallel={gathered_output}"
    )


def test_getitem_basic_ellipsis() -> None:
    """Test __getitem__ with Ellipsis expansion."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d), _DEVICE_TYPE)
    standalone_output = standalone_input[..., 1:3]

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input[..., 1:3]

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, (
        f"Ellipsis layout mismatch: expected {expected_layout}, got {dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"Ellipsis numerical mismatch: standalone={standalone_output}, parallel={gathered_output}"
    )


def test_getitem_advanced_single_list() -> None:
    """Test __getitem__ with list advanced indexing on replicated tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d), _DEVICE_TYPE)
    standalone_output = standalone_input[[0, 2, 1]]

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input[[0, 2, 1]]

    # Advanced index on dim0, dim1 kept → output has index shape (3,) + dim1 shape (5,) = (3, 5)
    expected_ndim = 2
    assert len(dist_output.layout.tensor_map) == expected_ndim, (
        f"Advanced index output ndim mismatch: expected {expected_ndim}, "
        f"got {len(dist_output.layout.tensor_map)}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"Advanced index numerical mismatch: standalone={standalone_output}, parallel={gathered_output}"
    )


def test_getitem_basic_tuple_int() -> None:
    """Test __getitem__ with tuple-of-ints key on a 3D replicated tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_3d), _DEVICE_TYPE)
    standalone_output = standalone_input[(0, 1, 2)]

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input[(0, 1, 2)]

    # Three ints remove dims 0,1,2 → 0-d tensor
    assert dist_output.layout.tensor_map == (), (
        f"Tuple-int output tensor_map mismatch: expected (), got {dist_output.layout.tensor_map}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"Tuple-int numerical mismatch: standalone={standalone_output}, parallel={gathered_output}"
    )


def test_getitem_zero_size_slice() -> None:
    """Test __getitem__ producing a zero-size dimension via slice."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d), _DEVICE_TYPE)
    standalone_output = standalone_input[2:2]

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input[2:2]

    assert dist_output.shape == standalone_output.shape, (
        f"Zero-size slice shape mismatch: expected {standalone_output.shape}, "
        f"got {dist_output.shape}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"Zero-size slice numerical mismatch: standalone={standalone_output}, parallel={gathered_output}"
    )


def test_getitem_advanced_paired() -> None:
    """Test __getitem__ with paired advanced indexing x[[0,1],[2,3]] on replicated tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d), _DEVICE_TYPE)
    standalone_output = standalone_input[[0, 1], [2, 3]]

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input[[0, 1], [2, 3]]

    # Paired advanced indexing: each pair selects one element → 1-d output
    assert dist_output.layout.tensor_map == (-1,), (
        f"Paired advanced tensor_map mismatch: expected (-1,), got {dist_output.layout.tensor_map}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"Paired advanced numerical mismatch: standalone={standalone_output}, parallel={gathered_output}"
    )


def test_getitem_advanced_multi_d_index() -> None:
    """Test __getitem__ with multi-dimensional index tensor x[ind_2x2] on replicated tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d), _DEVICE_TYPE)
    ind_2x2 = to_device(torch.tensor([[0, 1], [2, 3]]), _DEVICE_TYPE)
    standalone_output = standalone_input[ind_2x2]

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input[ind_2x2]

    # ind_2x2 shape (2,2) replaces dim0 → output shape (2, 2, 5)
    assert dist_output.shape == standalone_output.shape, (
        f"Multi-d index shape mismatch: expected {standalone_output.shape}, "
        f"got {dist_output.shape}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"Multi-d index numerical mismatch: standalone={standalone_output}, parallel={gathered_output}"
    )


def test_getitem_advanced_consecutive() -> None:
    """Test __getitem__ with consecutive advanced dims x[:,[0,2],[1,3]] on 3D replicated tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_3d), _DEVICE_TYPE)
    standalone_output = standalone_input[:, [0, 2], [1, 3]]

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input[:, [0, 2], [1, 3]]

    assert dist_output.shape == standalone_output.shape, (
        f"Consecutive advanced shape mismatch: expected {standalone_output.shape}, "
        f"got {dist_output.shape}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"Consecutive advanced numerical mismatch: standalone={standalone_output}, parallel={gathered_output}"
    )


def test_getitem_advanced_split() -> None:
    """Test __getitem__ with non-consecutive advanced dims x[[0,1],:,[2,3]] on 3D replicated tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_3d), _DEVICE_TYPE)
    standalone_output = standalone_input[[0, 1], :, [2, 3]]

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input[[0, 1], :, [2, 3]]

    assert dist_output.shape == standalone_output.shape, (
        f"Split advanced shape mismatch: expected {standalone_output.shape}, "
        f"got {dist_output.shape}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"Split advanced numerical mismatch: standalone={standalone_output}, parallel={gathered_output}"
    )


def test_getitem_advanced_keep_shard_outside() -> None:
    """Test __getitem__ with advanced index on a dim that is NOT the sharded dim."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d_shardable), _DEVICE_TYPE)
    standalone_output = standalone_input[[0, 2]]

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    # Shard dim1 (not dim0 — the advanced-indexed dim)
    x_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input[[0, 2]]

    assert dist_output.shape == standalone_output.shape, (
        f"Advanced keep shard outside shape mismatch: expected {standalone_output.shape}, "
        f"got {dist_output.shape}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"Advanced keep shard outside numerical mismatch: standalone={standalone_output}, "
        f"parallel={gathered_output}"
    )


def test_getitem_mixed_basic() -> None:
    """Test __getitem__ with int, full slice, ellipsis, and newaxis x[0, ::1, ..., None]."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d), _DEVICE_TYPE)
    standalone_output = standalone_input[0, ::1, ..., None]

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input[0, ::1, ..., None]

    # int removes dim0, slice keeps dim1, newaxis adds dim → ndim=2
    assert len(dist_output.layout.tensor_map) == 2, (
        f"Mixed basic output ndim mismatch: expected 2, got {len(dist_output.layout.tensor_map)}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"Mixed basic numerical mismatch: standalone={standalone_output}, parallel={gathered_output}"
    )


def test_getitem_chained() -> None:
    """Test chained __getitem__ calls."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(_standalone_2d), _DEVICE_TYPE)
    standalone_output = standalone_input[1:3][0]

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input[1:3][0]

    # After first [1:3]: (2, 5), after [0]: (5,)
    assert len(dist_output.layout.tensor_map) == 1, (
        f"Chained output ndim mismatch: expected 1, got {len(dist_output.layout.tensor_map)}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"Chained numerical mismatch: standalone={standalone_output}, parallel={gathered_output}"
    )
