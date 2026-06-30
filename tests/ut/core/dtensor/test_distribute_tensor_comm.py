#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
"""Unit tests for distribute_tensor src_data_rank (no distributed runtime required)."""
from unittest.mock import MagicMock, patch

from hyper_parallel.core.dtensor.dtensor import distribute_tensor, _distribute_tensor_with_communication
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


def test_distribute_tensor_none_uses_local_slice():
    """src_data_rank=None keeps legacy local slice path."""
    mesh = MagicMock()
    layout = MagicMock()
    layout.alias_placements = (Shard(0),)
    tensor = MagicMock()
    tensor.shape = (8, 4)
    local = MagicMock()

    with patch("hyper_parallel.core.dtensor.dtensor._build_layout", return_value=layout), \
         patch("hyper_parallel.core.dtensor.dtensor._get_slice_tensor_by_layout", return_value=local) as mock_slice, \
         patch("hyper_parallel.core.dtensor.dtensor.DTensor") as mock_dt:
        distribute_tensor(tensor, mesh, [Shard(0)], src_data_rank=None)
        mock_slice.assert_called_once_with(tensor, layout)
        mock_dt.assert_called_once_with(local, mesh, layout.alias_placements)


def test_distribute_tensor_default_uses_local_slice():
    """Default src_data_rank=None keeps legacy local slice path."""
    mesh = MagicMock()
    layout = MagicMock()
    layout.alias_placements = (Shard(0),)
    tensor = MagicMock()
    tensor.shape = (8, 4)
    local = MagicMock()

    with patch("hyper_parallel.core.dtensor.dtensor._build_layout", return_value=layout), \
         patch("hyper_parallel.core.dtensor.dtensor._get_slice_tensor_by_layout", return_value=local) as mock_slice, \
         patch("hyper_parallel.core.dtensor.dtensor.DTensor") as mock_dt:
        distribute_tensor(tensor, mesh, [Shard(0)])
        mock_slice.assert_called_once_with(tensor, layout)
        mock_dt.assert_called_once_with(local, mesh, layout.alias_placements)


def test_distribute_tensor_explicit_src_data_rank_uses_communication_path():
    """Explicit src_data_rank=0 routes through scatter/broadcast helper."""
    mesh = MagicMock()
    layout = MagicMock()
    layout.placements = (Shard(0),)
    layout.alias_placements = (Shard(0),)
    tensor = MagicMock()
    tensor.shape = (8, 4)
    local = MagicMock()

    with patch("hyper_parallel.core.dtensor.dtensor._build_layout", return_value=layout), \
         patch(
             "hyper_parallel.core.dtensor.dtensor._distribute_tensor_with_communication",
             return_value=local,
         ) as mock_comm, \
         patch("hyper_parallel.core.dtensor.dtensor.DTensor") as mock_dt:
        distribute_tensor(tensor, mesh, [Shard(0)], src_data_rank=0)
        mock_comm.assert_called_once_with(tensor, mesh, layout.placements, 0)
        mock_dt.assert_called_once_with(local, mesh, layout.alias_placements)


def test_distribute_shard_then_replicate():
    """Shard dim uses scatter; replicate dim uses broadcast."""
    mesh = MagicMock()
    mesh.ndim = 2
    mesh.size.side_effect = lambda dim=None: 2

    chunk0 = MagicMock(name="chunk0")
    chunk1 = MagicMock(name="chunk1")
    tensor = MagicMock()
    tensor.ndim = 2
    tensor.chunk.return_value = (chunk0, chunk1)
    scattered = MagicMock(name="scattered")
    broadcasted = MagicMock(name="broadcasted")

    with patch("hyper_parallel.core.dtensor.dtensor.mesh_scatter", return_value=scattered) as mock_scatter, \
         patch("hyper_parallel.core.dtensor.dtensor.mesh_broadcast", return_value=broadcasted) as mock_broadcast, \
         patch("hyper_parallel.core.dtensor.dtensor.platform.empty_like", return_value=MagicMock()):
        out = _distribute_tensor_with_communication(
            tensor, mesh, [Shard(0), Replicate()], src_data_rank=0
        )
    assert out is broadcasted
    mock_scatter.assert_called_once()
    mock_broadcast.assert_called_once()
