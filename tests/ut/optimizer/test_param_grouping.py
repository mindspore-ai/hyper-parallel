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
"""Unit tests for parameter grouping utilities (Muon optimizer support).

Tests cover extract_shard_info, calculate_replicate_group, and
group_parameters_by_sharding using mocked DTensor / DeviceMesh objects
so no actual distributed hardware is required.
"""

import unittest
from unittest.mock import MagicMock, patch

from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
from hyper_parallel.core.optimizer.param_grouping import (
    CommParamGroup,
    ShardInfo,
    calculate_replicate_group,
    extract_shard_info,
    group_parameters_by_sharding,
)


def _make_mock_dtensor(shape, placements, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"),
                       rank=0, rank_list=None):
    """Create a mock DTensor with the given shape and placements.

    Args:
        shape: Global tensor shape tuple.
        placements: Sequence of Placement objects (one per mesh dim).
        mesh_shape: Device mesh shape.
        mesh_dim_names: Names for each mesh dimension.
        rank: Current process rank.
        rank_list: Flat rank list; auto-generated from mesh_shape if None.

    Returns:
        MagicMock configured as a DTensor.
    """
    dtensor = MagicMock()
    dtensor.shape = shape

    # Build a mock DeviceMesh
    if rank_list is None:
        total = 1
        for s in mesh_shape:
            total *= s
        rank_list = tuple(range(total))

    mesh = MagicMock()
    mesh.ndim = len(mesh_shape)
    mesh.mesh_shape = mesh_shape
    mesh.rank = rank
    mesh.rank_list = rank_list
    mesh.mesh_dim_names = mesh_dim_names

    # get_devices_for_axis: return peer ranks along a mesh dimension
    def _get_devices_for_axis(mesh_dim_idx, query_rank):
        mesh_arr = list(rank_list)
        ndim = len(mesh_shape)
        idx = mesh_arr.index(query_rank)
        coord = [0] * ndim
        temp = idx
        for i in range(ndim - 1, -1, -1):
            coord[i] = temp % mesh_shape[i]
            temp //= mesh_shape[i]
        strides = [1] * ndim
        for i in range(ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * mesh_shape[i + 1]
        result = []
        for v in range(mesh_shape[mesh_dim_idx]):
            new_coord = coord.copy()
            new_coord[mesh_dim_idx] = v
            new_idx = 0
            for i in range(ndim):
                new_idx += new_coord[i] * strides[i]
            result.append(mesh_arr[new_idx])
        return result

    mesh.get_devices_for_axis = _get_devices_for_axis

    dtensor.device_mesh = mesh
    dtensor.placements = tuple(placements)

    return dtensor


class TestExtractShardInfo(unittest.TestCase):
    """Unit tests for extract_shard_info."""

    def test_fully_replicated_2d(self):
        """Fully replicated 2-D tensor: no shard dims, both mesh dims replicated."""
        dt = _make_mock_dtensor((8, 16), [Replicate(), Replicate()])
        info = extract_shard_info(dt)
        assert info.tensor_ndim == 2, f"Expected ndim=2, got {info.tensor_ndim}"
        assert info.shard_dims == set(), f"Expected empty shard_dims, got {info.shard_dims}"
        assert info.replicate_mesh_dims == [0, 1], (
            f"Expected [0, 1], got {info.replicate_mesh_dims}"
        )

    def test_shard_dim0_on_2d(self):
        """Shard(0) on 2-D tensor: shard_dims={0}, replicate on mesh dim 1."""
        dt = _make_mock_dtensor((8, 16), [Shard(0), Replicate()])
        info = extract_shard_info(dt)
        assert info.shard_dims == {0}, f"Expected {{0}}, got {info.shard_dims}"
        assert info.replicate_mesh_dims == [1], (
            f"Expected [1], got {info.replicate_mesh_dims}"
        )

    def test_shard_both_dims_on_2d(self):
        """Shard on both tensor dims: shard_dims={0, 1}, no replicate mesh dims."""
        dt = _make_mock_dtensor((8, 16), [Shard(0), Shard(1)])
        info = extract_shard_info(dt)
        assert info.shard_dims == {0, 1}, f"Expected {{0, 1}}, got {info.shard_dims}"
        assert info.replicate_mesh_dims == [], (
            f"Expected [], got {info.replicate_mesh_dims}"
        )

    def test_1d_tensor_raises(self):
        """1-D tensor raises ValueError — Muon requires >= 2-D parameters."""
        dt = _make_mock_dtensor((128,), [Shard(0), Replicate()])
        with self.assertRaises(ValueError) as ctx:
            extract_shard_info(dt)
        assert "at least 2 dimensions" in str(ctx.exception), (
            f"Expected 'at least 2 dimensions' in error, got {ctx.exception}"
        )

    def test_3d_tensor_shard_middle(self):
        """3-D tensor sharded on dim 1 only: shard_dims={1}."""
        dt = _make_mock_dtensor((4, 8, 16), [Replicate(), Shard(1)])
        info = extract_shard_info(dt)
        assert info.tensor_ndim == 3, f"Expected ndim=3, got {info.tensor_ndim}"
        assert info.shard_dims == {1}, f"Expected {{1}}, got {info.shard_dims}"
        assert info.replicate_mesh_dims == [0], (
            f"Expected [0], got {info.replicate_mesh_dims}"
        )


class TestCalculateReplicateGroup(unittest.TestCase):
    """Unit tests for calculate_replicate_group."""

    def test_no_replicate_dims(self):
        """All mesh dims are Shard: replicate_group = [current_rank]."""
        dt = _make_mock_dtensor((8, 16), [Shard(0), Shard(1)], rank=3)
        result = calculate_replicate_group(dt)
        assert result == [3], f"Expected [3], got {result}"

    def test_one_replicate_dim(self):
        """One replicate mesh dim (dp=2, tp=4): replicate group = 4 ranks along tp."""
        dt = _make_mock_dtensor(
            (8, 16), [Replicate(), Shard(1)],
            mesh_shape=(2, 4), rank=2,
        )
        result = calculate_replicate_group(dt)
        # rank 2 in (2,4) mesh: coord=(0,2). Replicate along dim 0 →
        # peers are (0,2)=2 and (1,2)=6.
        assert result == [2, 6], f"Expected [2, 6], got {result}"

    def test_both_replicate_dims(self):
        """Both mesh dims replicate: full mesh is the replicate group."""
        dt = _make_mock_dtensor(
            (8, 16), [Replicate(), Replicate()],
            mesh_shape=(2, 4), rank=0,
        )
        result = calculate_replicate_group(dt)
        assert result == [0, 1, 2, 3, 4, 5, 6, 7], (
            f"Expected [0..7], got {result}"
        )

    def test_with_precomputed_shard_info(self):
        """Passing shard_info avoids re-extraction."""
        dt = _make_mock_dtensor(
            (8, 16), [Replicate(), Shard(1)],
            mesh_shape=(2, 4), rank=5,
        )
        info = extract_shard_info(dt)
        result = calculate_replicate_group(dt, shard_info=info)
        # rank 5 in (2,4): coord=(1,1). Replicate dim 0 → (0,1)=1, (1,1)=5
        assert result == [1, 5], f"Expected [1, 5], got {result}"

    def test_3d_mesh_two_replicate_dims(self):
        """3-D mesh with 2 replicate dims: Cartesian product of peer lists."""
        dt = _make_mock_dtensor(
            (8, 16), [Replicate(), Replicate(), Shard(1)],
            mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "cp", "tp"),
            rank=0,
        )
        result = calculate_replicate_group(dt)
        # rank 0 coord=(0,0,0). Replicate dims 0 and 1.
        # dim 0 peers: (0,0,0)=0, (1,0,0)=4
        # dim 1 peers: (0,0,0)=0, (0,1,0)=2
        # Cartesian product union: {0, 2, 4, 6}
        assert result == [0, 2, 4, 6], f"Expected [0, 2, 4, 6], got {result}"


class TestGroupParametersBySharding(unittest.TestCase):
    """Unit tests for group_parameters_by_sharding."""

    def test_empty_input(self):
        """Empty parameter list returns empty groups."""
        no_comm, comm_groups = group_parameters_by_sharding([])
        assert no_comm == [], f"Expected [], got {no_comm}"
        assert comm_groups == [], f"Expected [], got {comm_groups}"

    def test_1d_params_raise_error(self):
        """1-D parameters raise ValueError — they cannot participate in Muon."""
        p1 = _make_mock_dtensor((128,), [Shard(0), Replicate()])
        with self.assertRaises(ValueError) as ctx:
            group_parameters_by_sharding([p1])
        assert "at least 2 dimensions" in str(ctx.exception), (
            f"Expected 'at least 2 dimensions' in error, got {ctx.exception}"
        )

    def test_fully_replicated_are_no_comm(self):
        """Fully replicated 2-D params go to no_comm_params."""
        p = _make_mock_dtensor((8, 16), [Replicate(), Replicate()])
        no_comm, comm_groups = group_parameters_by_sharding([p])
        assert len(no_comm) == 1, f"Expected 1 no_comm param, got {len(no_comm)}"
        assert len(comm_groups) == 0, f"Expected 0 comm groups, got {len(comm_groups)}"

    def test_shard_early_dim_is_no_comm(self):
        """3-D param [4,8,16] sharded only on dim 0 (not in last two) is no-comm."""
        p = _make_mock_dtensor((4, 8, 16), [Shard(0), Replicate()])
        no_comm, comm_groups = group_parameters_by_sharding([p])
        assert len(no_comm) == 1, f"Expected 1 no_comm param, got {len(no_comm)}"
        assert len(comm_groups) == 0, f"Expected 0 comm groups, got {len(comm_groups)}"

    def test_shard_last_dim_is_comm(self):
        """2-D param sharded on dim 1 (last dim) is a comm param."""
        p = _make_mock_dtensor((8, 16), [Replicate(), Shard(1)], rank=0)
        no_comm, comm_groups = group_parameters_by_sharding([p])
        assert len(no_comm) == 0, f"Expected 0 no_comm params, got {len(no_comm)}"
        assert len(comm_groups) == 1, f"Expected 1 comm group, got {len(comm_groups)}"
        assert len(comm_groups[0].params) == 1, (
            f"Expected 1 param in group, got {len(comm_groups[0].params)}"
        )
        assert comm_groups[0].replicate_group == [0, 4], (
            f"Expected [0, 4], got {comm_groups[0].replicate_group}"
        )

    def test_shard_both_last_two_dims_is_comm(self):
        """2-D param sharded on both dims 0 and 1 is a comm param."""
        p = _make_mock_dtensor((8, 16), [Shard(0), Shard(1)], rank=0)
        no_comm, comm_groups = group_parameters_by_sharding([p])
        assert len(no_comm) == 0, f"Expected 0 no_comm, got {len(no_comm)}"
        assert len(comm_groups) == 1, f"Expected 1 comm group, got {len(comm_groups)}"

    def test_3d_shard_middle_is_no_comm(self):
        """3-D param [4,8,16] sharded on dim 0 only is no-comm (last two=1,2)."""
        p = _make_mock_dtensor((4, 8, 16), [Shard(0), Replicate()])
        no_comm, comm_groups = group_parameters_by_sharding([p])
        assert len(no_comm) == 1, f"Expected 1 no_comm param, got {len(no_comm)}"
        assert len(comm_groups) == 0, f"Expected 0 comm groups, got {len(comm_groups)}"

    def test_3d_shard_last_two_is_comm(self):
        """3-D param [4,8,16] sharded on dim 2 (last) is a comm param."""
        p = _make_mock_dtensor((4, 8, 16), [Replicate(), Shard(2)], rank=0)
        no_comm, comm_groups = group_parameters_by_sharding([p])
        assert len(no_comm) == 0, f"Expected 0 no_comm, got {len(no_comm)}"
        assert len(comm_groups) == 1, f"Expected 1 comm group, got {len(comm_groups)}"

    def test_3d_shard_second_to_last_is_comm(self):
        """3-D param [4,8,16] sharded on dim 1 (second-to-last) is comm."""
        p = _make_mock_dtensor((4, 8, 16), [Replicate(), Shard(1)], rank=0)
        no_comm, comm_groups = group_parameters_by_sharding([p])
        assert len(comm_groups) == 1, f"Expected 1 comm group, got {len(comm_groups)}"

    def test_same_shard_grouped_together(self):
        """Two params with identical placements share one CommParamGroup."""
        p1 = _make_mock_dtensor((8, 16), [Replicate(), Shard(1)], rank=0)
        p2 = _make_mock_dtensor((4, 32), [Replicate(), Shard(1)], rank=0)
        no_comm, comm_groups = group_parameters_by_sharding([p1, p2])
        assert len(comm_groups) == 1, f"Expected 1 comm group, got {len(comm_groups)}"
        assert len(comm_groups[0].params) == 2, (
            f"Expected 2 params in group, got {len(comm_groups[0].params)}"
        )

    def test_different_shard_separate_groups(self):
        """Params with different placements go to separate CommParamGroups."""
        p1 = _make_mock_dtensor((8, 16), [Replicate(), Shard(1)], rank=0)
        p2 = _make_mock_dtensor((8, 16), [Shard(0), Shard(1)], rank=0)
        no_comm, comm_groups = group_parameters_by_sharding([p1, p2])
        assert len(comm_groups) == 2, f"Expected 2 comm groups, got {len(comm_groups)}"

    def test_mixed_no_comm_and_comm(self):
        """Mix of no-comm and comm params are correctly separated."""
        p_replicated = _make_mock_dtensor((8, 16), [Replicate(), Replicate()])
        p_shard_last = _make_mock_dtensor((8, 16), [Replicate(), Shard(1)], rank=0)
        no_comm, comm_groups = group_parameters_by_sharding(
            [p_replicated, p_shard_last]
        )
        assert len(no_comm) == 1, f"Expected 1 no_comm param, got {len(no_comm)}"
        assert len(comm_groups) == 1, f"Expected 1 comm group, got {len(comm_groups)}"

    def test_replicate_group_computed_once_per_group(self):
        """Replicate group is computed once for the first param in a group."""
        p1 = _make_mock_dtensor((8, 16), [Replicate(), Shard(1)], rank=0)
        p2 = _make_mock_dtensor((4, 32), [Replicate(), Shard(1)], rank=0)
        _, comm_groups = group_parameters_by_sharding([p1, p2])
        assert comm_groups[0].replicate_group == [0, 4], (
            f"Expected [0, 4], got {comm_groups[0].replicate_group}"
        )


class TestPlacementsKey(unittest.TestCase):
    """Unit tests for _placements_key helper."""

    def test_shard_equality(self):
        """Same Shard placements produce the same key."""
        from hyper_parallel.core.optimizer.param_grouping import _placements_key
        k1 = _placements_key([Shard(0), Replicate()])
        k2 = _placements_key([Shard(0), Replicate()])
        assert k1 == k2, f"Expected equal keys, got {k1} vs {k2}"

    def test_shard_inequality(self):
        """Different Shard placements produce different keys."""
        from hyper_parallel.core.optimizer.param_grouping import _placements_key
        k1 = _placements_key([Shard(0), Replicate()])
        k2 = _placements_key([Shard(1), Replicate()])
        assert k1 != k2, f"Expected different keys, got {k1} vs {k2}"

    def test_partial_in_key(self):
        """Partial placement is correctly included in the key."""
        from hyper_parallel.core.optimizer.param_grouping import _placements_key
        k1 = _placements_key([Partial("sum"), Shard(1)])
        k2 = _placements_key([Partial("sum"), Shard(1)])
        k3 = _placements_key([Partial("avg"), Shard(1)])
        assert k1 == k2, f"Expected equal keys, got {k1} vs {k2}"
        assert k1 != k3, f"Expected different keys, got {k1} vs {k3}"


if __name__ == "__main__":
    unittest.main()
