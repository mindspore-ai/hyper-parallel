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
"""Unit tests for :func:`hyper_parallel.core.dtensor.random._calc_shard_info`.

Guards the case where ``device_mesh.ndim != len(global_shape)`` (e.g. 3-D mesh
with a 2-D tensor). ``dim_map`` must be sized by tensor rank, not mesh rank.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.dtensor.random import _calc_shard_info, _calc_shard_linear_idx


def _make_mesh(mesh_shape: tuple[int, ...]) -> MagicMock:
    """Build a minimal ``DeviceMesh`` stand-in for shard-info tests."""
    mesh = MagicMock()
    mesh.mesh_shape = mesh_shape
    mesh.ndim = len(mesh_shape)
    return mesh


class TestCalcShardInfoMeshTensorRankMismatch(unittest.TestCase):
    """``_calc_shard_info`` when mesh rank and tensor rank differ."""

    def test_three_d_mesh_two_d_tensor_returns_tensor_rank_entries(self):
        """3-D mesh + 2-D tensor: output length follows ``global_shape``, not mesh."""
        mesh = _make_mesh((2, 2, 2))
        placements = [Shard(1), Replicate(), Shard(0)]
        global_shape = (8, 16)

        shard_idx, total_shards = _calc_shard_info(
            mesh_coordinate=(1, 0, 1),
            device_mesh=mesh,
            placements=placements,
            global_shape=global_shape,
        )

        self.assertEqual(len(shard_idx), len(global_shape))
        self.assertEqual(len(total_shards), len(global_shape))
        self.assertEqual(shard_idx, [1, 1])
        self.assertEqual(total_shards, [2, 2])
        self.assertEqual(_calc_shard_linear_idx(shard_idx, total_shards), 3)

    def test_three_d_mesh_two_d_tensor_docstring_shard_coords(self):
        """Shard coords from random.py docstring: ``dim_map = [2, 0]`` → ``(z, x)``."""
        mesh = _make_mesh((2, 2, 2))
        placements = [Shard(1), Replicate(), Shard(0)]
        global_shape = (8, 16)

        cases = {
            (0, 0, 0): (0, 0),
            (0, 0, 1): (1, 0),
            (0, 1, 0): (0, 0),
            (0, 1, 1): (1, 0),
            (1, 0, 0): (0, 1),
            (1, 0, 1): (1, 1),
            (1, 1, 0): (0, 1),
            (1, 1, 1): (1, 1),
        }
        for mesh_coordinate, expected_shard_coord in cases.items():
            with self.subTest(mesh_coordinate=mesh_coordinate):
                shard_idx, total_shards = _calc_shard_info(
                    mesh_coordinate=mesh_coordinate,
                    device_mesh=mesh,
                    placements=placements,
                    global_shape=global_shape,
                )
                self.assertEqual(tuple(shard_idx), expected_shard_coord)
                self.assertEqual(total_shards, [2, 2])

    def test_two_d_mesh_two_d_tensor_still_works(self):
        """Regression: equal mesh/tensor rank (existing 2-D sharding paths)."""
        mesh = _make_mesh((2, 2))
        global_shape = (8, 8)

        shard_idx, total_shards = _calc_shard_info(
            mesh_coordinate=(1, 0),
            device_mesh=mesh,
            placements=[Shard(0), Shard(1)],
            global_shape=global_shape,
        )
        self.assertEqual(shard_idx, [1, 0])
        self.assertEqual(total_shards, [2, 2])

        shard_idx, total_shards = _calc_shard_info(
            mesh_coordinate=(0, 1),
            device_mesh=mesh,
            placements=[Shard(0), Replicate()],
            global_shape=global_shape,
        )
        self.assertEqual(shard_idx, [0, 0])
        self.assertEqual(total_shards, [2, 1])

    def test_four_d_tensor_two_d_mesh_high_dim_shard(self):
        """Tensor rank > mesh rank: shard on dim 3 must not IndexError."""
        mesh = _make_mesh((2, 4))
        placements = [Shard(0), Shard(3)]
        global_shape = (8, 4, 4, 16)

        shard_idx, total_shards = _calc_shard_info(
            mesh_coordinate=(1, 2),
            device_mesh=mesh,
            placements=placements,
            global_shape=global_shape,
        )

        self.assertEqual(len(shard_idx), 4)
        self.assertEqual(shard_idx, [1, 0, 0, 2])
        self.assertEqual(total_shards, [2, 1, 1, 4])

    def test_mesh_coordinate_none_raises(self):
        """Missing coordinate is rejected before shard math."""
        mesh = _make_mesh((2, 2))
        with self.assertRaisesRegex(ValueError, "mesh_coordinate must not be None"):
            _calc_shard_info(
                mesh_coordinate=None,
                device_mesh=mesh,
                placements=[Shard(0), Shard(1)],
                global_shape=(8, 8),
            )


if __name__ == "__main__":
    unittest.main()
