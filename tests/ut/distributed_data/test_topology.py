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
"""Unit tests for distributed-data topology derivation."""

import unittest

from hyper_parallel.distributed_data.topology import DataTopology


class TestDataTopology(unittest.TestCase):
    """Validate data ownership independently of node-local rank numbering."""

    def test_selects_one_owner_for_each_dp_coordinate(self) -> None:
        """All CP/TP peers at one DP coordinate should share one owner."""
        topology = DataTopology.from_layout(
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp_shard", "cp", "tp"),
            rank_list=tuple(range(8)),
            global_rank=7,
        )

        self.assertEqual(topology.coordinate, (1, 1, 1))
        self.assertEqual(topology.data_rank, 1)
        self.assertEqual(topology.owner_ranks, (0, 4))
        self.assertEqual(topology.data_owner_rank, 4)
        self.assertEqual(topology.consumer_ranks, (4, 5, 6, 7))
        self.assertEqual((topology.cp_rank, topology.cp_size), (1, 2))
        self.assertEqual((topology.tp_rank, topology.tp_size), (1, 2))
        self.assertFalse(topology.is_data_owner)

    def test_uses_mesh_order_for_non_contiguous_global_ranks(self) -> None:
        """Ownership must not assume that global ranks are contiguous or sorted."""
        topology = DataTopology.from_layout(
            mesh_shape=(2, 2),
            mesh_dim_names=("dp_shard", "tp"),
            rank_list=(10, 12, 21, 25),
            global_rank=25,
        )

        self.assertEqual(topology.owner_ranks, (10, 21))
        self.assertEqual(topology.data_owner_rank, 21)
        self.assertEqual(topology.consumer_ranks, (21, 25))

    def test_without_dp_axis_uses_one_data_coordinate(self) -> None:
        """A pure TP mesh should have one owner and one data coordinate."""
        topology = DataTopology.from_layout(
            mesh_shape=(4,),
            mesh_dim_names=("tp",),
            rank_list=(0, 1, 2, 3),
            global_rank=2,
        )

        self.assertEqual(topology.data_world_size, 1)
        self.assertEqual(topology.data_rank, 0)
        self.assertEqual(topology.data_owner_rank, 0)
        self.assertEqual(topology.consumer_ranks, (0, 1, 2, 3))
