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
"""Unit tests for placement and tensor_map conversion."""
import os
import unittest
from unittest.mock import patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, _DEVICE_MESH_MAP
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard, StridedShard
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestPlacementConversion(unittest.TestCase):
    """Unit tests for Layout placement/tensor_map conversion."""

    def setUp(self):
        """Patch rank lookup and clear caches before each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.rank_patcher = patch("hyper_parallel.core.dtensor.device_mesh.platform.get_rank", return_value=0)
        self.rank_patcher.start()

    def tearDown(self):
        """Stop patchers and clear caches after each test."""
        self.rank_patcher.stop()
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def test_basic_conversion(self):
        """Test basic conversion between placements and tensor_map."""
        layout = Layout((2, 2), ("dp", "tp"), init_backend=False)

        layout.set_placements([Shard(0), Shard(1)])
        tensor_map = layout.placement_to_tensor_map(dim=2)
        self.assertEqual(tuple(tensor_map), (1, 0))

        layout.set_placements([Replicate(), Shard(0)])
        tensor_map = layout.placement_to_tensor_map(dim=1)
        self.assertEqual(tuple(tensor_map), (0,))

        layout.set_tensor_map((1, 0))
        layout.set_placements(None)
        placements = layout.tensor_map_to_placement()
        self.assertEqual(placements[0], Shard(0))
        self.assertEqual(placements[1], Shard(1))

    def test_shard_combinations(self):
        """Test various Shard combinations and round-trip conversion."""
        layout = Layout((4, 2), ("dp", "mp"), init_backend=False)

        layout.set_placements([Shard(1), Shard(0)])
        tensor_map = layout.placement_to_tensor_map(dim=2)
        self.assertEqual(tuple(tensor_map), (0, 1))

        placements = layout.tensor_map_to_placement()
        self.assertEqual(placements[0], Shard(1))
        self.assertEqual(placements[1], Shard(0))

        layout.set_placements([Shard(0), Replicate()])
        tensor_map = layout.placement_to_tensor_map(dim=3)
        self.assertEqual(tuple(tensor_map), (1, -1, -1))

        placements = layout.tensor_map_to_placement()
        self.assertEqual(placements[0], Shard(0))
        self.assertIsInstance(placements[1], Replicate)

    def test_full_replication(self):
        """Test full replication across all mesh dimensions."""
        layout = Layout((2, 2, 2), ("dp", "sp", "mp"), init_backend=False)

        layout.set_placements([Replicate(), Replicate(), Replicate()])
        tensor_map = layout.placement_to_tensor_map(dim=4)
        self.assertEqual(tuple(tensor_map), (-1, -1, -1, -1))

        placements = layout.tensor_map_to_placement()
        self.assertTrue(all(isinstance(placement, Replicate) for placement in placements))
        self.assertEqual(len(placements), 3)

    def test_partial_shard_mixed(self):
        """Test mixed Partial and Shard placement conversion."""
        layout = Layout((2, 2), ("dp", "mp"), init_backend=False)

        layout.set_placements([Partial("sum"), Shard(0)])
        tensor_map = layout.placement_to_tensor_map(dim=2)

        self.assertEqual(tuple(tensor_map), (0, -1))
        self.assertEqual(layout.partial[0], "sum")
        self.assertIsNone(layout.partial[1])

        placements = layout.tensor_map_to_placement()
        self.assertIsInstance(placements[0], Partial)
        self.assertEqual(placements[0].reduce_op, "sum")
        self.assertEqual(placements[1], Shard(0))

    def test_partial_reduce_ops(self):
        """Test min/max/avg partial reduction ops are preserved."""
        layout = Layout((2,), ("dp",), init_backend=False)

        for op in ("min", "max", "avg"):
            layout.set_placements([Partial(op)])
            tensor_map = layout.placement_to_tensor_map(dim=1)
            self.assertEqual(tuple(tensor_map), (-1,))
            self.assertEqual(layout.partial[0], op)

            placements = layout.tensor_map_to_placement()
            self.assertIsInstance(placements[0], Partial)
            self.assertEqual(placements[0].reduce_op, op)

    def test_conversion_errors(self):
        """Test error handling for invalid placement conversion inputs."""
        layout = Layout((2, 2), ("dp", "mp"), init_backend=False)

        layout.set_placements([Shard(5), Replicate()])
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            layout.placement_to_tensor_map(dim=2)

        layout.set_placements([Shard(0), Shard(0)])
        tensor_map = layout.placement_to_tensor_map(dim=2)
        self.assertEqual(tuple(tensor_map), ((1, 0), -1))

        with self.assertRaisesRegex(ValueError, "positive"):
            layout.placement_to_tensor_map(dim=-1)

        invalid_layout = Layout((2, 2, 2), ("dp", "tp", "ep"), init_backend=False)
        invalid_layout.set_placements([StridedShard(0, split_factor=2), Shard(0), Shard(0)])
        with self.assertRaisesRegex(ValueError, "split_factor mismatch"):
            invalid_layout.placement_to_tensor_map(dim=2)

    def test_strided_shard_conversion(self):
        """Test StridedShard converts to and from tuple tensor_map."""
        layout = Layout((2, 2), ("dp", "tp"), init_backend=False)

        layout.set_placements([StridedShard(0, split_factor=2), Shard(0)])
        tensor_map = layout.placement_to_tensor_map(dim=2)
        self.assertEqual(tuple(tensor_map), ((0, 1), -1))

        layout.set_tensor_map(((0, 1), -1))
        layout.set_placements(None)
        placements = layout.tensor_map_to_placement()
        self.assertEqual(placements[0], StridedShard(0, split_factor=2))
        self.assertEqual(placements[1], Shard(0))

    def test_uneven_shard_marker_value_semantics(self):
        """Test uneven markers participate in placement identity and representation."""
        marked_shard = Shard(0, uneven_shard=True)
        same_marked_shard = Shard(0, uneven_shard=True)
        balanced_shard = Shard(0)
        marked_strided_shard = StridedShard(0, split_factor=2, uneven_shard=True)

        self.assertEqual(marked_shard, same_marked_shard)
        self.assertEqual(hash(marked_shard), hash(same_marked_shard))
        self.assertNotEqual(marked_shard, balanced_shard)
        self.assertNotEqual(hash(marked_shard), hash(balanced_shard))
        self.assertEqual(repr(marked_shard), "Shard(dim=0, uneven_shard=True)")
        self.assertEqual(
            repr(marked_strided_shard),
            "StridedShard(dim=0, split_factor=2, uneven_shard=True)",
        )
        with self.assertRaisesRegex(TypeError, "uneven_shard must be bool"):
            Shard(0, uneven_shard=1)

    def test_uneven_shard_layout_round_trip(self):
        """Test tensor-map reconstruction retains the marked FSDP mesh placement."""
        layout = Layout((2, 2), ("fsdp", "tp"), init_backend=False)
        expected_placements = [
            StridedShard(0, split_factor=2, uneven_shard=True),
            Shard(0),
        ]

        layout.set_placements(expected_placements)
        tensor_map = layout.placement_to_tensor_map(dim=2)
        rebuilt_placements = layout.tensor_map_to_placement()

        self.assertEqual(tuple(tensor_map), ((0, 1), -1))
        self.assertEqual(rebuilt_placements, expected_placements)
        self.assertEqual(layout.uneven_shard_mesh_dims, (0,))
        self.assertTrue(layout.has_uneven_shard)

    def test_tensor_map_to_placement_preserves_left_to_right_sharding(self):
        """Test left-to-right tensor_map maps back to plain Shard placements."""
        layout = Layout((2, 2), ("dp", "tp"), init_backend=False)

        layout.set_tensor_map(((1, 0), -1))
        placements = layout.tensor_map_to_placement()
        self.assertEqual(placements[0], Shard(0))
        self.assertEqual(placements[1], Shard(0))

    def test_three_dim_strided_shard_round_trip(self):
        """Test round-trip conversion for chained 3D StridedShard placements."""
        layout = Layout((2, 2, 2), ("dp", "tp", "ep"), init_backend=False)

        layout.set_placements([StridedShard(0, split_factor=4), StridedShard(0, split_factor=2), Shard(0)])
        tensor_map = layout.placement_to_tensor_map(dim=2)
        self.assertEqual(tuple(tensor_map), ((0, 1, 2), -1))

        layout.set_tensor_map(((0, 1, 2), -1))
        layout.set_placements(None)
        placements = layout.tensor_map_to_placement()
        self.assertEqual(placements[0], StridedShard(0, split_factor=4))
        self.assertEqual(placements[1], StridedShard(0, split_factor=2))
        self.assertEqual(placements[2], Shard(0))

    def test_from_device_mesh(self):
        """Test Layout.from_device_mesh preserves mesh metadata."""
        mesh_shape = (4, 2)
        alias_name = ("dp", "mp")
        rank_list = tuple(range(8))
        device_mesh = DeviceMesh(
            "npu",
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            mesh_dim_names=alias_name,
            _init_backend=False,
        )

        layout = Layout.from_device_mesh(device_mesh)

        self.assertEqual(layout.mesh_shape, mesh_shape)
        self.assertEqual(layout.alias_name, alias_name)
        self.assertEqual(layout.rank_list, rank_list)
        self.assertIsNone(layout.tensor_map)
        self.assertIsNone(layout.placements)
        self.assertEqual(layout.partial, [None, None])

        layout.set_placements([Shard(0), Replicate()])
        tensor_map = layout.placement_to_tensor_map(dim=2)
        self.assertEqual(tuple(tensor_map), (1, -1))
        self.assertEqual(layout.mesh.mesh_shape, mesh_shape)


if __name__ == "__main__":
    unittest.main()
