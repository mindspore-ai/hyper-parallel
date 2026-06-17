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
"""Coverage supplement tests for hyper_parallel.core.dtensor.layout."""

import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import unittest
from unittest.mock import patch, MagicMock, Mock

import torch
import numpy as np

from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial, StridedShard
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


def _setup_mock_platform(platform_mock):
    """Configure a mock platform for DeviceMesh construction."""
    platform_mock.get_rank.return_value = 0
    platform_mock.get_world_size.return_value = 8
    platform_mock.Tensor = torch.Tensor
    mock_group = Mock()
    mock_group.group_name = "mock_group"

    def _split_group_side_effect(split_ranks, parent_pg=None, timeout=None, pg_options=None, group_desc=None):
        """ split group side effect."""
        for sr in split_ranks:
            key = str(tuple(sorted(sr)))
            EXISTING_COMM_GROUPS[key] = mock_group
        return mock_group

    platform_mock.split_group.side_effect = _split_group_side_effect
    platform_mock.get_created_group.return_value = None
    platform_mock.get_process_group_ranks.return_value = list(range(8))

    def mock_tensor_to_numpy(tensor):
        """Mock tensor to numpy."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    platform_mock.tensor_to_numpy.side_effect = mock_tensor_to_numpy


def _make_layout_with_mock(mesh_shape, alias_name, rank_list=None):
    """Create Layout within mocked context."""
    _DEVICE_MESH_MAP.clear()
    EXISTING_COMM_GROUPS.clear()
    from hyper_parallel.core.dtensor.layout import Layout
    return Layout(mesh_shape, alias_name, rank_list=rank_list, init_backend=False)


class _MockedLayoutTestCase(unittest.TestCase):
    """Base class that patches the device_mesh platform for all tests."""

    def setUp(self):
        """Set up test fixtures."""
        patcher_dm = patch("hyper_parallel.core.dtensor.device_mesh.platform")
        patcher_tensor = patch("hyper_parallel.core.dtensor.device_mesh.Tensor", torch.Tensor)
        self.mock_dm_platform = patcher_dm.start()
        patcher_tensor.start()
        _setup_mock_platform(self.mock_dm_platform)
        self.addCleanup(patcher_dm.stop)
        self.addCleanup(patcher_tensor.stop)
        self.addCleanup(_DEVICE_MESH_MAP.clear)
        self.addCleanup(EXISTING_COMM_GROUPS.clear)
        _DEVICE_MESH_MAP.clear()
        EXISTING_COMM_GROUPS.clear()

    def _make_layout(self, mesh_shape, alias_name, rank_list=None):
        """ make layout."""
        from hyper_parallel.core.dtensor.layout import Layout
        return Layout(mesh_shape, alias_name, rank_list=rank_list, init_backend=False)


class TestPlacementToTensorMap(_MockedLayoutTestCase):
    """Tests for PlacementToTensorMap."""
    def test_zero_dim_tensor(self):
        """Test zero dim tensor."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout.set_placements([Partial("sum"), Replicate()])
        result = layout.placement_to_tensor_map(0)
        self.assertEqual(result, [])
        self.assertEqual(layout.tensor_map, ())
        self.assertEqual(layout.partial[0], "sum")

    def test_basic_shard(self):
        """Test basic shard."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout.set_placements([Shard(0), Replicate()])
        result = layout.placement_to_tensor_map(2)
        self.assertIsNotNone(result)
        self.assertEqual(layout.tensor_map[0], 1)
        self.assertEqual(layout.tensor_map[1], -1)

    def test_negative_dim_raises(self):
        """Test negative dim raises."""
        layout = self._make_layout((2,), ("dp",))
        layout.set_placements([Shard(0)])
        with self.assertRaises(ValueError):
            layout.placement_to_tensor_map(-1)

    def test_shard_dim_out_of_bounds_raises(self):
        """Test shard dim out of bounds raises."""
        layout = self._make_layout((2,), ("dp",))
        layout.set_placements([Shard(5)])
        with self.assertRaises(ValueError):
            layout.placement_to_tensor_map(2)

    def test_strided_shard(self):
        """Test strided shard."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout.set_placements([StridedShard(dim=0, split_factor=4), Shard(0)])
        result = layout.placement_to_tensor_map(2)
        self.assertIsNotNone(result)

    def test_partial_placement(self):
        """Test partial placement."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout.set_placements([Partial("sum"), Shard(0)])
        result = layout.placement_to_tensor_map(2)
        self.assertEqual(layout.partial[0], "sum")


class TestTensorMapToPlacement(_MockedLayoutTestCase):
    """Tests for TensorMapToPlacement."""
    def test_basic_roundtrip(self):
        """Test basic roundtrip."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout.set_tensor_map((1, -1))
        placements = layout.tensor_map_to_placement()
        self.assertEqual(len(placements), 2)
        self.assertIsInstance(placements[0], Shard)
        self.assertIsInstance(placements[1], Replicate)

    def test_tuple_tensor_map(self):
        """Test tuple tensor map."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout.set_tensor_map(((1, 0),))
        placements = layout.tensor_map_to_placement()
        self.assertEqual(len(placements), 2)

    def test_none_tensor_map_raises(self):
        """Test none tensor map raises."""
        layout = self._make_layout((2,), ("dp",))
        with self.assertRaises(ValueError):
            layout.tensor_map_to_placement()

    def test_partial_preserved(self):
        """Test partial preserved."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout.set_tensor_map((1, -1))
        layout._partial[1] = "sum"
        placements = layout.tensor_map_to_placement()
        self.assertIsInstance(placements[1], Partial)


class TestSetPartialByDevAxis(_MockedLayoutTestCase):
    """Tests for SetPartialByDevAxis."""
    def test_valid_set(self):
        """Test valid set."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout.set_tensor_map((-1, -1))
        layout.tensor_map_to_placement()
        layout.set_partial_by_dev_axis("dp", "sum")
        self.assertEqual(layout.partial[0], "sum")

    def test_invalid_op_raises(self):
        """Test invalid op raises."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout.set_tensor_map((-1, -1))
        layout.tensor_map_to_placement()
        with self.assertRaises(ValueError):
            layout.set_partial_by_dev_axis("dp", "invalid_op")

    def test_shard_dim_raises(self):
        """Test shard dim raises."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout.set_tensor_map((1, -1))
        layout.tensor_map_to_placement()
        with self.assertRaises(ValueError):
            layout.set_partial_by_dev_axis("dp", "sum")


class TestGetDimSplitNum(_MockedLayoutTestCase):
    """Tests for GetDimSplitNum."""
    def test_none_mapping_returns_1(self):
        """Test none mapping returns 1."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout._alias_tensor_map = ("dp", "None")
        self.assertEqual(layout.get_dim_split_num(1), 1)

    def test_string_mapping(self):
        """Test string mapping."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout._alias_tensor_map = ("dp", "None")
        self.assertEqual(layout.get_dim_split_num(0), 2)

    def test_tuple_mapping(self):
        """Test tuple mapping."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout._alias_tensor_map = (("dp", "mp"), "None")
        self.assertEqual(layout.get_dim_split_num(0), 8)

    def test_out_of_bounds_returns_1(self):
        """Test out of bounds returns 1."""
        layout = self._make_layout((2,), ("dp",))
        layout._alias_tensor_map = ("dp",)
        self.assertEqual(layout.get_dim_split_num(5), 1)

    def test_no_alias_tensor_map_returns_1(self):
        """Test no alias tensor map returns 1."""
        layout = self._make_layout((2,), ("dp",))
        self.assertEqual(layout.get_dim_split_num(0), 1)


class TestGetSplitId(_MockedLayoutTestCase):
    """Tests for GetSplitId."""
    @patch("hyper_parallel.core.dtensor.layout.platform")
    def test_single_axis(self, mock_layout_platform):
        """Test single axis."""
        mock_layout_platform.get_rank.return_value = 0
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout._alias_tensor_map = ("dp", "None")
        result = layout.get_split_id(0)
        self.assertIsInstance(result, int)

    @patch("hyper_parallel.core.dtensor.layout.platform")
    def test_tuple_axis(self, mock_layout_platform):
        """Test tuple axis."""
        mock_layout_platform.get_rank.return_value = 0
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout._alias_tensor_map = (("dp", "mp"), "None")
        result = layout.get_split_id(0)
        self.assertIsInstance(result, int)

    @patch("hyper_parallel.core.dtensor.layout.platform")
    def test_none_mapping(self, mock_layout_platform):
        """Test none mapping."""
        mock_layout_platform.get_rank.return_value = 0
        layout = self._make_layout((2,), ("dp",))
        layout._alias_tensor_map = ("None",)
        self.assertEqual(layout.get_split_id(0), 0)

    @patch("hyper_parallel.core.dtensor.layout.platform")
    def test_no_alias_tensor_map(self, mock_layout_platform):
        """Test no alias tensor map."""
        mock_layout_platform.get_rank.return_value = 0
        layout = self._make_layout((2,), ("dp",))
        self.assertEqual(layout.get_split_id(0), 0)


class TestRepeatNum(_MockedLayoutTestCase):
    """Tests for RepeatNum."""
    def test_none_tensor_map_raises(self):
        """Test none tensor map raises."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        with self.assertRaises(ValueError):
            layout.repeat_num()

    def test_basic(self):
        """Test basic."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout.set_tensor_map((1, -1))
        self.assertEqual(layout.repeat_num(), 4)

    def test_tuple_tensor_map(self):
        """Test tuple tensor map."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout.set_tensor_map(((1, 0),))
        self.assertEqual(layout.repeat_num(), 1)


class TestToString(_MockedLayoutTestCase):
    """Tests for ToString."""
    def test_with_tensor_map(self):
        """Test with tensor map."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout.set_tensor_map((1, -1))
        s = layout.to_string()
        self.assertIn("Mesh shape", s)
        self.assertIn("Tensor Map", s)
        self.assertIn("dp", s)

    def test_without_tensor_map(self):
        """Test without tensor map."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        s = layout.to_string()
        self.assertIn("Not configured", s)

    def test_tuple_tensor_map(self):
        """Test tuple tensor map."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout.set_tensor_map(((1, 0),))
        s = layout.to_string()
        self.assertIn("Tensor Map", s)


class TestLayoutEq(_MockedLayoutTestCase):
    """Tests for LayoutEq."""
    def test_different_type(self):
        """Test different type."""
        layout = self._make_layout((2,), ("dp",))
        self.assertNotEqual(layout, "not_a_layout")

    def test_different_mesh_shape(self):
        """Test different mesh shape."""
        a = self._make_layout((2,), ("dp",))
        b = self._make_layout((4,), ("dp",))
        self.assertNotEqual(a, b)

    def test_one_none_tensor_map(self):
        """Test one none tensor map."""
        a = self._make_layout((2,), ("dp",))
        b = self._make_layout((2,), ("dp",))
        a.set_tensor_map((0,))
        self.assertNotEqual(a, b)

    def test_both_none_tensor_map(self):
        """Test both none tensor map."""
        a = self._make_layout((2,), ("dp",))
        b = self._make_layout((2,), ("dp",))
        self.assertEqual(a, b)

    def test_equal_with_tensor_map(self):
        """Test equal with tensor map."""
        a = self._make_layout((2, 4), ("dp", "mp"))
        b = self._make_layout((2, 4), ("dp", "mp"))
        a.set_tensor_map((1, -1))
        b.set_tensor_map((1, -1))
        self.assertEqual(a, b)


class TestLayoutStr(_MockedLayoutTestCase):
    """Tests for LayoutStr."""
    def test_str(self):
        """Test str."""
        layout = self._make_layout((2,), ("dp",))
        self.assertIn("Layout Configuration", str(layout))

    def test_repr(self):
        """Test repr."""
        layout = self._make_layout((2,), ("dp",))
        self.assertIn("Layout", repr(layout))


class TestLayoutCall(_MockedLayoutTestCase):
    """Tests for LayoutCall."""
    def test_alias_layout(self):
        """Test alias layout."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        new_layout = layout("dp", "mp")
        self.assertIsNotNone(new_layout.tensor_map)

    def test_placement_layout(self):
        """Test placement layout."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        new_layout = layout(Shard(0), Replicate())
        self.assertIsNotNone(new_layout.placements)

    def test_placement_list_layout(self):
        """Test placement list layout."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        new_layout = layout([Shard(0), Replicate()])
        self.assertIsNotNone(new_layout.placements)


class TestLayoutProperties(_MockedLayoutTestCase):
    """Tests for LayoutProperties."""
    def test_mesh_property(self):
        """Test mesh property."""
        layout = self._make_layout((2,), ("dp",))
        self.assertIsNotNone(layout.mesh)

    def test_rank_list(self):
        """Test rank list."""
        layout = self._make_layout((2,), ("dp",))
        self.assertEqual(layout.rank_list, (0, 1))

    def test_rank_list_setter(self):
        """Test rank list setter."""
        layout = self._make_layout((2,), ("dp",))
        layout.rank_list = (1, 0)
        self.assertEqual(layout.rank_list, (1, 0))

    def test_mesh_shape(self):
        """Test mesh shape."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        self.assertEqual(layout.mesh_shape, (2, 4))

    def test_alias_name(self):
        """Test alias name."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        self.assertEqual(layout.alias_name, ("dp", "mp"))

    def test_is_partial(self):
        """Test is partial."""
        layout = self._make_layout((2,), ("dp",))
        self.assertFalse(layout.is_partial())
        layout._partial[0] = "sum"
        self.assertTrue(layout.is_partial())

    def test_reset_partial(self):
        """Test reset partial."""
        layout = self._make_layout((2,), ("dp",))
        layout.set_tensor_map((-1,))
        layout._partial[0] = "sum"
        layout.reset_partial()
        self.assertFalse(layout.is_partial())

    def test_alias_placements_with_tuple(self):
        """Test alias placements with tuple."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout._alias_tensor_map = (("dp", "mp"),)
        self.assertEqual(layout.alias_placements, (("dp", "mp"),))

    def test_alias_placements_without_tuple(self):
        """Test alias placements without tuple."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout._alias_tensor_map = ("dp", "None")
        layout._placements = [Shard(0), Replicate()]
        self.assertEqual(layout.alias_placements, [Shard(0), Replicate()])

    def test_compact_str(self):
        """Test compact str."""
        layout = self._make_layout((2,), ("dp",))
        self.assertIsInstance(layout.compact_str, str)


class TestLayoutDeepCopy(_MockedLayoutTestCase):
    """Tests for LayoutDeepCopy."""
    def test_deepcopy(self):
        """Test deepcopy."""
        import copy
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout.set_tensor_map((1, -1))
        layout_copy = copy.deepcopy(layout)
        self.assertEqual(layout_copy.tensor_map, layout.tensor_map)
        self.assertIsNot(layout_copy, layout)


class TestLayoutSetState(_MockedLayoutTestCase):
    """Tests for LayoutSetState."""
    def test_setstate(self):
        """Test setstate."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        state = layout.__dict__.copy()
        from hyper_parallel.core.dtensor.layout import Layout
        new_layout = Layout.__new__(Layout)
        new_layout.__setstate__(state)
        self.assertEqual(new_layout.mesh_shape, (2, 4))


class TestLayoutFromDeviceMesh(_MockedLayoutTestCase):
    """Tests for LayoutFromDeviceMesh."""
    def test_from_device_mesh(self):
        """Test from device mesh."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        mesh = layout.mesh
        from hyper_parallel.core.dtensor.layout import Layout
        new_layout = Layout.from_device_mesh(mesh)
        self.assertEqual(new_layout.mesh_shape, (2, 4))
        self.assertIsNone(new_layout.tensor_map)


class TestToDict(_MockedLayoutTestCase):
    """Tests for ToDict."""
    def test_basic(self):
        """Test basic."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        layout.set_tensor_map((1, -1))
        d = layout.to_dict()
        self.assertEqual(d["mesh_shape"], (2, 4))
        self.assertEqual(d["tensor_map"], (1, -1))

    def test_none_tensor_map_raises(self):
        """Test none tensor map raises."""
        layout = self._make_layout((2,), ("dp",))
        with self.assertRaises(ValueError):
            layout.to_dict()


class TestLayoutSliceHelpers(_MockedLayoutTestCase):
    """Tests for LayoutSliceHelpers."""
    def test_is_dev_axis_apply_shard(self):
        """Test is dev axis apply shard."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        new_layout = layout("dp", "None")
        self.assertTrue(new_layout.is_dev_axis_apply_shard("dp"))
        self.assertFalse(new_layout.is_dev_axis_apply_shard("mp"))

    def test_get_dev_axis_apply_shard_axis(self):
        """Test get dev axis apply shard axis."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        new_layout = layout("dp", "mp")
        self.assertEqual(new_layout.get_dev_axis_apply_shard_axis("dp"), 0)
        self.assertEqual(new_layout.get_dev_axis_apply_shard_axis("mp"), 1)

    def test_get_dev_axis_apply_shard_axis_not_found(self):
        """Test get dev axis apply shard axis not found."""
        layout = self._make_layout((2, 4), ("dp", "mp"))
        new_layout = layout("dp", "None")
        self.assertIsNone(new_layout.get_dev_axis_apply_shard_axis("mp"))


if __name__ == "__main__":
    unittest.main()
