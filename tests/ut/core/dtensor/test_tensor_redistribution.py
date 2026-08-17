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
"""Coverage supplement tests for hyper_parallel.core.dtensor.tensor_redistribution."""

import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import unittest
from unittest.mock import patch, MagicMock, Mock

import torch
import numpy as np

from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


def _setup_mock_dm_platform(platform_mock):
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


class _MockedTestCase(unittest.TestCase):
    """Base class that patches both device_mesh and tensor_redistribution platforms."""

    def setUp(self):
        """Set up test fixtures."""
        self.patcher_dm = patch("hyper_parallel.core.dtensor.device_mesh.platform")
        self.patcher_dm_tensor = patch("hyper_parallel.core.dtensor.device_mesh.Tensor", torch.Tensor)
        self.patcher_tr = patch("hyper_parallel.core.dtensor.tensor_redistribution.platform")

        self.mock_dm_platform = self.patcher_dm.start()
        self.patcher_dm_tensor.start()
        self.mock_tr_platform = self.patcher_tr.start()

        _setup_mock_dm_platform(self.mock_dm_platform)
        # Also setup tr platform for rank
        self.mock_tr_platform.get_rank.return_value = 0

        _DEVICE_MESH_MAP.clear()
        EXISTING_COMM_GROUPS.clear()

        self.addCleanup(self.patcher_dm.stop)
        self.addCleanup(self.patcher_dm_tensor.stop)
        self.addCleanup(self.patcher_tr.stop)
        self.addCleanup(_DEVICE_MESH_MAP.clear)
        self.addCleanup(EXISTING_COMM_GROUPS.clear)

    def _make_layout(self, mesh_shape, alias_name, rank_list=None):
        """ make layout."""
        from hyper_parallel.core.dtensor.layout import Layout
        return Layout(mesh_shape, alias_name, rank_list=rank_list, init_backend=False)


class TestUnevenShardRedistribution(_MockedTestCase):
    """Tests for the intentionally unsupported live uneven-shard path."""

    def test_redistribution_rejects_uneven_shard_layout(self):
        """Live redistribution should defer uneven shards to FSDP collectives."""
        from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
        from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution

        source_layout = self._make_layout((2,), ("fsdp",), rank_list=(0, 1))
        source_layout.set_placements((Shard(0, uneven_shard=True),))
        source_layout.placement_to_tensor_map(dim=1)
        target_layout = self._make_layout((2,), ("fsdp",), rank_list=(0, 1))
        target_layout.set_placements((Replicate(),))
        target_layout.placement_to_tensor_map(dim=1)
        input_tensor = MagicMock()
        input_tensor.layout = source_layout

        with self.assertRaisesRegex(NotImplementedError, "uneven chunk-sharded"):
            TensorRedistribution().redistribution(input_tensor, target_layout)


# ===========================================================================
# TensorRedistribution method tests
# ===========================================================================

class TestConstructReshape(_MockedTestCase):
    """Tests for ConstructReshape."""
    def test_basic(self):
        """Test basic."""
        from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution
        tr = TensorRedistribution()
        x = MagicMock()
        x.view.return_value = "reshaped"
        result = tr._construct_reshape(x, 2, 3)
        x.view.assert_called_once_with((2, 3))
        self.assertEqual(result, "reshaped")


class TestConstructAllConcat(_MockedTestCase):
    """Tests for ConstructAllConcat."""
    def test_basic(self):
        """Test basic."""
        from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution
        tr = TensorRedistribution()
        x = MagicMock()
        self.mock_tr_platform.create_group.return_value = "group"
        self.mock_tr_platform.differentiable_all_gather_concat.return_value = "gathered"

        result = tr._construct_all_concat(x, 0, 1, 2, 3)
        # rank_list = (0, 1, 2), concat_dim = 3
        self.mock_tr_platform.create_group.assert_called_once_with((0, 1, 2))
        self.mock_tr_platform.differentiable_all_gather_concat.assert_called_once()
        self.assertEqual(result, "gathered")


class TestConstructStridedSlice(_MockedTestCase):
    """Tests for ConstructStridedSlice."""
    def test_basic(self):
        """Test basic."""
        from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution
        tr = TensorRedistribution()
        x = MagicMock()
        self.mock_tr_platform.construct_strided_slice.return_value = "sliced"
        # args: begin0, begin1, end0, end1, stride0, stride1
        result = tr._construct_strided_slice(x, 0, 0, 4, 4, 1, 1)
        self.mock_tr_platform.construct_strided_slice.assert_called_once()
        self.assertEqual(result, "sliced")


class TestConstructAllSplit(_MockedTestCase):
    """Tests for ConstructAllSplit."""
    def test_basic(self):
        """Test basic."""
        from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution
        tr = TensorRedistribution()
        tr.rank_id = 0
        x = MagicMock()
        self.mock_tr_platform.chunk.return_value = "chunked"

        # args: (split_dim, split_size, group)
        result = tr._construct_all_split(x, 0, 4, [0, 1, 2, 3])
        self.mock_tr_platform.chunk.assert_called_once_with(x, 0, 4, 0)
        self.assertEqual(result, "chunked")


class TestConstructAllToAll(_MockedTestCase):
    """Tests for ConstructAllToAll."""
    def test_same_dim(self):
        """Test same dim."""
        from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution
        tr = TensorRedistribution()
        # Create a real tensor for shape inspection
        x = torch.randn(8, 4)
        self.mock_tr_platform.create_group.return_value = "group"
        self.mock_tr_platform.differentiable_all_to_all.return_value = torch.randn(8, 4)

        # split_dim == concat_dim
        result = tr._construct_all_to_all(x, 0, 0, 2, [0, 1])
        self.mock_tr_platform.create_group.assert_called_once()
        self.mock_tr_platform.differentiable_all_to_all.assert_called_once()

    def test_different_dim(self):
        """Test different dim."""
        from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution
        tr = TensorRedistribution()
        x = torch.randn(8, 4)
        self.mock_tr_platform.create_group.return_value = "group"
        self.mock_tr_platform.differentiable_all_to_all.return_value = torch.randn(8, 4)

        result = tr._construct_all_to_all(x, 0, 1, 2, [0, 1])
        self.mock_tr_platform.create_group.assert_called_once()

    def test_uneven_split_raises(self):
        """Test uneven split raises."""
        from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution
        tr = TensorRedistribution()
        x = torch.randn(7, 4)  # 7 can't be evenly split into 2
        self.mock_tr_platform.create_group.return_value = "group"

        with self.assertRaises(ValueError):
            tr._construct_all_to_all(x, 0, 1, 2, [0, 1])


# ===========================================================================
# _apply_eazy_redistribute tests
# ===========================================================================

class TestApplyEazyRedistribute(_MockedTestCase):
    """Tests for ApplyEazyRedistribute."""
    def test_same_layout_returns_true(self):
        """Test same layout returns true."""
        from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution
        tr = TensorRedistribution()

        src = MagicMock()
        src.mesh_shape = (2, 4)
        src.rank_list = (0, 1, 2, 3, 4, 5, 6, 7)
        src.tensor_map = (1, -1)

        dst = MagicMock()
        dst.mesh_shape = (2, 4)
        dst.rank_list = (0, 1, 2, 3, 4, 5, 6, 7)
        dst.tensor_map = (-1, 0)

        self.assertTrue(tr._apply_eazy_redistribute(src, dst))

    def test_different_mesh_shape_returns_false(self):
        """Test different mesh shape returns false."""
        from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution
        tr = TensorRedistribution()

        src = MagicMock()
        src.mesh_shape = (2, 4)
        src.rank_list = (0, 1, 2, 3, 4, 5, 6, 7)

        dst = MagicMock()
        dst.mesh_shape = (4, 2)
        dst.rank_list = (0, 1, 2, 3, 4, 5, 6, 7)

        self.assertFalse(tr._apply_eazy_redistribute(src, dst))

    def test_different_rank_list_returns_false(self):
        """Test different rank list returns false."""
        from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution
        tr = TensorRedistribution()

        src = MagicMock()
        src.mesh_shape = (2, 4)
        src.rank_list = (0, 1, 2, 3, 4, 5, 6, 7)

        dst = MagicMock()
        dst.mesh_shape = (2, 4)
        dst.rank_list = (7, 6, 5, 4, 3, 2, 1, 0)

        self.assertFalse(tr._apply_eazy_redistribute(src, dst))

    def test_different_tensor_map_len_returns_false(self):
        """Test different tensor map len returns false."""
        from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution
        tr = TensorRedistribution()

        src = MagicMock()
        src.mesh_shape = (2, 4)
        src.rank_list = (0, 1, 2, 3, 4, 5, 6, 7)
        src.tensor_map = (1, -1)

        dst = MagicMock()
        dst.mesh_shape = (2, 4)
        dst.rank_list = (0, 1, 2, 3, 4, 5, 6, 7)
        dst.tensor_map = (1, -1, 0)

        self.assertFalse(tr._apply_eazy_redistribute(src, dst))


# ===========================================================================
# _allreduce_along_dev_dim tests
# ===========================================================================

class TestAllreduceAlongDevDim(_MockedTestCase):
    """Tests for AllreduceAlongDevDim."""
    def _make_mock_layout(self):
        """ make mock layout."""
        layout = MagicMock()
        layout.mesh_shape = (2, 4)
        layout.alias_name = ("dp", "mp")
        layout.get_comm_group_by_axis.return_value = "group"
        return layout

    def test_sum_op(self):
        """Test sum op."""
        from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution
        layout = self._make_mock_layout()
        x = torch.randn(4, 4)
        self.mock_tr_platform.differentiable_all_reduce.return_value = x

        result = TensorRedistribution._allreduce_along_dev_dim(x, 'sum', layout, 'dp')
        self.mock_tr_platform.differentiable_all_reduce.assert_called_once_with(x, 'sum', 'group')

    def test_avg_op(self):
        """Test avg op."""
        from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution
        layout = self._make_mock_layout()
        x = torch.randn(4, 4)
        self.mock_tr_platform.differentiable_all_reduce.return_value = x

        result = TensorRedistribution._allreduce_along_dev_dim(x, 'avg', layout, 'dp')
        self.mock_tr_platform.differentiable_all_reduce.assert_called_once_with(x, 'sum', 'group')

    def test_all_op(self):
        """Test all op."""
        from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution
        layout = self._make_mock_layout()
        x = torch.tensor([True, False, True])
        self.mock_tr_platform.tensor_type_cast.return_value = torch.tensor([1, 0, 1])
        self.mock_tr_platform.differentiable_all_reduce.return_value = torch.tensor([1, 0, 1])

        result = TensorRedistribution._allreduce_along_dev_dim(x, 'all', layout, 'dp')
        self.mock_tr_platform.tensor_type_cast.assert_called_once()
        self.mock_tr_platform.differentiable_all_reduce.assert_called_once()

    def test_zero_dim_tensor(self):
        """Test zero dim tensor."""
        from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution
        layout = self._make_mock_layout()
        x = torch.tensor(5.0)  # 0-dim
        self.mock_tr_platform.differentiable_all_reduce.return_value = torch.tensor([5.0])

        result = TensorRedistribution._allreduce_along_dev_dim(x, 'sum', layout, 'dp')
        self.assertEqual(result.dim(), 0)


# ===========================================================================
# _construct_layout_tuple_for_transform_operator_list tests
# ===========================================================================

class TestConstructLayoutTuple(_MockedTestCase):
    """Tests for ConstructLayoutTuple."""
    def test_basic(self):
        """Test basic."""
        from hyper_parallel.core.dtensor.tensor_redistribution import (
            _construct_layout_tuple_for_transform_operator_list,
        )
        from_layout = MagicMock()
        from_layout.to_dict.return_value = {
            "mesh_shape": (2, 4),
            "tensor_map": (1, -1),
            "interleaved_parallel": False,
            "alias_name": ("dp", "mp"),
            "rank_list": (0, 1, 2, 3, 4, 5, 6, 7),
        }
        to_layout = MagicMock()
        to_layout.to_dict.return_value = {
            "mesh_shape": (2, 4),
            "tensor_map": (-1, 0),
            "interleaved_parallel": False,
            "alias_name": ("dp", "mp"),
            "rank_list": (0, 1, 2, 3, 4, 5, 6, 7),
        }
        from_tuple, to_tuple = _construct_layout_tuple_for_transform_operator_list(
            from_layout, to_layout, (16, 8)
        )
        self.assertEqual(from_tuple[0], (2, 4))
        self.assertEqual(from_tuple[1], (1, -1))
        self.assertEqual(to_tuple[0], (2, 4))
        self.assertEqual(to_tuple[1], (-1, 0))
        # Both use from_full_shape
        self.assertEqual(from_tuple[2], [16, 8])
        self.assertEqual(to_tuple[2], [16, 8])


# ===========================================================================
# TensorRedistribution operator mapping tests
# ===========================================================================

class TestTensorRedistributionInit(_MockedTestCase):
    """Tests for TensorRedistributionInit."""
    def test_operator_mapping(self):
        """Test operator mapping."""
        from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution
        tr = TensorRedistribution()
        self.assertIn("Reshape", tr._construct_op_operator)
        self.assertIn("AllConcat", tr._construct_op_operator)
        self.assertIn("StridedSlice", tr._construct_op_operator)
        self.assertIn("all_concat", tr._construct_op_operator)
        self.assertIn("all_split", tr._construct_op_operator)
        self.assertIn("all_to_all", tr._construct_op_operator)
        self.assertFalse(tr.is_init)
        self.assertIsNone(tr.rank_id)


class TestConstructAllConcatNew(_MockedTestCase):
    """Tests for ConstructAllConcatNew."""
    def test_static_method(self):
        """Test static method."""
        from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution
        x = MagicMock()
        self.mock_tr_platform.create_group.return_value = "group"
        self.mock_tr_platform.differentiable_all_gather_concat.return_value = "gathered"

        # args: (concat_dim, concat_size, group_rank_list)
        result = TensorRedistribution._construct_all_concat_new(x, 0, 4, [0, 1, 2, 3])
        self.mock_tr_platform.create_group.assert_called_once_with([0, 1, 2, 3])
        self.mock_tr_platform.differentiable_all_gather_concat.assert_called_once()


if __name__ == "__main__":
    unittest.main()
