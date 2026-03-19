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
"""Unit tests for TensorRedistribution functionality.

This module contains comprehensive unit tests for the TensorRedistribution class
in hyper_parallel.core.dtensor.tensor_redistribution. Tests cover reshape, all_reduce,
reduce_scatter, reduce_partial, and layout transformation logic. All tests use
mocking to avoid dependencies on actual distributed communication or hardware.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure repo root is importable when running this file directly in CI.
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_TEST_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch

from hyper_parallel.core.dtensor.tensor_redistribution import (
    TensorRedistribution,
    _construct_layout_tuple_for_transform_operator_list,
)


class TestTensorRedistributionConstructOps(unittest.TestCase):
    """Unit tests for TensorRedistribution construct operator methods.

    Tests cover _construct_reshape, _apply_eazy_redistribute, and
    _construct_all_to_all error handling.
    """

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.redistribution = TensorRedistribution()
        self.device = torch.device("cpu")

    def test_construct_reshape_reshapes_tensor_correctly(self):
        """Test _construct_reshape correctly reshapes tensor.

        Verifies that _construct_reshape applies view with given shape args
        and returns tensor with expected shape.
        """
        # Arrange
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=self.device)
        args = (4,)

        # Act
        result = self.redistribution._construct_reshape(x, *args)

        # Assert
        self.assertEqual(result.shape, (4,))
        self.assertTrue(torch.equal(result, x.view(4)))

    def test_construct_reshape_with_multiple_dims(self):
        """Test _construct_reshape with multi-dimensional output shape."""
        # Arrange
        x = torch.arange(8, dtype=torch.float32, device=self.device)
        args = (2, 4)

        # Act
        result = self.redistribution._construct_reshape(x, *args)

        # Assert
        self.assertEqual(result.shape, (2, 4))

    def test_apply_eazy_redistribute_returns_false_when_mesh_shape_differs(self):
        """Test _apply_eazy_redistribute returns False when mesh shapes differ."""
        # Arrange
        mock_src = MagicMock()
        mock_src.mesh_shape = (2, 2)
        mock_src.rank_list = [0, 1, 2, 3]
        mock_src.tensor_map = (0, 1)
        mock_dst = MagicMock()
        mock_dst.mesh_shape = (4,)  # Different
        mock_dst.rank_list = [0, 1, 2, 3]
        mock_dst.tensor_map = (0, 1)

        # Act
        result = self.redistribution._apply_eazy_redistribute(mock_src, mock_dst)

        # Assert
        self.assertFalse(result)

    def test_apply_eazy_redistribute_returns_false_when_rank_list_differs(self):
        """Test _apply_eazy_redistribute returns False when rank lists differ."""
        # Arrange
        mock_src = MagicMock()
        mock_src.mesh_shape = (2, 2)
        mock_src.rank_list = [0, 1, 2, 3]
        mock_src.tensor_map = (0, 1)
        mock_dst = MagicMock()
        mock_dst.mesh_shape = (2, 2)
        mock_dst.rank_list = [0, 1]  # Different
        mock_dst.tensor_map = (0, 1)

        # Act
        result = self.redistribution._apply_eazy_redistribute(mock_src, mock_dst)

        # Assert
        self.assertFalse(result)

    def test_apply_eazy_redistribute_returns_false_when_tensor_map_size_differs(self):
        """Test _apply_eazy_redistribute returns False when tensor_map lengths differ."""
        # Arrange
        mock_src = MagicMock()
        mock_src.mesh_shape = (2, 2)
        mock_src.rank_list = [0, 1, 2, 3]
        mock_src.tensor_map = (0, 1)
        mock_dst = MagicMock()
        mock_dst.mesh_shape = (2, 2)
        mock_dst.rank_list = [0, 1, 2, 3]
        mock_dst.tensor_map = (0,)  # Different length

        # Act
        result = self.redistribution._apply_eazy_redistribute(mock_src, mock_dst)

        # Assert
        self.assertFalse(result)

    def test_apply_eazy_redistribute_returns_true_when_compatible(self):
        """Test _apply_eazy_redistribute returns True when layouts are compatible."""
        # Arrange
        mock_src = MagicMock()
        mock_src.mesh_shape = (2, 2)
        mock_src.rank_list = [0, 1, 2, 3]
        mock_src.tensor_map = (0, 1)
        mock_dst = MagicMock()
        mock_dst.mesh_shape = (2, 2)
        mock_dst.rank_list = [0, 1, 2, 3]
        mock_dst.tensor_map = (1, 0)

        # Act
        result = self.redistribution._apply_eazy_redistribute(mock_src, mock_dst)

        # Assert
        self.assertTrue(result)

    @patch('hyper_parallel.core.dtensor.tensor_redistribution.platform')
    def test_construct_all_to_all_raises_when_dim_not_evenly_divisible(
        self, mock_platform
    ):
        """Test _construct_all_to_all raises ValueError when split dim not divisible."""
        # Arrange - mock create_group to avoid dist init; ValueError is raised
        # before any collective call (at dim_size % split_count check)
        mock_platform.create_group.return_value = MagicMock()
        x = torch.randn(7, 4, device=self.device)  # 7 not divisible by 2
        args = (0, 1, 2, [0, 1])  # split_dim=0, concat_dim=1, split_count=2

        # Act & Assert
        with self.assertRaises(ValueError) as ctx:
            self.redistribution._construct_all_to_all(x, *args)
        self.assertIn("cannot be evenly split", str(ctx.exception))


class TestTensorRedistributionAllReduce(unittest.TestCase):
    """Unit tests for _allreduce_along_dev_dim with mocked platform."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.redistribution = TensorRedistribution()
        self.device = torch.device("cpu")

    @patch('hyper_parallel.core.dtensor.tensor_redistribution.platform')
    def test_allreduce_along_dev_dim_sum_op(self, mock_platform):
        """Test _allreduce_along_dev_dim with sum op calls differentiable_all_reduce."""
        # Arrange
        x = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        mock_layout = MagicMock()
        mock_layout.get_comm_group_by_axis.return_value = MagicMock()
        mock_layout.mesh_shape = (2,)
        mock_layout.alias_name = ["dp"]
        mock_platform.differentiable_all_reduce.return_value = x.clone()

        # Act
        result = self.redistribution._allreduce_along_dev_dim(
            x, 'sum', mock_layout, 'dp'
        )

        # Assert
        mock_platform.differentiable_all_reduce.assert_called_once_with(
            x, 'sum', mock_layout.get_comm_group_by_axis.return_value
        )
        self.assertTrue(torch.equal(result, x))

    @patch('hyper_parallel.core.dtensor.tensor_redistribution.platform')
    def test_allreduce_along_dev_dim_avg_op_divides_by_dev_num(self, mock_platform):
        """Test _allreduce_along_dev_dim with avg op divides result by dev_num."""
        # Arrange
        x = torch.tensor([2.0, 4.0, 6.0], device=self.device)
        mock_layout = MagicMock()
        mock_layout.get_comm_group_by_axis.return_value = MagicMock()
        mock_layout.mesh_shape = (4,)
        mock_alias = MagicMock()
        mock_alias.index.return_value = 0
        mock_layout.alias_name = mock_alias
        mock_platform.differentiable_all_reduce.return_value = torch.tensor(
            [8.0, 16.0, 24.0], device=self.device
        )

        # Act
        result = self.redistribution._allreduce_along_dev_dim(
            x, 'avg', mock_layout, 'dp'
        )

        # Assert
        expected = torch.tensor([2.0, 4.0, 6.0], device=self.device)
        self.assertTrue(torch.allclose(result, expected))

    @patch('hyper_parallel.core.dtensor.tensor_redistribution.platform')
    def test_allreduce_along_dev_dim_zero_dim_tensor(self, mock_platform):
        """Test _allreduce_along_dev_dim handles 0-dim tensor correctly."""
        # Arrange
        x = torch.tensor(3.0, device=self.device)
        mock_layout = MagicMock()
        mock_layout.get_comm_group_by_axis.return_value = MagicMock()
        mock_layout.mesh_shape = (2,)
        mock_layout.alias_name = ["dp"]
        mock_platform.differentiable_all_reduce.return_value = torch.tensor(
            [6.0], device=self.device
        )

        # Act
        result = self.redistribution._allreduce_along_dev_dim(
            x, 'sum', mock_layout, 'dp'
        )

        # Assert
        self.assertEqual(result.dim(), 0)
        self.assertAlmostEqual(result.item(), 6.0)

    @patch('hyper_parallel.core.dtensor.tensor_redistribution.platform')
    def test_allreduce_along_dev_dim_all_op_casts_to_bool(self, mock_platform):
        """Test _allreduce_along_dev_dim with all op casts to int32, reduces, then back to bool."""
        # Arrange
        x = torch.tensor([True, False, True], device=self.device)
        mock_layout = MagicMock()
        mock_layout.get_comm_group_by_axis.return_value = MagicMock()
        mock_layout.mesh_shape = (2,)
        mock_layout.alias_name = ["dp"]
        mock_platform.tensor_type_cast.return_value = torch.tensor(
            [1, 0, 1], dtype=torch.int32, device=self.device
        )
        mock_platform.differentiable_all_reduce.return_value = torch.tensor(
            [1, 0, 1], dtype=torch.int32, device=self.device
        )

        # Act
        result = self.redistribution._allreduce_along_dev_dim(
            x, 'all', mock_layout, 'dp'
        )

        # Assert
        mock_platform.tensor_type_cast.assert_called_once()
        mock_platform.differentiable_all_reduce.assert_called_once_with(
            mock_platform.tensor_type_cast.return_value,
            'all',
            mock_layout.get_comm_group_by_axis.return_value
        )
        self.assertEqual(result.dtype, torch.bool)
        self.assertTrue(torch.equal(result, torch.tensor([True, False, True])))


class TestTensorRedistributionReduceScatter(unittest.TestCase):
    """Unit tests for _reduce_scatter_along_dev_dim_with_axis.

    Verifies that the method correctly calls platform.differentiable_reduce_scatter
    (not self.platform.reduce_scatter) with proper parameters.
    """

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.redistribution = TensorRedistribution()
        self.device = torch.device("cpu")

    @patch('hyper_parallel.core.dtensor.tensor_redistribution.platform')
    def test_reduce_scatter_along_dev_dim_calls_differentiable_reduce_scatter(
        self, mock_platform
    ):
        """Test _reduce_scatter_along_dev_dim_with_axis calls platform.differentiable_reduce_scatter.

        Verifies the fix for the bug where self.platform.reduce_scatter was used
        instead of platform.differentiable_reduce_scatter.
        """
        # Arrange
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=self.device)
        mock_layout = MagicMock()
        mock_layout.mesh_shape = (2,)
        mock_alias = MagicMock()
        mock_alias.index.return_value = 0
        mock_layout.alias_name = mock_alias
        mock_layout.get_comm_group_by_axis.return_value = MagicMock()
        mock_output = torch.tensor([[2.0, 4.0]], device=self.device)
        mock_platform.differentiable_reduce_scatter.return_value = mock_output

        # Act
        result = self.redistribution._reduce_scatter_along_dev_dim_with_axis(
            x, axis=0, op='sum', layout=mock_layout, dev_dim='dp'
        )

        # Assert
        mock_platform.differentiable_reduce_scatter.assert_called_once_with(
            x, 2, 0, 'sum', mock_layout.get_comm_group_by_axis.return_value
        )
        self.assertTrue(torch.equal(result, mock_output))


class TestTensorRedistributionReducePartial(unittest.TestCase):
    """Unit tests for reduce_partial with mocked dependencies."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.redistribution = TensorRedistribution()
        self.device = torch.device("cpu")

    def test_reduce_partial_returns_unchanged_when_not_partial(self):
        """Test reduce_partial returns input unchanged when layout is not partial."""
        # Arrange
        mock_input = MagicMock()
        mock_input.layout = MagicMock()
        mock_input.layout.is_partial.return_value = False

        # Act
        result = self.redistribution.reduce_partial(mock_input, MagicMock())

        # Assert
        self.assertEqual(result, mock_input)

    def test_reduce_partial_returns_unchanged_when_layout_is_none(self):
        """Test reduce_partial returns input when from_layout is None."""
        # Arrange
        mock_input = MagicMock()
        mock_input.layout = None

        # Act
        result = self.redistribution.reduce_partial(mock_input, MagicMock())

        # Assert
        self.assertEqual(result, mock_input)

    def test_reduce_partial_raises_when_mesh_shape_mismatch(self):
        """Test reduce_partial raises ValueError when mesh shapes differ."""
        # Arrange
        mock_input = MagicMock()
        mock_input.layout = MagicMock()
        mock_input.layout.is_partial.return_value = True
        mock_input.layout.mesh_shape = (2, 2)
        mock_input.to_local.return_value = torch.tensor([1.0], device=self.device)
        mock_to_layout = MagicMock()
        mock_to_layout.mesh_shape = (4,)
        mock_to_layout.is_partial.return_value = False

        # Act & Assert
        with self.assertRaises(ValueError) as ctx:
            self.redistribution.reduce_partial(mock_input, mock_to_layout)
        self.assertIn("mesh_shape", str(ctx.exception))

    def test_reduce_partial_raises_when_to_layout_is_partial(self):
        """Test reduce_partial raises ValueError when to_layout is partial."""
        # Arrange
        mock_input = MagicMock()
        mock_input.layout = MagicMock()
        mock_input.layout.is_partial.return_value = True
        mock_input.layout.mesh_shape = (2, 2)
        mock_input.to_local.return_value = torch.tensor([1.0], device=self.device)
        mock_to_layout = MagicMock()
        mock_to_layout.mesh_shape = (2, 2)
        mock_to_layout.is_partial.return_value = True

        # Act & Assert
        with self.assertRaises(ValueError) as ctx:
            self.redistribution.reduce_partial(mock_input, mock_to_layout)
        self.assertIn("to_layout must be non-partial", str(ctx.exception))


class TestConstructLayoutTuple(unittest.TestCase):
    """Unit tests for _construct_layout_tuple_for_transform_operator_list."""

    def test_construct_layout_tuple_returns_correct_structure(self):
        """Test _construct_layout_tuple returns tuples with mesh_shape, tensor_map, shape."""
        # Arrange
        mock_from = MagicMock()
        mock_from.to_dict.return_value = {
            "mesh_shape": (2, 2),
            "tensor_map": (0, 1),
        }
        mock_to = MagicMock()
        mock_to.to_dict.return_value = {
            "mesh_shape": (2, 2),
            "tensor_map": (1, 0),
        }
        from_full_shape = (8, 16)

        # Act
        from_tuple, to_tuple = _construct_layout_tuple_for_transform_operator_list(
            mock_from, mock_to, from_full_shape
        )

        # Assert
        self.assertEqual(from_tuple[0], (2, 2))
        self.assertEqual(from_tuple[1], (0, 1))
        self.assertEqual(from_tuple[2], [8, 16])
        self.assertEqual(to_tuple[0], (2, 2))
        self.assertEqual(to_tuple[1], (1, 0))
        self.assertEqual(to_tuple[2], [8, 16])
