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
"""Unit tests for DeviceMesh functionality.

This module provides comprehensive unit tests for the DeviceMesh class and
related functions in the hyper_parallel framework. All tests use mocking to avoid
dependencies on actual hardware or distributed communication.
"""
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

# Set platform to torch for testing
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.platform import get_platform
from hyper_parallel.core.device_mesh import (
    DeviceMesh,
    _get_sub_rank_list,
    _create_device_mesh,
    init_device_mesh,
    _DEVICE_MESH_MAP,
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestDeviceMesh(unittest.TestCase):
    """Unit tests for DeviceMesh class and related functions."""

    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        # Clear global caches
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

        # Initialize platform
        self.platform = get_platform()

    def tearDown(self):
        """Clean up after each test method.

        Ensures global caches are cleared after each test to prevent
        test interference.
        """
        # Clear global caches
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_init_device_mesh_basic(self, mock_platform):
        """Test basic DeviceMesh construction with explicit mesh.

        Scenario: Create a 2x2 DeviceMesh directly with explicit mesh tensor.
        Expected behavior: DeviceMesh should be created with correct shape,
        dimension names, and rank list.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange: Set up mock platform behavior
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_group = MagicMock()
        mock_platform.split_group.return_value = mock_group
        mock_platform.tensor_to_numpy.side_effect = lambda t: np.array(t)

        # Act - use DeviceMesh directly to avoid caching issues with init_device_mesh
        mesh = DeviceMesh(
            device_type="npu",
            mesh=[[0, 1], [2, 3]],
            mesh_dim_names=("dp", "tp"),
            _init_backend=False,
        )

        # Assert: Verify DeviceMesh properties
        self.assertEqual(mesh.mesh_shape, (2, 2))
        self.assertEqual(mesh.mesh_dim_names, ("dp", "tp"))
        self.assertEqual(mesh.rank_list, (0, 1, 2, 3))
        self.assertEqual(mesh.ndim, 2)
        self.assertEqual(mesh.rank, 0)

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_init_device_mesh_caching(self, mock_platform):
        """Test that init_device_mesh caches results correctly.

        Scenario: Call init_device_mesh twice with same parameters.
        Expected behavior: Second call should return the same cached instance.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_group = MagicMock()
        mock_platform.split_group.return_value = mock_group

        # Act
        mesh1 = init_device_mesh("npu", (2, 2), mesh_dim_names=("dp", "tp"))
        mesh2 = init_device_mesh("npu", (2, 2), mesh_dim_names=("dp", "tp"))

        # Assert - same parameters should return same cached instance
        self.assertIs(mesh1, mesh2)

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_with_tensor(self, mock_platform):
        """Test DeviceMesh construction with custom mesh tensor.

        Scenario: Create DeviceMesh with a 2x3 tensor mesh.
        Expected behavior: DeviceMesh should have correct shape (2, 3).

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = self.platform.tensor([[0, 2], [1, 3]])

        # Act
        device_mesh = DeviceMesh("npu", mesh, mesh_dim_names=("dp", "tp"))

        # Assert
        self.assertEqual(device_mesh.mesh_shape, (2, 2))
        self.assertEqual(device_mesh.mesh_dim_names, ("dp", "tp"))
        self.assertEqual(device_mesh.rank_list, (0, 2, 1, 3))
        self.assertEqual(device_mesh.ndim, 2)

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_with_list(self, mock_platform):
        """Test DeviceMesh construction with list input."""
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        # Act
        device_mesh = DeviceMesh("npu", [[0, 2], [1, 3]], mesh_dim_names=("dp", "tp"))

        # Assert
        self.assertEqual(device_mesh.mesh_shape, (2, 2))
        self.assertEqual(device_mesh.rank_list, (0, 2, 1, 3))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_with_numpy(self, mock_platform):
        """Test DeviceMesh construction with numpy array mesh.

        Scenario: Create DeviceMesh with numpy array mesh.
        Expected behavior: DeviceMesh should be created successfully.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = np.array([[0, 2], [1, 3]], dtype=np.int64)

        # Act
        device_mesh = DeviceMesh("npu", mesh, mesh_dim_names=("dp", "tp"))

        # Assert
        self.assertEqual(device_mesh.mesh_shape, (2, 2))
        self.assertEqual(device_mesh.rank_list, (0, 2, 1, 3))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_getitem_single_dim(self, mock_platform):
        """Test DeviceMesh __getitem__ with single dimension."""
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "tp"),
        )

        # Act
        dp_mesh = mesh["dp"]
        tp_mesh = mesh["tp"]

        # Assert
        self.assertEqual(dp_mesh.mesh_shape, (2,))
        self.assertEqual(dp_mesh.mesh_dim_names, ("dp",))
        self.assertEqual(dp_mesh.root_mesh, mesh)
        self.assertEqual(dp_mesh.rank_list, (0, 4))

        self.assertEqual(tp_mesh.mesh_shape, (4,))
        self.assertEqual(tp_mesh.mesh_dim_names, ("tp",))
        self.assertEqual(tp_mesh.root_mesh, mesh)
        self.assertEqual(tp_mesh.rank_list, (0, 1, 2, 3))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_getitem_multiple_dims(self, mock_platform):
        """Test DeviceMesh __getitem__ with multiple dimensions."""
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "cp", "tp"),
        )

        # Act
        dp_cp_mesh = mesh[("dp", "cp")]

        # Assert
        self.assertEqual(dp_cp_mesh.mesh_shape, (2, 2))
        self.assertEqual(dp_cp_mesh.mesh_dim_names, ("dp", "cp"))
        self.assertEqual(dp_cp_mesh.rank_list, (0, 2, 4, 6))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_get_local_rank(self, mock_platform):
        """Test DeviceMesh get_local_rank method.

        Scenario: Get local rank within a specific mesh dimension.
        Expected behavior: Should return the local rank within the specified dimension.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "tp"),
        )

        # Act & Assert for rank 0
        self.assertEqual(mesh.get_local_rank("dp"), 0)
        self.assertEqual(mesh.get_local_rank("tp"), 0)
        self.assertEqual(mesh.get_local_rank(0), 0)
        self.assertEqual(mesh.get_local_rank(1), 0)

        # Test with mocked rank 5
        with patch.object(mesh, "_rank", 5):
            self.assertEqual(mesh.get_local_rank("dp"), 1)
            self.assertEqual(mesh.get_local_rank(1), 1)

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_flatten(self, mock_platform):
        """Test DeviceMesh flatten method.

        Scenario: Flatten a multi-dimensional mesh into a 1D mesh.
        Expected behavior: Should return a new DeviceMesh with flattened dimensions.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "cp", "tp"),
        )

        dp_cp_mesh = mesh[("dp", "cp")]

        # Act
        flat_mesh = dp_cp_mesh.flatten()

        # Assert
        self.assertEqual(flat_mesh.mesh_shape, (4,))
        self.assertEqual(flat_mesh.mesh_dim_names, ("dp_cp",))
        self.assertEqual(flat_mesh.rank_list, (0, 2, 4, 6))
        self.assertEqual(flat_mesh.root_mesh, mesh)

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_size(self, mock_platform):
        """Test DeviceMesh size method.

        Scenario: Get the total size of the mesh or size along a specific dimension.
        Expected behavior: size() should return total elements, size(dim) should return elements along that dimension.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "tp"),
        )

        # Act & Assert
        self.assertEqual(mesh.size(), 8)
        self.assertEqual(mesh.size(0), 2)
        self.assertEqual(mesh.size(1), 4)

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_properties(self, mock_platform):
        """Test DeviceMesh basic properties.

        Scenario: Access various properties of DeviceMesh instance.
        Expected behavior: Properties should return expected values for device_type, shape, mesh_shape, ndim, and rank.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "tp"),
        )

        # Act & Assert
        self.assertEqual(mesh.device_type, "npu")
        self.assertEqual(mesh.shape, (2, 4))
        self.assertEqual(mesh.mesh_shape, (2, 4))
        self.assertEqual(mesh.ndim, 2)
        self.assertEqual(mesh.rank, 0)
        # rank_list contains all ranks from 0 to world_size-1
        self.assertEqual(len(mesh.rank_list), 8)

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_repr(self, mock_platform):
        """Test DeviceMesh __repr__ method.

        Scenario: Get string representation of DeviceMesh instance.
        Expected behavior: __repr__ should return a string containing DeviceMesh class name and key properties.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2),
            mesh_dim_names=("dp", "tp"),
        )

        # Act
        repr_str = repr(mesh)

        # Assert - just check it contains expected substrings
        self.assertIn("DeviceMesh", repr_str)
        self.assertIn("device_type='npu'", repr_str)
        self.assertIn("mesh_shape=(2, 2)", repr_str)

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_axis_methods(self, mock_platform):
        """Test DeviceMesh axis-related methods.

        Scenario: Access axis-related information using axis_id, axis_index, and get_device_num_along_axis methods.
        Expected behavior: Methods should return correct dimension indices and device counts.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "tp"),
        )

        # Act & Assert
        self.assertEqual(mesh.axis_id("dp"), 1)
        self.assertEqual(mesh.axis_id("tp"), 0)
        self.assertEqual(mesh.axis_index("dp"), 0)
        self.assertEqual(mesh.axis_index("tp"), 1)
        self.assertEqual(mesh.get_device_num_along_axis("dp"), 2)
        self.assertEqual(mesh.get_device_num_along_axis("tp"), 4)

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_get_rank_list_along_axis(self, mock_platform):
        """Test DeviceMesh get_rank_list_along_axis method.

        Scenario: Get the list of ranks along a specific axis for the current rank.
        Expected behavior: Should return list of ranks that are in the same slice along the specified axis.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "tp"),
        )

        # Act
        rank_list_dp = mesh.get_rank_list_along_axis("dp")
        rank_list_tp = mesh.get_rank_list_along_axis("tp")

        # Assert
        self.assertEqual(rank_list_dp, [0, 4])
        self.assertEqual(rank_list_tp, [0, 1, 2, 3])

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_get_global_shape(self, mock_platform):
        """Test DeviceMesh get_global_shape method.

        Scenario: Calculate global tensor shape from local slice shape and tensor mapping.
        Expected behavior: Should return global shape accounting for sharding across mesh dimensions.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "tp"),
        )

        # Act - simulate tensor sharding where dimension 0 is sharded across dp (dim 1) and
        # dimension 1 is sharded across tp (dim 0 in mesh coordinate, reverse order)
        slice_shape = (4, 8)
        # Mapping: -1 = replicated, 0/1 = sharded across that mesh dimension
        # In reverse mesh order: tp is 0, dp is 1
        tensor_map = (1, 0)  # dim 0 sharded on dp, dim 1 sharded on tp

        global_shape = mesh.get_global_shape(slice_shape, tensor_map)

        # Assert
        # Expected: (4 * 2, 8 * 4) = (8, 32)
        self.assertEqual(global_shape, (8, 32))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_get_sub_rank_list(self, mock_platform):
        """Test _get_sub_rank_list helper function.

        Scenario: Extract sub-rank list for a sub-mesh from original mesh.
        Expected behavior: Should return list of ranks in the sub-mesh slice containing current rank.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mesh_shape = (2, 4)
        mesh_dim_names = ("dp", "tp")
        rank_list = (0, 1, 2, 3, 4, 5, 6, 7)
        sub_mesh_dim_names = ("dp",)
        current_rank = 0

        # Act
        sub_rank_list = _get_sub_rank_list(
            mesh_shape, mesh_dim_names, rank_list, sub_mesh_dim_names, current_rank
        )

        # Assert
        self.assertEqual(sub_rank_list, [0, 4])

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_with_none_mesh(self, mock_platform):
        """Test DeviceMesh with mesh=None (auto 1D mesh).

        Scenario: Create DeviceMesh without explicit mesh (mesh=None).
        Expected behavior: DeviceMesh should auto-generate 1D mesh based on world_size.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        # Act
        device_mesh = DeviceMesh("npu", mesh=None, _init_backend=False)

        # Assert
        self.assertEqual(device_mesh.mesh_shape, (4,))
        self.assertEqual(device_mesh.ndim, 1)
        self.assertEqual(device_mesh.rank_list, (0, 1, 2, 3))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_invalid_mesh_type(self, mock_platform):
        """Test DeviceMesh with invalid mesh type raises TypeError.

        Scenario: Create DeviceMesh with invalid mesh type (integer).
        Expected behavior: Should raise TypeError with appropriate message.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 1
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        # Act & Assert - scalar mesh should raise TypeError (not ValueError)
        # because 0 is not a valid Tensor, list, tuple or numpy array
        with self.assertRaises(TypeError) as context:
            DeviceMesh("npu", mesh=0, _init_backend=False)
        self.assertIn("mesh must be Tensor, list, tuple or numpy array", str(context.exception))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_invalid_mesh_dim_names_length(self, mock_platform):
        """Test DeviceMesh with mismatched mesh_dim_names length raises ValueError.

        Scenario: Create DeviceMesh with 2D mesh but 3 mesh_dim_names.
        Expected behavior: Should raise ValueError with appropriate message.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        # Act & Assert - mismatched dimension count should raise ValueError
        with self.assertRaises(ValueError) as context:
            DeviceMesh("npu", mesh=[[0, 1], [2, 3]], mesh_dim_names=("dp", "tp", "cp"), _init_backend=False)
        self.assertIn("mesh dimensions", str(context.exception))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_invalid_mesh_dim_names_duplicate(self, mock_platform):
        """Test DeviceMesh with duplicate mesh_dim_names raises ValueError.

        Scenario: Create DeviceMesh with duplicate dimension names.
        Expected behavior: Should raise ValueError with appropriate message.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        # Act & Assert - duplicate dimension names should raise ValueError
        with self.assertRaises(ValueError) as context:
            DeviceMesh("npu", mesh=[[0, 1], [2, 3]], mesh_dim_names=("dp", "dp"), _init_backend=False)
        self.assertIn("Each element of mesh_dim_names", str(context.exception))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_getitem_no_dim_names_raises(self, mock_platform):
        """Test DeviceMesh __getitem__ without mesh_dim_names raises RuntimeError.

        Scenario: Access submesh without setting mesh_dim_names.
        Expected behavior: Should raise RuntimeError with appropriate message.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = DeviceMesh("npu", mesh=[0, 1, 2, 3], _init_backend=False)

        # Act & Assert - accessing submesh without dim names should raise
        with self.assertRaises(RuntimeError) as context:
            _ = mesh["dp"]
        self.assertIn("Cannot slice a DeviceMesh without mesh_dim_names", str(context.exception))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_getitem_invalid_dim_name(self, mock_platform):
        """Test DeviceMesh __getitem__ with invalid dimension name raises KeyError.

        Scenario: Access submesh with invalid dimension name.
        Expected behavior: Should raise KeyError with appropriate message.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2),
            mesh_dim_names=("dp", "tp"),
        )

        # Act & Assert - invalid dimension name should raise
        with self.assertRaises(KeyError) as context:
            _ = mesh["invalid_dim"]
        self.assertIn("invalid_dim", str(context.exception))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_getitem_out_of_order(self, mock_platform):
        """Test DeviceMesh __getitem__ with out-of-order dimensions raises ValueError.

        Scenario: Access submesh with out-of-order dimension names.
        Expected behavior: Should raise ValueError with appropriate message.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "cp", "tp"),
        )

        # Act & Assert - out of order dimension should raise
        with self.assertRaises(ValueError) as context:
            _ = mesh[("cp", "dp")]  # Wrong order, should be ("dp", "cp")
        self.assertIn("must follow the order", str(context.exception))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_size_method(self, mock_platform):
        """Test DeviceMesh size method.

        Scenario: Get total size and size along specific dimensions.
        Expected behavior: size() should return total elements, size(dim)
        should return elements along that dimension.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "tp"),
        )

        # Act & Assert
        self.assertEqual(mesh.size(), 8)
        self.assertEqual(mesh.size(0), 2)
        self.assertEqual(mesh.size(1), 4)

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_get_coordinate(self, mock_platform):
        """Test DeviceMesh get_coordinate method.

        Scenario: Get coordinate of current rank in the mesh.
        Expected behavior: Should return coordinate tuple for current rank.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "tp"),
        )

        # Act
        coordinate = mesh.get_coordinate()

        # Assert - rank 0 in 2x4 mesh should be at position (0, 0)
        self.assertEqual(coordinate, (0, 0))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_root_mesh_property(self, mock_platform):
        """Test DeviceMesh root_mesh property.

        Scenario: Access root_mesh property of parent and child meshes.
        Expected behavior: Parent mesh should have None, child should reference parent.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "tp"),
        )

        # Act
        dp_mesh = mesh["dp"]

        # Assert
        self.assertIsNone(mesh.root_mesh)  # Root mesh has no parent
        self.assertEqual(dp_mesh.root_mesh, mesh)  # Submesh has parent

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_sub_mesh_property(self, mock_platform):
        """Test DeviceMesh sub_mesh property.

        Scenario: Access sub_mesh property after creating submeshes.
        Expected behavior: Should contain all created submeshes.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "tp"),
        )

        # Act
        self.assertEqual(len(mesh.sub_mesh), 0)  # Initially empty

        dp_mesh = mesh["dp"]

        # Assert
        self.assertEqual(len(mesh.sub_mesh), 1)
        self.assertEqual(dp_mesh, mesh.sub_mesh[0])

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_flatten_mapping(self, mock_platform):
        """Test DeviceMesh flatten mapping methods.

        Scenario: Create flattened mesh and check mapping.
        Expected behavior: get_flatten_mapping should contain the flattened mesh.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "tp"),
        )

        # Act & Assert - initially empty
        self.assertEqual(mesh.get_flatten_mapping(), {})

        # Add a flattened mesh
        flat_dp_tp = mesh.flatten()
        self.assertIn("dp_tp", mesh.get_flatten_mapping())
        self.assertEqual(mesh.get_flatten_mapping()["dp_tp"], flat_dp_tp)

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_to_hash(self, mock_platform):
        """Test DeviceMesh to_hash method.

        Scenario: Generate hash key from DeviceMesh.
        Expected behavior: to_hash should return tuple of shape, names, and rank list.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2),
            mesh_dim_names=("dp", "tp"),
        )

        # Act
        hash_key = mesh.to_hash()

        # Assert
        expected = ((2, 2), ("dp", "tp"), (0, 1, 2, 3))
        self.assertEqual(hash_key, expected)

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_get_all_groups(self, mock_platform):
        """Test DeviceMesh get_all_groups method.

        Scenario: Get all communication groups from DeviceMesh.
        Expected behavior: Should return list of all groups used in the mesh.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        # Mock group creation to avoid AssertionError
        mock_group = MagicMock()
        mock_platform.split_group.return_value = mock_group

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "tp"),
        )

        # Manually set up the dim_group_names and EXISTING_COMM_GROUPS to avoid assertion errors
        # This is needed because we're mocking the platform
        mesh._dim_group_names = ['(0, 1)', '(0, 1, 2, 3)']
        EXISTING_COMM_GROUPS['(0, 1)'] = mock_group
        EXISTING_COMM_GROUPS['(0, 1, 2, 3)'] = mock_group

        # Act
        all_groups = mesh.get_all_groups()

        # Assert
        self.assertEqual(len(all_groups), 2)

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_assert_axis(self, mock_platform):
        """Test DeviceMesh assert_axis method.

        Scenario: Validate axis name exists in mesh.
        Expected behavior: Should raise ValueError for invalid axis names.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "tp"),
        )

        # Act & Assert - valid axis should not raise
        mesh.assert_axis("dp", "test_op")
        mesh.assert_axis("tp", "test_op")

        # Invalid axis should raise
        with self.assertRaises(ValueError) as context:
            mesh.assert_axis("invalid", "test_op")
        self.assertIn("invalid", str(context.exception))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_without_dim_names(self, mock_platform):
        """Test DeviceMesh without mesh_dim_names.

        Scenario: Create DeviceMesh without mesh_dim_names.
        Expected behavior: DeviceMesh should work with default behavior.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = DeviceMesh("npu", mesh=[0, 1, 2, 3], _init_backend=False)

        # Act & Assert
        self.assertIsNone(mesh.mesh_dim_names)
        self.assertEqual(mesh.mesh_shape, (4,))
        self.assertEqual(mesh.rank_list, (0, 1, 2, 3))

        # axis_id without dim names should work
        self.assertEqual(mesh.axis_id("None"), -1)

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_from_group_with_1d_group(self, mock_platform):
        """Test DeviceMesh.from_group with 1D group.

        Scenario: Create DeviceMesh from a 1D communication group.
        Expected behavior: DeviceMesh should be created with ranks from the group.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_group = MagicMock()
        mock_platform.get_process_group_ranks.return_value = [0, 1, 2, 3]
        mock_platform.get_created_group.return_value = False
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        # Act - create DeviceMesh from 1D group
        device_mesh = DeviceMesh.from_group(
            group=mock_group,
            device_type="npu",
            mesh_dim_names=("dp",),
        )

        # Assert
        self.assertEqual(device_mesh.mesh_shape, (4,))
        self.assertEqual(device_mesh.rank_list, (0, 1, 2, 3))
        self.assertEqual(device_mesh.mesh_dim_names, ("dp",))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_from_group_with_nd_groups(self, mock_platform):
        """Test DeviceMesh.from_group with nD groups.

        Scenario: Create DeviceMesh from a list of communication groups for 2D mesh.
        Expected behavior: DeviceMesh should be created with appropriate rank layout.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_group_dp = MagicMock()
        mock_group_tp = MagicMock()
        mock_platform.get_process_group_ranks.side_effect = [
            [0, 1],  # dp group
            [0, 2],  # tp group
        ]
        mock_platform.get_created_group.return_value = False
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        # Act - create DeviceMesh from list of groups
        device_mesh = DeviceMesh.from_group(
            group=[mock_group_dp, mock_group_tp],
            device_type="npu",
            mesh=[[0, 1], [2, 3]],
            mesh_dim_names=("dp", "tp"),
        )

        # Assert
        self.assertEqual(device_mesh.mesh_shape, (2, 2))
        self.assertEqual(device_mesh.rank_list, (0, 1, 2, 3))
        self.assertEqual(device_mesh.mesh_dim_names, ("dp", "tp"))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_from_group_invalid_mesh(self, mock_platform):
        """Test DeviceMesh.from_group with invalid mesh raises ValueError.

        Scenario: Create DeviceMesh with 1D group but mismatched mesh.
        Expected behavior: Should raise ValueError when mesh doesn't match group ranks.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_group = MagicMock()
        mock_platform.get_process_group_ranks.return_value = [0, 1]
        mock_platform.get_created_group.return_value = False
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        # Act & Assert - mismatched mesh should raise ValueError
        with self.assertRaises(ValueError) as context:
            DeviceMesh.from_group(
                group=mock_group,
                device_type="npu",
                mesh=[0, 1, 2, 3],  # 4 ranks, but group only has 2
                mesh_dim_names=("dp",),
            )
        self.assertIn("Invalid mesh", str(context.exception))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_get_devices_for_axis_with_str(self, mock_platform):
        """Test DeviceMesh.get_devices_for_axis with string dimension name.

        Scenario: Get devices along a specific axis by dimension name.
        Expected behavior: Should return list of ranks along that dimension.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = DeviceMesh(
            "npu",
            mesh=[[0, 1], [2, 3]],
            mesh_dim_names=("dp", "tp"),
            _init_backend=False,
        )

        # Act - get devices for axis "dp" with rank 0
        devices = mesh.get_devices_for_axis("dp", 0)

        # Assert - rank 0 and 1 are in the same dp group
        self.assertEqual(sorted(devices), [0, 2])

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_get_devices_for_axis_with_int(self, mock_platform):
        """Test DeviceMesh.get_devices_for_axis with integer dimension index.

        Scenario: Get devices along a specific axis by dimension index.
        Expected behavior: Should return list of ranks along that dimension.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = DeviceMesh(
            "npu",
            mesh=[[0, 1], [2, 3]],
            mesh_dim_names=("dp", "tp"),
            _init_backend=False,
        )

        # Act - get devices for axis 0 (dp) with rank 0
        devices = mesh.get_devices_for_axis(0, 0)

        # Assert - rank 0 and 2 are in the same dp group
        self.assertEqual(sorted(devices), [0, 2])

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_get_devices_for_axis_invalid_dim(self, mock_platform):
        """Test DeviceMesh.get_devices_for_axis with invalid dimension raises ValueError.

        Scenario: Get devices with invalid dimension name.
        Expected behavior: Should raise ValueError with appropriate message.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = DeviceMesh(
            "npu",
            mesh=[[0, 1], [2, 3]],
            mesh_dim_names=("dp", "tp"),
            _init_backend=False,
        )

        # Act & Assert - invalid dimension name should raise
        with self.assertRaises(ValueError) as context:
            mesh.get_devices_for_axis("invalid", 0)
        self.assertIn("not found", str(context.exception))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_get_devices_for_axis_no_dim_names(self, mock_platform):
        """Test DeviceMesh.get_devices_for_axis without mesh_dim_names raises ValueError.

        Scenario: Get devices when mesh_dim_names is not set.
        Expected behavior: Should raise ValueError when using string dimension.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = DeviceMesh(
            "npu",
            mesh=[[0, 1], [2, 3]],
            _init_backend=False,
        )

        # Act & Assert - using string dim without dim names should raise
        with self.assertRaises(ValueError) as context:
            mesh.get_devices_for_axis("dp", 0)
        self.assertIn("not set", str(context.exception))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_get_devices_for_axis_invalid_rank(self, mock_platform):
        """Test DeviceMesh.get_devices_for_axis with invalid rank raises ValueError.

        Scenario: Get devices with rank not in mesh.
        Expected behavior: Should raise ValueError with appropriate message.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = DeviceMesh(
            "npu",
            mesh=[[0, 1], [2, 3]],
            mesh_dim_names=("dp", "tp"),
            _init_backend=False,
        )

        # Act & Assert - invalid rank should raise
        with self.assertRaises(ValueError) as context:
            mesh.get_devices_for_axis("dp", 100)
        self.assertIn("not found", str(context.exception))

    @patch("hyper_parallel.core.device_mesh.platform")
    def test_device_mesh_get_devices_for_axis_out_of_range(self, mock_platform):
        """Test DeviceMesh.get_devices_for_axis with out-of-range dimension index raises ValueError.

        Scenario: Get devices with dimension index out of valid range.
        Expected behavior: Should raise ValueError with appropriate message.

        Args:
            mock_platform: Mocked platform module for test isolation.
        """
        # Arrange
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, 'numpy') else np.array(t)

        mesh = DeviceMesh(
            "npu",
            mesh=[[0, 1], [2, 3]],
            mesh_dim_names=("dp", "tp"),
            _init_backend=False,
        )

        # Act & Assert - out of range dimension should raise
        with self.assertRaises(ValueError) as context:
            mesh.get_devices_for_axis(5, 0)
        self.assertIn("out of range", str(context.exception))


if __name__ == "__main__":
    unittest.main()