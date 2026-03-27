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
"""Unit tests for DTensor initialization functions.

This module provides comprehensive unit tests for DTensor initialization
functions (ones, zeros, empty, full) in the hyper_parallel framework.
All tests use mocking to avoid dependencies on actual hardware or
distributed communication.
"""
import unittest
from unittest.mock import patch, MagicMock, call

import os
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import (
    DTensor,
    _build_layout,
    _dtensor_init_helper,
    ones,
    zeros,
    empty,
    full,
    _LAYOUT_CACHE,
)
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.utils.shape_utils import compute_local_shape_and_global_offset


class TestBuildLayout(unittest.TestCase):
    """Unit tests for _build_layout function.

    This test class verifies that the _build_layout function correctly
    creates Layout objects from device meshes and placements, and properly
    caches the results to avoid redundant computations.
    """

    def setUp(self):
        """Clear layout cache before each test.

        Ensures each test starts with a clean state, preventing cache
        pollution from previous test runs.
        """
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        """Clear layout cache after each test.

        Cleans up any entries added during the test to maintain isolation
        between tests.
        """
        _LAYOUT_CACHE.clear()

    def test_build_layout_creates_layout_correctly(self):
        """Test that _build_layout creates layout with correct parameters.

        Test Scenario:
            Call _build_layout with a valid device mesh, shard/replicate
            placements, and tensor dimensions.

        Expected Behavior:
            - Layout.from_device_mesh() is called once with the mesh
            - layout(placements) is called to set placements
            - placement_to_tensor_map(tensor_dim) is called to map placements
            - The returned layout instance is the expected mock instance
        """
        # Arrange
        mock_mesh = MagicMock()
        mock_mesh.to_hash.return_value = "mesh_hash_123"
        mock_mesh.mesh_shape = (1, 1)  # len must match placement objects count
        placements = [Shard(0), Replicate()]
        tensor_dim = 3

        with patch('hyper_parallel.core.dtensor.dtensor.Layout') as mock_layout_class:
            mock_layout_instance = MagicMock()
            # Layout.from_device_mesh() returns the layout instance
            mock_layout_class.from_device_mesh.return_value = mock_layout_instance
            # layout(placements) returns the same instance after setting placements
            mock_layout_instance.return_value = mock_layout_instance

            # Act
            result = _build_layout(mock_mesh, placements, tensor_dim)

            # Assert
            mock_layout_class.from_device_mesh.assert_called_once_with(mock_mesh)
            # layout is called with placements as a list (not wrapped)
            mock_layout_instance.assert_called_once_with(placements)
            mock_layout_instance.placement_to_tensor_map.assert_called_once_with(tensor_dim)
            self.assertEqual(result, mock_layout_instance)

    def test_build_layout_uses_cache(self):
        """Test that _build_layout caches and reuses layouts.

        Test Scenario:
            Call _build_layout twice with identical parameters
            (same device mesh, placements, and tensor_dim).

        Expected Behavior:
            - Layout.from_device_mesh() is called only once (on first call)
            - Second call returns cached layout without creating new Layout
            - Both calls return the same layout instance (cached)
            - _LAYOUT_CACHE contains exactly 1 entry
        """
        # Arrange
        mock_mesh = MagicMock()
        mock_mesh.to_hash.return_value = "mesh_hash_456"
        mock_mesh.mesh_shape = (1,)  # len must match placement objects count
        placements = [Shard(0)]
        tensor_dim = 2

        with patch('hyper_parallel.core.dtensor.dtensor.Layout') as mock_layout_class:
            mock_layout_instance = MagicMock()
            mock_layout_class.from_device_mesh.return_value = mock_layout_instance
            mock_layout_instance.return_value = mock_layout_instance

            # Act - Call twice with same parameters
            result1 = _build_layout(mock_mesh, placements, tensor_dim)
            result2 = _build_layout(mock_mesh, placements, tensor_dim)

            # Assert - Layout should only be created once
            mock_layout_class.from_device_mesh.assert_called_once()
            self.assertEqual(result1, result2)
            self.assertEqual(len(_LAYOUT_CACHE), 1)
            # Verify the cached result is the same mock instance
            self.assertIs(result1, result2)

    def test_build_layout_different_params_creates_new_layout(self):
        """Test that different parameters create different cached layouts.

        Test Scenario:
            Call _build_layout with 3 different parameter combinations:
            1. Different mesh (mesh1) with same placement
            2. Same mesh (mesh1) with different placement
            3. Different mesh (mesh2) with same placement as case 1

        Expected Behavior:
            - Each unique combination creates a new Layout instance
            - _LAYOUT_CACHE contains exactly 3 entries (one for each unique call)
            - All three results are distinct layout instances
        """
        # Arrange
        mock_mesh1 = MagicMock()
        mock_mesh1.to_hash.return_value = "mesh_hash_1"
        mock_mesh1.mesh_shape = (1,)
        mock_mesh2 = MagicMock()
        mock_mesh2.to_hash.return_value = "mesh_hash_2"
        mock_mesh2.mesh_shape = (1,)
        placements1 = [Shard(0)]
        placements2 = [Replicate()]
        tensor_dim = 2

        with patch('hyper_parallel.core.dtensor.dtensor.Layout') as mock_layout_class:
            mock_layout_instances = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
            mock_layout_class.from_device_mesh.side_effect = lambda m: {
                mock_mesh1: mock_layout_instances[0],
                mock_mesh2: mock_layout_instances[2],
            }[m]

            # Act - Create layouts with different parameters
            result1 = _build_layout(mock_mesh1, placements1, tensor_dim)
            result2 = _build_layout(mock_mesh1, placements2, tensor_dim)
            result3 = _build_layout(mock_mesh2, placements1, tensor_dim)

            # Assert - Should have 3 different cached entries
            self.assertEqual(len(_LAYOUT_CACHE), 3)


class TestComputeLocalShape(unittest.TestCase):
    """Unit tests for compute_local_shape_and_global_offset function.

    This test class verifies that the compute_local_shape_and_global_offset
    function correctly calculates the local tensor shape based on the global
    shape, device mesh, and placement strategy (Shard or Replicate).
    """

    def test_compute_local_shape_with_shard(self):
        """Test local shape computation with Shard placement.

        Test Scenario:
            Compute local shape for a 3D tensor with global shape [16, 8, 12]
            using Shard(0) placement across 4 devices.

        Expected Behavior:
            - The first dimension is divided by the number of devices (16 / 4 = 4)
            - Other dimensions remain unchanged
            - Resulting local shape is [4, 8, 12]
        """
        # Arrange
        mock_mesh = MagicMock()
        mock_layout_instance = MagicMock()

        with patch('hyper_parallel.core.utils.shape_utils.Layout.from_device_mesh') as mock_from_mesh:
            mock_from_mesh.return_value = mock_layout_instance
            # layout(placement) returns the layout instance
            mock_layout_instance.return_value = mock_layout_instance
            mock_layout_instance.alias_tensor_map = ["dp", "None", "None"]
            mock_layout_instance.mesh.get_device_num_along_axis.return_value = 4

            global_shape = [16, 8, 12]
            placement = [Shard(0)]

            # Act
            local_shape = compute_local_shape_and_global_offset(
                global_shape, mock_mesh, placement
            )

            # Assert
            self.assertEqual(local_shape, [4, 8, 12])

    def test_compute_local_shape_with_replicate(self):
        """Test local shape computation with Replicate placement.

        Test Scenario:
            Compute local shape for a 3D tensor with global shape [16, 8, 12]
            using Replicate() placement across all devices.

        Expected Behavior:
            - Replicate placement means each device gets the full tensor
            - Local shape is identical to global shape [16, 8, 12]
            - No dimension is partitioned
        """
        # Arrange
        mock_mesh = MagicMock()
        mock_layout_instance = MagicMock()

        with patch('hyper_parallel.core.utils.shape_utils.Layout.from_device_mesh') as mock_from_mesh:
            mock_from_mesh.return_value = mock_layout_instance
            # layout(placement) returns the layout instance
            mock_layout_instance.return_value = mock_layout_instance
            mock_layout_instance.alias_tensor_map = ["None", "None", "None"]

            global_shape = [16, 8, 12]
            placement = [Replicate()]

            # Act
            local_shape = compute_local_shape_and_global_offset(
                global_shape, mock_mesh, placement
            )

            # Assert - Replicate should keep original shape
            self.assertEqual(local_shape, [16, 8, 12])


class TestDtensorInitHelper(unittest.TestCase):
    """Unit tests for _dtensor_init_helper function.

    This test class verifies that the _dtensor_init_helper function correctly
    initializes distributed tensors by computing local shapes, calling the
    appropriate platform initialization functions (full, ones, zeros, empty),
    and wrapping the results in DTensor objects.
    """

    def setUp(self):
        """Clear layout cache before each test.

        Ensures each test starts with a clean _LAYOUT_CACHE state to prevent
        interference between test cases.
        """
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        """Clear layout cache after each test.

        Cleans up any cache entries created during the test to maintain
        isolation between test cases.
        """
        _LAYOUT_CACHE.clear()

    @patch('hyper_parallel.core.dtensor.dtensor.DTensor')
    @patch('hyper_parallel.core.dtensor.dtensor.compute_local_shape_and_global_offset')
    def test_init_helper_with_full_op(self, mock_compute_local, mock_dtensor_class):
        """Test _dtensor_init_helper with full operation (requires fill_value).

        Test Scenario:
            Initialize a DTensor using platform.full with:
            - Global shape [4, 8, 12]
            - Shard(0) placement across devices
            - fill_value of 3.14
            - dtype of 'float32'

        Expected Behavior:
            - compute_local_shape_and_global_offset is called with size, mesh, placements
            - Local shape is computed as [2, 4, 6] (mocked return)
            - platform.full is called with local_shape, fill_value, and dtype
            - DTensor.from_local is called to wrap the local tensor
        """
        # Arrange
        mock_init_op = MagicMock()
        mock_local_tensor = MagicMock()
        mock_init_op.return_value = mock_local_tensor

        mock_compute_local.return_value = [2, 4, 6]

        mock_mesh = MagicMock()
        placements = [Shard(0)]
        size = [4, 8, 12]
        fill_value = 3.14

        # Act
        result = _dtensor_init_helper(
            mock_init_op,
            size,
            mock_mesh,
            placements,
            fill_value=fill_value,
            dtype='float32'
        )

        # Assert
        mock_compute_local.assert_called_once_with(size, mock_mesh, placements)
        # Check the init_op was called with correct arguments
        mock_init_op.assert_called_once()
        call_args = mock_init_op.call_args
        # First positional arg should be local_shape
        self.assertEqual(call_args[0][0], [2, 4, 6])
        # fill_value should be the second positional arg or in kwargs
        self.assertEqual(call_args[1].get('fill_value') or call_args[0][1], fill_value)
        mock_dtensor_class.from_local.assert_called_once_with(
            mock_local_tensor, mock_mesh, placements
        )

    @patch('hyper_parallel.core.dtensor.dtensor.DTensor')
    @patch('hyper_parallel.core.dtensor.dtensor.compute_local_shape_and_global_offset')
    def test_init_helper_with_ones_op(self, mock_compute_local, mock_dtensor_class):
        """Test _dtensor_init_helper with ones operation (no fill_value).

        Test Scenario:
            Initialize a DTensor using platform.ones with:
            - Global shape [2, 2]
            - Replicate() placement (no sharding)
            - dtype of 'float32'

        Expected Behavior:
            - compute_local_shape_and_global_offset is called with size, mesh, placements
            - Local shape is computed as [2, 2] (same as global for Replicate)
            - platform.ones is called with local_shape [2, 2] and dtype='float32'
            - No fill_value is passed (ones doesn't need it)
        """
        # Arrange
        mock_init_op = MagicMock()
        mock_local_tensor = MagicMock()
        mock_init_op.return_value = mock_local_tensor

        mock_compute_local.return_value = [2, 2]

        mock_mesh = MagicMock()
        placements = [Replicate()]
        size = [2, 2]

        # Act
        result = _dtensor_init_helper(
            mock_init_op,
            size,
            mock_mesh,
            placements,
            dtype='float32'
        )

        # Assert
        mock_compute_local.assert_called_once_with(size, mock_mesh, placements)
        mock_init_op.assert_called_once_with([2, 2], dtype='float32')


class TestOnes(unittest.TestCase):
    """Unit tests for ones function.

    This test class verifies that the ones() function correctly delegates
    to _dtensor_init_helper with the appropriate platform.ones initialization
    function and parameters.
    """

    @patch('hyper_parallel.core.dtensor.dtensor._dtensor_init_helper')
    @patch('hyper_parallel.core.dtensor.dtensor.platform')
    def test_ones_calls_helper_correctly(self, mock_platform, mock_helper):
        """Test that ones function calls _dtensor_init_helper with correct parameters.

        Test Scenario:
            Call ones() with a 3D shape [4, 8, 12], a device mesh, and
            Shard(0) placement for data parallel sharding.

        Expected Behavior:
            - _dtensor_init_helper is called exactly once
            - The helper receives platform.ones as the init_op
            - size parameter is [4, 8, 12]
            - device_mesh and placements are passed as keyword arguments
        """
        # Arrange
        mock_platform.ones = MagicMock()
        mock_helper.return_value = MagicMock()

        mock_mesh = MagicMock()
        placements = [Shard(0)]
        size = [4, 8, 12]

        # Act
        result = ones(size, mock_mesh, placements)

        # Assert - Check that helper was called with the correct init_op and kwargs
        mock_helper.assert_called_once_with(
            mock_platform.ones,  # init_op
            size,  # size
            device_mesh=mock_mesh,  # device_mesh as keyword arg
            placements=placements,  # placements as keyword arg
        )


class TestZeros(unittest.TestCase):
    """Unit tests for zeros function.

    This test class verifies that the zeros() function correctly delegates
    to _dtensor_init_helper with the appropriate platform.zeros initialization
    function and parameters.
    """

    @patch('hyper_parallel.core.dtensor.dtensor._dtensor_init_helper')
    @patch('hyper_parallel.core.dtensor.dtensor.platform')
    def test_zeros_calls_helper_correctly(self, mock_platform, mock_helper):
        """Test that zeros function calls _dtensor_init_helper with correct parameters.

        Test Scenario:
            Call zeros() with a 3D shape [4, 8, 12], a device mesh, and
            Shard(0) placement for data parallel sharding.

        Expected Behavior:
            - _dtensor_init_helper is called exactly once
            - The helper receives platform.zeros as the init_op
            - size parameter is [4, 8, 12]
            - device_mesh and placements are passed as keyword arguments
        """
        # Arrange
        mock_platform.zeros = MagicMock()
        mock_helper.return_value = MagicMock()

        mock_mesh = MagicMock()
        placements = [Shard(0)]
        size = [4, 8, 12]

        # Act
        result = zeros(size, mock_mesh, placements)

        # Assert
        mock_helper.assert_called_once_with(
            mock_platform.zeros,
            size,
            device_mesh=mock_mesh,
            placements=placements,
        )


class TestEmpty(unittest.TestCase):
    """Unit tests for empty function.

    This test class verifies that the empty() function correctly delegates
    to _dtensor_init_helper with the appropriate platform.empty initialization
    function and parameters. The empty() function creates a DTensor with
    uninitialized memory.
    """

    @patch('hyper_parallel.core.dtensor.dtensor._dtensor_init_helper')
    @patch('hyper_parallel.core.dtensor.dtensor.platform')
    def test_empty_calls_helper_correctly(self, mock_platform, mock_helper):
        """Test that empty function calls _dtensor_init_helper with correct parameters.

        Test Scenario:
            Call empty() with a 3D shape [4, 8, 12], a device mesh, and
            Shard(0) placement for data parallel sharding.

        Expected Behavior:
            - _dtensor_init_helper is called exactly once
            - The helper receives platform.empty as the init_op
            - size parameter is [4, 8, 12]
            - device_mesh and placements are passed as keyword arguments
        """
        # Arrange
        mock_platform.empty = MagicMock()
        mock_helper.return_value = MagicMock()

        mock_mesh = MagicMock()
        placements = [Shard(0)]
        size = [4, 8, 12]

        # Act
        result = empty(size, mock_mesh, placements)

        # Assert
        mock_helper.assert_called_once_with(
            mock_platform.empty,
            size,
            device_mesh=mock_mesh,
            placements=placements,
        )


class TestFull(unittest.TestCase):
    """Unit tests for full function.

    This test class verifies that the full() function correctly delegates
    to _dtensor_init_helper with the appropriate platform.full initialization
    function and parameters. The full() function creates a DTensor filled
    with a specified value.
    """

    @patch('hyper_parallel.core.dtensor.dtensor._dtensor_init_helper')
    @patch('hyper_parallel.core.dtensor.dtensor.platform')
    def test_full_calls_helper_correctly(self, mock_platform, mock_helper):
        """Test that full function calls _dtensor_init_helper with correct parameters.

        Test Scenario:
            Call full() with a 3D shape [4, 8, 12], fill_value of 123.4,
            a device mesh, and Shard(0) placement for data parallel sharding.

        Expected Behavior:
            - _dtensor_init_helper is called exactly once
            - The helper receives platform.full as the init_op
            - size parameter is [4, 8, 12]
            - fill_value is passed as 123.4
            - device_mesh and placements are passed as keyword arguments
        """
        # Arrange
        mock_platform.full = MagicMock()
        mock_helper.return_value = MagicMock()

        mock_mesh = MagicMock()
        placements = [Shard(0)]
        size = [4, 8, 12]
        fill_value = 123.4

        # Act
        result = full(size, fill_value, device_mesh=mock_mesh, placements=placements)

        # Assert
        mock_helper.assert_called_once_with(
            mock_platform.full,
            size,
            device_mesh=mock_mesh,
            placements=placements,
            fill_value=fill_value,
        )


if __name__ == "__main__":
    unittest.main()