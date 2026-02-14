# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test the DeviceMesh module."""
from unittest.mock import Mock, patch

import numpy as np
import pytest

from hyper_parallel.core.device_mesh import (
    init_device_mesh, DeviceMesh, _group_map, Tensor, _DEVICE_MESH_MAP
)


@pytest.fixture(name="mock_platform")
def fixture_mock_platform():
    """Mock platform-related interfaces (avoid dependency on real hardware/distributed environment)"""
    with patch("hyper_parallel.core.device_mesh.platform") as platform_mock:
        # Mock rank=0, world_size=8
        platform_mock.get_rank.return_value = 0
        platform_mock.get_world_size.return_value = 8
        # Mock communication group creation (return Mock object)
        mock_group = Mock()
        platform_mock.split_group.return_value = mock_group

        # Mock tensor_to_numpy to return actual numpy array from Tensor.asnumpy()
        def mock_tensor_to_numpy(tensor):
            """Convert Tensor to numpy array using asnumpy() method"""
            if hasattr(tensor, 'asnumpy'):
                return tensor.asnumpy()
            return tensor
        platform_mock.tensor_to_numpy.side_effect = mock_tensor_to_numpy
        yield platform_mock


@pytest.fixture(autouse=True)
def fixture_clear_group_map():
    """Auto clear global group cache to avoid test case pollution, effective for all test cases"""
    _group_map.clear()
    _DEVICE_MESH_MAP.clear()
    yield
    _group_map.clear()
    _DEVICE_MESH_MAP.clear()


@pytest.fixture(name="basic_2d_mesh")
def fixture_basic_2d_mesh(mock_platform):
    """Create basic 2D DeviceMesh: mesh_shape=(2,4), mesh_dim_names=("dp", "tp"), rank_list=(0-7)"""
    _ = mock_platform  # Ensure mock is active
    return init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp")
    )


@pytest.fixture(name="basic_3d_mesh")
def fixture_basic_3d_mesh(mock_platform):
    """Create basic 3D DeviceMesh: mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "cp", "tp"), rank_list=(0-7)"""
    _ = mock_platform  # Ensure mock is active
    return init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "cp", "tp")
    )


class TestDeviceMesh:
    """Test suite for DeviceMesh class and related functions"""

    def test_init_device_mesh_basic(self, mock_platform):
        """
        Feature: init_device_mesh function.
        Description: Test basic functionality including automatic rank_list generation
            and cache mechanism.
        Expectation: Run success, mesh properties match expected values,
            same parameters return the same cached instance.
        """
        _ = mock_platform  # Ensure mock is active
        # Automatically generate rank_list from mesh_shape
        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2),
            mesh_dim_names=("dp", "tp")
        )
        assert mesh.mesh_shape == (2, 2)
        assert mesh.mesh_dim_names == ("dp", "tp")
        assert mesh.rank_list == (0, 1, 2, 3)
        assert mesh.ndim == 2
        assert mesh.rank == 0

        # Cache mechanism (same parameters return the same instance)
        mesh1 = init_device_mesh("npu", (2, 2), mesh_dim_names=("dp", "tp"))
        mesh2 = init_device_mesh("npu", (2, 2), mesh_dim_names=("dp", "tp"))
        assert mesh1 is mesh2

    def test_device_mesh_direct_construction_with_tensor(self, mock_platform):
        """
        Feature: DeviceMesh direct construction with mesh tensor.
        Description: Test DeviceMesh construction with custom mesh tensor,
            verifying mesh_shape and rank_list are correctly extracted.
        Expectation: Run success, properties match expected values.
        """
        _ = mock_platform  # Ensure mock is active
        # Custom mesh layout using Tensor
        mesh = Tensor([[0, 2], [1, 3]])
        device_mesh = DeviceMesh("npu", mesh, mesh_dim_names=("dp", "tp"))

        assert device_mesh.mesh_shape == (2, 2)
        assert device_mesh.mesh_dim_names == ("dp", "tp")
        assert device_mesh.rank_list == (0, 2, 1, 3)  # Flattened from custom layout
        assert device_mesh.ndim == 2

    def test_device_mesh_direct_construction_with_list(self, mock_platform):
        """
        Feature: DeviceMesh direct construction with list.
        Description: Test DeviceMesh construction with list input,
            verifying automatic conversion to Tensor.
        Expectation: Run success, properties match expected values.
        """
        _ = mock_platform  # Ensure mock is active
        # Custom mesh layout using list
        device_mesh = DeviceMesh("npu", [[0, 2], [1, 3]], mesh_dim_names=("dp", "tp"))

        assert device_mesh.mesh_shape == (2, 2)
        assert device_mesh.mesh_dim_names == ("dp", "tp")
        assert device_mesh.rank_list == (0, 2, 1, 3)
        assert device_mesh.ndim == 2

    def test_device_mesh_direct_construction_with_numpy(self, mock_platform):
        """
        Feature: DeviceMesh direct construction with numpy array.
        Description: Test DeviceMesh construction with numpy array input,
            verifying automatic conversion to Tensor.
        Expectation: Run success, properties match expected values.
        """
        _ = mock_platform  # Ensure mock is active
        # Custom mesh layout using numpy array
        mesh = np.array([[0, 2], [1, 3]], dtype=np.int64)  # Use int64 to test conversion
        device_mesh = DeviceMesh("npu", mesh, mesh_dim_names=("dp", "tp"))

        assert device_mesh.mesh_shape == (2, 2)
        assert device_mesh.mesh_dim_names == ("dp", "tp")
        assert device_mesh.rank_list == (0, 2, 1, 3)
        assert device_mesh.ndim == 2

    def test_device_mesh_getitem_valid(self, basic_2d_mesh, basic_3d_mesh):
        """
        Feature: DeviceMesh.__getitem__ method.
        Description: Test valid sub-mesh acquisition via __getitem__ method,
            including single dimension specification via string and
            multiple dimensions specification via tuple.
        Expectation: Run success, sub-mesh properties match expected values,
            root_mesh reference is correctly set.
        """
        # Single dimension specified by string
        dp_mesh = basic_2d_mesh["dp"]
        assert dp_mesh.mesh_shape == (2,)
        assert dp_mesh.mesh_dim_names == ("dp",)
        assert dp_mesh.root_mesh == basic_2d_mesh
        assert dp_mesh.rank_list == (0, 4)
        assert dp_mesh in basic_2d_mesh.sub_mesh

        tp_mesh = basic_2d_mesh["tp"]
        assert tp_mesh.mesh_shape == (4,)
        assert tp_mesh.mesh_dim_names == ("tp",)
        assert tp_mesh.root_mesh == basic_2d_mesh
        assert tp_mesh.rank_list == (0, 1, 2, 3)

        # Multiple dimensions specified by tuple
        dp_cp_mesh = basic_3d_mesh[("dp", "cp")]
        assert dp_cp_mesh.mesh_shape == (2, 2)
        assert dp_cp_mesh.mesh_dim_names == ("dp", "cp")
        assert dp_cp_mesh.rank_list == (0, 2, 4, 6)

    def test_device_mesh_get_group_valid_via_name_and_index(self, basic_2d_mesh):
        """
        Feature: DeviceMesh.get_group method.
        Description: Test valid communication group acquisition via get_group method,
            including group acquisition by dimension name and by dimension index.
        Expectation: Run success, the groups arm same acquired by dimension name and by dimension index.
        """
        # Get group by dimension name
        dp_group_1 = basic_2d_mesh.get_group("dp")
        # Get group by dimension index
        dp_group_2 = basic_2d_mesh.get_group(0)
        assert dp_group_1 == dp_group_2

    def test_device_mesh_get_local_rank(self, basic_2d_mesh):
        """
        Feature: DeviceMesh.get_local_rank method.
        Description: Test valid local rank calculation via get_local_rank method,
            including local rank calculation for current rank=0 and mocked rank=5.
        Expectation: Run success, local rank values match expected positions
            in respective dimensions.
        """
        # Current rank=0 (mocked platform returns rank=0)
        assert basic_2d_mesh.get_local_rank("dp") == 0
        assert basic_2d_mesh.get_local_rank("tp") == 0

        # Mock current rank=5
        with patch.object(basic_2d_mesh, "_rank", 5):
            assert basic_2d_mesh.get_local_rank("dp") == 1
            assert basic_2d_mesh.get_local_rank(1) == 1

    def test_device_mesh_flatten(self, basic_3d_mesh, mock_platform):
        """
        Feature: DeviceMesh.flatten method.
        Description: Test valid mesh flattening via flatten method,
            verifying flattened mesh properties and group creation.
        Expectation: Run success, flattened mesh has correct shape, mesh_dim_names,
            rank_list, and root_mesh reference.
        """
        dp_cp_mesh = basic_3d_mesh[("dp", "cp")]
        flat_mesh = dp_cp_mesh.flatten()
        flat_mesh_group_1 = flat_mesh.get_group()
        flat_mesh_group_2 = basic_3d_mesh.get_group("dp_cp")
        mock_platform.split_group.assert_called_with(split_ranks=[[0, 2, 4, 6]], group_desc="mesh_dp_cp")

        assert flat_mesh.mesh_shape == (4,)
        assert flat_mesh.mesh_dim_names == ("dp_cp",)
        assert flat_mesh.rank_list == (0, 2, 4, 6)
        assert flat_mesh.root_mesh == basic_3d_mesh
        assert flat_mesh_group_1 is mock_platform.split_group.return_value
        assert flat_mesh_group_2 is mock_platform.split_group.return_value

    def test_device_mesh_mesh_property(self, mock_platform):
        """
        Feature: DeviceMesh.mesh property.
        Description: Test mesh property returns the mesh tensor.
        Expectation: Run success, mesh property returns correct tensor.
        """
        _ = mock_platform  # Ensure mock is active
        mesh = Tensor([[0, 1], [2, 3]])
        device_mesh = DeviceMesh("npu", mesh, mesh_dim_names=("dp", "tp"))

        np.testing.assert_array_equal(device_mesh.mesh, mesh)
        assert device_mesh.mesh.shape == (2, 2)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
