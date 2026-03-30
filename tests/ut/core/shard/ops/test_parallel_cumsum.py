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
"""parallel_cumsum test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_cumsum import CumsumDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = CumsumDistributedOp("cumsum")


class TestParallelCumsum(unittest.TestCase):
    """Unit tests for CumsumDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _setup_mock_platform(self, mock_platform, platform_type=None, world_size=8):
        """Configure common mock-platform attributes used across tests.

        Args:
            mock_platform: The MagicMock object injected by @patch.
            platform_type: Optional PlatformType to set on the mock.
            world_size: Value returned by mock_platform.get_world_size().
        """
        if platform_type is not None:
            mock_platform.platform_type = platform_type
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"))

    def _make_2x3x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x3x4 (dp, tp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=24)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 3, 4), mesh_dim_names=("dp", "tp", "mp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_layout_data_parallel(self, mock_platform):
        """
        Feature: Cumsum data parallel
        Description: Data parallel on non-cumsum dimension (dim=-1 unsharded)
        Expectation: Output layout identical to input layout
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), extra_args=(-1,))

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Data parallel cumsum failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        
        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, output_layout, (x_layout,), (-1,)) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, output_layout, (x_layout,), (-1,))}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_layout_tensor_parallel(self, mock_platform):
        """
        Feature: Cumsum tensor parallel
        Description: Tensor parallel on non-cumsum dimension (dim=0 unsharded)
        Expectation: Output layout identical to input layout
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), extra_args=(0,))

        expected_map = (-1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Tensor parallel cumsum failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_layout_mixed_parallel(self, mock_platform):
        """
        Feature: Cumsum mixed parallel
        Description: Mixed parallel with cumsum on unsharded middle dimension
        Expectation: Output layout identical to input layout
        """
        mesh = self._make_2x3x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        output_layout = op.infer_layout((x_layout,), extra_args=(1,))

        expected_map = (2, -1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Mixed parallel cumsum failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_layout_negative_dim(self, mock_platform):
        """
        Feature: Cumsum with negative dimension
        Description: Test negative dimension indexing (dim=-2) on 3D tensor
        Expectation: Correctly normalized and validated
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), extra_args=(-2,))

        expected_map = (-1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Negative dimension cumsum failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_layout_invalid_sharding_on_cumsum_dim(self, mock_platform):
        """
        Feature: Cumsum on sharded dimension
        Description: Attempt cumsum on a sharded dimension should fail
        Expectation: ValueError raised with clear message
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "Cannot perform sharding on normalized dimension 1"):
            op.infer_layout((x_layout,), extra_args=(-1,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_layout_dim_out_of_range_positive(self, mock_platform):
        """
        Feature: Cumsum with invalid positive dimension
        Description: Dimension index exceeds tensor rank
        Expectation: ValueError raised
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "Dimension 2 out of range for 2-dimensional input tensor"):
            op.infer_layout((x_layout,), extra_args=(2,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_layout_missing_dim_parameter(self, mock_platform):
        """
        Feature: Cumsum without dim parameter
        Description: extra_args missing required 'dim' parameter
        Expectation: ValueError raised
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "cumsum requires 'dim' parameter in extra_args"):
            op.infer_layout((x_layout,), extra_args=None)

        with self.assertRaisesRegex(ValueError, "cumsum requires 'dim' parameter in extra_args"):
            op.infer_layout((x_layout,), extra_args=(None,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_layout_invalid_dim_type(self, mock_platform):
        """
        Feature: Cumsum with non-integer dim
        Description: dim parameter must be integer
        Expectation: ValueError raised
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "'dim' must be an integer, got <class 'str'>"):
            op.infer_layout((x_layout,), extra_args=("invalid",))


if __name__ == "__main__":
    unittest.main()
