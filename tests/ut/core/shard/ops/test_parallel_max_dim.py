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
"""parallel_max_dim test"""
import os
import unittest
from unittest.mock import patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_reduce import MaxDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = MaxDistributedOp("MaxDim")


class TestParallelMaxDim(unittest.TestCase):
    """Unit tests for MaxDimDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

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
        """Set up mock and return a standard 2×4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"))

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2×2×2 (dp, cp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "cp", "mp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_dim_data_parallel_success(self, mock_platform):
        """
        Feature: MaxDim data parallel
        Description: Data parallel scenario with max on unsharded axis
        Expectation: Success, output layout correctly reduced
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        values_layout, index_layout = op.infer_layout((x_layout, None, None), (1, True))

        expected_map = (1, -1, -1)
        assert values_layout.tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert index_layout.tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map}, "
            f"got {index_layout.tensor_map}"
        )

        assert op.get_expand_impl(None, (values_layout, index_layout), (x_layout, None, None), (1, True)) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, (values_layout, index_layout), (x_layout, None, None), (1, True))}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_dim_model_parallel_success(self, mock_platform):
        """
        Feature: MaxDim model parallel
        Description: Model parallel scenario with max on unsharded batch dimension
        Expectation: Success, output layout correctly reduced
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        values_layout, index_layout = op.infer_layout((x_layout, None, None), (0, True))

        expected_map = (-1, 0, -1)
        assert values_layout.tensor_map == expected_map, (
            f"Model Parallel test failed. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert index_layout.tensor_map == expected_map, (
            f"Model Parallel test failed. Expected {expected_map}, "
            f"got {index_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_dim_hybrid_parallel_success(self, mock_platform):
        """
        Feature: MaxDim hybrid parallel
        Description: Hybrid parallel scenario with max on unsharded middle dimension
        Expectation: Success, output layout correctly reduced
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        values_layout, index_layout = op.infer_layout((x_layout, None, None), (1, True))

        expected_map = (2, -1, 0)
        assert values_layout.tensor_map == expected_map, (
            f"Hybrid Parallel test failed. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert index_layout.tensor_map == expected_map, (
            f"Hybrid Parallel test failed. Expected {expected_map}, "
            f"got {index_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_dim_all_replicated(self, mock_platform):
        """
        Feature: MaxDim all replicated
        Description: All dimensions replicated scenario
        Expectation: Success, output layout correctly reduced
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        values_layout, index_layout = op.infer_layout((x_layout, None, None), (0, True))

        expected_map = (-1, -1, -1)
        assert values_layout.tensor_map == expected_map, (
            f"All Replicated test failed. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert index_layout.tensor_map == expected_map, (
            f"All Replicated test failed. Expected {expected_map}, "
            f"got {index_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_dim_negative_dim(self, mock_platform):
        """
        Feature: MaxDim negative dimension index
        Description: Test negative dimension index (dim=-1)
        Expectation: Success, output layout correctly reduced
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        values_layout, index_layout = op.infer_layout((x_layout, None, None), (-1, True))

        expected_map = (1, -1, -1)
        assert values_layout.tensor_map == expected_map, (
            f"Negative dim test failed. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert index_layout.tensor_map == expected_map, (
            f"Negative dim test failed. Expected {expected_map}, "
            f"got {index_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_dim_keepdim_false(self, mock_platform):
        """
        Feature: MaxDim with keepdim=False
        Description: Test with keepdim=False, reduced dimension removed
        Expectation: Success, output layout has reduced dimension removed
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        values_layout, index_layout = op.infer_layout((x_layout, None, None), (1, False))

        expected_map = (1, -1)
        assert values_layout.tensor_map == expected_map, (
            f"Keepdim False test failed. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert index_layout.tensor_map == expected_map, (
            f"Keepdim False test failed. Expected {expected_map}, "
            f"got {index_layout.tensor_map}"
        )


    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_dim_3d_tensor(self, mock_platform):
        """
        Feature: MaxDim on 3D tensor
        Description: Test max on 3D tensor with mixed placements
        Expectation: Success
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        values_layout, index_layout = op.infer_layout((x_layout, None, None), (2, True))

        expected_map = (2, 1, -1)
        assert values_layout.tensor_map == expected_map, (
            f"3D tensor test failed. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert index_layout.tensor_map == expected_map, (
            f"3D tensor test failed. Expected {expected_map}, "
            f"got {index_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_dim_invalid_dim_type(self, mock_platform):
        """
        Feature: MaxDim invalid dim type
        Description: Pass non-integer dim
        Expectation: Raise TypeError
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        with self.assertRaisesRegex(TypeError, "The `dim` argument should not be a `Tensor` or a `str`"):
            op.infer_layout((x_layout, None, None), ("invalid", True))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_dim_dim_out_of_range(self, mock_platform):
        """
        Feature: MaxDim dim out of range
        Description: Pass dim value outside valid range
        Expectation: Raise ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        with self.assertRaisesRegex(ValueError, "Invalid reduce axis index"):
            op.infer_layout((x_layout, None, None), (10, True))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_dim_invalid_layouts_count(self, mock_platform):
        """
        Feature: MaxDim invalid layouts count
        Description: Pass wrong number of layouts
        Expectation: Raise ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        with self.assertRaisesRegex(ValueError, "requires at least one input layout"):
            op.infer_layout((), (1, True))


    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_dim_none_input_layout(self, mock_platform):
        """
        Feature: MaxDim None input layout
        Description: Pass None as input layout
        Expectation: Raise ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)

        with self.assertRaisesRegex(ValueError, "requires at least one input layout"):
            op.infer_layout((None, None, None), (1, True))


if __name__ == "__main__":
    unittest.main()
