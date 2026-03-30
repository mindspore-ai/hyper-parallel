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
"""parallel_topk test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_topk import TopKDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = TopKDistributedOp("TopK")
torch_op = TopKDistributedOp("topk")


class TestParallelTopK(unittest.TestCase):
    """Unit tests for TopKDistributedOp."""
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

    def _make_2x4x3_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4x3 (dp, mp, tp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=24)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4, 3), mesh_dim_names=("dp", "mp", "tp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_topk_layout_data_parallel(self, mock_platform):
        """
        Feature: TopK data parallel
        Description: Data parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        values_layout, indices_layout = op.infer_layout((x_layout,), ())
        expected_map = (1, -1)
        assert values_layout.tensor_map == indices_layout.tensor_map == expected_map, (
            f"Data Parallel with transpose_a test failed. Expected {expected_map},"
            f" got {values_layout.tensor_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, (values_layout, indices_layout), (x_layout,), ()) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, (values_layout, indices_layout), (x_layout,), ())}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_topk_layout_tensor_parallel(self, mock_platform):
        """
        Feature: TopK tensor parallel
        Description: Tensor parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)

        values_layout, indices_layout = op.infer_layout((x_layout,), ())
        expected_map = (0, -1)
        assert values_layout.tensor_map == indices_layout.tensor_map == expected_map, (
            f"Data Parallel with transpose_a test failed. Expected {expected_map},"
            f" got {values_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_topk_layout_tensor_and_data_parallel(self, mock_platform):
        """
        Feature: TopK tensor parallel
        Description: Tensor parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4x3_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)

        values_layout, indices_layout = op.infer_layout((x_layout,), ())
        expected_map = (2, 1, -1)
        assert values_layout.tensor_map == indices_layout.tensor_map == expected_map, (
            f"Data Parallel with transpose_a test failed. Expected {expected_map},"
            f" got {values_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_topk_layout_data_parallel(self, mock_platform):
        """
        Feature: TopK data parallel
        Description: Data parallel scenario (shard on first dim, topk on last unsharded dim)
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        values_layout, indices_layout = torch_op.infer_layout((x_layout,), ())
        expected_map = (1, -1)
        assert values_layout.tensor_map == indices_layout.tensor_map == expected_map, (
            f"Data Parallel with torch topk test failed. Expected {expected_map},"
            f" got {values_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_topk_layout_tensor_parallel(self, mock_platform):
        """
        Feature: TopK tensor parallel
        Description: Tensor parallel scenario (shard on last dim is NOT allowed, so shard on first dim with 'mp')
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)

        values_layout, indices_layout = torch_op.infer_layout((x_layout,), ())
        expected_map = (0, -1)
        assert values_layout.tensor_map == indices_layout.tensor_map == expected_map, (
            f"Data Parallel with torch topk test failed. Expected {expected_map},"
            f" got {values_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_topk_layout_mixed_parallel_invalid(self, mock_platform):
        """
        Feature: Test topk on a sharded dimension
        Description: Test topk on a sharded dimension
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        with self.assertRaisesRegex(ValueError, "Cannot perform sharding on params along the chosen dim"):
            torch_op.infer_layout((x_layout,), ())

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_topk_layout_error_dim_out_of_range(self, mock_platform):
        """
        Feature: Test indicating a invalid dim
        Description: Test indicating a invalid dim
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "Dimension out of range"):
            torch_op.infer_layout((x_layout,), extra_args=(2, 5))


if __name__ == "__main__":
    unittest.main()
