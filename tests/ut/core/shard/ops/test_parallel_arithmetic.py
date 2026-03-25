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
"""test ut for parallel arithmetic"""
import os
import unittest
from unittest.mock import patch
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_elementwise import ElementWiseDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelArithmetic(unittest.TestCase):
    """Unit tests for ElementWiseDistributedOp."""
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

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "mp"),
            init_backend=False
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_add_layout_hybrid_parallel(self, mock_platform):
        """
        Feature: add hybrid parallel
        Description: Data parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        op = ElementWiseDistributedOp("Add")

        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)
        w_placements = (Shard(0), Shard(1))
        w_layout = _build_layout(mesh, w_placements, 2)

        extra_args = {"input_shapes": [(4, 16), (4, 16)]}
        output_layout = op.infer_layout((x_layout, w_layout), (extra_args))
        expected_map = ("dp", "mp")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.alias_tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_add_layout_broadcast(self, mock_platform):
        """
        Feature: add hybrid parallel
        Description: Data parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        op = ElementWiseDistributedOp("Add")

        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)
        w_placements = (Shard(0), Replicate())
        w_layout = _build_layout(mesh, w_placements, 2)
        extra_args = {"input_shapes": [(4, 16), (4, 16)]}
        output_layout = op.infer_layout((x_layout, w_layout), (extra_args))
        expected_map = ("dp", "mp")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.alias_tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_mul_layout_broadcast(self, mock_platform):
        """
        Feature: mul hybrid parallel
        Description: Data parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        op = ElementWiseDistributedOp("Mul")

        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)
        w_placements = (Shard(0), Replicate())
        w_layout = _build_layout(mesh, w_placements, 2)
        extra_args = {"input_shapes": [(4, 16), (4, 16)]}
        output_layout = op.infer_layout((x_layout, w_layout), extra_args)
        expected_map = ("dp", "mp")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.alias_tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sub_layout_broadcast(self, mock_platform):
        """
        Feature: sub hybrid parallel
        Description: Data parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        op = ElementWiseDistributedOp("Sub")

        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)
        w_placements = (Shard(0), Replicate())
        w_layout = _build_layout(mesh, w_placements, 2)
        extra_args = {"input_shapes": [(4, 16), (4, 16)]}
        output_layout = op.infer_layout((x_layout, w_layout), extra_args)
        expected_map = ("dp", "mp")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.alias_tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_div_layout_broadcast(self, mock_platform):
        """
        Feature: div hybrid parallel
        Description: Data parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        op = ElementWiseDistributedOp("Div")

        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)
        w_placements = (Shard(0), Replicate())
        w_layout = _build_layout(mesh, w_placements, 2)
        extra_args = {"input_shapes": [(4, 16), (4, 16)]}
        output_layout = op.infer_layout((x_layout, w_layout), extra_args)
        expected_map = ("dp", "mp")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.alias_tensor_map}"
        )


if __name__ == "__main__":
    unittest.main()
