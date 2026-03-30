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
"""parallel_argsort test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_argsort import ArgsortDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = ArgsortDistributedOp("argsort")


class TestParallelArgsort(unittest.TestCase):
    """Unit tests for ArgsortDistributedOp."""
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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, tp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "tp", "mp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argsort_layout_inference_basic(self, mock_platform):
        """
        Feature: Argsort on an unsharded dimension
        Description: Perform argsort along the last dimension (dim=-1), which is fully replicated.
        Expectation: Output layout is identical to the input layout.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), extra_args=(-1,))

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Basic argsort failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, output_layout, (x_layout,), (-1,)) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, output_layout, (x_layout,), (-1,))}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argsort_layout_inference_specific_dim(self, mock_platform):
        """
        Feature: Argsort on a specific unsharded dimension with extra kwargs
        Description: Perform argsort on dim=0, with descending=True. dim=0 is Replicate.
        Expectation: Output layout is identical to the input layout.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), extra_args=(0, True, False))

        expected_map = (-1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Specific dim argsort failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argsort_layout_inference_negative_dim(self, mock_platform):
        """
        Feature: Argsort handling of negative dimensions
        Description: Perform argsort on a 3D tensor using dim=-2, which maps to the middle unsharded dimension.
        Expectation: Resolves the negative dimension correctly and returns the identical layout.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        output_layout = op.infer_layout((x_layout,), extra_args=(-2,))

        expected_map = (2, -1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Negative dim argsort failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argsort_layout_invalid_sharded_dim(self, mock_platform):
        """
        Feature: Argsort on a sharded dimension
        Description: Attempt to perform argsort along a dimension that is currently sharded.
        Expectation: ValueError is raised preventing the mathematically incorrect operation.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "Cannot perform argsort along dimension 0 because it is currently sharded"):
            op.infer_layout((x_layout,), extra_args=(0,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argsort_layout_invalid_out_of_bounds_dim(self, mock_platform):
        """
        Feature: Argsort with out-of-bounds dimension
        Description: Attempt to perform argsort using a dimension index larger than tensor rank.
        Expectation: ValueError is raised.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "is out of bounds for tensor of dimension 2"):
            op.infer_layout((x_layout,), extra_args=(2,))


if __name__ == "__main__":
    unittest.main()
