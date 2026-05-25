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
"""parallel_atleast_1d test"""
import os
import unittest
from unittest.mock import patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial
from hyper_parallel.core.shard.ops.parallel_atleast_1d import Atleast1DDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelAtleast1d(unittest.TestCase):
    """Unit tests for Atleast1DDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.op = Atleast1DDistributedOp("atleast_1d")

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
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "mp"),
            init_backend=False
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_atleast_1d_layout_inference_0d(self, mock_platform):
        """
        Feature: Convert 0D scalar to 1D
        Description: Input is a 0-dimensional scalar tensor.
        Expectation: Output layout is 1-dimensional and fully unsharded (Replicate).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 0)

        output_layout = self.op.infer_layout((x_layout,))

        expected_map = (-1,)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"0D to 1D conversion failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert self.op.get_expand_impl(None, output_layout, (x_layout,), None) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {self.op.get_expand_impl(None, output_layout, (x_layout,), None)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_atleast_1d_layout_inference_1d(self, mock_platform):
        """
        Feature: Preserve 1D tensor layout
        Description: Input is a 1-dimensional tensor sharded across 'dp'.
        Expectation: Output layout is identical to the input layout (preserves sharding).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 1)

        output_layout = self.op.infer_layout((x_layout,))

        expected_map = (1,)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"1D preservation failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_atleast_1d_layout_inference_nd(self, mock_platform):
        """
        Feature: Preserve N-dimensional tensor layout (N > 1)
        Description: Input is a 2-dimensional tensor with mixed sharding.
        Expectation: Output layout is identical to the input layout.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = self.op.infer_layout((x_layout,))

        expected_map = (1, 0)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"ND preservation failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_atleast_1d_invalid_partial(self, mock_platform):
        """
        Feature: Reject Partial status
        Description: Input layout contains a Partial reduction state.
        Expectation: ValueError raised with clear message (atleast_1d does not support partials).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Partial("sum"), Replicate())
        x_layout = _build_layout(mesh, x_placements, 1)

        with self.assertRaisesRegex(ValueError, "Partial status which is not allowed"):
            self.op.infer_layout((x_layout,))


if __name__ == "__main__":
    unittest.main()
