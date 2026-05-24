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
"""parallel_flatten test"""
import os
import unittest
from unittest.mock import patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_flatten import FlattenDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = FlattenDistributedOp("flatten")


class TestParallelFlatten(unittest.TestCase):
    """Unit tests for FlattenDistributedOp."""
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
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flatten_layout_inference_all_dims(self, mock_platform):
        """
        Feature: Flatten all dimensions (default behavior)
        Description: Flatten a 3D tensor across all dimensions (start_dim=0, end_dim=-1)
        Expectation: Output layout correctly merges dimensions into a 1D layout,
                     preserving the sharding of the leading sharded dimension.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        output_layout = op.infer_layout((x_layout,), extra_args=([(2, 4, 8)],))

        expected_map = (1,)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Flatten all dims failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, output_layout, (x_layout,), ([(2, 4, 8)],)) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, output_layout, (x_layout,), ([(2, 4, 8)],))}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flatten_layout_inference_specific_start_dim(self, mock_platform):
        """
        Feature: Flatten with specific start_dim
        Description: Flatten a 3D tensor starting from dim 1 (start_dim=1, end_dim=-1)
        Expectation: Leading dimension retains its original layout, and trailing dimensions
                     are merged correctly into the second dimension.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        output_layout = op.infer_layout((x_layout,), extra_args=(1, [(2, 4, 8)]))

        expected_map = (1, 0)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Flatten from start_dim failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flatten_layout_scalar(self, mock_platform):
        """
        Feature: Flatten a 0-D scalar tensor
        Description: PyTorch flatten converts a 0-D tensor into a 1-D tensor of shape (1,).
        Expectation: Output layout becomes a 1-D unsharded layout.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = ()
        x_layout = _build_layout(mesh, x_placements, 0)

        output_layout = op.infer_layout((x_layout,), extra_args=([()],))

        expected_map = (-1,)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Flatten scalar failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flatten_layout_no_op(self, mock_platform):
        """
        Feature: Flatten with start_dim > end_dim
        Description: When start_dim > end_dim, PyTorch semantics dictate that the tensor is returned unchanged.
        Expectation: Output layout exactly matches the input layout.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), extra_args=(1, 0, [(2, 4)]))

        expected_map = (1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Flatten no-op failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flatten_invalid_dimension(self, mock_platform):
        """
        Feature: Flatten with out-of-bounds dimension
        Description: Test input validation when start_dim or end_dim exceed tensor bounds.
        Expectation: ValueError is raised with clear dimension tracking.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaises(ValueError):
            op.infer_layout((x_layout,), extra_args=(2, [(2, 4)]))


if __name__ == "__main__":
    unittest.main()
