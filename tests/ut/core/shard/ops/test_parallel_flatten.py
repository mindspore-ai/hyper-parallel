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
from unittest.mock import MagicMock, patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
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
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

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

    def _infer_flatten(self, x_layout, input_shape, start_dim=0, end_dim=-1):
        """Infer flatten layout using the new cache_values calling convention."""
        output_layouts, extra_info = op.infer_layout([x_layout, start_dim, end_dim, input_shape])
        assert extra_info is None, f"Flatten extra_info should be None, got {extra_info}"
        return output_layouts[0]

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flatten_preprocess_builds_cache_values(self, mock_platform):
        """
        Feature: FlattenDistributedOp preprocess.
        Description: Verify that preprocess converts DTensor input to local tensor and caches layout/shape metadata.
        Expectation: local args, kwargs, and cache values follow the new dispatch convention.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        local_tensor = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.shape = (8, 4)
        mock_tensor.to_local.return_value = local_tensor

        local_args, local_kwargs, cache_values = op.preprocess((mock_tensor, 1), {})

        assert local_args == (local_tensor, 1, -1)
        assert not local_kwargs
        assert cache_values == [x_layout, 1, -1, (8, 4)]

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

        output_layout = self._infer_flatten(x_layout, (2, 4, 8))

        expected_map = (1,)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Flatten all dims failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

        def fake_flatten(x, start_dim=0, end_dim=-1):
            return x, start_dim, end_dim

        assert op.get_expand_impl(fake_flatten, ((output_layout,), None), [x_layout, 0, -1, (2, 4, 8)]) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(fake_flatten, ((output_layout,), None), [x_layout, 0, -1, (2, 4, 8)])}"
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

        output_layout = self._infer_flatten(x_layout, (2, 4, 8), start_dim=1)

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

        output_layout = self._infer_flatten(x_layout, ())

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

        output_layout = self._infer_flatten(x_layout, (2, 4), start_dim=1, end_dim=0)

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
            op.infer_layout([x_layout, 2, -1, (2, 4)])


if __name__ == "__main__":
    unittest.main()
