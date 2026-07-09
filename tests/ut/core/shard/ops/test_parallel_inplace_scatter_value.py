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
"""
Unit tests for InplaceScatterValueDistributedOp.
"""

import unittest
from unittest.mock import patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_inplace_scatter_value import InplaceScatterValueDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = InplaceScatterValueDistributedOp("InplaceScatterValue")


class TestInplaceScatterValue(unittest.TestCase):
    """Unit tests for InplaceScatterValueDistributedOp."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _setup_mock_platform(self, mock_platform, platform_type=None, world_size=8):
        """Configure common mock-platform attributes used across tests."""
        if platform_type is not None:
            mock_platform.platform_type = platform_type
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4 (dp, mp) mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "mp"),
            init_backend=False
        )

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, sp, mp) mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "sp", "mp"),
            init_backend=False
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_data_parallel_success(self, mock_platform):
        """Test with data parallel sharding - scatter on unsharded dimension."""
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        index_placements = (Shard(0), Replicate())
        index_layout = _build_layout(mesh, index_placements, 3)

        cache_values = [x_layout, index_layout, 1]
        infer_result = op.infer_layout(cache_values)
        output_layout = infer_result[0][0]

        assert output_layout.tensor_map == x_layout.tensor_map, (
            f"Data Parallel test failed. Expected {x_layout.tensor_map}, "
            f"got {output_layout.tensor_map}"
        )

        assert not output_layout.is_partial(), (
            f"Output layout should not be partial, got {output_layout.partial}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_model_parallel_success(self, mock_platform):
        """Test with model parallel sharding - scatter on unsharded dimension."""
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 3)

        index_placements = (Replicate(), Shard(1))
        index_layout = _build_layout(mesh, index_placements, 3)

        cache_values = [x_layout, index_layout, 0]
        infer_result = op.infer_layout(cache_values)
        output_layout = infer_result[0][0]

        assert output_layout.tensor_map == x_layout.tensor_map, (
            f"Model Parallel test failed. Expected {x_layout.tensor_map}, "
            f"got {output_layout.tensor_map}"
        )

        assert not output_layout.is_partial(), (
            f"Output layout should not be partial, got {output_layout.partial}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_hybrid_parallel_success(self, mock_platform):
        """Test with hybrid parallel sharding."""
        mesh = self._make_2x2x2_mesh(mock_platform)

        x_placements = (Shard(0), Replicate(), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        index_placements = (Shard(0), Replicate(), Shard(2))
        index_layout = _build_layout(mesh, index_placements, 3)

        cache_values = [x_layout, index_layout, 1]
        infer_result = op.infer_layout(cache_values)
        output_layout = infer_result[0][0]

        assert output_layout.tensor_map == x_layout.tensor_map, (
            f"Hybrid Parallel test failed. Expected {x_layout.tensor_map}, "
            f"got {output_layout.tensor_map}"
        )

        assert not output_layout.is_partial(), (
            f"Output layout should not be partial, got {output_layout.partial}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_all_replicated(self, mock_platform):
        """Test with all replicated layout."""
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        index_placements = (Replicate(), Replicate())
        index_layout = _build_layout(mesh, index_placements, 3)

        cache_values = [x_layout, index_layout, 2]
        infer_result = op.infer_layout(cache_values)
        output_layout = infer_result[0][0]

        assert output_layout.tensor_map == x_layout.tensor_map, (
            f"All Replicated test failed. Expected {x_layout.tensor_map}, "
            f"got {output_layout.tensor_map}"
        )

        assert not output_layout.is_partial(), (
            f"Output layout should not be partial, got {output_layout.partial}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_negative_dim(self, mock_platform):
        """Test with negative dimension index."""
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        index_placements = (Shard(0), Replicate())
        index_layout = _build_layout(mesh, index_placements, 3)

        cache_values = [x_layout, index_layout, -1]
        infer_result = op.infer_layout(cache_values)
        output_layout = infer_result[0][0]

        assert output_layout.tensor_map == x_layout.tensor_map, (
            f"Negative dim test failed. Expected {x_layout.tensor_map}, "
            f"got {output_layout.tensor_map}"
        )

        assert not output_layout.is_partial(), (
            f"Output layout should not be partial, got {output_layout.partial}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        self.assertIsNone(
            op.get_expand_impl(None, infer_result, cache_values),
            "get_expand_impl should return None",
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_sharded_dim_failure(self, mock_platform):
        """Test error when scattering on sharded dimension."""
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        index_placements = (Shard(0), Replicate())
        index_layout = _build_layout(mesh, index_placements, 3)

        cache_values = [x_layout, index_layout, 0]

        with self.assertRaisesRegex(ValueError, "scatter along sharded dimension"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_invalid_index_layout_failure(self, mock_platform):
        """Test error when index layout has different sharding on non-dim axes."""
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        index_placements = (Replicate(), Replicate())
        index_layout = _build_layout(mesh, index_placements, 3)

        cache_values = [x_layout, index_layout, 1]

        with self.assertRaisesRegex(ValueError, "input and index must use the same sharding"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_invalid_dim_failure(self, mock_platform):
        """Test error with out-of-bounds dimension index."""
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        index_placements = (Shard(0), Replicate())
        index_layout = _build_layout(mesh, index_placements, 3)

        cache_values = [x_layout, index_layout, 5]

        with self.assertRaisesRegex(ValueError, "dim .* is out of bounds"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_invalid_dim_type_failure(self, mock_platform):
        """Test error when dim is not an integer."""
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        index_placements = (Shard(0), Replicate())
        index_layout = _build_layout(mesh, index_placements, 3)

        cache_values = [x_layout, index_layout, "invalid"]

        with self.assertRaisesRegex(ValueError, "dim should be an integer"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_none_input_layout_failure(self, mock_platform):
        """Test error when input layout is None."""
        cache_values = [None, None, 0]

        with self.assertRaisesRegex(ValueError, "input layout should not be None"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_none_index_layout_failure(self, mock_platform):
        """Test error when index layout is None."""
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, None, 0]

        with self.assertRaisesRegex(ValueError, "index must be a DTensor"):
            op.infer_layout(cache_values)


if __name__ == "__main__":
    unittest.main()
