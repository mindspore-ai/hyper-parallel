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

import os
import unittest
from unittest.mock import patch
import numpy as np

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_inplace_scatter_value import InplaceScatterValueDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = InplaceScatterValueDistributedOp("InplaceScatterValue")


class TestInplaceScatterValue(unittest.TestCase):
    """Unit tests for InplaceScatterValueDistributedOp."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

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

        extra_args = [1, 10.0]

        output_layout = op.infer_layout((x_layout, None, index_layout, None), extra_args)

        assert output_layout.tensor_map == x_layout.tensor_map, (
            f"Data Parallel test failed. Expected {x_layout.tensor_map}, "
            f"got {output_layout.tensor_map}"
        )

        assert not output_layout.is_partial(), (
            f"Output layout should not be partial, got {output_layout.partial}"
        )

        assert op.get_expand_impl(None, output_layout, (x_layout, None, index_layout, None), extra_args) is None, (
            f"get_expand_impl should return None, "
            f"got {op.get_expand_impl(None, output_layout, (x_layout, None, index_layout, None), extra_args)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_model_parallel_success(self, mock_platform):
        """Test with model parallel sharding - scatter on unsharded dimension."""
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 3)

        index_placements = (Replicate(), Shard(1))
        index_layout = _build_layout(mesh, index_placements, 3)

        extra_args = [0, 5.0]

        output_layout = op.infer_layout((x_layout, None, index_layout, None), extra_args)

        assert output_layout.tensor_map == x_layout.tensor_map, (
            f"Model Parallel test failed. Expected {x_layout.tensor_map}, "
            f"got {output_layout.tensor_map}"
        )

        assert not output_layout.is_partial(), (
            f"Output layout should not be partial, got {output_layout.partial}"
        )

        assert op.get_expand_impl(None, output_layout, (x_layout, None, index_layout, None), extra_args) is None, (
            f"get_expand_impl should return None, "
            f"got {op.get_expand_impl(None, output_layout, (x_layout, None, index_layout, None), extra_args)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_hybrid_parallel_success(self, mock_platform):
        """Test with hybrid parallel sharding."""
        mesh = self._make_2x2x2_mesh(mock_platform)

        x_placements = (Shard(0), Replicate(), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        index_placements = (Shard(0), Replicate(), Shard(2))
        index_layout = _build_layout(mesh, index_placements, 3)

        extra_args = [1, 7.0]

        output_layout = op.infer_layout((x_layout, None, index_layout, None), extra_args)

        assert output_layout.tensor_map == x_layout.tensor_map, (
            f"Hybrid Parallel test failed. Expected {x_layout.tensor_map}, "
            f"got {output_layout.tensor_map}"
        )

        assert not output_layout.is_partial(), (
            f"Output layout should not be partial, got {output_layout.partial}"
        )

        assert op.get_expand_impl(None, output_layout, (x_layout, None, index_layout, None), extra_args) is None, (
            f"get_expand_impl should return None, "
            f"got {op.get_expand_impl(None, output_layout, (x_layout, None, index_layout, None), extra_args)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_all_replicated(self, mock_platform):
        """Test with all replicated layout."""
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        index_placements = (Replicate(), Replicate())
        index_layout = _build_layout(mesh, index_placements, 3)

        extra_args = [2, 3.0]

        output_layout = op.infer_layout((x_layout, None, index_layout, None), extra_args)

        assert output_layout.tensor_map == x_layout.tensor_map, (
            f"All Replicated test failed. Expected {x_layout.tensor_map}, "
            f"got {output_layout.tensor_map}"
        )

        assert not output_layout.is_partial(), (
            f"Output layout should not be partial, got {output_layout.partial}"
        )

        assert op.get_expand_impl(None, output_layout, (x_layout, None, index_layout, None), extra_args) is None, (
            f"get_expand_impl should return None, "
            f"got {op.get_expand_impl(None, output_layout, (x_layout, None, index_layout, None), extra_args)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_negative_dim(self, mock_platform):
        """Test with negative dimension index."""
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        index_placements = (Shard(0), Replicate())
        index_layout = _build_layout(mesh, index_placements, 3)

        extra_args = [-1, 9.0]

        output_layout = op.infer_layout((x_layout, None, index_layout, None), extra_args)

        assert output_layout.tensor_map == x_layout.tensor_map, (
            f"Negative dim test failed. Expected {x_layout.tensor_map}, "
            f"got {output_layout.tensor_map}"
        )

        assert not output_layout.is_partial(), (
            f"Output layout should not be partial, got {output_layout.partial}"
        )

        assert op.get_expand_impl(None, output_layout, (x_layout, None, index_layout, None), extra_args) is None, (
            f"get_expand_impl should return None, "
            f"got {op.get_expand_impl(None, output_layout, (x_layout, None, index_layout, None), extra_args)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_sharded_dim_failure(self, mock_platform):
        """Test error when scattering on sharded dimension."""
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        index_placements = (Shard(0), Replicate())
        index_layout = _build_layout(mesh, index_placements, 3)

        extra_args = [0, 12.0]

        with self.assertRaisesRegex(ValueError, "Scatter along sharded dimension"):
            op.infer_layout((x_layout, None, index_layout, None), extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_invalid_index_layout_failure(self, mock_platform):
        """Test error when index layout is not fully replicated."""
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        index_placements = (Replicate(), Replicate())
        index_layout = _build_layout(mesh, index_placements, 3)

        extra_args = [1, 13.0]

        with self.assertRaisesRegex(ValueError, "input and index must use the same sharding on non-dim axis"):
            op.infer_layout((x_layout, None, index_layout, None), extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_invalid_dim_failure(self, mock_platform):
        """Test error with invalid dimension index."""
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        index_placements = (Shard(0), Replicate())
        index_layout = _build_layout(mesh, index_placements, 3)

        extra_args = [5, 14.0]

        with self.assertRaisesRegex(ValueError, "dim .* is out of bounds"):
            op.infer_layout((x_layout, None, index_layout, None), extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_missing_dim_failure(self, mock_platform):
        """Test error when dim parameter is missing."""
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        index_placements = (Shard(0), Replicate())
        index_layout = _build_layout(mesh, index_placements, 3)

        extra_args = []

        with self.assertRaisesRegex(ValueError, "extra_args must contain exactly 2 elements"):
            op.infer_layout((x_layout, None, index_layout, None), extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_invalid_dim_type_failure(self, mock_platform):
        """Test error when dim is not an integer."""
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        index_placements = (Shard(0), Replicate())
        index_layout = _build_layout(mesh, index_placements, 3)

        extra_args = ["invalid", 15.0]

        with self.assertRaisesRegex(ValueError, "'dim' must be an integer"):
            op.infer_layout((x_layout, None, index_layout, None), extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_none_input_layout_failure(self, mock_platform):
        """Test error when input layout is None."""
        extra_args = [0, 16.0]

        with self.assertRaisesRegex(ValueError, "input tensor layout cannot be None"):
            op.infer_layout((None, None, None, None), extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_scatter_value_none_index_layout_failure(self, mock_platform):
        """Test error when index layout is None."""
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        extra_args = [0, 17.0]

        with self.assertRaisesRegex(ValueError, "index tensor layout cannot be None"):
            op.infer_layout((x_layout, None, None, None), extra_args)


if __name__ == "__main__":
    unittest.main()
