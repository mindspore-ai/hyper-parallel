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
"""parallel_chunk_view test"""
import unittest
from unittest.mock import patch

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_chunk_view import (
    ChunkViewDistributedOp,
    _normalize_chunk_view_args,
)
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP,
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelChunkView(unittest.TestCase):
    """Unit tests for ChunkViewDistributedOp."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()
        self.chunk_view_op = ChunkViewDistributedOp("chunk_view")

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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, mp, cp) mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "mp", "cp"),
            init_backend=False,
        )

    def _make_4x2_mesh(self, mock_platform):
        """Set up mock and return a standard 4x2 (dp, mp) mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(4, 2),
            mesh_dim_names=("dp", "mp"),
            init_backend=False,
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_data_parallel_success(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with data parallel
        Description: Test chunk_view with data parallel sharding (dim 0 sharded, split on dim 1)
        Expectation: Output layouts are correctly generated with same alias_tensor_map
        """
        mesh = self._make_4x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(0), Replicate()), 3)

        chunks = 4
        dim = 1
        input_shape = (8, 16, 32)
        cache_values = [input_layout, chunks, dim, input_shape]

        output_layouts, extra_info = self.chunk_view_op.infer_layout(cache_values)
        assert extra_info is None, f"extra_info should be None, got {extra_info}"

        assert len(output_layouts) == 4, (
            f"Expected 4 output layouts, got {len(output_layouts)}"
        )
        assert all(
            layout.alias_tensor_map == input_layout.alias_tensor_map
            for layout in output_layouts
        ), "Output layouts should have same alias_tensor_map as input"

        # Since get_expand_impl is not overridden, it returns None by default.
        # The same applies to other test cases, so it is unnecessary to test its return value.
        self.assertIsNone(
            self.chunk_view_op.get_expand_impl(None, (output_layouts, None), cache_values),
            "get_expand_impl should return None",
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_model_parallel_success(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with model parallel
        Description: Test chunk_view with model parallel sharding (dim 1 sharded, split on dim 0)
        Expectation: Output layouts are correctly generated with same alias_tensor_map
        """
        mesh = self._make_4x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Shard(1)), 3)

        chunks = 2
        dim = 0
        input_shape = (8, 16, 32)
        cache_values = [input_layout, chunks, dim, input_shape]

        output_layouts, extra_info = self.chunk_view_op.infer_layout(cache_values)
        assert extra_info is None, f"extra_info should be None, got {extra_info}"

        assert len(output_layouts) == 2, (
            f"Expected 2 output layouts, got {len(output_layouts)}"
        )
        assert all(
            layout.alias_tensor_map == input_layout.alias_tensor_map
            for layout in output_layouts
        ), "Output layouts should have same alias_tensor_map as input"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_hybrid_parallel_success(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with hybrid parallel
        Description: Test chunk_view with hybrid parallel (multiple dimensions sharded)
        Expectation: Output layouts are correctly generated with same alias_tensor_map
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)

        chunks = 3
        dim = 2
        input_shape = (8, 16, 12)
        cache_values = [input_layout, chunks, dim, input_shape]

        output_layouts, extra_info = self.chunk_view_op.infer_layout(cache_values)
        assert extra_info is None, f"extra_info should be None, got {extra_info}"

        assert len(output_layouts) == 3, (
            f"Expected 3 output layouts, got {len(output_layouts)}"
        )
        assert all(
            layout.alias_tensor_map == input_layout.alias_tensor_map
            for layout in output_layouts
        ), "Output layouts should have same alias_tensor_map as input"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_all_replicated(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with all replicated
        Description: Test chunk_view with no sharding
        Expectation: Output layouts are correctly generated with same alias_tensor_map
        """
        mesh = self._make_4x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        chunks = 5
        dim = 0
        input_shape = (20, 16)
        cache_values = [input_layout, chunks, dim, input_shape]

        output_layouts, extra_info = self.chunk_view_op.infer_layout(cache_values)
        assert extra_info is None, f"extra_info should be None, got {extra_info}"

        assert len(output_layouts) == 5, (
            f"Expected 5 output layouts, got {len(output_layouts)}"
        )
        assert all(
            layout.alias_tensor_map == input_layout.alias_tensor_map
            for layout in output_layouts
        ), "Output layouts should have same alias_tensor_map as input"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_negative_dim(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with negative dimension
        Description: Test chunk_view with negative dimension index
        Expectation: Negative dimension is correctly handled
        """
        mesh = self._make_4x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(0), Replicate()), 3)

        chunks = 2
        dim = -1
        input_shape = (8, 16, 32)
        cache_values = [input_layout, chunks, dim, input_shape]

        output_layouts, extra_info = self.chunk_view_op.infer_layout(cache_values)
        assert extra_info is None, f"extra_info should be None, got {extra_info}"

        assert len(output_layouts) == 2, (
            f"Expected 2 output layouts, got {len(output_layouts)}"
        )
        assert all(
            layout.alias_tensor_map == input_layout.alias_tensor_map
            for layout in output_layouts
        ), "Output layouts should have same alias_tensor_map as input"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_sharded_dim_failure(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with sharded dimension
        Description: Test error when trying to split a sharded dimension
        Expectation: ValueError is raised
        """
        mesh = self._make_4x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(0), Replicate()), 3)

        chunks = 2
        dim = 0
        input_shape = (8, 16, 32)
        cache_values = [input_layout, chunks, dim, input_shape]

        with self.assertRaisesRegex(ValueError, "cannot split tensor at sharded axis"):
            self.chunk_view_op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_dim_out_of_range(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with dimension out of range
        Description: Test error when dimension is out of valid range
        Expectation: ValueError is raised
        """
        mesh = self._make_4x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        chunks = 2
        dim = 5
        input_shape = (8, 16)
        cache_values = [input_layout, chunks, dim, input_shape]

        with self.assertRaisesRegex(ValueError, "dimension out of range"):
            self.chunk_view_op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_invalid_chunks(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with invalid chunks
        Description: Test error when chunks is less than 1
        Expectation: ValueError is raised
        """
        mesh = self._make_4x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        chunks = 0
        dim = 0
        input_shape = (8, 16)
        cache_values = [input_layout, chunks, dim, input_shape]

        with self.assertRaisesRegex(ValueError, "chunks must be greater than 0"):
            self.chunk_view_op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_none_input(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with None input
        Description: Test error when input layout is None
        Expectation: ValueError is raised
        """
        cache_values = [None, 2, 0, (8, 16)]

        with self.assertRaisesRegex(ValueError, "input layout should not be None"):
            self.chunk_view_op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_invalid_chunks_type(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with invalid chunks type
        Description: Test error when chunks is not an integer
        Expectation: TypeError is raised
        """
        mesh = self._make_4x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        chunks = 2.5
        dim = 0
        input_shape = (8, 16)
        cache_values = [input_layout, chunks, dim, input_shape]

        with self.assertRaisesRegex(TypeError, "chunks must be an integer"):
            self.chunk_view_op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_invalid_dim_type(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with invalid dim type
        Description: Test error when dim is not an integer
        Expectation: TypeError is raised
        """
        mesh = self._make_4x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        chunks = 2
        dim = 1.5
        input_shape = (8, 16)
        cache_values = [input_layout, chunks, dim, input_shape]

        with self.assertRaisesRegex(TypeError, "dim must be an integer"):
            self.chunk_view_op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_default_dim(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with default dimension
        Description: Test when dim defaults to 0 (split along first dimension)
        Expectation: Default dimension 0 is used and layout inference succeeds
        """
        mesh = self._make_4x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Shard(1)), 3)

        chunks = 2
        dim = 0
        input_shape = (8, 16, 32)
        cache_values = [input_layout, chunks, dim, input_shape]

        output_layouts, extra_info = self.chunk_view_op.infer_layout(cache_values)
        assert extra_info is None, f"extra_info should be None, got {extra_info}"

        assert len(output_layouts) == 2, (
            f"Expected 2 output layouts, got {len(output_layouts)}"
        )
        assert all(
            layout.alias_tensor_map == input_layout.alias_tensor_map
            for layout in output_layouts
        ), "Output layouts should have same alias_tensor_map as input"

    def test_normalize_chunk_view_args_all_positional(self):
        """
        Feature: ChunkView argument normalization
        Description: Test that _normalize_chunk_view_args correctly normalizes all positional args
        Expectation: Returns normalized args tuple and empty kwargs
        """
        input_obj = object()
        args, kwargs = _normalize_chunk_view_args(input_obj, 4, 1)
        assert args == (input_obj, 4, 1), f"Expected (input, 4, 1), got {args}"
        assert not kwargs, f"Expected empty kwargs, got {kwargs}"

    def test_normalize_chunk_view_args_default_dim(self):
        """
        Feature: ChunkView argument normalization with default dim
        Description: Test that _normalize_chunk_view_args defaults dim to 0 when not provided
        Expectation: dim defaults to 0
        """
        input_obj = object()
        args, kwargs = _normalize_chunk_view_args(input_obj, 2)
        assert args == (input_obj, 2, 0), (
            f"Expected (input, 2, 0) with dim defaulting to 0, got {args}"
        )
        assert not kwargs, f"Expected empty kwargs, got {kwargs}"


if __name__ == "__main__":
    unittest.main()
    