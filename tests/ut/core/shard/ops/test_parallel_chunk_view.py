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
import os
import unittest
from unittest.mock import patch
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_chunk_view import ChunkViewDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelChunkView(unittest.TestCase):
    """Unit tests for ChunkViewDistributedOp."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()
        self.chunk_view_op = ChunkViewDistributedOp("chunk_view")
    
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
    
    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, mp, cp) mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "mp", "cp"),
            init_backend=False
        )
    
    def _make_4x2_mesh(self, mock_platform):
        """Set up mock and return a standard 4x2 (dp, mp) mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(4, 2),
            mesh_dim_names=("dp", "mp"),
            init_backend=False
        )
    
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_data_parallel_success(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with data parallel
        Description: Test chunk_view with data parallel sharding (dim 0 sharded, split on dim 1)
        Expectation: Output layouts are correctly generated with same tensor_map
        """
        mesh = self._make_4x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(0), Replicate()), 3)
        
        chunks = 4
        dim = 1
        input_shape = (8, 16, 32)
        input_shapes = [input_shape]
        extra_args = [chunks, dim, input_shapes]
        
        output_layouts = self.chunk_view_op.infer_layout([input_layout], extra_args)
        
        assert len(output_layouts) == 4, (
            f"Expected 4 output layouts, got {len(output_layouts)}"
        )
        assert all(layout.tensor_map == input_layout.tensor_map for layout in output_layouts), (
            "Output layouts should have same tensor_map as input"
        )
        
        assert self.chunk_view_op.get_expand_impl(None, output_layouts, [input_layout], extra_args) is None, (
            f"get_expand_impl should return None" \
            f"got {self.chunk_view_op.get_expand_impl(None, output_layouts, [input_layout], extra_args)}"
        )
    
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_model_parallel_success(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with model parallel
        Description: Test chunk_view with model parallel sharding (dim 1 sharded, split on dim 0)
        Expectation: Output layouts are correctly generated with same tensor_map
        """
        mesh = self._make_4x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Shard(1)), 3)
        
        chunks = 2
        dim = 0
        input_shape = (8, 16, 32)
        input_shapes = [input_shape]
        extra_args = [chunks, dim, input_shapes]
        
        output_layouts = self.chunk_view_op.infer_layout([input_layout], extra_args)
        
        assert len(output_layouts) == 2, (
            f"Expected 2 output layouts, got {len(output_layouts)}"
        )
        assert all(layout.tensor_map == input_layout.tensor_map for layout in output_layouts), (
            "Output layouts should have same tensor_map as input"
        )
    
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_hybrid_parallel_success(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with hybrid parallel
        Description: Test chunk_view with hybrid parallel (multiple dimensions sharded)
        Expectation: Output layouts are correctly generated with same tensor_map
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)
        
        chunks = 3
        dim = 2
        input_shape = (8, 16, 12)
        input_shapes = [input_shape]
        extra_args = [chunks, dim, input_shapes]
        
        output_layouts = self.chunk_view_op.infer_layout([input_layout], extra_args)
        
        assert len(output_layouts) == 3, (
            f"Expected 3 output layouts, got {len(output_layouts)}"
        )
        assert all(layout.tensor_map == input_layout.tensor_map for layout in output_layouts), (
            "Output layouts should have same tensor_map as input"
        )
    
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_all_replicated(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with all replicated
        Description: Test chunk_view with no sharding
        Expectation: Output layouts are correctly generated with same tensor_map
        """
        mesh = self._make_4x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        
        chunks = 5
        dim = 0
        input_shape = (20, 16)
        input_shapes = [input_shape]
        extra_args = [chunks, dim, input_shapes]
        
        output_layouts = self.chunk_view_op.infer_layout([input_layout], extra_args)
        
        assert len(output_layouts) == 5, (
            f"Expected 5 output layouts, got {len(output_layouts)}"
        )
        assert all(layout.tensor_map == input_layout.tensor_map for layout in output_layouts), (
            "Output layouts should have same tensor_map as input"
        )
    
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
        input_shapes = [input_shape]
        extra_args = [chunks, dim, input_shapes]
        
        output_layouts = self.chunk_view_op.infer_layout([input_layout], extra_args)
        
        assert len(output_layouts) == 2, (
            f"Expected 2 output layouts, got {len(output_layouts)}"
        )
        assert all(layout.tensor_map == input_layout.tensor_map for layout in output_layouts), (
            "Output layouts should have same tensor_map as input"
        )
    
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
        input_shapes = [input_shape]
        extra_args = [chunks, dim, input_shapes]
        
        with self.assertRaisesRegex(ValueError, "Cannot split tensor at sharded axis"):
            self.chunk_view_op.infer_layout([input_layout], extra_args)
    
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
        input_shapes = [input_shape]
        extra_args = [chunks, dim, input_shapes]
        
        with self.assertRaisesRegex(ValueError, "Dimension out of range"):
            self.chunk_view_op.infer_layout([input_layout], extra_args)
    
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
        input_shapes = [input_shape]
        extra_args = [chunks, dim, input_shapes]
        
        with self.assertRaisesRegex(ValueError, "chunks must be greater than 0"):
            self.chunk_view_op.infer_layout([input_layout], extra_args)
    
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_missing_chunks(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with missing chunks
        Description: Test error when chunks is not provided
        Expectation: ValueError is raised
        """
        mesh = self._make_4x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        
        extra_args = []
        
        with self.assertRaisesRegex(ValueError, "chunk_view requires 'chunks' in extra_args"):
            self.chunk_view_op.infer_layout([input_layout], extra_args)
    
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_none_input(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with None input
        Description: Test error when input layout is None
        Expectation: ValueError is raised
        """
        input_shape = (8, 16)
        input_shapes = [input_shape]
        extra_args = [2, 0, input_shapes]
        
        with self.assertRaisesRegex(ValueError, "chunk_view requires a valid input tensor layout"):
            self.chunk_view_op.infer_layout([None], extra_args)
    
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
        input_shapes = [input_shape]
        extra_args = [chunks, dim, input_shapes]
        
        with self.assertRaisesRegex(TypeError, "chunks must be an integer"):
            self.chunk_view_op.infer_layout([input_layout], extra_args)
    
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
        input_shapes = [input_shape]
        extra_args = [chunks, dim, input_shapes]
        
        with self.assertRaisesRegex(TypeError, "dim must be an integer"):
            self.chunk_view_op.infer_layout([input_layout], extra_args)
    
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_default_dim(self, mock_platform):
        """
        Feature: ChunkView operator layout inference with default dimension
        Description: Test when dim is not provided (should default to 0)
        Expectation: Default dimension 0 is used
        """
        mesh = self._make_4x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Shard(1)), 3)
        
        chunks = 2
        input_shape = (8, 16, 32)
        input_shapes = [input_shape]
        extra_args = [chunks, input_shapes]
        
        output_layouts = self.chunk_view_op.infer_layout([input_layout], extra_args)
        
        assert len(output_layouts) == 2, (
            f"Expected 2 output layouts, got {len(output_layouts)}"
        )
        assert all(layout.tensor_map == input_layout.tensor_map for layout in output_layouts), (
            "Output layouts should have same tensor_map as input"
        )

if __name__ == "__main__":
    unittest.main()
