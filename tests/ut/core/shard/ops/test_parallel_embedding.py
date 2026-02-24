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
"""parallel_embedding test"""
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Set platform to mindspore for consistent test environment
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial
from hyper_parallel.core.shard.ops.parallel_embedding import EmbeddingDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

# Initialize distributed ops for both 'embedding' and 'Embedding'
embedding_ops = [
    EmbeddingDistributedOp("embedding"),
    EmbeddingDistributedOp("Embedding")
]


class TestParallelEmbedding(unittest.TestCase):
    """Unit tests for EmbeddingDistributedOp covering both functional and class interfaces."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _setup_mock_platform(self, mock_platform, world_size=8):
        """Configure common mock-platform attributes."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4 (dp, mp) mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"))

    def _make_2x2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "vp", "mp")):
        """Set up mock and return a standard 2x2x2 mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=mesh_dim_names)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_embedding_data_parallel(self, mock_platform):
        """
        Feature: Data Parallel for Embedding
        Description: Input sharded on batch dimension, weights fully replicated.
        Expectation: Output layout inherits input sharding, no partial state.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        # Input [batch, seq]: sharded on dp (dim 0) -> map (1, -1)
        input_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        # Weight [vocab, embed]: replicated -> map (-1, -1)
        weight_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        for op in embedding_ops:
            output_layout = op.infer_layout((input_layout, weight_layout))
            # Expected map: input(1, -1) + weight_embed(-1) = (1, -1, -1)
            expected_map = (1, -1, -1)
            self.assertEqual(output_layout.tensor_map, expected_map, f"Op {op.op_name} failed")
            self.assertFalse(output_layout.is_partial())
            # Implementation should be native (None) for DP
            self.assertIsNone(op.get_expand_impl(None, output_layout, (input_layout, weight_layout), None))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_embedding_column_parallel(self, mock_platform):
        """
        Feature: Column Parallel (Model Parallel) for Embedding
        Description: Weights sharded on the embedding dimension (dim 1).
        Expectation: Output layout inherits embed dimension sharding mapping.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        # Input [batch, seq]: sharded on dp -> map (1, -1)
        input_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        # Weight [vocab, embed]: sharded on mp (dim 1) -> map (-1, 0)
        weight_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        for op in embedding_ops:
            output_layout = op.infer_layout((input_layout, weight_layout))
            # Expected map: input(1, -1) + weight_embed(0) = (1, -1, 0)
            expected_map = (1, -1, 0)
            self.assertEqual(output_layout.tensor_map, expected_map, f"Op {op.op_name} failed")
            
            # CP requires wrapper to intercept max_norm
            impl = op.get_expand_impl(None, output_layout, (input_layout, weight_layout), None)
            self.assertTrue(callable(impl))
            with self.assertRaisesRegex(ValueError, "Column-Parallel.*does not support `max_norm`"):
                impl(MagicMock(), MagicMock(), max_norm=1.0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_embedding_row_parallel(self, mock_platform):
        """
        Feature: Row Parallel for Embedding (Vocab Sharding)
        Description: Weights sharded on vocab dimension (dim 0).
        Expectation: Output gets Partial Sum state on the vocab sharding axis (mp).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        # Weight vocab sharded on mp (Shard(0) on second mesh axis) -> map (0, -1)
        weight_layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)

        for op in embedding_ops:
            output_layout = op.infer_layout((input_layout, weight_layout))
            # Expected map: input(1, -1) + weight_embed(-1) = (1, -1, -1)
            self.assertEqual(output_layout.tensor_map, (1, -1, -1))
            self.assertTrue(output_layout.is_partial())
            
            # Partial should be on mp axis (index 1)
            mp_idx = mesh.axis_index("mp")
            self.assertEqual(output_layout.partial[mp_idx], "sum")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_embedding_row_and_column_parallel_3d(self, mock_platform):
        """
        Feature: Row and Column Parallel on 3D Mesh
        Description: Weight sharded on vocab (vp) and embed (mp) dimensions.
        Expectation: Output inherits embed sharding and generates partial sum on vp.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        # Input sharded on dp (dim 0) -> map (2, -1)
        input_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 2)
        # Weight vocab sharded on vp (dim 1), embed sharded on mp (dim 2) -> map (1, 0)
        weight_layout = _build_layout(mesh, (Replicate(), Shard(0), Shard(1)), 2)

        for op in embedding_ops:
            output_layout = op.infer_layout((input_layout, weight_layout))
            # Expected map: input(2, -1) + weight_embed(0) = (2, -1, 0)
            expected_map = (2, -1, 0)
            self.assertEqual(output_layout.tensor_map, expected_map)
            
            vp_idx = mesh.axis_index("vp")
            self.assertEqual(output_layout.partial[vp_idx], "sum")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_embedding_seq_dimension_sharding(self, mock_platform):
        """
        Feature: Sequence dimension sharding
        Description: Input sharded on sequence dimension (dim 1), weight replicated.
        Expectation: Output preserves sequence sharding, embed dim replicated.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        # Input map (-1, 0)
        input_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        weight_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        for op in embedding_ops:
            output_layout = op.infer_layout((input_layout, weight_layout))
            self.assertEqual(output_layout.tensor_map, (-1, 0, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_embedding_invalid_missing_weight_layout(self, mock_platform):
        """
        Feature: Input validation
        Description: Pass only input layout to infer_layout.
        Expectation: ValueError raised.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        for op in embedding_ops:
            with self.assertRaisesRegex(ValueError, "requires both input and weight layouts"):
                op.infer_layout((input_layout,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_embedding_partial_conflict_error(self, mock_platform):
        """
        Feature: Sharding/Partial conflict detection
        Description: Input batch sharded on 'dp', weight vocab also sharded on 'dp'.
        Expectation: ValueError raised (Partial and Shard on same axis).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        # Weight vocab sharded on dp (mesh index 0)
        weight_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        for op in embedding_ops:
            with self.assertRaisesRegex(ValueError, "Partial dim must be replicate"):
                op.infer_layout((input_layout, weight_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_embedding_row_parallel_params_validation(self, mock_platform):
        """
        Feature: Row Parallel runtime parameter validation
        Description: Attempt to use scale_grad_by_freq with row sharding.
        Expectation: ValueError raised by the impl wrapper.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        weight_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        for op in embedding_ops:
            output_layout = op.infer_layout((input_layout, weight_layout))
            impl = op.get_expand_impl(None, output_layout, (input_layout, weight_layout), None)
            
            with self.assertRaisesRegex(ValueError, "Row-Parallel.*does not support `scale_grad_by_freq=True`"):
                impl(MagicMock(), MagicMock(), scale_grad_by_freq=True)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_embedding_fully_sharded_weights(self, mock_platform):
        """
        Feature: Fully sharded weights (Vocab and Embed dimensions).
        Description: Weight sharded on dim 0 (Vocab) and dim 1 (Embed) across different mesh axes.
        Expectation: Output inherits embed sharding and has Partial Sum state from vocab sharding.
        """
        mesh = self._make_2x4_mesh(mock_platform) # (dp:2, mp:4)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        # Weight [vocab, embed]: Vocab sharded on dp (axis 0), Embed sharded on mp (axis 1)
        # tensor_map: (1, 0)
        weight_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        for op in embedding_ops:
            output_layout = op.infer_layout((input_layout, weight_layout))
            # Input map (-1, -1) + weight_embed(0) = (-1, -1, 0)
            self.assertEqual(output_layout.tensor_map, (-1, -1, 0))
            self.assertTrue(output_layout.is_partial())
            # Partial should be on dp axis (index 0) due to vocab sharding
            self.assertEqual(output_layout.partial[0], "sum")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_embedding_3d_input_sharding(self, mock_platform):
        """
        Feature: 3D input tensor sharding.
        Description: Input shape [batch, seq, feat] sharded on batch, weight Column Parallel.
        Expectation: Output shape [batch, seq, feat, embed] preserves input sharding and appends embed sharding.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        # Input [B, S, F]: Sharded on B (axis 0) -> map (1, -1, -1)
        input_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        # Weight [V, E]: Sharded on E (axis 1) -> map (-1, 0)
        weight_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        for op in embedding_ops:
            output_layout = op.infer_layout((input_layout, weight_layout))
            # Expected map: (1, -1, -1, 0)
            self.assertEqual(output_layout.tensor_map, (1, -1, -1, 0))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_embedding_input_multi_dim_sharding(self, mock_platform):
        """
        Feature: Multi-sharded input.
        Description: Input sharded on both dim 0 and dim 1, weight replicated.
        Expectation: Output preserves all input sharding dimensions.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        # Input [B, S]: B sharded on dp (1), S sharded on mp (0) -> map (1, 0)
        input_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
        weight_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        for op in embedding_ops:
            output_layout = op.infer_layout((input_layout, weight_layout))
            # Expected map: (1, 0, -1)
            self.assertEqual(output_layout.tensor_map, (1, 0, -1))



    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_embedding_column_parallel_scale_grad_freq(self, mock_platform):
        """
        Feature: CP Parameter validation.
        Description: scale_grad_by_freq is allowed in Column Parallel.
        Expectation: No error raised (unlike Row Parallel).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        weight_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2) # Column Parallel

        for op in embedding_ops:
            output_layout = op.infer_layout((input_layout, weight_layout))
            impl = op.get_expand_impl(MagicMock(), output_layout, (input_layout, weight_layout), None)
            
            # Should not raise ValueError for scale_grad_by_freq
            try:
                impl(MagicMock(), MagicMock(), scale_grad_by_freq=True)
            except ValueError as e:
                if "does not support `scale_grad_by_freq`" in str(e):
                    self.fail("CP Embedding should support scale_grad_by_freq")

if __name__ == "__main__":
    unittest.main()