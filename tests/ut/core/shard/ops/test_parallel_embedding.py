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
"""parallel_embedding test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_embedding import EmbeddingDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType

op = EmbeddingDistributedOp("Embedding")
torch_op = EmbeddingDistributedOp("embedding")


class TestParallelEmbedding(unittest.TestCase):
    """Unit tests for EmbeddingDistributedOp."""
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

    def _make_2x4_mesh(self, mock_platform, mesh_dim_names=("a", "b")):
        """Set up mock and return a standard 2x4 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=mesh_dim_names)

    def _make_1x8_mesh(self, mock_platform, mesh_dim_names=("a", "b")):
        """Set up mock and return a standard 1x8 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(1, 8), mesh_dim_names=mesh_dim_names)

    def _make_1d_mesh(self, mock_platform, world_size=8, mesh_name="dp"):
        """Set up mock and return a standard 1D mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=world_size)
        return init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=(mesh_name,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_embedding_layout_parallel_1(self, mock_platform):
        """
        Feature: Embedding parallel
        Description: Parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        w_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        output_layout = op.infer_layout(
            (
                x_layout,
                w_layout,
                None,
                None,
                None,
                None,
            ),
            (
                None,
                None,
            ),
        )
        expected_map = (-1, -1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Test failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_embedding_layout_parallel_2(self, mock_platform):
        """
        Feature: Embedding parallel
        Description: Parallel scenario
        Expectation: Success
        """
        mesh = self._make_1x8_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        output_layout = op.infer_layout(
            (
                x_layout,
                w_layout,
                None,
                None,
                None,
                None,
            ),
            (
                None,
                None,
            ),
        )
        expected_map = (-1, -1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Test failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_embedding_layout_dp_mp(self, mock_platform):
        """
        Feature: PyTorch Embedding parallel
        Description: Hybrid Parallel (Data Parallel on Input + Model Parallel on Weight Column)
        Logic:
            Input [Batch, Seq] -> Layout ("dp", "None") -> Map (1, -1)
            Weight [Vocab, Embed] -> Layout ("None", "mp") -> Map (-1, 0)
            Output [Batch, Seq, Embed] -> Should combine input map and weight last dim map
            Expected Map -> (1, -1, 0) corresponding to ("dp", "None", "mp")
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        output_layout = torch_op.infer_layout(
            [x_layout, w_layout],
            None
        )

        expected_map = (1, -1, 0)

        assert output_layout.tensor_map == expected_map, (
            f"Test failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_embedding_layout_pure_dp(self, mock_platform):
        """
        Feature: PyTorch Embedding parallel
        Description: Pure Data Parallel
        Logic:
            Input [Batch, Seq] -> Layout ("dp", "None") -> Map (0, -1) [Assuming 1D mesh]
            Weight [Vocab, Embed] -> Layout ("None", "None") -> Map (-1, -1)
            Output [Batch, Seq, Embed] -> Expected ("dp", "None", "None")
        Expectation: Success
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8, mesh_name="dp")
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        output_layout = torch_op.infer_layout(
            [x_layout, w_layout],
            None
        )

        expected_map = (0, -1, -1)

        assert output_layout.tensor_map == expected_map, (
            f"Test failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_embedding_layout_row_parallel(self, mock_platform):
        """
        Feature: PyTorch Embedding parallel
        Description: Row Parallel (Weight sharded on Vocab dim)
        Logic:
            Input [Batch, Seq] -> ("dp", "None")
            Weight [Vocab, Embed] -> ("mp", "None")

            According to the implementation logic provided:
            Output Indices = x_map + (w_map[-1],)
            x_map = ("dp", "None") -> (1, -1)
            w_map last dim = "None" -> -1
            Result -> (1, -1, -1) -> ("dp", "None", "None")

            Note: Real Row Parallel usually implies AllReduce or Partial output,
            but the provided implementation logic currently calculates pure sharding derivation.
        Expectation: Success based on current implementation logic
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)

        output_layout = torch_op.infer_layout(
            [x_layout, w_layout],
            None
        )

        expected_map = (1, -1, -1)

        assert output_layout.tensor_map == expected_map, (
            f"Test failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_embedding_inputs_check(self, mock_platform):
        """
        Feature: PyTorch Embedding parallel
        Description: Exception handling for insufficient inputs
        Expectation: ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "Embedding requires at least 2 layouts"):
            torch_op.infer_layout([x_layout], None)


if __name__ == "__main__":
    unittest.main()
