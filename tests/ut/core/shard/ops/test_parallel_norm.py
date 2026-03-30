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
"""parallel_norm test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_norm import NormDistributedOp, LayerNormDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = NormDistributedOp("RmsNorm")
torch_op = LayerNormDistributedOp("layer_norm")


class TestParallelNorm(unittest.TestCase):
    """Unit tests for NormDistributedOp and LayerNormDistributedOp."""
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

    def _make_1d_mesh(self, mock_platform, world_size=8, mesh_name="dp"):
        """Set up mock and return a standard 1D mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=world_size)
        return init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=(mesh_name,))

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_rmsnorm_layout_data_parallel(self, mock_platform):
        """
        Feature: RmsNorm data parallel
        Description: Data parallel scenario with no splitting on normalization axis
        Expectation: Success
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8, mesh_name="dp")

        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        gamma_layout = _build_layout(mesh, (Replicate(),), 1)
        beta_layout = gamma_layout

        input_layouts = (x_layout, gamma_layout, beta_layout)

        _, out_layout = op.infer_layout(input_layouts)

        expected_map = (0, -1)
        assert out_layout.tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map}, "
            f"got {out_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_rmsnorm_layout_tensor_parallel(self, mock_platform):
        """
        Feature: RmsNorm tensor parallel
        Description: Tensor parallel scenario with splitting on non-normalization axes
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)

        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)
        gamma_layout = _build_layout(mesh, (Replicate(),), 1)
        beta_layout = None

        input_layouts = (x_layout, gamma_layout, beta_layout)

        _, out_layout = op.infer_layout(input_layouts)

        expected_map = (1, 0, -1)
        assert out_layout.tensor_map == expected_map, (
            f"Tensor Parallel test failed. Expected {expected_map}, "
            f"got {out_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_rmsnorm_invalid_layout(self, mock_platform):
        """
        Feature: RmsNorm invalid layout
        Description: Test with invalid splitting on normalization axis
        Expectation: Raise ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)

        x_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        gamma_layout = _build_layout(mesh, (Replicate(), Shard(0)), 1)
        beta_layout = gamma_layout

        input_layouts = (x_layout, gamma_layout, beta_layout)

        with self.assertRaises(ValueError) as context:
            _, _ = op.infer_layout(input_layouts)

        self.assertIn("RmsNorm is disabled to support the splitting after", str(context.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_layernorm_layout_data_parallel(self, mock_platform):
        """
        Feature: LayerNorm data parallel
        Description: Data parallel scenario with no splitting on normalization axis
        Expectation: Success
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8, mesh_name="dp")
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        out_layout = torch_op.infer_layout((x_layout,), extra_args=((64,),))

        expected_map = (0, -1)
        assert out_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_layernorm_layout_tensor_parallel(self, mock_platform):
        """
        Feature: LayerNorm tensor parallel
        Description: Tensor parallel scenario with splitting on non-normalization axes
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)

        out_layout = torch_op.infer_layout((x_layout,), extra_args=((128,),))

        expected_map = (1, 0, -1)
        assert out_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_layernorm_layout_normalized_dim_sharded(self, mock_platform):
        """
        Feature: Test error when normalized dimension is sharded
        Description: Test error when normalized dimension is sharded
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        with self.assertRaisesRegex(ValueError, "Cannot perform sharding on normalized dimension 1"):
            torch_op.infer_layout((x_layout,), extra_args=((64,),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_layernorm_layout_normalized_shape_too_large(self, mock_platform):
        """
        Feature: Test normalized_shape larger than input ndim
        Description: Test normalized_shape larger than input ndim
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
        with self.assertRaisesRegex(ValueError, "larger than input ndim"):
            torch_op.infer_layout((x_layout,), extra_args=((128, 64, 56),))


if __name__ == "__main__":
    unittest.main()
