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
"""parallel_softmax test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_activation_with_axis import ActivationWithAxisDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = ActivationWithAxisDistributedOp("Softmax")
torch_op = ActivationWithAxisDistributedOp("softmax")


class TestParallelSoftmax(unittest.TestCase):
    """Unit tests for SoftmaxDistributedOp."""
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

    def _make_2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "mp")):
        """Set up mock and return a standard 2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=4)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=mesh_dim_names)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_softmax_layout_data_parallel(self, mock_platform):
        """
        Feature: Softmax data parallel
        Description: Data parallel scenario
        Expectation: Success
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8, mesh_name="dp")

        x_placements = (Shard(0),)
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), (-1,))
        expected_map = (0, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Data Parallel with transpose_a test failed. Expected {expected_map},"
            f" got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_softmax_layout_data_parallel_value_failed(self, mock_platform):
        """
        Feature: Softmax data parallel value failed
        Description: Data parallel scenario
        Expectation: Success
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8, mesh_name="dp")

        x_placements = (Shard(0),)
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaises(ValueError):
            _ = op.infer_layout((x_layout,), (0,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_softmax_data_parallel_success(self, mock_platform):
        """
        Feature: Softmax compatible with PyTorch (dim arg)
        Description: Standard Data Parallel scenario (Batch is sharded, Softmax on Feature dim)
        Expectation: Success, output layout equals input layout
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8, mesh_name="dp")

        x_placements = (Shard(0),)
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = torch_op.infer_layout((x_layout,), (1,))

        expected_map = x_layout.tensor_map
        assert output_layout.tensor_map == expected_map, (
            f"DP test failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_softmax_negative_dim_success(self, mock_platform):
        """
        Feature: Softmax compatible with PyTorch
        Description: Test negative index support (dim=-1) in Data Parallel
        Expectation: Success
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8, mesh_name="dp")

        x_placements = (Shard(0),)
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = torch_op.infer_layout((x_layout,), (-1,))

        expected_map = x_layout.tensor_map
        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_softmax_sharded_dim_failure(self, mock_platform):
        """
        Feature: Softmax distributed check
        Description: Attempting to compute Softmax on a sharded dimension
        Expectation: Raise ValueError
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8, mesh_name="dp")

        x_placements = (Shard(0),)
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaises(ValueError) as context:
            _ = torch_op.infer_layout((x_layout,), (0,))

        self.assertIn("is sharded", str(context.exception))
        self.assertIn("requires the reduction axis to be un-sharded", str(context.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_softmax_model_parallel_failure(self, mock_platform):
        """
        Feature: Softmax distributed check
        Description: Model Parallel scenario where the feature dimension is sharded
        Expectation: Raise ValueError when computing softmax on the MP axis
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8, mesh_name="mp")

        x_placements = (Shard(1),)
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaises(ValueError) as context:
            _ = torch_op.infer_layout((x_layout,), (-1,))

        self.assertIn("is sharded", str(context.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_softmax_model_parallel_success_on_other_axis(self, mock_platform):
        """
        Feature: Softmax distributed check
        Description: Model Parallel scenario, but computing softmax on the non-sharded axis
        Expectation: Success
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8, mesh_name="mp")

        x_placements = (Shard(1),)
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = torch_op.infer_layout((x_layout,), (0,))

        expected_map = x_layout.tensor_map
        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_softmax_input_consistency_failure(self, mock_platform):
        """
        Feature: Layout consistency check
        Description: Inputs have different layouts
        Expectation: Raise ValueError
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8, mesh_name="dp")

        layout1 = _build_layout(mesh, (Shard(0),), 2)
        layout2 = _build_layout(mesh, (Shard(1),), 2)

        with self.assertRaises(ValueError) as context:
            _ = torch_op.infer_layout((layout1, layout2), (1,))

        self.assertIn("requires all tensor inputs to have the same layout", str(context.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_softmax_multi_axis_mesh_success(self, mock_platform):
        """
        Feature: Softmax with multi-axis device mesh
        Description: 2D mesh (dp+mp), Softmax applied to unsharded dimension
        Expectation: Success, output layout matches input
        """
        mesh = self._make_2x4_mesh(mock_platform)

        x_placements = (Shard(0), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        output_layout = torch_op.infer_layout((x_layout,), (1,))

        expected_map = x_layout.tensor_map
        assert output_layout.tensor_map == expected_map, (
            f"Multi-axis mesh test failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_softmax_interleaved_parallel_success(self, mock_platform):
        """
        Feature: Softmax with interleaved parallel
        Description: Softmax applied to unsharded dimension under interleaved parallel layout
        Expectation: Success
        """
        mesh = self._make_2x2_mesh(mock_platform, mesh_dim_names=("dp", "interleaved_parallel"))

        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = torch_op.infer_layout((x_layout,), (1,))

        assert output_layout.tensor_map == x_layout.tensor_map
        assert output_layout.to_dict()["interleaved_parallel"] is True

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_softmax_non_contiguous_rank_list(self, mock_platform):
        """
        Feature: Softmax with non-contiguous rank list
        Description: Non-contiguous rank_list (simulating heterogeneous cluster), verify layout compatibility
        Expectation: Success if dim is unsharded
        """
        self._setup_mock_platform(mock_platform, world_size=4)
        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2),
            mesh_dim_names=("dp", "mp"),
            rank_list=(3, 1, 0, 2)
        )

        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = torch_op.infer_layout((x_layout,), (0,))

        assert output_layout.rank_list == (3, 1, 0, 2)
        assert output_layout.tensor_map == x_layout.tensor_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_softmax_multi_input_same_layout_success(self, mock_platform):
        """
        Feature: Softmax with multiple inputs (same layout)
        Description: Multiple inputs with consistent layout, verify validity
        Expectation: Success
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8, mesh_name="dp")

        layout1 = _build_layout(mesh, (Shard(0),), 2)
        layout2 = _build_layout(mesh, (Shard(0),), 2)

        output_layout = torch_op.infer_layout((layout1, layout2), (1,))

        assert output_layout.tensor_map == layout1.tensor_map


if __name__ == "__main__":
    unittest.main()
