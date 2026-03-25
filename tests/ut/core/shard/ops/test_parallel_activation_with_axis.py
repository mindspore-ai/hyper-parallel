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
"""parallel_activation_with_axis test"""
import os
import unittest
from unittest.mock import patch
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


class TestParallelActivationWithAxis(unittest.TestCase):
    """Unit tests for ActivationWithAxisDistributedOp."""
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

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "mp"),
            init_backend=False
        )

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, cp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "cp", "mp"),
            init_backend=False
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_activation_with_axis_data_parallel_success(self, mock_platform):
        """
        Feature: ActivationWithAxis data parallel
        Description: Data parallel scenario with softmax on unsharded axis
        Expectation: Success, output layout equals input layout
        """
        op = ActivationWithAxisDistributedOp("Softmax")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), (1,))

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_activation_with_axis_model_parallel_success(self, mock_platform):
        """
        Feature: ActivationWithAxis model parallel
        Description: Model parallel scenario with softmax on unsharded batch dimension
        Expectation: Success, output layout equals input layout
        """
        op = ActivationWithAxisDistributedOp("Softmax")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), (0,))

        expected_map = (-1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Model Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_activation_with_axis_hybrid_parallel_success(self, mock_platform):
        """
        Feature: ActivationWithAxis hybrid parallel
        Description: Hybrid parallel scenario with softmax on unsharded middle dimension
        Expectation: Success, output layout equals input layout
        """
        op = ActivationWithAxisDistributedOp("Swiglu")
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        output_layout = op.infer_layout((x_layout,), (1,))

        expected_map = (2, -1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Hybrid Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_activation_with_axis_all_replicated(self, mock_platform):
        """
        Feature: ActivationWithAxis all replicated
        Description: All dimensions replicated scenario
        Expectation: Success, output layout equals input layout
        """
        op = ActivationWithAxisDistributedOp("softmax")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), (0,))

        expected_map = (-1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"All Replicated test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_activation_with_axis_negative_dim(self, mock_platform):
        """
        Feature: ActivationWithAxis negative dimension index
        Description: Test negative dimension index (dim=-1)
        Expectation: Success, output layout equals input layout
        """
        op = ActivationWithAxisDistributedOp("Softmax")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), (-1,))

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Negative dim test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_activation_with_axis_sharded_dim_failure(self, mock_platform):
        """
        Feature: ActivationWithAxis sharded dimension check
        Description: Attempting to compute activation on a sharded dimension
        Expectation: Raise ValueError
        """
        op = ActivationWithAxisDistributedOp("Softmax")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "requires the reduction axis to be un-sharded"):
            op.infer_layout((x_layout,), (0,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_activation_with_axis_model_parallel_on_mp_axis_failure(self, mock_platform):
        """
        Feature: ActivationWithAxis model parallel check
        Description: Model Parallel scenario where the feature dimension is sharded
        Expectation: Raise ValueError when computing activation on the MP axis
        """
        op = ActivationWithAxisDistributedOp("Softmax")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "requires the reduction axis to be un-sharded"):
            op.infer_layout((x_layout,), (-1,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_activation_with_axis_multi_axis_tuple(self, mock_platform):
        """
        Feature: ActivationWithAxis with multiple axes
        Description: Test activation with tuple of axes
        Expectation: Success if all axes are unsharded
        """
        op = ActivationWithAxisDistributedOp("Softmax")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        output_layout = op.infer_layout((x_layout,), (0, 2))

        expected_map = (-1, -1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Multi axis tuple test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_activation_with_axis_multi_axis_sharded_failure(self, mock_platform):
        """
        Feature: ActivationWithAxis with multiple axes, one sharded
        Description: Test activation with tuple of axes where one is sharded
        Expectation: Raise ValueError
        """
        op = ActivationWithAxisDistributedOp("Softmax")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        with self.assertRaisesRegex(ValueError, "requires the reduction axis to be un-sharded"):
            op.infer_layout((x_layout,), (0, 2))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_activation_with_axis_input_consistency_failure(self, mock_platform):
        """
        Feature: ActivationWithAxis layout consistency check
        Description: Inputs have different layouts
        Expectation: Raise ValueError
        """
        op = ActivationWithAxisDistributedOp("Softmax")
        mesh = self._make_2x4_mesh(mock_platform)
        layout1 = _build_layout(mesh, (Shard(0), Replicate()), 2)
        layout2 = _build_layout(mesh, (Replicate(), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "requires all tensor inputs to have the same layout"):
            op.infer_layout((layout1, layout2), (1,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_activation_with_axis_multi_input_same_layout(self, mock_platform):
        """
        Feature: ActivationWithAxis with multiple inputs (same layout)
        Description: Multiple inputs with consistent layout
        Expectation: Success
        """
        op = ActivationWithAxisDistributedOp("Softmax")
        mesh = self._make_2x4_mesh(mock_platform)
        layout1 = _build_layout(mesh, (Shard(0), Replicate()), 2)
        layout2 = _build_layout(mesh, (Shard(0), Replicate()), 2)

        output_layout = op.infer_layout((layout1, layout2), (1,))

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Multi input same layout test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_activation_with_axis_3d_tensor(self, mock_platform):
        """
        Feature: ActivationWithAxis on 3D tensor
        Description: Test activation on 3D tensor with mixed placements
        Expectation: Success
        """
        op = ActivationWithAxisDistributedOp("Softmax")
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        output_layout = op.infer_layout((x_layout,), (2,))

        expected_map = (2, 1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"3D tensor test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_activation_with_axis_invalid_extra_args_type(self, mock_platform):
        """
        Feature: ActivationWithAxis invalid extra args type
        Description: Pass invalid type as extra args
        Expectation: Raise ValueError
        """
        op = ActivationWithAxisDistributedOp("Softmax")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "should be int or tuple"):
            op.infer_layout((x_layout,), ("invalid",))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_activation_with_axis_partial_input(self, mock_platform):
        """
        Feature: ActivationWithAxis with partial input
        Description: Input with partial state
        Expectation: Raise ValueError since _allow_partial_inputs is False
        """
        op = ActivationWithAxisDistributedOp("Softmax")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)
        x_layout.set_partial_by_dev_axis("dp", "sum")

        with self.assertRaisesRegex(ValueError, "has Partial status which is not allowed"):
            op.infer_layout((x_layout,), (1,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_activation_with_axis_get_expand_impl(self, mock_platform):
        """
        Feature: ActivationWithAxis get_expand_impl
        Description: Verify get_expand_impl returns None
        Expectation: Returns None
        """
        op = ActivationWithAxisDistributedOp("Softmax")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), (1,))

        assert op.get_expand_impl(None, output_layout, (x_layout,), (1,)) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, output_layout, (x_layout,), (1,))}"
        )


if __name__ == "__main__":
    unittest.main()
