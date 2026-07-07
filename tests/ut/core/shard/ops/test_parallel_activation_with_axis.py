# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
from unittest.mock import MagicMock, patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_activation_with_axis import ActivationWithAxisDistributedOp
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
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _infer_single_layout(self, op, cache_values):
        """Infer and unpack a single output layout with the new cache_values API."""
        output_layouts, _ = op.infer_layout(cache_values)
        return output_layouts[0]

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

        cache_values = [x_layout, 1]
        output_layout = self._infer_single_layout(op, cache_values)

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, ((output_layout,), None), cache_values) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, ((output_layout,), None), cache_values)}"
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

        cache_values = [x_layout, 0]
        output_layout = self._infer_single_layout(op, cache_values)

        expected_map = (-1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Model Parallel test failed. Expected {expected_map}, "
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

        cache_values = [x_layout, 0]
        output_layout = self._infer_single_layout(op, cache_values)

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

        cache_values = [x_layout, -1]
        output_layout = self._infer_single_layout(op, cache_values)

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
            op.infer_layout([x_layout, 0])

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
            op.infer_layout([x_layout, -1])

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

        cache_values = [x_layout, (0, 2)]
        output_layout = self._infer_single_layout(op, cache_values)

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
            op.infer_layout([x_layout, (0, 2)])

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
            op.infer_layout([layout1, layout2, 1])

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

        cache_values = [layout1, layout2, 1]
        output_layout = self._infer_single_layout(op, cache_values)

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

        cache_values = [x_layout, 2]
        output_layout = self._infer_single_layout(op, cache_values)

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
            op.infer_layout([x_layout, "invalid"])

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
            op.infer_layout([x_layout, 1])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_activation_with_axis_preprocess(self, mock_platform):
        """
        Feature: ActivationWithAxis preprocess
        Description: Convert DTensor input to local tensor and build cache_values
        Expectation: All runtime arguments are positional and cache contains layout and axis
        """
        op = ActivationWithAxisDistributedOp("Softmax")
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_local = MagicMock()
        mock_tensor.to_local.return_value = mock_local

        local_args, local_kwargs, cache_values = op.preprocess((mock_tensor,), {"dim": 1})

        assert local_args == (mock_local, 1)
        assert not local_kwargs
        assert cache_values == [x_layout, 1]

softmax_ms_op = ActivationWithAxisDistributedOp("Softmax")
softmax_torch_op = ActivationWithAxisDistributedOp("softmax")


class TestParallelSoftmax(unittest.TestCase):
    """Unit tests for SoftmaxDistributedOp (via ActivationWithAxisDistributedOp)."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _infer_single_layout(self, op, cache_values):
        """Infer and unpack a single output layout with the new cache_values API."""
        output_layouts, _ = op.infer_layout(cache_values)
        return output_layouts[0]

    def _setup_mock_platform(self, mock_platform, platform_type=None, world_size=8):
        """Configure common mock-platform attributes used across tests."""
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

        cache_values = [x_layout, -1]
        output_layout = self._infer_single_layout(softmax_ms_op, cache_values)
        expected_map = (0, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map},"
            f" got {output_layout.tensor_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert softmax_ms_op.get_expand_impl(None, ((output_layout,), None), cache_values) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {softmax_ms_op.get_expand_impl(None, ((output_layout,), None), cache_values)}"
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
            _ = softmax_ms_op.infer_layout([x_layout, 0])

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

        cache_values = [x_layout, 1]
        output_layout = self._infer_single_layout(softmax_torch_op, cache_values)

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

        cache_values = [x_layout, -1]
        output_layout = self._infer_single_layout(softmax_torch_op, cache_values)

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
            _ = softmax_torch_op.infer_layout([x_layout, 0])

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
            _ = softmax_torch_op.infer_layout([x_layout, -1])

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

        cache_values = [x_layout, 0]
        output_layout = self._infer_single_layout(softmax_torch_op, cache_values)

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
            _ = softmax_torch_op.infer_layout([layout1, layout2, 1])

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

        cache_values = [x_layout, 1]
        output_layout = self._infer_single_layout(softmax_torch_op, cache_values)

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

        cache_values = [x_layout, 1]
        output_layout = self._infer_single_layout(softmax_torch_op, cache_values)

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

        cache_values = [x_layout, 0]
        output_layout = self._infer_single_layout(softmax_torch_op, cache_values)

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

        cache_values = [layout1, layout2, 1]
        output_layout = self._infer_single_layout(softmax_torch_op, cache_values)

        assert output_layout.tensor_map == layout1.tensor_map


if __name__ == "__main__":
    unittest.main()
