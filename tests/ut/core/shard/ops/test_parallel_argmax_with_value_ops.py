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
"""parallel_argmax_with_value_ops test"""
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_argmax_with_value_ops import ArgMaxWithValueDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = ArgMaxWithValueDistributedOp("ArgMaxWithValue")


class TestParallelArgMaxWithValue(unittest.TestCase):
    """Unit tests for ArgMaxWithValueDistributedOp."""

    def setUp(self) -> None:
        """Clear global caches before each test to ensure isolation."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
        """Restore global cache state after each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

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
        """Set up mock and return a standard 2×4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"))

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2×2×2 (dp, cp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "cp", "mp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argmax_with_value_data_parallel_success(self, mock_platform):
        """
        Feature: ArgMaxWithValue data parallel
        Description: Data parallel scenario with argmax on unsharded axis
        Expectation: Success, output layout correctly reduced
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, 1, True]
        output_layouts, extra_info = op.infer_layout(cache_values)

        values_layout, indices_layout = output_layouts
        expected_map = (1, -1, -1)
        assert values_layout.tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert indices_layout.tensor_map == expected_map, (
            f"Data Parallel indices layout failed. Expected {expected_map}, "
            f"got {indices_layout.tensor_map}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test cases, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should return None, "
            f"got {op.get_expand_impl(None, (output_layouts, None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argmax_with_value_model_parallel_success(self, mock_platform):
        """
        Feature: ArgMaxWithValue model parallel
        Description: Model parallel scenario with argmax on unsharded batch dimension
        Expectation: Success, output layout correctly reduced
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, 0, True]
        output_layouts, extra_info = op.infer_layout(cache_values)

        values_layout, _ = output_layouts
        expected_map = (-1, 0, -1)
        assert values_layout.tensor_map == expected_map, (
            f"Model Parallel test failed. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argmax_with_value_hybrid_parallel_success(self, mock_platform):
        """
        Feature: ArgMaxWithValue hybrid parallel
        Description: Hybrid parallel scenario with argmax on unsharded middle dimension
        Expectation: Success, output layout correctly reduced
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, 1, True]
        output_layouts, extra_info = op.infer_layout(cache_values)

        values_layout, _ = output_layouts
        expected_map = (2, -1, 0)
        assert values_layout.tensor_map == expected_map, (
            f"Hybrid Parallel test failed. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argmax_with_value_all_replicated(self, mock_platform):
        """
        Feature: ArgMaxWithValue all replicated
        Description: All dimensions replicated scenario
        Expectation: Success, output layout correctly reduced
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, 0, True]
        output_layouts, extra_info = op.infer_layout(cache_values)

        values_layout, _ = output_layouts
        expected_map = (-1, -1, -1)
        assert values_layout.tensor_map == expected_map, (
            f"All Replicated test failed. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argmax_with_value_negative_dim(self, mock_platform):
        """
        Feature: ArgMaxWithValue negative dimension index
        Description: Test negative dimension index (dim=-1)
        Expectation: Success, output layout correctly reduced
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, -1, True]
        output_layouts, extra_info = op.infer_layout(cache_values)

        values_layout, _ = output_layouts
        expected_map = (1, -1, -1)
        assert values_layout.tensor_map == expected_map, (
            f"Negative dim test failed. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argmax_with_value_keep_dims_false(self, mock_platform):
        """
        Feature: ArgMaxWithValue with keep_dims=False
        Description: Test with keep_dims=False, reduced dimension removed
        Expectation: Success, output layout has reduced dimension removed
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, 1, False]
        output_layouts, extra_info = op.infer_layout(cache_values)

        values_layout, _ = output_layouts
        expected_map = (1, -1)
        assert values_layout.tensor_map == expected_map, (
            f"Keep dims False test failed. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argmax_with_value_sharded_dim_failure(self, mock_platform):
        """
        Feature: ArgMaxWithValue sharded dimension check
        Description: Attempting to compute argmax on a sharded dimension
        Expectation: Raise ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, 0, True]
        with self.assertRaisesRegex(ValueError, "cannot perform sharding on axis dim"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argmax_with_value_model_parallel_on_mp_axis_failure(self, mock_platform):
        """
        Feature: ArgMaxWithValue model parallel check
        Description: Model Parallel scenario where the feature dimension is sharded
        Expectation: Raise ValueError when computing argmax on the MP axis
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, 1, True]
        with self.assertRaisesRegex(ValueError, "cannot perform sharding on axis dim"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argmax_with_value_3d_tensor(self, mock_platform):
        """
        Feature: ArgMaxWithValue on 3D tensor
        Description: Test argmax on 3D tensor with mixed placements
        Expectation: Success
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, 2, True]
        output_layouts, extra_info = op.infer_layout(cache_values)

        values_layout, _ = output_layouts
        expected_map = (2, 1, -1)
        assert values_layout.tensor_map == expected_map, (
            f"3D tensor test failed. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argmax_with_value_partial_input(self, mock_platform):
        """
        Feature: ArgMaxWithValue with partial input
        Description: Input with partial state
        Expectation: Raise ValueError since _allow_partial_inputs is False
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)
        x_layout.set_partial_by_dev_axis("dp", "sum")

        cache_values = [x_layout, 1, True]
        with self.assertRaisesRegex(ValueError, "has Partial status which is not allowed"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argmax_with_value_axis_out_of_range(self, mock_platform):
        """
        Feature: ArgMaxWithValue axis out of range
        Description: Pass an axis value outside the valid range for the input rank
        Expectation: Raise ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, 5, True]
        with self.assertRaisesRegex(ValueError, "axis out of range"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argmax_with_value_axis_not_int(self, mock_platform):
        """
        Feature: ArgMaxWithValue axis type validation
        Description: Pass a non-int axis value
        Expectation: Raise ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, 1.0, True]
        with self.assertRaisesRegex(ValueError, "axis should be int"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argmax_with_value_preprocess(self, mock_platform):
        """
        Feature: ArgMaxWithValueDistributedOp preprocess
        Description: Verify preprocess routes all args as positional for MindSpore Primitive
        Expectation: local_kwargs is empty; local_args has 3 elements; cache_values has 3 elements
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = MagicMock()

        local_args, local_kwargs, cache_values = op.preprocess((mock_tensor, 1, True), {})

        assert not local_kwargs, (
            f"For ArgMaxWithValue, local_kwargs should be empty, got {local_kwargs}"
        )
        assert len(local_args) == 3, (
            f"For ArgMaxWithValue, local_args should have 3 elements "
            f"(tensor, axis, keep_dims), got {len(local_args)}"
        )
        assert local_args[1] == 1, (
            f"axis should be 1, got {local_args[1]}"
        )
        assert local_args[2] is True, (
            f"keep_dims should be True, got {local_args[2]}"
        )
        assert len(cache_values) == 3, (
            f"cache_values should have 3 elements, got {len(cache_values)}"
        )
        assert cache_values[1] == 1, (
            f"cache_values[1] (axis) should be 1, got {cache_values[1]}"
        )
        assert cache_values[2] is True, (
            f"cache_values[2] (keep_dims) should be True, got {cache_values[2]}"
        )


if __name__ == "__main__":
    unittest.main()
    