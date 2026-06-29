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
"""parallel_split test"""
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_split import (
    SplitDistributedOp,
    SplitWithSizeDistributedOp,
    SplitTensorDistributedOp,
    TensorSplitDistributedOp,
)
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelSplit(unittest.TestCase):
    """Unit tests for SplitDistributedOp, SplitWithSizeDistributedOp, and SplitTensorDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()
        self.split_op = SplitDistributedOp("split")
        self.torch_split_op = SplitDistributedOp("split")
        self.split_with_size_op = SplitWithSizeDistributedOp("split_with_size")
        self.split_tensor_op = SplitTensorDistributedOp("split_tensor")

    def tearDown(self):
        """Clean up after each test method."""
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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, mp, cp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "mp", "cp"),
            init_backend=False
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_normal(self, mock_platform):
        """
        Feature: Split operator layout inference under normal conditions
        Description: Test normal split where axis is not sharded
        Expectation: Output layouts are correctly generated with same tensor_map
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Shard(0), Shard(2)), 3)
        axis = 1
        # split_size=2, input_shape=(12, 16, 20), output_num = ceil(16/2) = 8
        cache_values = [input_layout, axis, 8]

        infer_result = self.split_op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        assert len(output_layouts) == 8, (
            f"Expected 8 output layouts, got {len(output_layouts)}"
        )
        assert all(layout.tensor_map == input_layout.tensor_map for layout in output_layouts), (
            f"Output tensor_map should match "
            f"input_tensor_map={input_layout.tensor_map}, "
            f"got {[l.tensor_map for l in output_layouts]}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert self.split_op.get_expand_impl(None, infer_result, cache_values) is None, (
            f"get_expand_impl should return None, "
            f"got {self.split_op.get_expand_impl(None, infer_result, cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_invalid_axis(self, mock_platform):
        """
        Feature: Split operator layout inference with invalid axis
        Description: Test when trying to split a sharded axis (which is not allowed)
        Expectation: ValueError is raised
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Shard(0), Shard(1)), 3)
        axis = 0
        cache_values = [input_layout, axis, 8]

        with self.assertRaisesRegex(ValueError, "can not split tensor at sharded axis"):
            self.split_op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_axis_out_of_range(self, mock_platform):
        """
        Feature: Test axis out of range
        Description: Test axis out of range
        Expectation: ValueError is raised
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        axis = 5
        cache_values = [input_layout, axis, 2]

        with self.assertRaisesRegex(ValueError, "dimension should be in range"):
            self.split_op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_default_dim(self, mock_platform):
        """
        Feature: Split operator layout inference with default dim
        Description: When dim is not explicitly provided, it defaults to 0
        Expectation: Split works correctly on dim=0
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        axis = 0
        cache_values = [input_layout, axis, 6]

        infer_result = self.split_op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        assert len(output_layouts) == 6, (
            f"Expected 6 output layouts, got {len(output_layouts)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_with_size(self, mock_platform):
        """
        Feature: Split operator layout inference with sections list
        Description: Test split using a list of section sizes
        Expectation: Output number matches the length of sections list
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Shard(1), Replicate()), 3)
        axis = 2
        split_sections = [2, 3, 3]
        cache_values = [input_layout, axis, len(split_sections)]

        infer_result = self.split_with_size_op.infer_layout(cache_values)
        output_layouts = infer_result[0]

        assert len(output_layouts) == len(split_sections), (
            f"Expected {len(split_sections)} output layouts, got {len(output_layouts)}"
        )
        assert all(layout.tensor_map == input_layout.tensor_map for layout in output_layouts), (
            f"Output tensor_map should match "
            f"input_tensor_map={input_layout.tensor_map}, "
            f"got {[l.tensor_map for l in output_layouts]}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_with_size_invalid_axis(self, mock_platform):
        """
        Feature: Split operator layout inference with invalid axis
        Description: Test when trying to split a sharded axis (which is not allowed)
        Expectation: ValueError is raised
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Shard(1), Replicate()), 3)
        axis = 1
        cache_values = [input_layout, axis, 3]

        with self.assertRaisesRegex(ValueError, "can not split tensor at sharded axis"):
            self.split_with_size_op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_split_tensor_infer_layout_with_remainder(self, mock_platform):
        """
        Feature: Split operator layout inference with non-divisible size
        Description: Test split when input shape is not divisible by split size
        Expectation: Output count includes an extra tensor for the remainder
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(2)), 3)
        axis = 1
        # split_size=3, input_shape[1]=7, output_num = 7//3 + 1 = 3
        cache_values = [input_layout, axis, 3]

        infer_result = self.split_tensor_op.infer_layout(cache_values)
        output_layouts = infer_result[0]

        assert len(output_layouts) == 3, (
            f"Expected 3 output layouts, got {len(output_layouts)}"
        )
        assert all(layout.tensor_map == input_layout.tensor_map for layout in output_layouts), (
            f"Output tensor_map should match "
            f"input_tensor_map={input_layout.tensor_map}, "
            f"got {[l.tensor_map for l in output_layouts]}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_split_tensor_infer_layout_with_remainder_invalid_axis(self, mock_platform):
        """
        Feature: Split operator layout inference with invalid axis
        Description: Test when trying to split a sharded axis (which is not allowed)
        Expectation: ValueError is raised
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(2)), 3)
        axis = 2
        cache_values = [input_layout, axis, 3]

        with self.assertRaisesRegex(ValueError, "can not split tensor at sharded axis"):
            self.split_tensor_op.infer_layout(cache_values)




tensor_split_op = TensorSplitDistributedOp("tensor_split")


class TestParallelTensorSplit(unittest.TestCase):
    """Unit tests for TensorSplitDistributedOp."""

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
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"))

    def _make_2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=4)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "mp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tensor_split_integer_default_dim(self, mock_platform):
        """
        Feature: Basic tensor_split with integer sections
        Description: Split tensor into 3 sections with default dim=0.
        Expectation: Output contains 3 identical layouts, dim 0 must be unsharded.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, 0, 3]
        infer_result = tensor_split_op.infer_layout(cache_values)
        output_layouts = infer_result[0]

        assert len(output_layouts) == 3, (
            f"Expected 3 output layouts, got {len(output_layouts)}"
        )
        for out_layout in output_layouts:
            expected_tm = x_layout.to_dict()["tensor_map"]
            actual_tm = out_layout.to_dict()["tensor_map"]
            assert actual_tm == expected_tm, (
                f"Output layout should be identical to the input layout, "
                f"expected={expected_tm}, got={actual_tm}"
            )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert tensor_split_op.get_expand_impl(None, infer_result, cache_values) is None, (
            f"get_expand_impl should return None, "
            f"got {tensor_split_op.get_expand_impl(None, infer_result, cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tensor_split_tuple_specific_dim(self, mock_platform):
        """
        Feature: tensor_split with tuple/list indices and specific dim
        Description: Split tensor using list indices [1, 3] on dim=1.
        Expectation: Output contains len(indices) + 1 = 3 identical layouts.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, 1, 3]
        infer_result = tensor_split_op.infer_layout(cache_values)
        output_layouts = infer_result[0]

        assert len(output_layouts) == 3, (
            f"Expected 3 output layouts, got {len(output_layouts)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tensor_split_negative_dim(self, mock_platform):
        """
        Feature: tensor_split with negative dimension
        Description: Split tensor using negative dim (-1) which resolves to the last dimension.
        Expectation: Operator accurately calculates the true dimension and infers correctly.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, -1, 2]
        infer_result = tensor_split_op.infer_layout(cache_values)
        output_layouts = infer_result[0]

        assert len(output_layouts) == 2, (
            f"Expected 2 output layouts, got {len(output_layouts)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tensor_split_1d_tensor(self, mock_platform):
        """
        Feature: tensor_split with 1D tensor as indices
        Description: Split tensor using a 1D tensor (mocked) for indices_or_sections.
        Expectation: Output contains shape[0] + 1 identical layouts.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, 0, 5]
        infer_result = tensor_split_op.infer_layout(cache_values)
        output_layouts = infer_result[0]

        assert len(output_layouts) == 5, (
            f"Expected 5 output layouts, got {len(output_layouts)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tensor_split_sharded_dim_error(self, mock_platform):
        """
        Feature: Error handling for splitting on a sharded dimension
        Description: Attempt to split along an axis that is actively sharded across the device mesh.
        Expectation: ValueError is raised.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, 0, 2]
        with self.assertRaisesRegex(ValueError, r"can not split tensor at sharded axis\[0\]"):
            tensor_split_op.infer_layout(cache_values)

    def test_tensor_split_invalid_type_error(self):
        """
        Feature: Error handling for invalid indices_or_sections type
        Description: Pass a string type as indices_or_sections.
        Expectation: TypeError is raised from preprocess.
        """
        mock_input = MagicMock()
        mock_input.to_local.return_value = "local_tensor"
        mock_input.layout = MagicMock()

        error_msg = "indices_or_sections must be an integer"
        with self.assertRaisesRegex(TypeError, error_msg):
            tensor_split_op.preprocess((mock_input, "invalid_string"), {})

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tensor_split_out_of_bounds_dim(self, mock_platform):
        """
        Feature: Error handling for out of bounds dimension
        Description: Pass a dimension index that exceeds the tensor's rank.
        Expectation: ValueError is raised.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, 2, 2]
        with self.assertRaisesRegex(ValueError, r"dimension should be in range \[0, 2\), but got 2."):
            tensor_split_op.infer_layout(cache_values)


if __name__ == "__main__":
    unittest.main()
    