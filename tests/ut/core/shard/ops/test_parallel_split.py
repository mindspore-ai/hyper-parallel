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
"""parallel_split test"""
import os
import unittest
from unittest.mock import patch
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_split import (
    SplitDistributedOp,
    SplitWithSizeDistributedOp,
    SplitTensorDistributedOp
)
from hyper_parallel.platform import get_platform
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
        self.platform = get_platform()
        self.split_op = SplitDistributedOp("split")
        self.torch_split_op = SplitDistributedOp("split")
        self.split_with_size_op = SplitWithSizeDistributedOp("split_with_size")
        self.split_tensor_op = SplitTensorDistributedOp("split_tensor")

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
        split_size = 2
        input_shape = ((12, 16, 20), None)
        extra_args = [split_size, axis, input_shape]

        output_layouts = self.split_op.infer_layout([input_layout], extra_args)
        assert len(output_layouts) == 8
        assert all(layout.tensor_map == input_layout.tensor_map for layout in output_layouts)

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
        split_size = 2
        input_shape = ((12, 16, 20), None)
        extra_args = [split_size, axis, input_shape]

        with self.assertRaisesRegex(ValueError, "Can not split tensor at sharded axis"):
            self.split_op.infer_layout([input_layout], extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_axis_out_of_range(self, mock_platform):
        """
        Feature: Test axis out of range
        Description: Test axis out of range
        Expectation: Success
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        axis = 5
        split_size = 2
        input_shape = ((12, 16, 20), None)
        extra_args = [split_size, axis, input_shape]

        with self.assertRaisesRegex(ValueError, "Dimension out of range"):
            self.split_op.infer_layout([input_layout], extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_insufficient_extra_args(self, mock_platform):
        """
        Feature: Test missing extra_arg
        Description: Test missing extra_arg
        Expectation: Success
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        extra_args = [0]

        with self.assertRaisesRegex(ValueError, "Split ops extra_args requires 'axis' and contains 'output_num' optionally."):
            self.split_op.infer_layout([input_layout], extra_args)

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
        split_size_or_sections = [2, 3, 3]
        extra_args = [split_size_or_sections, axis]

        output_layouts = self.split_with_size_op.infer_layout([input_layout], extra_args)

        assert len(output_layouts) == len(split_size_or_sections)
        assert all(layout.tensor_map == input_layout.tensor_map for layout in output_layouts)

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
        split_size_or_sections = [2, 3, 3]
        extra_args = [split_size_or_sections, axis]

        with self.assertRaises(ValueError):
            self.split_with_size_op.infer_layout([input_layout], extra_args)

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
        split_size = 3
        input_shape = [5, 7, 9]
        extra_args = [split_size, axis, [input_shape, ]]

        output_layouts = self.split_tensor_op.infer_layout([input_layout], extra_args)

        expected_output_num = input_shape[axis] // split_size + 1
        assert len(output_layouts) == expected_output_num
        assert all(layout.tensor_map == input_layout.tensor_map for layout in output_layouts)

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
        split_size = 3
        input_shape = [5, 7, 9]
        extra_args = [split_size, axis, [input_shape, ]]

        with self.assertRaises(ValueError):
            self.split_tensor_op.infer_layout([input_layout], extra_args)


if __name__ == "__main__":
    unittest.main()
