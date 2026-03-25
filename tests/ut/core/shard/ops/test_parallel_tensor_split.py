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
"""parallel_tensor_split test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_split import TensorSplitDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = TensorSplitDistributedOp("tensor_split")


class MockTensor:
    """Mock class to simulate a 1D Tensor for testing indices_or_sections."""
    def __init__(self, shape):
        self.shape = shape


class TestParallelTensorSplit(unittest.TestCase):
    """Unit tests for TensorSplitDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()

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

        output_layouts = op.infer_layout((x_layout,), extra_args=(3,))

        assert len(output_layouts) == 3, f"Expected 3 output layouts, got {len(output_layouts)}"
        for out_layout in output_layouts:
            assert out_layout.to_dict()["tensor_map"] == x_layout.to_dict()["tensor_map"], (
                "Output layout should be identical to the input layout."
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

        output_layouts = op.infer_layout((x_layout,), extra_args=([1, 3], 1))

        assert len(output_layouts) == 3, f"Expected 3 output layouts, got {len(output_layouts)}"

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

        output_layouts = op.infer_layout((x_layout,), extra_args=(2, -1))

        assert len(output_layouts) == 2, f"Expected 2 output layouts, got {len(output_layouts)}"

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

        mock_tensor = MockTensor(shape=(4,))
        output_layouts = op.infer_layout((x_layout,), extra_args=(mock_tensor, 0))

        assert len(output_layouts) == 5, f"Expected 5 output layouts, got {len(output_layouts)}"

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

        with self.assertRaisesRegex(ValueError, r"Cannot perform tensor_split on sharded axis\[0\]"):
            op.infer_layout((x_layout,), extra_args=(2, 0))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tensor_split_invalid_type_error(self, mock_platform):
        """
        Feature: Error handling for invalid indices_or_sections type
        Description: Pass a string type as indices_or_sections.
        Expectation: TypeError is raised.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        error_msg = "tensor_split: indices_or_sections must be an integer, list, tuple, or 1D tensor."
        with self.assertRaisesRegex(TypeError, error_msg):
            op.infer_layout((x_layout,), extra_args=("invalid_string", 0))

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

        with self.assertRaisesRegex(ValueError, r"Dimension out of range \(expected \[0, 2\), got 2\)."):
            op.infer_layout((x_layout,), extra_args=(2, 2))


if __name__ == "__main__":
    unittest.main()
