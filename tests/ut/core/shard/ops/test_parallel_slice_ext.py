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
"""parallel_slice_ext test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_slice_ext import SliceExtDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

slice_op = SliceExtDistributedOp("SliceExt")


class TestParallelSliceExt(unittest.TestCase):
    """Unit tests for SliceExtDistributedOp."""
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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, mp, cp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "mp", "cp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_normal(self, mock_platform):
        """
        Feature: Slice operator layout inference under normal conditions
        Description: Test normal split where axis is not sharded
        Expectation: Output layouts are correctly generated with same tensor_map
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Shard(0), Shard(2)), 3)
        axis = 1
        extra_args = [axis, 0, 2, 1]
        output_layout = slice_op.infer_layout([input_layout], extra_args)
        assert output_layout == input_layout

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_invalid_axis(self, mock_platform):
        """
        Feature: Slice operator layout inference with invalid axis
        Description: Test when trying to split a sharded axis (which is not allowed)
        Expectation: ValueError is raised
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Shard(0), Shard(2)), 3)
        axis = 0
        extra_args = [axis, 0, 2, 1]
        with self.assertRaises(ValueError):
            slice_op.infer_layout([input_layout], extra_args)


if __name__ == "__main__":
    unittest.main()
