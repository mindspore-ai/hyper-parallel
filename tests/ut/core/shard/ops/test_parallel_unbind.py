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
"""parallel_unbind test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_unbind import UnbindDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = UnbindDistributedOp("unbind")


class TestParallelUnbind(unittest.TestCase):
    """Unit tests for UnbindDistributedOp."""
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

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"))

    def _make_2x2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "tp", "mp")):
        """Set up mock and return a standard 2x2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=mesh_dim_names)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_unbind_layout_inference_dim0(self, mock_platform):
        """
        Feature: Unbind on unsharded dimension 0
        Description: Unbind input (3, 8) on dim 0. Dim 0 is unsharded, Dim 1 is sharded.
        Expectation: Return tuple of layouts, size equals dim 0 size. Remaining dimension preserves sharding.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)
        input_shape = (3, 8)

        output_layouts = op.infer_layout((x_layout,), extra_args=(0, [input_shape]))

        assert isinstance(output_layouts, tuple)
        assert len(output_layouts) == 3

        expected_map = (0,)
        for layout in output_layouts:
            assert layout.to_dict()["tensor_map"] == expected_map, (
                f"Unbind dim0 failed. Expected {expected_map}, got {layout.to_dict()['tensor_map']}"
            )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_unbind_layout_inference_dim1(self, mock_platform):
        """
        Feature: Unbind on unsharded dimension 1
        Description: Unbind input (4, 4) on dim 1. Dim 0 is sharded, Dim 1 is unsharded.
        Expectation: Return tuple of layouts. Dim 0 preserves sharding.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        input_shape = (4, 4)

        output_layouts = op.infer_layout((x_layout,), extra_args=(1, [input_shape]))

        assert len(output_layouts) == 4

        expected_map = (1,)
        for layout in output_layouts:
            assert layout.to_dict()["tensor_map"] == expected_map, (
                f"Unbind dim1 failed. Expected {expected_map}, got {layout.to_dict()['tensor_map']}"
            )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_unbind_layout_inference_negative_dim(self, mock_platform):
        """
        Feature: Unbind with negative dimension
        Description: Unbind input (2, 4, 8) on dim -1 (last dim). Only last dim is unsharded.
        Expectation: Correctly resolves negative index and infers layout.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)
        input_shape = (2, 4, 8)

        output_layouts = op.infer_layout((x_layout,), extra_args=(-1, [input_shape]))

        assert len(output_layouts) == 8

        expected_map = (2, 1)
        for layout in output_layouts:
            assert layout.to_dict()["tensor_map"] == expected_map, (
                f"Unbind negative dim failed. Expected {expected_map}, got {layout.to_dict()['tensor_map']}"
            )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_unbind_layout_sharded_dim_error(self, mock_platform):
        """
        Feature: Unbind sharded dimension error
        Description: Attempt to unbind a dimension that is currently sharded.
        Expectation: ValueError raised.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        input_shape = (4, 4)

        with self.assertRaisesRegex(ValueError, "Unbinding a sharded dimension is not supported"):
            op.infer_layout((x_layout,), extra_args=(0, [input_shape]))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_unbind_layout_dim_out_of_range(self, mock_platform):
        """
        Feature: Unbind dimension out of range
        Description: Attempt to unbind a dimension index larger than rank.
        Expectation: ValueError raised.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        input_shape = (4, 4)

        with self.assertRaisesRegex(ValueError, "Dimension out of range"):
            op.infer_layout((x_layout,), extra_args=(2, [input_shape]))


if __name__ == "__main__":
    unittest.main()
