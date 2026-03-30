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
"""parallel_sort test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_sort import SortDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = SortDistributedOp("sort")


class TestParallelSort(unittest.TestCase):
    """Unit tests for SortDistributedOp."""
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

    def _make_2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2 (dp, tp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=4)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, tp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "tp", "mp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_layout_inference_basic(self, mock_platform):
        """
        Feature: Sort along an unsharded dimension
        Description: Input is sharded on dim0, sorting is performed on dim1 (unsharded).
        Expectation: Returns a tuple of two layouts (values, indices), both identical to input layout.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layouts = op.infer_layout((x_layout,), extra_args=(1, False, False))

        assert isinstance(output_layouts, tuple) and len(output_layouts) == 2, (
            "Sort must return a tuple of two layouts (values, indices)"
        )

        values_layout, indices_layout = output_layouts

        expected_map = (1, -1)

        assert values_layout.tensor_map == expected_map, (
            f"Values layout incorrect. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert indices_layout.tensor_map == expected_map, (
            f"Indices layout incorrect. Expected {expected_map}, "
            f"got {indices_layout.tensor_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, output_layouts, (x_layout,), (1, False, False)) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, output_layouts, (x_layout,), (1, False, False))}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_layout_inference_sharded_dim_error(self, mock_platform):
        """
        Feature: Sort along a sharded dimension
        Description: Input is sharded on dim0, attempt to sort on dim0.
        Expectation: Should raise ValueError because sorting requires global data along the sort axis.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "sorting along a sharded dimension .* is not supported"):
            op.infer_layout((x_layout,), extra_args=(0, True, False))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_layout_inference_negative_dim(self, mock_platform):
        """
        Feature: Sort with negative dimension index
        Description: Input (2D) sharded on dim0, sort on dim=-1 (last dim, which is unsharded).
        Expectation: Successfully infers layout, converting -1 to correct dimension index.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layouts = op.infer_layout((x_layout,), extra_args=(-1, False, True))

        values_layout, indices_layout = output_layouts
        expected_map = (1, -1)

        assert values_layout.tensor_map == expected_map
        assert indices_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_layout_inference_preserve_other_dims(self, mock_platform):
        """
        Feature: Sort preserves sharding on other dimensions
        Description: 3D input sharded on dim0 and dim2. Sort on dim1 (unsharded).
        Expectation: Output layouts preserve sharding on dim0 and dim2.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        output_layouts = op.infer_layout((x_layout,), extra_args=(1, False, False))

        values_layout, indices_layout = output_layouts
        expected_map = (2, -1, 0)

        assert values_layout.tensor_map == expected_map
        assert indices_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_layout_inference_all_replicate(self, mock_platform):
        """
        Feature: Sort on fully replicated tensor
        Description: Input is fully replicated. Sort on any dimension.
        Expectation: Output is fully replicated.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layouts = op.infer_layout((x_layout,), extra_args=(0, False, False))

        values_layout, _ = output_layouts
        expected_map = (-1, -1)

        assert values_layout.tensor_map == expected_map


if __name__ == "__main__":
    unittest.main()
