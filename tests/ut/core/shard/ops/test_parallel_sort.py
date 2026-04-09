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
from unittest.mock import MagicMock, patch

import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_sort import SortDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = SortDistributedOp("sort")


class TestParallelSort(unittest.TestCase):
    """Test parallel_sort ops."""
    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _make_2x4_mesh(self, mock_platform):
        """Mock a 2x4 device mesh."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.platform_type = MagicMock()
        return init_device_mesh(device_type="cpu", mesh_shape=(2, 4),
                                mesh_dim_names=("dp", "mp"), init_backend=False)

    def _make_2x2_mesh(self, mock_platform):
        """Mock a 2x2 device mesh."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.platform_type = MagicMock()
        return init_device_mesh(device_type="cpu", mesh_shape=(2, 2),
                                mesh_dim_names=("dp", "tp"), init_backend=False)

    def _make_2x2x2_mesh(self, mock_platform):
        """Mock a 2x2x2 device mesh."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.platform_type = MagicMock()
        return init_device_mesh(device_type="cpu", mesh_shape=(2, 2, 2),
                                mesh_dim_names=("dp", "tp", "mp"), init_backend=False)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_layout_inference_basic(self, mock_platform):
        """Test Sort layout inference with basic sharding."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, 1]
        output_layouts, extra_info = op.infer_layout(cache_values)

        assert isinstance(output_layouts, tuple) and len(output_layouts) == 2, (
            "Sort must return a tuple of two layouts (values, indices)"
        )
        assert extra_info is None, f"Sort extra_info should be None, got {extra_info}"

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

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_layout_inference_sharded_dim_error(self, mock_platform):
        """Test Sort layout inference with sharded dimension error."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, 0]
        with self.assertRaisesRegex(ValueError, "sharded dimension"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_layout_inference_negative_dim(self, mock_platform):
        """Test Sort layout inference with negative dimension."""
        mesh = self._make_2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, -1]
        output_layouts, extra_info = op.infer_layout(cache_values)

        values_layout, indices_layout = output_layouts
        expected_map = (1, -1)

        assert values_layout.tensor_map == expected_map
        assert indices_layout.tensor_map == expected_map
        assert extra_info is None

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_layout_inference_preserve_other_dims(self, mock_platform):
        """Test Sort layout inference with preserve other dims."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, 1]
        output_layouts, extra_info = op.infer_layout(cache_values)

        values_layout, indices_layout = output_layouts
        expected_map = (2, -1, 0)

        assert values_layout.tensor_map == expected_map
        assert indices_layout.tensor_map == expected_map
        assert extra_info is None

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_layout_inference_all_replicate(self, mock_platform):
        """Test Sort layout inference with all Replicate."""
        mesh = self._make_2x2_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, 0]
        output_layouts, extra_info = op.infer_layout(cache_values)

        values_layout, _ = output_layouts
        expected_map = (-1, -1)

        assert values_layout.tensor_map == expected_map
        assert extra_info is None


if __name__ == "__main__":
    unittest.main()
