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
"""parallel_linear test"""
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_matmul import LinearDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = LinearDistributedOp("Linear")


class TestParallelLinear(unittest.TestCase):
    """Test Parallel Linear Distributed Operator."""
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

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_layout_data_parallel(self, mock_platform):
        """Test Linear layout with Data Parallel."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        cache_values = [x_layout, w_layout, None]
        output_layouts, extra_info = op.infer_layout(cache_values)
        output_layout = output_layouts[0]
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Data Parallel with transpose.a test failed. Expected {expected_map},"
            f" got {output_layout.tensor_map}"
        )

        assert extra_info is None, f"extra_info should be None, got {extra_info}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_layout_hybrid_parallel(self, mock_platform):
        """Test Linear layout with Hybrid Parallel."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)
        bias_layout = _build_layout(mesh, (Replicate(), Shard(0)), 1)
        cache_values = [x_layout, w_layout, bias_layout]
        output_layouts, extra_info = op.infer_layout(cache_values)
        output_layout = output_layouts[0]
        expected_map = (1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Hybrid Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_layout_hybrid_tensor_parallel(self, mock_platform):
        """Test Linear layout with Hybrid Tensor Parallel."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        cache_values = [x_layout, w_layout, None]
        output_layouts, extra_info = op.infer_layout(cache_values)
        output_layout = output_layouts[0]
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Hybrid Tensor Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_layout_hybrid_tensor_parallel_with_bias(self, mock_platform):
        """Test Linear layout with Hybrid Tensor Parallel and Bias."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        bias_layout = _build_layout(mesh, (Shard(0),), 1)
        cache_values = [x_layout, w_layout, bias_layout]
        with self.assertRaisesRegex(ValueError, "Output dimensions must have same sharding"):
            _ = op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_layout_partial_with_sharded_contract_dim(self, mock_platform):
        """Test Linear layout with Partial status with sharded contract dim."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        cache_values = [x_layout, w_layout, None]
        output_layouts, extra_info = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        expected_partial = [None, 'sum']
        assert output_layout.partial == expected_partial, (
            f"Partial status test failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )

        assert extra_info is None, f"extra_info should be None, got {extra_info}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_layout_partial_without_sharded_contract_dim(self, mock_platform):
        """Test Linear layout with Partial status without sharded contract dim."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        cache_values = [x_layout, w_layout, None]
        output_layouts, extra_info = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        expected_partial = [None, None]
        assert output_layout.partial == expected_partial, (
            f"Partial status test failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"


if __name__ == "__main__":
    unittest.main()
