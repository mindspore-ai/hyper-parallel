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
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_matmul import LinearDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = LinearDistributedOp("Linear")


class TestParallelLinear(unittest.TestCase):
    """Unit tests for LinearDistributedOp."""
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

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_layout_data_parallel(self, mock_platform):
        """
        Feature: Linear data parallel
        Description: Data parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        output_layout = op.infer_layout((x_layout, w_layout, None), ())
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Data Parallel with transpose_a test failed. Expected {expected_map},"
            f" got {output_layout.tensor_map}"
        )

        assert op.get_expand_impl(None, output_layout, (x_layout, w_layout, None), ())is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {self.op.get_expand_impl(None, output_layout, (x_layout,), None)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_layout_hybrid_parallel(self, mock_platform):
        """
        Feature: Linear hybrid parallel
        Description: Hybrid parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)
        bias_layout = _build_layout(mesh, (Replicate(), Shard(0)), 1)
        output_layout = op.infer_layout((x_layout, w_layout, bias_layout), ())
        expected_map = (1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Hybrid Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        assert op.get_expand_impl(None, output_layout, (x_layout, w_layout, bias_layout), ())is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {self.op.get_expand_impl(None, output_layout, (x_layout,), None)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_layout_hybrid_tensor_parallel(self, mock_platform):
        """
        Feature: Linear hybrid tensor parallel
        Description: Hybrid tensor parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        output_layout = op.infer_layout((x_layout, w_layout, None), ())
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Hybrid Tensor Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_layout_hybrid_tensor_parallel_with_bias(self, mock_platform):
        """
        Feature: Linear hybrid tensor parallel
        Description: Hybrid tensor parallel scenario
        Expectation: raise error
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        bias_layout = _build_layout(mesh, (Shard(0),), 1)
        with self.assertRaisesRegex(ValueError, "Output dimensions must have same sharding"):
            _ = op.infer_layout((x_layout, w_layout, bias_layout), ())

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_layout_partial_with_sharded_contract_dim(self, mock_platform):
        """
        Feature: Linear partial status with sharded contract dimension
        Description: Test that partial status is set when contract dimension is sharded
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
        w_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        output_layout = op.infer_layout((x_layout, w_layout, None), ())

        expected_partial = [None, 'sum']
        assert output_layout.partial == expected_partial, (
            f"Partial status test failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )

        assert op.get_expand_impl(None, output_layout, (x_layout, w_layout, None), ())is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {self.op.get_expand_impl(None, output_layout, (x_layout,), None)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_linear_layout_partial_without_sharded_contract_dim(self, mock_platform):
        """
        Feature: Linear partial status without sharded contract dimension
        Description: Test that partial status is None when contract dimension is not sharded
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        w_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        output_layout = op.infer_layout((x_layout, w_layout, None), ())

        expected_partial = [None, None]
        assert output_layout.partial == expected_partial, (
            f"Partial status test failed. Expected {expected_partial}, "
            f"got {output_layout.partial}"
        )
        assert op.get_expand_impl(None, output_layout, (x_layout, w_layout, None), ())is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {self.op.get_expand_impl(None, output_layout, (x_layout,), None)}"
        )

if __name__ == "__main__":
    unittest.main()
