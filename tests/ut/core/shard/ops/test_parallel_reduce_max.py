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
"""parallel_reduce_max test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_reduce import ReduceMaxDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelReduceMax(unittest.TestCase):
    """Unit tests for ReduceMaxDistributedOp."""
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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, cp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "cp", "mp"))

    def _run_scenario(self, op, x_layout, expected_map, extra_args):
        """Infer layout of reduce max operator"""
        output_layout = op.infer_layout((x_layout,), extra_args)
        assert output_layout.tensor_map == expected_map, (
            f"ReduceMax failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

        assert op.get_expand_impl(None, output_layout, (x_layout,), extra_args) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, output_layout, (x_layout,), extra_args)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reduce_max_data_parallel_1(self, mock_platform):
        """
        Feature: Data parallel.
        Description: reduce dp axis, keepdim=False.
        Expectation: dp axis reduced.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        op = ReduceMaxDistributedOp("ReduceMax")
        self._run_scenario(op, x_layout, expected_map=(-1, -1), extra_args=[0, False])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reduce_max_model_parallel_2(self, mock_platform):
        """
        Feature: Model parallel.
        Description: reduce mp axis, keepdim=True.
        Expectation: mp axis -> None.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(2)), 3)

        op = ReduceMaxDistributedOp("ReduceMax")
        self._run_scenario(op, x_layout, expected_map=(-1, -1, -1), extra_args=[2, True])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reduce_max_hybrid_parallel_3(self, mock_platform):
        """
        Feature: Hybrid parallel.
        Description: reduce cp axis, keepdim=False.
        Expectation: cp reduced, dp/mp kept.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        op = ReduceMaxDistributedOp("ReduceMax")
        self._run_scenario(op, x_layout, expected_map=(2, 0), extra_args=[1, False])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reduce_max_reduce_multiple_dims_4(self, mock_platform):
        """
        Feature: Reduce over multiple dims.
        Description: reduce (0, 2), keepdim=True.
        Expectation: dp/mp -> None, cp kept.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        op = ReduceMaxDistributedOp("ReduceMax")
        self._run_scenario(op, x_layout, expected_map=(-1, 1, -1), extra_args=[(0, 2), True])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reduce_max_reduce_all_dims_5(self, mock_platform):
        """
        Feature: Reduce over all dims.
        Description: dim=None, keepdim=False.
        Expectation: all reduced.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        op = ReduceMaxDistributedOp("ReduceMax")
        self._run_scenario(op, x_layout, expected_map=(), extra_args=[None, False])


if __name__ == "__main__":
    unittest.main()
