# Copyright 2025-2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""parallel_sum_mean_ext test"""
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_reduce import SumExtDistributedOp, MeanExtDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP,
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelSumMeanProdExt(unittest.TestCase):
    """Unit tests for SumExtDistributedOp and MeanExtDistributedOp."""

    def setUp(self) -> None:
        """Clear global caches before each test to ensure isolation."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
        """Restore global cache state after each test."""
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
        """Set up mock and return a standard 2×2×2 (dp, cp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "cp", "mp"))

    def _run_scenario(self, op_name, expected_map, cache_values):
        """Infer layout of reduce operator using the new cache_values API."""
        if op_name == "SumExt":
            op = SumExtDistributedOp(op_name)
        else:
            op = MeanExtDistributedOp(op_name)
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        assert output_layout.tensor_map == expected_map, (
            f"{op_name} failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

        # get_expand_impl is not overridden for reduce ops — returns None.
        assert op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, (output_layouts, None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sum_ext_data_parallel_1(self, mock_platform):
        """
        Feature: Data parallel.
        Description: reduce dp axis, keepdim=False.
        Expectation: dp axis reduced.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        cache_values = [x_layout, 0, False]
        self._run_scenario("SumExt", expected_map=(-1, -1), cache_values=cache_values)
        self._run_scenario("MeanExt", expected_map=(-1, -1), cache_values=cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sum_ext_model_parallel_2(self, mock_platform):
        """
        Feature: Model parallel.
        Description: reduce mp axis, keepdim=True.
        Expectation: mp axis -> None.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(2)), 3)

        cache_values = [x_layout, 2, True]
        self._run_scenario("SumExt", expected_map=(-1, -1, -1), cache_values=cache_values)
        self._run_scenario("MeanExt", expected_map=(-1, -1, -1), cache_values=cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sum_ext_hybrid_parallel_3(self, mock_platform):
        """
        Feature: Hybrid parallel.
        Description: reduce cp axis, keepdim=False.
        Expectation: cp reduced, dp/mp kept.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        cache_values = [x_layout, 1, False]
        self._run_scenario("SumExt", expected_map=(2, 0), cache_values=cache_values)
        self._run_scenario("MeanExt", expected_map=(2, 0), cache_values=cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sum_ext_reduce_multiple_dims_4(self, mock_platform):
        """
        Feature: Reduce over multiple dims.
        Description: reduce (0, 2), keepdim=True.
        Expectation: dp/mp -> None, cp kept.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        cache_values = [x_layout, (0, 2), True]
        self._run_scenario("SumExt", expected_map=(-1, 1, -1), cache_values=cache_values)
        self._run_scenario("MeanExt", expected_map=(-1, 1, -1), cache_values=cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sum_ext_reduce_all_dims_5(self, mock_platform):
        """
        Feature: Reduce over all dims.
        Description: dim=None, keepdim=False.
        Expectation: all reduced.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        cache_values = [x_layout, None, False]
        self._run_scenario("SumExt", expected_map=(), cache_values=cache_values)
        self._run_scenario("MeanExt", expected_map=(), cache_values=cache_values)


if __name__ == "__main__":
    unittest.main()
    