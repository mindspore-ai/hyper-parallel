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
"""test ut for parallel slice infer_layout"""
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_slice import SliceDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = SliceDistributedOp("Slice")


class TestParallelSlice(unittest.TestCase):
    """Unit tests for SliceDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        """Clean up after each test method."""
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

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "mp"),
            init_backend=False
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_slice_layout_1(self, mock_platform):
        """
        Feature: MatMul data parallel
        Description: Data parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        cache_values = [x_layout, (0, 0), (8, 4), (8, 8)]
        output_layouts, extra_info = op.infer_layout(cache_values)
        assert output_layouts == (x_layout,)
        assert extra_info[0] == (0, 0)
        assert extra_info[1] == (4, 4)

        calls = []

        def fake_slice(x, begin, end):
            calls.append((x, begin, end))
            return "sliced"

        impl = op.get_expand_impl(fake_slice, (output_layouts, extra_info), cache_values)
        assert impl is not None
        assert impl("local_x", (0, 0), (8, 4)) == "sliced"
        assert calls == [("local_x", (0, 0), (4, 4))]

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_slice_layout_2(self, mock_platform):
        """
        Feature: MatMul data parallel
        Description: Data parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        with self.assertRaises(ValueError):
            _ = op.infer_layout([x_layout, (0, 0), (4, 4), (8, 8)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_slice_preprocess(self, mock_platform):
        """
        Feature: Slice preprocess
        Description: Verify local args and cache values are built in new dispatch format
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        local_tensor = MagicMock()
        input_tensor = MagicMock()
        input_tensor.layout = x_layout
        input_tensor.shape = (8, 8)
        input_tensor.to_local.return_value = local_tensor

        local_args, local_kwargs, cache_values = op.preprocess((input_tensor, (0, 0), (8, 4)), {})

        assert local_args == (local_tensor, (0, 0), (8, 4))
        assert not local_kwargs
        assert cache_values == [x_layout, (0, 0), (8, 4), (8, 8)]

if __name__ == "__main__":
    unittest.main()
