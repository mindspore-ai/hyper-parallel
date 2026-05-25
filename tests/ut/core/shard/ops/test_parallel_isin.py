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
"""parallel_isin test"""
import os
import unittest
from unittest.mock import patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_isin import IsinDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = IsinDistributedOp("isin")


class TestParallelIsin(unittest.TestCase):
    """Unit tests for IsinDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

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

    def _make_2x3x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x3x4 (dp, tp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=24)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 3, 4), mesh_dim_names=("dp", "tp", "mp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_isin_layout_data_parallel(self, mock_platform):
        """
        Feature: Isin data parallel
        Description: elements sharded on data parallel dimension, test_elements fully replicated
        Expectation: Output layout identical to elements layout
        """
        mesh = self._make_2x4_mesh(mock_platform)
        elements_placements = (Shard(0), Replicate())
        elements_layout = _build_layout(mesh, elements_placements, 2)
        test_elements_placements = (Replicate(), Replicate())
        test_elements_layout = _build_layout(mesh, test_elements_placements, 2)

        output_layout = op.infer_layout((elements_layout, test_elements_layout), extra_args=None)

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Data parallel isin failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, output_layout, (elements_layout, test_elements_layout), 
                                       extra_args=None) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"""got {op.get_expand_impl(None, output_layout, (elements_layout, test_elements_layout),
                                            extra_args=None)}"""
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_isin_layout_mixed_parallel(self, mock_platform):
        """
        Feature: Isin mixed parallel
        Description: elements with mixed sharding (dp on dim0, mp on dim2), test_elements fully replicated
        Expectation: Output layout identical to elements layout
        """
        mesh = self._make_2x3x4_mesh(mock_platform)
        elements_placements = (Shard(0), Replicate(), Shard(2))
        elements_layout = _build_layout(mesh, elements_placements, 3)
        test_elements_placements = (Replicate(), Replicate(), Replicate())
        test_elements_layout = _build_layout(mesh, test_elements_placements, 3)

        output_layout = op.infer_layout((elements_layout, test_elements_layout), extra_args=None)

        expected_map = (2, -1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Mixed parallel isin failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_isin_layout_invalid_test_elements_sharded(self, mock_platform):
        """
        Feature: Isin with test_elements sharded on dim0
        Description: test_elements sharded on first dimension violates global view requirement
        Expectation: ValueError raised with clear message
        """
        mesh = self._make_2x4_mesh(mock_platform)
        elements_placements = (Replicate(), Replicate())
        elements_layout = _build_layout(mesh, elements_placements, 2)
        test_elements_placements = (Shard(0), Replicate())
        test_elements_layout = _build_layout(mesh, test_elements_placements, 2)

        with self.assertRaisesRegex(ValueError, "'test_elements' must be unsharded. Current tensor_map:"):
            op.infer_layout((elements_layout, test_elements_layout), extra_args=None)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_isin_layout_missing_test_elements(self, mock_platform):
        """
        Feature: Isin without test_elements layout
        Description: Missing second layout in tuple
        Expectation: ValueError raised
        """
        mesh = self._make_2x4_mesh(mock_platform)
        elements_placements = (Replicate(), Replicate())
        elements_layout = _build_layout(mesh, elements_placements, 2)

        with self.assertRaisesRegex(ValueError, "'test_elements' requires a valid tensor layout"):
            op.infer_layout((elements_layout,), extra_args=None)

        with self.assertRaisesRegex(ValueError, "'test_elements' requires a valid tensor layout"):
            op.infer_layout((elements_layout, None), extra_args=None)


if __name__ == "__main__":
    unittest.main()
