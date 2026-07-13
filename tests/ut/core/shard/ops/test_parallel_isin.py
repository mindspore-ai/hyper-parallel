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
from unittest.mock import MagicMock, patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
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
    def setUp(self) -> None:
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
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

        cache_values = [elements_layout, test_elements_layout]
        output_layouts, extra_info = op.infer_layout(cache_values)

        assert extra_info is None, f"extra_info should be None, got {extra_info}"
        output_layout = output_layouts[0]
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Data parallel isin failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should return None for isin, "
            f"got {op.get_expand_impl(None, (output_layouts, None), cache_values)}"
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

        cache_values = [elements_layout, test_elements_layout]
        output_layouts, extra_info = op.infer_layout(cache_values)

        assert extra_info is None, f"extra_info should be None, got {extra_info}"
        output_layout = output_layouts[0]
        expected_map = (2, -1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Mixed parallel isin failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        # No need to verify get_expand_impl here - already verified in test_isin_layout_data_parallel

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

        cache_values = [elements_layout, test_elements_layout]
        with self.assertRaisesRegex(ValueError, "'test_elements' must be unsharded"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_isin_layout_missing_elements(self, mock_platform):
        """
        Feature: Isin without elements layout
        Description: elements layout is None or empty cache_values
        Expectation: ValueError raised
        """
        with self.assertRaisesRegex(ValueError, "'elements' requires a valid tensor layout"):
            op.infer_layout([])

        with self.assertRaisesRegex(ValueError, "'elements' requires a valid tensor layout"):
            op.infer_layout([None, None])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_isin_preprocess(self, mock_platform):
        """
        Feature: IsinDistributedOp preprocess routes keyword-only params into local_kwargs.
        Description: torch.isin has assume_unique and invert as keyword-only parameters (*).
        Expectation: local_args has 2 tensors; local_kwargs has assume_unique and invert;
            cache_values has 2 layouts.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        elements_placements = (Shard(0), Replicate())
        elements_layout = _build_layout(mesh, elements_placements, 2)
        test_elements_placements = (Replicate(), Replicate())
        test_elements_layout = _build_layout(mesh, test_elements_placements, 2)

        mock_elements = MagicMock()
        mock_elements.layout = elements_layout
        mock_elements.to_local.return_value = MagicMock()
        mock_test_elements = MagicMock()
        mock_test_elements.layout = test_elements_layout
        mock_test_elements.to_local.return_value = MagicMock()

        local_args, local_kwargs, cache_values = op.preprocess(
            (mock_elements, mock_test_elements), {}
        )

        assert len(local_args) == 2, (
            f"local_args should have 2 elements (elements, test_elements), "
            f"got {len(local_args)}"
        )
        assert local_kwargs == {'assume_unique': False, 'invert': False}, (
            f"local_kwargs should be {{'assume_unique': False, 'invert': False}}, "
            f"got {local_kwargs}"
        )
        assert len(cache_values) == 2, (
            f"cache_values should have 2 layouts, got {len(cache_values)}"
        )
        assert cache_values[0] is elements_layout, (
            f"cache_values[0] should be elements_layout, got {cache_values[0]}"
        )
        assert cache_values[1] is test_elements_layout, (
            f"cache_values[1] should be test_elements_layout, got {cache_values[1]}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_isin_preprocess_custom_kwargs(self, mock_platform):
        """
        Feature: IsinDistributedOp preprocess forwards custom assume_unique and invert.
        Description: Verify that assume_unique=True and invert=True are correctly forwarded.
        Expectation: local_kwargs reflects the custom values.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        elements_placements = (Replicate(), Replicate())
        elements_layout = _build_layout(mesh, elements_placements, 2)
        test_elements_placements = (Replicate(), Replicate())
        test_elements_layout = _build_layout(mesh, test_elements_placements, 2)

        mock_elements = MagicMock()
        mock_elements.layout = elements_layout
        mock_elements.to_local.return_value = MagicMock()
        mock_test_elements = MagicMock()
        mock_test_elements.layout = test_elements_layout
        mock_test_elements.to_local.return_value = MagicMock()

        local_args, local_kwargs, cache_values = op.preprocess(
            (mock_elements, mock_test_elements),
            {'assume_unique': True, 'invert': True}
        )

        assert local_kwargs == {'assume_unique': True, 'invert': True}, (
            f"local_kwargs should be {{'assume_unique': True, 'invert': True}}, "
            f"got {local_kwargs}"
        )


if __name__ == "__main__":
    unittest.main()
