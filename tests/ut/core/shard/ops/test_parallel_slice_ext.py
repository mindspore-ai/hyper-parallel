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
"""parallel_slice_ext test"""
import os
import unittest
from unittest.mock import patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_slice_ext import SliceExtDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

slice_op = SliceExtDistributedOp("SliceExt")


class TestParallelSliceExt(unittest.TestCase):
    """Unit tests for SliceExtDistributedOp."""

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
        """Configure common mock-platform attributes used across tests."""
        if platform_type is not None:
            mock_platform.platform_type = platform_type
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, mp, cp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "mp", "cp"))

    def _make_2x2_mesh(self, mock_platform):
        """Set up mock and return a 2x2 (dp, tp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=4)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_normal(self, mock_platform):
        """
        Feature: SliceExt operator layout inference under normal conditions.
        Description: Test normal slice where axis is not sharded.
        Expectation: Output layout preserves the same sharding as input.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        # placements: (Replicate on dp, Shard(0) on mp, Shard(2) on cp)
        # alias_tensor_map: ("mp", "None", "cp")
        input_layout = _build_layout(mesh, (Replicate(), Shard(0), Shard(2)), 3)
        axis = 1  # dim 1 is unsharded ("None")
        cache_values = [input_layout, axis]

        output_layouts, extra_info = slice_op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        assert extra_info is None, f"extra_info should be None, got {extra_info}"
        assert output_layout.tensor_map == input_layout.tensor_map, (
            f"Output layout tensor_map should match input, "
            f"expected {input_layout.tensor_map}, got {output_layout.tensor_map}"
        )

        # SliceExtDistributedOp does not override get_expand_impl → always None.
        # Verified once here; other test cases omit this check as per testing conventions.
        assert slice_op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should return None for SliceExt, "
            f"got {slice_op.get_expand_impl(None, (output_layouts, None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_sharded_axis_error(self, mock_platform):
        """
        Feature: SliceExt operator layout inference with sharded axis.
        Description: Test slice along a sharded dimension is rejected.
        Expectation: ValueError is raised with "sharded axis" message.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        # alias_tensor_map: ("mp", "None", "cp"), dim 0 is sharded on "mp"
        input_layout = _build_layout(mesh, (Replicate(), Shard(0), Shard(2)), 3)
        axis = 0  # dim 0 is sharded ("mp") → should raise
        cache_values = [input_layout, axis]

        with self.assertRaisesRegex(ValueError, "sharded axis"):
            slice_op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_negative_dim(self, mock_platform):
        """
        Feature: SliceExt operator handles negative dimension index.
        Description: Negative axis is normalized and validated correctly.
        Expectation: Negative dim that resolves to unsharded dim succeeds;
                     negative dim that resolves to sharded dim raises.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        # alias_tensor_map: ("mp", "None", "cp")
        input_layout = _build_layout(mesh, (Replicate(), Shard(0), Shard(2)), 3)

        # axis=-2 → normalized to dim 1 (unsharded "None") → should pass
        cache_values = [input_layout, -2]
        output_layouts, extra_info = slice_op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        assert extra_info is None, f"extra_info should be None, got {extra_info}"
        assert output_layout.tensor_map == input_layout.tensor_map, (
            f"Negative dim test: output layout tensor_map should match input, "
            f"expected {input_layout.tensor_map}, got {output_layout.tensor_map}"
        )

        # axis=-1 → normalized to dim 2 (sharded "cp") → should raise
        cache_values_invalid = [input_layout, -1]
        with self.assertRaisesRegex(ValueError, "sharded axis"):
            slice_op.infer_layout(cache_values_invalid)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_all_replicate(self, mock_platform):
        """
        Feature: SliceExt operator layout inference with all Replicate placements.
        Description: All dimensions are replicated; any axis is valid.
        Expectation: Output layout preserves the fully replicated sharding.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        axis = 0
        cache_values = [input_layout, axis]

        output_layouts, extra_info = slice_op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        assert extra_info is None, f"extra_info should be None, got {extra_info}"
        assert output_layout.tensor_map == input_layout.tensor_map, (
            f"All replicate test: output layout tensor_map should match input, "
            f"expected {input_layout.tensor_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_partial_input_error(self, mock_platform):
        """
        Feature: SliceExt operator rejects Partial status inputs.
        Description: Input layout has Partial status set.
        Expectation: ValueError is raised about Partial status.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        input_layout.set_partial_by_dev_axis("dp", "sum")
        cache_values = [input_layout, 1]

        with self.assertRaisesRegex(ValueError, "Partial status"):
            slice_op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_invalid_axis_type(self, mock_platform):
        """
        Feature: SliceExt operator validates axis is int.
        Description: Non-int axis value is rejected.
        Expectation: ValueError is raised about axis type.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        cache_values = [input_layout, 1.5]

        with self.assertRaisesRegex(ValueError, "axis should be int"):
            slice_op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_axis_out_of_range(self, mock_platform):
        """
        Feature: SliceExt operator validates axis is in range.
        Description: Axis value exceeds tensor dimensions.
        Expectation: ValueError is raised about axis out of range.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        cache_values = [input_layout, 3]

        with self.assertRaisesRegex(ValueError, "axis out of range"):
            slice_op.infer_layout(cache_values)


if __name__ == "__main__":
    unittest.main()
