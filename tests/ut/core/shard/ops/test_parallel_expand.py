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
"""parallel_expand test"""
import unittest
from unittest.mock import patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_expand import ExpandDistributedOp, ExpandAsDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = ExpandDistributedOp("expand")
op2 = ExpandAsDistributedOp("expand_as")


class TestParallelExpand(unittest.TestCase):
    """Unit tests for ExpandDistributedOp."""
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
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"))

    def _make_2x2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "tp", "mp")):
        """Set up mock and return a standard 2x2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=mesh_dim_names)

    def _make_2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "mp")):
        """Set up mock and return a standard 2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=4)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=mesh_dim_names)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_layout_inference(self, mock_platform):
        """
        Feature: Expand unsharded singleton dimension
        Description: Expand last dimension (unsharded) while preserving sharded first dimension
        Expectation: Output layout preserves sharding on preserved dimension, expanded dimension unsharded
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, (2, 1), (-1, 5)]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        expected_map = ("dp", "None")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Basic expand failed. Expected {expected_map}, got {output_layout.alias_tensor_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should return None, "
            f"got {op.get_expand_impl(None, (output_layouts, None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_same_size_preserve_sharding(self, mock_platform):
        """
        Feature: Expand with explicit same-size dimensions preserves sharding.
        Description: Input shape (8, 4) with layout ("dp", "None"), expand(8, 4).
        Expectation: Sharding is preserved on dim 0 since size is unchanged.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, (8, 4), (8, 4)]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        expected_map = ("dp", "None")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Same-size sharded dim failed. Expected {expected_map}, "
            f"got {output_layout.alias_tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_layout_inference_3d(self, mock_platform):
        """
        Feature: Expand with -1 preservation
        Description: Multiple dimensions with -1 preservation and one expansion on unsharded dim
        Expectation: Preserved dimensions keep original sharding, expanded dimension becomes unsharded
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, (3, 1, 4), (-1, 10, -1)]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        expected_map = ("dp", "None", "mp")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Preserve with -1 failed. Expected {expected_map}, "
            f"got {output_layout.alias_tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_layout_prepend_new_dimensions(self, mock_platform):
        """
        Feature: Expand prepending multiple new dimensions
        Description: Prepend two new dimensions to 2D tensor with mixed sharding
        Expectation: Both new dimensions unsharded, existing dimensions preserve sharding
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, (5, 6), (2, 3, -1, -1)]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        expected_map = ("None", "None", "None", "mp")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Prepend multiple new dimensions failed. Expected {expected_map}, "
            f"got {output_layout.alias_tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_layout_scalar_expansion(self, mock_platform):
        """
        Feature: Expand scalar tensor
        Description: Expand 0-D scalar tensor to 2D shape (3,4)
        Expectation: Output layout fully unsharded (both dimensions unsharded)
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = ()
        x_layout = _build_layout(mesh, x_placements, 0)

        cache_values = [x_layout, (), (3, 4)]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        expected_map = ("None", "None")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Scalar expansion failed. Expected {expected_map}, got {output_layout.alias_tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_layout_invalid_expand_sharded_dim(self, mock_platform):
        """
        Feature: Expand sharded dimension
        Description: Attempt to expand a sharded dimension (should fail)
        Expectation: ValueError raised with clear message
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, (1, 3), (5, -1)]
        with self.assertRaisesRegex(ValueError, "cannot expand dimension 0"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_layout_invalid_minus_one_for_new_dim(self, mock_platform):
        """
        Feature: Expand with -1 for new dimension
        Description: Attempt to use -1 for prepended dimension (invalid per PyTorch semantics)
        Expectation: ValueError raised
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, (5, 6), (-1, 3, 4)]
        with self.assertRaisesRegex(ValueError, "cannot use -1 for new dimension at position 0"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_layout_invalid_dimension_reduction(self, mock_platform):
        """
        Feature: Expand reducing dimensions
        Description: Attempt to reduce dimensions with expand (output_ndim < input_ndim)
        Expectation: ValueError raised
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, (3, 4), (5,)]
        with self.assertRaisesRegex(ValueError, "cannot reduce dimensions with expand"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_as_layout_basic_expansion(self, mock_platform):
        """
        Feature: Basic expand_as with unsharded singleton dimension
        Description: Input (8,1) sharded on dim0, target (8,16) -> expand dim1 (unsharded)
        Expectation: Output layout preserves sharding on dim0, dim1 unsharded
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, (8, 1), (8, 16)]
        output_layouts, _ = op2.infer_layout(cache_values)
        output_layout = output_layouts[0]

        expected_map = ("dp", "None")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Basic expand_as failed. Expected {expected_map}, got {output_layout.alias_tensor_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op2.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should return None, "
            f"got {op2.get_expand_impl(None, (output_layouts, None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_as_layout_3d_preservation(self, mock_platform):
        """
        Feature: 3D expand_as with middle dimension expansion
        Description: Input (4,1,6) sharded on dim0/dim2, target (4,10,6) -> expand dim1
        Expectation: Preserved dims keep sharding, expanded dim unsharded
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, (4, 1, 6), (4, 10, 6)]
        output_layouts, _ = op2.infer_layout(cache_values)
        output_layout = output_layouts[0]

        expected_map = ("dp", "None", "mp")
        assert output_layout.alias_tensor_map == expected_map, (
            f"3D expand_as preservation failed. Expected {expected_map}, "
            f"got {output_layout.alias_tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_as_layout_prepend_dimensions(self, mock_platform):
        """
        Feature: Expand_as with prepended dimensions (rank promotion)
        Description: Input (8,1) sharded on dim0, target (2,3,8,16) -> prepend 2 dims + expand dim1
        Expectation: New dims unsharded, dim2 preserved (sharded), dim3 expanded (unsharded)
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, (8, 1), (2, 3, 8, 16)]
        output_layouts, _ = op2.infer_layout(cache_values)
        output_layout = output_layouts[0]

        expected_map = ("None", "None", "dp", "None")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Prepend dimensions failed. Expected {expected_map}, "
            f"got {output_layout.alias_tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_as_layout_scalar_expansion(self, mock_platform):
        """
        Feature: Scalar tensor expand_as
        Description: Input () scalar -> target (3,4,5)
        Expectation: Output layout fully unsharded (all dimensions unsharded)
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = ()
        x_layout = _build_layout(mesh, x_placements, 0)

        cache_values = [x_layout, (), (3, 4, 5)]
        output_layouts, _ = op2.infer_layout(cache_values)
        output_layout = output_layouts[0]

        expected_map = ("None", "None", "None")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Scalar expand_as failed. Expected {expected_map}, got {output_layout.alias_tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_as_layout_invalid_sharded_singleton(self, mock_platform):
        """
        Feature: Expand_as on sharded singleton dimension
        Description: Input (8,1) with dim1 sharded -> target (8,16) (invalid: cannot shard singleton)
        Expectation: ValueError raised with clear message about sharded dimension
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, (8, 1), (8, 16)]
        with self.assertRaisesRegex(ValueError, "cannot expand sharded dimension 1"):
            op2.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_as_layout_invalid_non_singleton_mismatch(self, mock_platform):
        """
        Feature: Non-singleton dimension size mismatch
        Description: Input (8,3) -> target (8,5) (invalid: 3!=5 and 3!=1)
        Expectation: ValueError raised for incompatible dimension sizes
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, (8, 3), (8, 5)]
        with self.assertRaisesRegex(ValueError, "cannot expand dimension 1 from size 3 to 5"):
            op2.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_as_layout_invalid_rank_reduction(self, mock_platform):
        """
        Feature: Target rank smaller than input rank
        Description: Input (4,5,6) -> target (4,5) (invalid: cannot reduce dimensions with expand)
        Expectation: ValueError raised for rank reduction
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, (4, 5, 6), (4, 5)]
        with self.assertRaisesRegex(ValueError, "target shape.*cannot be smaller than input shape"):
            op2.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_as_layout_right_aligned_broadcast(self, mock_platform):
        """
        Feature: Right-aligned dimension matching (PyTorch broadcast semantics)
        Description: Input (1,4) -> target (3,1,4) - implicit leading singleton added
        Expectation: Leading dimension unsharded (new), middle dimension unsharded (expanded), last preserved
        """
        mesh = self._make_2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, (4, 1), (3, 4, 1)]
        output_layouts, _ = op2.infer_layout(cache_values)
        output_layout = output_layouts[0]

        expected_map = ("None", "dp", "None")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Right-aligned broadcast failed. Expected {expected_map}, "
            f"got {output_layout.alias_tensor_map}"
        )


if __name__ == "__main__":
    unittest.main()
