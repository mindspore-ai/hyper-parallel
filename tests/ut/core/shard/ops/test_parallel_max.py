# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""parallel_max test"""
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_reduce import MaxDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP,
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = MaxDistributedOp("max")


class TestParallelMax(unittest.TestCase):
    """Unit tests for MaxDistributedOp."""

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

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2×4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"))

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2×2×2 (dp, tp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "tp", "mp"))

    def _make_2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2×2 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=4)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "mp"))

    # ------------------------------------------------------------------
    # Reduction — dim specified
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_reduce_dim_sharded(self, mock_platform):
        """
        Feature: Max reduction on sharded dimension
        Description: Reduce dim0 (sharded) of a 2D tensor
        Expectation: Returns tuple of layouts (values, indices).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        cache_values = [x_layout, 0, False]
        output_layouts, _ = op.infer_layout(cache_values)

        assert isinstance(output_layouts, tuple)
        assert len(output_layouts) == 2

        val_layout, idx_layout = output_layouts
        expected_map = (-1,)

        assert val_layout.tensor_map == expected_map, (
            f"Values layout incorrect. Expected {expected_map}, got {val_layout.tensor_map}"
        )
        assert idx_layout.tensor_map == expected_map, (
            f"Indices layout incorrect. Expected {expected_map}, got {idx_layout.tensor_map}"
        )

        # get_expand_impl is not overridden — returns None by default.
        assert op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, (output_layouts, None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_reduce_dim_replicated(self, mock_platform):
        """Max reduction on replicated dimension — dim removed."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        cache_values = [x_layout, 1, False]
        output_layouts, _ = op.infer_layout(cache_values)

        assert isinstance(output_layouts, tuple)
        val_layout, _ = output_layouts
        expected_map = (1,)
        assert val_layout.tensor_map == expected_map, (
            f"Values layout incorrect. Expected {expected_map}, got {val_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_keepdim(self, mock_platform):
        """Max reduction with keepdim=True — dim becomes None/-1."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        cache_values = [x_layout, 0, True]
        output_layouts, _ = op.infer_layout(cache_values)

        assert isinstance(output_layouts, tuple)
        val_layout, _ = output_layouts
        expected_map = (-1, -1)
        assert val_layout.tensor_map == expected_map, (
            f"Keepdim layout incorrect. Expected {expected_map}, got {val_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_negative_dim(self, mock_platform):
        """Max reduction with negative dimension index."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        cache_values = [x_layout, -1, False]
        output_layouts, _ = op.infer_layout(cache_values)

        val_layout, idx_layout = output_layouts
        expected_map = (1,)
        assert val_layout.tensor_map == expected_map
        assert idx_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_keepdim_negative(self, mock_platform):
        """Max reduction with keepdim and negative dim."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        cache_values = [x_layout, -1, True]
        output_layouts, _ = op.infer_layout(cache_values)

        val_layout, _ = output_layouts
        expected_map = (1, -1)
        assert val_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_reduce_keepdim_false_explicit(self, mock_platform):
        """Max reduction with explicit keepdim=False."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        cache_values = [x_layout, 0, False]
        output_layouts, _ = op.infer_layout(cache_values)

        val_layout, _ = output_layouts
        expected_map = (-1,)
        assert val_layout.tensor_map == expected_map

    # ------------------------------------------------------------------
    # Reduction — 3D / multi-dim
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_3d_sharded(self, mock_platform):
        """Max reduction on 3D tensor — remove middle dim."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)

        cache_values = [x_layout, 1, False]
        output_layouts, _ = op.infer_layout(cache_values)

        val_layout, _ = output_layouts
        expected_map = (2, -1)
        assert val_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_3d_reduce_last_dim(self, mock_platform):
        """Max reduction on last dimension of 3D tensor."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)

        cache_values = [x_layout, 2, False]
        output_layouts, _ = op.infer_layout(cache_values)

        val_layout, _ = output_layouts
        expected_map = (2, 1)
        assert val_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_4d_complex(self, mock_platform):
        """Max reduction on 4D tensor with mixed placements."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1), Replicate()), 4)

        cache_values = [x_layout, 2, False]
        output_layouts, _ = op.infer_layout(cache_values)

        val_layout, _ = output_layouts
        expected_map = (1, -1, -1)
        assert val_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_all_sharded_reduce_0(self, mock_platform):
        """Max reduction on fully sharded tensor — reduce dim0."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        cache_values = [x_layout, 0, False]
        output_layouts, _ = op.infer_layout(cache_values)

        val_layout, _ = output_layouts
        expected_map = (0,)
        assert val_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_1d_sharded_reduce(self, mock_platform):
        """Max reduction on 1D sharded tensor."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0),), 1)

        cache_values = [x_layout, 0, False]
        output_layouts, _ = op.infer_layout(cache_values)

        val_layout, idx_layout = output_layouts
        expected_map = ()
        assert val_layout.tensor_map == expected_map
        assert idx_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_replicated_input(self, mock_platform):
        """Max reduction on fully replicated tensor."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        cache_values = [x_layout, 0, False]
        output_layouts, _ = op.infer_layout(cache_values)

        val_layout, _ = output_layouts
        expected_map = (-1,)
        assert val_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_replicate_reduce_keepdim(self, mock_platform):
        """Max reduction on Replicated tensor with keepdim."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        cache_values = [x_layout, 0, True]
        output_layouts, _ = op.infer_layout(cache_values)

        val_layout, _ = output_layouts
        expected_map = (-1, -1)
        assert val_layout.tensor_map == expected_map

    # ------------------------------------------------------------------
    # Global reduction (dim=None)
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_global(self, mock_platform):
        """Global Max reduction — returns single layout."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        cache_values = [x_layout, None, False]
        output_layouts, _ = op.infer_layout(cache_values)

        assert len(output_layouts) == 1
        output_layout = output_layouts[0]
        expected_map = ()
        assert output_layout.tensor_map == expected_map, (
            f"Global reduce layout incorrect. Expected {expected_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_global_keepdim(self, mock_platform):
        """Global Max reduction with keepdim."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        cache_values = [x_layout, None, True]
        output_layouts, _ = op.infer_layout(cache_values)

        output_layout = output_layouts[0]
        expected_map = (-1, -1)
        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_global_fully_sharded(self, mock_platform):
        """Global Max reduction on fully sharded tensor."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        cache_values = [x_layout, None, False]
        output_layouts, _ = op.infer_layout(cache_values)

        output_layout = output_layouts[0]
        expected_map = ()
        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_scalar_input(self, mock_platform):
        """Max reduction on scalar input."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (), 0)

        cache_values = [x_layout, None, False]
        output_layouts, _ = op.infer_layout(cache_values)

        output_layout = output_layouts[0]
        expected_map = ()
        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_global_scalar_input(self, mock_platform):
        """Global Max reduction on Scalar (legacy empty extra_args path)."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (), 0)

        cache_values = [x_layout, None, False]
        output_layouts, _ = op.infer_layout(cache_values)

        output_layout = output_layouts[0]
        expected_map = ()
        assert output_layout.tensor_map == expected_map

    # ------------------------------------------------------------------
    # Element-wise (two-tensor inputs)
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_elementwise(self, mock_platform):
        """Element-wise max(a, b) — propagates first input layout."""
        mesh = self._make_2x4_mesh(mock_platform)
        a_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        b_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        cache_values = [a_layout, b_layout]
        output_layouts, _ = op.infer_layout(cache_values)

        output_layout = output_layouts[0]
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_elementwise_mismatched(self, mock_platform):
        """Element-wise max with mismatched sharding — adopts first input's layout."""
        mesh = self._make_2x4_mesh(mock_platform)
        layout_a = _build_layout(mesh, (Shard(0), Replicate()), 2)
        layout_b = _build_layout(mesh, (Replicate(), Replicate()), 2)

        cache_values = [layout_a, layout_b]
        output_layouts, _ = op.infer_layout(cache_values)

        output_layout = output_layouts[0]
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_binary_same_layout(self, mock_platform):
        """Element-wise max with identical sharded layouts."""
        mesh = self._make_2x4_mesh(mock_platform)
        layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        cache_values = [layout, layout]
        output_layouts, _ = op.infer_layout(cache_values)

        output_layout = output_layouts[0]
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_binary_mixed_shard_replicate(self, mock_platform):
        """Element-wise max with mixed Shard/Replicate."""
        mesh = self._make_2x4_mesh(mock_platform)
        layout_a = _build_layout(mesh, (Shard(0), Replicate()), 2)
        layout_b = _build_layout(mesh, (Replicate(), Replicate()), 2)

        cache_values = [layout_a, layout_b]
        output_layouts, _ = op.infer_layout(cache_values)

        output_layout = output_layouts[0]
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_binary_broadcast_scalar(self, mock_platform):
        """Element-wise max with scalar broadcasting."""
        mesh = self._make_2x4_mesh(mock_platform)
        layout_a = _build_layout(mesh, (Shard(0), Replicate()), 2)
        layout_b = _build_layout(mesh, (), 0)

        cache_values = [layout_a, layout_b]
        output_layouts, _ = op.infer_layout(cache_values)

        output_layout = output_layouts[0]
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_binary_orthogonal_sharding(self, mock_platform):
        """Element-wise max with orthogonal sharding — propagates first input's layout."""
        mesh = self._make_2x2_mesh(mock_platform)
        layout_a = _build_layout(mesh, (Shard(0), Replicate()), 2)
        layout_b = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        cache_values = [layout_a, layout_b]
        output_layouts, _ = op.infer_layout(cache_values)

        output_layout = output_layouts[0]
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_dim_out_of_range(self, mock_platform):
        """Invalid dimension index — raises Exception."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "Invalid reduce axis"):
            op.infer_layout([x_layout, 5, False])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_max_layout_inference_reduce_invalid_dim_type(self, mock_platform):
        """Max reduction with invalid dim type (float) — raises ValueError."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        with self.assertRaisesRegex(TypeError, "should be `None`, `int`"):
            op.infer_layout([x_layout, 1.5, False])


if __name__ == "__main__":
    unittest.main()
    