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
"""parallel_sort test"""
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_sort import SortDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS
op = SortDistributedOp("sort")
op_ms = SortDistributedOp("SortExt")


class TestParallelSort(unittest.TestCase):
    """Test parallel_sort ops."""

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

    def _make_2x4_mesh(self, mock_platform):
        """Mock a 2x4 device mesh."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )
        mock_platform.platform_type = MagicMock()
        return init_device_mesh(device_type="cpu", mesh_shape=(2, 4),
                                mesh_dim_names=("dp", "mp"), init_backend=False)

    def _make_2x2_mesh(self, mock_platform):
        """Mock a 2x2 device mesh."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )
        mock_platform.platform_type = MagicMock()
        return init_device_mesh(device_type="cpu", mesh_shape=(2, 2),
                                mesh_dim_names=("dp", "tp"), init_backend=False)

    def _make_2x2x2_mesh(self, mock_platform):
        """Mock a 2x2x2 device mesh."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )
        mock_platform.platform_type = MagicMock()
        return init_device_mesh(device_type="cpu", mesh_shape=(2, 2, 2),
                                mesh_dim_names=("dp", "tp", "mp"), init_backend=False)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_layout_inference_basic(self, mock_platform):
        """
        Feature: SortDistributedOp layout inference on an unsharded sort dimension.
        Description: Input is sharded on dim 0 (Shard(0)); sort is applied on the replicated dim 1.
        Expectation: Output values and indices layouts are identical to the input layout.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, 1]
        output_layouts, extra_info = op.infer_layout(cache_values)

        assert isinstance(output_layouts, tuple) and len(output_layouts) == 2, (
            "Sort must return a tuple of two layouts (values, indices)"
        )
        assert extra_info is None, f"Sort extra_info should be None, got {extra_info}"

        values_layout, indices_layout = output_layouts
        expected_map = (1, -1)

        assert values_layout.tensor_map == expected_map, (
            f"Values layout incorrect. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert indices_layout.tensor_map == expected_map, (
            f"Indices layout incorrect. Expected {expected_map}, "
            f"got {indices_layout.tensor_map}"
        )
        # SortDistributedOp does not override get_expand_impl → always None.
        # Verified once here; other test cases omit this check as per testing conventions.
        assert op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should return None for sort, "
            f"got {op.get_expand_impl(None, (output_layouts, None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_layout_inference_sharded_dim_error(self, mock_platform):
        """
        Feature: SortDistributedOp rejects sorting along a sharded dimension.
        Description: Input is sharded on dim 0; attempting to sort on that same dim 0.
        Expectation: ValueError is raised indicating sorting along a sharded dimension.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, 0]
        with self.assertRaisesRegex(ValueError, "sharded dimension"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_layout_inference_negative_dim(self, mock_platform):
        """
        Feature: SortDistributedOp handles negative dimension index correctly.
        Description: Input is sharded on dim 0; sort is applied on dim -1 (last dim, replicated).
        Expectation: Output layouts match the input layout with sharding preserved on dim 0.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, -1]
        output_layouts, extra_info = op.infer_layout(cache_values)

        values_layout, indices_layout = output_layouts
        expected_map = (1, -1)

        assert values_layout.tensor_map == expected_map, (
            f"Negative dim test failed. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert indices_layout.tensor_map == expected_map, (
            f"Negative dim indices layout failed. Expected {expected_map}, "
            f"got {indices_layout.tensor_map}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_layout_inference_preserve_other_dims(self, mock_platform):
        """
        Feature: SortDistributedOp preserves sharding on all non-sorted dimensions.
        Description: 3D input sharded on dims 0 and 2; sort is on unsharded dim 1.
        Expectation: Output layouts preserve all existing sharding on dims 0 and 2.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, 1]
        output_layouts, extra_info = op.infer_layout(cache_values)

        values_layout, indices_layout = output_layouts
        expected_map = (2, -1, 0)

        assert values_layout.tensor_map == expected_map, (
            f"Preserve other dims test failed. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert indices_layout.tensor_map == expected_map, (
            f"Indices preserve dims failed. Expected {expected_map}, "
            f"got {indices_layout.tensor_map}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_layout_inference_all_replicate(self, mock_platform):
        """
        Feature: SortDistributedOp layout inference when all placements are Replicate.
        Description: Input has no sharding (all Replicate); sort is on dim 0.
        Expectation: Output layouts are also fully replicated with all-(-1) tensor_map.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, 0]
        output_layouts, extra_info = op.infer_layout(cache_values)

        values_layout, _ = output_layouts
        expected_map = (-1, -1)

        assert values_layout.tensor_map == expected_map, (
            f"All replicate test failed. Expected {expected_map}, "
            f"got {values_layout.tensor_map}"
        )
        assert extra_info is None, f"extra_info should be None, got {extra_info}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_partial_input_raises_error(self, mock_platform):
        """
        Feature: SortDistributedOp rejects inputs with Partial status.
        Description: Input has Partial status set on dp axis (pending AllReduce).
        Expectation: ValueError is raised about Partial status not being allowed.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("dp", "sum")
        cache_values = [x_layout, 1]
        with self.assertRaisesRegex(ValueError, "Partial status"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_layout_multiaxis_tuple_sharded_dim_error(self, mock_platform):
        """
        Feature: SortDistributedOp rejects StridedShard multi-axis mapping on sort dim.
        Description: Dim 0 is mapped to a tuple of mesh axes via StridedShard + Shard(0) combo.
        Expectation: ValueError is raised about sorting along a sharded dimension.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        # Build a layout where dim 0 is mapped to a tuple of mesh axes via alias.
        # Use two Shard placements on the same tensor dim to produce a multi-axis mapping.
        from hyper_parallel.core.dtensor.placement_types import StridedShard  # pylint: disable=C0415
        x_layout = _build_layout(mesh, (StridedShard(0, split_factor=2), Shard(0), Replicate()), 2)

        # Sort on dim 0 which is now a multi-axis mapping → should raise
        cache_values = [x_layout, 0]
        with self.assertRaisesRegex(ValueError, "sharded dimension"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_preprocess_torch_stable_in_kwargs(self, mock_platform):
        """
        Feature: SortDistributedOp preprocess routes stable into kwargs for PyTorch.
        Description: PyTorch torch.sort declares stable keyword-only (after *); op_name is 'sort'.
        Expectation: local_args has 3 elements (tensor, dim, descending); stable is in local_kwargs.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        # Create a mock DTensor with the layout
        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = MagicMock()

        local_args, local_kwargs, cache_values = op.preprocess((mock_tensor,), {})

        assert local_kwargs == {'dim': -1, 'descending': False, 'stable': False}, (
            f"For PyTorch 'sort', local_kwargs should be {{'dim': -1, 'descending': False, 'stable': False}}, "
            f"got local_kwargs={local_kwargs}"
        )
        assert len(local_args) == 1, (
            f"For PyTorch 'sort', local_args should have 3 elements "
            f"(tensor, dim, descending), got {len(local_args)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sort_preprocess_mindspore_stable_in_args(self, mock_platform):
        """
        Feature: SortDistributedOp preprocess routes stable into positional args for MindSpore Primitive.
        Description: MindSpore SortExt Primitive does not accept kwargs; op_name is 'SortExt'.
        Expectation: local_kwargs is empty; local_args has 4 elements with stable as the 4th arg.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = MagicMock()

        local_args, local_kwargs, cache_values = op_ms.preprocess((mock_tensor,), {})

        assert not local_kwargs, (
            f"For MindSpore 'SortExt', local_kwargs should be empty, got {local_kwargs}"
        )
        assert len(local_args) == 4, (
            f"For MindSpore 'SortExt', local_args should have 4 elements "
            f"(tensor, dim, descending, stable), got {len(local_args)}"
        )
        assert local_args[3] is False, (
            f"stable default should be False, got {local_args[3]}"
        )


if __name__ == "__main__":
    unittest.main()
