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
"""parallel_repeat test"""
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_repeat import RepeatDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = RepeatDistributedOp("repeat")


class TestParallelRepeat(unittest.TestCase):
    """Unit tests for RepeatDistributedOp."""
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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, tp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "tp", "mp"),
            init_backend=False
        )

    def _infer_repeat(self, x_layout, *sizes):
        """Infer repeat output layout through the new cache_values calling convention."""
        cache_values = [x_layout, sizes]
        output_layouts, extra_info = op.infer_layout(cache_values)
        assert extra_info is None, f"Repeat extra_info should be None, got {extra_info}"
        return output_layouts[0], output_layouts, cache_values

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeat_layout_inference(self, mock_platform):
        """
        Feature: Repeat unsharded dimension
        Description: Repeat last dimension (unsharded) while preserving sharded first dimension (repeat=1)
        Expectation: Output layout preserves sharding on preserved dimension, repeated dimension unsharded
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout, output_layouts, cache_values = self._infer_repeat(x_layout, 1, 5)

        expected_map = (1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Basic repeat failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, (output_layouts, None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeat_layout_inference_3d(self, mock_platform):
        """
        Feature: Repeat with preservation (repeat=1)
        Description: Multiple dimensions with repeat=1 preservation and one repetition on unsharded dim
        Expectation: Preserved dimensions keep original sharding, repeated dimension becomes unsharded
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        output_layout, _, _ = self._infer_repeat(x_layout, 1, 10, 1)

        expected_map = (2, -1, 0)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Preserve with repeat=1 failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeat_layout_prepend_new_dimensions(self, mock_platform):
        """
        Feature: Repeat prepending multiple new dimensions
        Description: Prepend two new dimensions to 2D tensor with mixed sharding
        Expectation: Both new dimensions unsharded, existing non-repeated dimensions preserve sharding
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout, _, _ = self._infer_repeat(x_layout, 2, 3, 1, 1)

        expected_map = (-1, -1, -1, 0)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Prepend multiple new dimensions failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeat_layout_scalar_expansion(self, mock_platform):
        """
        Feature: Repeat scalar tensor
        Description: Repeat 0-D scalar tensor to 2D shape (3,4)
        Expectation: Output layout fully unsharded (both dimensions unsharded)
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = ()
        x_layout = _build_layout(mesh, x_placements, 0)

        output_layout, _, _ = self._infer_repeat(x_layout, 3, 4)

        expected_map = (-1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Scalar repeat failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeat_layout_invalid_repeat_sharded_dim(self, mock_platform):
        """
        Feature: Repeat sharded dimension
        Description: Attempt to repeat a sharded dimension > 1 times (should fail)
        Expectation: ValueError raised with clear message
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "Cannot repeat dimension 0 which is sharded"):
            op.infer_layout([x_layout, (5, 1)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeat_layout_packed_tuple_args(self, mock_platform):
        """
        Feature: Repeat with tuple args
        Description: Attempt to pass repeat args as a single packed tuple, ensuring unpack logic works
        Expectation: Output layout correctly inferred without crashing
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout, _, _ = self._infer_repeat(x_layout, (1, 5))

        expected_map = (1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Packed tuple repeat failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeat_layout_all_ones(self, mock_platform):
        """
        Feature: Repeat with all ones (shape-preserving)
        Description: Repeat tensor with 1s across all existing dimensions.
        Expectation: All dimensions preserve their original sharding.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout, _, _ = self._infer_repeat(x_layout, 1, 1)

        expected_map = (1, 0)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"All ones repeat failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeat_layout_zero_repeat(self, mock_platform):
        """
        Feature: Repeat with 0 (creating a zero-size tensor)
        Description: Repeat an unsharded dimension 0 times.
        Expectation: Dimension becomes 0-size but remains unsharded (-1).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout, _, _ = self._infer_repeat(x_layout, 1, 0)

        expected_map = (1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Zero repeat failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeat_layout_invalid_zero_repeat_sharded(self, mock_platform):
        """
        Feature: Repeat sharded dimension with 0
        Description: Attempt to repeat a sharded dimension 0 times.
        Expectation: ValueError raised because repeat_times != 1 on a sharded dimension.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "Cannot repeat dimension 0 which is sharded"):
            op.infer_layout([x_layout, (0, 1)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeat_layout_1d_to_4d_prepend(self, mock_platform):
        """
        Feature: Prepend multiple dimensions to 1D tensor
        Description: Prepend 3 new dimensions to a 1D sharded tensor.
        Expectation: 3 prepended dims are unsharded, the original 1D sharding is preserved.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0),)
        x_layout = _build_layout(mesh, x_placements, 1)

        output_layout, _, _ = self._infer_repeat(x_layout, 2, 3, 4, 1)

        expected_map = (-1, -1, -1, 1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"1D to 4D prepend failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeat_layout_fully_replicated(self, mock_platform):
        """
        Feature: Repeat fully replicated tensor
        Description: Tensor is unsharded everywhere, repeated across all dimensions.
        Expectation: Output remains completely unsharded (-1) across all dimensions.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout, _, _ = self._infer_repeat(x_layout, 5, 6)

        expected_map = (-1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Fully replicated repeat failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeat_layout_missing_layout(self, mock_platform):
        """
        Feature: Validate empty layouts
        Description: Call infer_layout with an empty layout or None layout.
        Expectation: ValueError is raised complaining about invalid input layout.
        """
        with self.assertRaisesRegex(ValueError, "requires a valid input tensor layout"):
            op.infer_layout([None, (1, 2)])

        with self.assertRaisesRegex(ValueError, "requires a valid input tensor layout"):
            op.infer_layout([])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeat_layout_missing_extra_args(self, mock_platform):
        """
        Feature: Validate empty repeat sizes
        Description: Call infer_layout without passing repeat sizes in cache_values.
        Expectation: ValueError is raised complaining about missing sizes.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(),)
        x_layout = _build_layout(mesh, x_placements, 1)

        with self.assertRaisesRegex(ValueError, "requires repeat sizes in cache_values"):
            op.infer_layout([x_layout, tuple()])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeat_layout_list_packed_args(self, mock_platform):
        """
        Feature: Robust parsing of cache_values as list
        Description: User passes repeat sizes as a list inside cache_values.
        Expectation: Layout is correctly inferred by unpacking the list.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout, _, _ = self._infer_repeat(x_layout, [1, 5])

        expected_map = (1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"List packed args failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeat_layout_float_args_cast(self, mock_platform):
        """
        Feature: Convert float arguments to integer
        Description: User passes float values as repeat sizes.
        Expectation: Op implementation safely casts floats to ints and infers layout.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout, _, _ = self._infer_repeat(x_layout, 1.0, 5.0)

        expected_map = (1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Float args cast failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeat_layout_complex_3d_mesh(self, mock_platform):
        """
        Feature: Repeat over 3D DeviceMesh
        Description: 4D tensor mapped to a 3D mesh. Prepend 2 new dimensions, preserve existing 4, repeat the last.
        Expectation: Prepended dimensions are unsharded, preserved keep layout, repeated becomes/remains unsharded.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Shard(2), Replicate())
        x_layout = _build_layout(mesh, x_placements, 4)

        output_layout, _, _ = self._infer_repeat(x_layout, 2, 2, 1, 1, 1, 5)

        expected_map = (-1, -1, 2, 1, 0, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Complex 3D mesh repeat failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeat_layout_partial_input_raises_error(self, mock_platform):
        """
        Feature: Validate Partial input
        Description: Input layout has Partial status.
        Expectation: ValueError is raised because repeat does not allow Partial input.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("dp", "sum")

        with self.assertRaisesRegex(ValueError, "Partial status"):
            op.infer_layout([x_layout, (1, 1)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeat_preprocess(self, mock_platform):
        """
        Feature: RepeatDistributedOp preprocess.
        Description: Convert DTensor input to local tensor and build cache_values.
        Expectation: Repeat sizes stay positional and cache_values carries layout plus sizes.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        local_tensor = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = local_tensor

        local_args, local_kwargs, cache_values = op.preprocess((mock_tensor, 1, 5), {})

        assert local_args == (local_tensor, 1, 5), f"Unexpected local_args: {local_args}"
        assert not local_kwargs, f"Repeat should not use kwargs, got {local_kwargs}"
        assert cache_values == [x_layout, (1, 5)], f"Unexpected cache_values: {cache_values}"


if __name__ == "__main__":
    unittest.main()
