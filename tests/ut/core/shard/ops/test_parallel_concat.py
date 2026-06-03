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
"""parallel_concat test"""
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_concat import ConcatDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = ConcatDistributedOp("cat")


class TestParallelCat(unittest.TestCase):
    """Unit tests for ConcatDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()
        self.platform = get_platform()

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

    def _make_2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "mp")):
        """Set up mock and return a standard 2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=4)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=mesh_dim_names)

    def _make_2x2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "tp", "mp")):
        """Set up mock and return a standard 2x2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=mesh_dim_names)

    def _make_1d_mesh(self, mock_platform, world_size=4):
        """Set up mock and return a standard 1D mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=world_size)
        return init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_mismatch(self, mock_platform):
        """
        Feature: Cat layout inference with mismatched input layouts
        Description: Attempt to concatenate tensors that have different sharding strategies
        Expectation: ValueError is raised indicating that all input tensors must have the same layout
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        y_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        with self.assertRaisesRegex(ValueError, "All input tensors must have the same layout"):
            op.infer_layout([x_layout, y_layout, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_get_expand_impl_unsharded_dim(self, mock_platform):
        """
        Feature: Get execution implementation for unsharded dimension
        Description: Try to get expand implementation when concatenating along a dimension that is NOT sharded
        Expectation: get_expand_impl returns None, delegating the operation to original PyTorch cat
        """
        mesh = self._make_2x4_mesh(mock_platform)
        placements = (Shard(0), Replicate())
        layout = _build_layout(mesh, placements, 2)

        cache_values = [layout, layout, 1]
        output_layouts, extra_info = op.infer_layout(cache_values)

        assert op.get_expand_impl(None, (output_layouts, extra_info), cache_values) is None, (
            "Concatenating on an unsharded dimension should return None to use the fallback PyTorch cat."
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_multiple_inputs(self, mock_platform):
        """
        Feature: Cat layout inference with more than 2 inputs
        Description: Concatenate 3 tensors with identical layouts
        Expectation: Output layout is identical to the base input layout without errors
        """
        mesh = self._make_2x4_mesh(mock_platform)
        placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, placements, 2)

        cache_values = [x_layout, x_layout, x_layout, 1]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        assert output_layout == x_layout, "Output layout should be identical to the input layout."
        assert cache_values[-1] == 1, "Dimension should remain 1."

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_negative_dim_3d(self, mock_platform):
        """
        Feature: Negative dimension normalization for 3D tensors
        Description: Concatenate 3D tensors using dim=-2
        Expectation: Output layout matches when dim=-2 resolves to dim=1
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        placements = (Shard(0), Replicate(), Shard(2))
        x_layout = _build_layout(mesh, placements, 3)

        output_layouts, _ = op.infer_layout([x_layout, x_layout, -2])

        assert output_layouts[0] == x_layout, "Negative dimension -2 should be accepted as dim 1."

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_dim_minus_ndim(self, mock_platform):
        """
        Feature: Negative dimension normalization boundary case (dim = -ndim)
        Description: Concatenate 2D tensors using dim=-2
        Expectation: dim=-2 is accepted as dim=0
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        output_layouts, _ = op.infer_layout([x_layout, x_layout, -2])

        assert output_layouts[0] == x_layout, "Dimension -2 on a 2D tensor should be accepted as dim 0."

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_get_expand_impl_unsharded_dim_3d(self, mock_platform):
        """
        Feature: Execution implementation for unsharded dimension in 3D
        Description: Concatenating 3D tensors along an unsharded middle dimension
        Expectation: get_expand_impl returns None
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(2)), 3)

        cache_values = [x_layout, x_layout, 1]
        output_layouts, extra_info = op.infer_layout(cache_values)

        assert op.get_expand_impl(None, (output_layouts, extra_info), cache_values) is None, (
            "Concatenating on an unsharded dimension should return None."
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_mismatch_multiple(self, mock_platform):
        """
        Feature: Mismatched layouts across multiple inputs
        Description: Provide 3 input tensors where the 3rd one has a different layout
        Expectation: ValueError is raised indicating inconsistency
        """
        mesh = self._make_2x4_mesh(mock_platform)
        layout_1 = _build_layout(mesh, (Shard(0), Replicate()), 2)
        layout_2 = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        with self.assertRaisesRegex(ValueError, "All input tensors must have the same layout"):
            op.infer_layout([layout_1, layout_1, layout_2, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_get_expand_impl_all_replicated(self, mock_platform):
        """
        Feature: Execution implementation when fully replicated
        Description: Tensors are fully replicated on all dimensions
        Expectation: get_expand_impl returns None for any concatenation dimension
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        cache_values_0 = [x_layout, x_layout, 0]
        output_layouts_0, extra_info_0 = op.infer_layout(cache_values_0)
        cache_values_1 = [x_layout, x_layout, 1]
        output_layouts_1, extra_info_1 = op.infer_layout(cache_values_1)

        assert op.get_expand_impl(None, (output_layouts_0, extra_info_0), cache_values_0) is None, (
            "Fully replicated tensor should return None for dim=0."
        )
        assert op.get_expand_impl(None, (output_layouts_1, extra_info_1), cache_values_1) is None, (
            "Fully replicated tensor should return None for dim=1."
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_sharded_dim_raises(self, mock_platform):
        """
        Feature: Cat layout inference on a sharded dimension
        Description: Attempt to concatenate along dim=0 which is sharded
        Expectation: ValueError is raised because concatenation on a sharded dim is not supported
        """
        mesh = self._make_2x4_mesh(mock_platform)
        layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "Concatenation along a sharded dimension"):
            op.infer_layout([layout, layout, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_negative_sharded_dim_raises(self, mock_platform):
        """
        Feature: Cat layout inference on a negative sharded dimension
        Description: Attempt to concatenate along dim=-1 where the last dimension is sharded
        Expectation: ValueError is raised with the properly normalized dimension
        """
        mesh = self._make_2x4_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        with self.assertRaisesRegex(ValueError, r"normalized_dim=1\) is not supported"):
            op.infer_layout([layout, layout, -1])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_empty_inputs(self, mock_platform):
        """
        Feature: Cat layout inference with empty inputs
        Description: Pass an empty tuple to layouts
        Expectation: ValueError is raised requiring at least one input DTensor
        """
        with self.assertRaisesRegex(ValueError, "cat requires at least one input DTensor"):
            op.infer_layout([0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_scalar_args_ignored(self, mock_platform):
        """
        Feature: Cat layout inference with scalar inputs mixed in
        Description: Pass a tuple of (layout, None, layout) simulating a scalar argument interleave
        Expectation: The None layout is ignored and the output layout is correctly inferred
        """
        mesh = self._make_2x4_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        output_layouts, _ = op.infer_layout([layout, None, layout, 0])
        output_layout = output_layouts[0]
        assert output_layout == layout, "Output layout should match the valid base layout."

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_default_dim(self, mock_platform):
        """
        Feature: Cat layout inference with default dimension
        Description: Pass the default normalized dim=0 produced by preprocess
        Expectation: The default dimension 0 is accepted
        """
        mesh = self._make_2x4_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        tensor_1 = MagicMock()
        tensor_2 = MagicMock()
        tensor_1.layout = layout
        tensor_2.layout = layout
        tensor_1.to_local.return_value = MagicMock()
        tensor_2.to_local.return_value = MagicMock()

        local_args, local_kwargs, cache_values = op.preprocess(((tensor_1, tensor_2),), {})

        assert not local_kwargs
        assert local_args[1] == 0
        assert len(cache_values) == 3
        assert cache_values[-1] == 0, "Default dimension 0 should be appended to cache_values."

        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        assert output_layout == layout

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_default_dim_sharded(self, mock_platform):
        """
        Feature: Cat layout inference with default dimension (sharded)
        Description: Default dim is 0, but dim 0 is sharded
        Expectation: ValueError is raised for sharded dimension 0
        """
        mesh = self._make_2x4_mesh(mock_platform)
        layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "Concatenation along a sharded dimension"):
            op.infer_layout([layout, layout, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_single_input(self, mock_platform):
        """
        Feature: Cat layout inference with a single input tensor
        Description: Concatenate a single tensor (trivial case)
        Expectation: Returns the same layout without errors
        """
        mesh = self._make_2x4_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        output_layouts, _ = op.infer_layout([layout, 0])
        output_layout = output_layouts[0]
        assert output_layout == layout

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_sharded_dim_multiple_inputs(self, mock_platform):
        """
        Feature: Cat layout inference on a sharded dim with >2 inputs
        Description: Concatenate 3 tensors along a sharded middle dimension
        Expectation: ValueError is raised
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Shard(1), Replicate()), 3)

        with self.assertRaisesRegex(ValueError, "Concatenation along a sharded dimension"):
            op.infer_layout([layout, layout, layout, 1])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_all_none_layouts(self, mock_platform):
        """
        Feature: Cat layout inference when all layouts are None
        Description: Extreme edge case where no DTensor layout is provided (e.g. all inputs are scalar/normal tensors)
        Expectation: ValueError is raised indicating at least one DTensor is required
        """
        with self.assertRaisesRegex(ValueError, "cat requires at least one input DTensor"):
            op.infer_layout([None, None, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_1d_mesh(self, mock_platform):
        """
        Feature: Cat layout inference on 1D DeviceMesh
        Description: Concatenate tensors on a 1D device mesh without sharding
        Expectation: Output layout is properly inferred
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=4)
        layout = _build_layout(mesh, (Replicate(),), 1)

        output_layouts, _ = op.infer_layout([layout, layout, 0])
        output_layout = output_layouts[0]

        assert output_layout == layout, "1D mesh concatenation on Replicate dim should succeed."

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_4d_tensor_dim2(self, mock_platform):
        """
        Feature: Cat layout inference on 4D tensors
        Description: Concatenate 4D tensors along dim=2, where dim 0 and 1 are sharded
        Expectation: Output layout is identical to the input layout
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate(), Replicate()), 4)

        output_layouts, _ = op.infer_layout([layout, layout, 2])
        output_layout = output_layouts[0]

        assert output_layout == layout, "Concatenating 4D tensor on unsharded dim 2 should succeed."

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_last_dim(self, mock_platform):
        """
        Feature: Cat layout inference on the last dimension
        Description: Concatenate along the highest positive dimension index (ndim - 1)
        Expectation: Success when the last dimension is Replicated
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        output_layouts, _ = op.infer_layout([layout, layout, 1])
        output_layout = output_layouts[0]

        assert output_layout == layout

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_get_expand_impl_cache_values(self, mock_platform):
        """
        Feature: get_expand_impl with cache values
        Description: Call get_expand_impl with the new infer_result and cache_values arguments
        Expectation: Returns None, delegating to native cat
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)

        cache_values = [layout, layout]
        infer_result = ((layout,), None)

        assert op.get_expand_impl(None, infer_result, cache_values) is None, (
            "Cache values should be safely handled by get_expand_impl."
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_interleaved_parallel(self, mock_platform):
        """
        Feature: Cat layout inference with interleaved parallel (virtual sharding)
        Description: Use an interleaved_parallel mesh dim name and concatenate on an unsharded dimension
        Expectation: Output layout is correctly inferred
        """
        mesh = self._make_2x2_mesh(mock_platform, mesh_dim_names=("dp", "interleaved_parallel"))
        layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        output_layouts, _ = op.infer_layout([layout, layout, 1])
        output_layout = output_layouts[0]

        assert output_layout == layout

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_negative_dim_replicate(self, mock_platform):
        """
        Feature: Negative dimension normalization on a replicated dimension
        Description: Use dim=-1 on a 2D tensor where the last dimension is Replicate
        Expectation: Normalizes to dim=1 and succeeds
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        output_layouts, _ = op.infer_layout([layout, layout, -1])
        output_layout = output_layouts[0]

        assert output_layout == layout

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_interspersed_nones(self, mock_platform):
        """
        Feature: Cat layout inference with interspersed None layouts
        Description: Pass a tuple with multiple None values simulating multiple scalar/non-tensor arguments
        Expectation: The valid layout is correctly identified and returned
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        output_layouts, _ = op.infer_layout([None, layout, None, layout, None, 1])
        output_layout = output_layouts[0]

        assert output_layout == layout, "Should correctly ignore all None layouts and return the valid base layout."

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_multi_axis_sharded_dim_raises(self, mock_platform):
        """
        Feature: Cat layout inference on a multi-axis mesh
        Description: Attempt to concatenate along a dimension that is sharded across the mesh
        Expectation: ValueError is raised
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)

        with self.assertRaisesRegex(ValueError, "Concatenation along a sharded dimension"):
            op.infer_layout([layout, layout, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_out_of_bounds_dim_positive(self, mock_platform):
        """
        Feature: Cat layout inference with out-of-bounds dimension
        Description: Pass a dimension index that exceeds the tensor's dimensionality
        Expectation: ValueError is raised by dim range validation
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        with self.assertRaises(ValueError):
            op.infer_layout([layout, layout, 5])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_get_expand_impl_multi_layout_tuple(self, mock_platform):
        """
        Feature: get_expand_impl with multiple inputs
        Description: Request expand implementation for 3 concatenated tensors
        Expectation: Returns None, delegating entirely to PyTorch backend
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)

        cache_values = [layout, layout, layout, 1]
        output_layouts, extra_info = op.infer_layout(cache_values)

        assert op.get_expand_impl(None, (output_layouts, extra_info), cache_values) is None, (
            "Should return None to use native cat implementation."
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_mismatch_shard_vs_replicate(self, mock_platform):
        """
        Feature: Layout mismatch detection (Shard vs Replicate)
        Description: Concatenate two tensors where one is sharded and the other is replicated
        Expectation: ValueError is raised due to mismatched layouts
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout1 = _build_layout(mesh, (Shard(0), Replicate()), 2)
        layout2 = _build_layout(mesh, (Replicate(), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "All input tensors must have the same layout"):
            op.infer_layout([layout1, layout2, 1])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_mismatch_shard0_vs_shard1(self, mock_platform):
        """
        Feature: Layout mismatch detection (Different Shard Mesh Dimensions)
        Description: Concatenate two tensors sharded on different mesh dimensions
        Expectation: ValueError is raised due to mismatched layouts
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout1 = _build_layout(mesh, (Shard(0), Replicate()), 2)
        layout2 = _build_layout(mesh, (Shard(1), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "All input tensors must have the same layout"):
            op.infer_layout([layout1, layout2, 1])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cat_layout_inference_negative_dim_normalization_3d(self, mock_platform):
        """
        Feature: Negative dimension normalization for 3D tensors (dim=-1)
        Description: Use dim=-1 on a 3D tensor where the last dimension is Replicate
        Expectation: Normalizes to dim=2 and successfully infers layout
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        output_layouts, _ = op.infer_layout([layout, layout, -1])

        assert output_layouts[0] == layout, "Negative dim -1 for a 3D tensor should be accepted as dim 2."


if __name__ == "__main__":
    unittest.main()
