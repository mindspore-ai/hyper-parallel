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
"""parallel_stack test"""
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_stack import StackDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS
from tests.custom_ops.parallel_stack_ext import StackExtDistributedOp

op = StackDistributedOp("stack")


class TestParallelStack(unittest.TestCase):
    """Test parallel_stack ops."""
    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _make_2x4_mesh(self, mock_platform):
        """Mock a 2x4 device mesh."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )
        mock_platform.platform_type = MagicMock()
        return init_device_mesh(
            device_type="cpu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "mp"),
            init_backend=False
        )


    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_layout_inference_dim_0(self, mock_platform):
        """Test Stack layout inference inserting a new dimension at dim 0."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        # Base layout tensor_map: (1, -1) -> dp is sharded, mp is replicate
        x_layout = _build_layout(mesh, x_placements, 2)

        # Stacking along dim 0
        cache_values = [x_layout, 0]
        output_layouts, extra_info = op.infer_layout(cache_values)

        self.assertTrue(isinstance(output_layouts, tuple))
        self.assertEqual(len(output_layouts), 1)
        self.assertIsNone(extra_info)

        out_layout = output_layouts[0]

        # New dimension should be inserted at index 0 and be mapped to -1 (Replicate)
        # Expected map: (-1, 1, -1)
        expected_map = (-1, 1, -1)
        self.assertEqual(out_layout.tensor_map, expected_map,
                         f"Expected {expected_map}, got {out_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_layout_inference_dim_last(self, mock_platform):
        """Test Stack layout inference inserting a new dimension at the end."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        # Stacking along dim 2 (the end of a 2D tensor)
        cache_values = [x_layout, 2]
        output_layouts, extra_info = op.infer_layout(cache_values)

        out_layout = output_layouts[0]

        # New dimension should be inserted at the end
        # Expected map: (1, -1, -1)
        expected_map = (1, -1, -1)
        self.assertEqual(out_layout.tensor_map, expected_map,
                         f"Expected {expected_map}, got {out_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_layout_inference_all_replicate(self, mock_platform):
        """Test Stack layout inference with fully replicated inputs."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        # Stacking along dim 1
        cache_values = [x_layout, 1]
        output_layouts, _ = op.infer_layout(cache_values)

        out_layout = output_layouts[0]

        # Everything should be Replicate (-1)
        expected_map = (-1, -1, -1)
        self.assertEqual(out_layout.tensor_map, expected_map)


    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_preprocess_single_tensor(self, mock_platform):
        """Test Stack preprocess handles a single input tensor correctly."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        t1 = MagicMock()
        t1.layout = x_layout
        t1.to_local.return_value = "local_t1"

        # Preprocess with a sequence containing only one tensor
        local_args, local_kwargs, cache_values = op.preprocess(
            ((t1,),), {'dim': 0}
        )

        self.assertEqual(local_args[0], ("local_t1",))
        self.assertEqual(local_kwargs['dim'], 0)
        self.assertEqual(cache_values[0], x_layout)
        self.assertEqual(cache_values[1], 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_layout_inference_1d_tensor(self, mock_platform):
        """Test Stack layout inference when stacking 1D tensors into a 2D tensor."""
        mesh = self._make_2x4_mesh(mock_platform)
        # 1D tensor sharded along mesh dimension 0 (dp)
        x_placements = (Shard(0),)
        x_layout = _build_layout(mesh, x_placements, 1)

        # tensor_map for 1D tensor should be (1,) because mesh has 2 dims (index 1 is dp, index 0 is mp)
        self.assertEqual(x_layout.tensor_map, (1,))

        # Stacking along dim 1 (creating a new dimension at the end)
        cache_values = [x_layout, 1]
        output_layouts, extra_info = op.infer_layout(cache_values)

        out_layout = output_layouts[0]

        # New dimension is inserted at index 1 and is unsharded (-1)
        # Expected map: (1, -1)
        expected_map = (1, -1)
        self.assertEqual(out_layout.tensor_map, expected_map)
        self.assertIsNone(extra_info)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_layout_inference_0d_tensor(self, mock_platform):
        """Test Stack layout inference when stacking 0D tensors (scalars) into a 1D tensor."""
        mesh = self._make_2x4_mesh(mock_platform)
        # 0D tensor has no dimensions, so placements are just Replicate for all mesh dims
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 0)

        self.assertEqual(x_layout.tensor_map, ())

        # Stacking scalars creates a 1D tensor. The only valid dim is 0.
        cache_values = [x_layout, 0]
        output_layouts, _ = op.infer_layout(cache_values)

        out_layout = output_layouts[0]

        # New dimension is inserted at index 0 and is unsharded (-1)
        expected_map = (-1,)
        self.assertEqual(out_layout.tensor_map, expected_map)

    def _make_2x2x2_mesh(self, mock_platform):
        """Mock a 2x2x2 device mesh for complex tests."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )
        mock_platform.platform_type = MagicMock()
        return init_device_mesh(device_type="cpu", mesh_shape=(2, 2, 2),
                                mesh_dim_names=("dp", "tp", "mp"), init_backend=False)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_layout_inference_complex_mesh(self, mock_platform):
        """Test Stack layout inference on a 3D mesh with mixed sharding."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        # 2D tensor on 3D mesh: sharded on dp, replicated on tp, sharded on mp
        x_placements = (Shard(0), Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        # Mesh dims: dp (idx 2), tp (idx 1), mp (idx 0)
        # Input tensor_map should be (2, 0)
        self.assertEqual(x_layout.tensor_map, (2, 0))

        # Stacking along dim 1 (inserting between the two existing dimensions)
        cache_values = [x_layout, 1]
        output_layouts, _ = op.infer_layout(cache_values)

        out_layout = output_layouts[0]

        # New dimension is inserted at index 1 and is unsharded (-1)
        # Expected map: (2, -1, 0)
        expected_map = (2, -1, 0)
        self.assertEqual(out_layout.tensor_map, expected_map)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_layout_inference_negative_dim_last(self, mock_platform):
        """Test Stack layout inference using dim=-1 (stacking at the end)."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        # For a 2D tensor, dim=-1 is equivalent to dim=2 in terms of insertion
        cache_values = [x_layout, -1]
        output_layouts, extra_info = op.infer_layout(cache_values)

        out_layout = output_layouts[0]

        # New dimension should be inserted at the end and mapped to -1
        # Expected map: (1, -1, -1)
        expected_map = (1, -1, -1)
        self.assertEqual(out_layout.tensor_map, expected_map,
                         f"Expected {expected_map}, got {out_layout.tensor_map}")
        self.assertIsNone(extra_info)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_layout_inference_negative_dim_first(self, mock_platform):
        """Test Stack layout inference using dim=-3 for a 2D tensor (stacking at the beginning)."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        # For a 2D tensor, ndim=2, dim=-3 translates to index 0 (-3 + 2 + 1 = 0)
        cache_values = [x_layout, -3]
        output_layouts, _ = op.infer_layout(cache_values)

        out_layout = output_layouts[0]

        # New dimension should be inserted at the start and mapped to -1
        # Expected map: (-1, 1, -1)
        expected_map = (-1, 1, -1)
        self.assertEqual(out_layout.tensor_map, expected_map,
                         f"Expected {expected_map}, got {out_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_preprocess_multiple_tensors(self, mock_platform):
        """Test Stack preprocess properly extracts locals from multiple tensors."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        t1 = MagicMock()
        t1.layout = x_layout
        t1.to_local.return_value = "local_t1"

        t2 = MagicMock()
        t2.layout = x_layout
        t2.to_local.return_value = "local_t2"

        # Preprocess with a sequence containing multiple tensors
        local_args, local_kwargs, cache_values = op.preprocess(
            ((t1, t2),), {'dim': 1}
        )

        self.assertEqual(local_args[0], ("local_t1", "local_t2"))
        self.assertEqual(local_kwargs['dim'], 1)
        # cache_values should contain all tensor layouts followed by the dimension
        self.assertEqual(len(cache_values), 3)
        self.assertEqual(cache_values[0], x_layout)
        self.assertEqual(cache_values[1], x_layout)
        self.assertEqual(cache_values[2], 1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_layout_inference_mismatched_layouts(self, mock_platform):
        """Test Stack layout inference raises ValueError for mismatched input layouts."""
        mesh = self._make_2x4_mesh(mock_platform)
        layout1 = _build_layout(mesh, (Shard(0), Replicate()), 2)
        layout2 = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        # Input cache values matching different layouts from different tensors
        cache_values = [layout1, layout2, 0]

        with self.assertRaisesRegex(ValueError, "All input tensors must have the same layout"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_layout_inference_empty_inputs(self, mock_platform):
        """Test Stack layout inference raises ValueError when no layouts are provided."""
        # Only dimension is passed in cache_values, mimicking an empty tensor list
        cache_values = [0]

        with self.assertRaisesRegex(ValueError, "stack requires at least one input tensor"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_layout_inference_dim_out_of_bounds(self, mock_platform):
        """Test Stack layout inference raises ValueError when dimension is out of valid bounds."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        # For a 2D tensor, valid dims for stack are [-3, 2]. Pass 3.
        cache_values_pos = [x_layout, 3]
        with self.assertRaisesRegex(ValueError, "Dimension out of range"):
            op.infer_layout(cache_values_pos)

        # For a 2D tensor, valid dims for stack are [-3, 2]. Pass -4.
        cache_values_neg = [x_layout, -4]
        with self.assertRaisesRegex(ValueError, "Dimension out of range"):
            op.infer_layout(cache_values_neg)

class TestParallelStackExt(unittest.TestCase):
    """Unit tests for StackExtDistributedOp."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

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
        """Set up mock and return a standard 2x2x2 (dp, cp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "cp", "mp"))

    def _make_2x2x1_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x1 (dp, cp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=4)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 1), mesh_dim_names=("dp", "cp", "mp"))

    def _infer_layout(self, stack_ext_op, layouts, axis):
        return stack_ext_op.infer_layout(tuple(layouts), extra_args=[axis])

    def _run_scenario(self, layouts, axis, expected_map):
        stack_ext_op = StackExtDistributedOp("StackExt")
        out_layout = self._infer_layout(stack_ext_op, layouts, axis)

        got_map = out_layout.tensor_map
        assert got_map == expected_map, (
            f"StackExt failed. Expected {expected_map}, got {got_map}"
        )

        assert stack_ext_op.get_expand_impl(None, out_layout, tuple(layouts), [axis]) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {stack_ext_op.get_expand_impl(None, out_layout, tuple(layouts), [axis])}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_ext_dispatch_and_layout(self, mock_platform):
        """
        Feature: StackExt distributed op dispatch and layout inference
        Description: Get StackExtDistributedOp via direct import (preferred) or registry (fallback),
                     then infer output layout for two inputs with identical layout using axis=0.
        Expectation: infer_layout succeeds and output tensor_map equals (-1, 2, 1, 0).
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x1 = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)
        x2 = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        stack_ext_op = StackExtDistributedOp("StackExt")
        out_layout = self._infer_layout(stack_ext_op, [x1, x2], axis=0)
        assert out_layout.tensor_map == (-1, 2, 1, 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_ext_layout_cache(self, mock_platform):
        """
        Feature: StackExt layout inference stability (cache-like behavior)
        Description: Call infer_layout twice with the same inputs and axis to verify repeated calls
                     produce consistent layout results.
        Expectation: Both infer_layout calls succeed and produce identical tensor_map results.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x1 = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)
        x2 = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        stack_ext_op = StackExtDistributedOp("StackExt")
        out1 = self._infer_layout(stack_ext_op, [x1, x2], axis=1)
        out2 = self._infer_layout(stack_ext_op, [x1, x2], axis=1)

        assert out1.tensor_map == out2.tensor_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_ext_axis0_same_layout_1(self, mock_platform):
        """
        Feature: StackExt output layout inference with inserted dimension
        Description: Inputs share the same layout (dp, cp, mp) and use axis=0 to insert a new
                     leading dimension on the output layout.
        Expectation: infer_layout succeeds and output tensor_map equals (-1, 2, 1, 0).
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x1 = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)
        x2 = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        self._run_scenario(layouts=[x1, x2], axis=0, expected_map=(-1, 2, 1, 0))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_ext_axis1_same_layout_2(self, mock_platform):
        """
        Feature: StackExt output layout inference with inserted dimension
        Description: Inputs share the same layout (dp, cp, mp) and use axis=1 to insert a new
                     dimension after dp on the output layout.
        Expectation: infer_layout succeeds and output tensor_map equals (2, -1, 1, 0).
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x1 = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)
        x2 = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        self._run_scenario(layouts=[x1, x2], axis=1, expected_map=(2, -1, 1, 0))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_ext_axis_minus1_3(self, mock_platform):
        """
        Feature: StackExt output layout inference with inserted dimension
        Description: Inputs share the same layout (dp, cp, mp) and use axis=-1 to append a new
                     trailing dimension on the output layout.
        Expectation: infer_layout succeeds and output tensor_map equals (2, 1, 0, -1).
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x1 = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)
        x2 = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        self._run_scenario(layouts=[x1, x2], axis=-1, expected_map=(2, 1, 0, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_ext_with_none_layout_4(self, mock_platform):
        """
        Feature: StackExt layout inference with optional None-layout inputs
        Description: Provide one normal layout input and one None layout input (constant-like).
                     The None layout input should be ignored in layout consistency checks.
        Expectation: infer_layout succeeds and output tensor_map follows the base layout insertion,
                     resulting in (-1, 2, 1, 0) for axis=0.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x1 = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)
        x2 = None

        self._run_scenario(layouts=[x1, x2], axis=0, expected_map=(-1, 2, 1, 0))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_ext_mismatch_tensor_map_should_raise_5(self, mock_platform):
        """
        Feature: StackExt input layout validation
        Description: Provide two non-None inputs with the same mesh_shape but different tensor_map.
        Expectation: infer_layout raises ValueError due to tensor_map mismatch among non-None inputs.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x1 = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)
        x2 = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)

        stack_ext_op = StackExtDistributedOp("StackExt")
        with self.assertRaises(ValueError):
            _ = self._infer_layout(stack_ext_op, [x1, x2], axis=0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_ext_mismatch_mesh_shape_should_raise_6(self, mock_platform):
        """
        Feature: StackExt input layout validation
        Description: Provide two non-None inputs with different mesh_shape shapes.
        Expectation: infer_layout raises ValueError due to mesh_shape mismatch among inputs.
        """
        mesh1 = self._make_2x2x2_mesh(mock_platform)
        x1 = _build_layout(mesh1, (Shard(0), Shard(1), Shard(2)), 3)

        mesh2 = self._make_2x2x1_mesh(mock_platform)
        x2 = _build_layout(mesh2, (Shard(0), Shard(1), Shard(2)), 3)

        stack_ext_op = StackExtDistributedOp("StackExt")
        with self.assertRaises(ValueError):
            _ = self._infer_layout(stack_ext_op, [x1, x2], axis=0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_ext_axis_out_of_range_should_raise_7(self, mock_platform):
        """
        Feature: StackExt axis range validation
        Description: Input rank=3 so output rank=4; axis must be within [-4, 3].
                     Provide an invalid axis=4 to trigger out-of-range validation.
        Expectation: infer_layout raises ValueError because axis is out of the valid range.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x1 = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)
        x2 = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        stack_ext_op = StackExtDistributedOp("StackExt")
        with self.assertRaises(ValueError):
            _ = self._infer_layout(stack_ext_op, [x1, x2], axis=4)


if __name__ == "__main__":
    unittest.main()
