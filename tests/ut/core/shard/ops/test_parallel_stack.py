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
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_stack import StackDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

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
        mock_platform.platform_type = MagicMock()
        return init_device_mesh(device_type="cpu", mesh_shape=(2, 4),
                                mesh_dim_names=("dp", "mp"), init_backend=False)


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

if __name__ == "__main__":
    unittest.main()