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
"""
Unit tests for StackExt distributed operators
"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.custom_ops.parallel_stack_ext import StackExtDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelStackExt(unittest.TestCase):
    """Unit tests for StackExtDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()

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

    def _infer_layout(self, op, layouts, axis):
        return op.infer_layout(tuple(layouts), extra_args=[axis])

    def _run_scenario(self, layouts, axis, expected_map):
        op = StackExtDistributedOp("StackExt")
        out_layout = self._infer_layout(op, layouts, axis)

        got_map = out_layout.tensor_map
        assert got_map == expected_map, (
            f"StackExt failed. Expected {expected_map}, got {got_map}"
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

        op = StackExtDistributedOp("StackExt")
        out_layout = self._infer_layout(op, [x1, x2], axis=0)
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

        op = StackExtDistributedOp("StackExt")
        out1 = self._infer_layout(op, [x1, x2], axis=1)
        out2 = self._infer_layout(op, [x1, x2], axis=1)

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

        self._run_scenario(
            layouts=[x1, x2],
            axis=0,
            expected_map=(-1, 2, 1, 0),
        )

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

        self._run_scenario(
            layouts=[x1, x2],
            axis=1,
            expected_map=(2, -1, 1, 0),
        )

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

        self._run_scenario(
            layouts=[x1, x2],
            axis=-1,
            expected_map=(2, 1, 0, -1),
        )

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

        self._run_scenario(
            layouts=[x1, x2],
            axis=0,
            expected_map=(-1, 2, 1, 0),
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_ext_mismatch_tensor_map_should_raise_5(self, mock_platform):
        """
        Feature: StackExt input layout validation
        Description: Provide two non-None inputs with the same mesh_shape but different tensor_map
                     (e.g., one uses mp while the other uses None on the last dimension).
        Expectation: infer_layout raises ValueError due to tensor_map mismatch among non-None inputs.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x1 = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)
        x2 = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)

        op = StackExtDistributedOp("StackExt")
        with self.assertRaises(ValueError):
            _ = self._infer_layout(op, [x1, x2], axis=0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_stack_ext_mismatch_mesh_shape_should_raise_6(self, mock_platform):
        """
        Feature: StackExt input layout validation
        Description: Provide two non-None inputs with different mesh_shape shapes.
                     Note: rank_list length must match prod(mesh_shape) for each layout instance.
        Expectation: infer_layout raises ValueError due to mesh_shape mismatch among inputs.
        """
        mesh1 = self._make_2x2x2_mesh(mock_platform)
        x1 = _build_layout(mesh1, (Shard(0), Shard(1), Shard(2)), 3)

        mesh2 = self._make_2x2x1_mesh(mock_platform)
        x2 = _build_layout(mesh2, (Shard(0), Shard(1), Shard(2)), 3)

        op = StackExtDistributedOp("StackExt")
        with self.assertRaises(ValueError):
            _ = self._infer_layout(op, [x1, x2], axis=0)

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

        op = StackExtDistributedOp("StackExt")
        with self.assertRaises(ValueError):
            _ = self._infer_layout(op, [x1, x2], axis=4)


if __name__ == "__main__":
    unittest.main()
