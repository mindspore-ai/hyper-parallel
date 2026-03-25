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
"""Unit tests for element-wise distributed operators"""
import os
import unittest
from unittest.mock import patch
import copy
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_elementwise import ElementWiseDistributedOp, AddDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelElementwiseOps(unittest.TestCase):
    """Unit tests for ElementWiseDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()
        self.op = ElementWiseDistributedOp("element_wise")
        self.op_with_partial = AddDistributedOp("element_wise_with_partial")

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, sp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "sp", "mp"),
            init_backend=False
        )

    def _run_scenario(self, x_layout, y_layout, expected_map, extra_args):
        """Infer layout of element-wise operator"""
        if x_layout.is_partial() or y_layout.is_partial():
            op_ = self.op_with_partial
        else:
            op_ = self.op
        output_layout = op_.infer_layout((x_layout, y_layout), extra_args)
        got_map = output_layout.tensor_map
        assert got_map == expected_map, (
            f"Element-wise failed. Expected {expected_map}, got {got_map}"
        )

    def _run_single_input_scenario(self, x_layout, expected_map, extra_args):
        """Infer layout of element-wise operator with single input"""
        if x_layout.is_partial():
            op_ = self.op_with_partial
        else:
            op_ = self.op
        output_layout = op_.infer_layout((x_layout,), extra_args)
        got_map = output_layout.tensor_map
        assert got_map == expected_map, (
            f"Element-wise failed. Expected {expected_map}, got {got_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_single_input_partial_0(self, mock_platform):
        """
        Feature: Element-wise operator with single input
        Description: Single input with Partial status
        Expectation: Output keeps input layout with Partial
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        x_layout.set_partial_by_dev_axis("dp", "sum")

        with self.assertRaisesRegex(ValueError, "has Partial status which is not allowed"):
            self.op.infer_layout((x_layout,), extra_args={"input_shapes": [(4, 8, 16)]})

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_single_input_replicate_1(self, mock_platform):
        """
        Feature: Element-wise operator with single input
        Description: Single input with all dimensions replicated
        Expectation: Output keeps input layout
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)

        self._run_single_input_scenario(
            x_layout,
            expected_map=(-1, -1, -1),
            extra_args={"input_shapes": [(4, 8, 16)]},
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_single_input_sharded_2(self, mock_platform):
        """
        Feature: Element-wise operator with single input
        Description: Single input with all dimensions sharded
        Expectation: Output keeps input layout
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        self._run_single_input_scenario(
            x_layout,
            expected_map=(2, 1, 0),
            extra_args={"input_shapes": [(4, 8, 16)]},
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_single_input_partial_3(self, mock_platform):
        """
        Feature: Element-wise operator with single input
        Description: Single input with Partial status
        Expectation: Output keeps input layout with Partial
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        x_layout.set_partial_by_dev_axis("dp", "sum")

        output_layout = self.op_with_partial.infer_layout((x_layout,), extra_args={"input_shapes": [(4, 8, 16)]})

        assert output_layout.partial[mesh.axis_index("dp")] == "sum"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_both_replicate_4(self, mock_platform):
        """
        Feature: Element-wise operator with two inputs
        Description: Both inputs replicated on all dimensions
        Expectation: Output replicated on all dimensions
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        y_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)

        self._run_scenario(
            x_layout, y_layout,
            expected_map=(-1, -1, -1),
            extra_args={"input_shapes": [(4, 8, 16), (4, 8, 16)]},
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_both_sharded_same_5(self, mock_platform):
        """
        Feature: Element-wise operator with two inputs
        Description: Both inputs sharded on same dimensions
        Expectation: Output sharded on same dimensions
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(2)), 3)
        y_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(2)), 3)

        self._run_scenario(
            x_layout, y_layout,
            expected_map=(2, -1, 0),
            extra_args={"input_shapes": [(4, 8, 16), (4, 8, 16)]},
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_one_sharded_one_replicate_6(self, mock_platform):
        """
        Feature: Element-wise operator with two inputs
        Description: One input sharded, other replicated
        Expectation: Output sharded
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(2)), 3)
        y_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)

        self._run_scenario(
            x_layout, y_layout,
            expected_map=(2, -1, 0),
            extra_args={"input_shapes": [(4, 8, 16), (4, 8, 16)]},
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_different_sharding_conflict_7(self, mock_platform):
        """
        Feature: Element-wise operator with two inputs
        Description: Different sharding patterns on same dimension
        Expectation: Raise ValueError (requires communication)
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        y_layout = _build_layout(mesh, (Replicate(), Shard(0), Replicate()), 3)

        extra_args = {"input_shapes": [(4, 8, 16), (4, 8, 16)]}

        with self.assertRaisesRegex(ValueError, "should have same sharding pattern"):
            self.op.infer_layout((x_layout, y_layout), extra_args=extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sharded_broadcasts_to_replicated_8(self, mock_platform):
        """
        Feature: Element-wise operator with broadcasting
        Description: Sharded input broadcasts to replicated (size 1 -> N)
        Expectation: Raise ValueError (requires communication)
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        y_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        y_layout._partial = [None] * len(y_layout._partial)
        extra_args = {"input_shapes": [(1, 8, 16), (4, 8, 16)]}

        with self.assertRaisesRegex(ValueError, "Broadcasting dimension cannot be sharded"):
            self.op.infer_layout((x_layout, y_layout), extra_args=extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_replicated_broadcasts_to_sharded_9(self, mock_platform):
        """
        Feature: Element-wise operator with broadcasting
        Description: Replicated input broadcasts to sharded (size 1 -> N)
        Expectation: Output sharded
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        y_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(2)), 3)
        x_layout._partial = [None] * len(x_layout._partial)

        self._run_scenario(
            x_layout, y_layout,
            expected_map=(2, -1, 0),
            extra_args={"input_shapes": [(1, 8, 16), (4, 8, 16)]},
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_both_need_broadcast_10(self, mock_platform):
        """
        Feature: Element-wise operator with broadcasting
        Description: Both inputs need broadcast on different dimensions
        Expectation: Output merges sharding
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        y_layout = _build_layout(mesh, (Replicate(), Shard(1), Replicate()), 3)

        self._run_scenario(
            x_layout, y_layout,
            expected_map=(2, 1, -1),
            extra_args={"input_shapes": [(4, 1, 16), (1, 8, 16)]},
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_complex_broadcast_11(self, mock_platform):
        """
        Feature: Element-wise operator with broadcasting
        Description: Complex broadcast with multiple dimensions
        Expectation: Output merges sharding correctly
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)
        y_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(2)), 3)

        self._run_scenario(
            x_layout, y_layout,
            expected_map=(2, 1, 0),
            extra_args={"input_shapes": [(4, 8, 1), (1, 1, 16)]},
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scalar_broadcast_12(self, mock_platform):
        """
        Feature: Element-wise operator with broadcasting
        Description: Scalar broadcasts to sharded tensor
        Expectation: Output sharded
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 0)
        y_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        self._run_scenario(
            x_layout, y_layout,
            expected_map=(2, 1, 0),
            extra_args={"input_shapes": [(), (4, 8, 16)]},
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_partial_with_replicate_13(self, mock_platform):
        """
        Feature: Element-wise operator with Partial
        Description: Partial input with replicated input
        Expectation: Output Partial
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        x_layout.set_partial_by_dev_axis("dp", "sum")
        y_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)

        extra_args = {"input_shapes": [(4, 8, 16), (4, 8, 16)]}
        output_layout = self.op_with_partial.infer_layout((x_layout, y_layout), extra_args)

        assert output_layout.partial[mesh.axis_index("dp")] == "sum"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_partial_with_partial_same_14(self, mock_platform):
        """
        Feature: Element-wise operator with Partial
        Description: Both inputs have same Partial operation
        Expectation: Output Partial with same operation
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        x_layout.set_partial_by_dev_axis("dp", "sum")
        y_layout = copy.deepcopy(x_layout)
        y_layout.set_partial_by_dev_axis("dp", "sum")

        extra_args = {"input_shapes": [(4, 8, 16), (4, 8, 16)]}
        output_layout = self.op_with_partial.infer_layout((x_layout, y_layout), extra_args)

        assert output_layout.partial[mesh.axis_index("dp")] == "sum"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_partial_with_partial_different_15(self, mock_platform):
        """
        Feature: Element-wise operator with Partial
        Description: Both inputs have different Partial operations
        Expectation: Raise ValueError
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        x_layout.set_partial_by_dev_axis("dp", "sum")
        y_layout = copy.deepcopy(x_layout)
        y_layout.set_partial_by_dev_axis("dp", "max")

        extra_args = {"input_shapes": [(4, 8, 16), (4, 8, 16)]}

        with self.assertRaisesRegex(ValueError, "partial operations should be same"):
            self.op_with_partial.infer_layout((x_layout, y_layout), extra_args=extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_shard_with_partial_conflict_16(self, mock_platform):
        """
        Feature: Element-wise operator with Shard and Partial
        Description: One input sharded, other Partial on same device axis
        Expectation: Raise ValueError
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        y_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        y_layout.set_partial_by_dev_axis("dp", "sum")

        extra_args = {"input_shapes": [(4, 8, 16), (4, 8, 16)]}

        with self.assertRaisesRegex(ValueError, "Shard and Partial should not coexist on same device axis"):
            self.op_with_partial.infer_layout((x_layout, y_layout), extra_args=extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_partial_broadcasts_to_sharded_17(self, mock_platform):
        """
        Feature: Element-wise operator with Partial broadcasting
        Description: Partial input broadcasts to sharded input
        Expectation: Raise ValueError
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        x_layout.set_partial_by_dev_axis("dp", "sum")
        y_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        extra_args = {"input_shapes": [(1, 8, 16), (4, 8, 16)]}

        with self.assertRaisesRegex(ValueError, "Shard and Partial should not coexist on same device axis"):
            self.op_with_partial.infer_layout((x_layout, y_layout), extra_args=extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_multiple_inputs_18(self, mock_platform):
        """
        Feature: Element-wise operator
        Description: 3 inputs
        Expectation: Run Successfully
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        y_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        z_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)

        extra_args = {"input_shapes": [(4, 8, 16), (1, 8, 16), (1, 8, 16)]}
        expected_map = (2, -1, -1)

        output_layout = self.op.infer_layout((x_layout, y_layout, z_layout), extra_args=extra_args)
        got_map = output_layout.tensor_map
        assert got_map == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_output_shard_partial_conflict_19(self, mock_platform):
        """
        Feature: Element-wise operator
        Description: Output would have both Shard and Partial on same axis
        Expectation: Raise ValueError
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        y_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        y_layout.set_partial_by_dev_axis("dp", "sum")

        extra_args = {"input_shapes": [(4, 8, 16), (4, 8, 16)]}

        with self.assertRaisesRegex(ValueError, "Shard and Partial should not coexist on same device axis"):
            self.op_with_partial.infer_layout((x_layout, y_layout), extra_args=extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_incompatible_broadcast_shapes_20(self, mock_platform):
        """
        Feature: Element-wise operator with broadcasting
        Description: Incompatible shapes for broadcasting
        Expectation: Raise ValueError
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        y_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)

        extra_args = {"input_shapes": [(4, 8, 16), (4, 7, 16)]}

        with self.assertRaisesRegex(ValueError, "cannot be broadcast"):
            self.op_with_partial.infer_layout((x_layout, y_layout), extra_args=extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_1d_to_3d_broadcast_21(self, mock_platform):
        """
        Feature: Element-wise operator with different dimension counts
        Description: 1D tensor broadcasts to 3D tensor
        Expectation: Output maintains 3D sharding
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(0)), 1)
        y_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)

        self._run_scenario(
            x_layout, y_layout,
            expected_map=(2, 1, 0),
            extra_args={"input_shapes": [(16,), (4, 8, 16)]},
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_no_input_shapes_provided_22(self, mock_platform):
        """
        Feature: Element-wise operator
        Description: No input shapes provided
        Expectation: Raise ValueError
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        y_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)

        with self.assertRaisesRegex(ValueError, "cannot infer layout without shapes"):
            self.op.infer_layout((x_layout, y_layout), extra_args={})


if __name__ == "__main__":
    unittest.main()
