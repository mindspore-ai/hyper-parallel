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
"""Unit tests for element-wise distributed operators"""
import unittest
from unittest.mock import MagicMock, patch
import copy
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_elementwise import ElementWiseDistributedOp, AddDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


def _extract_test_input_shapes(extra_args):
    """Extract input shapes from extra_args dict."""
    if isinstance(extra_args, dict):
        return extra_args.get("input_shapes", None)
    return None


def _make_cache_values(layouts, extra_args=None):
    """Build cache_values for the new preprocess/infer_layout protocol."""
    return [*layouts, _extract_test_input_shapes(extra_args)]


def _infer_output_layout(op, layouts, extra_args=None):
    """Infer and return the single output layout."""
    output_layouts, extra_info = op.infer_layout(_make_cache_values(layouts, extra_args))
    assert extra_info is None
    return output_layouts[0]


def _get_expand_impl(op, func, output_layout, layouts, extra_args=None):
    """Call get_expand_impl with new-protocol infer_result and cache_values."""
    cache_values = _make_cache_values(layouts, extra_args)
    return op.get_expand_impl(func, ((output_layout,), None), cache_values)


class TestParallelElementwiseOps(unittest.TestCase):
    """Unit tests for ElementWiseDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()
        self.op = ElementWiseDistributedOp("element_wise")
        self.op_with_partial = AddDistributedOp("element_wise_with_partial")

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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, sp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "sp", "mp"),
            init_backend=False
        )

    def _run_scenario(self, x_layout, y_layout, expected_map, extra_args, flag=False):
        """Infer layout of element-wise operator"""
        if x_layout.is_partial() or y_layout.is_partial():
            op_ = self.op_with_partial
        else:
            op_ = self.op

        cache_values = [
            x_layout,
            y_layout,
            extra_args.get("input_shapes"),
        ]

        infer_result = op_.infer_layout(cache_values)
        output_layout = infer_result[0][0]

        got_map = output_layout.tensor_map
        assert got_map == expected_map, (
            f"Element-wise failed. Expected {expected_map}, got {got_map}"
        )
        impl = op_.get_expand_impl(None, infer_result, cache_values)
        if flag:
            assert impl is not None, (
                f"get_expand_impl test failed. Expected non-None, got {impl}"
            )
            assert callable(impl), "Returned impl should be a callable function"
        else:
            assert impl is None, (
                f"get_expand_impl test failed. Expected None, got {impl}"
            )

    def _run_single_input_scenario(self, x_layout, expected_map, extra_args, flag=False):
        """Infer layout of element-wise operator with single input"""
        if x_layout.is_partial():
            op_ = self.op_with_partial
        else:
            op_ = self.op

        cache_values = [
            x_layout,
            extra_args.get("input_shapes"),
        ]

        infer_result = op_.infer_layout(cache_values)
        output_layout = infer_result[0][0]

        got_map = output_layout.tensor_map
        assert got_map == expected_map, (
            f"Element-wise failed. Expected {expected_map}, got {got_map}"
        )
        impl = op_.get_expand_impl(None, infer_result, cache_values)
        if flag:
            assert impl is not None, (
                f"get_expand_impl test failed. Expected non-None, got {impl}"
            )
            assert callable(impl), "Returned impl should be a callable function"
        else:
            assert impl is None, (
                f"get_expand_impl test failed. Expected None, got {impl}"
            )

    def test_preprocess_unwraps_dtensors_and_builds_cache_values(self):
        """
        Feature: Element-wise preprocess
        Description: DTensor-like inputs are converted to local tensors and cached with layout/shape.
        Expectation: local args/kwargs use local tensors and cache_values carries layouts first.
        """
        x_layout = MagicMock()
        y_layout = MagicMock()
        x_local = MagicMock()
        y_local = MagicMock()

        x_tensor = MagicMock()
        x_tensor.layout = x_layout
        x_tensor.shape = (4, 8)
        x_tensor.to_local.return_value = x_local
        x_tensor._layout = x_layout

        y_tensor = MagicMock()
        y_tensor.layout = y_layout
        y_tensor.shape = (4, 8)
        y_tensor.to_local.return_value = y_local
        y_tensor._layout = y_layout

        local_args, local_kwargs, cache_values = self.op.preprocess((x_tensor,), {"other": y_tensor})

        assert local_args == (x_local,)
        assert local_kwargs == {"other": y_local}
        assert cache_values[0] is x_layout
        assert cache_values[1] is y_layout
        assert cache_values[2] == [(4, 8), (4, 8)]

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
            self.op.infer_layout(_make_cache_values((x_layout,), {"input_shapes": [(4, 8, 16)]}))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_single_input_replicate_1(self, mock_platform):
        """
        Feature: Element-wise operator with single input
        Description: Single input with all dimensions replicated
        Expectation: Output keeps input layout
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        x_layout._partial = [None] * len(x_layout._partial)

        self._run_single_input_scenario(
            x_layout,
            expected_map=(-1, -1, -1),
            extra_args={"input_shapes": [(4, 8, 16)]},
            flag=False
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
            flag=False
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

        output_layout = _infer_output_layout(
            self.op_with_partial, (x_layout,), {"input_shapes": [(4, 8, 16)]}
        )

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
            flag=False
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
            flag=False
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
            flag=False
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
            self.op.infer_layout(_make_cache_values((x_layout, y_layout), extra_args))

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
            self.op.infer_layout(_make_cache_values((x_layout, y_layout), extra_args))

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
            flag=False
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
            flag=False
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
            flag=False
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
            flag=False
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
        y_layout = copy.deepcopy(_build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3))
        y_layout._partial = [None] * len(y_layout._partial)

        extra_args = {"input_shapes": [(4, 8, 16), (4, 8, 16)]}
        output_layout = _infer_output_layout(self.op_with_partial, (x_layout, y_layout), extra_args)

        assert output_layout.partial[mesh.axis_index("dp")] == "sum"

        impl = _get_expand_impl(self.op_with_partial, None, output_layout, (x_layout, y_layout), extra_args)
        assert impl is not None, (
            f"get_expand_impl test failed. Expected non-None, got {impl}"
        )
        assert callable(impl), "Returned impl should be a callable function"

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
        output_layout = _infer_output_layout(self.op_with_partial, (x_layout, y_layout), extra_args)

        assert output_layout.partial[mesh.axis_index("dp")] == "sum"

        impl = _get_expand_impl(self.op_with_partial, None, output_layout, (x_layout, y_layout), extra_args)
        assert impl is None, (
            f"get_expand_impl test failed. Expected None, got {impl}"
        )

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
            self.op_with_partial.infer_layout(_make_cache_values((x_layout, y_layout), extra_args))

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
            self.op_with_partial.infer_layout(_make_cache_values((x_layout, y_layout), extra_args))

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
            self.op_with_partial.infer_layout(_make_cache_values((x_layout, y_layout), extra_args))

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

        output_layout = _infer_output_layout(self.op, (x_layout, y_layout, z_layout), extra_args)
        got_map = output_layout.tensor_map
        assert got_map == expected_map

        assert _get_expand_impl(self.op, None, output_layout,
                                (x_layout, y_layout, z_layout), extra_args) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"""got {_get_expand_impl(self.op, None, output_layout,
                                    (x_layout, y_layout, z_layout), extra_args)}"""
        )

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
            self.op_with_partial.infer_layout(_make_cache_values((x_layout, y_layout), extra_args))

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
            self.op_with_partial.infer_layout(_make_cache_values((x_layout, y_layout), extra_args))

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
            flag=False
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
            self.op.infer_layout(_make_cache_values((x_layout, y_layout), {}))


class TestParallelArithmetic(unittest.TestCase):
    """Unit tests for ElementWiseDistributedOp (arithmetic ops)."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        """Clean up after each test method."""
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

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "mp"),
            init_backend=False
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_add_layout_hybrid_parallel(self, mock_platform):
        """
        Feature: add hybrid parallel
        Description: Data parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        op = ElementWiseDistributedOp("Add")

        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)
        w_placements = (Shard(0), Shard(1))
        w_layout = _build_layout(mesh, w_placements, 2)

        extra_args = {"input_shapes": [(4, 16), (4, 16)]}
        output_layout = _infer_output_layout(op, (x_layout, w_layout), extra_args)
        expected_map = ("dp", "mp")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.alias_tensor_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        impl = _get_expand_impl(op, None, output_layout, (x_layout, w_layout), extra_args)
        assert impl is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {impl}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_add_layout_broadcast(self, mock_platform):
        """
        Feature: add hybrid parallel
        Description: Data parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        op = ElementWiseDistributedOp("Add")

        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)
        w_placements = (Shard(0), Replicate())
        w_layout = _build_layout(mesh, w_placements, 2)
        extra_args = {"input_shapes": [(4, 16), (4, 16)]}
        output_layout = _infer_output_layout(op, (x_layout, w_layout), extra_args)
        expected_map = ("dp", "mp")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.alias_tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_mul_layout_broadcast(self, mock_platform):
        """
        Feature: mul hybrid parallel
        Description: Data parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        op = ElementWiseDistributedOp("Mul")

        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)
        w_placements = (Shard(0), Replicate())
        w_layout = _build_layout(mesh, w_placements, 2)
        extra_args = {"input_shapes": [(4, 16), (4, 16)]}
        output_layout = _infer_output_layout(op, (x_layout, w_layout), extra_args)
        expected_map = ("dp", "mp")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.alias_tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sub_layout_broadcast(self, mock_platform):
        """
        Feature: sub hybrid parallel
        Description: Data parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        op = ElementWiseDistributedOp("Sub")

        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)
        w_placements = (Shard(0), Replicate())
        w_layout = _build_layout(mesh, w_placements, 2)
        extra_args = {"input_shapes": [(4, 16), (4, 16)]}
        output_layout = _infer_output_layout(op, (x_layout, w_layout), extra_args)
        expected_map = ("dp", "mp")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.alias_tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_div_layout_broadcast(self, mock_platform):
        """
        Feature: div hybrid parallel
        Description: Data parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        op = ElementWiseDistributedOp("Div")

        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)
        w_placements = (Shard(0), Replicate())
        w_layout = _build_layout(mesh, w_placements, 2)
        extra_args = {"input_shapes": [(4, 16), (4, 16)]}
        output_layout = _infer_output_layout(op, (x_layout, w_layout), extra_args)
        expected_map = ("dp", "mp")
        assert output_layout.alias_tensor_map == expected_map, (
            f"Data Parallel test failed. Expected {expected_map}, "
            f"got {output_layout.alias_tensor_map}"
        )


class TestParallelZerosLike(unittest.TestCase):
    """Unit tests for ElementWiseDistributedOp used by zeros_like."""

    def setUp(self):
        """Clear global caches and initialise platform before each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()
        self.op = ElementWiseDistributedOp("zeros_like")

    def tearDown(self):
        """Restore global state after each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _setup_mock_platform(self, mock_platform, world_size=8):
        """Configure minimal mock-platform attributes needed by init_device_mesh."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )

    def _make_2x4_mesh(self, mock_platform):
        """Return a 2x4 (dp, mp) mesh with mocked backend."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "mp"),
            init_backend=False,
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_zeros_like_data_parallel(self, mock_platform):
        """
        Feature: zeros_like data parallel
        Description: Input sharded on batch dim; output layout must match input.
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, placements, 2)

        cache_values = [x_layout, None]
        infer_result = self.op.infer_layout(cache_values)
        output_layout = infer_result[0][0]

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"zeros_like data parallel layout mismatch: "
            f"expected={expected_map}, got={output_layout.tensor_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test cases, so it is unnecessary to test its return value.
        assert self.op.get_expand_impl(None, infer_result, cache_values) is None, (
            "get_expand_impl should return None"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_zeros_like_all_replicated(self, mock_platform):
        """
        Feature: zeros_like all replicated
        Description: Fully replicated input; output must also be fully replicated.
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, placements, 2)

        cache_values = [x_layout, None]
        infer_result = self.op.infer_layout(cache_values)
        output_layout = infer_result[0][0]

        expected_map = (-1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"zeros_like all-replicated layout mismatch: "
            f"expected={expected_map}, got={output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_zeros_like_partial_input_raises(self, mock_platform):
        """
        Feature: zeros_like partial input error
        Description: Input with Partial status must be rejected.
        Expectation: ValueError is raised
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("dp", "sum")

        with self.assertRaisesRegex(ValueError, "has Partial status which is not allowed"):
            self.op.infer_layout([x_layout, None])


class TestParallelSoftplusExt(unittest.TestCase):
    """Unit tests for ElementWiseDistributedOp used by SoftplusExt."""

    def setUp(self):
        """Clear global caches and initialise platform before each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()
        self.op = ElementWiseDistributedOp("SoftplusExt")

    def tearDown(self):
        """Restore global state after each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _setup_mock_platform(self, mock_platform, world_size=8):
        """Configure minimal mock-platform attributes needed by init_device_mesh."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )

    def _make_2x4_mesh(self, mock_platform):
        """Return a 2x4 (dp, mp) mesh with mocked backend."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "mp"),
            init_backend=False,
        )

    def test_preprocess_unwraps_dtensor_for_softplus(self):
        """
        Feature: SoftplusExt preprocess
        Description: DTensor input is converted to local tensor and cached with layout/shape.
        Expectation: local args use local tensor and cache_values carries layout.
        """
        x_layout = MagicMock()
        x_local = MagicMock()
        x_tensor = MagicMock()
        x_tensor.layout = x_layout
        x_tensor.shape = (4, 8)
        x_tensor.to_local.return_value = x_local
        x_tensor._layout = x_layout

        local_args, local_kwargs, cache_values = self.op.preprocess((x_tensor,), {})

        assert local_args == (x_local,)
        assert local_kwargs == {}
        assert cache_values[0] is x_layout

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_softplus_ext_data_parallel(self, mock_platform):
        """
        Feature: SoftplusExt data parallel
        Description: Input sharded on batch dim; output layout must match input.
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, placements, 2)

        cache_values = [x_layout, None]
        infer_result = self.op.infer_layout(cache_values)
        output_layout = infer_result[0][0]

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"SoftplusExt data parallel layout mismatch: "
            f"expected={expected_map}, got={output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_softplus_ext_all_replicated(self, mock_platform):
        """
        Feature: SoftplusExt all replicated
        Description: Fully replicated input; output must also be fully replicated.
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, placements, 2)

        cache_values = [x_layout, None]
        infer_result = self.op.infer_layout(cache_values)
        output_layout = infer_result[0][0]

        expected_map = (-1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"SoftplusExt all-replicated layout mismatch: "
            f"expected={expected_map}, got={output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_softplus_ext_partial_input_raises(self, mock_platform):
        """
        Feature: SoftplusExt partial input error
        Description: Input with Partial status must be rejected.
        Expectation: ValueError is raised
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("dp", "sum")

        with self.assertRaisesRegex(ValueError, "has Partial status which is not allowed"):
            self.op.infer_layout([x_layout, None])


class TestParallelSqrt(unittest.TestCase):
    """Unit tests for ElementWiseDistributedOp used by Sqrt."""

    def setUp(self):
        """Clear global caches and initialise platform before each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()
        self.op = ElementWiseDistributedOp("Sqrt")

    def tearDown(self):
        """Restore global state after each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _setup_mock_platform(self, mock_platform, world_size=8):
        """Configure minimal mock-platform attributes needed by init_device_mesh."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )

    def _make_2x4_mesh(self, mock_platform):
        """Return a 2x4 (dp, mp) mesh with mocked backend."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "mp"),
            init_backend=False,
        )

    def test_preprocess_unwraps_dtensor_for_sqrt(self):
        """
        Feature: Sqrt preprocess
        Description: DTensor input is converted to local tensor and cached with layout/shape.
        Expectation: local args use local tensor and cache_values carries layout.
        """
        x_layout = MagicMock()
        x_local = MagicMock()
        x_tensor = MagicMock()
        x_tensor.layout = x_layout
        x_tensor.shape = (4, 8)
        x_tensor.to_local.return_value = x_local
        x_tensor._layout = x_layout

        local_args, local_kwargs, cache_values = self.op.preprocess((x_tensor,), {})

        assert local_args == (x_local,)
        assert local_kwargs == {}
        assert cache_values[0] is x_layout

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sqrt_data_parallel(self, mock_platform):
        """
        Feature: Sqrt data parallel
        Description: Input sharded on batch dim; output layout must match input.
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, placements, 2)

        cache_values = [x_layout, None]
        infer_result = self.op.infer_layout(cache_values)
        output_layout = infer_result[0][0]

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Sqrt data parallel layout mismatch: "
            f"expected={expected_map}, got={output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sqrt_all_replicated(self, mock_platform):
        """
        Feature: Sqrt all replicated
        Description: Fully replicated input; output must also be fully replicated.
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, placements, 2)

        cache_values = [x_layout, None]
        infer_result = self.op.infer_layout(cache_values)
        output_layout = infer_result[0][0]

        expected_map = (-1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Sqrt all-replicated layout mismatch: "
            f"expected={expected_map}, got={output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sqrt_model_parallel(self, mock_platform):
        """
        Feature: Sqrt model parallel
        Description: Input sharded on column dim; output layout must match input.
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, placements, 2)

        cache_values = [x_layout, None]
        infer_result = self.op.infer_layout(cache_values)
        output_layout = infer_result[0][0]

        expected_map = (-1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Sqrt model parallel layout mismatch: "
            f"expected={expected_map}, got={output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sqrt_partial_input_raises(self, mock_platform):
        """
        Feature: Sqrt partial input error
        Description: Input with Partial status must be rejected.
        Expectation: ValueError is raised
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("dp", "sum")

        with self.assertRaisesRegex(ValueError, "has Partial status which is not allowed"):
            self.op.infer_layout([x_layout, None])


if __name__ == "__main__":
    unittest.main()
