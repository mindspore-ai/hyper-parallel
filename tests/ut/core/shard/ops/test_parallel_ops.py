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
"""Unit tests for hyper_parallel.core.shard.ops.parallel_ops (DistributedOp base class)"""
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import numpy as np

from hyper_parallel.core.dtensor.dtensor import DTensor, _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.core.shard.ops.parallel_ops import DistributedOp
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


def _make_mesh(mock_platform, mesh_shape, mesh_dim_names):
    EXISTING_COMM_GROUPS.clear()
    _DEVICE_MESH_MAP.clear()
    mock_platform.get_rank.return_value = 0
    mock_platform.get_world_size.return_value = int(np.prod(mesh_shape))
    mock_platform.tensor_to_numpy.side_effect = (
        lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
    )
    return init_device_mesh(
        device_type="npu",
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
        init_backend=False,
    )


class TestDistributedOpCheckPartialInputs(unittest.TestCase):
    """
    Feature: DistributedOp._check_partial_inputs
    Description: Raise ValueError when any input layout has partial status.
    Expectation: ValueError for partial layouts; no error for non-partial ones.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_partial_input_raises_value_error(self, mock_platform):
        """Partial layout in inputs raises ValueError."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 2)
        layout.set_partial_by_dev_axis("dp", "sum")

        op = DistributedOp("test_partial_check")
        with self.assertRaises(ValueError) as ctx:
            op._check_partial_inputs((layout,))
        self.assertIn("Partial status", str(ctx.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_non_partial_input_no_error(self, mock_platform):
        """Non-partial layout does not raise."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 2)

        op = DistributedOp("test_no_partial")
        op._check_partial_inputs((layout,))  # should not raise

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_none_layout_skipped(self, mock_platform):
        """None layout in inputs is skipped without error."""
        op = DistributedOp("test_none_layout")
        op._check_partial_inputs((None,))  # should not raise


class TestDistributedOpInferLayout(unittest.TestCase):
    """
    Feature: DistributedOp.infer_layout
    Description: Default implementation returns first input layout; raises for partial inputs.
    Expectation: Returns (layouts[0],) for valid inputs; ValueError for partial.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_returns_first_layout(self, mock_platform):
        """Default infer_layout returns (layouts[0],)."""
        mesh = _make_mesh(mock_platform, (4,), ("tp",))
        layout = _build_layout(mesh, (Replicate(),), 2)

        op = DistributedOp("test_infer_first")
        result = op.infer_layout((layout,))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], layout)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_empty_layouts_returns_none(self, mock_platform):
        """Empty layouts returns None."""
        op = DistributedOp("test_empty")
        result = op.infer_layout(())
        self.assertIsNone(result)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_partial_input_raises(self, mock_platform):
        """Partial input layout raises ValueError via _check_partial_inputs."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 2)
        layout.set_partial_by_dev_axis("dp", "sum")

        op = DistributedOp("test_partial_infer")
        with self.assertRaises(ValueError):
            op.infer_layout((layout,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_allow_partial_flag_bypasses_check(self, mock_platform):
        """When _allow_partial_inputs=True, partial inputs are allowed."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 2)
        layout.set_partial_by_dev_axis("dp", "sum")

        op = DistributedOp("test_allow_partial")
        op._allow_partial_inputs = True
        result = op.infer_layout((layout,))
        self.assertIsNotNone(result)


class TestDistributedOpPreprocess(unittest.TestCase):
    """
    Feature: DistributedOp.preprocess
    Description: Default implementation returns None (legacy dispatch fallback).
    Expectation: Always returns None regardless of input.
    """

    def test_preprocess_returns_none_for_non_dtensor(self):
        """Default preprocess returns None."""
        op = DistributedOp("test_preprocess")
        self.assertIsNone(op.preprocess((1, 2, 3), {"a": "b"}))

    def test_preprocess_returns_none_for_empty_args(self):
        """Default preprocess returns None for empty args."""
        op = DistributedOp("test_empty_preprocess")
        self.assertIsNone(op.preprocess((), {}))


class TestDistributedOpGetExpandImpl(unittest.TestCase):
    """
    Feature: DistributedOp.get_expand_impl
    Description: Default implementation returns None.
    Expectation: Always returns None.
    """

    def test_get_expand_impl_returns_none(self):
        """Default get_expand_impl returns None."""
        op = DistributedOp("test_expand_impl")
        self.assertIsNone(op.get_expand_impl(None, None, ()))


class TestDistributedOpWrapOutput(unittest.TestCase):
    """
    Feature: DistributedOp.wrap_output
    Description: Wrap local tensor outputs into DTensors using inferred layouts.
    Expectation: Single tensor → DTensor; tuple → tuple of DTensors; size mismatch → RuntimeError.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.dtensor.dtensor.platform")
    def test_single_tensor_wrapped_in_dtensor(self, mock_dtensor_platform, mock_mesh_platform):
        """Single local tensor output is wrapped as a DTensor."""
        mesh = _make_mesh(mock_mesh_platform, (1,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 1)

        mock_dtensor_platform.tensor_to_numpy.side_effect = (
            lambda t: t if isinstance(t, np.ndarray) else np.array(t)
        )

        local_t = np.array([1.0, 2.0])
        op = DistributedOp("test_wrap")

        with patch.object(DTensor, "from_local", return_value=MagicMock(spec=DTensor)) as mock_from_local:
            result = op.wrap_output(local_t, (layout,))
            mock_from_local.assert_called_once()
            self.assertIsNotNone(result)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.dtensor.dtensor.platform")
    def test_tuple_output_wraps_each_element(self, mock_dtensor_platform, mock_mesh_platform):
        """Tuple of local tensors is wrapped element-wise."""
        mesh = _make_mesh(mock_mesh_platform, (1,), ("dp",))
        layout1 = _build_layout(mesh, (Replicate(),), 1)
        layout2 = _build_layout(mesh, (Replicate(),), 1)

        mock_dtensor_platform.tensor_to_numpy.side_effect = (
            lambda t: t if isinstance(t, np.ndarray) else np.array(t)
        )

        local_t1 = np.array([1.0])
        local_t2 = np.array([2.0])
        op = DistributedOp("test_wrap_tuple")

        with patch.object(DTensor, "from_local", return_value=MagicMock(spec=DTensor)):
            result = op.wrap_output((local_t1, local_t2), (layout1, layout2))
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tuple_output_size_mismatch_raises(self, mock_platform):
        """Output tuple size != layout tuple size raises RuntimeError."""
        mesh = _make_mesh(mock_platform, (1,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 1)
        op = DistributedOp("test_wrap_mismatch")

        with self.assertRaises(RuntimeError):
            op.wrap_output((np.array([1.0]), np.array([2.0])), (layout,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_single_output_with_multiple_layouts_raises(self, mock_platform):
        """Single (non-tuple) output with multiple layouts raises RuntimeError."""
        mesh = _make_mesh(mock_platform, (1,), ("dp",))
        layout1 = _build_layout(mesh, (Replicate(),), 1)
        layout2 = _build_layout(mesh, (Replicate(),), 1)
        op = DistributedOp("test_single_multi_layout")

        with self.assertRaises(RuntimeError):
            op.wrap_output(np.array([1.0]), (layout1, layout2))


class TestDistributedOpRegistration(unittest.TestCase):
    """
    Feature: DistributedOp registration
    Description: Creating a DistributedOp instance registers it under the given name.
    Expectation: Op is retrievable via get_distributed_op.
    """

    def test_op_registered_on_creation(self):
        """DistributedOp registers itself in the global registry on __init__."""
        from hyper_parallel.core.shard.ops.parallel_ops_register import get_distributed_op
        op = DistributedOp("_ut_test_op_unique_xyz")
        retrieved = get_distributed_op("_ut_test_op_unique_xyz")
        self.assertIs(retrieved, op)


if __name__ == "__main__":
    unittest.main()
