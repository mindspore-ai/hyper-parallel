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
"""Unit tests for TupleElementWiseDistributedOp"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.core.shard.ops.parallel_tuple_elementwise import TupleElementWiseDistributedOp
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


def _infer_layout(op, cache_values):
    """Infer layouts with the new preprocess/infer_layout protocol."""
    result = op.infer_layout(cache_values)
    if result is None:
        return None
    output_layouts, extra_info = result
    assert extra_info is None
    return output_layouts


class TestTupleElementWiseDistributedOpInferLayout(unittest.TestCase):
    """
    Feature: TupleElementWiseDistributedOp.infer_layout
    Description: Returns all input layouts as output layouts (element-wise op).
    Expectation: Output equals the full input layouts tuple; empty input returns None.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_empty_layouts_returns_none(self, mock_platform):
        """Empty input layouts returns None."""
        op = TupleElementWiseDistributedOp("tuple_elementwise_empty")
        result = op.infer_layout([])
        self.assertIsNone(result)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_single_layout_returned_as_tuple(self, mock_platform):
        """Single input layout is returned unchanged as the layouts object."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 2)
        op = TupleElementWiseDistributedOp("tuple_elementwise_single")
        result = _infer_layout(op, [layout])
        self.assertEqual(result, (layout,))
        self.assertIsNot(result[0], layout)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_multiple_layouts_returned_as_is(self, mock_platform):
        """Multiple input layouts are returned as-is (the full tuple)."""
        mesh = _make_mesh(mock_platform, (2, 2), ("dp", "mp"))
        layout1 = _build_layout(mesh, (Replicate(), Replicate()), 2)
        layout2 = _build_layout(mesh, (Shard(0), Replicate()), 2)
        op = TupleElementWiseDistributedOp("tuple_elementwise_multi")
        result = _infer_layout(op, [layout1, layout2])
        self.assertEqual(result, (layout1, layout2))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sharded_layout_returned(self, mock_platform):
        """Sharded input layouts are returned correctly."""
        mesh = _make_mesh(mock_platform, (4,), ("dp",))
        layout = _build_layout(mesh, (Shard(0),), 2)
        op = TupleElementWiseDistributedOp("tuple_elementwise_sharded")
        result = _infer_layout(op, [layout])
        self.assertEqual(result, (layout,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_none_layout_in_inputs(self, mock_platform):
        """None layout in inputs is returned as part of the result."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 2)
        op = TupleElementWiseDistributedOp("tuple_elementwise_none")
        result = _infer_layout(op, [layout, None])
        self.assertEqual(result, (layout, None))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_three_layouts_returned(self, mock_platform):
        """Three input layouts are returned as-is."""
        mesh = _make_mesh(mock_platform, (2, 2, 2), ("dp", "cp", "mp"))
        layout1 = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        layout2 = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        layout3 = _build_layout(mesh, (Replicate(), Replicate(), Shard(2)), 3)
        op = TupleElementWiseDistributedOp("tuple_elementwise_three")
        result = _infer_layout(op, [layout1, layout2, layout3])
        self.assertEqual(result, (layout1, layout2, layout3))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_partial_input_raises_error(self, mock_platform):
        """Partial input layout is rejected."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 2)
        layout.set_partial_by_dev_axis("dp", "sum")
        op = TupleElementWiseDistributedOp("tuple_elementwise_partial")
        with self.assertRaisesRegex(ValueError, "Partial status"):
            op.infer_layout([layout])


class TestTupleElementWiseInheritance(unittest.TestCase):
    """
    Feature: TupleElementWiseDistributedOp inherits DistributedOp
    Description: TupleElementWiseDistributedOp inherits base class behavior for
                 preprocess, get_expand_impl, wrap_output.
    Expectation: Inherited methods behave as defined in DistributedOp.
    """

    def test_preprocess_expands_tuple_and_unwraps_dtensors(self):
        """preprocess unwraps local tensors and expands tuple/list layouts for cache values."""
        layout1 = object()
        layout2 = object()
        local1 = object()
        local2 = object()
        tensor1 = MagicMock()
        tensor1._layout = layout1
        tensor1.layout = layout1
        tensor1.to_local.return_value = local1
        tensor2 = MagicMock()
        tensor2._layout = layout2
        tensor2.layout = layout2
        tensor2.to_local.return_value = local2
        plain_slot = object()

        op = TupleElementWiseDistributedOp("tuple_elementwise_preprocess")
        local_args, local_kwargs, cache_values = op.preprocess(
            ((tensor1, plain_slot, tensor2),),
            {"kw": tensor1}
        )

        self.assertEqual(local_args, ((local1, plain_slot, local2),))
        self.assertEqual(local_kwargs, {"kw": local1})
        self.assertEqual(cache_values, [layout1, None, layout2, layout1])

    def test_get_expand_impl_returns_none(self):
        """Inherited get_expand_impl returns None."""
        op = TupleElementWiseDistributedOp("tuple_elementwise_expand")
        self.assertIsNone(op.get_expand_impl(None, ((object(),), None), [object()]))


if __name__ == "__main__":
    unittest.main()
