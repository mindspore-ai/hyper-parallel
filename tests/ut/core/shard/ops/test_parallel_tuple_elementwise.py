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
import os
import unittest
from unittest.mock import patch

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout
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


class TestTupleElementWiseDistributedOpInferLayout(unittest.TestCase):
    """
    Feature: TupleElementWiseDistributedOp.infer_layout
    Description: Returns all input layouts as output layouts (element-wise op).
    Expectation: Output equals the full input layouts tuple; empty input returns None.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_empty_layouts_returns_none(self, mock_platform):
        """Empty input layouts returns None."""
        op = TupleElementWiseDistributedOp("tuple_elementwise_empty")
        result = op.infer_layout(())
        self.assertIsNone(result)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_single_layout_returned_as_tuple(self, mock_platform):
        """Single input layout is returned unchanged as the layouts object."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 2)
        inputs = (layout,)
        op = TupleElementWiseDistributedOp("tuple_elementwise_single")
        result = op.infer_layout(inputs)
        self.assertIs(result, inputs)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_multiple_layouts_returned_as_is(self, mock_platform):
        """Multiple input layouts are returned as-is (the full tuple)."""
        mesh = _make_mesh(mock_platform, (2, 2), ("dp", "mp"))
        layout1 = _build_layout(mesh, (Replicate(), Replicate()), 2)
        layout2 = _build_layout(mesh, (Shard(0), Replicate()), 2)
        inputs = (layout1, layout2)
        op = TupleElementWiseDistributedOp("tuple_elementwise_multi")
        result = op.infer_layout(inputs)
        self.assertIs(result, inputs)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sharded_layout_returned(self, mock_platform):
        """Sharded input layouts are returned correctly."""
        mesh = _make_mesh(mock_platform, (4,), ("dp",))
        layout = _build_layout(mesh, (Shard(0),), 2)
        op = TupleElementWiseDistributedOp("tuple_elementwise_sharded")
        result = op.infer_layout((layout,))
        self.assertEqual(result, (layout,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_extra_args_ignored(self, mock_platform):
        """extra_args parameter is ignored for tuple element-wise ops."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 2)
        op = TupleElementWiseDistributedOp("tuple_elementwise_extra_args")
        result = op.infer_layout((layout,), extra_args=(42, "some_arg"))
        self.assertEqual(result, (layout,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_three_layouts_returned(self, mock_platform):
        """Three input layouts are returned as-is."""
        mesh = _make_mesh(mock_platform, (2, 2, 2), ("dp", "cp", "mp"))
        layout1 = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        layout2 = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        layout3 = _build_layout(mesh, (Replicate(), Replicate(), Shard(2)), 3)
        inputs = (layout1, layout2, layout3)
        op = TupleElementWiseDistributedOp("tuple_elementwise_three")
        result = op.infer_layout(inputs)
        self.assertIs(result, inputs)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_none_layout_in_inputs(self, mock_platform):
        """None layout in inputs is returned as part of the result."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 2)
        inputs = (layout, None)
        op = TupleElementWiseDistributedOp("tuple_elementwise_none")
        result = op.infer_layout(inputs)
        self.assertIs(result, inputs)


class TestTupleElementWiseInheritance(unittest.TestCase):
    """
    Feature: TupleElementWiseDistributedOp inherits DistributedOp
    Description: TupleElementWiseDistributedOp inherits base class behavior for
                 preprocess, get_expand_impl, wrap_output.
    Expectation: Inherited methods behave as defined in DistributedOp.
    """

    def test_preprocess_returns_none(self):
        """Inherited preprocess returns None (legacy fallback)."""
        op = TupleElementWiseDistributedOp("tuple_elementwise_preprocess")
        self.assertIsNone(op.preprocess((1, 2), {}))

    def test_get_expand_impl_returns_none(self):
        """Inherited get_expand_impl returns None."""
        op = TupleElementWiseDistributedOp("tuple_elementwise_expand")
        self.assertIsNone(op.get_expand_impl(None, None, ()))


if __name__ == "__main__":
    unittest.main()
