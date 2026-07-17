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
"""parallel_index_add test"""
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np

from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _LAYOUT_CACHE, _build_layout
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.core.shard._op_dispatch import OpDispatcher
from hyper_parallel.core.shard.ops.parallel_index_add import IndexAddDistributedOp
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


def _infer_one(distributed_op, cache_values):
    """Run infer_layout(cache_values) and return the single output layout."""
    output_layouts, _ = distributed_op.infer_layout(cache_values)
    return output_layouts[0]


class _MockDTensor:
    """Small DTensor-like object for preprocess tests."""

    def __init__(self, layout: Any, local_value: Any = None) -> None:
        """Initialize the mock with a layout and optional local value."""
        self.layout = layout
        self._layout = layout
        self._local_value = local_value if local_value is not None else object()

    def to_local(self) -> Any:
        """Return the value representing the local tensor."""
        return self._local_value


class TestParallelIndexAdd(unittest.TestCase):
    """Unit tests for IndexAddDistributedOp."""

    def setUp(self) -> None:
        """Set up isolated mesh/layout caches before each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()
        self.op = IndexAddDistributedOp("index_add")

    def tearDown(self) -> None:
        """Clean up global caches after each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    @staticmethod
    def _setup_mock_platform(mock_platform, world_size=4):
        """Configure common mock platform attributes."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )

    def _make_2x2_mesh(self, mock_platform):
        """Set up mock platform and return a 2x2 mesh."""
        self._setup_mock_platform(mock_platform)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2),
            mesh_dim_names=("dp", "tp"),
            init_backend=False,
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_preserves_alpha_and_out(self, mock_platform):
        """
        Feature: IndexAdd argument preprocessing.
        Description: Localize DTensor inputs and keep alpha/out keyword arguments.
        Expectation: Local args, local kwargs, and layout cache values are correct.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        index_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        source_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        out_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        input_tensor = _MockDTensor(input_layout, local_value="input")
        index = _MockDTensor(index_layout, local_value="index")
        source = _MockDTensor(source_layout, local_value="source")
        out = _MockDTensor(out_layout, local_value="out")

        local_args, local_kwargs, cache_values = self.op.preprocess(
            (input_tensor, 0, index, source), {"alpha": 0.5, "out": out}
        )

        self.assertEqual(local_args, ("input", 0, "index", "source"))
        self.assertEqual(local_kwargs, {"alpha": 0.5, "out": "out"})
        self.assertEqual(cache_values, [input_layout, index_layout, source_layout, out_layout, True, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_replicated_target_dim_with_non_target_shard(self, mock_platform):
        """
        Feature: IndexAdd layout inference.
        Description: Infer layout when the target dimension is replicated and a non-target axis is sharded.
        Expectation: Output layout matches input layout.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        index_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        source_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        output_layout = _infer_one(self.op, [input_layout, index_layout, source_layout, None, False, 0])

        self.assertEqual(output_layout.tensor_map, input_layout.tensor_map)
        self.assertEqual(output_layout.alias_tensor_map, input_layout.alias_tensor_map)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_negative_dim(self, mock_platform):
        """
        Feature: IndexAdd negative dimension handling.
        Description: Infer layout with a negative target dimension.
        Expectation: Negative dimension is normalized and output layout matches input layout.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        index_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        source_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        output_layout = _infer_one(self.op, [input_layout, index_layout, source_layout, None, False, -1])

        self.assertEqual(output_layout.tensor_map, input_layout.tensor_map)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_inplace_op_uses_same_layout_rule(self, mock_platform):
        """
        Feature: In-place IndexAdd layout inference.
        Description: Infer layout for index_add_ with the same input/source sharding as index_add.
        Expectation: Output layout matches input layout.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        op = IndexAddDistributedOp("index_add_")
        input_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        index_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        source_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        output_layout = _infer_one(op, [input_layout, index_layout, source_layout, None, False, 0])

        self.assertEqual(output_layout.tensor_map, input_layout.tensor_map)

    def test_restore_mutable_dtensor_result_for_index_add_and_out(self):
        """
        Feature: IndexAdd mutable result restoration.
        Description: Restore wrapper results for in-place index_add_ and index_add with out.
        Expectation: Dispatcher returns the original input or out DTensor wrapper.
        """
        input_dtensor = _MockDTensor(layout=object())
        out_dtensor = _MockDTensor(layout=object())
        output = object()

        restored_inplace = OpDispatcher._restore_mutable_dtensor_result(
            "index_add_", (input_dtensor,), {}, output
        )
        restored_out = OpDispatcher._restore_mutable_dtensor_result(
            "index_add", (), {"out": out_dtensor}, output
        )

        self.assertIs(restored_inplace, input_dtensor)
        self.assertIs(restored_out, out_dtensor)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reject_partial_inputs(self, mock_platform):
        """
        Feature: IndexAdd partial input validation.
        Description: Infer layout when input layout has Partial status.
        Expectation: ValueError is raised for Partial status.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Partial(), Replicate()), 2)
        index_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        source_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "Partial status"):
            _infer_one(self.op, [input_layout, index_layout, source_layout, None, False, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reject_missing_source_layout(self, mock_platform):
        """
        Feature: IndexAdd source layout validation.
        Description: Infer layout when source is not a DTensor layout.
        Expectation: ValueError is raised for missing source layout.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        index_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)

        with self.assertRaisesRegex(ValueError, "source must be a DTensor"):
            _infer_one(self.op, [input_layout, index_layout, None, None, False, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_allows_plain_replicated_index(self, mock_platform):
        """
        Feature: Plain index support.
        Description: Infer layout when every rank supplies the same ordinary 1-D index Tensor.
        Expectation: Output layout matches input layout.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        source_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

        output_layout = _infer_one(self.op, [input_layout, None, source_layout, None, False, 0])

        self.assertEqual(output_layout.tensor_map, input_layout.tensor_map)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_caches_plain_out_as_flag_only(self, mock_platform):
        """
        Feature: Plain out cache representation.
        Description: Preprocess an ordinary out Tensor without storing the object in cache_values.
        Expectation: out_layout is None and out_provided is True.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        index_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        source_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        input_tensor = _MockDTensor(input_layout, local_value="input")
        index = _MockDTensor(index_layout, local_value="index")
        source = _MockDTensor(source_layout, local_value="source")
        out = object()

        _, local_kwargs, cache_values = self.op.preprocess((input_tensor, 0, index, source), {"out": out})

        self.assertIs(local_kwargs["out"], out)
        self.assertEqual(cache_values, [input_layout, index_layout, source_layout, None, True, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reject_missing_out_layout(self, mock_platform):
        """
        Feature: IndexAdd out layout validation.
        Description: Infer layout when out is a plain local object.
        Expectation: ValueError is raised for missing out layout.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        index_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        source_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "out must be a DTensor"):
            _infer_one(self.op, [input_layout, index_layout, source_layout, None, True, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reject_out_layout_mismatch(self, mock_platform):
        """
        Feature: IndexAdd out layout validation.
        Description: Infer layout when out layout differs from input layout.
        Expectation: ValueError is raised for mismatched out layout.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        index_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        source_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        out_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "out layout"):
            _infer_one(self.op, [input_layout, index_layout, source_layout, out_layout, True, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reject_non_integer_dim(self, mock_platform):
        """
        Feature: IndexAdd dimension validation.
        Description: Infer layout with a non-integer dimension argument.
        Expectation: ValueError is raised for invalid dimension type.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        index_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        source_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "dim should be an integer"):
            _infer_one(self.op, [input_layout, index_layout, source_layout, None, False, "0"])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reject_dim_out_of_bounds(self, mock_platform):
        """
        Feature: IndexAdd dimension validation.
        Description: Infer layout with a dimension outside input rank bounds.
        Expectation: ValueError is raised for out-of-bounds dimension.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        index_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        source_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "out of bounds"):
            _infer_one(self.op, [input_layout, index_layout, source_layout, None, False, 2])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reject_source_rank_mismatch(self, mock_platform):
        """
        Feature: IndexAdd source rank validation.
        Description: Infer layout when input and source ranks differ.
        Expectation: ValueError is raised for source rank mismatch.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        index_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        source_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)

        with self.assertRaisesRegex(ValueError, "same number of dimensions"):
            _infer_one(self.op, [input_layout, index_layout, source_layout, None, False, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reject_index_rank_not_one(self, mock_platform):
        """
        Feature: IndexAdd index rank validation.
        Description: Infer layout when index is not a one-dimensional DTensor.
        Expectation: ValueError is raised for invalid index rank.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        index_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        source_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "index must be a 1-D DTensor"):
            _infer_one(self.op, [input_layout, index_layout, source_layout, None, False, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reject_sharded_index(self, mock_platform):
        """
        Feature: IndexAdd index sharding validation.
        Description: Infer layout when index is sharded.
        Expectation: ValueError is raised for sharded index layout.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        index_layout = _build_layout(mesh, (Shard(0), Replicate()), 1)
        source_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "sharded index is not supported"):
            _infer_one(self.op, [input_layout, index_layout, source_layout, None, False, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reject_sharded_input_target_dim(self, mock_platform):
        """
        Feature: IndexAdd target dimension validation.
        Description: Infer layout when input target dimension is sharded.
        Expectation: ValueError is raised for sharded input target dimension.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        index_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        source_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "sharded input dimension"):
            _infer_one(self.op, [input_layout, index_layout, source_layout, None, False, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reject_sharded_source_target_dim(self, mock_platform):
        """
        Feature: IndexAdd source target dimension validation.
        Description: Infer layout when source target dimension is sharded.
        Expectation: ValueError is raised for sharded source target dimension.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        index_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        source_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        with self.assertRaisesRegex(ValueError, "source target dimension"):
            _infer_one(self.op, [input_layout, index_layout, source_layout, None, False, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reject_non_target_source_sharding_mismatch(self, mock_platform):
        """
        Feature: IndexAdd non-target axis validation.
        Description: Infer layout when input and source differ on a non-target axis.
        Expectation: ValueError is raised for non-target sharding mismatch.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        index_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        source_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "same sharding"):
            _infer_one(self.op, [input_layout, index_layout, source_layout, None, False, 0])


if __name__ == "__main__":
    unittest.main()
