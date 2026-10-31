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
"""Unit tests for TrilDistributedOp."""
import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _LAYOUT_CACHE, _build_layout
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.shard.ops.parallel_ops_register import get_distributed_op
from hyper_parallel.core.shard.ops.parallel_tril import TrilDistributedOp
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestTrilDistributedOp(unittest.TestCase):
    """Unit tests for distributed Tril layout and expansion behavior."""

    def setUp(self) -> None:
        """Clear distributed caches before each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
        """Clear distributed caches after each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    @staticmethod
    def _get_op(name="tril"):
        return get_distributed_op(name)

    @staticmethod
    def _make_mesh(mock_platform, mesh_shape, mesh_dim_names, rank=0):
        mock_platform.get_rank.return_value = rank
        mock_platform.get_world_size.return_value = math.prod(mesh_shape)
        mock_platform.tensor_to_numpy.side_effect = (
            lambda tensor: tensor.numpy() if hasattr(tensor, "numpy") else np.array(tensor)
        )
        mock_platform.platform_type = MagicMock()
        return init_device_mesh(
            device_type="cpu",
            mesh_shape=mesh_shape,
            mesh_dim_names=mesh_dim_names,
            init_backend=False,
        )

    @staticmethod
    def _mock_dtensor(layout, shape=(6, 8)):
        tensor = MagicMock()
        tensor.layout = layout
        tensor.shape = shape
        tensor.to_local.return_value = MagicMock()
        return tensor

    @staticmethod
    def _run_expand(op, layout, shape, diagonal):
        """Infer a layout and invoke its local expansion once."""
        cache_values = [layout, shape]
        output_layouts, _ = op.infer_layout(cache_values)
        func = MagicMock(return_value=object())
        expand_fn = op.get_expand_impl(func, (output_layouts, None), cache_values)
        if expand_fn is None:
            return None, func
        local_input = object()
        expand_fn(local_input, diagonal)
        return expand_fn, func

    def test_tril_registered_for_both_platforms(self):
        """PyTorch tril and MindSpore TrilExt resolve to TrilDistributedOp."""
        self.assertIsInstance(self._get_op("tril"), TrilDistributedOp)
        self.assertIsInstance(self._get_op("TrilExt"), TrilDistributedOp)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_preprocess_default(self, mock_platform):
        """Default preprocess keeps diagonal positional and caches shape."""
        mesh = self._make_mesh(mock_platform, (2,), ("tp",))
        layout = _build_layout(mesh, (Replicate(),), 2)
        input_tensor = self._mock_dtensor(layout)

        local_args, local_kwargs, cache_values = self._get_op().preprocess((input_tensor,), {})

        self.assertEqual(local_args, (input_tensor.to_local.return_value, 0))
        self.assertEqual(local_kwargs, {})
        self.assertEqual(cache_values, [layout, (6, 8)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_preprocess_accepts_input_keyword(self, mock_platform):
        """Tril preprocess accepts the native input= keyword."""
        mesh = self._make_mesh(mock_platform, (2,), ("tp",))
        layout = _build_layout(mesh, (Replicate(),), 2)
        input_tensor = self._mock_dtensor(layout)

        local_args, local_kwargs, cache_values = self._get_op().preprocess(
            (), {"input": input_tensor, "diagonal": 2}
        )

        self.assertEqual(local_args, (input_tensor.to_local.return_value, 2))
        self.assertEqual(local_kwargs, {})
        self.assertEqual(cache_values, [layout, (6, 8)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_preprocess_mindspore_routing(self, mock_platform):
        """MindSpore TrilExt keeps input and diagonal as positional arguments."""
        mesh = self._make_mesh(mock_platform, (2,), ("tp",))
        layout = _build_layout(mesh, (Replicate(),), 2)
        input_tensor = self._mock_dtensor(layout)

        local_args, local_kwargs, cache_values = self._get_op("TrilExt").preprocess(
            (input_tensor,), {"diagonal": -1}
        )

        self.assertEqual(local_args, (input_tensor.to_local.return_value, -1))
        self.assertEqual(local_kwargs, {})
        self.assertEqual(cache_values, [layout, (6, 8)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_preprocess_cache_excludes_diagonal(self, mock_platform):
        """Diagonal does not enter cache_values while global shape does."""
        mesh = self._make_mesh(mock_platform, (2,), ("tp",))
        layout = _build_layout(mesh, (Replicate(),), 2)
        input_tensor = self._mock_dtensor(layout)

        cache_one = self._get_op().preprocess((input_tensor, 1), {})[2]
        cache_two = self._get_op().preprocess((input_tensor, -3), {})[2]

        self.assertEqual(cache_one, [layout, (6, 8)])
        self.assertEqual(cache_one, cache_two)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_all_replicated(self, mock_platform):
        """Replicated matrix preserves layout and needs no expansion."""
        mesh = self._make_mesh(mock_platform, (2, 2), ("dp", "tp"))
        layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        cache_values = [layout, (6, 8)]

        output_layouts, _ = self._get_op().infer_layout(cache_values)

        self.assertEqual(output_layouts[0].alias_tensor_map, layout.alias_tensor_map)
        self.assertIsNot(output_layouts[0], layout)
        self.assertIsNone(self._get_op().get_expand_impl(None, (output_layouts, None), cache_values))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_data_parallel(self, mock_platform):
        """Batch-only sharding preserves layout and needs no diagonal adjustment."""
        mesh = self._make_mesh(mock_platform, (2, 2), ("dp", "tp"))
        layout = _build_layout(mesh, (Shard(0), Replicate()), 3)
        cache_values = [layout, (4, 6, 8)]

        output_layouts, _ = self._get_op().infer_layout(cache_values)

        self.assertEqual(output_layouts[0].alias_tensor_map, ("dp", "None", "None"))
        self.assertIsNone(self._get_op().get_expand_impl(None, (output_layouts, None), cache_values))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_row_sharded_negative_dim(self, mock_platform):
        """Shard(-2) adjusts diagonal by the row shard's global offset."""
        mesh = self._make_mesh(mock_platform, (2,), ("tp",), rank=1)
        layout = _build_layout(mesh, (Shard(-2),), 2)

        expand_fn, func = self._run_expand(self._get_op(), layout, (6, 8), 0)

        self.assertIsNotNone(expand_fn)
        func.assert_called_once_with(unittest.mock.ANY, 3)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_col_sharded_negative_dim(self, mock_platform):
        """Shard(-1) adjusts diagonal by the negative column global offset."""
        mesh = self._make_mesh(mock_platform, (2,), ("tp",), rank=1)
        layout = _build_layout(mesh, (Shard(-1),), 2)

        expand_fn, func = self._run_expand(self._get_op(), layout, (6, 8), 0)

        self.assertIsNotNone(expand_fn)
        func.assert_called_once_with(unittest.mock.ANY, -4)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_row_col_sharded(self, mock_platform):
        """Independent row and column mesh coordinates compose additively."""
        mesh = self._make_mesh(mock_platform, (2, 2), ("row_tp", "col_tp"), rank=3)
        layout = _build_layout(mesh, (Shard(-2), Shard(-1)), 2)

        expand_fn, func = self._run_expand(self._get_op(), layout, (6, 8), -1)

        self.assertIsNotNone(expand_fn)
        func.assert_called_once_with(unittest.mock.ANY, -2)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_hybrid_batch_row(self, mock_platform):
        """Hybrid DP plus row sharding adjusts only the row offset."""
        mesh = self._make_mesh(mock_platform, (2, 2), ("dp", "tp"), rank=1)
        layout = _build_layout(mesh, (Shard(0), Shard(-2)), 3)

        expand_fn, func = self._run_expand(self._get_op(), layout, (4, 6, 8), 1)

        self.assertIsNotNone(expand_fn)
        func.assert_called_once_with(unittest.mock.ANY, 4)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_positive_diagonal_runtime_reuse(self, mock_platform):
        """One cached row-shard closure reads each positive runtime diagonal."""
        mesh = self._make_mesh(mock_platform, (2,), ("tp",), rank=1)
        layout = _build_layout(mesh, (Shard(0),), 2)
        cache_values = [layout, (6, 8)]
        output_layouts, _ = self._get_op().infer_layout(cache_values)
        func = MagicMock(return_value=object())
        expand_fn = self._get_op().get_expand_impl(func, (output_layouts, None), cache_values)

        expand_fn(object(), 1)
        expand_fn(object(), 5)

        self.assertEqual([call.args[1] for call in func.call_args_list], [4, 8])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_negative_diagonal_runtime_reuse(self, mock_platform):
        """One cached column-shard closure reads each negative runtime diagonal."""
        mesh = self._make_mesh(mock_platform, (2,), ("tp",), rank=1)
        layout = _build_layout(mesh, (Shard(1),), 2)
        cache_values = [layout, (6, 8)]
        output_layouts, _ = self._get_op().infer_layout(cache_values)
        func = MagicMock(return_value=object())
        expand_fn = self._get_op().get_expand_impl(func, (output_layouts, None), cache_values)

        expand_fn(object(), -1)
        expand_fn(object(), -5)

        self.assertEqual([call.args[1] for call in func.call_args_list], [-5, -9])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_diagonal_above_matrix_width(self, mock_platform):
        """Large positive diagonal retains all values after row adjustment."""
        mesh = self._make_mesh(mock_platform, (2,), ("tp",), rank=1)
        layout = _build_layout(mesh, (Shard(0),), 2)
        cache_values = [layout, (6, 8)]
        output_layouts, _ = self._get_op().infer_layout(cache_values)
        expand_fn = self._get_op().get_expand_impl(torch.tril, (output_layouts, None), cache_values)
        local_input = torch.ones(3, 8)

        result = expand_fn(local_input, 9)

        torch.testing.assert_close(result, local_input)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_diagonal_below_matrix_height(self, mock_platform):
        """Large negative diagonal clears all values after column adjustment."""
        mesh = self._make_mesh(mock_platform, (2,), ("tp",), rank=1)
        layout = _build_layout(mesh, (Shard(1),), 2)
        cache_values = [layout, (6, 8)]
        output_layouts, _ = self._get_op().infer_layout(cache_values)
        expand_fn = self._get_op().get_expand_impl(torch.tril, (output_layouts, None), cache_values)
        local_input = torch.ones(6, 4)

        result = expand_fn(local_input, -7)

        torch.testing.assert_close(result, torch.zeros_like(local_input))

    def test_tril_out_not_supported(self):
        """A non-None PyTorch out argument is rejected explicitly."""
        with self.assertRaisesRegex(ValueError, "out keyword is not supported"):
            self._get_op().preprocess((MagicMock(),), {"out": object()})

    def test_tril_missing_input_layout_rejected(self):
        """A missing input layout is rejected during layout inference."""
        input_tensor = MagicMock()
        input_tensor.layout = None
        input_tensor.shape = (6, 8)

        with self.assertRaisesRegex(ValueError, "input layout"):
            cache_values = self._get_op().preprocess((input_tensor,), {})[2]
            self._get_op().infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_partial_input_error(self, mock_platform):
        """Partial input layouts are rejected."""
        mesh = self._make_mesh(mock_platform, (2,), ("tp",))
        layout = _build_layout(mesh, (Replicate(),), 2)
        layout.set_partial_by_dev_axis("tp", "sum")

        with self.assertRaisesRegex(ValueError, "Partial status"):
            self._get_op().infer_layout([layout, (6, 8)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_one_dimensional_input_error(self, mock_platform):
        """Rank-one input layouts are rejected."""
        mesh = self._make_mesh(mock_platform, (2,), ("tp",))
        layout = _build_layout(mesh, (Replicate(),), 1)

        with self.assertRaisesRegex(ValueError, "at least 2 dimensions"):
            self._get_op().infer_layout([layout, (6,)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_shape_layout_rank_mismatch(self, mock_platform):
        """Cached shape rank must match the layout rank."""
        mesh = self._make_mesh(mock_platform, (2,), ("tp",))
        layout = _build_layout(mesh, (Replicate(),), 2)

        with self.assertRaisesRegex(ValueError, "shape rank must match layout rank"):
            self._get_op().infer_layout([layout, (6, 8, 10)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_row_multi_axis_shard_error(self, mock_platform):
        """A row dimension mapped to multiple mesh axes is rejected."""
        mesh = self._make_mesh(mock_platform, (2, 2), ("dp", "tp"))
        layout = _build_layout(mesh, (("dp", "tp"), "None"), 2)

        with self.assertRaisesRegex(ValueError, "row dimension"):
            self._get_op().infer_layout([layout, (8, 8)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_col_multi_axis_shard_error(self, mock_platform):
        """A column dimension mapped to multiple mesh axes is rejected."""
        mesh = self._make_mesh(mock_platform, (2, 2), ("dp", "tp"))
        layout = _build_layout(mesh, ("None", ("dp", "tp")), 2)

        with self.assertRaisesRegex(ValueError, "column dimension"):
            self._get_op().infer_layout([layout, (8, 8)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_non_divisible_row_shard_error(self, mock_platform):
        """A non-divisible row dimension is rejected before offset calculation."""
        mesh = self._make_mesh(mock_platform, (2,), ("tp",))
        layout = _build_layout(mesh, (Shard(-2),), 2)

        with self.assertRaisesRegex(ValueError, "row dimension size 7 should be divisible"):
            self._get_op().infer_layout([layout, (7, 8)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tril_non_divisible_col_shard_error(self, mock_platform):
        """A non-divisible column dimension is rejected before offset calculation."""
        mesh = self._make_mesh(mock_platform, (2,), ("tp",))
        layout = _build_layout(mesh, (Shard(-1),), 2)

        with self.assertRaisesRegex(ValueError, "column dimension size 7 should be divisible"):
            self._get_op().infer_layout([layout, (8, 7)])

if __name__ == "__main__":
    unittest.main()
