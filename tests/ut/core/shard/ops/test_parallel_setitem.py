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
"""parallel_setitem test"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_setitem import SetItemDistributedOp
from hyper_parallel.core.shard.ops.parallel_getitem import (
    GetItemDistributedOp,
    _BOOL_MASK,
    _key_cache_descriptor,
)
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP,
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

setitem_op = SetItemDistributedOp("__setitem__")
getitem_op = GetItemDistributedOp("__getitem__")


class TestSetItemDistributedOp(unittest.TestCase):
    """Unit tests for SetItemDistributedOp."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
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

    def _make_2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2 (dp, mp) mesh."""
        self._setup_mock_platform(mock_platform, world_size=4)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "mp"))

    def _make_mock_dtensor(self, layout, to_local_return="local_tensor", shape=(8, 10)):
        """Create a mock DTensor with given layout."""
        mock_tensor = MagicMock()
        mock_tensor.layout = layout
        mock_tensor.to_local.return_value = to_local_return
        mock_tensor.shape = shape
        return mock_tensor

    # ===== infer_layout tests (success) =====

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_setitem_scalar_replicated(self, mock_platform):
        """
        Feature: setitem with scalar value on replicated tensor.
        Description: x[1:3] = 0.0 on fully replicated tensor.
        Expectation: Output layout equals self layout (in-place);
                     scalar branch does not trigger value broadcast validation.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        key_desc, kind = _key_cache_descriptor(slice(1, 3))
        cache_values = [self_layout, key_desc, (8, 10), kind, None, (2, 10)]

        result = setitem_op.infer_layout(cache_values)
        self.assertIsNone(result[1])
        self.assertEqual(len(result[0]), 1)
        out_layout = result[0][0]
        self.assertIs(out_layout, self_layout)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_setitem_tensor_replicated(self, mock_platform):
        """
        Feature: setitem with plain tensor value on replicated tensor.
        Description: x[1:3] = torch.zeros(2, 5) on replicated tensor.
        Expectation: Output layout equals self layout (in-place);
                     plain tensor triggers _validate_value_broadcast.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        key_desc, kind = _key_cache_descriptor(slice(1, 3))
        value_desc = ("plain_tensor", (2, 5), "torch.float32")
        cache_values = [self_layout, key_desc, (8, 5), kind, value_desc, (2, 5)]

        result = setitem_op.infer_layout(cache_values)
        self.assertIsNone(result[1])
        self.assertEqual(len(result[0]), 1)
        out_layout = result[0][0]
        self.assertIs(out_layout, self_layout)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_setitem_shard_kept_dim(self, mock_platform):
        """
        Feature: setitem with DTensor value on sharded dim that is NOT indexed.
        Description: self shard dim1 (dp), x[1:3, :] = v(2,5),
                     v has matching sharding ("None","dp").
        Expectation: Output layout equals self layout (in-place);
                     DTensor value layout matches expected output layout.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, ("None", "dp"), 2)
        val_layout = _build_layout(mesh, ("None", "dp"), 2)

        key_desc, kind = _key_cache_descriptor((slice(1, 3), slice(None)))
        value_desc = ("dtensor", val_layout, (2, 5))
        cache_values = [self_layout, key_desc, (8, 5), kind, value_desc, (2, 5)]

        result = setitem_op.infer_layout(cache_values)
        self.assertIsNone(result[1])
        self.assertEqual(len(result[0]), 1)
        out_layout = result[0][0]
        self.assertIs(out_layout, self_layout)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_setitem_advanced_single_list(self, mock_platform):
        """
        Feature: setitem with advanced list index and DTensor value.
        Description: x[[0, 2]] = v(2, 10) on fully replicated tensor.
        Expectation: Output layout equals self layout (in-place);
                     advanced index on dim0, value layout matches expected output.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        val_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        key_desc, kind = _key_cache_descriptor([0, 2])
        value_desc = ("dtensor", val_layout, (2, 10))
        cache_values = [self_layout, key_desc, (8, 10), kind, value_desc, (2, 10)]

        result = setitem_op.infer_layout(cache_values)
        self.assertIsNone(result[1])
        self.assertEqual(len(result[0]), 1)
        out_layout = result[0][0]
        self.assertIs(out_layout, self_layout)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_setitem_inplace_view_propagation(self, mock_platform):
        """
        Feature: setitem value layout must match getitem output layout.
        Description: For basic indexing x[1:3] = v, v's layout should equal
                     getitem's inferred output layout to preserve sharding.
        Expectation: setitem accepts value whose layout matches getitem output;
                     output layout equals self layout (in-place).
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        key_desc, kind = _key_cache_descriptor(slice(1, 3))
        getitem_cache = [self_layout, key_desc, (8, 5), kind]
        getitem_result = getitem_op.infer_layout(getitem_cache)
        getitem_out_layout = getitem_result[0][0]

        val_layout = getitem_out_layout
        value_desc = ("dtensor", val_layout, (2, 5))
        setitem_cache = [self_layout, key_desc, (8, 5), kind, value_desc, (2, 5)]

        result = setitem_op.infer_layout(setitem_cache)
        self.assertIs(result[0][0], self_layout)
        self.assertEqual(val_layout.alias_tensor_map, getitem_out_layout.alias_tensor_map)

    # ===== Error cases =====

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.shard.ops.parallel_setitem.platform")
    def test_setitem_bool_mask(self, mock_op_platform, mock_dt_platform):
        """
        Feature: Error on BoolTensor mask LHS.
        Description: x[x > 5] = 0 should error.
        Expectation: ValueError with "boolean-mask".
        """
        mesh = self._make_2x2_mesh(mock_dt_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        mock_bool = MagicMock()
        mock_op_platform.bool = mock_bool

        key_desc = (("bool_mask", (8, 10)),)
        cache_values = [self_layout, key_desc, (8, 10), _BOOL_MASK, None, None]

        with self.assertRaises(ValueError) as ctx:
            setitem_op.infer_layout(cache_values)
        self.assertIn("boolean-mask", str(ctx.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.shard.ops.parallel_setitem.platform")
    def test_setitem_shard_dim_write(self, mock_op_platform, mock_dt_platform):
        """
        Feature: Error when writing to sharded dimension.
        Description: x[2] = 0 with shard on dim0.
        Expectation: ValueError with "non-replicate dim".
        """
        mesh = self._make_2x2_mesh(mock_dt_platform)
        self_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        mock_bool = MagicMock()
        mock_op_platform.bool = mock_bool

        key_desc, kind = _key_cache_descriptor(2)
        cache_values = [self_layout, key_desc, (8, 10), kind, None, (10,)]

        with self.assertRaises(ValueError) as ctx:
            setitem_op.infer_layout(cache_values)
        self.assertIn("non-replicate dim 0", str(ctx.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.shard.ops.parallel_setitem.platform")
    def test_setitem_value_layout_mismatch(self, mock_op_platform, mock_dt_platform):
        """
        Feature: Error when DTensor value layout mismatches LHS expected layout.
        Description: x[:, 1:3] = v where v has different sharding.
        Expectation: ValueError with "value layout mismatch".
        """
        mesh = self._make_2x2_mesh(mock_dt_platform)
        self_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        val_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        mock_bool = MagicMock()
        mock_op_platform.bool = mock_bool

        key_desc, kind = _key_cache_descriptor((slice(None), slice(1, 3)))
        value_desc = ("dtensor", val_layout, (8, 2))
        cache_values = [self_layout, key_desc, (8, 10), kind, value_desc, (8, 2)]

        with self.assertRaises(ValueError) as ctx:
            setitem_op.infer_layout(cache_values)
        self.assertIn("value layout mismatch", str(ctx.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.shard.ops.parallel_setitem.platform")
    def test_setitem_value_shape_mismatch(self, mock_op_platform, mock_dt_platform):
        """
        Feature: Error when value shape cannot broadcast to LHS slice.
        Description: x[1:3] = zeros(3, 5) with LHS shape (2, 5).
        Expectation: ValueError with "cannot broadcast".
        """
        mesh = self._make_2x2_mesh(mock_dt_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        mock_bool = MagicMock()
        mock_op_platform.bool = mock_bool

        key_desc, kind = _key_cache_descriptor(slice(1, 3))
        value_desc = ("plain_tensor", (3, 5), "torch.float32")
        cache_values = [self_layout, key_desc, (8, 5), kind, value_desc, (2, 5)]

        with self.assertRaises(ValueError) as ctx:
            setitem_op.infer_layout(cache_values)
        self.assertIn("cannot broadcast", str(ctx.exception))

    # ===== get_expand_impl =====

    def test_get_expand_impl_returns_none(self):
        """
        Feature: get_expand_impl returns None.
        Description: __setitem__ doesn't override get_expand_impl.
        Expectation: Returns None.
        """
        # Since `get_expand_impl` is not overridden, it returns None by default.
        # Verified once here; other tests do not need to repeat this check.
        assert setitem_op.get_expand_impl(None, None, None) is None, (
            f"get_expand_impl should return None, "
            f"got {setitem_op.get_expand_impl(None, None, None)}"
        )

    # ===== wrap_output =====

    def test_wrap_output_returns_none(self):
        """
        Feature: wrap_output returns None for __setitem__.
        Description: __setitem__ is in-place with no return value.
        Expectation: Returns None.
        """
        self.assertIsNone(setitem_op.wrap_output(None, None))
