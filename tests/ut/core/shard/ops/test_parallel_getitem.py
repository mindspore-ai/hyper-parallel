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
"""parallel_getitem test"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_getitem import (
    GetItemDistributedOp,
    _key_cache_descriptor,
    _descriptor_to_expanded_actions,
    _BASIC,
    _ADVANCED,
    _BOOL_MASK,
)
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP,
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS
from hyper_parallel.core.shard._op_dispatch import _OP_DISPATCHER

getitem_op = GetItemDistributedOp("__getitem__")


class TestGetItemDistributedOp(unittest.TestCase):
    """Unit tests for GetItemDistributedOp."""

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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, mp, cp) mesh for 3D tensor tests."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "mp", "cp"))

    def _make_mock_dtensor(self, layout, to_local_return="local_tensor"):
        """Create a mock DTensor with given layout."""
        mock_tensor = MagicMock()
        mock_tensor.layout = layout
        mock_tensor.to_local.return_value = to_local_return
        mock_tensor.shape = (8, 10)
        return mock_tensor

    # ===== classify_key tests =====

    def test_classify_basic_int(self):
        """Test _key_cache_descriptor with single int key.

        x[2] on 2D tensor.
        """
        key_desc, kind = _key_cache_descriptor(2)
        self.assertEqual(kind, _BASIC)
        actions = _descriptor_to_expanded_actions(key_desc, 2)
        self.assertEqual(len(actions), 2)  # int + implicit trailing full slice
        self.assertEqual(actions[0][0], "int")
        self.assertEqual(actions[1][0], "slice")

    def test_classify_basic_slice(self):
        """Test _key_cache_descriptor with single slice key.

        x[1:3] on 2D tensor.
        """
        key_desc, kind = _key_cache_descriptor(slice(1, 3))
        self.assertEqual(kind, _BASIC)
        actions = _descriptor_to_expanded_actions(key_desc, 2)
        self.assertEqual(len(actions), 2)  # slice + implicit trailing full slice
        self.assertEqual(actions[0][0], "slice")
        self.assertEqual(actions[1][0], "slice")

    def test_classify_basic_newaxis(self):
        """Test _key_cache_descriptor with None (newaxis).

        x[None] on 2D tensor.
        """
        key_desc, kind = _key_cache_descriptor(None)
        self.assertEqual(kind, _BASIC)
        actions = _descriptor_to_expanded_actions(key_desc, 2)
        self.assertEqual(len(actions), 3)  # newaxis + 2 implicit trailing full slices
        self.assertEqual(actions[0][0], "newaxis")

    def test_classify_basic_ellipsis(self):
        """Test _key_cache_descriptor with Ellipsis."""
        key_desc, kind = _key_cache_descriptor(Ellipsis)
        self.assertEqual(kind, _BASIC)
        actions = _descriptor_to_expanded_actions(key_desc, 2)
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0][0], "slice")
        self.assertEqual(actions[1][0], "slice")

    def test_classify_basic_tuple(self):
        """Test _key_cache_descriptor with mixed tuple."""
        key_desc, kind = _key_cache_descriptor((0, slice(None), None))
        self.assertEqual(kind, _BASIC)
        actions = _descriptor_to_expanded_actions(key_desc, 2)
        self.assertEqual(len(actions), 3)
        self.assertEqual(actions[0][0], "int")
        self.assertEqual(actions[1][0], "slice")

    def test_classify_advanced_list(self):
        """Test _key_cache_descriptor with list key."""
        key_desc, kind = _key_cache_descriptor([0, 2])
        self.assertEqual(kind, _ADVANCED)

    # ===== Positive cases: infer_layout (plan §3.5 order) =====

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_basic_int_index_replicated(self, mock_platform):
        """
        Feature: Basic int index on fully replicated tensor.
        Description: x[2] on 2D replicated input removes dim 0.
        Expectation: Output layout has 1 dim, replicated.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor(2)
        cache_values = [self_layout, key_desc, global_shape, kind]

        result = getitem_op.infer_layout(cache_values)
        out_layout = result[0][0]
        self.assertEqual(out_layout.alias_tensor_map, ("None",))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_basic_slice_replicated(self, mock_platform):
        """
        Feature: Basic slice on fully replicated tensor.
        Description: x[1:3] on 2D replicated input.
        Expectation: Output keeps 2 dims, all replicated.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor(slice(1, 3))
        cache_values = [self_layout, key_desc, global_shape, kind]

        result = getitem_op.infer_layout(cache_values)
        out_layout = result[0][0]
        self.assertEqual(out_layout.alias_tensor_map, ("None", "None"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_basic_ellipsis(self, mock_platform):
        """
        Feature: Ellipsis on replicated tensor.
        Description: x[..., 1:3] with 2D input.
        Expectation: Output maintains 2 dims.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor((Ellipsis, slice(1, 3)))
        cache_values = [self_layout, key_desc, global_shape, kind]

        result = getitem_op.infer_layout(cache_values)
        out_layout = result[0][0]
        self.assertEqual(out_layout.alias_tensor_map, ("None", "None"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_basic_newaxis_front(self, mock_platform):
        """
        Feature: Newaxis at front on replicated tensor.
        Description: x[None] prepends a dimension.
        Expectation: Output has 3 dims, first is replicated.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor(None)
        cache_values = [self_layout, key_desc, global_shape, kind]

        result = getitem_op.infer_layout(cache_values)
        out_layout = result[0][0]
        self.assertEqual(out_layout.alias_tensor_map, ("None", "None", "None"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_basic_newaxis_middle(self, mock_platform):
        """
        Feature: Newaxis in middle on replicated tensor.
        Description: x[:, None, :] inserts a dimension.
        Expectation: Output has 3 dims, newaxis replicated.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor((slice(None), None, slice(None)))
        cache_values = [self_layout, key_desc, global_shape, kind]

        result = getitem_op.infer_layout(cache_values)
        out_layout = result[0][0]
        self.assertEqual(out_layout.alias_tensor_map, ("None", "None", "None"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_basic_mixed(self, mock_platform):
        """
        Feature: Mixed basic indexing with int, full slice, Ellipsis, and newaxis.
        Description: x[0, ::1, ..., None] on 3D replicated input.
        Expectation: int removes dim0, slice keeps dim1, Ellipsis fills dim2, None adds dim.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        global_shape = (8, 10, 6)

        key_desc, kind = _key_cache_descriptor((0, slice(None, None, 1), Ellipsis, None))
        cache_values = [self_layout, key_desc, global_shape, kind]

        result = getitem_op.infer_layout(cache_values)
        out_layout = result[0][0]
        # int(0) removes dim0, ::1 keeps dim1, Ellipsis fills dim2, None adds dim
        self.assertEqual(out_layout.alias_tensor_map, ("None", "None", "None"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_basic_tuple_form(self, mock_platform):
        """
        Feature: Tuple of ints indexing produces scalar output.
        Description: x[(0, 1)] on 2D replicated input.
        Expectation: Both dims removed, output is scalar (0-d).
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor((0, 1))
        cache_values = [self_layout, key_desc, global_shape, kind]

        result = getitem_op.infer_layout(cache_values)
        out_layout = result[0][0]
        # Both dims removed by int indices → 0-d scalar
        self.assertEqual(out_layout.alias_tensor_map, ())

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_basic_empty_tuple(self, mock_platform):
        """
        Feature: Empty tuple indexing returns unchanged tensor.
        Description: x[()] is equivalent to full slice on all dims.
        Expectation: Output layout equals input layout.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor(())
        cache_values = [self_layout, key_desc, global_shape, kind]

        result = getitem_op.infer_layout(cache_values)
        out_layout = result[0][0]
        self.assertEqual(out_layout.alias_tensor_map, self_layout.alias_tensor_map)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_basic_shard_keep_dim0(self, mock_platform):
        """
        Feature: Shard dim0 kept when indexing dim1 only.
        Description: x[:, 1:3] with shard on dim0.
        Expectation: Output keeps shard on dim0.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor((slice(None), slice(1, 3)))
        cache_values = [self_layout, key_desc, global_shape, kind]

        result = getitem_op.infer_layout(cache_values)
        out_layout = result[0][0]
        self.assertEqual(out_layout.alias_tensor_map, ("dp", "None"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_basic_shard_keep_dim1(self, mock_platform):
        """
        Feature: Shard dim1 kept when indexing dim0 only.
        Description: x[1:3, :] with shard on dim1.
        Expectation: Output keeps shard on dim1.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, ("None", "dp"), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor((slice(1, 3), slice(None)))
        cache_values = [self_layout, key_desc, global_shape, kind]

        result = getitem_op.infer_layout(cache_values)
        out_layout = result[0][0]
        self.assertEqual(out_layout.alias_tensor_map, ("None", "dp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_advanced_single_list_replicated(self, mock_platform):
        """
        Feature: Advanced indexing with single list on replicated tensor.
        Description: x[[0, 2]] on 2D replicated input.
        Expectation: Advanced index dim replaced by replicated list dim.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor([0, 2])
        cache_values = [self_layout, key_desc, global_shape, kind]

        result = getitem_op.infer_layout(cache_values)
        out_layout = result[0][0]
        # Advanced index on dim0: broadcast dim (replicated) + dim1 kept (replicated)
        self.assertEqual(out_layout.alias_tensor_map, ("None", "None"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_advanced_keep_shard_outside(self, mock_platform):
        """
        Feature: Advanced indexing keeps shard on non-indexed dims.
        Description: x[[0, 2]] with shard on dim1.
        Expectation: Output dim0 is replicated (advanced), dim1 keeps shard.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, ("None", "dp"), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor([0, 2])
        cache_values = [self_layout, key_desc, global_shape, kind]

        result = getitem_op.infer_layout(cache_values)
        out_layout = result[0][0]
        # Advanced index on dim0 → broadcast dim (replicated) + dim1 (kept shard)
        self.assertEqual(out_layout.alias_tensor_map, ("None", "dp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_advanced_paired_indices_replicated(self, mock_platform):
        """
        Feature: Paired advanced indices on replicated tensor.
        Description: x[[0, 1], [2, 3]] on 2D replicated input.
        Expectation: Both dims advanced-indexed, broadcast shape replaces both → 1 dim.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor(([0, 1], [2, 3]))
        cache_values = [self_layout, key_desc, global_shape, kind]

        result = getitem_op.infer_layout(cache_values)
        out_layout = result[0][0]
        # Both dims replaced by broadcast shape (2,) → 1 dim replicated
        self.assertEqual(out_layout.alias_tensor_map, ("None",))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_advanced_multi_d_index(self, mock_platform):
        """
        Feature: Multi-dimensional LongTensor advanced index on replicated tensor.
        Description: x[ind_2x2] where ind_2x2 is a 2D LongTensor of shape (2, 2).
        Expectation: Index shape (2,2) replaces dim0, dim1 kept → 3 dims replicated.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        global_shape = (8, 10)

        # Simulate _key_cache_descriptor output for 2D LongTensor key
        key_desc = (("idx_tensor", (2, 2)),)
        cache_values = [self_layout, key_desc, global_shape, _ADVANCED]

        result = getitem_op.infer_layout(cache_values)
        out_layout = result[0][0]
        # Index shape (2,2) replaces dim0 → 2 broadcast dims + dim1 kept → 3 dims
        self.assertEqual(out_layout.alias_tensor_map, ("None", "None", "None"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_advanced_consecutive_with_basic(self, mock_platform):
        """
        Feature: Consecutive advanced indices mixed with basic slice on 3D tensor.
        Description: x[:, [0, 2], [1, 3]] on 3D replicated input.
        Expectation: Slice dim0 kept, consecutive advanced on dim1-2 replaced by broadcast.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        global_shape = (8, 10, 6)

        key_desc, kind = _key_cache_descriptor((slice(None), [0, 2], [1, 3]))
        cache_values = [self_layout, key_desc, global_shape, kind]

        result = getitem_op.infer_layout(cache_values)
        out_layout = result[0][0]
        # dim0 kept (slice), dim1-2 replaced by broadcast (2,) → 2 dims
        self.assertEqual(out_layout.alias_tensor_map, ("None", "None"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_advanced_split_with_basic(self, mock_platform):
        """
        Feature: Non-consecutive advanced indices mixed with basic slice on 3D tensor.
        Description: x[[0, 1], :, [2, 3]] on 3D replicated input.
        Expectation: Broadcast dims placed at position 0, then basic slice dim kept.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        global_shape = (8, 10, 6)

        key_desc, kind = _key_cache_descriptor(([0, 1], slice(None), [2, 3]))
        cache_values = [self_layout, key_desc, global_shape, kind]

        result = getitem_op.infer_layout(cache_values)
        out_layout = result[0][0]
        # Non-consecutive → broadcast (2,) at front + dim1 kept → 2 dims
        self.assertEqual(out_layout.alias_tensor_map, ("None", "None"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_zero_size_slice(self, mock_platform):
        """
        Feature: Zero-size slice on replicated tensor.
        Description: x[2:2] produces zero-size dim0, but layout is unchanged.
        Expectation: Output keeps 2 dims replicated.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor(slice(2, 2))
        cache_values = [self_layout, key_desc, global_shape, kind]

        result = getitem_op.infer_layout(cache_values)
        out_layout = result[0][0]
        # Non-full slice but dim0 is replicate (valid), dim1 kept
        self.assertEqual(out_layout.alias_tensor_map, ("None", "None"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_zero_d_long_tensor_index(self, mock_platform):
        """
        Feature: 0-D LongTensor index treated as int on replicated tensor.
        Description: x[tensor(2)] is equivalent to x[2] — removes dim0.
        Expectation: Output has 1 dim replicated.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        global_shape = (8, 10)

        # 0-D LongTensor is treated as int by _key_cache_descriptor
        key_desc = (("int", 2),)
        cache_values = [self_layout, key_desc, global_shape, _BASIC]

        result = getitem_op.infer_layout(cache_values)
        out_layout = result[0][0]
        # 0-D long tensor → int(2) → removes dim0 → 1 dim
        self.assertEqual(out_layout.alias_tensor_map, ("None",))

    # ===== Error cases: infer_layout (plan §3.5 order) =====

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_int_on_sharded_dim(self, mock_platform):
        """
        Feature: Error when int index on sharded dimension.
        Description: x[2] with shard on dim0.
        Expectation: ValueError with "non-replicate dim".
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor(2)
        cache_values = [self_layout, key_desc, global_shape, kind]

        with self.assertRaises(ValueError) as ctx:
            getitem_op.infer_layout(cache_values)
        self.assertIn("non-replicate dim 0", str(ctx.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_partial_slice_on_sharded_dim(self, mock_platform):
        """
        Feature: Error when non-full slice on sharded dimension.
        Description: x[1:3] with shard on dim0.
        Expectation: ValueError with "non-replicate dim".
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor(slice(1, 3))
        cache_values = [self_layout, key_desc, global_shape, kind]

        with self.assertRaises(ValueError) as ctx:
            getitem_op.infer_layout(cache_values)
        self.assertIn("non-replicate dim 0", str(ctx.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_advanced_on_sharded_dim(self, mock_platform):
        """
        Feature: Error when advanced index on sharded dimension.
        Description: x[[0, 2]] with shard on dim0.
        Expectation: ValueError with "non-replicate dim".
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor([0, 2])
        cache_values = [self_layout, key_desc, global_shape, kind]

        with self.assertRaises(ValueError) as ctx:
            getitem_op.infer_layout(cache_values)
        self.assertIn("non-replicate dim 0", str(ctx.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_advanced_index_tensor_sharded(self, mock_platform):
        """
        Feature: Error when advanced index tensor is sharded.
        Description: x[ind] where ind is a sharded DTensor (not replicated).
        Expectation: ValueError with "index tensor must be replicated".
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        global_shape = (8, 10)

        # idx_tensor with alias ("dp",) → sharded, not replicated
        key_desc = (("idx_tensor", (2,), ("dp",)),)
        cache_values = [self_layout, key_desc, global_shape, _ADVANCED]

        with self.assertRaises(ValueError) as ctx:
            getitem_op.infer_layout(cache_values)
        self.assertIn("index tensor must be replicated", str(ctx.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.shard.ops.parallel_getitem._is_bool_tensor", return_value=True)
    def test_bool_mask_1d(self, mock_is_bool, mock_platform):
        """
        Feature: Error when 1D BoolTensor mask is used.
        Description: x[mask1d] where mask1d is a 1-dimensional BoolTensor.
        Expectation: ValueError with "boolean-mask".
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        global_shape = (8, 10)

        # Full flow: _key_cache_descriptor detects 1D BoolTensor → _BOOL_MASK
        mask1d = MagicMock()
        key_desc, kind = _key_cache_descriptor(mask1d)
        self.assertEqual(kind, _BOOL_MASK)
        cache_values = [self_layout, key_desc, global_shape, kind]

        with self.assertRaises(ValueError) as ctx:
            getitem_op.infer_layout(cache_values)
        self.assertIn("boolean-mask", str(ctx.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.shard.ops.parallel_getitem._is_bool_tensor", return_value=True)
    def test_bool_mask_full(self, mock_is_bool, mock_platform):
        """
        Feature: Error when full BoolTensor mask (e.g. from comparison) is used.
        Description: x[x > 5] where the comparison produces a BoolTensor.
        Expectation: ValueError with "boolean-mask".
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        global_shape = (8, 10)

        # Full flow: _key_cache_descriptor detects BoolTensor in tuple → _BOOL_MASK
        mask_full = MagicMock()
        key_desc, kind = _key_cache_descriptor((mask_full,))
        self.assertEqual(kind, _BOOL_MASK)
        cache_values = [self_layout, key_desc, global_shape, kind]

        with self.assertRaises(ValueError) as ctx:
            getitem_op.infer_layout(cache_values)
        self.assertIn("boolean-mask", str(ctx.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_slice_step_2(self, mock_platform):
        """
        Feature: Error when slice step is not 1 or None.
        Description: x[::2] with step=2.
        Expectation: ValueError with "slice step".
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor(slice(None, None, 2))
        cache_values = [self_layout, key_desc, global_shape, kind]

        with self.assertRaises(ValueError) as ctx:
            getitem_op.infer_layout(cache_values)
        self.assertIn("slice step", str(ctx.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_negative_step(self, mock_platform):
        """
        Feature: Error when slice step is negative.
        Description: x[::-1].
        Expectation: ValueError with "slice step".
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor(slice(None, None, -1))
        cache_values = [self_layout, key_desc, global_shape, kind]

        with self.assertRaises(ValueError) as ctx:
            getitem_op.infer_layout(cache_values)
        self.assertIn("slice step", str(ctx.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_int_out_of_range(self, mock_platform):
        """
        Feature: Error when int index is out of range.
        Description: x[100] with ndim=2, dim0 size=8.
        Expectation: ValueError with "out of range".
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        global_shape = (8, 10)

        key_desc, kind = _key_cache_descriptor(100)
        cache_values = [self_layout, key_desc, global_shape, kind]

        with self.assertRaises(ValueError) as ctx:
            getitem_op.infer_layout(cache_values)
        self.assertIn("out of range", str(ctx.exception))

    def test_unsupported_key_type(self):
        """
        Feature: Error when key contains an unsupported type.
        Description: x[object()] where key is not int/slice/None/Ellipsis/list/Tensor.
        Expectation: ValueError with "unsupported index type".
        """
        with self.assertRaises(ValueError) as ctx:
            _key_cache_descriptor(object())
        self.assertIn("unsupported index type", str(ctx.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_partial_input(self, mock_platform):
        """
        Feature: Error when input has Partial status.
        Description: __getitem__ with partial input.
        Expectation: ValueError with "Partial".
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        self_layout._partial = ["sum", None, None]

        key_desc, kind = _key_cache_descriptor(slice(None))
        cache_values = [self_layout, key_desc, (8, 10), kind]

        with self.assertRaises(ValueError) as ctx:
            getitem_op.infer_layout(cache_values)
        self.assertIn("Partial", str(ctx.exception))

    # ===== get_expand_impl =====

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_get_expand_impl_returns_none(self, mock_platform):
        """
        Feature: get_expand_impl returns None.
        Description: __getitem__ doesn't need expand impl.
        Expectation: Returns None.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        key_desc, kind = _key_cache_descriptor(slice(None))
        cache_values = [self_layout, key_desc, (8, 10), kind]
        infer_result = ((self_layout,), None)

        assert getitem_op.get_expand_impl(None, infer_result, cache_values) is None, (
            f"get_expand_impl should return None, "
            f"got {getitem_op.get_expand_impl(None, infer_result, cache_values)}"
        )

    # ===== preprocess tests =====

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.shard.ops.parallel_getitem.platform")
    def test_preprocess_basic(self, mock_op_platform, mock_dt_platform):
        """
        Feature: preprocess for basic indexing.
        Description: Verify preprocess normalizes args and builds cache_values.
        Expectation: Returns local_args, empty kwargs, and cache_values with 4 elements.
        """
        mesh = self._make_2x2_mesh(mock_dt_platform)
        self_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        mock_bool = MagicMock()
        mock_op_platform.bool = mock_bool

        mock_tensor = self._make_mock_dtensor(self_layout, "local_tensor_data")
        local_args, local_kwargs, cache_values = getitem_op.preprocess(
            (mock_tensor, 2), {}
        )
        self.assertEqual(len(local_args), 2)
        self.assertEqual(local_kwargs, {})
        self.assertEqual(len(cache_values), 4)
        self.assertEqual(cache_values[3], _BASIC)


class TestDispatchWhitelist(unittest.TestCase):
    """Verify __getitem__ and __setitem__ are NOT in the dispatch whitelist."""

    def test_getitem_not_in_whitelist(self):
        """
        Feature: __getitem__ not in whitelist.
        Description: __getitem__ should be routed through distributed op dispatch.
        Expectation: Not in _OP_DISPATCHER.whitelist.
        """
        self.assertNotIn("__getitem__", _OP_DISPATCHER.whitelist)

    def test_setitem_not_in_whitelist(self):
        """
        Feature: __setitem__ not in whitelist.
        Description: __setitem__ should be routed through distributed op dispatch.
        Expectation: Not in _OP_DISPATCHER.whitelist.
        """
        self.assertNotIn("__setitem__", _OP_DISPATCHER.whitelist)

    def test_descriptor_get_still_in_whitelist(self):
        """
        Feature: __get__ still in whitelist.
        Description: __get__ is a descriptor protocol method, not __getitem__.
        Expectation: Still in _OP_DISPATCHER.whitelist.
        """
        self.assertIn("__get__", _OP_DISPATCHER.whitelist)
