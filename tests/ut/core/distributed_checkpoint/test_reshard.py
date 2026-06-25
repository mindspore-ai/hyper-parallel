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
"""UT for :mod:`hyper_parallel.core.distributed_checkpoint.reshard`."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from hyper_parallel.core.distributed_checkpoint.reshard import (
    ReshardHandler,
    check_layout,
    infer_intersection,
    infer_slice_area_by_rank,
    rank_id_to_dev_id_list,
)


def _layout_with_attrs(mesh_shape, tensor_map, rank_list):
    """Build a layout mock satisfying :func:`check_layout` attribute requirements."""
    layout = MagicMock()
    layout.mesh_shape = mesh_shape
    layout._tensor_map = tensor_map
    layout._rank_list = rank_list
    layout.to_dict.return_value = {
        "mesh_shape": mesh_shape,
        "tensor_map": tensor_map,
        "rank_list": rank_list,
    }
    return layout


class TestReshard(unittest.TestCase):
    """Tests for reshard geometry helpers and ReshardHandler."""

    def test_rank_id_to_dev_id_list_2x2_mesh(self):
        """
        Feature: rank_id_to_dev_id_list conversion.
        Description: Map rank ids 0-3 on a (2, 2) device mesh.
        Expectation: Device coordinates match row-major mesh ordering.
        """
        mesh = (2, 2)
        self.assertEqual(rank_id_to_dev_id_list(mesh, 0), [0, 0])
        self.assertEqual(rank_id_to_dev_id_list(mesh, 1), [0, 1])
        self.assertEqual(rank_id_to_dev_id_list(mesh, 2), [1, 0])
        self.assertEqual(rank_id_to_dev_id_list(mesh, 3), [1, 1])

    def test_infer_intersection_overlapping_1d(self):
        """
        Feature: infer_intersection for overlapping 1-D ranges.
        Description: Intersect [0, 4) with [2, 6).
        Expectation: Returns ((2, 4),).
        """
        area_a = ((0, 4),)
        area_b = ((2, 6),)
        self.assertEqual(infer_intersection(area_a, area_b), ((2, 4),))

    def test_infer_intersection_disjoint_returns_none(self):
        """
        Feature: infer_intersection for disjoint ranges.
        Description: Intersect [0, 2) with [3, 5).
        Expectation: Returns None when there is no overlap.
        """
        self.assertIsNone(infer_intersection(((0, 2),), ((3, 5),)))

    def test_infer_intersection_dimension_mismatch_raises(self):
        """
        Feature: infer_intersection dimension validation.
        Description: Pass areas with different numbers of dimensions.
        Expectation: ValueError is raised.
        """
        with self.assertRaises(ValueError):
            infer_intersection(((0, 2),), ((0, 2), (0, 2)))

    def test_infer_slice_area_by_rank_row_shard(self):
        """
        Feature: infer_slice_area_by_rank for row-wise sharding.
        Description: 8x4 tensor sharded on dim 0 across 2 devices (mesh 2x1).
        Expectation: Rank 0 owns rows [0, 4), rank 1 owns [4, 8).
        """
        mesh_shape = (1, 2)
        tensor_map = (0, -1)
        full_shape = (8, 4)
        self.assertEqual(
            infer_slice_area_by_rank(mesh_shape, tensor_map, 0, full_shape),
            ((0, 4), (0, 4)),
        )
        self.assertEqual(
            infer_slice_area_by_rank(mesh_shape, tensor_map, 1, full_shape),
            ((4, 8), (0, 4)),
        )

    def test_check_layout_none_is_noop(self):
        """
        Feature: check_layout with falsy layout.
        Description: Pass None as layout.
        Expectation: Returns without raising.
        """
        check_layout(None, "layout")

    def test_check_layout_missing_attr_raises(self):
        """
        Feature: check_layout required attributes.
        Description: Layout object missing _rank_list attribute.
        Expectation: ValueError mentions the missing attribute.
        """
        bad = SimpleNamespace(mesh_shape=(2,), _tensor_map=(0,))
        with self.assertRaises(ValueError) as ctx:
            check_layout(bad, "bad_layout")
        self.assertIn("_rank_list", str(ctx.exception))

    def test_reshard_handler_replicate_to_shard(self):
        """
        Feature: ReshardHandler infer_all_tensor_offset and get_real_tensor.
        Description: Reshard from replicated (None from_layout) to 2-way row shard.
        Expectation: Target rank receives the correct slice assembled from source data.
        """
        to_layout = _layout_with_attrs((1, 2), (0, -1), [0, 1])
        handler = ReshardHandler(
            param_name="weight",
            full_shape=(4, 2),
            from_layout=None,
            to_layout=to_layout,
            to_rank_id=1,
        )
        local_map = handler.infer_all_tensor_offset()
        self.assertIn(0, local_map)
        full = np.arange(8, dtype=np.float32).reshape(4, 2)
        result = handler.get_real_tensor({0: full[2:4, :]})
        np.testing.assert_array_equal(result, full[2:4, :])

    def test_reshard_handler_both_layouts_none_raises(self):
        """
        Feature: ReshardHandler constructor validation.
        Description: Pass None for both from_layout and to_layout.
        Expectation: ValueError is raised during initialization.
        """
        with self.assertRaises(ValueError):
            ReshardHandler("w", (2, 2), None, None, 0)

    def test_reshard_handler_missing_slice_raises(self):
        """
        Feature: ReshardHandler.get_real_tensor input validation.
        Description: Omit a required source rank slice from from_tensor_map.
        Expectation: ValueError mentions the missing rank.
        """
        to_layout = _layout_with_attrs((1, 2), (0, -1), [0, 1])
        handler = ReshardHandler("w", (4, 2), None, to_layout, to_rank_id=1)
        handler.infer_all_tensor_offset()
        with self.assertRaises(ValueError) as ctx:
            handler.get_real_tensor({99: np.zeros((2, 2), dtype=np.float32)})
        self.assertIn("Missing slice", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
