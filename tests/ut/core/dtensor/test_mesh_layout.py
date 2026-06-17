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
"""Unit tests for hyper_parallel.core.dtensor._mesh_layout (pure logic, no external deps)."""

import unittest

import numpy as np

from hyper_parallel.core.dtensor._mesh_layout import (
    _is_int,
    _as_tuple,
    _flatten_inttuple,
    _match_structure,
    _numel,
    _contiguous_strides,
    _scale_inttuple,
    _enumerate_offsets,
    _canonicalize_axis,
    _nested_from_flat,
    _FlatLayout,
    _MeshLayout,
)


# ===========================================================================
# Utility function tests
# ===========================================================================

class TestIsInt(unittest.TestCase):
    """Tests for IsInt."""
    def test_int_returns_true(self):
        """Test int returns true."""
        self.assertTrue(_is_int(0))
        self.assertTrue(_is_int(42))
        self.assertTrue(_is_int(-1))

    def test_bool_returns_false(self):
        """Test bool returns false."""
        self.assertFalse(_is_int(True))
        self.assertFalse(_is_int(False))

    def test_float_returns_false(self):
        """Test float returns false."""
        self.assertFalse(_is_int(1.0))

    def test_str_returns_false(self):
        """Test str returns false."""
        self.assertFalse(_is_int("1"))


class TestAsTuple(unittest.TestCase):
    """Tests for AsTuple."""
    def test_int_wrapped(self):
        """Test int wrapped."""
        self.assertEqual(_as_tuple(3), (3,))

    def test_tuple_unchanged(self):
        """Test tuple unchanged."""
        self.assertEqual(_as_tuple((1, 2)), (1, 2))

    def test_nested_tuple(self):
        """Test nested tuple."""
        self.assertEqual(_as_tuple(((1, 2), 3)), ((1, 2), 3))


class TestFlattenInttuple(unittest.TestCase):
    """Tests for FlattenInttuple."""
    def test_single_int(self):
        """Test single int."""
        self.assertEqual(_flatten_inttuple(5), (5,))

    def test_flat_tuple(self):
        """Test flat tuple."""
        self.assertEqual(_flatten_inttuple((2, 3)), (2, 3))

    def test_nested_tuple(self):
        """Test nested tuple."""
        self.assertEqual(_flatten_inttuple(((2, 3), 4)), (2, 3, 4))

    def test_deeply_nested(self):
        """Test deeply nested."""
        self.assertEqual(_flatten_inttuple(((1, (2, 3)), 4)), (1, 2, 3, 4))


class TestMatchStructure(unittest.TestCase):
    """Tests for MatchStructure."""
    def test_int_int_match(self):
        """Test int int match."""
        self.assertTrue(_match_structure(2, 4))

    def test_int_tuple_no_match(self):
        """Test int tuple no match."""
        self.assertFalse(_match_structure(2, (4,)))

    def test_tuple_length_mismatch(self):
        """Test tuple length mismatch."""
        self.assertFalse(_match_structure((2, 3), (4,)))

    def test_nested_match(self):
        """Test nested match."""
        self.assertTrue(_match_structure((2, (3, 4)), (5, (6, 7))))

    def test_nested_mismatch(self):
        """Test nested mismatch."""
        self.assertFalse(_match_structure((2, (3, 4)), (5, 6)))


class TestNumel(unittest.TestCase):
    """Tests for Numel."""
    def test_single_int(self):
        """Test single int."""
        self.assertEqual(_numel(5), 5)

    def test_tuple(self):
        """Test tuple."""
        self.assertEqual(_numel((2, 3)), 6)

    def test_nested_tuple(self):
        """Test nested tuple."""
        self.assertEqual(_numel(((2, 3), 4)), 24)


class TestContiguousStrides(unittest.TestCase):
    """Tests for ContiguousStrides."""
    def test_empty(self):
        """Test empty."""
        self.assertEqual(_contiguous_strides(()), ())

    def test_1d(self):
        """Test 1d."""
        self.assertEqual(_contiguous_strides((4,)), (1,))

    def test_2d(self):
        """Test 2d."""
        self.assertEqual(_contiguous_strides((2, 3)), (3, 1))

    def test_3d(self):
        """Test 3d."""
        self.assertEqual(_contiguous_strides((2, 3, 4)), (12, 4, 1))


class TestScaleInttuple(unittest.TestCase):
    """Tests for ScaleInttuple."""
    def test_scalar(self):
        """Test scalar."""
        self.assertEqual(_scale_inttuple(3, 2), 6)

    def test_tuple(self):
        """Test tuple."""
        self.assertEqual(_scale_inttuple((2, 3), 4), (8, 12))

    def test_nested(self):
        """Test nested."""
        self.assertEqual(_scale_inttuple((2, (3, 4)), 5), (10, (15, 20)))


class TestEnumerateOffsets(unittest.TestCase):
    """Tests for EnumerateOffsets."""
    def test_scalar_shape(self):
        """Test scalar shape."""
        self.assertEqual(_enumerate_offsets(3, 2), [0, 2, 4])

    def test_multi_dim(self):
        """Test multi dim."""
        offsets = _enumerate_offsets((2, 3), (3, 1))
        self.assertEqual(sorted(offsets), [0, 1, 2, 3, 4, 5])


class TestCanonicalizeAxis(unittest.TestCase):
    """Tests for CanonicalizeAxis."""
    def test_removes_size_1(self):
        """Test removes size 1."""
        shape, stride = _canonicalize_axis((1, 4), (4, 1))
        self.assertEqual(shape, (4,))
        self.assertEqual(stride, (1,))

    def test_merges_contiguous(self):
        """Test merges contiguous."""
        shape, stride = _canonicalize_axis((2, 4), (4, 1))
        self.assertEqual(shape, (8,))
        self.assertEqual(stride, (1,))

    def test_non_contiguous_preserved(self):
        """Test non contiguous preserved."""
        shape, stride = _canonicalize_axis((2, 4), (8, 1))
        self.assertEqual(shape, (2, 4))
        self.assertEqual(stride, (8, 1))

    def test_negative_shape_raises(self):
        """Test negative shape raises."""
        with self.assertRaises(ValueError):
            _canonicalize_axis((-1, 4), (4, 1))

    def test_shape_stride_length_mismatch_raises(self):
        """Test shape stride length mismatch raises."""
        with self.assertRaises(ValueError):
            _canonicalize_axis((2, 3), (1,))

    def test_none_stride_auto_computed(self):
        """Test none stride auto computed."""
        shape, stride = _canonicalize_axis((2, 4), None)
        self.assertEqual(shape, (8,))
        self.assertEqual(stride, (1,))


class TestNestedFromFlat(unittest.TestCase):
    """Tests for NestedFromFlat."""
    def test_single_element(self):
        """Test single element."""
        self.assertEqual(_nested_from_flat((5,)), 5)

    def test_multiple_elements(self):
        """Test multiple elements."""
        self.assertEqual(_nested_from_flat((2, 3)), (2, 3))


# ===========================================================================
# _FlatLayout tests
# ===========================================================================

class TestFlatLayout(unittest.TestCase):
    """Tests for FlatLayout."""
    def test_basic_construction(self):
        """Test basic construction."""
        fl = _FlatLayout(4)
        self.assertEqual(fl.shape, (4,))
        self.assertEqual(fl.stride, (1,))

    def test_construction_with_stride(self):
        """Test construction with stride."""
        fl = _FlatLayout((2, 4), (8, 1))
        self.assertEqual(fl.shape, (2, 4))
        self.assertEqual(fl.stride, (8, 1))

    def test_canonicalization_merge(self):
        """Test canonicalization merge."""
        fl = _FlatLayout((2, 4), (4, 1))
        self.assertEqual(fl.shape, (8,))
        self.assertEqual(fl.stride, (1,))

    def test_canonicalization_removes_size_1(self):
        """Test canonicalization removes size 1."""
        fl = _FlatLayout((1, 4), (4, 1))
        self.assertEqual(fl.shape, (4,))
        self.assertEqual(fl.stride, (1,))

    def test_numel(self):
        """Test numel."""
        fl = _FlatLayout((2, 3))
        self.assertEqual(fl.numel(), 6)

    def test_numel_empty_shape(self):
        """Test numel empty shape."""
        fl = _FlatLayout(1)  # size=1 gets removed
        self.assertEqual(fl.shape, ())
        self.assertEqual(fl.numel(), 1)

    def test_cosize(self):
        """Test cosize."""
        fl = _FlatLayout((2, 3))
        self.assertEqual(fl.cosize(), 6)

    def test_cosize_non_contiguous(self):
        """Test cosize non contiguous."""
        fl = _FlatLayout((2, 3), (6, 1))
        self.assertEqual(fl.cosize(), 9)

    def test_check_sorted_true(self):
        """Test check sorted true."""
        fl = _FlatLayout((2, 3), (6, 1))
        self.assertTrue(fl.check_sorted())

    def test_check_sorted_false(self):
        """Test check sorted false."""
        fl = _FlatLayout((3, 2), (1, 6))
        self.assertFalse(fl.check_sorted())

    def test_check_orthogonal_single_dim(self):
        """Test check orthogonal single dim."""
        fl = _FlatLayout(4)
        self.assertTrue(fl.check_orthogonal())

    def test_check_orthogonal_true(self):
        """Test check orthogonal true."""
        fl = _FlatLayout((2, 3), (3, 1))
        self.assertTrue(fl.check_orthogonal())

    def test_check_orthogonal_false(self):
        """Test check orthogonal false."""
        fl = _FlatLayout((2, 3), (2, 1))
        self.assertFalse(fl.check_orthogonal())

    def test_all_ranks_from_zero(self):
        """Test all ranks from zero."""
        fl = _FlatLayout((2, 3), (3, 1))
        self.assertEqual(fl.all_ranks_from_zero(), [0, 1, 2, 3, 4, 5])

    def test_all_ranks_from_zero_empty(self):
        """Test all ranks from zero empty."""
        fl = _FlatLayout(1)
        self.assertEqual(fl.all_ranks_from_zero(), [0])

    def test_frozen(self):
        """Test frozen."""
        fl = _FlatLayout(4)
        with self.assertRaises(AttributeError):
            fl.shape = (2,)


# ===========================================================================
# _MeshLayout tests
# ===========================================================================

class TestMeshLayoutConstruction(unittest.TestCase):
    """Tests for MeshLayoutConstruction."""
    def test_from_flat_layout_list(self):
        """Test from flat layout list."""
        axes = [_FlatLayout(2), _FlatLayout(4)]
        ml = _MeshLayout(axes)
        self.assertEqual(ml.shape, (2, 4))
        # Each _FlatLayout has its own stride; _MeshLayout preserves them as-is
        self.assertEqual(ml.stride, (1, 1))

    def test_from_shape_stride(self):
        """Test from shape stride."""
        ml = _MeshLayout((2, 4), (4, 1))
        self.assertEqual(ml.shape, (2, 4))
        self.assertEqual(ml.stride, (4, 1))

    def test_structure_mismatch_raises(self):
        """Test structure mismatch raises."""
        with self.assertRaises(ValueError):
            _MeshLayout((2, 3), (4,))

    def test_from_sizes_strides_no_stride(self):
        """Test from sizes strides no stride."""
        ml = _MeshLayout.from_sizes_strides((2, 4))
        self.assertEqual(ml.shape, (2, 4))
        self.assertEqual(ml.stride, (4, 1))

    def test_from_sizes_strides_with_stride(self):
        """Test from sizes strides with stride."""
        ml = _MeshLayout.from_sizes_strides((2, 4), (8, 1))
        self.assertEqual(ml.shape, (2, 4))
        self.assertEqual(ml.stride, (8, 1))


class TestMeshLayoutLen(unittest.TestCase):
    """Tests for MeshLayoutLen."""
    def test_tuple_shape(self):
        """Test tuple shape."""
        ml = _MeshLayout((2, 4), (4, 1))
        self.assertEqual(len(ml), 2)

    def test_int_shape(self):
        """Test int shape."""
        ml = _MeshLayout(6, 1)
        self.assertEqual(len(ml), 1)


class TestMeshLayoutGetItem(unittest.TestCase):
    """Tests for MeshLayoutGetItem."""
    def test_normal_index(self):
        """Test normal index."""
        ml = _MeshLayout((2, 4), (4, 1))
        sub = ml[0]
        self.assertEqual(sub.shape, 2)
        self.assertEqual(sub.stride, 4)

    def test_negative_index(self):
        """Test negative index."""
        ml = _MeshLayout((2, 4), (4, 1))
        sub = ml[-1]
        self.assertEqual(sub.shape, 4)
        self.assertEqual(sub.stride, 1)

    def test_out_of_bounds_raises(self):
        """Test out of bounds raises."""
        ml = _MeshLayout((2, 4), (4, 1))
        with self.assertRaises(IndexError):
            _ = ml[2]

    def test_1d_index_0(self):
        """Test 1d index 0."""
        ml = _MeshLayout(6, 1)
        sub = ml[0]
        self.assertEqual(sub.shape, 6)

    def test_1d_index_neg1(self):
        """Test 1d index neg1."""
        ml = _MeshLayout(6, 1)
        sub = ml[-1]
        self.assertEqual(sub.shape, 6)

    def test_1d_bad_index_raises(self):
        """Test 1d bad index raises."""
        ml = _MeshLayout(6, 1)
        with self.assertRaises(IndexError):
            _ = ml[1]


class TestMeshLayoutEq(unittest.TestCase):
    """Tests for MeshLayoutEq."""
    def test_equal(self):
        """Test equal."""
        a = _MeshLayout((2, 4), (4, 1))
        b = _MeshLayout((2, 4), (4, 1))
        self.assertEqual(a, b)

    def test_not_equal(self):
        """Test not equal."""
        a = _MeshLayout((2, 4), (4, 1))
        b = _MeshLayout((4, 2), (2, 1))
        self.assertNotEqual(a, b)

    def test_not_meshlayout(self):
        """Test not meshlayout."""
        a = _MeshLayout((2, 4), (4, 1))
        self.assertNotEqual(a, "not_a_layout")


class TestMeshLayoutProperties(unittest.TestCase):
    """Tests for MeshLayoutProperties."""
    def test_sizes_and_strides(self):
        """Test sizes and strides."""
        ml = _MeshLayout((2, 4), (4, 1))
        self.assertEqual(ml.sizes, (2, 4))
        self.assertEqual(ml.strides, (4, 1))

    def test_numel(self):
        """Test numel."""
        ml = _MeshLayout((2, 4), (4, 1))
        self.assertEqual(ml.numel(), 8)

    def test_top_level_sizes(self):
        """Test top level sizes."""
        ml = _MeshLayout((2, (3, 4)), (12, (4, 1)))
        self.assertEqual(ml.top_level_sizes, (2, 12))

    def test_all_ranks_from_zero(self):
        """Test all ranks from zero."""
        ml = _MeshLayout((2, 3), (3, 1))
        self.assertEqual(ml.all_ranks_from_zero(), [0, 1, 2, 3, 4, 5])

    def test_check_non_overlap_true(self):
        """Test check non overlap true."""
        ml = _MeshLayout((2, 3), (3, 1))
        self.assertTrue(ml.check_non_overlap())

    def test_check_non_overlap_false(self):
        """Test check non overlap false."""
        ml = _MeshLayout((2, 3), (1, 1))
        self.assertFalse(ml.check_non_overlap())

    def test_axes(self):
        """Test axes."""
        ml = _MeshLayout((2, 4), (4, 1))
        axes = ml.axes
        self.assertEqual(len(axes), 2)
        self.assertIsInstance(axes[0], _FlatLayout)

    def test_iter(self):
        """Test iter."""
        ml = _MeshLayout((2, 4), (4, 1))
        items = list(ml)
        self.assertEqual(len(items), 2)

    def test_repr(self):
        """Test repr."""
        ml = _MeshLayout((2, 4), (4, 1))
        self.assertIn("_MeshLayout", repr(ml))


class TestMeshLayoutCoalesce(unittest.TestCase):
    """Tests for MeshLayoutCoalesce."""
    def test_int_shape_returns_self(self):
        """Test int shape returns self."""
        ml = _MeshLayout(6, 1)
        self.assertIs(ml.coalesce(), ml)

    def test_contiguous_merge(self):
        """Test contiguous merge."""
        ml = _MeshLayout((2, 4), (4, 1))
        coalesced = ml.coalesce()
        self.assertEqual(coalesced.shape, 8)
        self.assertEqual(coalesced.stride, 1)

    def test_non_contiguous_no_merge(self):
        """Test non contiguous no merge."""
        ml = _MeshLayout((2, 4), (8, 1))
        coalesced = ml.coalesce()
        self.assertEqual(coalesced.shape, (2, 4))
        self.assertEqual(coalesced.stride, (8, 1))

    def test_partial_merge_to_single(self):
        """Test partial merge to single."""
        ml = _MeshLayout((2, 3, 4), (12, 4, 1))
        coalesced = ml.coalesce()
        self.assertEqual(coalesced.shape, 24)
        self.assertEqual(coalesced.stride, 1)


class TestMeshLayoutComposition(unittest.TestCase):
    """Tests for MeshLayoutComposition."""
    def test_normal(self):
        """Test normal."""
        base = _MeshLayout(4, 2)
        inner = _MeshLayout((2, 2), (2, 1))
        result = base.composition(inner)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.stride, (4, 2))

    def test_non_scalar_stride_raises(self):
        """Test non scalar stride raises."""
        base = _MeshLayout((2, 2), (4, 1))
        inner = _MeshLayout(2, 1)
        with self.assertRaises(NotImplementedError):
            base.composition(inner)


class TestMeshLayoutNest(unittest.TestCase):
    """Tests for MeshLayoutNest."""
    def test_multi_dim(self):
        """Test multi dim."""
        ml = _MeshLayout((2, 4), (4, 1))
        nested = ml.nest()
        self.assertEqual(nested.shape, ((2, 4),))
        self.assertEqual(nested.stride, ((4, 1),))

    def test_single_dim_returns_self(self):
        """Test single dim returns self."""
        ml = _MeshLayout(6, 1)
        self.assertIs(ml.nest(), ml)


class TestMeshLayoutSplice(unittest.TestCase):
    """Tests for MeshLayoutSplice."""
    def test_replace_range(self):
        """Test replace range."""
        ml = _MeshLayout((2, 3, 4), (12, 4, 1))
        replacement = _MeshLayout((6,), (2,))
        result = ml.splice(1, 2, replacement)
        self.assertEqual(result.shape, (2, 6, 4))
        self.assertEqual(result.stride, (12, 2, 1))

    def test_result_1d(self):
        """Test result 1d."""
        ml = _MeshLayout((2, 4), (4, 1))
        replacement = _MeshLayout(8, 1)
        result = ml.splice(0, 2, replacement)
        self.assertEqual(result.shape, 8)
        self.assertEqual(result.stride, 1)


class TestMeshLayoutCollapse(unittest.TestCase):
    """Tests for MeshLayoutCollapse."""
    def test_returns_flat_layout(self):
        """Test returns flat layout."""
        ml = _MeshLayout((2, 4), (4, 1))
        fl = ml.collapse()
        self.assertIsInstance(fl, _FlatLayout)
        self.assertEqual(fl.numel(), 8)


class TestMeshLayoutRemapToNumpy(unittest.TestCase):
    """Tests for MeshLayoutRemapToNumpy."""
    def test_normal_remap(self):
        """Test normal remap."""
        ml = _MeshLayout((2, 3), (3, 1))
        rank_map = list(range(6))
        result = ml.remap_to_numpy(rank_map)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 2, 3))

    def test_empty_layout_single_rank(self):
        """Test empty layout single rank."""
        # _MeshLayout(1, 1) canonicalizes to shape=1, stride=1
        # After collapse: shape=(), offsets=[0], each anchor covers [anchor+0]
        # This actually works fine for world_size=1
        ml = _MeshLayout(1, 1)
        result = ml.remap_to_numpy([42])
        self.assertEqual(result.shape, (1, 1))
        self.assertEqual(result.flat[0], 42)

    def test_incomplete_partition_raises(self):
        """Test incomplete partition raises."""
        ml = _MeshLayout((2, 2), (4, 1))
        # world_size=8 but layout only covers 4 ranks with stride 4,1
        # offsets: [0, 1, 4, 5]; anchors 0->[0,1,4,5], anchor 2 -> [2,3,6,7]
        # This should work for 8 ranks.
        # Let's use a layout that definitely won't partition evenly.
        ml2 = _MeshLayout((3,), (1,))
        # world_size=4, offsets=[0,1,2], anchor 0 covers [0,1,2], anchor 3 covers [3,4,5] -> out of bounds
        # So used={0,1,2} != 4
        with self.assertRaises(ValueError):
            ml2.remap_to_numpy([0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
