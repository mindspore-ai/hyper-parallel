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
"""Unit tests for hyper_parallel.core.dtensor.redistribute_infer (pure logic)."""

import unittest

from hyper_parallel.core.dtensor.redistribute_infer import (
    TensorMap,
    DevMat,
    RedistributionOperatorInfer,
    Status,
    CONCAT_BY_AXIS,
    SPLIT_BY_AXIS,
    PERMUTE_BY_AXIS,
    NONE,
)


# ===========================================================================
# TensorMap tests
# ===========================================================================

class TestTensorMap(unittest.TestCase):
    """Tests for TensorMap."""
    def test_get_dim_by_idx_valid(self):
        """Test get dim by idx valid."""
        tm = TensorMap([0, 1, 2])
        self.assertEqual(tm.get_dim_by_idx(0), 0)
        self.assertEqual(tm.get_dim_by_idx(2), 2)

    def test_get_dim_by_idx_out_of_range(self):
        """Test get dim by idx out of range."""
        tm = TensorMap([0, 1])
        self.assertEqual(tm.get_dim_by_idx(5), NONE)

    def test_get_index_by_value_found(self):
        """Test get index by value found."""
        tm = TensorMap([0, 1, (2, 3)])
        self.assertEqual(tm.get_index_by_value(1), 1)
        self.assertEqual(tm.get_index_by_value((2, 3)), 2)

    def test_get_index_by_value_not_found(self):
        """Test get index by value not found."""
        tm = TensorMap([0, 1])
        self.assertEqual(tm.get_index_by_value(5), NONE)

    def test_get_index_contain_value_int_match(self):
        """Test get index contain value int match."""
        tm = TensorMap([0, (1, 2)])
        self.assertEqual(tm.get_index_contain_value(2), 1)

    def test_get_index_contain_value_tuple_suffix_match(self):
        """Test get index contain value tuple suffix match."""
        tm = TensorMap([(0, 1, 2), 3])
        self.assertEqual(tm.get_index_contain_value((1, 2)), 0)

    def test_get_index_contain_value_no_match(self):
        """Test get index contain value no match."""
        tm = TensorMap([0, 1])
        self.assertEqual(tm.get_index_contain_value(5), NONE)

    def test_get_index_contain_value_non_tuple_dim_skipped(self):
        """Test get index contain value non tuple dim skipped."""
        tm = TensorMap([0, 1, 2])
        self.assertEqual(tm.get_index_contain_value(2), NONE)


# ===========================================================================
# DevMat tests
# ===========================================================================

class TestDevMat(unittest.TestCase):
    """Tests for DevMat."""
    def test_get_dim_by_reverse_idx_int(self):
        """Test get dim by reverse idx int."""
        dm = DevMat([2, 4, 8])
        # reverse idx 0 -> last dim (8), 1 -> middle (4), 2 -> first (2)
        self.assertEqual(dm.get_dim_by_reverse_idx(0), 8)
        self.assertEqual(dm.get_dim_by_reverse_idx(1), 4)
        self.assertEqual(dm.get_dim_by_reverse_idx(2), 2)

    def test_get_dim_by_reverse_idx_tuple(self):
        """Test get dim by reverse idx tuple."""
        dm = DevMat([2, 4, 8])
        # tuple (0, 1) -> product of dim[2]*dim[1] = 8*4 = 32
        self.assertEqual(dm.get_dim_by_reverse_idx((0, 1)), 32)

    def test_get_dim_by_reverse_idx_cache(self):
        """Test get dim by reverse idx cache."""
        dm = DevMat([2, 4])
        result1 = dm.get_dim_by_reverse_idx((0, 1))
        result2 = dm.get_dim_by_reverse_idx((0, 1))
        self.assertEqual(result1, result2)
        self.assertIn((0, 1), dm._combined_dims)

    def test_get_devices_along_dim_basic_dim0(self):
        """Test get devices along dim basic dim0."""
        dm = DevMat([2, 3])
        rank_list = [0, 1, 2, 3, 4, 5]
        # dim=0: stride=3, groups of size 2
        group = dm._get_devices_along_dim(0, rank_list, 0)
        self.assertEqual(group, [0, 3])

    def test_get_devices_along_dim_basic_dim1(self):
        """Test get devices along dim basic dim1."""
        dm = DevMat([2, 3])
        rank_list = [0, 1, 2, 3, 4, 5]
        # dim=1: stride=1, groups of size 3
        group = dm._get_devices_along_dim(0, rank_list, 1)
        self.assertEqual(group, [0, 1, 2])

    def test_get_devices_along_dim_out_of_range(self):
        """Test get devices along dim out of range."""
        dm = DevMat([2, 3])
        with self.assertRaises(ValueError):
            dm._get_devices_along_dim(0, [0, 1, 2, 3, 4, 5], 2)

    def test_get_devices_along_dim_rank_not_in_list(self):
        """Test get devices along dim rank not in list."""
        dm = DevMat([2, 3])
        with self.assertRaises(ValueError):
            dm._get_devices_along_dim(99, [0, 1, 2, 3, 4, 5], 0)

    def test_get_devices_along_dim_rank_list_mismatch(self):
        """Test get devices along dim rank list mismatch."""
        dm = DevMat([2, 3])
        with self.assertRaises(ValueError):
            dm._get_devices_along_dim(0, [0, 1, 2], 0)

    def test_get_devices_along_dim_size_1(self):
        """Test get devices along dim size 1."""
        dm = DevMat([1, 4])
        rank_list = [0, 1, 2, 3]
        group = dm._get_devices_along_dim(2, rank_list, 0)
        self.assertEqual(group, [2])

    def test_get_devices_along_dim_single_call(self):
        """Test get devices along dim single call."""
        dm = DevMat([2, 3])
        rank_list = [0, 1, 2, 3, 4, 5]
        group = dm.get_devices_along_dim(1, rank_list, 1)
        self.assertEqual(group, [0, 1, 2])

    def test_get_devices_along_dim_multi_dims(self):
        """Test get devices along dim multi dims."""
        dm = DevMat([2, 2, 2])
        rank_list = list(range(8))
        # dim=[0, 1] means vary dim0 and dim1 while fixing dim2
        group = dm.get_devices_along_dim(0, rank_list, [0, 1])
        # rank 0 is at (0,0,0), fixing dim2=0, varying dim0 and dim1
        # dim0 group for rank 0: [0, 4], then for each, get dim1 group
        # For 0: dim1 group = [0, 2]; For 4: dim1 group = [4, 6]
        self.assertEqual(sorted(group), [0, 2, 4, 6])


# ===========================================================================
# RedistributionOperatorInfer tests
# ===========================================================================

class TestRedistributionOperatorInfer(unittest.TestCase):
    """Tests for RedistributionOperatorInfer."""
    def test_no_change_no_ops(self):
        """Test no change no ops."""
        inferrer = RedistributionOperatorInfer(
            dev_mat=[2, 4],
            in_tensor_map=[0, 1],
            out_tensor_map=[0, 1],
        )
        status = inferrer.infer_redistribution_operator()
        self.assertEqual(status, Status.SUCCESS)
        self.assertEqual(len(inferrer.operator_list_), 0)

    def test_simple_split(self):
        """Test simple split."""
        # in: [NONE, NONE] -> out: [0, NONE]
        inferrer = RedistributionOperatorInfer(
            dev_mat=[2, 4],
            in_tensor_map=[NONE, NONE],
            out_tensor_map=[0, NONE],
        )
        status = inferrer.infer_redistribution_operator()
        self.assertEqual(status, Status.SUCCESS)
        # Should have a split operator
        split_ops = [op for op in inferrer.operator_list_ if op[0] == SPLIT_BY_AXIS]
        self.assertGreater(len(split_ops), 0)

    def test_simple_concat(self):
        """Test simple concat."""
        # in: [0, NONE] -> out: [NONE, NONE]
        inferrer = RedistributionOperatorInfer(
            dev_mat=[2, 4],
            in_tensor_map=[0, NONE],
            out_tensor_map=[NONE, NONE],
        )
        status = inferrer.infer_redistribution_operator()
        self.assertEqual(status, Status.SUCCESS)
        concat_ops = [op for op in inferrer.operator_list_ if op[0] == CONCAT_BY_AXIS]
        self.assertGreater(len(concat_ops), 0)

    def test_permute_with_use_permute_true(self):
        """Test permute with use permute true."""
        # in: [0, NONE] -> out: [NONE, 0] requires permute/all-to-all
        inferrer = RedistributionOperatorInfer(
            dev_mat=[2, 4],
            in_tensor_map=[0, NONE],
            out_tensor_map=[NONE, 0],
            use_permute=True,
        )
        status = inferrer.infer_redistribution_operator()
        self.assertEqual(status, Status.SUCCESS)
        permute_ops = [op for op in inferrer.operator_list_ if op[0] == PERMUTE_BY_AXIS]
        self.assertGreater(len(permute_ops), 0)

    def test_permute_with_use_permute_false(self):
        """Test permute with use permute false."""
        # Same as above but use_permute=False -> should use concat+split instead
        inferrer = RedistributionOperatorInfer(
            dev_mat=[2, 4],
            in_tensor_map=[0, NONE],
            out_tensor_map=[NONE, 0],
            use_permute=False,
        )
        status = inferrer.infer_redistribution_operator()
        self.assertEqual(status, Status.SUCCESS)
        permute_ops = [op for op in inferrer.operator_list_ if op[0] == PERMUTE_BY_AXIS]
        self.assertEqual(len(permute_ops), 0)
        # Should have concat and split ops instead
        self.assertGreater(len(inferrer.operator_list_), 0)

    def test_fallback_concat_branch(self):
        """Test fallback concat branch."""
        # When no split/permute/concat makes progress, the fallback branch should fire
        # in: [0, 1] -> out: [1, 0] - forces a complex redistribution
        inferrer = RedistributionOperatorInfer(
            dev_mat=[2, 4],
            in_tensor_map=[0, 1],
            out_tensor_map=[1, 0],
            use_permute=True,
        )
        status = inferrer.infer_redistribution_operator()
        self.assertEqual(status, Status.SUCCESS)
        self.assertGreater(len(inferrer.operator_list_), 0)

    def test_insert_operator(self):
        """Test insert operator."""
        inferrer = RedistributionOperatorInfer(
            dev_mat=[2], in_tensor_map=[NONE], out_tensor_map=[NONE]
        )
        result = inferrer.insert_operator(SPLIT_BY_AXIS, (0, 0, 2))
        self.assertEqual(result, Status.SUCCESS)
        self.assertEqual(len(inferrer.operator_list_), 1)


class TestHandleTupleSplitCase(unittest.TestCase):
    """Tests for HandleTupleSplitCase."""
    def test_in_dim_prefix_of_out_dim_tuple(self):
        """Test in dim prefix of out dim tuple."""
        # in_dim=0 is prefix of out_dim=(0,1)
        inferrer = RedistributionOperatorInfer(
            dev_mat=[2, 4],
            in_tensor_map=[0, NONE],
            out_tensor_map=[(0, 1), NONE],
        )
        status = inferrer.infer_redistribution_operator()
        self.assertEqual(status, Status.SUCCESS)

    def test_in_dim_tuple_prefix(self):
        """Test in dim tuple prefix."""
        # in_dim=(0,) is prefix of out_dim=(0,1)
        inferrer = RedistributionOperatorInfer(
            dev_mat=[2, 4],
            in_tensor_map=[(0,), NONE],
            out_tensor_map=[(0, 1), NONE],
        )
        status = inferrer.infer_redistribution_operator()
        self.assertEqual(status, Status.SUCCESS)


class TestHandleNoneDimTuplePermuteCase(unittest.TestCase):
    """Tests for HandleNoneDimTuplePermuteCase."""
    def test_none_to_tuple_conflict(self):
        """Test none to tuple conflict."""
        # in: [NONE, (0,1)] -> out: [(0,1), NONE]
        # in_dim=NONE at index 0, out_dim=(0,1) at index 0, map has (0,1) at index 1 -> conflict
        inferrer = RedistributionOperatorInfer(
            dev_mat=[2, 4],
            in_tensor_map=[NONE, (0, 1)],
            out_tensor_map=[(0, 1), NONE],
            use_permute=True,
        )
        status = inferrer.infer_redistribution_operator()
        self.assertEqual(status, Status.SUCCESS)


class TestHandleTupleDimPermuteCase(unittest.TestCase):
    """Tests for HandleTupleDimPermuteCase."""
    def test_in_dim_prefix_out_dim_with_conflict(self):
        """Test in dim prefix out dim with conflict."""
        # in_dim=0 at idx 0, out_dim=(0,1) at idx 0
        # map has value 1 elsewhere -> conflict on out_dim_rest=1
        inferrer = RedistributionOperatorInfer(
            dev_mat=[2, 4],
            in_tensor_map=[0, 1],
            out_tensor_map=[(0, 1), NONE],
            use_permute=True,
        )
        status = inferrer.infer_redistribution_operator()
        self.assertEqual(status, Status.SUCCESS)


class TestHandleTupleConcatCase(unittest.TestCase):
    """Tests for HandleTupleConcatCase."""
    def test_tuple_in_dim_to_none_strided_shard_reorder(self):
        """Test tuple in dim to none strided shard reorder."""
        # in_dim=(1,0) at idx 0, out_dim=NONE -> not descending, so concat last element
        inferrer = RedistributionOperatorInfer(
            dev_mat=[2, 4],
            in_tensor_map=[(1, 0), NONE],
            out_tensor_map=[NONE, NONE],
        )
        status = inferrer.infer_redistribution_operator()
        self.assertEqual(status, Status.SUCCESS)

    def test_tuple_in_dim_to_prefix_match(self):
        """Test tuple in dim to prefix match."""
        # in_dim=(0,1) at idx 0, out_dim=0 -> prefix match, concat rest
        inferrer = RedistributionOperatorInfer(
            dev_mat=[2, 4],
            in_tensor_map=[(0, 1), NONE],
            out_tensor_map=[0, NONE],
        )
        status = inferrer.infer_redistribution_operator()
        self.assertEqual(status, Status.SUCCESS)

    def test_tuple_in_dim_descending_to_none_no_special_concat(self):
        """Test tuple in dim descending to none no special concat."""
        # in_dim=(1,0) descending -> plain same-dim Shard, no special concat
        # Actually (1,0): 1>0 is descending, so it won't trigger the special branch
        inferrer = RedistributionOperatorInfer(
            dev_mat=[2, 4],
            in_tensor_map=[(1, 0)],
            out_tensor_map=[NONE],
        )
        status = inferrer.infer_redistribution_operator()
        self.assertEqual(status, Status.SUCCESS)


class TestInferOpsList(unittest.TestCase):
    """Tests for InferOpsList."""
    def test_split_and_concat_ops(self):
        """Test split and concat ops."""
        # in: [0, NONE] -> out: [NONE, 0]
        inferrer = RedistributionOperatorInfer(
            dev_mat=[2, 4],
            in_tensor_map=[0, NONE],
            out_tensor_map=[NONE, 0],
            use_permute=False,
        )
        rank_list = list(range(8))
        ops = inferrer.infer_ops_list(0, rank_list)
        self.assertIsInstance(ops, list)
        for op in ops:
            self.assertIn(op[0], ("all_concat", "all_split", "all_to_all"))

    def test_permute_ops(self):
        """Test permute ops."""
        inferrer = RedistributionOperatorInfer(
            dev_mat=[2, 4],
            in_tensor_map=[0, NONE],
            out_tensor_map=[NONE, 0],
            use_permute=True,
        )
        rank_list = list(range(8))
        ops = inferrer.infer_ops_list(0, rank_list)
        all_to_all_ops = [op for op in ops if op[0] == "all_to_all"]
        self.assertGreater(len(all_to_all_ops), 0)

    def test_size_1_ops_skipped(self):
        """Test size 1 ops skipped."""
        # When dev_mat dimension is 1, the op should be skipped
        inferrer = RedistributionOperatorInfer(
            dev_mat=[1, 4],
            in_tensor_map=[NONE, NONE],
            out_tensor_map=[1, NONE],  # dim 1 reverse -> dev_mat[0] = 1
        )
        rank_list = list(range(4))
        ops = inferrer.infer_ops_list(0, rank_list)
        # The split with size 1 should be skipped
        split_ops = [op for op in ops if op[0] == "all_split"]
        self.assertEqual(len(split_ops), 0)

    def test_tuple_tensor_map_in_ops_list(self):
        """Test tuple tensor map in ops list."""
        # Test with tuple tensor_map values
        inferrer = RedistributionOperatorInfer(
            dev_mat=[2, 4],
            in_tensor_map=[(0, 1)],
            out_tensor_map=[NONE],
        )
        rank_list = list(range(8))
        ops = inferrer.infer_ops_list(0, rank_list)
        self.assertIsInstance(ops, list)

    def test_no_change_empty_ops(self):
        """Test no change empty ops."""
        inferrer = RedistributionOperatorInfer(
            dev_mat=[2, 4],
            in_tensor_map=[0, 1],
            out_tensor_map=[0, 1],
        )
        rank_list = list(range(8))
        ops = inferrer.infer_ops_list(0, rank_list)
        self.assertEqual(len(ops), 0)


if __name__ == "__main__":
    unittest.main()
