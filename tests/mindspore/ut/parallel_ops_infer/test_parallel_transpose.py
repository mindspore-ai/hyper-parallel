# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test ut for parallel transpose infer layout"""
import math
import pytest
from hyper_parallel import Layout
from hyper_parallel.core.shard.ops.parallel_transpose import TransposeDistributedOp



class TestTransposeDistributedOp:
    """Test cases for TransposeDistributedOp.infer_layout method."""

    def test_basic_transpose_operation(self):
        """
        Feature: Transpose distributed operator basic functionality
        Description: Test transpose operation with valid 2D layout and axis permutation
        Expectation: Output layout tensor map correctly permuted according to axis
        """
        dev_mat = (2, 2)
        aliases = ("dp", "tp")
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        layout_with_tensor_map = input_layout("tp", "dp")

        op = TransposeDistributedOp('Transpose')
        result_layout = op.infer_layout([layout_with_tensor_map], ((1, 0),))

        assert result_layout.tensor_map == (1, 0)
        assert result_layout.mesh_shape == input_layout.mesh_shape
        assert result_layout.alias_name == input_layout.alias_name

    def test_3d_transpose_operation(self):
        """
        Feature: Transpose distributed operator with 3D tensor
        Description: Test transpose operation with 3D layout and complex axis permutation
        Expectation: Output layout tensor map correctly follows the given permutation
        """
        dev_mat = (2, 2, 2)
        aliases = ("dp", "cp", "tp")
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        layout_with_tensor_map = input_layout("tp", "cp", "dp")

        op = TransposeDistributedOp('Transpose')
        result_layout = op.infer_layout([layout_with_tensor_map], ((2, 0, 1),))

        assert result_layout.tensor_map == (2, 0, 1)
        assert result_layout.mesh_shape == input_layout.mesh_shape
        assert result_layout.alias_name == input_layout.alias_name

    def test_dimension_mismatch_error(self):
        """
        Feature: Transpose distributed operator error handling
        Description: Test transpose operation with mismatched tensor map and axis dimensions
        Expectation: Raise ValueError with appropriate message
        """
        dev_mat = (2, 2)
        aliases = ("dp", "tp")
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        layout_with_tensor_map = input_layout("tp", "dp")

        op = TransposeDistributedOp('Transpose')

        with pytest.raises(ValueError, match="Input tensor shape and permutation must have the same size"):
            op.infer_layout([layout_with_tensor_map], ((1, 0, 2),))

    def test_negative_index_error(self):
        """
        Feature: Transpose distributed operator validation
        Description: Test transpose operation with negative index in axis permutation
        Expectation: Raise ValueError for invalid permutation
        """
        dev_mat = (2, 2, 2)
        aliases = ("dp", "cp", "tp")
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        layout_with_tensor_map = input_layout("tp", "cp", "dp")

        op = TransposeDistributedOp('Transpose')

        with pytest.raises(ValueError, match="Invalid permutation"):
            op.infer_layout([layout_with_tensor_map], ((-1, 0, 1),))

    def test_out_of_range_index_error(self):
        """
        Feature: Transpose distributed operator bounds checking
        Description: Test transpose operation with axis index exceeding tensor dimensions
        Expectation: Raise ValueError for invalid permutation due to out of range index
        """
        dev_mat = (2, 2)
        aliases = ("dp", "tp")
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        layout_with_tensor_map = input_layout("tp", "dp")

        op = TransposeDistributedOp('Transpose')

        with pytest.raises(ValueError, match="Invalid permutation"):
            op.infer_layout([layout_with_tensor_map], ((0, 2),))

    def test_duplicate_indices_error(self):
        """
        Feature: Transpose distributed operator uniqueness validation
        Description: Test transpose operation with duplicate indices in axis permutation
        Expectation: Raise ValueError for invalid permutation due to duplicate indices
        """
        dev_mat = (2, 2, 2)
        aliases = ("dp", "cp", "tp")
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        layout_with_tensor_map = input_layout("tp", "cp", "dp")

        op = TransposeDistributedOp('Transpose')

        with pytest.raises(ValueError, match="Invalid permutation"):
            op.infer_layout([layout_with_tensor_map], ((1, 1, 0),))

    def test_identity_transpose(self):
        """
        Feature: Transpose distributed operator identity operation
        Description: Test transpose operation with identity permutation (no change)
        Expectation: Output layout identical to input layout
        """
        dev_mat = (2, 2, 2)
        aliases = ("dp", "cp", "tp")
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        layout_with_tensor_map = input_layout("cp", "tp", "dp")

        op = TransposeDistributedOp('Transpose')
        result_layout = op.infer_layout([layout_with_tensor_map], ((0, 1, 2),))

        assert result_layout.tensor_map == (1, 0, 2)
        assert result_layout.mesh_shape == input_layout.mesh_shape
        assert result_layout.alias_name == input_layout.alias_name


class TestPermuteDistributedOp:
    """Test cases for Permute Distributed Operator (mapped to TransposeDistributedOp)."""

    def test_permute_3d_basic(self):
        """
        Feature: Permute distributed operator basic functionality (3D)
        Description: Test permute operation with standard 3D layout (PyTorch style dims: 2, 0, 1)
        Expectation: Output layout tensor map correctly permuted according to axis
        """
        # Layout: [TP, CP, DP] -> (2, 2, 2)
        dev_mat = (2, 2, 2)
        aliases = ("tp", "cp", "dp")
        rank_list = list(range(math.prod(dev_mat)))

        # Input Tensor Map: ("tp", "cp", "dp") -> corresponds to indices 2, 1, 0
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        layout_with_tensor_map = input_layout("tp", "cp", "dp")

        # Init op with 'permute' name
        op = TransposeDistributedOp('permute')

        # Pytorch: x.permute(2, 0, 1) -> moves dim 2 to pos 0, dim 0 to pos 1...
        # Expectation: Output Tensor Map should be ("dp", "tp", "cp")
        # Note: We pass ((2, 0, 1),) because the implementation does axis = extra_args[0]
        result_layout = op.infer_layout([layout_with_tensor_map], ((2, 0, 1),))

        # Check Logic:
        # in_map = [0:tp, 1:cp, 2:dp]
        # perm = (2, 0, 1)
        # out[0] = in[2] = dp (2)
        # out[1] = in[0] = tp (0)
        # out[2] = in[1] = cp (1)
        assert result_layout.tensor_map == (0, 2, 1)
        assert result_layout.mesh_shape == input_layout.mesh_shape
        assert result_layout.alias_name == input_layout.alias_name

    def test_permute_4d_complex(self):
        """
        Feature: Permute distributed operator with 4D tensor
        Description: Test permute operation with 4D layout to verify higher dimension support
        Expectation: Output layout tensor map correctly follows the given permutation
        """
        dev_mat = (2, 2, 2, 2)
        aliases = ("dp", "mp", "cp", "tp") # 0, 1, 2, 3
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        # Input map: 3, 2, 1, 0
        layout_with_tensor_map = input_layout("dp", "mp", "cp", "tp")

        op = TransposeDistributedOp('permute')

        # Permute: (0, 3, 1, 2) -> (dp, tp, mp, cp)
        result_layout = op.infer_layout([layout_with_tensor_map], ((0, 3, 1, 2),))

        assert result_layout.tensor_map == (3, 0, 2, 1)

    def test_permute_no_change(self):
        """
        Feature: Permute identity check
        Description: Test permute with order that matches input (no change)
        Expectation: Output tensor map equals input tensor map
        """
        dev_mat = (2, 2)
        aliases = ("dp", "tp")
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        layout_with_tensor_map = input_layout("dp", "tp")#(1,0)

        op = TransposeDistributedOp('permute')
        # Pytorch: x.permute(0, 1)
        result_layout = op.infer_layout([layout_with_tensor_map], ((0, 1),))

        assert result_layout.tensor_map == (1, 0)

    def test_permute_shape_mismatch(self):
        """
        Feature: Permute error handling (Shape Mismatch)
        Description: Input has 3 dims, but permute args has 2 dims
        Expectation: ValueError
        """
        dev_mat = (2, 2, 2)
        aliases = ("dp", "cp", "tp")
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        layout_with_tensor_map = input_layout("dp", "cp", "tp")

        op = TransposeDistributedOp('permute')

        with pytest.raises(ValueError, match="Input tensor shape and permutation must have the same size"):
            op.infer_layout([layout_with_tensor_map], ((0, 1),))

    def test_permute_duplicate_dims(self):
        """
        Feature: Permute error handling (Duplicate Dims)
        Description: Permute arguments contain duplicate dimensions
        Expectation: ValueError
        """
        dev_mat = (2, 2, 2)
        aliases = ("dp", "cp", "tp")
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        layout_with_tensor_map = input_layout("dp", "cp", "tp")

        op = TransposeDistributedOp('permute')

        # Invalid: dimension 0 is used twice
        with pytest.raises(ValueError, match="Invalid permutation"):
            op.infer_layout([layout_with_tensor_map], ((0, 0, 1),))

    def test_permute_out_of_bounds(self):
        """
        Feature: Permute error handling (Out of Bounds)
        Description: Permute arguments contain index larger than rank
        Expectation: ValueError
        """
        dev_mat = (2, 2)
        aliases = ("dp", "tp")
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        layout_with_tensor_map = input_layout("dp", "tp")

        op = TransposeDistributedOp('permute')

        # Invalid: dimension 2 does not exist for 2D tensor
        with pytest.raises(ValueError, match="Invalid permutation"):
            op.infer_layout([layout_with_tensor_map], ((2, 1),))

    def test_permute_negative_index_error(self):
        """
        Feature: Permute error handling (Negative Index)
        Description: Current implementation does not support negative indices in backend
        Expectation: ValueError
        """
        # Note: While PyTorch supports negative indices (e.g. permute(0, -1)),
        # the provided backend code explicitly checks 'if v < 0' and raises error.
        # This test ensures the backend safety check works.
        dev_mat = (2, 2)
        aliases = ("dp", "tp")
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        layout_with_tensor_map = input_layout("dp", "tp")

        op = TransposeDistributedOp('permute')

        with pytest.raises(ValueError, match="Invalid permutation"):
            op.infer_layout([layout_with_tensor_map], ((0, -1),))



class TesttransposeDistributedOp:
    """Test cases for Torch-style Transpose Distributed Operator (swapping two dimensions)."""

    def test_torch_transpose_basic(self):
        """
        Feature: Torch Transpose distributed operator basic functionality
        Description: Test transpose(dim0, dim1) with 2D layout
        Expectation: Output layout tensor map has dimensions swapped
        """
        dev_mat = (2, 2)
        aliases = ("dp", "tp")
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        # tensor_map: dp(1), tp(0) -> (1, 0)
        layout_with_tensor_map = input_layout("dp", "tp")

        op = TransposeDistributedOp('transpose')
        # swap 0 and 1 -> (0, 1)
        # extra_args should be passed as (dim0, dim1)
        result_layout = op.infer_layout([layout_with_tensor_map], (0, 1))

        assert result_layout.tensor_map == (0, 1)
        assert result_layout.mesh_shape == input_layout.mesh_shape

    def test_torch_transpose_3d_complex(self):
        """
        Feature: Torch Transpose with 3D tensor
        Description: Test transpose(dim0, dim1) on 3D layout
        Expectation: Only specified dimensions are swapped in tensor map
        """
        dev_mat = (2, 2, 2)
        aliases = ("dp", "cp", "tp") # indices 0, 1, 2. map vals: 2, 1, 0
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        # dp, cp, tp -> (2, 1, 0)
        layout_with_tensor_map = input_layout("dp", "cp", "tp")

        op = TransposeDistributedOp('transpose')
        # transpose(0, 2) -> swap first and last
        # input: (2, 1, 0). swap 0 and 2 -> (0, 1, 2)
        result_layout = op.infer_layout([layout_with_tensor_map], (0, 2))

        assert result_layout.tensor_map == (0, 1, 2)

    def test_torch_transpose_negative_indices(self):
        """
        Feature: Torch Transpose negative indices
        Description: Test transpose with negative dimension indices
        Expectation: Negative indices are correctly interpreted and dimensions swapped
        """
        dev_mat = (2, 2, 2)
        aliases = ("dp", "cp", "tp")
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        # (2, 1, 0)
        layout_with_tensor_map = input_layout("dp", "cp", "tp")

        op = TransposeDistributedOp('transpose')
        # transpose(0, -1) -> swap 0 and 2
        result_layout = op.infer_layout([layout_with_tensor_map], (0, -1))

        assert result_layout.tensor_map == (0, 1, 2)

    def test_torch_transpose_arg_count_error(self):
        """
        Feature: Torch Transpose error handling
        Description: Test with incorrect number of arguments
        Expectation: ValueError
        """
        dev_mat = (2, 2)
        aliases = ("dp", "tp")
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        layout_with_tensor_map = input_layout("dp", "tp")

        op = TransposeDistributedOp('transpose')

        with pytest.raises(ValueError, match=r"expected \(dim0, dim1\)"):
            op.infer_layout([layout_with_tensor_map], (0,))

    def test_torch_transpose_type_error(self):
        """
        Feature: Torch Transpose type validation
        Description: Test with non-integer dimensions
        Expectation: ValueError
        """
        dev_mat = (2, 2)
        aliases = ("dp", "tp")
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        layout_with_tensor_map = input_layout("dp", "tp")

        op = TransposeDistributedOp('transpose')

        with pytest.raises(ValueError, match="Dimensions must be integers"):
            op.infer_layout([layout_with_tensor_map], (0, "1"))

    def test_torch_transpose_out_of_bounds(self):
        """
        Feature: Torch Transpose bounds checking
        Description: Test with dimension index out of bounds
        Expectation: ValueError
        """
        dev_mat = (2, 2)
        aliases = ("dp", "tp")
        rank_list = list(range(math.prod(dev_mat)))
        input_layout = Layout(mesh_shape=dev_mat, alias_name=aliases, rank_list=rank_list)
        layout_with_tensor_map = input_layout("dp", "tp")

        op = TransposeDistributedOp('transpose')

        with pytest.raises(ValueError, match="Transpose dimensions out of bounds"):
            op.infer_layout([layout_with_tensor_map], (0, 2))
