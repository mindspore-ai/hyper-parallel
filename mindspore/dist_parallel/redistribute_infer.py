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
"""redistribute_infer"""
from typing import Dict, List, Tuple, Union

class Status:
    SUCCESS = 0
    FAILED = 1


CONCAT_BY_AXIS = 0
SPLIT_BY_AXIS = 1
PERMUTE_BY_AXIS = 2
NONE = -1

class TensorMap:
    """Enhanced tensor map struct supporting tuples for combined dimensions"""
    def __init__(self, dims: List[Union[int, Tuple[int, ...]]]):
        self.dims = dims

    def GetDimByIdx(self, index: int) -> Union[int, Tuple[int, ...]]:
        return self.dims[index] if index < len(self.dims) else NONE

    def GetIndexByValue(self, value: Union[int, Tuple[int, ...]]) -> int:
        for i, dim in enumerate(self.dims):
            if dim == value:
                return i
        return NONE

    def GetIndexContainValue(self, value: Union[int, Tuple[int, ...]]) -> int:
        for i, dim in enumerate(self.dims):
            if not isinstance(dim, tuple):
                continue
            if isinstance(value, tuple) and value == dim[len(dim) - len(value):]:
                return i
            if not isinstance(value, tuple) and value == dim[-1]:
                return i
        return NONE

class DevMat:
    """
    Represents a multi-dimensional grid of devices where each dimension has a specific size.
    Supports operations to retrieve device groups along single or combined dimensions.

    Attributes:
        dims (List[int]): Sizes of each dimension in the device matrix.
        _combined_dims (Dict[Tuple[int, ...], int]): Cache for precomputed combined dimension sizes.
    """

    def __init__(self, dims: List[int]):
        """
        Initialize device matrix dimensions.

        Args:
            dims: List of integers representing the size of each dimension.
        """
        self.dims = dims
        self._combined_dims: Dict[Tuple[int, ...], int] = {}

    def GetDimByReverseIdx(self, idx: Union[int, Tuple[int, ...]]) -> int:
        """
        Get dimension size by reverse index or product of combined dimensions.

        For a single integer index `i`, returns the size of the dimension at reverse
        position (i.e., `dims[len(dims)-1-i]`). For a tuple of indices, returns the
        product of sizes for the specified reverse-indexed dimensions.

        Args:
            idx: Integer dimension index or tuple of indices.

        Returns:
            Dimension size (for integer) or product of sizes (for tuple).
        """
        if isinstance(idx, tuple):
            return self._GetCombinedSize(idx)
        return self.dims[len(self.dims) - 1 - idx]

    def _GetCombinedSize(self, dims: Union[int, Tuple[int, ...]]) -> int:
        """
        Compute and cache the product of sizes for combined dimensions.

        Args:
            dims: Tuple of dimension indices (reverse-indexed).

        Returns:
            Product of sizes for the specified dimensions.
        """
        if dims in self._combined_dims:
            return self._combined_dims[dims]
        size = 1
        for d in dims:
            size *= self.dims[len(self.dims) - 1 - d]
        self._combined_dims[dims] = size
        return size

    def _GetDevicesAlongDim(self, rank: int, rank_list: List[int], dim: int) -> List[int]:
        """
        Get devices sharing the same coordinates.

        Devices are grouped such that only the specified dimension varies. The device matrix
        is assumed to be in row-major order (last dimension changes fastest).

        Args:
            rank: Target device rank.
            rank_list: Flattened list of all devices in row-major order.
            dim: Target dimension index (0-indexed from outermost).

        Returns:
            List of devices in the same group as `rank` along `dim`.

        Raises:
            ValueError: For invalid dimension or mismatched rank_list size.
        """
        if dim < 0 or dim >= len(self.dims):
            raise ValueError(f"Dimension {dim} out of range [0, {len(self.dims)})")

        # Trivial case: dimension size is 1
        if self.dims[dim] == 1:
            return [rank]

        total_devices = 1
        for d in self.dims:
            total_devices *= d

        # Validate rank_list length
        if len(rank_list) != total_devices:
            raise ValueError(f"rank_list length ({len(rank_list)}) doesn't match "
                             f"device matrix product ({total_devices})")

        # Compute stride for the dimension
        stride = 1
        for i in range(dim + 1, len(self.dims)):
            stride *= self.dims[i]

        # Find local index of rank in rank_list
        try:
            local_index = rank_list.index(rank)
        except ValueError:
            raise ValueError(f"Rank {rank} not in rank_list")

        # Calculate base index and generate group
        index_in_dim = (local_index // stride) % self.dims[dim]
        base = local_index - index_in_dim * stride
        group = [rank_list[base + k * stride] for k in range(self.dims[dim])]

        return group

    def GetDevicesAlongDim(self, rank: int, rank_list: List[int], dim: Union[int, List[int]]) -> List[int]:
        """
        Get devices sharing the same coordinates.

        For a single dimension, returns devices where only that dimension varies.
        For a tuple of dimensions, returns devices where ONLY the specified dimensions vary,
        sharing fixed coordinates in all other dimensions.

        Args:
            rank: Target device rank.
            rank_list: Flattened list of all devices in row-major order.
            dim: Single dimension index or tuple of indices.

        Returns:
            List of devices in the same hyperplane as `rank` orthogonal to `dim`.

        Raises:
            ValueError: For invalid dimensions or mismatched rank_list size.
        """
        if isinstance(dim, list):
            result = self._GetDevicesAlongDim(rank, rank_list, dim[0])
            current_layer_len = len(result)
            current_layer_step = 0
            dim_index = 1
            while dim_index < len(dim):
                sub_rank = result.pop(0)
                result.extend(self._GetDevicesAlongDim(sub_rank, rank_list, dim[dim_index]))
                current_layer_step += 1
                if current_layer_step == current_layer_len:
                    dim_index += 1
                    current_layer_step = 0
                    current_layer_len = len(result)
            return result
        return self._GetDevicesAlongDim(rank, rank_list, dim)


class RedistributionOperatorInfer:
    """
    Infers communication operators for tensor redistribution in distributed systems.

    Determines the sequence of communication operations (split, concat, permute)
    required to transform a tensor from an input device mapping to an output device mapping.

    Args:
        dev_mat: Device matrix dimensions representing the device grid
        in_tensor_map: Input tensor's device mapping for each tensor dimension
        out_tensor_map: Output tensor's device mapping for each tensor dimension
        use_permute: Whether to use permute operator (all-to-all) when possible (default: True)
    """
    def __init__(self, dev_mat: List[int],
                 in_tensor_map: List[Union[int, Tuple[int, ...]]],
                 out_tensor_map: List[Union[int, Tuple[int, ...]]],
                 use_permute: bool = True):

        self.operator_list_: List[Tuple[int, Tuple]] = []
        self.map_: Dict[int, Union[int, Tuple[int, ...]]] = {}
        self.use_permute = use_permute

        # Initialize with expanded dimensions
        self.dev_ranks = len(dev_mat)
        self.dev_mat_ = DevMat(dev_mat)
        self.in_tensor_map_ = TensorMap(in_tensor_map)
        self.out_tensor_map_ = TensorMap(out_tensor_map)

        self.map_ = {i: self.in_tensor_map_.GetDimByIdx(i)
                     for i in range(len(in_tensor_map))}

    def InsertOperator(self, op_type: int, args: Tuple) -> int:
        """
        Adds an operator to the internal operator sequence.

        Args:
            op_type: Operator type constant (SPLIT_BY_AXIS, CONCAT_BY_AXIS, PERMUTE_BY_AXIS)
            args: Operator-specific arguments tuple

        Returns:
            Status.SUCCESS on success, Status.FAILED on error
        """
        self.operator_list_.append((op_type, args))
        return Status.SUCCESS

    def InferRedistributionOperator(self) -> int:
        """
        Main inference driver coordinating the redistribution sequence.

        Executes in 3 phases until mapping is resolved:
        1. Split operations
        2. Permute/All-to-All operations
        3. Concat operations

        Returns:
            Status.SUCCESS if full sequence inferred, Status.FAILED otherwise
        """
        while self.map_:
            len_global = len(self.operator_list_)

            while self.map_:
                len_split_by_axis = len(self.operator_list_)

                # Step 1: infer split op
                if self.InferSplitByAxis() == Status.FAILED:
                    return Status.FAILED

                # Step 2: infer alltoall op
                while self.map_:
                    len_permute_by_axis = len(self.operator_list_)
                    if self.InferPermuteByAxis() == Status.FAILED:
                        return Status.FAILED
                    if len_permute_by_axis == len(self.operator_list_):
                        break

                if len_split_by_axis == len(self.operator_list_):
                    break

            # Step 3: infer allconcat op
            if self.InferConcatByAxis() == Status.FAILED:
                return Status.FAILED

            if len_global == len(self.operator_list_) and self.map_:
                index = next(iter(self.map_.keys()))
                in_dim = self.map_[index]
                self.map_[index] = NONE
                dev_dim = self.dev_mat_.GetDimByReverseIdx(in_dim)
                args = (index, in_dim, dev_dim)
                if self.InsertOperator(CONCAT_BY_AXIS, args) == Status.FAILED:
                    return Status.FAILED

        return Status.SUCCESS

    def _HandleSimpleSplitCase(self, index: int, in_dim: Union[int, Tuple[int, ...]],
                               out_dim: Union[int, Tuple[int, ...]]) -> bool:
        """Handle the simple case where input dimension is None and output dimension is not conflicting"""
        if in_dim != NONE:
            return False

        conflict = any(v == out_dim for v in self.map_.values())
        if isinstance(out_dim, tuple):
            conflict_tuple = any(isinstance(v, tuple) and v[len(v) - len(out_dim):] == out_dim
                                 for v in self.map_.values())
        else:
            conflict_tuple = any(isinstance(v, tuple) and v[-1] == out_dim for v in self.map_.values())

        if not conflict and not conflict_tuple:
            dev_dim = self.dev_mat_.GetDimByReverseIdx(out_dim)
            args = (index, out_dim, dev_dim)
            return self.InsertOperator(SPLIT_BY_AXIS, args) == Status.SUCCESS

        return False

    def _HandleTupleSplitCase(self, index: int, in_dim: Union[int, Tuple[int, ...]],
                              out_dim: Union[int, Tuple[int, ...]]) -> bool:
        """Handle the case where output dimension is a tuple and input dimension matches prefix"""
        if not isinstance(out_dim, tuple):
            return False

        if ((not isinstance(in_dim, tuple) and in_dim == out_dim[0]) or
                (isinstance(in_dim, tuple) and in_dim == out_dim[:len(in_dim)])):

            if isinstance(in_dim, tuple):
                out_dim_rest = out_dim[-1] if len(out_dim[len(in_dim):]) == 1 else out_dim[len(in_dim):]
            else:
                out_dim_rest = out_dim[-1] if len(out_dim[1:]) == 1 else out_dim[1:]

            conflict = any(v == out_dim_rest for v in self.map_.values())
            if not conflict:
                dev_dim = self.dev_mat_.GetDimByReverseIdx(out_dim_rest)
                args = (index, out_dim_rest, dev_dim)
                return self.InsertOperator(SPLIT_BY_AXIS, args) == Status.SUCCESS

        return False

    def InferSplitByAxis(self) -> int:
        """
        Infers split operations for the current mapping state.

        Conditions for split:
        - Tensor dimension changes from unmapped to mapped
        - No conflicts in target device dimension

        Updates internal mapping state and operator list.

        Returns:
            Status.SUCCESS if operations inferred, Status.FAILED on error
        """
        keys = list(self.map_.keys())
        for index in keys:
            if index not in self.map_:
                continue

            in_dim = self.map_[index]
            out_dim = self.out_tensor_map_.GetDimByIdx(index)

            if in_dim == out_dim:
                del self.map_[index]
                continue

            # Handle simple case: input dimension is None
            if self._HandleSimpleSplitCase(index, in_dim, out_dim):
                del self.map_[index]
                continue

            # Handle tuple case: output dimension is a tuple
            if self._HandleTupleSplitCase(index, in_dim, out_dim):
                del self.map_[index]
                continue

        return Status.SUCCESS

    def _HandleNoneDimPermuteCase(self, index: int, in_dim: Union[int, Tuple[int, ...]],
                                  out_dim: Union[int, Tuple[int, ...]]) -> bool:
        """Handle permute case where input dimension is None"""
        if in_dim != NONE:
            return False

        # Check for conflicts in output dimension
        conflict = any(v == out_dim for v in self.map_.values())
        if not conflict:
            return False

        # Handle regular dimension conflict
        concat_axis = self.in_tensor_map_.GetIndexByValue(out_dim)
        if concat_axis is None:
            return False

        split_dev_num = self.dev_mat_.GetDimByReverseIdx(out_dim)

        if self.use_permute:
            # concat tensor map value, to get the communication group
            concat_map = self.in_tensor_map_.GetDimByIdx(concat_axis)
            concat_dev_num = self.dev_mat_.GetDimByReverseIdx(concat_map)
            args_permute = (concat_dev_num, index, concat_axis, concat_map, split_dev_num)

            if self.InsertOperator(PERMUTE_BY_AXIS, args_permute) == Status.FAILED:
                return False
        else:
            args_concat = (concat_axis, out_dim, split_dev_num)
            args_split = (index, out_dim, split_dev_num)

            if self.InsertOperator(CONCAT_BY_AXIS, args_concat) == Status.FAILED:
                return False
            if self.InsertOperator(SPLIT_BY_AXIS, args_split) == Status.FAILED:
                return False

        del self.map_[index]
        self.map_[concat_axis] = NONE
        return True

    def _HandleNoneDimTuplePermuteCase(self, index: int, in_dim: Union[int, Tuple[int, ...]],
                                       out_dim: Union[int, Tuple[int, ...]]) -> bool:
        """Handle permute case where input dimension is None and output dimension is a tuple with conflicts"""
        if in_dim != NONE:
            return False

        if isinstance(out_dim, tuple):
            conflict_tuple = any(isinstance(v, tuple) and v[len(v) - len(out_dim):] == out_dim
                                 for v in self.map_.values())
        else:
            conflict_tuple = any(isinstance(v, tuple) and v[-1] == out_dim for v in self.map_.values())

        if not conflict_tuple:
            return False

        concat_axis = self.in_tensor_map_.GetIndexContainValue(out_dim)
        if concat_axis is None:
            return False

        split_dev_num = self.dev_mat_.GetDimByReverseIdx(out_dim)

        if self.use_permute:
            # concat tensor map value, to get the communication group
            concat_map = out_dim
            concat_dev_num = self.dev_mat_.GetDimByReverseIdx(concat_map)
            args_permute = (concat_dev_num, index, concat_axis, concat_map, split_dev_num)

            if self.InsertOperator(PERMUTE_BY_AXIS, args_permute) == Status.FAILED:
                return False
        else:
            args_concat = (concat_axis, out_dim, split_dev_num)
            args_split = (index, out_dim, split_dev_num)

            if self.InsertOperator(CONCAT_BY_AXIS, args_concat) == Status.FAILED:
                return False
            if self.InsertOperator(SPLIT_BY_AXIS, args_split) == Status.FAILED:
                return False

        del self.map_[index]
        out_dim_len = 1 if not isinstance(out_dim, tuple) else len(out_dim)
        rest_size = len(self.map_[concat_axis]) - out_dim_len
        new_map_item = self.map_[concat_axis][:rest_size] if rest_size > 1 else self.map_[concat_axis][0]
        self.map_[concat_axis] = new_map_item
        return True

    def _HandleTupleDimPermuteCase(self, index: int, in_dim: Union[int, Tuple[int, ...]],
                                   out_dim: Union[int, Tuple[int, ...]]) -> bool:
        """Handle permute case where both input and output dimensions are tuples"""
        if not isinstance(out_dim, tuple):
            return False

        if not ((not isinstance(in_dim, tuple) and in_dim == out_dim[0]) or
                (isinstance(in_dim, tuple) and in_dim == out_dim[:len(in_dim)])):
            return False

        if isinstance(in_dim, tuple):
            out_dim_rest = out_dim[-1] if len(out_dim[len(in_dim):]) == 1 else out_dim[len(in_dim):]
        else:
            out_dim_rest = out_dim[-1] if len(out_dim[1:]) == 1 else out_dim[1:]

        conflict = any(v == out_dim_rest for v in self.map_.values())
        if not conflict:
            return False

        concat_axis = self.in_tensor_map_.GetIndexByValue(out_dim_rest)
        if concat_axis is None:
            return False

        split_dev_num = self.dev_mat_.GetDimByReverseIdx(out_dim_rest)

        if self.use_permute:
            # concat tensor map value, to get the communication group
            concat_map = out_dim_rest
            concat_dev_num = self.dev_mat_.GetDimByReverseIdx(concat_map)
            args_permute = (concat_dev_num, index, concat_axis, concat_map, split_dev_num)

            if self.InsertOperator(PERMUTE_BY_AXIS, args_permute) == Status.FAILED:
                return False
        else:
            args_concat = (concat_axis, out_dim_rest, split_dev_num)
            args_split = (index, out_dim_rest, split_dev_num)

            if self.InsertOperator(CONCAT_BY_AXIS, args_concat) == Status.FAILED:
                return False
            if self.InsertOperator(SPLIT_BY_AXIS, args_split) == Status.FAILED:
                return False

        del self.map_[index]
        self.map_[concat_axis] = NONE
        return True

    def InferPermuteByAxis(self) -> int:
        """
        Infers permutation (all-to-all) operations for dimension conflicts.

        Handles cases where:
        - Input dimension is unmapped but output dimension is already occupied
        - Uses either permute operator or split+concat pair based on use_permute flag

        Returns:
            Status.SUCCESS if operations inferred, Status.FAILED on error
        """
        keys = list(self.map_.keys())
        for index in keys:
            if index not in self.map_:
                continue

            in_dim = self.map_[index]
            out_dim = self.out_tensor_map_.GetDimByIdx(index)

            if in_dim == out_dim:
                del self.map_[index]
                continue

            # Handle different permute cases
            if self._HandleNoneDimPermuteCase(index, in_dim, out_dim):
                continue

            if self._HandleNoneDimTuplePermuteCase(index, in_dim, out_dim):
                continue

            if self._HandleTupleDimPermuteCase(index, in_dim, out_dim):
                continue

        return Status.SUCCESS

    def _HandleTupleConcatCase(self, index: int, in_dim: Union[int, Tuple[int, ...]],
                               out_dim: Union[int, Tuple[int, ...]]) -> bool:
        """Handle concat case where input dimension is a tuple and output matches prefix"""
        if not isinstance(in_dim, tuple):
            return False

        if not ((not isinstance(out_dim, tuple) and out_dim == in_dim[0]) or
                (isinstance(out_dim, tuple) and out_dim == in_dim[:len(out_dim)])):
            return False

        if isinstance(out_dim, tuple):
            in_dim_rest = in_dim[-1] if len(in_dim[len(out_dim):]) == 1 else in_dim[len(out_dim):]
        else:
            in_dim_rest = in_dim[-1] if len(in_dim[1:]) == 1 else in_dim[1:]

        concat_dev_num = self.dev_mat_.GetDimByReverseIdx(in_dim_rest)
        args = (index, in_dim_rest, concat_dev_num)

        if self.InsertOperator(CONCAT_BY_AXIS, args) == Status.FAILED:
            return False

        del self.map_[index]
        return True

    def _HandleSimpleConcatCase(self, index: int, in_dim: Union[int, Tuple[int, ...]],
                                out_dim: Union[int, Tuple[int, ...]]) -> bool:
        """Handle simple concat case where input dimension is mapped but output is not"""
        if in_dim == NONE:
            return False

        if self.out_tensor_map_.GetIndexByValue(in_dim) != NONE:
            return False

        concat_dev_num = self.dev_mat_.GetDimByReverseIdx(in_dim)
        args = (index, in_dim, concat_dev_num)

        if self.InsertOperator(CONCAT_BY_AXIS, args) == Status.FAILED:
            return False

        if out_dim == NONE:
            del self.map_[index]
        else:
            self.map_[index] = NONE

        return True

    def InferConcatByAxis(self) -> int:
        """
        Infers concat operations for the current mapping state.

        Conditions for concat:
        - Input dimension is mapped but output is unmapped
        - Device dimension needs consolidation

        Returns:
            Status.SUCCESS if operations inferred, Status.FAILED on error
        """
        keys = list(self.map_.keys())
        for index in keys:
            if index not in self.map_:
                continue

            in_dim = self.map_[index]
            out_dim = self.out_tensor_map_.GetDimByIdx(index)

            # Handle tuple concat case
            if self._HandleTupleConcatCase(index, in_dim, out_dim):
                continue

            # Handle simple concat case
            if self._HandleSimpleConcatCase(index, in_dim, out_dim):
                continue

        return Status.SUCCESS

    def InferOpsList(self, rank: int, rank_list: List[int]):
        """
        Converts internal operator sequence to executable communication operations.

        Args:
            rank: Current device rank
            rank_list: Full list of device ranks in row-major order

        Returns:
            List of executable communication operations as tuples:
            - ("all_concat", (dim, size, group))
            - ("all_split", (dim, size, group))
            - ("all_to_all", (split_dim, concat_dim, size, group))
        """
        self.InferRedistributionOperator()
        ops_list = []
        for op in self.operator_list_:
            if op[0] == CONCAT_BY_AXIS:
                tensor_map = [self.dev_ranks - 1 - d for d in op[1][1]] if isinstance(op[1][1], tuple) \
                    else self.dev_ranks  - 1 - op[1][1]
                group = self.dev_mat_.GetDevicesAlongDim(rank, rank_list, tensor_map)
                concat_dim = op[1][0]
                concat_size = op[1][2]
                if concat_size == 1:
                    continue
                ops_list.append(("all_concat", (concat_dim, concat_size, group)))
            elif op[0] == SPLIT_BY_AXIS:
                tensor_map = [self.dev_ranks - 1 - d for d in op[1][1]] if isinstance(op[1][1], tuple) \
                    else self.dev_ranks  - 1 - op[1][1]
                group = self.dev_mat_.GetDevicesAlongDim(rank, rank_list, tensor_map)
                split_dim = op[1][0]
                split_size = op[1][2]
                if split_size == 1:
                    continue
                ops_list.append(("all_split", (split_dim, split_size, group)))
            else:
                tensor_map = [self.dev_ranks - 1 - d for d in op[1][3]] if isinstance(op[1][3], tuple) \
                    else self.dev_ranks  - 1 - op[1][3]
                group = self.dev_mat_.GetDevicesAlongDim(rank, rank_list, tensor_map)
                concat_dim = op[1][2]
                split_dim = op[1][1]
                permute_size = op[1][0]
                if permute_size == 1:
                    continue
                ops_list.append(("all_to_all", (split_dim, concat_dim, permute_size, group)))
        return ops_list
