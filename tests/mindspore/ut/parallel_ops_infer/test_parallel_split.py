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
"""parallel_split test"""

import pytest
from hyper_parallel.core.shard.ops.parallel_split import (SplitDistributedOp, SplitWithSizeDistributedOp,
                                                                    SplitTensorDistributedOp)
from hyper_parallel import Layout

# 初始化一个SplitDistributedOp实例
split_op = SplitDistributedOp("split")
torch_split_op = SplitDistributedOp("split")


# 定义一个辅助函数来创建Layout对象
def create_layout(tensor_map):
    base_mesh_shape = (2, 2, 2)
    base_alias_name = ("dp", "mp", "cp")
    base_rank_list = list(range(8))
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    alias_names = [(base_alias_name[-idx - 1] if idx >= 0 else "None") for idx in tensor_map]
    x_layout = x_layout(*alias_names)
    return x_layout


def test_infer_layout_normal():
    """
    Feature: Split operator layout inference under normal conditions
    Description: Test normal split where axis is not sharded
    Expectation: Output layouts are correctly generated with same tensor_map
    """
    input_layout = create_layout([1, -1, 0])
    axis = 1
    split_size = 2
    input_shape = ((12, 16, 20), None)
    extra_args = [split_size, axis, input_shape]

    output_layouts = split_op.infer_layout([input_layout], extra_args)
    assert len(output_layouts) == 8
    assert all(layout.tensor_map == input_layout.tensor_map for layout in output_layouts)


def test_infer_layout_invalid_axis():
    """
    Feature: Split operator layout inference with invalid axis
    Description: Test when trying to split a sharded axis (which is not allowed)
    Expectation: ValueError is raised
    """
    input_layout = create_layout([1, 0, -1])
    axis = 0
    split_size = 2
    input_shape = ((12, 16, 20), None)
    extra_args = [split_size, axis, input_shape]

    with pytest.raises(ValueError, match="Can not split tensor at sharded axis"):
        split_op.infer_layout([input_layout], extra_args)


def test_infer_layout_axis_out_of_range():
    """
    Feature: Test axis out of range
    Description: Test axis out of range
    Expectation: Success
    """
    input_layout = create_layout([-1, -1])
    axis = 5  # invalid for 2D tensor
    split_size = 2
    input_shape = ((12, 16, 20), None)
    extra_args = [split_size, axis, input_shape]

    with pytest.raises(ValueError, match="Dimension out of range"):
        split_op.infer_layout([input_layout], extra_args)


def test_infer_layout_insufficient_extra_args():
    """
    Feature: Test missing extra_arg
    Description: Test missing extra_arg
    Expectation: Success
    """
    input_layout = create_layout([-1, -1])
    extra_args = [0]  # missing output_num

    with pytest.raises(ValueError, match="Split ops extra_args requires 'axis' and contains 'output_num' optionally."):
        split_op.infer_layout([input_layout], extra_args)


split_with_size_op = SplitWithSizeDistributedOp("split_with_size")


def test_infer_layout_with_size():
    """
    Feature: Split operator layout inference with sections list
    Description: Test split using a list of section sizes
    Expectation: Output number matches the length of sections list
    """
    input_layout = create_layout([-1, 1, -1])
    axis = 2
    split_size_or_sections = [2, 3, 3]
    extra_args = [split_size_or_sections, axis]

    output_layouts = split_with_size_op.infer_layout([input_layout], extra_args)

    assert len(output_layouts) == len(split_size_or_sections)
    assert all(layout.tensor_map == input_layout.tensor_map for layout in output_layouts)


def test_infer_layout_with_size_invalid_axis():
    """
    Feature: Split operator layout inference with invalid axis
    Description: Test when trying to split a sharded axis (which is not allowed)
    Expectation: ValueError is raised
    """
    input_layout = create_layout([-1, 1, -1])
    axis = 1
    split_size_or_sections = [2, 3, 3]
    extra_args = [split_size_or_sections, axis]

    with pytest.raises(ValueError):
        split_with_size_op.infer_layout([input_layout], extra_args)


split_tensor_op = SplitTensorDistributedOp("split_tensor")


def test_split_tensor_infer_layout_with_remainder():
    """
    Feature: Split operator layout inference with non-divisible size
    Description: Test split when input shape is not divisible by split size
    Expectation: Output count includes an extra tensor for the remainder
    """
    input_layout = create_layout([-1, -1, 0])
    axis = 1
    split_size = 3
    input_shape = [5, 7, 9]
    extra_args = [split_size, axis, [input_shape, ]]

    output_layouts = split_tensor_op.infer_layout([input_layout], extra_args)

    expected_output_num = input_shape[axis] // split_size + 1
    assert len(output_layouts) == expected_output_num
    assert all(layout.tensor_map == input_layout.tensor_map for layout in output_layouts)


def test_split_tensor_infer_layout_with_remainder_invalid_axis():
    """
    Feature: Split operator layout inference with invalid axis
    Description: Test when trying to split a sharded axis (which is not allowed)
    Expectation: ValueError is raised
    """
    input_layout = create_layout([-1, -1, 0])
    axis = 2
    split_size = 3
    input_shape = [5, 7, 9]
    extra_args = [split_size, axis, [input_shape, ]]

    with pytest.raises(ValueError):
        split_tensor_op.infer_layout([input_layout], extra_args)
