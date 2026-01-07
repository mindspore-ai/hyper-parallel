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
"""parallel_slice_ext test"""

import pytest
from hyper_parallel.core.shard.ops.parallel_slice_ext import SliceExtDistributedOp
from hyper_parallel import Layout

slice_op = SliceExtDistributedOp("SliceExt")


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
    Feature: Slice operator layout inference under normal conditions
    Description: Test normal split where axis is not sharded
    Expectation: Output layouts are correctly generated with same tensor_map
    """
    input_layout = create_layout([1, -1, 0])
    axis = 1
    extra_args = [axis, 0, 2, 1]
    output_layout = slice_op.infer_layout([input_layout], extra_args)
    assert output_layout == input_layout


def test_infer_layout_invalid_axis():
    """
    Feature: Slice operator layout inference with invalid axis
    Description: Test when trying to split a sharded axis (which is not allowed)
    Expectation: ValueError is raised
    """
    input_layout = create_layout([1, -1, 0])
    axis = 0
    extra_args = [axis, 0, 2, 1]
    with pytest.raises(ValueError):
        slice_op.infer_layout([input_layout], extra_args)
