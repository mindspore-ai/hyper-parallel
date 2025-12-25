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
"""test ut for parallel slice infer_layout"""
import pytest
from hyper_parallel import Layout
from hyper_parallel.core.tensor_parallel.ops.parallel_slice import SliceDistributedOp

op = SliceDistributedOp("Slice")


def test_slice_layout_1():
    """
    Feature: MatMul data parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    # Data Parallel (DP)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")

    output_info = op.infer_layout((x_layout,), ((0, 0), (8, 4), (8, 8)))
    assert x_layout == output_info[0]
    assert output_info[1] == (0, 0)
    assert output_info[2] == (4, 4)


def test_slice_layout_2():
    """
    Feature: MatMul data parallel
    Description: Data parallel scenario
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    # Data Parallel (DP)
    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")

    with pytest.raises(ValueError):
        _ = op.infer_layout((x_layout,), ((0, 0), (4, 4), (8, 8)))
