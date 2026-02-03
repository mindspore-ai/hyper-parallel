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
"""parallel_repeat_interleave test"""

import pytest
from hyper_parallel import Layout
from hyper_parallel.core.shard.ops.parallel_repeat_interleave import RepeatInterleaveDistributedOp
import numpy as np

op = RepeatInterleaveDistributedOp("repeat_interleave")

def test_torch_repeat_interleave_layout_data_parallel():
    """
    Feature: RepeatInterleave data parallel
    Description: Data parallel scenario (shard on first dim, repeat on last unsharded dim)
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "tp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")  # tensor_map = (1, -1)  len(alias_name)-1-i
    repeats = 2
    dim = 1
    output_layout = op.infer_layout(x_layout, (repeats, dim))

    expected_map = (1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Data Parallel with torch repeat_interleave test failed. Expected {expected_map}," \
        f" got {output_layout.to_dict()['tensor_map']}"


def test_torch_repeat_interleave_layout_tensor_parallel():
    """
    Feature: RepeatInterleave tensor parallel
    Description: Tensor parallel scenario (shard on first dim with 'tp', repeat on last unsharded dim)
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "tp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("tp", "None")  # tensor_map = (0, -1)

    repeats = 2
    dim = 1
    output_layout = op.infer_layout((x_layout,), (repeats, dim))
    expected_map = (0, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Tensor Parallel with torch repeat_interleave test failed. Expected {expected_map}," \
        f" got {output_layout.to_dict()['tensor_map']}"


def test_torch_repeat_interleave_with_tensor_layout_data_parallel():
    """
    Feature: RepeatInterleave data parallel
    Description: Data parallel scenario (shard on first dim, repeat on last unsharded dim)
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "tp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")  # tensor_map = (1, -1)  len(alias_name)-1-i
    repeats_tensor = [2,1,1,1]

    dim = 1
    output_layout = op.infer_layout((x_layout,), (repeats_tensor, dim))
    expected_map = (1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Data Parallel with torch repeat_interleave test failed. Expected {expected_map}," \
        f" got {output_layout.to_dict()['tensor_map']}"


def test_torch_repeat_interleave_with_tensor_layout_tensor_parallel():
    """
    Feature: RepeatInterleave tensor parallel
    Description: Tensor parallel scenario (shard on first dim with 'tp', repeat on last unsharded dim)
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "tp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("tp", "None")  # tensor_map = (0, -1)
    repeats_tensor = [2,1,1,1]

    dim = 1
    output_layout = op.infer_layout((x_layout,), (repeats_tensor, dim))
    expected_map = (0, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Tensor Parallel with torch repeat_interleave test failed. Expected {expected_map}," \
        f" got {output_layout.to_dict()['tensor_map']}"
def test_torch_repeat_interleave_dim_none_layout_data_parallel():
    """
    Feature: RepeatInterleave data parallel
    Description: Data parallel scenario (shard on first dim, repeat on last unsharded dim)
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "tp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "None")
    repeats = 2
    output_layout = op.infer_layout((x_layout,), (repeats,))
    expected_map = (1,)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Data Parallel with dim None repeat_interleave test failed. Expected {expected_map}," \
        f" got {output_layout.to_dict()['tensor_map']}"


def test_torch_repeat_interleave_dim_none_layout_tensor_parallel():
    """
    Feature: RepeatInterleave tensor parallel
    Description: Tensor parallel scenario (shard on first dim with 'tp', repeat on last unsharded dim)
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "tp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("tp", "None")

    repeats = 2
    output_layout = op.infer_layout((x_layout,), (repeats,))
    expected_map = (0,)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Tensor Parallel dim None repeat_interleave test failed. Expected {expected_map}," \
        f" got {output_layout.to_dict()['tensor_map']}"

def test_torch_repeat_interleave_layout_sharded_dim_error():
    """
    Feature: RepeatInterleave on sharded dimension
    Description: Repeat on a sharded dimension should raise error
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "tp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("dp", "tp")  # tensor_map = (1, 0)

    repeats = 2
    dim = 1
    with pytest.raises(ValueError, match="Cannot perform sharding on params along the chosen dim"):
        op.infer_layout((x_layout,), (repeats, dim))

def test_torch_repeat_interleave_layout_error_dim_out_of_range():
    """
    Feature: Test indicating a invalid dim
    Description: Test indicating a invalid dim.
    Expectation: Success
    """
    base_mesh_shape = (2, 4)
    base_alias_name = ("dp", "tp")
    base_rank_list = list(range(8))

    x_layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "None")

    repeats = 2
    dim = 5

    with pytest.raises(ValueError, match="Dimension out of range"):
        op.infer_layout((x_layout,), extra_args=(repeats, dim))  # dim=5 invalid
