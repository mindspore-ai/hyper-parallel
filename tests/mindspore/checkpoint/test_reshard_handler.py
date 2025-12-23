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
"""Test Resharding"""
import operator
from functools import reduce
import pytest
import numpy as np

from mindspore.parallel import Layout

from hyper_parallel.core.checkpoint.reshard import ReshardHandler
from hyper_parallel.core.checkpoint.reshard import infer_slice_area_by_rank


def get_slice_data(full_data, offset):
    area = ()
    for begin, end in offset:
        area += (slice(begin, end),)
    return full_data[area]


def reshard_tensor_func(param_name, full_shape, from_layout, to_layout, to_rank_id):
    """reshard tensor and verify"""
    reshard = ReshardHandler(param_name, full_shape, from_layout, to_layout, to_rank_id)
    reshard.infer_all_tensor_offset()
    all_offset = reshard.global_union_area_map

    # Generate fake from data
    ele_num = reduce(operator.mul, full_shape)
    full_data = np.array(range(ele_num), np.int32).reshape(full_shape)
    from_tensor_map = {}
    for rank, offset in all_offset.items():
        from_tensor_map[rank] = get_slice_data(full_data, offset)

    # Transfer and verify
    actual_result = reshard.get_real_tensor(from_tensor_map)
    to_offset = infer_slice_area_by_rank(
        reshard.to_dev_matrix,
        reshard.to_tensor_map,
        reshard.to_rank_list.index(reshard.to_rank_id),
        full_shape
    )
    expect_result = get_slice_data(full_data, to_offset)
    assert np.all(actual_result == expect_result)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_reshard_between_layout_and_none_layout():
    """
    Feature: Resharding between a normal layout and a none layout.
    Description: Test reshard between a normal layout and a none layout.
    Expectation: The reshard_tensor_func executes successfully without throwing exceptions,
                 completing bidirectional tensor resharding between a normal layout and a none layout.
    """
    param_name = "weight"
    full_shape = (64, 64)

    layout = Layout((2, 2, 2, 2), ('dp', 'cp', 'rep', 'tp'), rank_list=list(range(16, 32)))
    layout = layout(('rep', 'cp'), 'tp')

    rank_id = 19
    reshard_tensor_func(param_name, full_shape, layout, None, 0)
    reshard_tensor_func(param_name, full_shape, None, layout, rank_id)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_wrong_to_rank_id_while_to_layout_not_none():
    """
    Feature: Resharding when to_layout is normal, but to_rank_id is not in to_rank_list.
    Description: Test reshard when to_rank_id is wrong and layout not none.
    Expectation: A ValueError exception is raised when calling ReshardHandler()
                 with to_layout is normal, but to_rank_id is not in to_rank_list.
    """
    param_name = "weight"
    full_shape = (64, 64)

    layout = Layout((2, 2, 2, 2), ('dp', 'cp', 'rep', 'tp'), rank_list=list(range(16, 32)))
    layout = layout(('rep', 'cp'), 'tp')

    rank_id = 32
    with pytest.raises(ValueError, match=r'to_rank_id is not in to_rank_list.'):
        ReshardHandler(param_name, full_shape, None, layout, rank_id)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_wrong_to_rank_id_while_to_layout_is_none():
    """
    Feature: Resharding between a normal layout and a none layout, to_layout is none and to_rank_id is not 0.
    Description: Test reshard when to_rank_id is wrong and layout is none.
    Expectation: The reshard_tensor_func executes successfully without throwing exceptions,
                 completing bidirectional tensor resharding between the fully sharded layout and none layout.
    """
    param_name = "weight"
    full_shape = (64, 64)

    layout = Layout((2, 2, 2, 2), ('dp', 'cp', 'rep', 'tp'), rank_list=list(range(16, 32)))
    layout = layout(('rep', 'cp'), 'tp')

    reshard_tensor_func(param_name, full_shape, layout, None, 2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_shape_not_even_divided_by_dev_matrix():
    """
    Feature: Resharding when dev can not divide tensor shape.
    Description: Test reshard shape when dev can not divide tensor shape.
    Expectation: A ValueError exception is raised when calling reshard_tensor_func()
                 with message Shape can not divide.
    """
    param_name = "weight"
    full_shape = (64, 65)

    layout = Layout((2, 2, 2, 2), ('dp', 'cp', 'rep', 'tp'), rank_list=list(range(16, 32)))
    layout = layout(('rep', 'cp'), 'tp')

    with pytest.raises(ValueError, match=r'Shape can not divided'):
        reshard_tensor_func(param_name, full_shape, layout, None, 0)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_reshard_between_fully_shard():
    """
    Feature: Resharding between fully sharded modes.
    Description: Test bidirectional tensor resharding between two different fully sharded layouts.
    Expectation: The reshard_tensor_func executes successfully without throwing exceptions,
                 completing bidirectional tensor resharding between the two fully sharded layouts.
    """
    param_name = "weight"
    full_shape = (64, 64)

    layout_0 = Layout((2, 2, 2, 2), ('dp', 'cp', 'rep', 'tp'), rank_list=list(range(16, 32)))
    from_layout = layout_0(('rep', 'cp'), 'tp')

    layout_1 = Layout((2, 4), ('dp', 'tp'), rank_list=list(range(8, 16)))
    to_layout = layout_1('dp', 'tp')

    from_rank_id = 19
    to_rank_id = 14
    reshard_tensor_func(param_name, full_shape, from_layout, to_layout, to_rank_id)
    reshard_tensor_func(param_name, full_shape, to_layout, from_layout, from_rank_id)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_reshard_between_fully_shard_and_not_shard():
    """
    Feature: Resharding between fully sharded and non-sharded modes.
    Description: Test bidirectional tensor resharding between a fully sharded layout and a non-sharded layout.
    Expectation: The reshard_tensor_func executes successfully without throwing exceptions,
                 completing bidirectional tensor resharding between the fully sharded and non-sharded layouts.
    """
    param_name = "weight"
    full_shape = (64, 64)

    layout_0 = Layout((2, 4, 4), ('dp', 'cp', 'tp'), rank_list=list(range(32, 64)))
    from_layout = layout_0(('cp', 'dp'), 'tp')

    layout_1 = Layout((8,), ('dp',), rank_list=list(range(0, 8)))
    to_layout = layout_1('None', 'None')

    from_rank_id = 35
    to_rank_id = 7
    reshard_tensor_func(param_name, full_shape, from_layout, to_layout, to_rank_id)
    reshard_tensor_func(param_name, full_shape, to_layout, from_layout, from_rank_id)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_reshard_between_not_fully_shard():
    """
    Feature: Resharding between non-fully sharded modes.
    Description: Test bidirectional tensor resharding between two different non-fully sharded layouts.
    Expectation: The reshard_tensor_func executes successfully without throwing exceptions,
                 completing bidirectional tensor resharding between the two non-fully sharded layouts.
    """
    param_name = "weight"
    full_shape = (64, 64)

    layout_0 = Layout((2, 4, 4), ('dp', 'cp', 'tp'), rank_list=list(range(32, 64)))
    from_layout = layout_0('cp', 'tp')

    layout_1 = Layout((2, 4, 4), ('dp', 'cp', 'tp'), rank_list=list(range(0, 32)))
    to_layout = layout_1('tp', 'dp')

    from_rank_id = 35
    to_rank_id = 7
    reshard_tensor_func(param_name, full_shape, from_layout, to_layout, to_rank_id)
    reshard_tensor_func(param_name, full_shape, to_layout, from_layout, from_rank_id)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_from_tensor_map_missing_rank():
    """
    Feature: Exception handling for incomplete from_tensor_map.
    Description: Test the scenario where from_tensor_map is missing a required rank.
    Expectation: A ValueError exception is raised when calling get_real_tensor()
                 with an incomplete from_tensor_map that lacks a necessary rank entry.
    """
    param_name = "weight"
    full_shape = (64, 64)

    layout_0 = Layout((2, 4, 4), ('dp', 'cp', 'tp'), rank_list=list(range(32, 64)))
    from_layout = layout_0(('cp', 'dp'), 'tp')

    layout_1 = Layout((8,), ('dp',), rank_list=list(range(0, 8)))
    to_layout = layout_1('None', 'None')

    to_rank_id = 7
    reshard = ReshardHandler(param_name, full_shape, from_layout, to_layout, to_rank_id)
    reshard.infer_all_tensor_offset()
    all_offset = reshard.global_union_area_map

    # Generate fake from data
    ele_num = reduce(operator.mul, full_shape)
    full_data = np.array(range(ele_num), np.int32).reshape(full_shape)
    from_tensor_map = {}
    pop_rank = 0
    for rank, offset in all_offset.items():
        pop_rank = rank
        from_tensor_map[rank] = get_slice_data(full_data, offset)
    from_tensor_map.pop(pop_rank)
    with pytest.raises(ValueError):
        reshard.get_real_tensor(from_tensor_map)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_from_tensor_map_has_unexpected_data():
    """
    Feature: Exception handling for invalid data in from_tensor_map.
    Description: Test the scenario where a rank in from_tensor_map contains data with an unexpected shape.
    Expectation: A ValueError exception is raised when calling get_real_tensor()
                 with from_tensor_map containing a rank with data that has an unexpected shape.
    """
    param_name = "weight"
    full_shape = (64, 64)

    layout_0 = Layout((2, 4, 4), ('dp', 'cp', 'tp'), rank_list=list(range(32, 64)))
    from_layout = layout_0(('cp', 'dp'), 'tp')

    layout_1 = Layout((8,), ('dp',), rank_list=list(range(0, 8)))
    to_layout = layout_1('None', 'None')

    to_rank_id = 7
    reshard = ReshardHandler(param_name, full_shape, from_layout, to_layout, to_rank_id)
    reshard.infer_all_tensor_offset()
    all_offset = reshard.global_union_area_map

    # Generate fake from data
    ele_num = reduce(operator.mul, full_shape)
    full_data = np.array(range(ele_num), np.int32).reshape(full_shape)
    from_tensor_map = {}
    modify_rank = 0
    for rank, offset in all_offset.items():
        modify_rank = rank
        from_tensor_map[rank] = get_slice_data(full_data, offset)
    modify_shape = from_tensor_map[modify_rank].shape
    modify_shape = modify_shape[:-1] + (modify_shape[-1] + 2,)
    from_tensor_map[modify_rank] = np.zeros(modify_shape, full_data.dtype)
    with pytest.raises(ValueError):
        reshard.get_real_tensor(from_tensor_map)
