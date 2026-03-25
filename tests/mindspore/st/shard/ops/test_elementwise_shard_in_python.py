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
"""parallel_elementwise_shell test"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

ELEMENTWISE_SHARD_IN_PYTHON = "elementwise_shard_in_python.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_elementwise_shard_in_python_group1():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_minimum_same_shape_parallel_1
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ELEMENTWISE_SHARD_IN_PYTHON, "test_minimum_same_shape_parallel_1", 11300, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_elementwise_shard_in_python_group2():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_less_equal_same_shape_parallel_2
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ELEMENTWISE_SHARD_IN_PYTHON, "test_less_equal_same_shape_parallel_2", 11301, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_elementwise_shard_in_python_group3():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_greater_equal_same_shape_parallel_3
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ELEMENTWISE_SHARD_IN_PYTHON, "test_greater_equal_same_shape_parallel_3", 11302, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_elementwise_shard_in_python_group4():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_logical_or_same_shape_parallel_4
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ELEMENTWISE_SHARD_IN_PYTHON, "test_logical_or_same_shape_parallel_4", 11303, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_elementwise_shard_in_python_group5():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_minimum_broadcast_dim0_parallel_5
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ELEMENTWISE_SHARD_IN_PYTHON, "test_minimum_broadcast_dim0_parallel_5", 11304, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_elementwise_shard_in_python_group6():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_minimum_broadcast_dim1_parallel_6
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ELEMENTWISE_SHARD_IN_PYTHON, "test_minimum_broadcast_dim1_parallel_6", 11305, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_elementwise_shard_in_python_group7():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_minimum_broadcast_dim2_parallel_7
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ELEMENTWISE_SHARD_IN_PYTHON, "test_minimum_broadcast_dim2_parallel_7", 11306, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_elementwise_shard_in_python_group8():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_minimum_broadcast_rank_mismatch_parallel_8
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ELEMENTWISE_SHARD_IN_PYTHON, "test_minimum_broadcast_rank_mismatch_parallel_8", 11307, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_elementwise_shard_in_python_group9():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_minimum_broadcast_scalar_like_parallel_9
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ELEMENTWISE_SHARD_IN_PYTHON, "test_minimum_broadcast_scalar_like_parallel_9", 11308, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_elementwise_shard_in_python_group10():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_less_equal_broadcast_multi_dim_parallel_10
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ELEMENTWISE_SHARD_IN_PYTHON, "test_less_equal_broadcast_multi_dim_parallel_10", 11309, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_elementwise_shard_in_python_group11():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_minimum_partial_shard_parallel_11
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ELEMENTWISE_SHARD_IN_PYTHON, "test_minimum_partial_shard_parallel_11", 11310, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_elementwise_shard_in_python_group12():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_mod_same_shape_parallel_12
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ELEMENTWISE_SHARD_IN_PYTHON, "test_mod_same_shape_parallel_12", 11311, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_elementwise_shard_in_python_group13():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_mod_broadcast_dim0_parallel_13
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ELEMENTWISE_SHARD_IN_PYTHON, "test_mod_broadcast_dim0_parallel_13", 11312, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_elementwise_shard_in_python_group14():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_mod_broadcast_dim1_parallel_14
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ELEMENTWISE_SHARD_IN_PYTHON, "test_mod_broadcast_dim1_parallel_14", 11313, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_elementwise_shard_in_python_group15():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_mod_broadcast_dim2_parallel_15
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ELEMENTWISE_SHARD_IN_PYTHON, "test_mod_broadcast_dim2_parallel_15", 11314, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_elementwise_shard_in_python_group16():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_mod_broadcast_rank_mismatch_parallel_16
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ELEMENTWISE_SHARD_IN_PYTHON, "test_mod_broadcast_rank_mismatch_parallel_16", 11315, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_elementwise_shard_in_python_group17():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_mod_tensor_scalar_parallel_17
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ELEMENTWISE_SHARD_IN_PYTHON, "test_mod_tensor_scalar_parallel_17", 11316, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_elementwise_shard_in_python_group18():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_mod_partial_shard_parallel_18
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ELEMENTWISE_SHARD_IN_PYTHON, "test_mod_partial_shard_parallel_18", 11317, 8, 8, 2),
    ])
