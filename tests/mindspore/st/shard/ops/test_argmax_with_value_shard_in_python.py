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
"""parallel_argmax_with_value shell test"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

ARGMAX_WITH_VALUE_SHARD_IN_PYTHON = "argmax_with_value_shard_in_python.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_argmax_with_value_shard_in_python_group1():
    """
    Feature: parallel run case in argmax_with_value_shard_in_python
    Description:
        1. test_argmax_with_value_data_parallel_1
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ARGMAX_WITH_VALUE_SHARD_IN_PYTHON, "test_argmax_with_value_data_parallel_1", 11600, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_argmax_with_value_shard_in_python_group2():
    """
    Feature: parallel run case in argmax_with_value_shard_in_python
    Description:
        1. test_argmax_with_value_model_parallel_2
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ARGMAX_WITH_VALUE_SHARD_IN_PYTHON, "test_argmax_with_value_model_parallel_2", 11601, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_argmax_with_value_shard_in_python_group3():
    """
    Feature: parallel run case in argmax_with_value_shard_in_python
    Description:
        1. test_argmax_with_value_hybrid_parallel_3
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ARGMAX_WITH_VALUE_SHARD_IN_PYTHON, "test_argmax_with_value_hybrid_parallel_3", 11602, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_argmax_with_value_shard_in_python_group4():
    """
    Feature: parallel run case in argmax_with_value_shard_in_python
    Description:
        1. test_argmax_with_value_all_replicated_4
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ARGMAX_WITH_VALUE_SHARD_IN_PYTHON, "test_argmax_with_value_all_replicated_4", 11603, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_argmax_with_value_shard_in_python_group5():
    """
    Feature: parallel run case in argmax_with_value_shard_in_python
    Description:
        1. test_argmax_with_value_negative_dim_5
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ARGMAX_WITH_VALUE_SHARD_IN_PYTHON, "test_argmax_with_value_negative_dim_5", 11604, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_argmax_with_value_shard_in_python_group6():
    """
    Feature: parallel run case in argmax_with_value_shard_in_python
    Description:
        1. test_argmax_with_value_keep_dims_false_6
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ARGMAX_WITH_VALUE_SHARD_IN_PYTHON, "test_argmax_with_value_keep_dims_false_6", 11605, 8, 8, 2),
    ])
