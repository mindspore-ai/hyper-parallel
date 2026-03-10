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
"""parallel_one_hot_ext_shell test"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

ONE_HOT_EXT_SHARD_IN_PYTHON = "one_hot_ext_shard_in_python.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_one_hot_ext_shard_in_python_group1():
    """
    Feature: parallel run case in one_hot_ext_shard_in_python
    Description:
        1. test_one_hot_ext_data_parallel_1d_int64_1
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ONE_HOT_EXT_SHARD_IN_PYTHON, "test_one_hot_ext_data_parallel_1d_int64_1", 12300, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_one_hot_ext_shard_in_python_group2():
    """
    Feature: parallel run case in one_hot_ext_shard_in_python
    Description:
        1. test_one_hot_ext_data_parallel_2d_int64_2
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ONE_HOT_EXT_SHARD_IN_PYTHON, "test_one_hot_ext_data_parallel_2d_int64_2", 12301, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_one_hot_ext_shard_in_python_group3():
    """
    Feature: parallel run case in one_hot_ext_shard_in_python
    Description:
        1. test_one_hot_ext_replicate_all_int64_3
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ONE_HOT_EXT_SHARD_IN_PYTHON, "test_one_hot_ext_replicate_all_int64_3", 12302, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_one_hot_ext_shard_in_python_group4():
    """
    Feature: parallel run case in one_hot_ext_shard_in_python
    Description:
        1. test_one_hot_ext_3d_data_parallel_int64_4
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ONE_HOT_EXT_SHARD_IN_PYTHON, "test_one_hot_ext_3d_data_parallel_int64_4", 12303, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_one_hot_ext_shard_in_python_group5():
    """
    Feature: parallel run case in one_hot_ext_shard_in_python
    Description:
        1. test_one_hot_ext_auto_depth_skewed_distribution_int64_5
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ONE_HOT_EXT_SHARD_IN_PYTHON, "test_one_hot_ext_auto_depth_skewed_distribution_int64_5", 12304, 8,
                      8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_one_hot_ext_shard_in_python_group6():
    """
    Feature: parallel run case in one_hot_ext_shard_in_python
    Description:
        1. test_one_hot_ext_auto_depth_all_same_local_max_int64_6
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(ONE_HOT_EXT_SHARD_IN_PYTHON, "test_one_hot_ext_auto_depth_all_same_local_max_int64_6", 12305, 8,
                      8, 2),
    ])
