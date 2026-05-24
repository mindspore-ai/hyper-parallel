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

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_one_hot_ext.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_one_hot_ext_group1():
    """
    Feature: parallel run case in one_hot_ext_shard_in_python
    Description:
        1. test_one_hot_ext_data_parallel_1d_int64_1
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_one_hot_ext_data_parallel_1d_int64_1", worker_num=8, local_worker_num=8,
            glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_one_hot_ext_group2():
    """
    Feature: parallel run case in one_hot_ext_shard_in_python
    Description:
        1. test_one_hot_ext_data_parallel_2d_int64_2
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_one_hot_ext_data_parallel_2d_int64_2", worker_num=8, local_worker_num=8,
        glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_one_hot_ext_group3():
    """
    Feature: parallel run case in one_hot_ext_shard_in_python
    Description:
        1. test_one_hot_ext_replicate_all_int64_3
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_one_hot_ext_replicate_all_int64_3", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_one_hot_ext_group4():
    """
    Feature: parallel run case in one_hot_ext_shard_in_python
    Description:
        1. test_one_hot_ext_3d_data_parallel_int64_4
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_one_hot_ext_3d_data_parallel_int64_4", worker_num=8, local_worker_num=8,
        glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_one_hot_ext_group5():
    """
    Feature: parallel run case in one_hot_ext_shard_in_python
    Description:
        1. test_one_hot_ext_auto_depth_skewed_distribution_int64_5
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_one_hot_ext_auto_depth_skewed_distribution_int64_5", worker_num=8,
        local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_one_hot_ext_group6():
    """
    Feature: parallel run case in one_hot_ext_shard_in_python
    Description:
        1. test_one_hot_ext_auto_depth_all_same_local_max_int64_6
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_one_hot_ext_auto_depth_all_same_local_max_int64_6", worker_num=8,
        local_worker_num=8, glog_v=2),
    ])
