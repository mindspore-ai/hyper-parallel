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

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_elementwise.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_elementwise_group1():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_minimum_same_shape_parallel_1
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_minimum_same_shape_parallel_1", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_elementwise_group2():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_less_equal_same_shape_parallel_2
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_less_equal_same_shape_parallel_2", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_elementwise_group3():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_greater_equal_same_shape_parallel_3
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_greater_equal_same_shape_parallel_3", worker_num=8, local_worker_num=8,
            glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_elementwise_group4():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_logical_or_same_shape_parallel_4
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_logical_or_same_shape_parallel_4", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_elementwise_group5():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_minimum_broadcast_dim0_parallel_5
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_minimum_broadcast_dim0_parallel_5", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_elementwise_group6():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_minimum_broadcast_dim1_parallel_6
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_minimum_broadcast_dim1_parallel_6", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_elementwise_group7():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_minimum_broadcast_dim2_parallel_7
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_minimum_broadcast_dim2_parallel_7", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_elementwise_group8():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_minimum_broadcast_rank_mismatch_parallel_8
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_minimum_broadcast_rank_mismatch_parallel_8", worker_num=8, local_worker_num=8,
        glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_elementwise_group9():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_minimum_broadcast_scalar_like_parallel_9
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_minimum_broadcast_scalar_like_parallel_9", worker_num=8, local_worker_num=8,
        glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_elementwise_group10():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_less_equal_broadcast_multi_dim_parallel_10
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_less_equal_broadcast_multi_dim_parallel_10", worker_num=8, local_worker_num=8,
        glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_elementwise_group11():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_minimum_partial_shard_parallel_11
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_minimum_partial_shard_parallel_11", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_elementwise_group12():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_mod_same_shape_parallel_12
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_mod_same_shape_parallel_12", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_elementwise_group13():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_mod_broadcast_dim0_parallel_13
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_mod_broadcast_dim0_parallel_13", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_elementwise_group14():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_mod_broadcast_dim1_parallel_14
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_mod_broadcast_dim1_parallel_14", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_elementwise_group15():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_mod_broadcast_dim2_parallel_15
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_mod_broadcast_dim2_parallel_15", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_elementwise_group16():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_mod_broadcast_rank_mismatch_parallel_16
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_mod_broadcast_rank_mismatch_parallel_16", worker_num=8, local_worker_num=8,
        glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_elementwise_group17():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_mod_tensor_scalar_parallel_17
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_mod_tensor_scalar_parallel_17", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_elementwise_group18():
    """
    Feature: parallel run case in elementwise_shard_in_python
    Description:
        1. test_mod_partial_shard_parallel_18
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_mod_partial_shard_parallel_18", worker_num=8, local_worker_num=8, glog_v=2),
    ])
