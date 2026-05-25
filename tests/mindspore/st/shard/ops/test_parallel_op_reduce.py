# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""Test runner for reduce and max_dim distributed ST (MindSpore)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_reduce.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_reduce_group1():
    """
    Feature: parallel run case in _test_parallel_op_reduce
    Description:
        1. test_sum_ext_dim_partial_model_parallel_1 — SumExt partial model parallel
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_sum_ext_dim_partial_model_parallel_1", worker_num=8, local_worker_num=8,
            glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_reduce_group2():
    """
    Feature: parallel run case in _test_parallel_op_reduce
    Description:
        1. test_mean_ext_partial_model_parallel_2 — MeanExt partial model parallel
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_mean_ext_partial_model_parallel_2", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_reduce_group3():
    """
    Feature: parallel run case in _test_parallel_op_reduce
    Description:
        1. test_reduce_max_partial_model_parallel_3 — ReduceMax partial model parallel
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_reduce_max_partial_model_parallel_3", worker_num=8, local_worker_num=8,
        glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_reduce_group4():
    """
    Feature: parallel run case in _test_parallel_op_reduce
    Description:
        1. test_reduce_max_backward_gradient_4 — ReduceMax backward gradient
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_reduce_max_backward_gradient_4", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_reduce_group5():
    """
    Feature: parallel run case in _test_parallel_op_reduce
    Description:
        1. test_max_dim_data_parallel_1 — MaxDim data parallel
        2. test_max_dim_model_parallel_2 — MaxDim model parallel
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_max_dim_data_parallel_1", worker_num=4, local_worker_num=4, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_max_dim_model_parallel_2", worker_num=4, local_worker_num=4, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_reduce_group6():
    """
    Feature: parallel run case in _test_parallel_op_reduce
    Description:
        1. test_max_dim_negative_dim_3 — MaxDim negative dimension
        2. test_max_dim_keepdim_false_4 — MaxDim keepdim=False
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_max_dim_negative_dim_3", worker_num=4, local_worker_num=4, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_max_dim_keepdim_false_4", worker_num=4, local_worker_num=4, glog_v=2),
    ])
