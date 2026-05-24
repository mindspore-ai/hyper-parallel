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
"""parallel_scatter_update_shell test"""

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_scatter_update.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_scatter_update_group1():
    """
    Feature: parallel run case in scatter_update_shard_in_python
    Description:
        1. test_scatter_update_data_parallel_1
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_scatter_update_data_parallel_1", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_scatter_update_group2():
    """
    Feature: parallel run case in scatter_update_shard_in_python
    Description:
        1. test_scatter_update_model_parallel_2
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_scatter_update_model_parallel_2", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_scatter_update_group3():
    """
    Feature: parallel run case in scatter_update_shard_in_python
    Description:
        1. test_scatter_update_hybrid_parallel_3
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_scatter_update_hybrid_parallel_3", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_scatter_update_group4():
    """
    Feature: parallel run case in scatter_update_shard_in_python
    Description:
        1. test_scatter_update_multi_dim_indices_4
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_scatter_update_multi_dim_indices_4", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_scatter_update_group5():
    """
    Feature: parallel run case in scatter_update_shard_in_python
    Description:
        1. test_scatter_update_replicate_all_5
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_scatter_update_replicate_all_5", worker_num=8, local_worker_num=8, glog_v=2),
    ])
