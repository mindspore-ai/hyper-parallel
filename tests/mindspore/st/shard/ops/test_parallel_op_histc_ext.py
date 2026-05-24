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
"""
Shell file for HistcExt distributed operator integration tests.
"""

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_histc_ext.py")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential"
)
def test_parallel_op_histc_ext_group1():
    """
    Feature: parallel run case in gather_nd_shard_in_python
    Description:
        1. test_histc_ext_data_parallel1
        2. test_histc_ext_model_parallel2
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_histc_ext_data_parallel1", worker_num=4, local_worker_num=4, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_histc_ext_model_parallel2", worker_num=4, local_worker_num=4, glog_v=2),
    ])


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential"
)
def test_parallel_op_histc_ext_group2():
    """
    Feature: parallel run case in gather_nd_shard_in_python
    Description:
        1. test_histc_ext_hybrid_parallel3
        2. test_histc_ext_all_replicated4
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_histc_ext_hybrid_parallel3", worker_num=4, local_worker_num=4, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_histc_ext_all_replicated4", worker_num=4, local_worker_num=4, glog_v=2),
    ])
