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
"""test_symmetric_memory.py"""
from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_symmetric_memory_put_mem():
    """
    Feature: symmetric memory put mem.
    Description: Test symmetric memory put mem api.
    Expectation: symmetric memory put mem api performs as expected.
    """
    glog_v = 2
    file_name = "symmetric_memory.py"
    case_name = "test_ms_symmetric_memory_put_mem"
    master_port = 10134
    worker_num = 2
    msrun_case(glog_v, file_name, case_name, master_port, worker_num, worker_num)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_symmetric_memory_shmem_put_with_signal():
    """
    Feature: symmetric memory put mem with signal.
    Description: Test symmetric memory put mem with signal api.
    Expectation: symmetric memory put mem with signal api performs as expected.
    """
    glog_v = 2
    file_name = "symmetric_memory.py"
    case_name = "test_ms_symmetric_memory_shmem_put_with_signal"
    master_port = 10135
    worker_num = 2
    msrun_case(glog_v, file_name, case_name, master_port, worker_num, worker_num)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_symmetric_memory_shmem_allgather():
    """
    Feature: symmetric memory allgather.
    Description: Test symmetric memory allgather api.
    Expectation: symmetric memory allgather api performs as expected.
    """
    glog_v = 2
    file_name = "symmetric_memory.py"
    case_name = "test_ms_symmetric_memory_shmem_allgather"
    master_port = 10136
    worker_num = 4
    msrun_case(glog_v, file_name, case_name, master_port, worker_num, worker_num)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_symmetric_memory_shmem_wait_for_signal():
    """
    Feature: symmetric memory wait for signal.
    Description: Test symmetric memory wait for signal api.
    Expectation: symmetric memory wait for signal api performs as expected.
    """
    glog_v = 2
    file_name = "symmetric_memory.py"
    case_name = "test_ms_symmetric_memory_shmem_wait_for_signal"
    master_port = 10137
    worker_num = 2
    msrun_case(glog_v,file_name, case_name, master_port, worker_num, worker_num)
