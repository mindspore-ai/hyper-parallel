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
"""test_process_group.py"""
from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_process_group():
    """
    Feature: Init process group, get rank list and backend in group, then destroy the group.
    Description: Test init process group with backend is ``hccl``, then try to get rank list and backend in this
        process group. After that, create a sub process group and get rank list and backend in sub process group.
        Finally, destroy the sub process group and process group.
    Expectation: Run success.
    """
    glog_v = 3
    file_name = "process_group.py"
    case_name = "test_process_group"
    master_port = 10111
    msrun_case(glog_v, file_name, case_name, master_port)
