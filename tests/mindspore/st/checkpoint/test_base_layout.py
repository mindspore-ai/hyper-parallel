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
"""layout save load test"""

from hyper_parallel.core.checkpoint.layout import combine_layout
from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_base_layout():
    """
    Feature: layout save, load and combine.
    Description: save layout, load layout, and combine layout.
    Expectation: Run success.
    """
    glog_v = 3
    file_name = "base_shard.py"
    case_name = "test_base_layout"
    master_port = 11222
    msrun_case(glog_v, file_name, case_name, master_port)
    layout_dict = combine_layout(".")
    assert isinstance(layout_dict, dict)
