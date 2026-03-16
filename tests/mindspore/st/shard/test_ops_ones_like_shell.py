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
"""parallel_ones_like_shell test"""

from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_ones_like_basic_3d_1():
    """
    Feature: OnesLike operator.
    Description: Test OnesLike on a 3D tensor in python shard.
    Expectation: Run success.
    """
    glog_v = 2
    file_name = "ones_like_shard_in_python.py"
    case_name = "test_ones_like_basic_3d_1"
    master_port = 11310
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_ones_like_with_dtype_2():
    """
    Feature: OnesLike operator.
    Description: Test OnesLike with dtype conversion in python shard.
    Expectation: Run success.
    """
    glog_v = 2
    file_name = "ones_like_shard_in_python.py"
    case_name = "test_ones_like_with_dtype_2"
    master_port = 11311
    msrun_case(glog_v, file_name, case_name, master_port)
