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
"""parallel_transpose_ext_view_shell test"""

from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_transpose_ext_view_basic_3d_1():
    """
    Feature: TransposeExtView operator.
    Description: Test TransposeExtView swaps two dims on a 3D tensor in python shard.
    Expectation: Run success.
    """
    glog_v = 2
    file_name = "transpose_ext_view_shard_in_python.py"
    case_name = "test_transpose_ext_view_basic_3d_1"
    master_port = 11294
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_transpose_ext_view_negative_dims_2():
    """
    Feature: TransposeExtView operator.
    Description: Test TransposeExtView supports negative dims in python shard.
    Expectation: Run success.
    """
    glog_v = 2
    file_name = "transpose_ext_view_shard_in_python.py"
    case_name = "test_transpose_ext_view_negative_dims_2"
    master_port = 11295
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_transpose_ext_view_same_dims_noop_3():
    """
    Feature: TransposeExtView operator.
    Description: Test TransposeExtView is a no-op when dim0 == dim1 in python shard.
    Expectation: Run success.
    """
    glog_v = 2
    file_name = "transpose_ext_view_shard_in_python.py"
    case_name = "test_transpose_ext_view_same_dims_noop_3"
    master_port = 11296
    msrun_case(glog_v, file_name, case_name, master_port)
