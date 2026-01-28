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

from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_one_hot_ext_cell_shard_1():
    """
    Feature: OneHotExt operator.
    Description: Test cell shard in python with 1D tensor, data parallel on tensor dim 0.
    Expectation: Run success.
    """
    glog_v = 2
    file_name = "one_hot_ext_shard_in_python.py"
    case_name = "test_one_hot_ext_data_parallel_1d_int64_1"
    master_port = 12300
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_one_hot_ext_cell_shard_2():
    """
    Feature: OneHotExt operator.
    Description: Test cell shard in python with 2D tensor, data parallel on tensor dim 0.
    Expectation: Run success.
    """
    glog_v = 2
    file_name = "one_hot_ext_shard_in_python.py"
    case_name = "test_one_hot_ext_data_parallel_2d_int64_2"
    master_port = 12301
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_one_hot_ext_cell_shard_3():
    """
    Feature: OneHotExt operator.
    Description: Test cell shard in python with full replication.
    Expectation: Run success.
    """
    glog_v = 2
    file_name = "one_hot_ext_shard_in_python.py"
    case_name = "test_one_hot_ext_replicate_all_int64_3"
    master_port = 12302
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_one_hot_ext_cell_shard_4():
    """
    Feature: OneHotExt operator.
    Description: Test cell shard in python with 3D tensor, data parallel on first dim only.
    Expectation: Run success.
    """
    glog_v = 2
    file_name = "one_hot_ext_shard_in_python.py"
    case_name = "test_one_hot_ext_3d_data_parallel_int64_4"
    master_port = 12303
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_one_hot_ext_cell_shard_5():
    """
    Feature: OneHotExt operator with auto depth.
    Description:
        Test cell shard in python with 1D tensor, data parallel, num_classes=-1.
        Only one dp shard has the global maximum to enforce global max reduction.
    Expectation: Run success with correct global max reduction.
    """
    glog_v = 2
    file_name = "one_hot_ext_shard_in_python.py"
    case_name = "test_one_hot_ext_auto_depth_skewed_distribution_int64_5"
    master_port = 12304
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_one_hot_ext_cell_shard_6():
    """
    Feature: OneHotExt operator with auto depth.
    Description:
        Test cell shard in python with 1D tensor, data parallel, num_classes=-1.
        All dp shards share the same local max (easy case).
    Expectation: Run success with correct depth.
    """
    glog_v = 2
    file_name = "one_hot_ext_shard_in_python.py"
    case_name = "test_one_hot_ext_auto_depth_all_same_local_max_int64_6"
    master_port = 12305
    msrun_case(glog_v, file_name, case_name, master_port)
