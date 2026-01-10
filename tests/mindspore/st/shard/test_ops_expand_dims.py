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
"""parallel_expand_dims_shell test"""

from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_expanddims_cell_shard_1():
    '''
    Feature: ExpandDims operator.
    Description: Test cell shard in python with data parallel.
    Expectation: Run success.
    '''
    glog_v = 2
    file_name = "expand_dims_shard_in_python.py"
    case_name = "test_expanddims_data_parallel_1"
    master_port = 11300
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_expanddims_cell_shard_2():
    '''
    Feature: ExpandDims operator.
    Description: Test cell shard in python with model parallel.
    Expectation: Run success.
    '''
    glog_v = 2
    file_name = "expand_dims_shard_in_python.py"
    case_name = "test_expanddims_model_parallel_2"
    master_port = 11301
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_expanddims_cell_shard_3():
    '''
    Feature: ExpandDims operator.
    Description: Test cell shard in python with hybrid parallel.
    Expectation: Run success.
    '''
    glog_v = 2
    file_name = "expand_dims_shard_in_python.py"
    case_name = "test_expanddims_hybrid_parallel_3"
    master_port = 11302
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_expanddims_cell_shard_4():
    '''
    Feature: ExpandDims operator.
    Description: Test cell shard in python, insert in middle.
    Expectation: Run success.
    '''
    glog_v = 2
    file_name = "expand_dims_shard_in_python.py"
    case_name = "test_expanddims_insert_middle_4"
    master_port = 11303
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_expanddims_cell_shard_5():
    '''
    Feature: ExpandDims operator.
    Description: Test cell shard in python with negative axis.
    Expectation: Run success.
    '''
    glog_v = 2
    file_name = "expand_dims_shard_in_python.py"
    case_name = "test_expanddims_negative_axis_5"
    master_port = 11304
    msrun_case(glog_v, file_name, case_name, master_port)
