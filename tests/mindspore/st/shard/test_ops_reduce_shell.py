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
"""parallel_reduce_shell test"""

from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sum_ext_cell_shard_1():
    '''
    Feature: sum operator.
    Description: Test cell shard in python.
    Expectation: Run success.
    '''
    glog_v = 2
    file_name = "reduce_shard_in_python.py"
    case_name = "test_sum_ext_dim_partial_model_parallel_1"
    master_port = 11290
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_mean_ext_cell_shard_2():
    '''
    Feature: mean operator.
    Description: Test cell shard in python.
    Expectation: Run success.
    '''
    glog_v = 2
    file_name = "reduce_shard_in_python.py"
    case_name = "test_mean_ext_partial_model_parallel_2"
    master_port = 11291
    msrun_case(glog_v, file_name, case_name, master_port)



@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_reduce_max_cell_shard_3():
    '''
    Feature: ReduceMax operator.
    Description: Test cell shard in python.
    Expectation: Run success.
    '''
    glog_v = 2
    file_name = "reduce_shard_in_python.py"
    case_name = "test_reduce_max_partial_model_parallel_3"
    master_port = 11292
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_reduce_max_backward_gradient_4():
    '''
    Feature: ReduceMax operator backward gradient.
    Description: Test ReduceMax backward gradient in distributed training.
    Expectation: Run success.
    '''
    glog_v = 2
    file_name = "reduce_shard_in_python.py"
    case_name = "test_reduce_max_backward_gradient_4"
    master_port = 11293
    msrun_case(glog_v, file_name, case_name, master_port)
