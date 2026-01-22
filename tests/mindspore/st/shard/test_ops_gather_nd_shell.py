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
"""gathernd_shell test"""

from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gathernd_partial_model_parallel_1():
    '''
    Feature: GatherNd operator.
    Description: Test GatherNd model parallel in python shard.
    Expectation: Run success.
    '''
    glog_v = 2
    file_name = "gather_nd_shard_in_python.py"
    case_name = "test_gathernd_partial_model_parallel_1"
    master_port = 11300
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gathernd_partial_data_parallel_2():
    '''
    Feature: GatherNd operator.
    Description: Test GatherNd data parallel in python shard.
    Expectation: Run success.
    '''
    glog_v = 2
    file_name = "gather_nd_shard_in_python.py"
    case_name = "test_gathernd_partial_data_parallel_2"
    master_port = 11301
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gathernd_params_plain_tensor_3():
    '''
    Feature: GatherNd operator.
    Description: Test GatherNd where params is plain Tensor in python shard.
    Expectation: Run success.
    '''
    glog_v = 2
    file_name = "gather_nd_shard_in_python.py"
    case_name = "test_gathernd_params_plain_tensor"
    master_port = 11303
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gathernd_k1_trailing_dims_shard_4():
    '''
    Feature: GatherNd operator.
    Description: Test GatherNd with K=1 and shard params trailing dims in python shard.
    Expectation: Run success.
    '''
    glog_v = 2
    file_name = "gather_nd_shard_in_python.py"
    case_name = "test_gathernd_k1_trailing_dims_shard_3"
    master_port = 11303
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gathernd_k2_trailing_dim_shard_5():
    '''
    Feature: GatherNd operator.
    Description: Test GatherNd with K=2 and shard params trailing dims in python shard.
    Expectation: Run success.
    '''
    glog_v = 2
    file_name = "gather_nd_shard_in_python.py"
    case_name = "test_gathernd_k2_trailing_dim_shard_4"
    master_port = 11304
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gathernd_k3_no_trailing_dims_6():
    '''
    Feature: GatherNd operator.
    Description: Test GatherNd with K=3 and no params trailing dims in python shard.
    Expectation: Run success.
    '''
    glog_v = 2
    file_name = "gather_nd_shard_in_python.py"
    case_name = "test_gathernd_k3_no_trailing_dims_5"
    master_port = 11305
    msrun_case(glog_v, file_name, case_name, master_port)
