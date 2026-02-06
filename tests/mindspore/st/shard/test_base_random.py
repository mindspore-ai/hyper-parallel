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
"""parallel_base_shard test"""

from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_tracker_initialization():
    '''
    Feature: with no_init_parameters + cell shard + hsdp + init param + loss repeat + partial.
    Description: Test base shard.
    Expectation: Run success.
    '''
    glog_v = 3
    file_name = "base_random.py"
    case_name = "test_tracker_initialization"
    master_port = 11333
    msrun_case(glog_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distribute_region_disabled():
    '''
    Feature: with no_init_parameters + cell shard + hsdp + init param + loss repeat + partial.
    Description: Test base shard.
    Expectation: Run success.
    '''
    glog_v = 3
    file_name = "base_random.py"
    case_name = "test_distribute_region_disabled"
    master_port = 11333
    msrun_case(glog_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_multi_dim_sharding_offset():
    '''
    Feature: with no_init_parameters + cell shard + hsdp + init param + loss repeat + partial.
    Description: Test base shard.
    Expectation: Run success.
    '''
    glog_v = 3
    file_name = "base_random.py"
    case_name = "test_multi_dim_sharding_offset"
    master_port = 11333
    msrun_case(glog_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_rng_tracker():
    '''
    Feature: with no_init_parameters + cell shard + hsdp + init param + loss repeat + partial.
    Description: Test base shard.
    Expectation: Run success.
    '''
    glog_v = 3
    file_name = "base_random.py"
    case_name = "test_rng_tracker"
    master_port = 11333
    msrun_case(glog_v, file_name, case_name, master_port)
