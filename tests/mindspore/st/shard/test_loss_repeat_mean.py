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
"""loss_repeat_mean test"""

from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_loss_repeat_mean_0():
    '''
    Feature: loss repeat mean + partial + hsdp fully shard.
    Description: Test loss repeat mean.
    Expectation: Run success.
    '''
    glog_v = 3
    file_name = "loss_repeat_mean.py"
    case_name = "test_loss_repeat_mean_0"
    master_port = 11296
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_loss_repeat_mean_1():
    '''
    Feature: loss repeat mean + partial + hsdp no fully shard.
    Description: Test loss repeat mean.
    Expectation: Run success.
    '''
    glog_v = 3
    file_name = "loss_repeat_mean.py"
    case_name = "test_loss_repeat_mean_1"
    master_port = 11297
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_loss_repeat_mean_2():
    '''
    Feature: loss repeat mean + partial + hsdp no shard.
    Description: Test loss repeat mean.
    Expectation: Run success.
    '''
    glog_v = 3
    file_name = "loss_repeat_mean.py"
    case_name = "test_loss_repeat_mean_2"
    master_port = 11298
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_loss_repeat_mean_3():
    '''
    Feature: loss repeat mean + partial + hsdp shard size is -1.
    Description: Test loss repeat mean.
    Expectation: Run success.
    '''
    glog_v = 3
    file_name = "loss_repeat_mean.py"
    case_name = "test_loss_repeat_mean_3"
    master_port = 11299
    msrun_case(glog_v, file_name, case_name, master_port)
