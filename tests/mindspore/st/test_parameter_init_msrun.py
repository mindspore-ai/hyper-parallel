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
"""test param init with msrun multi card"""
import os
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_param_init_with_tp_dp():
    '''
    Feature: parameter init.
    Description: parameter init with tp and dp.
    Expectation: tp slice init data is difference and dp slice init data is same
    '''
    case_name = "test_param_init_with_tp_dp"
    log = f"log_{case_name}"
    ret = os.system(
        f"msrun --worker_num=8 --local_worker_num=8 --log_dir={log} --join=True --master_port=18281\
            pytest -s test_parameter_init.py::{case_name}"
    )
    assert ret == 0
