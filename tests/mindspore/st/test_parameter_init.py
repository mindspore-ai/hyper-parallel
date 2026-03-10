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
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

PARAMETER_INIT = "parameter_init.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_parameter_init_group1():
    """
    Feature: parallel run case in parameter_init
    Description:
        1. test_init_parameters_with_hsdp
        2. test_init_parameters_with_dtensor_and_hsdp
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(PARAMETER_INIT, "test_init_parameters_with_hsdp", 18281),
        MindSporeCase(PARAMETER_INIT, "test_init_parameters_with_dtensor_and_hsdp", 18282)
    ])


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_parameter_init_group2():
    """
    Feature: parallel run case in parameter_init
    Description:
        1. test_init_parameters_mem
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(PARAMETER_INIT, "test_init_parameters_mem", 18283)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parameter_init_group3():
    """
    Feature: parallel run case in parameter_init
    Description:
        1. test_param_init_with_tp_dp
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(PARAMETER_INIT, "test_param_init_with_tp_dp", 18284, 8, 8)
    ])
