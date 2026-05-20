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
"""test in-place division (div_) dtensor"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_div__identical_sharding():
    '''
    Feature: test parallel op div_ (identical sharding).
    Description: test parallel op div_.
    Expectation: Run success.
    '''

    file_name = "parallel_op_div_.py"
    case_name = "test_div__identical_sharding"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_div__identical_sharding():
    '''
    Feature: test parallel op div_ (identical sharding, gloo cpu).
    Description: test parallel op div_.
    Expectation: Run success.
    '''

    file_name = "parallel_op_div_.py"
    case_name = "test_div__identical_sharding"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_div__broadcast():
    '''
    Feature: test parallel op div_ (broadcasting).
    Description: test parallel op div_.
    Expectation: Run success.
    '''

    file_name = "parallel_op_div_.py"
    case_name = "test_div__broadcast"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_div__broadcast():
    '''
    Feature: test parallel op div_ (broadcasting, gloo cpu).
    Description: test parallel op div_.
    Expectation: Run success.
    '''

    file_name = "parallel_op_div_.py"
    case_name = "test_div__broadcast"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_div__scalar():
    '''
    Feature: test parallel op div_ (scalar).
    Description: test parallel op div_.
    Expectation: Run success.
    '''

    file_name = "parallel_op_div_.py"
    case_name = "test_div__scalar"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_div__scalar():
    '''
    Feature: test parallel op div_ (scalar, gloo cpu).
    Description: test parallel op div_.
    Expectation: Run success.
    '''

    file_name = "parallel_op_div_.py"
    case_name = "test_div__scalar"
    torchrun_case(file_name, case_name)
