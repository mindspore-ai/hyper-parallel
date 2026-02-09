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
"""test parallel op squeeze"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_squeeze_basic():
    '''
    Feature: test parallel op squeeze.
    Description: test parallel op squeeze.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_squeeze.py"
    case_name = "test_distributed_squeeze_basic"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_squeeze_no_args_all_dims():
    '''
    Feature: test parallel op squeeze.
    Description: test parallel op squeeze.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_squeeze.py"
    case_name = "test_distributed_squeeze_no_args_all_dims"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_squeeze_specific_axis_negative():
    '''
    Feature: test parallel op squeeze.
    Description: test parallel op squeeze.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_squeeze.py"
    case_name = "test_distributed_squeeze_specific_axis_negative"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_squeeze_error_non_singleton():
    '''
    Feature: test parallel op squeeze.
    Description: test parallel op squeeze.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_squeeze.py"
    case_name = "test_distributed_squeeze_error_non_singleton"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_squeeze_scalar_like():
    '''
    Feature: test parallel op squeeze.
    Description: test parallel op squeeze.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_squeeze.py"
    case_name = "test_distributed_squeeze_scalar_like"
    torchrun_case(file_name, case_name, master_port)
