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
"""test base dtensor remainder operation"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_remainder_basic():
    '''
    Feature: test parallel op remainder.
    Description: test parallel op remainder with basic identical layout tensors.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_remainder.py"
    case_name = "test_distributed_remainder_basic"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_remainder_scalar():
    '''
    Feature: test parallel op remainder.
    Description: test parallel op remainder with scalar operand.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_remainder.py"
    case_name = "test_distributed_remainder_scalar"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_remainder_broadcast():
    '''
    Feature: test parallel op remainder.
    Description: test parallel op remainder with broadcasting shapes.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_remainder.py"
    case_name = "test_distributed_remainder_broadcast"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_remainder_operator_overload():
    '''
    Feature: test parallel op remainder.
    Description: test parallel op modulo (%) overload.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_remainder.py"
    case_name = "test_distributed_remainder_operator_overload"
    torchrun_case(file_name, case_name, master_port)
