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
"""test new_ones dtensor"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_new_ones_tuple_size():
    '''
    Feature: test parallel op new_ones.
    Description: test parallel op new_ones with tuple size.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_new_ones.py"
    case_name = "test_distributed_new_ones_tuple_size"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_new_ones_list_size():
    '''
    Feature: test parallel op new_ones.
    Description: test parallel op new_ones with list size.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_new_ones.py"
    case_name = "test_distributed_new_ones_list_size"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_new_ones_int_size():
    '''
    Feature: test parallel op new_ones.
    Description: test parallel op new_ones with int size.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_new_ones.py"
    case_name = "test_distributed_new_ones_int_size"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_new_ones_scalar():
    '''
    Feature: test parallel op new_ones.
    Description: test parallel op new_ones scalar.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_new_ones.py"
    case_name = "test_distributed_new_ones_scalar"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_new_ones_input_sharding_ignored():
    '''
    Feature: test parallel op new_ones.
    Description: test parallel op new_ones ignores input sharding.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_new_ones.py"
    case_name = "test_distributed_new_ones_input_sharding_ignored"
    torchrun_case(file_name, case_name, master_port)
