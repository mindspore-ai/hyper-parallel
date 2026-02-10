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
"""test base dtensor unbind"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_unbind_dim0():
    '''
    Feature: test parallel op unbind.
    Description: test parallel op unbind on dim 0.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_unbind.py"
    case_name = "test_distributed_unbind_dim0"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_unbind_dim1():
    '''
    Feature: test parallel op unbind.
    Description: test parallel op unbind on dim 1.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_unbind.py"
    case_name = "test_distributed_unbind_dim1"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_unbind_negative_dim():
    '''
    Feature: test parallel op unbind.
    Description: test parallel op unbind with negative dim.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_unbind.py"
    case_name = "test_distributed_unbind_negative_dim"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_unbind_sharded_dim_error():
    '''
    Feature: test parallel op unbind.
    Description: test parallel op unbind error handling.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_unbind.py"
    case_name = "test_distributed_unbind_sharded_dim_error"
    torchrun_case(file_name, case_name, master_port)
