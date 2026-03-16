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
"""test base dtensor with atleast_1d"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_atleast_1d_0d():
    '''
    Feature: test parallel op atleast_1d.
    Description: test parallel op atleast_1d with 0-dimensional scalar tensor.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_atleast_1d.py"
    case_name = "test_distributed_atleast_1d_0d"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_atleast_1d_1d():
    '''
    Feature: test parallel op atleast_1d.
    Description: test parallel op atleast_1d with 1-dimensional sharded tensor.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_atleast_1d.py"
    case_name = "test_distributed_atleast_1d_1d"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_atleast_1d_2d():
    '''
    Feature: test parallel op atleast_1d.
    Description: test parallel op atleast_1d with 2-dimensional tensor with mixed sharding.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_atleast_1d.py"
    case_name = "test_distributed_atleast_1d_2d"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_atleast_1d_multiple_tensors():
    '''
    Feature: test parallel op atleast_1d.
    Description: test parallel op atleast_1d with multiple tensors as input arguments.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_atleast_1d.py"
    case_name = "test_distributed_atleast_1d_multiple_tensors"
    torchrun_case(file_name, case_name, master_port)
