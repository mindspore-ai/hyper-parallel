# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""test base dtensor"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_softmax_data_parallel():
    '''
    Feature: test parallel op softmax.
    Description: test parallel op softmax data parallel.
    Expectation: Run success.
    '''

    file_name = "parallel_op_softmax.py"
    case_name = "test_softmax_data_parallel"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_softmax_data_parallel():
    '''
    Feature: test parallel op softmax (gloo cpu).
    Description: test parallel op softmax data parallel.
    Expectation: Run success.
    '''

    file_name = "parallel_op_softmax.py"
    case_name = "test_softmax_data_parallel"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_softmax_model_parallel():
    '''
    Feature: test parallel op softmax.
    Description: test parallel op softmax model parallel.
    Expectation: Run success.
    '''

    file_name = "parallel_op_softmax.py"
    case_name = "test_softmax_model_parallel"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_softmax_model_parallel():
    '''
    Feature: test parallel op softmax (gloo cpu).
    Description: test parallel op softmax model parallel.
    Expectation: Run success.
    '''

    file_name = "parallel_op_softmax.py"
    case_name = "test_softmax_model_parallel"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_softmax_hybrid_parallel():
    '''
    Feature: test parallel op softmax.
    Description: test parallel op softmax hybrid parallel.
    Expectation: Run success.
    '''

    file_name = "parallel_op_softmax.py"
    case_name = "test_softmax_hybrid_parallel"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_softmax_hybrid_parallel():
    '''
    Feature: test parallel op softmax (gloo cpu).
    Description: test parallel op softmax hybrid parallel.
    Expectation: Run success.
    '''

    file_name = "parallel_op_softmax.py"
    case_name = "test_softmax_hybrid_parallel"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_softmax_all_replicated():
    '''
    Feature: test parallel op softmax.
    Description: test parallel op softmax all replicated.
    Expectation: Run success.
    '''

    file_name = "parallel_op_softmax.py"
    case_name = "test_softmax_all_replicated"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_softmax_all_replicated():
    '''
    Feature: test parallel op softmax (gloo cpu).
    Description: test parallel op softmax all replicated.
    Expectation: Run success.
    '''

    file_name = "parallel_op_softmax.py"
    case_name = "test_softmax_all_replicated"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_softmax_negative_dim():
    '''
    Feature: test parallel op softmax.
    Description: test parallel op softmax negative dim.
    Expectation: Run success.
    '''

    file_name = "parallel_op_softmax.py"
    case_name = "test_softmax_negative_dim"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_softmax_negative_dim():
    '''
    Feature: test parallel op softmax (gloo cpu).
    Description: test parallel op softmax negative dim.
    Expectation: Run success.
    '''

    file_name = "parallel_op_softmax.py"
    case_name = "test_softmax_negative_dim"
    torchrun_case(file_name, case_name)
