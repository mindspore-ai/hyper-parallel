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
"""test base dtensor"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_isin():
    '''
    Feature: test parallel op isin.
    Description: test parallel op isin.
    Expectation: Run success.
    '''
    master_port = 10458
    file_name = "parallel_op_isin.py"
    case_name = "test_distributed_isin_layout_inference"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_isin_sharded_test_elements():
    '''
    Feature: test parallel op isin.
    Description: test parallel op isin.
    Expectation: Run success.
    '''
    master_port = 10458
    file_name = "parallel_op_isin.py"
    case_name = "test_distributed_isin_sharded_test_elements_error"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_isin_invert_and_assume_unique():
    '''
    Feature: test parallel op isin.
    Description: test parallel op isin.
    Expectation: Run success.
    '''
    master_port = 10458
    file_name = "parallel_op_isin.py"
    case_name = "test_distributed_isin_invert_and_assume_unique"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_isin_mixed_parallel_3d_mixed_parallel_3d():
    '''
    Feature: test parallel op isin.
    Description: test parallel op isin.
    Expectation: Run success.
    '''
    master_port = 10458
    file_name = "parallel_op_isin.py"
    case_name = "test_distributed_isin_mixed_parallel_3d"
    torchrun_case(file_name, case_name, master_port)
