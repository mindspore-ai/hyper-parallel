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
def test_distributed_permute():
    '''
    Feature: test parallel op permute.
    Description: test parallel op permute layout inference and correctness.
    Expectation: Run success.
    '''
    master_port = 10889  # Ensure distinct port if run in parallel
    file_name = "parallel_op_transpose_permute.py"
    case_name = "test_distributed_permute_layout_inference"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_transpose():
    '''
    Feature: test parallel op transpose.
    Description: test parallel op transpose layout inference and correctness.
    Expectation: Run success.
    '''
    master_port = 10890
    file_name = "parallel_op_transpose_permute.py"
    case_name = "test_distributed_transpose_layout_inference"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_transpose_neg():
    '''
    Feature: test parallel op transpose with negative dims.
    Description: test parallel op transpose negative dimension support.
    Expectation: Run success.
    '''
    master_port = 10891
    file_name = "parallel_op_transpose_permute.py"
    case_name = "test_distributed_transpose_negative_dim"
    torchrun_case(file_name, case_name, master_port)
