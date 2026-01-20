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
def test_distributed_reshape():
    '''
    Feature: test parallel op reshape.
    Description: test parallel op reshape (flatten and expand).
    Expectation: Run success.
    '''
    master_port = 10889
    file_name = "parallel_op_reshape_view.py"
    case_name = "test_distributed_reshape_layout_inference"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_view():
    '''
    Feature: test parallel op view.
    Description: test parallel op view (tuple args and preserving sharded dim).
    Expectation: Run success.
    '''
    master_port = 10890
    file_name = "parallel_op_reshape_view.py"
    case_name = "test_distributed_view_layout_inference"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_reshape_error():
    '''
    Feature: test parallel op reshape error.
    Description: test parallel op reshape invalid cases.
    Expectation: Run success.
    '''
    master_port = 10891
    file_name = "parallel_op_reshape_view.py"
    case_name = "test_distributed_reshape_fail_mismatch"
    torchrun_case(file_name, case_name, master_port)
