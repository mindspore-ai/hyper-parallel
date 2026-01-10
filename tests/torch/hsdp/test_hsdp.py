# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test hybrid shard data parallel"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_pure_data_parallel():
    '''
    Feature: pure data parallel
    Description: 
    Expectation: Run successfully.
    '''
    master_port = 12341
    file_name = "hsdp.py"
    case_name = "test_pure_data_parallel"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero1_fully_shard():
    '''
    Feature: zero1 fully shard data parallel
    Description: 
    Expectation: Run successfully.
    '''
    master_port = 12342
    file_name = "hsdp.py"
    case_name = "test_zero1_fully_shard"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero1_partial_shard():
    '''
    Feature: zero1 partial shard data parallel
    Description: 
    Expectation: Run successfully.
    '''
    master_port = 12343
    file_name = "hsdp.py"
    case_name = "test_zero1_partial_shard"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero2_fully_shard():
    '''
    Feature: zero2 fully shard data parallel
    Description: 
    Expectation: Run successfully.
    '''
    master_port = 12344
    file_name = "hsdp.py"
    case_name = "test_zero2_fully_shard"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero2_partial_shard():
    '''
    Feature: zero2 partial shard data parallel
    Description: 
    Expectation: Run successfully.
    '''
    master_port = 12345
    file_name = "hsdp.py"
    case_name = "test_zero2_partial_shard"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero3_fully_shard():
    '''
    Feature: zero3 fully shard data parallel
    Description: 
    Expectation: Run successfully.
    '''
    master_port = 12346
    file_name = "hsdp.py"
    case_name = "test_zero3_fully_shard"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero3_partial_shard():
    '''
    Feature: zero3 partial shard data parallel
    Description: 
    Expectation: Run successfully.
    '''
    master_port = 12347
    file_name = "hsdp.py"
    case_name = "test_zero3_partial_shard"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero1_fully_shard_with_acc_grad():
    '''
    Feature: zero1 fully shard data parallel with acc grad
    Description: 
    Expectation: Run successfully.
    '''
    master_port = 12342
    file_name = "hsdp.py"
    case_name = "test_zero1_fully_shard_with_acc_grad"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero1_partial_shard_with_acc_grad():
    '''
    Feature: zero1 partial shard data parallel with acc grad
    Description: 
    Expectation: Run successfully.
    '''
    master_port = 12343
    file_name = "hsdp.py"
    case_name = "test_zero1_partial_shard_with_acc_grad"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero2_fully_shard_with_acc_grad():
    '''
    Feature: zero2 fully shard data parallel with acc grad
    Description: 
    Expectation: Run successfully.
    '''
    master_port = 12344
    file_name = "hsdp.py"
    case_name = "test_zero2_fully_shard_with_acc_grad"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero2_partial_shard_with_acc_grad():
    '''
    Feature: zero2 partial shard data parallel with acc grad
    Description: 
    Expectation: Run successfully.
    '''
    master_port = 12345
    file_name = "hsdp.py"
    case_name = "test_zero2_partial_shard_with_acc_grad"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero3_fully_shard_with_acc_grad():
    '''
    Feature: zero3 fully shard data parallel with acc grad
    Description: 
    Expectation: Run successfully.
    '''
    master_port = 12346
    file_name = "hsdp.py"
    case_name = "test_zero3_fully_shard_with_acc_grad"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero3_partial_shard_with_acc_grad():
    '''
    Feature: zero3 partial shard data parallel with acc grad
    Description: 
    Expectation: Run successfully.
    '''
    master_port = 12347
    file_name = "hsdp.py"
    case_name = "test_zero3_partial_shard_with_acc_grad"
    torchrun_case(file_name, case_name, master_port)
