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
"""test sub_mesh"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_full_mesh_shard_forward_1():
    '''
    Feature: dtensor redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11654
    file_name = "submesh.py"
    case_name = "test_full_mesh_shard_forward_1"
    torchrun_case(file_name, case_name, master_port, num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_column_parallel_forward():
    '''
    Feature: dtensor redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11654
    file_name = "submesh.py"
    case_name = "test_sub_mesh_column_parallel_forward"
    torchrun_case(file_name, case_name, master_port, num_proc=4)


# @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_full_mesh_shard_forward_2():
    '''
    Feature: dtensor redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11654
    file_name = "submesh.py"
    case_name = "test_full_mesh_shard_forward_2"
    torchrun_case(file_name, case_name, master_port, num_proc=4)


# @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_row_parallel_forward():
    '''
    Feature: dtensor redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11654
    file_name = "submesh.py"
    case_name = "test_sub_mesh_row_parallel_forward"
    torchrun_case(file_name, case_name, master_port, num_proc=4)


# @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_row_parallel_redistribute_forward():
    '''
    Feature: dtensor redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11654
    file_name = "submesh.py"
    case_name = "test_sub_mesh_row_parallel_redistribute_forward"
    torchrun_case(file_name, case_name, master_port, num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_redistribute_1():
    '''
    Feature: dtensor redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11654
    file_name = "submesh.py"
    case_name = "test_sub_mesh_redistribute_1"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_redistribute_2():
    '''
    Feature: dtensor redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11654
    file_name = "submesh.py"
    case_name = "test_sub_mesh_redistribute_2"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_redistribute_3():
    '''
    Feature: dtensor redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11654
    file_name = "submesh.py"
    case_name = "test_sub_mesh_redistribute_3"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_redistribute_4():
    '''
    Feature: dtensor redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11654
    file_name = "submesh.py"
    case_name = "test_sub_mesh_redistribute_4"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_redistribute_5():
    '''
    Feature: dtensor redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11654
    file_name = "submesh.py"
    case_name = "test_sub_mesh_redistribute_5"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_redistribute_6():
    '''
    Feature: dtensor redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11654
    file_name = "submesh.py"
    case_name = "test_sub_mesh_redistribute_6"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_redistribute_7():
    '''
    Feature: dtensor redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11654
    file_name = "submesh.py"
    case_name = "test_sub_mesh_redistribute_7"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_redistribute_8():
    '''
    Feature: dtensor redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11654
    file_name = "submesh.py"
    case_name = "test_sub_mesh_redistribute_8"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_redistribute_9():
    '''
    Feature: dtensor redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11654
    file_name = "submesh.py"
    case_name = "test_sub_mesh_redistribute_9"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sub_mesh_redistribute_10():
    '''
    Feature: dtensor redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11654
    file_name = "submesh.py"
    case_name = "test_sub_mesh_redistribute_10"
    torchrun_case(file_name, case_name, master_port)
