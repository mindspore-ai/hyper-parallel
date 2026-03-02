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
"""test base dtensor scatter"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_scatter_basic():
    '''
    Feature: test parallel op scatter.
    Description: test parallel op scatter with basic sharding.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_scatter.py"
    case_name = "test_distributed_scatter_basic"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_scatter_scalar_src():
    '''
    Feature: test parallel op scatter.
    Description: test parallel op scatter with scalar source.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_scatter.py"
    case_name = "test_distributed_scatter_scalar_src"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_scatter_sharded_dim_error():
    '''
    Feature: test parallel op scatter.
    Description: test error raising when scattering on sharded dim.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_scatter.py"
    case_name = "test_distributed_scatter_sharded_dim_error"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_scatter_layout_mismatch_index():
    '''
    Feature: test parallel op scatter.
    Description: test error raising when index layout mismatch.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_scatter.py"
    case_name = "test_distributed_scatter_layout_mismatch_index"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_scatter_layout_mismatch_src():
    '''
    Feature: test parallel op scatter.
    Description: test error raising when src layout mismatch.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_scatter.py"
    case_name = "test_distributed_scatter_layout_mismatch_src"
    torchrun_case(file_name, case_name, master_port)
