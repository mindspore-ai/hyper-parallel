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
def test_distributed_expand_basic_unsharded():
    '''
    Feature: test parallel op expand.
    Description: test parallel op expand.
    Expectation: Run success.
    '''
    master_port = 10358
    file_name = "parallel_op_expand.py"
    case_name = "test_distributed_expand_basic_unsharded"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_expand_3d():
    '''
    Feature: test parallel op expand.
    Description: test parallel op expand.
    Expectation: Run success.
    '''
    master_port = 10358
    file_name = "parallel_op_expand.py"
    case_name = "test_distributed_expand_3d"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_expand_prepend_new_dimensions():
    '''
    Feature: test parallel op expand.
    Description: test parallel op expand.
    Expectation: Run success.
    '''
    master_port = 10358
    file_name = "parallel_op_expand.py"
    case_name = "test_distributed_expand_prepend_new_dimensions"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_expand_sharded_dim_error():
    '''
    Feature: test parallel op expand.
    Description: test parallel op expand.
    Expectation: Run success.
    '''
    master_port = 10358
    file_name = "parallel_op_expand.py"
    case_name = "test_distributed_expand_sharded_dim_error"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_expand_scalar_tensor():
    '''
    Feature: test parallel op expand.
    Description: test parallel op expand.
    Expectation: Run success.
    '''
    master_port = 10358
    file_name = "parallel_op_expand.py"
    case_name = "test_distributed_expand_scalar_tensor"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_expand_as_basic():
    '''
    Feature: test parallel op expand.
    Description: test parallel op expand.
    Expectation: Run success.
    '''
    master_port = 10358
    file_name = "parallel_op_expand.py"
    case_name = "test_distributed_expand_as_basic"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_expand_as_3d_preservation():
    '''
    Feature: test parallel op expand.
    Description: test parallel op expand.
    Expectation: Run success.
    '''
    master_port = 10358
    file_name = "parallel_op_expand.py"
    case_name = "test_distributed_expand_as_3d_preservation"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_expand_as_prepend_dimensions():
    '''
    Feature: test parallel op expand.
    Description: test parallel op expand.
    Expectation: Run success.
    '''
    master_port = 10358
    file_name = "parallel_op_expand.py"
    case_name = "test_distributed_expand_as_prepend_dimensions"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_expand_as_scalar_to_tensor():
    '''
    Feature: test parallel op expand.
    Description: test parallel op expand.
    Expectation: Run success.
    '''
    master_port = 10358
    file_name = "parallel_op_expand.py"
    case_name = "test_distributed_expand_as_scalar_to_tensor"
    torchrun_case(file_name, case_name, master_port)
