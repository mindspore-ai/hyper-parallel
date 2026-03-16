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
"""test base dtensor dropout op"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_dropout_basic_sharded():
    '''
    Feature: test parallel op dropout.
    Description: test parallel op dropout with basic sharding.
    Expectation: Run success.
    '''
    master_port = 10360
    file_name = "parallel_op_dropout.py"
    case_name = "test_distributed_dropout_basic_sharded"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_dropout_p0_exact_match():
    '''
    Feature: test parallel op dropout.
    Description: test parallel op dropout exact numerical match when p=0.0.
    Expectation: Run success.
    '''
    master_port = 10360
    file_name = "parallel_op_dropout.py"
    case_name = "test_distributed_dropout_p0_exact_match"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_dropout_3d():
    '''
    Feature: test parallel op dropout.
    Description: test parallel op dropout with a 3D sharded tensor.
    Expectation: Run success.
    '''
    master_port = 10360
    file_name = "parallel_op_dropout.py"
    case_name = "test_distributed_dropout_3d"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_dropout_replicate():
    '''
    Feature: test parallel op dropout.
    Description: test parallel op dropout with a fully replicated tensor.
    Expectation: Run success.
    '''
    master_port = 10360
    file_name = "parallel_op_dropout.py"
    case_name = "test_distributed_dropout_replicate"
    torchrun_case(file_name, case_name, master_port)
