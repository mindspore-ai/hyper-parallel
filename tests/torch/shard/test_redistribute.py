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
"""test redistribute"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_shard_to_replicate():
    '''
    Feature: dtensor redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11333
    case_name = "redistribute.py::test_shard_to_replicate"
    torchrun_case(master_port, case_name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_replicate_to_shard():
    '''
    Feature: dtensor redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11334
    case_name = "redistribute.py::test_replicate_to_shard"
    torchrun_case(master_port, case_name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_different_mesh():
    '''
    Feature: dtensor redistribute.
    Description: 
    Expectation: Run success.
    '''
    master_port = 11335
    case_name = "redistribute.py::test_different_mesh"
    torchrun_case(master_port, case_name)
