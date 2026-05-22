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
"""test base dtensor with isinf"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_isinf_replicate():
    '''
    Feature: test parallel op isinf.
    Description: test parallel op isinf on a fully replicated tensor.
    Expectation: Run success.
    '''

    file_name = "parallel_op_isinf.py"
    case_name = "test_isinf_replicate"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_isinf_replicate():
    '''
    Feature: test parallel op isinf (gloo cpu).
    Description: test parallel op isinf on a fully replicated tensor.
    Expectation: Run success.
    '''

    file_name = "parallel_op_isinf.py"
    case_name = "test_isinf_replicate"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_isinf_1d_shard():
    '''
    Feature: test parallel op isinf.
    Description: test parallel op isinf on a 1D sharded tensor.
    Expectation: Run success.
    '''

    file_name = "parallel_op_isinf.py"
    case_name = "test_isinf_1d_shard"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_isinf_1d_shard():
    '''
    Feature: test parallel op isinf (gloo cpu).
    Description: test parallel op isinf on a 1D sharded tensor.
    Expectation: Run success.
    '''

    file_name = "parallel_op_isinf.py"
    case_name = "test_isinf_1d_shard"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_isinf_2d_shard_dim0():
    '''
    Feature: test parallel op isinf.
    Description: test parallel op isinf on a 2D tensor sharded on dim0.
    Expectation: Run success.
    '''

    file_name = "parallel_op_isinf.py"
    case_name = "test_isinf_2d_shard_dim0"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_isinf_2d_shard_dim0():
    '''
    Feature: test parallel op isinf (gloo cpu).
    Description: test parallel op isinf on a 2D tensor sharded on dim0.
    Expectation: Run success.
    '''

    file_name = "parallel_op_isinf.py"
    case_name = "test_isinf_2d_shard_dim0"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_isinf_2d_shard_dim1():
    '''
    Feature: test parallel op isinf.
    Description: test parallel op isinf on a 2D tensor sharded on dim1.
    Expectation: Run success.
    '''

    file_name = "parallel_op_isinf.py"
    case_name = "test_isinf_2d_shard_dim1"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_isinf_2d_shard_dim1():
    '''
    Feature: test parallel op isinf (gloo cpu).
    Description: test parallel op isinf on a 2D tensor sharded on dim1.
    Expectation: Run success.
    '''

    file_name = "parallel_op_isinf.py"
    case_name = "test_isinf_2d_shard_dim1"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_isinf_3d_multi_shard():
    '''
    Feature: test parallel op isinf.
    Description: test parallel op isinf on a 3D tensor with multi-dimension sharding.
    Expectation: Run success.
    '''

    file_name = "parallel_op_isinf.py"
    case_name = "test_isinf_3d_multi_shard"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_isinf_3d_multi_shard():
    '''
    Feature: test parallel op isinf (gloo cpu).
    Description: test parallel op isinf on a 3D tensor with multi-dimension sharding.
    Expectation: Run success.
    '''

    file_name = "parallel_op_isinf.py"
    case_name = "test_isinf_3d_multi_shard"
    torchrun_case(file_name, case_name)
