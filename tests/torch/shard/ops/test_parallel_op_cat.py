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
"""test parallel op cat"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_cat_basic():
    '''
    Feature: test parallel op cat.
    Description: basic cat with aligned shards.
    Expectation: Run success.
    '''

    file_name = "parallel_op_cat.py"
    case_name = "test_cat_basic"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_cat_basic():
    '''
    Feature: test parallel op cat (gloo cpu).
    Description: basic cat with aligned shards.
    Expectation: Run success.
    '''

    file_name = "parallel_op_cat.py"
    case_name = "test_cat_basic"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_cat_3d_complex():
    '''
    Feature: test parallel op cat.
    Description: 3D cat on complex device mesh.
    Expectation: Run success.
    '''

    file_name = "parallel_op_cat.py"
    case_name = "test_cat_3d_complex"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_cat_3d_complex():
    '''
    Feature: test parallel op cat (gloo cpu).
    Description: 3D cat on complex device mesh.
    Expectation: Run success.
    '''

    file_name = "parallel_op_cat.py"
    case_name = "test_cat_3d_complex"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_cat_multiple_tensors():
    '''
    Feature: test parallel op cat.
    Description: Concatenate more than 2 tensors at once.
    Expectation: Run success.
    '''

    file_name = "parallel_op_cat.py"
    case_name = "test_cat_multiple_tensors"
    torchrun_case(file_name, case_name, num_proc=8)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_cat_multiple_tensors():
    '''
    Feature: test parallel op cat (gloo cpu).
    Description: Concatenate more than 2 tensors at once.
    Expectation: Run success.
    '''

    file_name = "parallel_op_cat.py"
    case_name = "test_cat_multiple_tensors"
    torchrun_case(file_name, case_name, num_proc=8)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_cat_mismatched_shapes():
    '''
    Feature: test parallel op cat.
    Description: Concatenate tensors with differing sizes in the target dimension.
    Expectation: Run success.
    '''

    file_name = "parallel_op_cat.py"
    case_name = "test_cat_mismatched_shapes"
    torchrun_case(file_name, case_name, num_proc=4)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_cat_mismatched_shapes():
    '''
    Feature: test parallel op cat (gloo cpu).
    Description: Concatenate tensors with differing sizes in the target dimension.
    Expectation: Run success.
    '''

    file_name = "parallel_op_cat.py"
    case_name = "test_cat_mismatched_shapes"
    torchrun_case(file_name, case_name, num_proc=4)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_cat_with_empty():
    '''
    Feature: test parallel op cat.
    Description: Concatenate with a dimension size of zero (empty tensor).
    Expectation: Run success.
    '''

    file_name = "parallel_op_cat.py"
    case_name = "test_cat_with_empty"
    torchrun_case(file_name, case_name, num_proc=4)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_cat_with_empty():
    '''
    Feature: test parallel op cat (gloo cpu).
    Description: Concatenate with a dimension size of zero (empty tensor).
    Expectation: Run success.
    '''

    file_name = "parallel_op_cat.py"
    case_name = "test_cat_with_empty"
    torchrun_case(file_name, case_name, num_proc=4)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_cat_4d_tensor():
    '''
    Feature: test parallel op cat.
    Description: Concatenate 4D tensors on a 2D device mesh.
    Expectation: Run success.
    '''

    file_name = "parallel_op_cat.py"
    case_name = "test_cat_4d_tensor"
    torchrun_case(file_name, case_name, num_proc=8)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_cat_4d_tensor():
    '''
    Feature: test parallel op cat (gloo cpu).
    Description: Concatenate 4D tensors on a 2D device mesh.
    Expectation: Run success.
    '''

    file_name = "parallel_op_cat.py"
    case_name = "test_cat_4d_tensor"
    torchrun_case(file_name, case_name, num_proc=8)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_cat_5d_mixed_placements():
    '''
    Feature: test parallel op cat.
    Description: 5D tensors with mixed sharding and replication placements.
    Expectation: Run success.
    '''

    file_name = "parallel_op_cat.py"
    case_name = "test_cat_5d_mixed_placements"
    torchrun_case(file_name, case_name, num_proc=8)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_cat_5d_mixed_placements():
    '''
    Feature: test parallel op cat (gloo cpu).
    Description: 5D tensors with mixed sharding and replication placements.
    Expectation: Run success.
    '''

    file_name = "parallel_op_cat.py"
    case_name = "test_cat_5d_mixed_placements"
    torchrun_case(file_name, case_name, num_proc=8)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_cat_shard_last_cat_first():
    '''
    Feature: test parallel op cat.
    Description: Shard on the last dimension but concatenate on the first.
    Expectation: Run success.
    '''

    file_name = "parallel_op_cat.py"
    case_name = "test_cat_shard_last_cat_first"
    torchrun_case(file_name, case_name, num_proc=8)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_cat_shard_last_cat_first():
    '''
    Feature: test parallel op cat (gloo cpu).
    Description: Shard on the last dimension but concatenate on the first.
    Expectation: Run success.
    '''

    file_name = "parallel_op_cat.py"
    case_name = "test_cat_shard_last_cat_first"
    torchrun_case(file_name, case_name, num_proc=8)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_cat_singleton_dimension():
    '''
    Feature: test parallel op cat.
    Description: Concatenate tensors along a singleton dimension.
    Expectation: Run success.
    '''

    file_name = "parallel_op_cat.py"
    case_name = "test_cat_singleton_dimension"
    torchrun_case(file_name, case_name, num_proc=4)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_cat_singleton_dimension():
    '''
    Feature: test parallel op cat (gloo cpu).
    Description: Concatenate tensors along a singleton dimension.
    Expectation: Run success.
    '''

    file_name = "parallel_op_cat.py"
    case_name = "test_cat_singleton_dimension"
    torchrun_case(file_name, case_name, num_proc=4)
