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

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_min_element_wise():
    '''
    Feature: test parallel op min.
    Description: test parallel op min element wise.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_element_wise"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_min_element_wise():
    '''
    Feature: test parallel op min (gloo cpu).
    Description: test parallel op min element wise.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_element_wise"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_min_dim_reduce_sharded():
    '''
    Feature: test parallel op min.
    Description: test parallel op min reduce sharded dim.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_dim_reduce_sharded"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_min_dim_reduce_sharded():
    '''
    Feature: test parallel op min (gloo cpu).
    Description: test parallel op min reduce sharded dim.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_dim_reduce_sharded"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_min_dim_reduce_replicated():
    '''
    Feature: test parallel op min.
    Description: test parallel op min reduce replicated dim.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_dim_reduce_replicated"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_min_dim_reduce_replicated():
    '''
    Feature: test parallel op min (gloo cpu).
    Description: test parallel op min reduce replicated dim.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_dim_reduce_replicated"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_min_global_reduce():
    '''
    Feature: test parallel op min.
    Description: test parallel op min global reduce.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_global_reduce"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_min_global_reduce():
    '''
    Feature: test parallel op min (gloo cpu).
    Description: test parallel op min global reduce.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_global_reduce"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_min_keepdim():
    '''
    Feature: test parallel op min.
    Description: test parallel op min keepdim.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_keepdim"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_min_keepdim():
    '''
    Feature: test parallel op min (gloo cpu).
    Description: test parallel op min keepdim.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_keepdim"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_min_3d_element_wise():
    '''
    Feature: test parallel op min.
    Description: test parallel op min element wise on 3d tensor.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_3d_element_wise"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_min_3d_element_wise():
    '''
    Feature: test parallel op min (gloo cpu).
    Description: test parallel op min element wise on 3d tensor.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_3d_element_wise"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_min_3d_reduce_negative_dim():
    '''
    Feature: test parallel op min.
    Description: test parallel op min reduce on negative replicated dim.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_3d_reduce_negative_dim"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_min_3d_reduce_negative_dim():
    '''
    Feature: test parallel op min (gloo cpu).
    Description: test parallel op min reduce on negative replicated dim.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_3d_reduce_negative_dim"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_min_3d_reduce_sharded_dim():
    '''
    Feature: test parallel op min.
    Description: test parallel op min reduce on sharded 3d dim.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_3d_reduce_sharded_dim"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_min_3d_reduce_sharded_dim():
    '''
    Feature: test parallel op min (gloo cpu).
    Description: test parallel op min reduce on sharded 3d dim.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_3d_reduce_sharded_dim"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_min_1d_mesh_global_reduce():
    '''
    Feature: test parallel op min.
    Description: test parallel op min global reduce on 1D mesh.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_1d_mesh_global_reduce"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_min_1d_mesh_global_reduce():
    '''
    Feature: test parallel op min (gloo cpu).
    Description: test parallel op min global reduce on 1D mesh.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_1d_mesh_global_reduce"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_min_1d_mesh_element_wise():
    '''
    Feature: test parallel op min.
    Description: test parallel op min element-wise on 1D mesh.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_1d_mesh_element_wise"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_min_1d_mesh_element_wise():
    '''
    Feature: test parallel op min (gloo cpu).
    Description: test parallel op min element-wise on 1D mesh.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_1d_mesh_element_wise"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_min_keepdim_negative_dim():
    '''
    Feature: test parallel op min.
    Description: test parallel op min keepdim using negative dim indexing.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_keepdim_negative_dim"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_min_keepdim_negative_dim():
    '''
    Feature: test parallel op min (gloo cpu).
    Description: test parallel op min keepdim using negative dim indexing.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_keepdim_negative_dim"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_min_4_cards():
    '''
    Feature: test parallel op min.
    Description: test parallel op min reduce explicitly using 4 cards.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_4_cards"
    torchrun_case(file_name, case_name, num_proc=4)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_min_4_cards():
    '''
    Feature: test parallel op min (gloo cpu).
    Description: test parallel op min reduce explicitly using 4 cards.
    Expectation: Run success.
    '''

    file_name = "parallel_op_min.py"
    case_name = "test_min_4_cards"
    torchrun_case(file_name, case_name, num_proc=4)
