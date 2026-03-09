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
def test_distributed_index_select_basic():
    '''
    Feature: test parallel op index_select.
    Description: test parallel op index_select basic unsharded dimension.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_index_select.py"
    case_name = "test_distributed_index_select_basic"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_index_select_3d():
    '''
    Feature: test parallel op index_select.
    Description: test parallel op index_select on 3D tensor with multiple sharded dimensions.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_index_select.py"
    case_name = "test_distributed_index_select_3d"
    torchrun_case(file_name, case_name, master_port, num_proc=4)



@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_index_select_negative_dim():
    '''
    Feature: test parallel op index_select.
    Description: test parallel op index_select with a negative dimension index.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_index_select.py"
    case_name = "test_distributed_index_select_negative_dim"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_index_select_2d_dim0():
    """
    Feature: test parallel op index_select.
    Description: test parallel op index_select on dim 0 with a 2D tensor.
    Expectation: Run success.
    """
    master_port = 10360
    file_name = "parallel_op_index_select.py"
    case_name = "test_distributed_index_select_2d_dim0"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_index_select_2d_dim1():
    """
    Feature: test parallel op index_select.
    Description: test parallel op index_select on dim 1 with a 2D tensor.
    Expectation: Run success.
    """
    master_port = 10361
    file_name = "parallel_op_index_select.py"
    case_name = "test_distributed_index_select_2d_dim1"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_index_select_3d_dim1():
    """
    Feature: test parallel op index_select.
    Description: test parallel op index_select on an unsharded dimension of a 3D tensor.
    Expectation: Run success.
    """
    master_port = 10362
    file_name = "parallel_op_index_select.py"
    case_name = "test_distributed_index_select_3d_dim1"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_index_select_single_element():
    """
    Feature: test parallel op index_select.
    Description: test parallel op index_select using a single-element index.
    Expectation: Run success.
    """
    master_port = 10364
    file_name = "parallel_op_index_select.py"
    case_name = "test_distributed_index_select_single_element"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_index_select_sharded_dim0_2d():
    """
    Feature: test parallel op index_select.
    Description: test parallel op index_select on a sharded dimension 0 of a 2D tensor.
    Expectation: Run success.
    """
    master_port = 10400
    file_name = "parallel_op_index_select.py"
    case_name = "test_distributed_index_select_sharded_dim0_2d"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_index_select_sharded_dim1_2d():
    """
    Feature: test parallel op index_select.
    Description: test parallel op index_select on a sharded dimension 1 of a 2D tensor.
    Expectation: Run success.
    """
    master_port = 10401
    file_name = "parallel_op_index_select.py"
    case_name = "test_distributed_index_select_sharded_dim1_2d"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_index_select_sharded_dim2_3d():
    """
    Feature: test parallel op index_select.
    Description: test parallel op index_select on a sharded dimension 2 of a 3D tensor.
    Expectation: Run success.
    """
    master_port = 10402
    file_name = "parallel_op_index_select.py"
    case_name = "test_distributed_index_select_sharded_dim2_3d"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_index_select_duplicate_indices_sharded():
    """
    Feature: test parallel op index_select.
    Description: test parallel op index_select on a sharded dimension with duplicate indices.
    Expectation: Run success.
    """
    master_port = 10403
    file_name = "parallel_op_index_select.py"
    case_name = "test_distributed_index_select_duplicate_indices_sharded"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_index_select_out_of_order_sharded():
    """
    Feature: test parallel op index_select.
    Description: test parallel op index_select on a sharded dimension with out-of-order indices.
    Expectation: Run success.
    """
    master_port = 10404
    file_name = "parallel_op_index_select.py"
    case_name = "test_distributed_index_select_out_of_order_sharded"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_index_select_fully_replicated():
    """
    Feature: test parallel op index_select.
    Description: test parallel op index_select on a fully replicated tensor.
    Expectation: Run success.
    """
    master_port = 10405
    file_name = "parallel_op_index_select.py"
    case_name = "test_distributed_index_select_fully_replicated"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_index_select_negative_sharded_dim():
    """
    Feature: test parallel op index_select.
    Description: test parallel op index_select using a negative index on a sharded dimension.
    Expectation: Run success.
    """
    master_port = 10406
    file_name = "parallel_op_index_select.py"
    case_name = "test_distributed_index_select_negative_sharded_dim"
    torchrun_case(file_name, case_name, master_port)
