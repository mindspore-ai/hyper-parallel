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
def test_distributed_flatten_all_dims():
    '''
    Feature: test parallel op flatten.
    Description: test parallel op flatten.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_flatten.py"
    case_name = "test_distributed_flatten_all_dims"
    torchrun_case(file_name, case_name, master_port)



@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_flatten_middle_dims():
    '''
    Feature: dtensor + torch.Tensor.flatten on middle dimensions with partial sharding
    Description:
        - Flatten dimensions 1 and 2 of a 4D distributed tensor.
        - Input tensor has shape (4, 2, 4, 6), sharded on dim0 ("dp") and dim1 ("tp").
        - Only one of the flattened dimensions (dim1) is sharded.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_flatten.py"
    case_name = "test_distributed_flatten_middle_dims"
    torchrun_case(file_name, case_name, master_port, num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_flatten_unsharded():
    '''
    Feature: dtensor + torch.Tensor.flatten on unsharded middle dimensions
    Description:
        - Flatten dimensions 1 and 2 of a 3D distributed tensor.
        - Input tensor has shape (8, 4, 6), sharded only on dim0 ("dp").
        - The dimensions being flattened (dim1, dim2) are both replicated.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_flatten.py"
    case_name = "test_distributed_flatten_unsharded"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_flatten_negative_dims():
    '''
    Feature: dtensor + torch.Tensor.flatten with negative dimension indices
    Description:
        - Flatten dimensions using negative indices (-2, -1) on a 3D distributed tensor.
        - Input tensor has shape (8, 4, 6), sharded on dim0 ("dp") and dim1 ("tp").
        - The flattened dimensions correspond to dim1 and dim2, where dim1 is sharded.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_flatten.py"
    case_name = "test_distributed_flatten_negative_dims"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_flatten_scalar():
    '''
    Feature: dtensor + torch.Tensor.flatten on a distributed scalar
    Description:
        - Apply flatten(0, -1) to a distributed scalar tensor.
        - A scalar has no dimensions, so flatten should conceptually have no effect on its shape.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_flatten.py"
    case_name = "test_distributed_flatten_scalar"
    torchrun_case(file_name, case_name, master_port)
@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_flatten_default_args():
    '''
    Feature: dtensor + torch.Tensor.flatten with default arguments
    Description:
        - Apply flatten() without explicit start_dim and end_dim.
        - The default behavior is flattening all dimensions.
    Expectation: Run success.
    '''
    master_port = 10360
    file_name = "parallel_op_flatten.py"
    case_name = "test_distributed_flatten_default_args"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_flatten_single_dim():
    '''
    Feature: dtensor + torch.Tensor.flatten with start_dim == end_dim
    Description:
        - Flatten a single dimension (e.g., start_dim=1, end_dim=1).
        - Conceptually, this operation should not change the shape or layout.
    Expectation: Run success.
    '''
    master_port = 10361
    file_name = "parallel_op_flatten.py"
    case_name = "test_distributed_flatten_single_dim"
    torchrun_case(file_name, case_name, master_port, num_proc=4)



@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_flatten_2d_to_1d():
    '''
    Feature: dtensor + torch.Tensor.flatten from 2D to 1D
    Description:
        - Flatten a 2D tensor to 1D (start_dim=0, end_dim=1).
        - Input tensor has shape (8, 4), sharded on dim0 ("dp").
    Expectation: Run success.
    '''
    master_port = 10364
    file_name = "parallel_op_flatten.py"
    case_name = "test_distributed_flatten_2d_to_1d"
    torchrun_case(file_name, case_name, master_port)
