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
"""test parallel op conv3d"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_conv3d_data_parallel():
    '''
    Feature: test parallel op conv3d data parallel.
    Description: test parallel op conv3d data parallel.
    Expectation: Run success.
    '''
    master_port = 10359
    file_name = "parallel_op_conv3d.py"
    case_name = "test_distributed_conv3d_data_parallel"
    torchrun_case(file_name, case_name, master_port, num_proc=4)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_conv3d_column_parallel():
    '''
    Feature: test parallel op conv3d column parallel.
    Description: test parallel op conv3d column parallel.
    Expectation: Run success.
    '''
    master_port = 10360
    file_name = "parallel_op_conv3d.py"
    case_name = "test_distributed_conv3d_column_parallel"
    torchrun_case(file_name, case_name, master_port, num_proc=4)
@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_conv3d_spatial_parallel():
    '''
    Feature: test parallel op conv3d spatial parallel.
    Description: test parallel op conv3d spatial parallel.
    Expectation: Run success.
    '''
    master_port = 10361
    file_name = "parallel_op_conv3d.py"
    case_name = "test_distributed_conv3d_spatial_parallel"
    torchrun_case(file_name, case_name, master_port, num_proc=4)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_conv3d_with_bias():
    '''
    Feature: test parallel op conv3d with bias.
    Description: test parallel op conv3d with bias.
    Expectation: Run success.
    '''
    master_port = 10362
    file_name = "parallel_op_conv3d.py"
    case_name = "test_distributed_conv3d_with_bias"
    torchrun_case(file_name, case_name, master_port, num_proc=4)
@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_conv3d_row_parallel():
    '''
    Feature: test parallel op conv3d row parallel.
    Description: test parallel op conv3d row parallel.
    Expectation: Run success.
    '''
    master_port = 10363
    file_name = "parallel_op_conv3d.py"
    case_name = "test_distributed_conv3d_row_parallel"
    torchrun_case(file_name, case_name, master_port, num_proc=4)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_conv3d_dp_cp():
    '''
    Feature: test parallel op conv3d data + column parallel.
    Description: test parallel op conv3d data + column parallel on a 2D mesh.
    Expectation: Run success.
    '''
    master_port = 10364
    file_name = "parallel_op_conv3d.py"
    case_name = "test_distributed_conv3d_dp_cp"
    torchrun_case(file_name, case_name, master_port, num_proc=4)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_conv3d_dp_rp():
    '''
    Feature: test parallel op conv3d data + row parallel.
    Description: test parallel op conv3d data + row parallel on a 2D mesh.
    Expectation: Run success.
    '''
    master_port = 10365
    file_name = "parallel_op_conv3d.py"
    case_name = "test_distributed_conv3d_dp_rp"
    torchrun_case(file_name, case_name, master_port, num_proc=4)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_conv3d_spatial_h():
    '''
    Feature: test parallel op conv3d spatial parallel on Height.
    Description: test parallel op conv3d spatial parallel on Height axis.
    Expectation: Run success.
    '''
    master_port = 10366
    file_name = "parallel_op_conv3d.py"
    case_name = "test_distributed_conv3d_spatial_h"
    torchrun_case(file_name, case_name, master_port, num_proc=4)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_conv3d_spatial_w():
    '''
    Feature: test parallel op conv3d spatial parallel on Width.
    Description: test parallel op conv3d spatial parallel on Width axis.
    Expectation: Run success.
    '''
    master_port = 10367
    file_name = "parallel_op_conv3d.py"
    case_name = "test_distributed_conv3d_spatial_w"
    torchrun_case(file_name, case_name, master_port, num_proc=4)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_conv3d_groups_dp():
    '''
    Feature: test parallel op conv3d with groups and data parallel.
    Description: test parallel op conv3d with groups > 1 and data parallel.
    Expectation: Run success.
    '''
    master_port = 10368
    file_name = "parallel_op_conv3d.py"
    case_name = "test_distributed_conv3d_groups_dp"
    torchrun_case(file_name, case_name, master_port, num_proc=4)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_conv3d_groups_cp():
    '''
    Feature: test parallel op conv3d with groups and column parallel.
    Description: test parallel op conv3d with groups > 1 and column parallel.
    Expectation: Run success.
    '''
    master_port = 10369
    file_name = "parallel_op_conv3d.py"
    case_name = "test_distributed_conv3d_groups_cp"
    torchrun_case(file_name, case_name, master_port, num_proc=4)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_conv3d_groups_cp_with_bias():
    '''
    Feature: test parallel op conv3d with groups, column parallel and bias.
    Description: test parallel op conv3d with groups > 1, column parallel and bias.
    Expectation: Run success.
    '''
    master_port = 10370
    file_name = "parallel_op_conv3d.py"
    case_name = "test_distributed_conv3d_groups_cp_with_bias"
    torchrun_case(file_name, case_name, master_port, num_proc=4)
