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
def test_distributed_repeat_basic_unsharded():
    """
    Feature: test parallel op repeat.
    Description: test parallel op repeat.
    Expectation: Run success.
    """
    master_port = 10359
    file_name = "parallel_op_repeat.py"
    case_name = "test_distributed_repeat_basic_unsharded"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_repeat_3d():
    """
    Feature: test parallel op repeat.
    Description: test parallel op repeat.
    Expectation: Run success.
    """
    master_port = 10359
    file_name = "parallel_op_repeat.py"
    case_name = "test_distributed_repeat_3d"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_repeat_scalar_tensor():
    """
    Feature: test parallel op repeat.
    Description: test parallel op repeat.
    Expectation: Run success.
    """
    master_port = 10359
    file_name = "parallel_op_repeat.py"
    case_name = "test_distributed_repeat_scalar_tensor"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_repeat_replicated_dim():
    """
    Feature: test parallel op repeat on a replicated dimension.
    Description: Verify repeat preserves replication and produces correct output.
    Expectation: Run success.
    """
    master_port = 10359
    file_name = "parallel_op_repeat.py"
    case_name = "test_distributed_repeat_replicated_dim"
    torchrun_case(file_name, case_name, master_port, num_proc=4)



@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_repeat_zero_times():
    """
    Feature: test parallel op repeat with zero repetitions.
    Description: Verify repeat with 0 produces an empty tensor correctly.
    Expectation: Run success.
    """
    master_port = 10359
    file_name = "parallel_op_repeat.py"
    case_name = "test_distributed_repeat_zero_times"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_repeat_4d_input():
    """
    Feature: test parallel op repeat with 4D input tensor.
    Description: Verify repeat works for higher-dimensional tensors with mixed sharding.
    Expectation: Run success.
    """
    master_port = 10359
    file_name = "parallel_op_repeat.py"
    case_name = "test_distributed_repeat_4d_input"
    torchrun_case(file_name, case_name, master_port, num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_repeat_sharded_dim_repeat_one():
    """
    Feature: test parallel op repeat on sharded dimension with repeat count of 1.
    Description: Verify repeat with count 1 on a sharded dim is allowed and preserves sharding.
    Expectation: Run success.
    """
    master_port = 10359
    file_name = "parallel_op_repeat.py"
    case_name = "test_distributed_repeat_sharded_dim_repeat_one"
    torchrun_case(file_name, case_name, master_port)



@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_repeat_all_dims_replicated():
    """
    Feature: test parallel op repeat when all input dimensions are replicated.
    Description: Verify repeat maintains replication for all dimensions.
    Expectation: Run success.
    """
    master_port = 10359
    file_name = "parallel_op_repeat.py"
    case_name = "test_distributed_repeat_all_dims_replicated"
    torchrun_case(file_name, case_name, master_port, num_proc=4)
