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
"""test parallel op stack"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

PARALLEL_OP_STACK = "parallel_op_stack.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_stack_group1():
    """
    Feature: parallel run case in shard
    Description:
        1.test_distributed_stack_basic_dim0
        2.test_distributed_stack_dim1
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_STACK, "test_distributed_stack_basic_dim0", 10400, 4),
        TorchCase(PARALLEL_OP_STACK, "test_distributed_stack_dim1", 10401, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_stack_group2():
    """
    Feature: parallel run case in shard
    Description:
        1.test_distributed_stack_negative_dim
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_STACK, "test_distributed_stack_negative_dim", 10402, 4),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_stack_group3():
    """
    Feature: parallel run case in shard
    Description:
        1.test_distributed_stack_multiple_tensors
        2.test_distributed_stack_3d_tensors
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_STACK, "test_distributed_stack_multiple_tensors", 10404, 4),
        TorchCase(PARALLEL_OP_STACK, "test_distributed_stack_3d_tensors", 10405, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_stack_group4():
    """
    Feature: parallel run case in shard
    Description:
        1.test_distributed_stack_scalars
        2.test_distributed_stack_fully_replicated
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_STACK, "test_distributed_stack_scalars", 10406, 4),
        TorchCase(PARALLEL_OP_STACK, "test_distributed_stack_fully_replicated", 10407, 4),
    ])
