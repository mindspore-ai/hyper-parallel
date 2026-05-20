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
"""test parallel op masked_scatter"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

PARALLEL_OP_MASKED_SCATTER = "parallel_op_masked_scatter.py"

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_masked_scatter_group1():
    """
    Feature: parallel run case in shard
    Description:
        1.test_masked_scatter_basic_replicated
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_MASKED_SCATTER, "test_masked_scatter_basic_replicated", num_proc=4)
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_masked_scatter_group1_gloo():
    """
    Feature: parallel run case in shard (gloo cpu)
    Description:
        1.test_masked_scatter_basic_replicated
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_MASKED_SCATTER, "test_masked_scatter_basic_replicated", num_proc=4)
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_masked_scatter_group2():
    """
    Feature: parallel run case in shard
    Description:
        1.test_masked_scatter_1d_replicated
        2.test_masked_scatter_3d_broadcast
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_MASKED_SCATTER, "test_masked_scatter_1d_replicated", num_proc=4),
        TorchCase(PARALLEL_OP_MASKED_SCATTER, "test_masked_scatter_3d_broadcast", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_masked_scatter_group2_gloo():
    """
    Feature: parallel run case in shard (gloo cpu)
    Description:
        1.test_masked_scatter_1d_replicated
        2.test_masked_scatter_3d_broadcast
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_MASKED_SCATTER, "test_masked_scatter_1d_replicated", num_proc=4),
        TorchCase(PARALLEL_OP_MASKED_SCATTER, "test_masked_scatter_3d_broadcast", num_proc=4),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_masked_scatter_group3():
    """
    Feature: parallel run case in shard
    Description:
        1.test_masked_scatter_oversized_source
        2.test_masked_scatter_all_false_mask
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_MASKED_SCATTER, "test_masked_scatter_oversized_source", num_proc=4),
        TorchCase(PARALLEL_OP_MASKED_SCATTER, "test_masked_scatter_all_false_mask", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_masked_scatter_group3_gloo():
    """
    Feature: parallel run case in shard (gloo cpu)
    Description:
        1.test_masked_scatter_oversized_source
        2.test_masked_scatter_all_false_mask
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_MASKED_SCATTER, "test_masked_scatter_oversized_source", num_proc=4),
        TorchCase(PARALLEL_OP_MASKED_SCATTER, "test_masked_scatter_all_false_mask", num_proc=4),
    ])
