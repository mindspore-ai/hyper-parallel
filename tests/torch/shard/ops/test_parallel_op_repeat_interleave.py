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
"""test parallel op repeat_interleave"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

PARALLEL_OP_REPEAT_INTERLEAVE = "parallel_op_repeat_interleave.py"

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_repeat_interleave_group1():
    """
    Feature: parallel run case in shard
    Description:
        1.test_repeat_interleave_layout_inference
        2.test_repeat_interleave_with_tensor
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_REPEAT_INTERLEAVE, "test_repeat_interleave_layout_inference", num_proc=4),
        TorchCase(PARALLEL_OP_REPEAT_INTERLEAVE, "test_repeat_interleave_with_tensor", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_repeat_interleave_group1_gloo():
    """
    Feature: parallel run case in shard (gloo cpu)
    Description:
        1.test_repeat_interleave_layout_inference
        2.test_repeat_interleave_with_tensor
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_REPEAT_INTERLEAVE, "test_repeat_interleave_layout_inference", num_proc=4),
        TorchCase(PARALLEL_OP_REPEAT_INTERLEAVE, "test_repeat_interleave_with_tensor", num_proc=4),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_repeat_interleave_group2():
    """
    Feature: parallel run case in shard
    Description:
        1.test_repeat_interleave_dim_none
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_REPEAT_INTERLEAVE, "test_repeat_interleave_dim_none", num_proc=4)
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_repeat_interleave_group2_gloo():
    """
    Feature: parallel run case in shard (gloo cpu)
    Description:
        1.test_repeat_interleave_dim_none
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_REPEAT_INTERLEAVE, "test_repeat_interleave_dim_none", num_proc=4)
    ])
