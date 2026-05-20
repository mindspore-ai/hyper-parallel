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
"""test parallel op isin"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

PARALLEL_OP_ISIN = "parallel_op_isin.py"

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_isin_group1():
    """
    Feature: parallel run case in shard
    Description:
        1.test_isin_layout_inference
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_ISIN, "test_isin_layout_inference", num_proc=4)
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_isin_group1_gloo():
    """
    Feature: parallel run case in shard (gloo cpu)
    Description:
        1.test_isin_layout_inference
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_ISIN, "test_isin_layout_inference", num_proc=4)
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_isin_group2():
    """
    Feature: parallel run case in shard
    Description:
        1.test_isin_invert_and_assume_unique
        2.test_isin_mixed_parallel_3d
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_ISIN, "test_isin_invert_and_assume_unique", num_proc=4),
        TorchCase(PARALLEL_OP_ISIN, "test_isin_mixed_parallel_3d", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_isin_group2_gloo():
    """
    Feature: parallel run case in shard (gloo cpu)
    Description:
        1.test_isin_invert_and_assume_unique
        2.test_isin_mixed_parallel_3d
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_ISIN, "test_isin_invert_and_assume_unique", num_proc=4),
        TorchCase(PARALLEL_OP_ISIN, "test_isin_mixed_parallel_3d", num_proc=4),
    ])
