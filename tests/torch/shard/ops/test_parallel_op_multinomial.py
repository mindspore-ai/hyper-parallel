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
"""test parallel op multinomial"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

PARALLEL_OP_MULTINOMIAL = "parallel_op_multinomial.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_multinomial_group1():
    """
    Feature: parallel run case in shard
    Description:
        1.test_distributed_multinomial_1d_replicated
        2.test_distributed_multinomial_2d_batch_sharded
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_MULTINOMIAL, "test_distributed_multinomial_1d_replicated", 10359, 4),
        TorchCase(PARALLEL_OP_MULTINOMIAL, "test_distributed_multinomial_2d_batch_sharded", 10360, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_multinomial_group2():
    """
    Feature: parallel run case in shard
    Description:
        1.test_distributed_multinomial_2d_fully_replicated
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_MULTINOMIAL, "test_distributed_multinomial_2d_fully_replicated", 10361, 4),
    ])
