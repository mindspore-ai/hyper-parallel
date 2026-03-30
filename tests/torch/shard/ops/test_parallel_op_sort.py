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
"""test parallel op sort"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

PARALLEL_OP_SORT = "parallel_op_sort.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_sort_group1():
    """
    Feature: parallel run case in shard
    Description:
        1.test_distributed_sort_basic
        2.test_distributed_sort_descending
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_SORT, "test_distributed_sort_basic", 10359, 4),
        TorchCase(PARALLEL_OP_SORT, "test_distributed_sort_descending", 10360, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_sort_group2():
    """
    Feature: parallel run case in shard
    Description:
        1.test_distributed_sort_middle_dim
        2.test_distributed_sort_negative_dim
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(PARALLEL_OP_SORT, "test_distributed_sort_middle_dim", 10361, 4),
        TorchCase(PARALLEL_OP_SORT, "test_distributed_sort_negative_dim", 10362, 4),
    ])
