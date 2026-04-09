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
"""parallel_masked_fill_scalar test"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

MASKED_FILL_SCALAR_SHARD_IN_PYTHON = "masked_fill_scalar_shard_in_python.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_masked_fill_scalar_shard_in_python_group1():
    """
    Feature: parallel run case in masked_fill_scalar_shard_in_python
    Description:
        1. test_masked_fill_scalar_same_shape_parallel_1
        2. test_masked_fill_scalar_broadcast_dim0_parallel_2
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(MASKED_FILL_SCALAR_SHARD_IN_PYTHON, "test_masked_fill_scalar_same_shape_parallel_1",
                      11400, 4, 4, 2),
        MindSporeCase(MASKED_FILL_SCALAR_SHARD_IN_PYTHON, "test_masked_fill_scalar_broadcast_dim0_parallel_2",
                      11401, 4, 4, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_masked_fill_scalar_shard_in_python_group2():
    """
    Feature: parallel run case in masked_fill_scalar_shard_in_python
    Description:
        1. test_masked_fill_scalar_broadcast_dim1_parallel_3
        2. test_masked_fill_scalar_partial_shard_parallel_4
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(MASKED_FILL_SCALAR_SHARD_IN_PYTHON, "test_masked_fill_scalar_broadcast_dim1_parallel_3",
                      11402, 4, 4, 2),
        MindSporeCase(MASKED_FILL_SCALAR_SHARD_IN_PYTHON, "test_masked_fill_scalar_partial_shard_parallel_4",
                      11403, 4, 4, 2),
    ])
