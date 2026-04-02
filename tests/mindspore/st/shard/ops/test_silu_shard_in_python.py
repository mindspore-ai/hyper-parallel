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
"""silu_shard_in_python test"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

SILU_SHARD_IN_PYTHON = "silu_shard_in_python.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_silu_shard_in_python_group1():
    """
    Feature: parallel run case in silu_shard_in_python
    Description:
        1. test_silu_data_parallel_1
        2. test_silu_model_parallel_2
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SILU_SHARD_IN_PYTHON, "test_silu_data_parallel_1", 11400, 4, 4, 2),
        MindSporeCase(SILU_SHARD_IN_PYTHON, "test_silu_model_parallel_2", 11401, 4, 4, 2),
    ])
