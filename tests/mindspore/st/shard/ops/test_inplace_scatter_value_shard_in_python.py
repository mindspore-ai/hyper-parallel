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
"""parallel_inplace_scatter_value_shell test"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

INPLACE_SCATTER_VALUE_SHARD_IN_PYTHON = "inplace_scatter_value_shard_in_python.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_inplace_scatter_value_shard_in_python_group1():
    """
    Feature: parallel run case in inplace_scatter_value_shard_in_python
    Description:
        1. test_inplace_scatter_value_data_parallel_1
        2. test_inplace_scatter_value_model_parallel_2
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(INPLACE_SCATTER_VALUE_SHARD_IN_PYTHON, "test_inplace_scatter_value_data_parallel_1",
                      12300, 4, 4, 2),
        MindSporeCase(INPLACE_SCATTER_VALUE_SHARD_IN_PYTHON, "test_inplace_scatter_value_model_parallel_2",
                      12301, 4, 4, 2)
    ])
