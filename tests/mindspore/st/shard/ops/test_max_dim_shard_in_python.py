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
"""parallel_max_dim shell test"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

MAX_DIM_SHARD_IN_PYTHON = "max_dim_shard_in_python.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_max_dim_shard_in_python_group1():
    """
    Feature: parallel run case in max_dim_shard_in_python
    Description:
        1. test_max_dim_data_parallel_1
        2. test_max_dim_model_parallel_2
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(MAX_DIM_SHARD_IN_PYTHON, "test_max_dim_data_parallel_1", 11700, 4, 4, 2),
        MindSporeCase(MAX_DIM_SHARD_IN_PYTHON, "test_max_dim_model_parallel_2", 11701, 4, 4, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_max_dim_shard_in_python_group2():
    """
    Feature: parallel run case in max_dim_shard_in_python
    Description:
        1. test_max_dim_negative_dim_3
        2. test_max_dim_keepdim_false_4
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(MAX_DIM_SHARD_IN_PYTHON, "test_max_dim_negative_dim_3", 11702, 4, 4, 2),
        MindSporeCase(MAX_DIM_SHARD_IN_PYTHON, "test_max_dim_keepdim_false_4", 11703, 4, 4, 2),
    ])
