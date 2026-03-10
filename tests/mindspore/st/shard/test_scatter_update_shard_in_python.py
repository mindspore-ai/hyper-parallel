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
"""parallel_scatter_update_shell test"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

SCATTER_UPDATE_SHARD_IN_PYTHON = "scatter_update_shard_in_python.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_scatter_update_shard_in_python_group1():
    """
    Feature: parallel run case in scatter_update_shard_in_python
    Description:
        1. test_scatter_update_data_parallel_1
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SCATTER_UPDATE_SHARD_IN_PYTHON, "test_scatter_update_data_parallel_1", 12290, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_scatter_update_shard_in_python_group2():
    """
    Feature: parallel run case in scatter_update_shard_in_python
    Description:
        1. test_scatter_update_model_parallel_2
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SCATTER_UPDATE_SHARD_IN_PYTHON, "test_scatter_update_model_parallel_2", 12291, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_scatter_update_shard_in_python_group3():
    """
    Feature: parallel run case in scatter_update_shard_in_python
    Description:
        1. test_scatter_update_hybrid_parallel_3
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SCATTER_UPDATE_SHARD_IN_PYTHON, "test_scatter_update_hybrid_parallel_3", 12292, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_scatter_update_shard_in_python_group4():
    """
    Feature: parallel run case in scatter_update_shard_in_python
    Description:
        1. test_scatter_update_multi_dim_indices_4
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SCATTER_UPDATE_SHARD_IN_PYTHON, "test_scatter_update_multi_dim_indices_4", 12293, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_scatter_update_shard_in_python_group5():
    """
    Feature: parallel run case in scatter_update_shard_in_python
    Description:
        1. test_scatter_update_replicate_all_5
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(SCATTER_UPDATE_SHARD_IN_PYTHON, "test_scatter_update_replicate_all_5", 12294, 8, 8, 2),
    ])
