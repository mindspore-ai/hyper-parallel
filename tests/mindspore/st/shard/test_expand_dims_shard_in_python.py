# Copyright 2025 Huawei Technologies Co., Ltd
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
"""parallel_expand_dims_shell test"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

EXPAND_DIMS_SHARD_IN_PYTHON = "expand_dims_shard_in_python.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_expand_dims_shard_in_python_group1():
    """
    Feature: parallel run case in expand_dims_shard_in_python
    Description:
        1. test_expanddims_data_parallel_1
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(EXPAND_DIMS_SHARD_IN_PYTHON, "test_expanddims_data_parallel_1", 18290, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_expand_dims_shard_in_python_group2():
    """
    Feature: parallel run case in expand_dims_shard_in_python
    Description:
        1. test_expanddims_model_parallel_2
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(EXPAND_DIMS_SHARD_IN_PYTHON, "test_expanddims_model_parallel_2", 18292, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_expand_dims_shard_in_python_group3():
    """
    Feature: parallel run case in expand_dims_shard_in_python
    Description:
        1. test_expanddims_hybrid_parallel_3
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(EXPAND_DIMS_SHARD_IN_PYTHON, "test_expanddims_hybrid_parallel_3", 18294, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_expand_dims_shard_in_python_group4():
    """
    Feature: parallel run case in expand_dims_shard_in_python
    Description:
        1. test_expanddims_insert_middle_4
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(EXPAND_DIMS_SHARD_IN_PYTHON, "test_expanddims_insert_middle_4", 18295, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_expand_dims_shard_in_python_group5():
    """
    Feature: parallel run case in expand_dims_shard_in_python
    Description:
        1. test_expanddims_negative_axis_5
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(EXPAND_DIMS_SHARD_IN_PYTHON, "test_expanddims_negative_axis_5", 18298, 8, 8, 2),
    ])
