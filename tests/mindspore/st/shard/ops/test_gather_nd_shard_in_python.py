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
"""gathernd_shell test"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

GATHER_ND_SHARD_IN_PYTHON = "gather_nd_shard_in_python.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_gather_nd_shard_in_python_group1():
    """
    Feature: parallel run case in gather_nd_shard_in_python
    Description:
        1. test_gathernd_partial_model_parallel_1
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(GATHER_ND_SHARD_IN_PYTHON, "test_gathernd_partial_model_parallel_1", 18291, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_gather_nd_shard_in_python_group2():
    """
    Feature: parallel run case in gather_nd_shard_in_python
    Description:
        1. test_gathernd_partial_data_parallel_2
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(GATHER_ND_SHARD_IN_PYTHON, "test_gathernd_partial_data_parallel_2", 18293, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_gather_nd_shard_in_python_group3():
    """
    Feature: parallel run case in gather_nd_shard_in_python
    Description:
        1. test_gathernd_params_plain_tensor
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(GATHER_ND_SHARD_IN_PYTHON, "test_gathernd_params_plain_tensor", 18296, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_gather_nd_shard_in_python_group4():
    """
    Feature: parallel run case in gather_nd_shard_in_python
    Description:
        1. test_gathernd_k1_trailing_dims_shard_3
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(GATHER_ND_SHARD_IN_PYTHON, "test_gathernd_k1_trailing_dims_shard_3", 18297, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_gather_nd_shard_in_python_group5():
    """
    Feature: parallel run case in gather_nd_shard_in_python
    Description:
        1. test_gathernd_k2_trailing_dim_shard_4
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(GATHER_ND_SHARD_IN_PYTHON, "test_gathernd_k2_trailing_dim_shard_4", 18299, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_gather_nd_shard_in_python_group6():
    """
    Feature: parallel run case in gather_nd_shard_in_python
    Description:
        1. test_gathernd_k3_no_trailing_dims_5
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(GATHER_ND_SHARD_IN_PYTHON, "test_gathernd_k3_no_trailing_dims_5", 18300, 8, 8, 2),
    ])
