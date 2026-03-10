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
"""parallel_transpose_ext_view_shell test"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

TRANSPOSE_EXT_VIEW_SHARD_IN_PYTHON = "transpose_ext_view_shard_in_python.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_transpose_ext_view_shard_in_python_group1():
    """
    Feature: parallel run case in transpose_ext_view_shard_in_python
    Description:
        1. test_transpose_ext_view_basic_3d_1
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(TRANSPOSE_EXT_VIEW_SHARD_IN_PYTHON, "test_transpose_ext_view_basic_3d_1", 11294, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_transpose_ext_view_shard_in_python_group2():
    """
    Feature: parallel run case in transpose_ext_view_shard_in_python
    Description:
        1. test_transpose_ext_view_negative_dims_2
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(TRANSPOSE_EXT_VIEW_SHARD_IN_PYTHON, "test_transpose_ext_view_negative_dims_2", 11295, 8, 8, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_transpose_ext_view_shard_in_python_group3():
    """
    Feature: parallel run case in transpose_ext_view_shard_in_python
    Description:
        1. test_transpose_ext_view_same_dims_noop_3
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(TRANSPOSE_EXT_VIEW_SHARD_IN_PYTHON, "test_transpose_ext_view_same_dims_noop_3", 18288, 8, 8, 2),
    ])
