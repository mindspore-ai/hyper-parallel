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
"""
Shell file for HistcExt distributed operator integration tests.
"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPLEMENTATION_FILE = "histc_ext_shard_in_python.py"


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential"
)
def test_histc_ext_shard_in_python_group1():
    """
    Feature: parallel run case in gather_nd_shard_in_python
    Description:
        1. test_histc_ext_data_parallel1
        2. test_histc_ext_model_parallel2
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPLEMENTATION_FILE, "test_histc_ext_data_parallel1", 11500, 4, 4, 2),
        MindSporeCase(IMPLEMENTATION_FILE, "test_histc_ext_model_parallel2", 11501, 4, 4, 2),
    ])


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential"
)
def test_histc_ext_shard_in_python_group2():
    """
    Feature: parallel run case in gather_nd_shard_in_python
    Description:
        1. test_histc_ext_hybrid_parallel3
        2. test_histc_ext_all_replicated4
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPLEMENTATION_FILE, "test_histc_ext_hybrid_parallel3", 11502, 4, 4, 2),
        MindSporeCase(IMPLEMENTATION_FILE, "test_histc_ext_all_replicated4", 11503, 4, 4, 2),
    ])
