# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""parallel_base_custom_shard test"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

BASE_DTENSOR_INIT = "base_dtensor_init.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_base_dtensor_init_group1():
    """
    Feature: parallel run case in base_dtensor_init
    Description:
        1. test_ones
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(BASE_DTENSOR_INIT, "test_ones", 18302, 8, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_base_dtensor_init_group2():
    """
    Feature: parallel run case in base_dtensor_init
    Description:
        1. test_empty
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(BASE_DTENSOR_INIT, "test_empty", 18303, 2, 2)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_base_dtensor_init_group3():
    """
    Feature: parallel run case in base_dtensor_init
    Description:
        1. test_full
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(BASE_DTENSOR_INIT, "test_full", 18304, 8, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_base_dtensor_init_group4():
    """
    Feature: parallel run case in base_dtensor_init
    Description:
        1. test_zeros
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(BASE_DTENSOR_INIT, "test_zeros", 18305, 8, 8)
    ])
