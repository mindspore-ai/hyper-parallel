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
"""parallel_base_shard test"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

BASE_RANDOM = "base_random.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_base_random_group1():
    """
    Feature: parallel run case in base_random
    Description:
        1. test_tracker_initialization
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(BASE_RANDOM, "test_tracker_initialization", 18306, 2, 2)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_base_random_group2():
    """
    Feature: parallel run case in base_random
    Description:
        1. test_distribute_region_disabled
        2. test_multi_dim_sharding_offset
        3. test_rng_tracker
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(BASE_RANDOM, "test_distribute_region_disabled", 18309, 2, 2),
        MindSporeCase(BASE_RANDOM, "test_multi_dim_sharding_offset", 18310, 4, 4),
        MindSporeCase(BASE_RANDOM, "test_rng_tracker", 11336, 2, 2)
    ])
