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
"""test base random"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

BASE_RANDOM = "base_random.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_base_random_group1():
    """
    Feature: parallel run case in shard
    Description:
        1.test_tracker_initialization
        2.test_distribute_region_disabled
        3.test_rng_tracker
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(BASE_RANDOM, "test_tracker_initialization", 11335, 2),
        TorchCase(BASE_RANDOM, "test_distribute_region_disabled", 11336, 2),
        TorchCase(BASE_RANDOM, "test_rng_tracker", 11337, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_base_random_group2():
    """
    Feature: parallel run case in shard
    Description:
        1.test_multi_dim_sharding_offset
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(BASE_RANDOM, "test_multi_dim_sharding_offset", 11338, 4)
    ])
