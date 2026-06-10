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
"""msrun launcher for fully_shard 1F1B-style micro-batching tests."""
import os

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

_TEST_FILE = os.path.join(os.path.dirname(__file__), "_test_fully_shard_simu_pp.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_fully_shard_simu_pp_suite():
    """
    Feature: fully_shard simulating 1F1B-style micro-batching.
    Description: Run both unshard-toggle variants together on a 4-card dp mesh; each variant
                 compares per-rank loss + grad against a single-card baseline that consumes
                 the full global input.
    Expectation: fully_shard loss/grad match the baseline on every rank for both variants.
    """
    parallel_run([
        MindSporeCase(_TEST_FILE, "test_fully_shard_simu_pp_implicit_unshard_reshard", 18401, 4, 4),
        MindSporeCase(_TEST_FILE, "test_fully_shard_simu_pp_explicit_unshard_no_reshard", 18402, 4, 4),
    ])
