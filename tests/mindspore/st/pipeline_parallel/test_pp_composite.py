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
"""msrun launcher for Pipeline Parallel + fully_shard composite tests."""
import os
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

_TEST_FILE = os.path.join(os.path.dirname(__file__), "_test_pp_composite.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_pp_plus_fully_shard_replicate():
    """
    Feature: Pipeline parallel composed with fully_shard (replicate-style DP) under GPipe and 1F1B.
    Description: Run the GPipe and 1F1B schedules on fully_shard-wrapped stages in parallel,
                 each occupying 4 cards, and compare last-stage micro-batch losses against a
                 no-FSDP pipeline reference.
    Expectation: Distributed last-stage losses match the reference pipeline for both schedules.
    """
    parallel_run([
        MindSporeCase(_TEST_FILE, "test_fully_shard_pp_gpipe", 18301, 4, 4),
        MindSporeCase(_TEST_FILE, "test_fully_shard_pp_1f1b", 18302, 4, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_pp_plus_fully_shard_vpp():
    """
    Feature: Pipeline parallel (VPP / Interleaved 1F1B) composed with fully_shard (replicate-style DP).
    Description: Run the Interleaved 1F1B schedule on per-virtual-stage fully_shard-wrapped layers
                 on 4 cards, and compare last-virtual-stage micro-batch losses plus per-stage
                 gradients against a full-model serial reference.
    Expectation: Distributed last-virtual-stage losses and per-virtual-stage grad shards match
                 the reference under the additive-loss accumulation pattern.
    """
    parallel_run([
        MindSporeCase(_TEST_FILE, "test_fully_shard_pp_vpp", 18303, 4, 4),
    ])
