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
"""Tests for init_empty_weights -> fully_shard -> init weight consistency."""
from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_TEST_INIT_WEIGHTS = str(Path(__file__).resolve().with_name("_test_init_weights.py"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_init_weights_group1() -> None:
    """Verify seeded, randn_like, and replicated initialization on two NPUs."""
    torchrun_case(_TEST_INIT_WEIGHTS, (
        "test_init_weights_consistency",
        "test_init_weights_with_randn_like",
        "test_replicate_init_weights_consistency",
    ), num_proc=2, timeout=420)
