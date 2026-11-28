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
"""Ascend Level1 coverage for checkpoint exclusion."""

import os
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import TorchCase, parallel_run

_ROOT = Path(__file__).resolve().parent
_EXCLUDE = str(_ROOT / "checkpoint_exclude.py")
_REENTRANT = str(_ROOT / "reentrant_checkpoint_cases.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_checkpoint_exclude_wrapper() -> None:
    """Keep independent Level1 swap, RNG, numerical, and memory checks on Ascend."""
    cases = [
        TorchCase(_EXCLUDE, "test_rmsnorm_matmul_checkpoint_exclude_memory", num_proc=1),
        TorchCase(_EXCLUDE, "test_checkpoint_exclude_save_chain_swap_matrix", num_proc=1),
        TorchCase(_EXCLUDE, "test_reentrant_checkpoint_exclude_swap", num_proc=1),
        TorchCase(_EXCLUDE, "test_adjacent_save_regions_execute_once_and_replay_cached_outputs", num_proc=1),
        TorchCase(_REENTRANT, "test_checkpoint_exclude_numerics_and_rng", num_proc=1),
        TorchCase(_REENTRANT, "test_memory_lifecycle", num_proc=1),
    ]
    budget = int(os.environ.get("HYPER_PARALLEL_TEST_CARD_COUNT", "8"))
    if not 1 <= budget <= 8:
        raise ValueError("HYPER_PARALLEL_TEST_CARD_COUNT must be between 1 and 8")
    for offset in range(0, len(cases), budget):
        parallel_run(cases[offset:offset + budget], global_num_proc=budget)
