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
"""Validate replay behavior for adjacent PyTorch checkpoint SAVE regions."""

import os
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

CHECKPOINT_EXCLUDE = str(Path(__file__).with_name("checkpoint_exclude.py"))
REENTRANT_CASES = str(Path(__file__).with_name("reentrant_checkpoint_cases.py"))


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_checkpoint_exclude_wrapper():
    """
    Feature: checkpoint exclude
    Description:
        1. test_adjacent_save_regions_execute_once_and_replay_cached_outputs
        2. test_checkpoint_exclude_save_chain_swap
        3. test_checkpoint_exclude_save_chain_group_swap_cpu_pool
        4. test_rmsnorm_matmul_checkpoint_exclude_memory
        5. test_reentrant_checkpoint_exclude_swap
        6. test_operator_numerics
        7. test_rng_autocast
        8. test_memory_lifecycle
    Expectation: Run success.
    """
    cases = [
        TorchCase(CHECKPOINT_EXCLUDE, "test_adjacent_save_regions_execute_once_and_replay_cached_outputs", num_proc=1),
        TorchCase(CHECKPOINT_EXCLUDE, "test_checkpoint_exclude_save_chain_swap", num_proc=1),
        TorchCase(CHECKPOINT_EXCLUDE, "test_checkpoint_exclude_save_chain_group_swap_cpu_pool", num_proc=1),
        TorchCase(CHECKPOINT_EXCLUDE, "test_rmsnorm_matmul_checkpoint_exclude_memory", num_proc=1),
        TorchCase(CHECKPOINT_EXCLUDE, "test_reentrant_checkpoint_exclude_swap", num_proc=1),
        TorchCase(REENTRANT_CASES, "test_operator_numerics", num_proc=1),
        TorchCase(REENTRANT_CASES, "test_rng_autocast", num_proc=1),
        TorchCase(REENTRANT_CASES, "test_memory_lifecycle", num_proc=1),
    ]
    # The parent must remain torch-free; smaller hosts opt in with an explicit device budget.
    budget = int(os.environ.get("HYPER_PARALLEL_TEST_CARD_COUNT", "8"))
    if budget < 1 or budget > 8:
        raise ValueError("HYPER_PARALLEL_TEST_CARD_COUNT must be between 1 and 8")
    for offset in range(0, len(cases), budget):
        parallel_run(cases[offset:offset + budget], global_num_proc=budget)
