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
"""One lightweight gate for checkpoint exclusion and split pipeline integration."""

import os
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase


_MEMORY = Path(__file__).resolve().parent
_TORCH = _MEMORY.parent
_EXCLUDE = str(_MEMORY / "checkpoint_exclude.py")
_REENTRANT = str(_MEMORY / "reentrant_checkpoint_cases.py")
_FSDP = str(_MEMORY / "reentrant_composite_cases.py")
_PP = str(_TORCH / "pipeline_parallel" / "reentrant_checkpoint_worker.py")


def _device_budget() -> int:
    """Read the explicit contiguous physical-device budget used by parallel_run."""
    budget = int(os.environ.get("HYPER_PARALLEL_TEST_CARD_COUNT", "8"))
    if not 1 <= budget <= 8:
        raise ValueError("HYPER_PARALLEL_TEST_CARD_COUNT must be between 1 and 8")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible is not None and visible.split(",") != [str(index) for index in range(budget)]:
        raise ValueError("parallel_run requires ASCEND_RT_VISIBLE_DEVICES to map physical cards 0..budget-1")
    return budget


def _run_with_budget(cases: list[TorchCase], budget: int) -> None:
    """Pack disjoint device slices while omitting ranks unavailable on smaller hosts."""
    batch = []
    used = 0
    for case in cases:
        if case.num_proc > budget:
            continue
        if used + case.num_proc > budget:
            parallel_run(batch, global_num_proc=budget)
            batch = []
            used = 0
        batch.append(case)
        used += case.num_proc
    if batch:
        parallel_run(batch, global_num_proc=budget)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards",
          essential_mark="essential")
def test_reentrant_exclude_suite() -> None:
    """Run two mixed batches, PP8, and the two-rank swap/split case."""
    budget = _device_budget()
    _run_with_budget([
        TorchCase(_PP, "test_reentrant_pp4", num_proc=4),
        TorchCase(_FSDP, "test_fsdp_checkpoint_wrapper_exclude", num_proc=2),
        TorchCase(_EXCLUDE, "test_checkpoint_exclude_save_chain_swap", num_proc=1),
        TorchCase(_REENTRANT, "test_operator_numerics", num_proc=1),
    ], budget)
    if budget == 8:
        parallel_run([TorchCase(_PP, "test_reentrant_pp8", num_proc=8)])
    _run_with_budget([
        TorchCase(_PP, "test_reentrant_pp2", num_proc=2),
        TorchCase(_FSDP, "test_fsdp_checkpoint_api_exclude", num_proc=2),
        TorchCase(_EXCLUDE, "test_checkpoint_exclude_save_chain_group_swap_cpu_pool", num_proc=1),
        TorchCase(_EXCLUDE, "test_reentrant_checkpoint_exclude_swap", num_proc=1),
        TorchCase(_REENTRANT, "test_rng_autocast", num_proc=1),
        TorchCase(_REENTRANT, "test_memory_lifecycle", num_proc=1),
    ], budget)
    _run_with_budget([
        TorchCase(_PP, "test_reentrant_swap_stage_split_npu", num_proc=2),
        TorchCase(_EXCLUDE, "test_adjacent_save_regions_execute_once_and_replay_cached_outputs", num_proc=1),
        TorchCase(_EXCLUDE, "test_rmsnorm_matmul_checkpoint_exclude_memory", num_proc=1),
    ], budget)
