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
"""Lightweight launcher for the representative Torch MegaMoe profiling ST."""

from __future__ import annotations

import json
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import TorchCase, parallel_run
from tests.torch.multicore._test_env import (
    multicore_adapter_is_available,
    prepare_multicore_test_environment,
    without_inherited_rank_environment,
)
from tests.common.port_utils import allocate_port


_WORKER = str(Path(__file__).resolve().parent / "_test_mega_moe_profiling.py")
_WORLD_SIZE = 2
_EXPECTED_RECORD_COUNT = 1411


def _prepare_torch_multicore_test_environment() -> None:
    """Activate multicore and require a payload built for Torch."""
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError("MegaMoe Torch ST requires a wheel or PYTHONPATH payload built with " "--multicore on")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_standalone_profiler(monkeypatch, tmp_path: Path) -> None:
    """Export complete steady-state forward traces for both ranks."""
    _prepare_torch_multicore_test_environment()
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_WORLD_SIZE))
    monkeypatch.setenv("HP_MEGA_MOE_PROFILE_RESULT_DIR", str(tmp_path))
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_forward_profiling",
                    num_proc=_WORLD_SIZE,
                )
            ],
            global_num_proc=_WORLD_SIZE,
        )

    for rank in range(_WORLD_SIZE):
        trace_path = tmp_path / f"rank{rank}_mega_kernel_trace.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        metadata = trace["megaKernelCycleTrace"]
        assert metadata["invocationCount"] == 1
        assert metadata["recordCount"] == _EXPECTED_RECORD_COUNT
        assert metadata["droppedRecordCount"] == 0
