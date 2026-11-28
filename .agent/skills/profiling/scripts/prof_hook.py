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
"""In-run profiler hook: profile a training loop's steady-state steps.

Imported by the training script, inside the training process, so ``torch_npu``
is imported at module top as elsewhere in this repository. It is never imported
by ``collect_prof.py``, which runs on the launching host.

``collect_prof.py --mode inrun`` exports ``PROF_OUT`` / ``PROF_WARMUP`` /
``PROF_ACTIVE`` / ``PROF_LEVEL``; this module is what reads them, so the launch
side and the run side agree on one schedule::

    import sys
    sys.path.insert(0, ".agent/skills/profiling/scripts")
    from prof_hook import profile_steps

    with profile_steps() as prof:
        for batch in loader:
            train_step(batch)
            prof.step()          # required: drives the warmup/active schedule

The defaults (skip warmup, profile 3 steady-state steps, ``Level1``) are the
configuration ``breakdown.py``'s consistency checks expect; ``Level1`` is what
emits ``step_trace_time.csv``, without which there is no step breakdown.
"""
from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

import torch_npu
from torch_npu.profiler import (
    ProfilerActivity,
    ProfilerLevel,
    _ExperimentalConfig,
    profile,
    schedule,
    tensorboard_trace_handler,
)

DEFAULT_OUT = "prof"
DEFAULT_WARMUP = 3
DEFAULT_ACTIVE = 3
_LEVELS = {"Level0": ProfilerLevel.Level0,
           "Level1": ProfilerLevel.Level1,
           "Level2": ProfilerLevel.Level2}


def _env_int(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back on bad input."""
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value >= 0 else default


@contextlib.contextmanager
def profile_steps(out_dir: str | None = None, warmup: int | None = None,
                  active: int | None = None, level: str | None = None) -> Iterator[profile]:
    """Profile ``active`` steady-state steps after skipping ``warmup``.

    Args:
        out_dir: Output directory; defaults to ``$PROF_OUT`` or ``./prof``.
        warmup: Steps to skip; defaults to ``$PROF_WARMUP`` or 3.
        active: Steady-state steps to profile; defaults to ``$PROF_ACTIVE`` or 3.
        level: ``Level0`` / ``Level1`` / ``Level2``; defaults to ``$PROF_LEVEL``
            or ``Level1``. Below ``Level1`` there is no ``step_trace_time.csv``.

    Yields:
        The profiler; call ``.step()`` once per training step.
    """
    out = Path(out_dir or os.environ.get("PROF_OUT") or DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)
    warmup = _env_int("PROF_WARMUP", DEFAULT_WARMUP) if warmup is None else warmup
    active = _env_int("PROF_ACTIVE", DEFAULT_ACTIVE) if active is None else active
    name = level or os.environ.get("PROF_LEVEL") or "Level1"
    experimental = _ExperimentalConfig(profiler_level=_LEVELS.get(name, ProfilerLevel.Level1))
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            schedule=schedule(wait=0, warmup=warmup, active=active, repeat=1),
            on_trace_ready=tensorboard_trace_handler(str(out)),
            experimental_config=experimental,
            record_shapes=True,
            with_stack=True,
    ) as prof:
        yield prof
    torch_npu.npu.synchronize()
