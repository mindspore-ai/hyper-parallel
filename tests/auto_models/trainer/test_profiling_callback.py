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
"""Profiling callback lifecycle tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hyper_parallel.auto_models.trainer.callbacks.base import TrainerState
from hyper_parallel.auto_models.trainer.callbacks.profiling_callback import ProfilingCallback


def _trainer(*, enabled: bool = True, rank: int = 0, global_rank: int = 0) -> SimpleNamespace:
    config = SimpleNamespace(
        enabled=enabled,
        start_step=3,
        end_step=4,
        trace_dir="/tmp/profile",
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        rank=rank,
    )
    return SimpleNamespace(
        config=SimpleNamespace(profiling=config),
        global_rank=global_rank,
        world_size=8,
        mesh=None,
    )


def test_profiling_callback_drives_profiler_lifecycle() -> None:
    """Enabled target rank creates, advances, and stops one profiler."""
    profiler = MagicMock()
    callback = ProfilingCallback(_trainer())
    state = TrainerState()

    with patch(
        "hyper_parallel.auto_models.trainer.callbacks.profiling_callback.helper.create_profiler",
        return_value=profiler,
    ) as create_profiler:
        callback.on_train_begin(state)
        callback.on_step_end(state)
        callback.on_train_end(state)

    create_profiler.assert_called_once()
    profiler.start.assert_called_once_with()
    profiler.step.assert_called_once_with()
    profiler.stop.assert_called_once_with()


def test_profiling_callback_ignores_non_target_rank() -> None:
    """Only the configured rank records a trace."""
    callback = ProfilingCallback(_trainer(rank=0, global_rank=1))

    with patch(
        "hyper_parallel.auto_models.trainer.callbacks.profiling_callback.helper.create_profiler"
    ) as create_profiler:
        callback.on_train_begin(TrainerState())

    create_profiler.assert_not_called()


@pytest.mark.parametrize(
    ("start_step", "end_step"),
    [(0, 4), (3, 3), (4, 3)],
)
def test_profiling_callback_rejects_invalid_window(start_step: int, end_step: int) -> None:
    """Enabled profiling requires a non-empty positive step interval."""
    trainer = _trainer()
    trainer.config.profiling.start_step = start_step
    trainer.config.profiling.end_step = end_step

    with pytest.raises(ValueError, match="profiling"):
        ProfilingCallback(trainer)
