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
"""Tests for structured Trainer terminal logging."""

import logging
from types import SimpleNamespace

from hyper_parallel.auto_models.trainer.callbacks import LoggingCallback, TrainerState


class _FakeTqdmCallback:
    """Record tqdm-compatible terminal messages."""

    def __init__(self, handles_message: bool = True) -> None:
        self.handles_message = handles_message
        self.messages: list[str] = []

    def write(self, message: str) -> bool:
        """Record or decline one terminal message."""
        if not self.handles_message:
            return False
        self.messages.append(message)
        return True


def _build_trainer(
    logging_steps: int,
    global_rank: int = 0,
    tqdm_callback: _FakeTqdmCallback | None = None,
) -> SimpleNamespace:
    """Build the callback's minimal Trainer dependency surface."""
    return SimpleNamespace(
        mesh=None,
        global_rank=global_rank,
        config=SimpleNamespace(
            training=SimpleNamespace(logging_steps=logging_steps),
        ),
        tqdm_callback=tqdm_callback,
        step_env_metrics={
            "training/total_loss": 2.5,
            "training/grad_norm": 0.75,
            "training/lr": 1.0e-4,
            "performance/tokens_per_second": 128.1254,
        },
    )


def test_logging_callback_writes_complete_metrics_through_tqdm() -> None:
    """Write one complete line without duplicating the same global step."""
    tqdm_callback = _FakeTqdmCallback()
    callback = LoggingCallback(
        _build_trainer(logging_steps=2, tqdm_callback=tqdm_callback)
    )

    callback.on_step_end(TrainerState(global_step=1, epoch=0))
    callback.on_step_end(TrainerState(global_step=2, epoch=0))
    callback.on_step_end(TrainerState(global_step=2, epoch=0))

    assert tqdm_callback.messages == [
        "step=2 epoch=0 performance/tokens_per_second=128.125 "
        "training/grad_norm=0.75 training/lr=0.0001 training/total_loss=2.5"
    ]


def test_logging_callback_falls_back_to_standard_logging(caplog) -> None:
    """Use standard logging when no active tqdm bar handles the message."""
    callback = LoggingCallback(
        _build_trainer(
            logging_steps=1,
            tqdm_callback=_FakeTqdmCallback(handles_message=False),
        )
    )

    with caplog.at_level(logging.INFO):
        callback.on_step_end(TrainerState(global_step=1, epoch=3))

    assert "step=1 epoch=3" in caplog.text
    assert "training/total_loss=2.5" in caplog.text


def test_logging_callback_skips_disabled_and_nonzero_rank() -> None:
    """Suppress output when cadence is disabled or the process is not rank zero."""
    disabled_tqdm = _FakeTqdmCallback()
    nonzero_tqdm = _FakeTqdmCallback()
    disabled = LoggingCallback(
        _build_trainer(logging_steps=0, tqdm_callback=disabled_tqdm)
    )
    nonzero = LoggingCallback(
        _build_trainer(
            logging_steps=1,
            global_rank=1,
            tqdm_callback=nonzero_tqdm,
        )
    )

    disabled.on_step_end(TrainerState(global_step=1))
    nonzero.on_step_end(TrainerState(global_step=1))

    assert disabled_tqdm.messages == []
    assert nonzero_tqdm.messages == []
