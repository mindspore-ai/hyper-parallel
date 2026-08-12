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
"""Tests for the Trainer tqdm progress callback."""

from types import SimpleNamespace
from typing import Any

from hyper_models.trainer.callbacks import TempLogCallback, TqdmCallback, TrainerState


class _FakeProgressBar:
    """Record progress-bar updates without writing terminal output."""

    def __init__(self, total: int, initial: int) -> None:
        self.total = total
        self.initial = initial
        self.postfix = None
        self.updates = 0
        self.closed = False
        self.messages = []

    def set_postfix(self, postfix: dict) -> None:
        """Record the latest metric postfix."""
        self.postfix = postfix

    def update(self, amount: int) -> None:
        """Record a progress increment."""
        self.updates += amount

    def close(self) -> None:
        """Record progress-bar closure."""
        self.closed = True

    def write(self, message: str, file: Any = None) -> None:
        """Record a line written above the progress bar."""
        del file
        self.messages.append(message)


def _build_trainer(global_rank: int) -> SimpleNamespace:
    """Build the callback's minimal Trainer dependency surface."""
    return SimpleNamespace(
        global_rank=global_rank,
        mesh=None,
        train_steps=5,
        step_train_metrics={
            "training/total_loss": 2.5,
            "training/foundation_loss": 2.25,
            "training/grad_norm": 0.75,
            "training/lr": 1.0e-4,
        },
        step_env_metrics={
            "performance/tokens_per_second": 128.1254,
            "performance/step_time": 0.4567,
        },
    )


def test_tqdm_callback_updates_shared_metrics_once(monkeypatch) -> None:
    """Display shared metrics and reject duplicate step dispatch."""
    callback = TqdmCallback(_build_trainer(global_rank=0))
    state = TrainerState(global_step=2, epoch=1)
    progress_bar = _FakeProgressBar(total=5, initial=2)
    monkeypatch.setattr(callback, "_create_progress_bar", lambda total, initial: progress_bar)

    callback.on_train_begin(state)
    state.global_step = 3
    callback.on_step_end(state, loss=999.0)
    callback.on_step_end(state, loss=999.0)

    assert progress_bar.total == 5
    assert progress_bar.initial == 2
    assert progress_bar.updates == 1
    assert progress_bar.postfix == {
        "loss": "2.5",
        "grad_norm": "0.75",
        "lr": "1.000e-04",
        "foundation_loss": "2.25",
        "tokens/s": "128.125",
        "step_time": "0.457s",
    }

    callback.on_train_end(state)
    callback.on_train_end(state)

    assert progress_bar.closed
    assert callback._progress_bar is None


def test_tqdm_callback_skips_nonzero_global_rank(monkeypatch) -> None:
    """Avoid duplicate progress bars across distributed nodes."""
    callback = TqdmCallback(_build_trainer(global_rank=1))
    calls = []
    monkeypatch.setattr(callback, "_create_progress_bar", lambda total, initial: calls.append((total, initial)))

    callback.on_train_begin(TrainerState(global_step=1, epoch=0))

    assert calls == []
    assert callback._progress_bar is None


def test_tqdm_callback_writes_above_active_progress_bar(monkeypatch) -> None:
    """Expose a stable tqdm-compatible writer for metric logging."""
    callback = TqdmCallback(_build_trainer(global_rank=0))
    progress_bar = _FakeProgressBar(total=5, initial=0)
    monkeypatch.setattr(callback, "_create_progress_bar", lambda total, initial: progress_bar)
    callback.on_train_begin(TrainerState())

    assert callback.write("step=1 loss=2.5")
    assert progress_bar.messages == ["step=1 loss=2.5"]

    callback.on_train_end(TrainerState())

    assert not callback.write("step=2 loss=2.4")


def test_temp_log_callback_is_compatibility_alias() -> None:
    """Keep existing imports working while Trainer registers TqdmCallback."""
    assert TempLogCallback is TqdmCallback
