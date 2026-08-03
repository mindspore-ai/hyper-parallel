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
"""Tests for the temporary Trainer tqdm progress callback."""

from types import SimpleNamespace

from hyper_models.trainer.callbacks import TempLogCallback, TrainerState


class _FakeProgressBar:
    """Record progress-bar updates without writing terminal output."""

    def __init__(self, total: int, initial: int) -> None:
        self.total = total
        self.initial = initial
        self.postfix = None
        self.updates = 0
        self.closed = False

    def set_postfix(self, postfix: dict) -> None:
        """Record the latest metric postfix."""
        self.postfix = postfix

    def update(self, amount: int) -> None:
        """Record a progress increment."""
        self.updates += amount

    def close(self) -> None:
        """Record progress-bar closure."""
        self.closed = True


def _build_trainer(global_rank: int) -> SimpleNamespace:
    """Build the callback's minimal Trainer dependency surface."""
    optimizer = SimpleNamespace(param_groups=[{"lr": 1.0e-4}])
    return SimpleNamespace(
        global_rank=global_rank,
        mesh=None,
        optimizer=[optimizer],
        train_steps=5,
    )


def test_temp_log_callback_updates_progress_on_rank_zero(monkeypatch) -> None:
    """Update one rank-zero progress bar with optimizer-step metrics."""
    callback = TempLogCallback(_build_trainer(global_rank=0))
    state = TrainerState(global_step=3, epoch=1)
    progress_bar = _FakeProgressBar(total=5, initial=3)
    monkeypatch.setattr(
        callback,
        "_create_progress_bar",
        lambda total, initial: progress_bar,
    )

    callback.on_train_begin(state)
    callback.on_step_end(
        state,
        loss=2.5,
        loss_dict={"foundation_loss": 2.5},
        grad_norm=0.75,
    )

    assert progress_bar.total == 5
    assert progress_bar.initial == 3
    assert progress_bar.updates == 1
    assert progress_bar.postfix == {
        "loss": "2.5",
        "grad_norm": "0.75",
        "lr": "0.0001",
        "foundation_loss": "2.5",
    }

    callback.on_train_end(state)

    assert progress_bar.closed
    assert callback._progress_bar is None


def test_temp_log_callback_skips_nonzero_rank(monkeypatch) -> None:
    """Avoid duplicate progress bars from nonzero distributed ranks."""
    callback = TempLogCallback(_build_trainer(global_rank=1))
    calls = []
    monkeypatch.setattr(
        callback,
        "_create_progress_bar",
        lambda total, initial: calls.append((total, initial)),
    )

    callback.on_train_begin(TrainerState(global_step=1, epoch=0))

    assert calls == []
    assert callback._progress_bar is None
