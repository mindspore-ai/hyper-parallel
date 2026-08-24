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
"""Tests for scheduled Trainer evaluation placeholder."""

import logging
from types import SimpleNamespace

from hyper_parallel.auto_models.trainer.callbacks import EvaluateCallback, TrainerState


def _build_trainer(eval_steps: int, eval_epochs: int) -> SimpleNamespace:
    """Build the callback's minimal Trainer dependency surface."""
    training = SimpleNamespace(
        eval_steps=eval_steps,
        eval_epochs=eval_epochs,
    )
    return SimpleNamespace(
        mesh=None,
        global_rank=0,
        model=SimpleNamespace(training=True),
        config=SimpleNamespace(training=training),
    )


def test_evaluate_callback_schedules_and_deduplicates_same_step(monkeypatch) -> None:
    """Trigger step evaluation once when step and epoch cadences overlap."""
    callback = EvaluateCallback(_build_trainer(eval_steps=2, eval_epochs=1))
    calls = []
    monkeypatch.setattr(
        callback,
        "_evaluate",
        lambda state, trigger: calls.append((state.global_step, trigger)),
    )
    state = TrainerState(global_step=2, epoch=0)

    callback.on_step_end(state)
    callback.on_step_end(state)
    callback.on_epoch_end(state)

    assert calls == [(2, "step")]


def test_evaluate_callback_supports_epoch_cadence(monkeypatch) -> None:
    """Trigger evaluation after the configured number of completed epochs."""
    callback = EvaluateCallback(_build_trainer(eval_steps=0, eval_epochs=2))
    calls = []
    monkeypatch.setattr(
        callback,
        "_evaluate",
        lambda state, trigger: calls.append((state.global_step, trigger)),
    )

    callback.on_epoch_end(TrainerState(global_step=3, epoch=0))
    callback.on_epoch_end(TrainerState(global_step=6, epoch=1))

    assert calls == [(6, "epoch")]


def test_evaluate_callback_warns_without_changing_model_mode(caplog) -> None:
    """Make placeholder behavior explicit without entering evaluation mode."""
    trainer = _build_trainer(eval_steps=1, eval_epochs=0)
    callback = EvaluateCallback(trainer)

    with caplog.at_level(logging.WARNING):
        callback.on_step_end(TrainerState(global_step=1))

    assert trainer.model.training
    assert "Evaluation triggered at step 1 by step" in caplog.text
    assert "validation loop is not implemented yet" in caplog.text
