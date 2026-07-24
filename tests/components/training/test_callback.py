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
"""Tests for the callback system — StepState / CallbackManager / built-ins (03 §4.2)."""

import dataclasses
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hyper_models.components.training.callback import (
    CallbackManager,
    CheckpointCallback,
    EvaluateCallback,
    GCCallback,
    LoggingCallback,
    SIGTERMHandler,
    StepState,
    TqdmCallback,
    TrainingCallback,
    WandbCallback,
    build_callback_manager,
)


def _make_state(**overrides):
    defaults = dict(
        step=1, epoch=0, is_final_step=False,
        is_ckpt_step=False, is_val_step=False, is_log_step=False,
        is_gc_step=False, sigterm_received=False,
        loss=0.5, grad_norm=1.0, lr=1e-3, tps=100.0, mfu=0.3, num_tokens=128,
    )
    defaults.update(overrides)
    return StepState(**defaults)


# ── StepState ──

def test_step_state_frozen():
    state = _make_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.step = 99


def test_step_state_all_fields():
    state = _make_state(step=7, epoch=2, loss=0.25, num_tokens=64)
    assert state.step == 7
    assert state.epoch == 2
    assert state.loss == 0.25
    assert state.num_tokens == 64
    assert state.is_final_step is False
    assert state.grad_norm == 1.0


# ── CallbackManager ──

def test_callback_manager_init():
    manager = CallbackManager()
    assert manager._callbacks == []


def test_callback_manager_register():
    manager = CallbackManager()
    manager.register(TrainingCallback())
    manager.register(TrainingCallback())
    assert len(manager._callbacks) == 2


def test_callback_manager_on_step_end():
    calls = []

    class Recorder(TrainingCallback):
        def __init__(self, tag):
            self.tag = tag

        def on_step_end(self, state):
            calls.append((self.tag, state.step))

    manager = CallbackManager()
    manager.register(Recorder("a"))
    manager.register(Recorder("b"))
    manager.on_step_end(_make_state(step=3))
    assert calls == [("a", 3), ("b", 3)]  # 按注册顺序调用


def test_callback_manager_on_train_begin():
    cb = MagicMock(spec=TrainingCallback)
    manager = CallbackManager()
    manager.register(cb)
    manager.on_train_begin()
    cb.on_train_begin.assert_called_once()


def test_callback_manager_on_train_end():
    cb = MagicMock(spec=TrainingCallback)
    manager = CallbackManager()
    manager.register(cb)
    manager.on_train_end()
    cb.on_train_end.assert_called_once()


def test_callback_manager_empty():
    manager = CallbackManager()
    manager.on_step_end(_make_state())
    manager.on_train_begin()
    manager.on_train_end()  # 不抛异常


def test_training_callback_base():
    cb = TrainingCallback()
    cb.on_step_end(_make_state())
    cb.on_train_begin()
    cb.on_train_end()  # 默认空实现不抛异常


# ── CheckpointCallback ──

def _make_recipe():
    recipe = MagicMock()
    recipe.cfg.checkpoint.checkpoint_dir = "/tmp/ckpt"
    return recipe


def test_checkpoint_callback_skip_non_ckpt_step():
    recipe = _make_recipe()
    cb = CheckpointCallback(recipe)
    cb.on_step_end(_make_state(is_ckpt_step=False))
    recipe.save_checkpoint.assert_not_called()


def test_checkpoint_callback_skip_final_step():
    recipe = _make_recipe()
    cb = CheckpointCallback(recipe)
    cb.on_step_end(_make_state(is_ckpt_step=True, is_final_step=True))
    recipe.save_checkpoint.assert_not_called()  # 最终步由训练循环统一保存


def test_checkpoint_callback_saves():
    recipe = _make_recipe()
    recipe._last_val_losses = {"validation": {"loss": 0.1, "num_tokens": 10}}
    cb = CheckpointCallback(recipe)
    cb.on_step_end(_make_state(is_ckpt_step=True, epoch=1, step=100, loss=0.7))
    recipe.save_checkpoint.assert_called_once_with(
        "/tmp/ckpt", 1, 100, 0.7,
        val_losses={"validation": {"loss": 0.1, "num_tokens": 10}},
    )
    recipe.step_scheduler.mark_epoch_ckpt_saved.assert_called_once()


# ── EvaluateCallback ──

def test_evaluate_callback_runs_on_val_step():
    recipe = _make_recipe()
    recipe.val_dataloaders = {"validation": MagicMock()}
    recipe._run_validation_epoch.return_value = {"loss": 0.2, "num_tokens": 8}
    cb = EvaluateCallback(recipe)
    cb.on_step_end(_make_state(is_val_step=True))
    recipe._run_validation_epoch.assert_called_once()
    assert recipe._last_val_losses == {"validation": {"loss": 0.2, "num_tokens": 8}}
    recipe.log_val_metrics.assert_called_once()


def test_evaluate_callback_skip():
    recipe = _make_recipe()
    recipe.val_dataloaders = {"validation": MagicMock()}
    cb = EvaluateCallback(recipe)
    cb.on_step_end(_make_state(is_val_step=False))
    recipe._run_validation_epoch.assert_not_called()


# ── LoggingCallback ──

def test_logging_callback():
    recipe = _make_recipe()
    cb = LoggingCallback(recipe)
    with patch("hyper_models.components.training.callback.logger") as mock_logger:
        cb.on_step_end(_make_state(is_log_step=True))
        mock_logger.info.assert_called_once()
        mock_logger.reset_mock()
        cb.on_step_end(_make_state(is_log_step=False))
        mock_logger.info.assert_not_called()


# ── TqdmCallback ──

def test_tqdm_callback():
    recipe = _make_recipe()
    recipe.step_scheduler = None  # getattr 兜底 → initial=0
    cb = TqdmCallback(recipe, total=10)
    with patch("hyper_models.components.training.callback._is_rank_0", return_value=True), \
         patch("tqdm.tqdm") as mock_tqdm:
        pbar = MagicMock()
        mock_tqdm.return_value = pbar
        cb.on_train_begin()
        mock_tqdm.assert_called_once()
        cb.on_step_end(_make_state(loss=0.5, lr=1e-3))
        pbar.set_postfix.assert_called_once()
        pbar.update.assert_called_once_with(1)
        cb.on_train_end()
        pbar.close.assert_called_once()


def test_tqdm_callback_no_pbar():
    cb = TqdmCallback(MagicMock(), total=10)
    cb.on_step_end(_make_state())  # pbar is None → 直接返回
    cb.on_train_end()


# ── GCCallback ──

def test_gc_callback():
    recipe = _make_recipe()
    cb = GCCallback(recipe)
    cb.on_step_end(_make_state(is_gc_step=True))
    recipe._maybe_collect_garbage.assert_called_once()
    recipe.reset_mock()
    cb.on_step_end(_make_state(is_gc_step=False))
    recipe._maybe_collect_garbage.assert_not_called()


# ── WandbCallback ──

def test_wandb_callback():
    mock_wandb = MagicMock()
    recipe = _make_recipe()
    cb = WandbCallback(recipe, project="proj")
    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        cb.on_step_end(_make_state(is_log_step=True, step=5))
        mock_wandb.log.assert_called_once()
        assert mock_wandb.log.call_args[0][0]["step"] == 5
        mock_wandb.reset_mock()
        cb.on_step_end(_make_state(is_log_step=False))
        mock_wandb.log.assert_not_called()


# ── SIGTERMHandler ──

def test_sigterm_handler():
    recipe = _make_recipe()
    cb = SIGTERMHandler(recipe)
    cb.on_step_end(_make_state(sigterm_received=False))
    recipe.step_scheduler.cleanup.assert_not_called()
    cb.on_step_end(_make_state(sigterm_received=True, step=42))
    recipe.step_scheduler.cleanup.assert_called_once()
    assert recipe.step_scheduler.max_steps == 42


# ── build_callback_manager ──

def test_build_callback_manager():
    cfg = SimpleNamespace(
        wandb=None,
        step_scheduler=SimpleNamespace(gc_every_steps=None),
    )
    manager = build_callback_manager(MagicMock(), cfg)
    assert isinstance(manager, CallbackManager)
    # 默认 5 个：Checkpoint / Evaluate / Logging / Tqdm / SIGTERM
    # （wandb 未启用、gc_every_steps 未设置时不注册）
    assert len(manager._callbacks) == 5
    types = [type(cb) for cb in manager._callbacks]
    assert types[0] is CheckpointCallback
    assert types[-1] is SIGTERMHandler


def test_build_callback_manager_with_wandb_and_gc():
    cfg = SimpleNamespace(
        wandb=SimpleNamespace(enabled=True, project="proj"),
        step_scheduler=SimpleNamespace(gc_every_steps=100),
    )
    manager = build_callback_manager(MagicMock(), cfg)
    types = [type(cb) for cb in manager._callbacks]
    assert WandbCallback in types
    assert GCCallback in types
    assert len(manager._callbacks) == 7
