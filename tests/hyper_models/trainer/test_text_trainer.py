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
"""Tests for the TextTrainer composition and training loop."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import hyper_models.trainer.text_trainer as text_trainer_module


class _FakeBaseTrainer:
    """Record the explicit TextTrainer construction stages."""

    calls: list[str]

    def _record(self, stage: str) -> None:
        self.calls.append(stage)

    def _setup(self) -> None:
        self._record("setup")
        self.distributed_setup = object()
        self.mesh = object()
        self.dp_cp_mesh = object()

    def _build_model(self) -> None:
        self._record("model")

    def _build_model_assets(self) -> None:
        self._record("model_assets")

    def _build_data_transform(self) -> None:
        self._record("data_transform")

    def _build_dataset(self) -> None:
        self._record("dataset")

    def _build_collate_fn(self) -> None:
        self._record("collate_fn")

    def _build_dataloader(self) -> None:
        self._record("dataloader")

    def _compute_train_steps(self) -> None:
        self._record("train_steps")

    def _build_optimizer(self) -> None:
        self._record("optimizer")

    def _build_lr_scheduler(self) -> None:
        self._record("lr_scheduler")

    def _build_training_context(self) -> None:
        self._record("training_context")

    def _init_callbacks(self) -> None:
        self._record("callbacks")


def test_text_trainer_keeps_explicit_base_stage_order(monkeypatch) -> None:
    """Keep the VeOmni-style composition while using shared build stages."""
    monkeypatch.setattr(
        text_trainer_module,
        "BaseTrainer",
        _FakeBaseTrainer,
    )
    _FakeBaseTrainer.calls = []
    config = object()

    trainer = text_trainer_module.TextTrainer(config)

    assert trainer.base.config is config
    assert _FakeBaseTrainer.calls == [
        "setup",
        "model",
        "model_assets",
        "data_transform",
        "dataset",
        "collate_fn",
        "dataloader",
        "train_steps",
        "optimizer",
        "lr_scheduler",
        "training_context",
        "callbacks",
    ]
    assert trainer.distributed_setup is trainer.base.distributed_setup
    assert trainer.mesh is trainer.base.mesh
    assert trainer.dp_cp_mesh is trainer.base.dp_cp_mesh


def test_text_train_step_uses_base_distributed_runtime(monkeypatch) -> None:
    """Use BaseTrainer FSDP synchronization and optional scheduler behavior."""
    optimizer = SimpleNamespace(step=Mock(), zero_grad=Mock())
    base = SimpleNamespace(
        config=SimpleNamespace(
            training=SimpleNamespace(max_grad_norm=1.0),
        ),
        state=SimpleNamespace(global_step=0),
        model=object(),
        optimizer=optimizer,
        lr_scheduler=None,
        model_reshard=Mock(),
        _configure_fsdp_gradient_sync=Mock(),
        forward_backward_step=Mock(
            return_value=(
                torch.tensor(2.0),
                {"foundation_loss": torch.tensor(2.0)},
            )
        ),
        on_step_begin=Mock(),
        on_step_end=Mock(),
    )
    trainer = text_trainer_module.TextTrainer.__new__(
        text_trainer_module.TextTrainer
    )
    trainer.base = base
    micro_batches = [
        {"input_ids": torch.ones(2, 3), "labels": torch.ones(2, 3)}
    ]
    sync_calls: list[str] = []
    monkeypatch.setattr(text_trainer_module, "synchronize", lambda: None)
    monkeypatch.setattr(
        text_trainer_module,
        "hsdp_sync_stream",
        lambda: sync_calls.append("hsdp"),
    )
    monkeypatch.setattr(
        text_trainer_module,
        "clip_grad_norm_",
        lambda model, max_norm: torch.tensor(0.5),
    )
    monkeypatch.setattr(
        text_trainer_module,
        "SkipDTensorDispatch",
        nullcontext,
    )

    metrics = trainer.train_step(iter([micro_batches]))

    assert metrics == {"loss": 2.0, "grad_norm": 0.5}
    assert base.state.global_step == 1
    base.model_reshard.assert_called_once_with(0, 1)
    base._configure_fsdp_gradient_sync.assert_called_once_with(0, 1)
    assert sync_calls == ["hsdp"]
    optimizer.step.assert_called_once_with()
    optimizer.zero_grad.assert_called_once_with()


def test_text_train_step_does_not_advance_after_dataloader_end() -> None:
    """Leave global step unchanged when the dataloader is exhausted."""
    trainer = text_trainer_module.TextTrainer.__new__(
        text_trainer_module.TextTrainer
    )
    trainer.base = SimpleNamespace(
        config=SimpleNamespace(),
        state=SimpleNamespace(global_step=3),
    )

    with pytest.raises(StopIteration):
        trainer.train_step(iter(()))

    assert trainer.base.state.global_step == 3
