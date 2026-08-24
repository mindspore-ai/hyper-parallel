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
"""Tests for configurable Trainer loss modules."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch
from torch import nn

import hyper_parallel.auto_models.trainer.base as base_module
from hyper_parallel.auto_models.components.loss import ModelOutputLoss
from hyper_parallel.auto_models.trainer.base import BaseTrainer
from hyper_parallel.auto_models.trainer.config import Target, TrainerConfig


def _unused_target() -> Target:
    """Return a target required by ``TrainerConfig`` but unused here."""
    return Target(lambda: None, target_path="tests.unused")


def _non_module_target() -> object:
    """Return an invalid configured loss value."""
    return object()


class _ConfiguredLoss(nn.Module):
    """Small configurable loss module used to verify target construction."""

    def __init__(self, scale: float = 1.0) -> None:
        """Store the configured scale."""
        super().__init__()
        self.scale = scale

    def forward(
        self,
        *,
        model_output: Any,
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Scale the model-provided loss."""
        if labels is None:
            raise ValueError("labels are required")
        return model_output.loss * self.scale


class _RecordingModel(nn.Module):
    """Record the complete model input and expose differentiable logits."""

    def __init__(self) -> None:
        """Create the differentiable recording model."""
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.forward_inputs: Optional[dict[str, Any]] = None

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        """Return an output while retaining every received input."""
        self.forward_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "use_cache": use_cache,
        }
        logits = input_ids.float() * self.weight
        return SimpleNamespace(loss=logits.mean() + 100.0, logits=logits)


class _RecordingLoss(nn.Module):
    """Record the model output and labels received from the Trainer."""

    def __init__(self) -> None:
        """Create empty call records."""
        super().__init__()
        self.model_output: Any = None
        self.labels: Optional[torch.Tensor] = None

    def forward(
        self,
        *,
        model_output: Any,
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return a custom loss derived from logits and labels."""
        if labels is None:
            raise ValueError("labels are required")
        self.model_output = model_output
        self.labels = labels
        return model_output.logits.mean() + labels.float().mean() * 0.0


def test_model_output_loss_returns_output_loss() -> None:
    """Return the exact loss object supplied by the model output."""
    expected = torch.tensor(3.0)
    output = SimpleNamespace(loss=expected)

    actual = ModelOutputLoss()(
        model_output=output,
        labels=torch.tensor([1]),
    )

    assert actual is expected


def test_trainer_builds_model_output_loss_by_default() -> None:
    """Use ``ModelOutputLoss`` when no loss target is configured."""
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.config = TrainerConfig(
        model=_unused_target(),
        optimizer=_unused_target(),
    )

    trainer._build_loss()

    assert isinstance(trainer.loss_fn, ModelOutputLoss)


def test_trainer_builds_configured_loss_module() -> None:
    """Build a concrete loss module through the configured target."""
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.config = TrainerConfig(
        model=_unused_target(),
        optimizer=_unused_target(),
        loss_fn=Target(
            _ConfiguredLoss,
            target_path="tests.ConfiguredLoss",
            scale=2.5,
        ),
    )

    trainer._build_loss()

    assert isinstance(trainer.loss_fn, _ConfiguredLoss)
    assert trainer.loss_fn.scale == pytest.approx(2.5)


def test_trainer_rejects_configured_loss_that_is_not_module() -> None:
    """Reject configured loss targets that do not build ``nn.Module`` objects."""
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.config = TrainerConfig(
        model=_unused_target(),
        optimizer=_unused_target(),
        loss_fn=Target(
            _non_module_target,
            target_path="tests.non_module_target",
        ),
    )

    with pytest.raises(ValueError, match="must build a torch.nn.Module"):
        trainer._build_loss()


def test_forward_passes_full_batch_to_model_and_labels_to_loss(
    monkeypatch,
) -> None:
    """Keep labels in model inputs and also pass them to the loss module."""
    recorded_aggregation: dict[str, Any] = {}

    def _mean_global_loss(
        losses: torch.Tensor,
        micro_batch_token_len: dict[str, torch.Tensor],
        micro_batches_token_len: dict[str, torch.Tensor],
        device_mesh: object,
    ) -> dict[str, torch.Tensor]:
        recorded_aggregation.update(
            losses=losses,
            micro_batch_token_len=micro_batch_token_len,
            micro_batches_token_len=micro_batches_token_len,
            device_mesh=device_mesh,
        )
        return {"foundation_loss": losses}

    monkeypatch.setattr(base_module, "mean_global_loss", _mean_global_loss)

    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.device = torch.device("cpu")
    trainer.model = _RecordingModel()
    trainer.loss_fn = _RecordingLoss()
    trainer.model_fwd_context = nullcontext()
    trainer.model_bwd_context = nullcontext()
    trainer.micro_batch_token_len = {"foundation_tokens": torch.tensor(4)}
    trainer.micro_batches_token_len = {"foundation_tokens": torch.tensor(4)}
    trainer.mesh = object()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4),
        "labels": torch.tensor([[2, 3, 4, 5]]),
    }

    loss, loss_dict = trainer.forward_backward_step(batch)

    assert trainer.model.forward_inputs is not None
    assert torch.equal(trainer.model.forward_inputs["input_ids"], batch["input_ids"])
    assert torch.equal(
        trainer.model.forward_inputs["attention_mask"],
        batch["attention_mask"],
    )
    assert trainer.model.forward_inputs["use_cache"] is False
    assert trainer.loss_fn.labels is trainer.model.forward_inputs["labels"]
    assert trainer.loss_fn.model_output.logits is not None
    assert recorded_aggregation["losses"] is loss_dict["foundation_loss"]
    assert torch.allclose(loss, loss_dict["foundation_loss"])
    assert trainer.model.weight.grad is not None
