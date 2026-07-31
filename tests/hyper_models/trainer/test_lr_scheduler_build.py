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
"""Tests for Trainer-owned learning-rate scheduler construction."""

import torch

from hyper_models.components.optim import cosine_with_warmup
from hyper_models.trainer.base import BaseTrainer
from hyper_models.trainer.config import Target, TrainerConfig, TrainingConfig


def _unused_target() -> Target:
    """Return a target required by ``TrainerConfig`` but unused in this test."""
    return Target(lambda: None, target_path="tests.unused")


def test_trainer_builds_lr_scheduler_from_training_max_steps() -> None:
    """Pass only optimizer and dataset-derived update count to the LR builder."""
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.config = TrainerConfig(
        model=_unused_target(),
        optimizer=_unused_target(),
        training=TrainingConfig(max_steps=7),
        lr_scheduler=Target(
            cosine_with_warmup,
            target_path=(
                "hyper_models.components.optim.lr_scheduler.lr_scheduler."
                "cosine_with_warmup"
            ),
            lr_warmup_steps=1,
        ),
    )
    trainer.optimizer = optimizer
    trainer.train_steps = trainer.config.training.max_steps

    trainer._build_lr_scheduler()

    assert trainer.lr_scheduler.optimizer is optimizer


def test_trainer_allows_lr_scheduler_to_be_disabled() -> None:
    """Keep the LR scheduler optional without creating another component."""
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.config = TrainerConfig(
        model=_unused_target(),
        optimizer=_unused_target(),
        training=TrainingConfig(max_steps=7),
        lr_scheduler=None,
    )
    trainer.optimizer = object()

    trainer._build_lr_scheduler()

    assert trainer.lr_scheduler is None
