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

from hyper_models.components.optim import LRSchedulerConfig
from hyper_models.trainer.base import BaseTrainer
from hyper_models.trainer.config import ModelConfig, TrainerConfig, TrainingConfig


def test_trainer_builds_lr_scheduler_from_training_max_steps() -> None:
    """Pass only optimizer and Trainer update count to the LR builder."""
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.config = TrainerConfig(
        model=ModelConfig(name="dummy"),
        training=TrainingConfig(max_steps=7),
        lr_scheduler=LRSchedulerConfig(lr_warmup_steps=1),
    )
    trainer.optimizer = [optimizer]

    trainer._build_lr_scheduler()

    assert isinstance(trainer.lr_scheduler, list)
    assert len(trainer.lr_scheduler) == 1
    assert trainer.lr_scheduler[0].optimizer is optimizer


def test_trainer_allows_lr_scheduler_to_be_disabled() -> None:
    """Keep the LR scheduler optional without creating another component."""
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.config = TrainerConfig(
        model=ModelConfig(name="dummy"),
        training=TrainingConfig(max_steps=7),
        lr_scheduler=None,
    )
    trainer.optimizer = []

    trainer._build_lr_scheduler()

    assert trainer.lr_scheduler is None
