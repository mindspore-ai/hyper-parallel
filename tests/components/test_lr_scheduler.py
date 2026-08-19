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
"""Tests for direct optimizer-bound learning-rate scheduler construction."""

import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR

from hyper_models.components.optim import cosine_with_warmup


def _build_optimizer() -> torch.optim.Optimizer:
    parameter = torch.nn.Parameter(torch.ones(1))
    return torch.optim.AdamW([parameter], lr=1e-3)


def test_lr_scheduler_builds_without_warmup() -> None:
    """Bind one cosine scheduler directly to one optimizer."""
    optimizer = _build_optimizer()

    scheduler = cosine_with_warmup(
        optimizer=optimizer,
        train_steps=5,
        lr_warmup_steps=0,
        min_lr=0.0,
    )

    assert isinstance(scheduler, CosineAnnealingLR)
    assert scheduler.optimizer is optimizer


def test_lr_scheduler_builds_with_warmup() -> None:
    """Compose linear warmup and cosine decay without a config wrapper."""
    optimizer = _build_optimizer()

    scheduler = cosine_with_warmup(
        optimizer=optimizer,
        train_steps=10,
        lr_warmup_steps=2,
        init_lr=1e-4,
        max_lr=1e-3,
    )

    assert isinstance(scheduler, SequentialLR)
    assert scheduler.optimizer is optimizer


@pytest.mark.parametrize("train_steps", [0, -1])
def test_lr_scheduler_rejects_invalid_train_steps(train_steps: int) -> None:
    """Reject invalid optimizer-update counts."""
    with pytest.raises(ValueError, match="train_steps must be at least 1"):
        cosine_with_warmup(
            optimizer=_build_optimizer(),
            train_steps=train_steps,
        )
