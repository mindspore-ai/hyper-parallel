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
"""Tests for optimizer-bound learning-rate scheduler construction."""

import pytest
import torch

from hyper_models.components.optim import (
    LRSchedulerConfig,
    RatioBasedLRSchedulerConfig,
)


def _build_optimizer() -> torch.optim.Optimizer:
    parameter = torch.nn.Parameter(torch.ones(1))
    return torch.optim.AdamW([parameter], lr=1e-3)


def test_lr_scheduler_builds_without_step_scheduler() -> None:
    """Bind LR schedulers using only optimizer and total update count."""
    optimizers = [_build_optimizer(), _build_optimizer()]
    config = LRSchedulerConfig(lr_warmup_steps=1, min_lr=0.0)

    schedulers = config.build(optimizers, max_steps=5)

    assert len(schedulers) == len(optimizers)
    assert all(scheduler.optimizer is optimizer for scheduler, optimizer in zip(schedulers, optimizers))


def test_ratio_lr_scheduler_does_not_mutate_config() -> None:
    """Resolve ratio settings on a copied configuration."""
    optimizer = _build_optimizer()
    config = RatioBasedLRSchedulerConfig(
        warmup_steps_ratio=0.2,
        min_lr_ratio=0.1,
    )

    schedulers = config.build(optimizer, max_steps=10)

    assert len(schedulers) == 1
    assert config.lr_warmup_steps is None
    assert config.lr_decay_steps is None
    assert config.min_lr is None


@pytest.mark.parametrize("max_steps", [0, -1, True])
def test_lr_scheduler_rejects_invalid_max_steps(max_steps: int) -> None:
    """Reject missing or invalid optimizer-update counts."""
    with pytest.raises(ValueError, match="max_steps must be a positive integer"):
        LRSchedulerConfig().build(_build_optimizer(), max_steps=max_steps)
