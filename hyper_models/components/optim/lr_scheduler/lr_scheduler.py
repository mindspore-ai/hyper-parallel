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
"""YAML-callable learning-rate scheduler factories."""

from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)


def cosine_with_warmup(
    *,
    optimizer: Optimizer,
    train_steps: int,
    lr_warmup_steps: Optional[int] = None,
    lr_decay_steps: Optional[int] = None,
    init_lr: Optional[float] = None,
    max_lr: Optional[float] = None,
    min_lr: Optional[float] = None,
) -> LRScheduler:
    """Create a cosine scheduler with an optional linear warmup.

    Args:
        optimizer: Runtime optimizer injected by the trainer.
        train_steps: Total number of optimizer steps.
        lr_warmup_steps: Number of linear warmup steps.
        lr_decay_steps: Number of cosine decay steps. Defaults to the
            remaining steps after warmup.
        init_lr: Learning rate at the start of warmup.
        max_lr: Learning rate used to normalize the warmup start factor.
        min_lr: Minimum learning rate reached by cosine decay.

    Returns:
        A single runtime learning-rate scheduler.

    Raises:
        ValueError: If the configured step counts or warmup rates are invalid.
    """
    if train_steps < 1:
        raise ValueError("train_steps must be at least 1")

    warmup_steps = 0 if lr_warmup_steps is None else lr_warmup_steps
    if warmup_steps < 0:
        raise ValueError("lr_warmup_steps must not be negative")

    decay_steps = (
        train_steps - warmup_steps
        if lr_decay_steps is None
        else lr_decay_steps
    )
    if decay_steps < 1:
        raise ValueError("lr_decay_steps must be at least 1")

    optimizer_lr = optimizer.param_groups[0]["lr"]
    minimum_lr = 0.0 if min_lr is None else min_lr
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=decay_steps,
        eta_min=minimum_lr,
    )
    if warmup_steps == 0:
        return cosine

    initial_lr = optimizer_lr if init_lr is None else init_lr
    maximum_lr = optimizer_lr if max_lr is None else max_lr
    if maximum_lr <= 0:
        raise ValueError("max_lr must be positive when warmup is enabled")

    start_factor = initial_lr / maximum_lr
    if not 0.0 < start_factor <= 1.0:
        raise ValueError("init_lr must be greater than 0 and no greater than max_lr")

    warmup = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


__all__ = ["cosine_with_warmup"]
