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
"""Typed learning-rate scheduler configurations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)


class LRSchedulerConfig(ABC):
    """Configuration contract for constructing runtime LR schedulers."""

    @abstractmethod
    def build(
        self,
        optimizers: Optimizer | list[Optimizer],
        total_steps: int,
    ) -> list[LRScheduler]: ...


@dataclass(kw_only=True, slots=True)
class CosineWithWarmup(LRSchedulerConfig):
    """Step-based learning-rate scheduler configuration."""

    # ── LR decay ──
    lr_warmup_steps: Optional[int] = None
    lr_decay_steps: Optional[int] = None
    lr_decay_style: str = "cosine"
    init_lr: Optional[float] = None
    max_lr: Optional[float] = None
    min_lr: Optional[float] = None

    # ── Weight Decay scheduling ──
    start_wd: Optional[float] = None
    end_wd: Optional[float] = None
    wd_incr_steps: Optional[int] = None
    wd_incr_style: str = "constant"

    # ── WSD mode ──
    wsd_decay_steps: Optional[int] = None
    lr_wsd_decay_style: Optional[str] = None

    # ── Advanced ──
    use_checkpoint_opt_param_scheduler: bool = True
    override_opt_param_scheduler: bool = False

    def build(
        self,
        optimizers: Optimizer | list[Optimizer],
        total_steps: int,
    ) -> list[LRScheduler]:
        """Build one scheduler for each optimizer."""
        if total_steps < 1:
            raise ValueError("total_steps must be at least 1")

        lr_warmup_steps = self.lr_warmup_steps or 0
        lr_decay_steps = self.lr_decay_steps or (total_steps - lr_warmup_steps)
        optimizer_list = optimizers if isinstance(optimizers, list) else [optimizers]
        first_optimizer = optimizer_list[0]
        init_lr = (
            self.init_lr
            if self.init_lr is not None
            else first_optimizer.param_groups[0]["lr"]
        )
        max_lr = (
            self.max_lr
            if self.max_lr is not None
            else first_optimizer.param_groups[0]["lr"]
        )
        min_lr = self.min_lr if self.min_lr is not None else 0.0

        schedulers = []
        for optimizer in optimizer_list:
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(1, lr_decay_steps),
                eta_min=min_lr,
            )
            if lr_warmup_steps == 0:
                schedulers.append(cosine)
                continue

            warmup = LinearLR(
                optimizer,
                start_factor=init_lr / max_lr,
                end_factor=1.0,
                total_iters=lr_warmup_steps,
            )
            schedulers.append(
                SequentialLR(
                    optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[lr_warmup_steps],
                )
            )
        return schedulers




__all__ = ["CosineWithWarmup", "LRSchedulerConfig"]