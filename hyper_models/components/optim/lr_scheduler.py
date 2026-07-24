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
"""Learning-rate scheduler component configuration types — following design doc §9.6."""

from dataclasses import dataclass, field
from typing import Any, Optional

from hyper_models.config.configurable import Configurable


class LRScheduler(Configurable):
    """Base category for learning-rate scheduler components."""

    @dataclass
    class Config(Configurable.Config):
        """Base configuration accepted by the scheduler slot."""


class CosineWithWarmup(LRScheduler):
    """Cosine scheduler with a warmup phase."""

    @dataclass
    class Config(LRScheduler.Config):
        """Cosine-with-warmup scheduler parameters."""

        warmup_ratio: float = 0.1
        min_lr: float = 1e-5
        batch_size_warmup_ratio: float = 0.0

    def __init__(self, config: "CosineWithWarmup.Config") -> None:
        self.config = config


# ── LRSchedulerConfig — step-based (AutoModel compatible) ──

@dataclass
class LRSchedulerConfig:
    """LR scheduler typed config — step-based (AutoModel compatible).

    Following design doc §9.6. All step fields are absolute (not ratio).
    """

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

    def build(self, optimizer, step_scheduler) -> list:
        """Build OptimizerParamScheduler list.

        Stub — returns a simple lambda scheduler that wraps torch.optim.lr_scheduler.
        Full implementation requires porting OptimizerParamScheduler from nemo_automodel.

        Args:
            optimizer: list[Optimizer] or single Optimizer.
            step_scheduler: StepScheduler instance.

        Returns:
            list of LR schedulers.
        """
        max_steps = step_scheduler.max_steps if step_scheduler.max_steps > 0 else 1000
        lr_warmup_steps = self.lr_warmup_steps if self.lr_warmup_steps is not None else 0
        lr_decay_steps = self.lr_decay_steps or (max_steps - lr_warmup_steps)

        opt = optimizer if not isinstance(optimizer, list) else optimizer[0]
        init_lr = self.init_lr if self.init_lr is not None else opt.param_groups[0]["lr"]
        max_lr = self.max_lr if self.max_lr is not None else opt.param_groups[0]["lr"]
        min_lr = self.min_lr if self.min_lr is not None else 0.0

        # Stub: use CosineAnnealingLR with linear warmup via LambdaLR
        schedulers = []
        for single_opt in (optimizer if isinstance(optimizer, list) else [optimizer]):
            if lr_warmup_steps > 0:
                from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

                warmup_sch = LinearLR(
                    single_opt, start_factor=init_lr / max_lr,
                    end_factor=1.0, total_iters=lr_warmup_steps,
                )
                cosine_sch = CosineAnnealingLR(
                    single_opt, T_max=max(1, lr_decay_steps),
                    eta_min=min_lr,
                )
                schedulers.append(SequentialLR(
                    single_opt,
                    schedulers=[warmup_sch, cosine_sch],
                    milestones=[lr_warmup_steps],
                ))
            else:
                from torch.optim.lr_scheduler import CosineAnnealingLR
                schedulers.append(CosineAnnealingLR(
                    single_opt, T_max=max(1, lr_decay_steps), eta_min=min_lr,
                ))

        return schedulers


@dataclass
class RatioBasedLRSchedulerConfig(LRSchedulerConfig):
    """Accepts ratio parameters, converts to absolute steps in build()."""

    warmup_steps_ratio: float = 0.1
    min_lr_ratio: float = 0.0

    def build(self, optimizer, step_scheduler):
        self.lr_warmup_steps = int(step_scheduler.max_steps * self.warmup_steps_ratio) if step_scheduler.max_steps > 0 else 0
        self.lr_decay_steps = step_scheduler.max_steps - self.lr_warmup_steps if step_scheduler.max_steps > 0 else 1000
        opt = optimizer if not isinstance(optimizer, list) else optimizer[0]
        max_lr = self.max_lr or opt.param_groups[0]["lr"]
        self.min_lr = max_lr * self.min_lr_ratio
        return super().build(optimizer, step_scheduler)


__all__ = [
    "LRScheduler", "CosineWithWarmup",
    "LRSchedulerConfig", "RatioBasedLRSchedulerConfig",
]