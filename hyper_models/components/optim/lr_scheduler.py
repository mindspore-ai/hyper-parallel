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

from dataclasses import dataclass, replace
from typing import Optional

from torch.optim import Optimizer as TorchOptimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler as TorchLRScheduler,
    SequentialLR,
)

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

@dataclass(kw_only=True, slots=True)
class LRSchedulerConfig(LRScheduler.Config):
    """LR scheduler typed config — step-based (AutoModel compatible).

    Following design doc §9.6. All step fields are absolute (not ratio).
    Inherits from :class:`LRScheduler.Config` so it can be used in the typed
    ``TrainerConfig.lr_scheduler`` slot.
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

    def build(
        self,
        optimizer: TorchOptimizer | list[TorchOptimizer],
        *,
        max_steps: int,
    ) -> list[TorchLRScheduler]:
        """Build OptimizerParamScheduler list.

        Stub — returns a simple lambda scheduler that wraps torch.optim.lr_scheduler.
        Full implementation requires porting OptimizerParamScheduler from nemo_automodel.

        Args:
            optimizer: list[Optimizer] or single Optimizer.
            max_steps: Total number of optimizer updates in the training run.

        Returns:
            list of LR schedulers.

        Raises:
            ValueError: If no optimizer is provided or the configured step
                counts are invalid.
        """
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError(f"max_steps must be a positive integer, but got {max_steps!r}")

        optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
        if not optimizers:
            raise ValueError("optimizer must contain at least one optimizer")

        lr_warmup_steps = self.lr_warmup_steps if self.lr_warmup_steps is not None else 0
        if lr_warmup_steps < 0 or lr_warmup_steps >= max_steps:
            raise ValueError(
                f"lr_warmup_steps must be in [0, max_steps), but got "
                f"lr_warmup_steps={lr_warmup_steps}, max_steps={max_steps}"
            )

        lr_decay_steps = (
            self.lr_decay_steps
            if self.lr_decay_steps is not None
            else max_steps - lr_warmup_steps
        )
        if lr_decay_steps <= 0:
            raise ValueError(f"lr_decay_steps must be positive, but got {lr_decay_steps!r}")

        opt = optimizers[0]
        init_lr = self.init_lr if self.init_lr is not None else opt.param_groups[0]["lr"]
        max_lr = self.max_lr if self.max_lr is not None else opt.param_groups[0]["lr"]
        min_lr = self.min_lr if self.min_lr is not None else 0.0
        if lr_warmup_steps > 0 and max_lr <= 0:
            raise ValueError(f"max_lr must be positive when warmup is enabled, but got {max_lr!r}")

        # Stub: use CosineAnnealingLR with linear warmup via LambdaLR
        schedulers = []
        for single_opt in optimizers:
            if lr_warmup_steps > 0:
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
                schedulers.append(CosineAnnealingLR(
                    single_opt, T_max=max(1, lr_decay_steps), eta_min=min_lr,
                ))

        return schedulers


@dataclass(kw_only=True, slots=True)
class RatioBasedLRSchedulerConfig(LRSchedulerConfig):
    """Accepts ratio parameters, converts to absolute steps in build()."""

    warmup_steps_ratio: float = 0.1
    min_lr_ratio: float = 0.0

    def build(
        self,
        optimizer: TorchOptimizer | list[TorchOptimizer],
        *,
        max_steps: int,
    ) -> list[TorchLRScheduler]:
        """Resolve ratio settings and build schedulers without mutating the config."""
        optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
        if not optimizers:
            raise ValueError("optimizer must contain at least one optimizer")

        lr_warmup_steps = int(max_steps * self.warmup_steps_ratio)
        max_lr = self.max_lr if self.max_lr is not None else optimizers[0].param_groups[0]["lr"]
        resolved_config = replace(
            self,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=max_steps - lr_warmup_steps,
            min_lr=max_lr * self.min_lr_ratio,
        )
        return LRSchedulerConfig.build(
            resolved_config,
            optimizer,
            max_steps=max_steps,
        )


__all__ = [
    "LRScheduler", "CosineWithWarmup",
    "LRSchedulerConfig", "RatioBasedLRSchedulerConfig",
]
