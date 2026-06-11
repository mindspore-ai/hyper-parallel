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

"""Learning rate schedule utilities for HyperParallel optimizers."""

import copy
import logging
import math
from typing import Any, Dict, Iterator, List, Optional

from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer

logger = logging.getLogger(__name__)

SUPPORTED_DECAY_STYLES = {"constant", "linear", "cosine", "WSD"}
SUPPORTED_WSD_STYLES = {"linear", "cosine", "exponential", "minus_sqrt"}


class OptimizerParamScheduler:
    """Anneals learning rate and weight decay for a SINGLE optimizer.

    Aligned with LambdaLR-based scheduler:
    - LR values are bit-identical for constant / linear / cosine decay styles.
    - Different param groups (e.g. muon vs adamw) naturally get different peak LRs
      via ``param_group['initial_lr']`` (set by optimizer constructor), without
      injecting extra keys into param_groups.
    """

    def __init__(
            self,
            optimizer: Any,
            init_lr: float,
            lr_start: float,
            min_lr: float,
            lr_warmup_steps: int,
            lr_decay_steps: int,
            lr_decay_style: str,
            lr_decay_ratio: float = 1.0,
            wsd_decay_steps: Optional[int] = None,
            lr_wsd_decay_style: str = "exponential",
            override_opt_param_scheduler: bool = False
    ):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.lr_start = lr_start
        self.min_lr = min_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_style = lr_decay_style
        self.lr_decay_ratio = lr_decay_ratio

        self.wsd_decay_steps = wsd_decay_steps
        self.lr_wsd_decay_style = lr_wsd_decay_style

        self.override_opt_param_scheduler = override_opt_param_scheduler
        self.num_steps = 0

        # Ensure every param_group has 'initial_lr' (same as PyTorch LambdaLR does).
        # This is the peak LR for each group — set by the optimizer constructor
        # (e.g. muon lr=0.001, adamw lr=0.0001), so different sub-optimizers
        # naturally get different peak LRs without injecting extra keys.
        for pg in self.optimizer.param_groups:
            pg.setdefault('initial_lr', pg['lr'])

        self._validate_params()

        # Set the learning rate
        self.step(0)

    def _validate_params(self) -> None:
        """Validate initialization parameters to ensure logical correctness."""
        if self.min_lr < 0.0:
            raise ValueError("min_lr must be >= 0.0")

        if self.init_lr < self.min_lr:
            raise ValueError("init_lr must be >= min_lr")

        if self.lr_decay_steps <= 0:
            raise ValueError("lr_decay_steps must be > 0")

        if self.lr_warmup_steps >= self.lr_decay_steps:
            raise ValueError("warmup_steps must be < decay_steps")

        if self.lr_decay_style == "WSD":
            if self.wsd_decay_steps is None:
                raise ValueError("wsd_decay_steps must be not None")

            if self.wsd_decay_steps <= 0:
                raise ValueError("wsd_decay_steps must be > 0")

            if self.wsd_decay_steps > self.lr_decay_steps:
                raise ValueError("wsd_decay_steps must be <= lr_decay_steps")

    def get_lr(self, param_group: Dict[str, Any]) -> float:
        """Calculate and return the learning rate based on current step and decay style."""
        max_lr = param_group['initial_lr']
        init_lr = self.init_lr
        min_lr_ratio = self.min_lr / init_lr if init_lr > 0 else 0.0

        # 1. Linear warmup (exclusive boundary: step < warmup_steps, same as origin).
        if self.lr_warmup_steps > 0 and self.num_steps < self.lr_warmup_steps:
            progress = float(self.num_steps) / float(self.lr_warmup_steps)
            factor = (self.lr_start + (init_lr - self.lr_start) * progress) / init_lr
            return factor * max_lr

        # If the learning rate is constant, just return the peak value.
        if self.lr_decay_style == 'constant':
            return max_lr

        # 2. Decay period
        # 2.1 WSD decay
        if self.lr_decay_style == 'WSD':
            # WSD: Warmup -> Stable -> Decay
            # For WSD, lr_decay_steps is the total schedule length (no lr_decay_ratio applied).
            if self.num_steps > self.lr_decay_steps:
                return min_lr_ratio * max_lr

            wsd_anneal_start = self.lr_decay_steps - (self.wsd_decay_steps or 0)

            if self.num_steps <= wsd_anneal_start:
                return max_lr  # Stable Phase: keep max_lr without decaying

            # Final decay phase of WSD
            wsd_decay_ratio = float(self.num_steps - wsd_anneal_start) / float(self.wsd_decay_steps or 1)

            if self.lr_wsd_decay_style == "linear":
                coeff = 1.0 - wsd_decay_ratio
            elif self.lr_wsd_decay_style == "cosine":
                coeff = 0.5 * (math.cos(math.pi * wsd_decay_ratio) + 1.0)
            elif self.lr_wsd_decay_style == "exponential":
                coeff = (2.0 * (0.5 ** wsd_decay_ratio)) - 1.0
            else:  # minus_sqrt fallback
                coeff = 1.0 - math.sqrt(wsd_decay_ratio)

            factor = max(0.0, coeff) * (1 - min_lr_ratio) + min_lr_ratio
            return factor * max_lr

        # 2.2 Non-WSD decay: use lr_decay_ratio to compute effective decay_steps
        lr_decay_steps = int(self.lr_decay_steps * self.lr_decay_ratio)
        if self.num_steps > lr_decay_steps:
            return min_lr_ratio * max_lr

        decay_ratio = float(self.num_steps - self.lr_warmup_steps) / float(
            max(1, lr_decay_steps - self.lr_warmup_steps)
        )

        decay_ratio = max(0.0, min(1.0, decay_ratio))

        if self.lr_decay_style == 'linear':
            factor = max(min_lr_ratio, 1.0 - decay_ratio)
            return factor * max_lr

        if self.lr_decay_style == 'cosine':
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            factor = coeff * (1 - min_lr_ratio) + min_lr_ratio
            return max(0.0, factor) * max_lr

        raise ValueError(f"Unsupported decay style: {self.lr_decay_style}")

    def step(self, increment: int = 1) -> None:
        """Advance the scheduler steps and set lr for all parameters groups."""
        self.num_steps += increment
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(param_group)

    def get_last_lr(self) -> List[float]:
        """Return the current learning rate for all parameter groups."""
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the scheduler as a dict."""
        num_groups = len(self.optimizer.param_groups)

        # LambdaLR.load_state_dict compatible layout
        state_dict = {
            "last_epoch": self.num_steps,
            "base_lrs": [pg['initial_lr'] for pg in self.optimizer.param_groups],
            "_last_lr": [pg['lr'] for pg in self.optimizer.param_groups],
            "_step_count": self.num_steps + 1,
            "lr_lambdas": [None] * num_groups,
        }

        # OptimizerParamScheduler config
        state_dict.update({
            "initial_lr": self.init_lr,
            "lr_warmup_steps": self.lr_warmup_steps,
            "lr_decay_steps": self.lr_decay_steps,
            "lr_decay_style": self.lr_decay_style,
            "lr_decay_ratio": self.lr_decay_ratio,
            "min_lr": self.min_lr,
            "wsd_decay_steps": self.wsd_decay_steps,
            "lr_wsd_decay_style": self.lr_wsd_decay_style
        })

        return state_dict

    def _check_and_set(self, cls_value: Any, sd_value: Any, name: str) -> Any:
        """Strong validation logic during checkpoint loading."""
        if self.override_opt_param_scheduler:
            return cls_value
        if cls_value != sd_value:
            logger.warning("Scheduler Config Override: %s changed from %s to %s.", name, sd_value, cls_value)
        return cls_value

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state and immediately refresh the underlying optimizer's learning rate.

        Compatible with both:
        - OptimizerParamScheduler state dicts (last_epoch, lr_warmup_steps, ...)
        - PyTorch LambdaLR state dicts (last_epoch, base_lrs, _step_count, ...)
        """
        # Restore num_steps from either format
        if 'last_epoch' in state_dict:
            self.num_steps = state_dict['last_epoch']
        elif 'num_steps' in state_dict:
            self.num_steps = state_dict['num_steps']
        else:
            self.num_steps = 0

        # Restore scheduler config if present (OptimizerParamScheduler format only;
        # LambdaLR checkpoints lack these keys and will be silently skipped).
        config_keys = [
            ('lr_warmup_steps', 'warmup_steps'),
            ('lr_decay_steps', 'decay_steps'),
            ('lr_decay_style', 'decay_style'),
            ('lr_decay_ratio', 'decay_ratio'),
            ('min_lr', 'min_lr'),
            ('init_lr', 'initial_lr'),
            ('wsd_decay_steps', 'wsd_decay_steps'),
            ('lr_wsd_decay_style', 'lr_wsd_decay_style'),
        ]
        for attr, log_name in config_keys:
            if attr in state_dict:
                setattr(self, attr, self._check_and_set(getattr(self, attr), state_dict[attr], log_name))

        # Recompute LR for current step without advancing the counter
        self.step(0)


class LRSchedulersContainer:
    """Container for multiple learning rate schedulers.

    Each scheduler is keyed by the same name as its corresponding sub-optimizer
    in ``ChainedOptimizer.optimizers_dict`` (e.g. ``"muon"``, ``"adamw"``).
    This ensures that ``state_dict`` / ``load_state_dict`` are robust to
    insertion-order differences between the save and load environments.
    """

    def __init__(self, optimizers: ChainedOptimizer, scheduler_kwargs: Dict[str, Any]) -> None:
        self._names: List[str] = list(optimizers.optimizers_dict.keys())
        self._schedulers_by_name: Dict[str, OptimizerParamScheduler] = {}
        for name, opt in optimizers.optimizers_dict.items():
            self._schedulers_by_name[name] = OptimizerParamScheduler(
                optimizer=opt, **scheduler_kwargs,
            )
        self.schedulers: List[OptimizerParamScheduler] = [
            self._schedulers_by_name[name] for name in self._names
        ]

    def __iter__(self) -> Iterator[OptimizerParamScheduler]:
        """Iterate over the registered schedulers."""
        return iter(self.schedulers)

    def __len__(self) -> int:
        """Return the total number of schedulers in the container."""
        return len(self.schedulers)

    def step(self) -> None:
        """Advance the step for all schedulers."""
        for scheduler in self.schedulers:
            scheduler.step()

    def get_last_lr(self) -> List[float]:
        """Return a flattened list of the last learning rates across all sub-schedulers."""
        param_last_lr: List[float] = []
        for scheduler in self.schedulers:
            param_last_lr.extend(scheduler.get_last_lr())
        return param_last_lr

    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler states keyed by sub-optimizer name.

        Compatible with veomni's ``{'muon': ..., 'adamw': ...}`` format.
        """
        return {name: self._schedulers_by_name[name].state_dict() for name in self._names}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler states keyed by sub-optimizer name.

        Matches by name rather than position, so the checkpoint key order
        (e.g. ``muon, adamw``) need not match the local creation order
        (e.g. ``adamw, muon``).
        """
        if len(self._names) != len(state_dict):
            raise RuntimeError(
                f"Scheduler count mismatch! Current has {len(self._names)}, "
                f"but checkpoint contains {len(state_dict)} states."
            )
        for name in self._names:
            if name not in state_dict:
                raise RuntimeError(
                    f"Missing state for scheduler '{name}' in state_dict. "
                    f"Available keys: {sorted(state_dict.keys())}, "
                    f"expected keys: {sorted(self._names)}."
                )
            self._schedulers_by_name[name].load_state_dict(
                copy.deepcopy(state_dict[name]),
            )


def get_hyper_lr_scheduler(
        optimizer: ChainedOptimizer,
        total_steps: int,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.0,
        decay_style: str = "cosine",
        lr: float = 1e-4,
        lr_min: float = 1e-7,
        lr_start: float = 0.0,
        lr_decay_ratio: float = 1.0,
        wsd_decay_steps: Optional[int] = None,
        lr_wsd_decay_style: str = "exponential",
        override_opt_param_scheduler: bool = False,
) -> LRSchedulersContainer:
    """Create a learning rate scheduler compatible with HyperParallel optimizers.

    Example:
        from hyper_parallel.core.optimizer.lr_scheduler import get_hyper_lr_scheduler

        lr_scheduler = get_hyper_lr_scheduler(
                optimizer=optimizer,
                total_steps=train_steps,
                warmup_steps=0,
                warmup_ratio=lr_warmup_ratio,
                decay_style=lr_decay_style,
                lr_decay_ratio=lr_decay_ratio,
                lr_min=lr_min,
                lr=lr,
                lr_start=lr_start,
            )
        return lr_scheduler
    """

    if decay_style not in SUPPORTED_DECAY_STYLES:
        raise ValueError(
            f"Unknown decay_style '{decay_style}'. "
            f"Supported: {sorted(SUPPORTED_DECAY_STYLES)}"
        )

    if decay_style == "WSD" and lr_wsd_decay_style not in SUPPORTED_WSD_STYLES:
        raise ValueError(
            f"Unknown lr_wsd_decay_style '{lr_wsd_decay_style}'. "
            f"Supported: {sorted(SUPPORTED_WSD_STYLES)}"
        )

    # pylint: disable=chained-comparison
    if warmup_steps <= 0 and warmup_ratio > 0:
        warmup_steps = int(total_steps * warmup_ratio)

    scheduler_kwargs = {
        "init_lr": lr,
        "lr_start": lr_start,
        "min_lr": lr_min,
        "lr_warmup_steps": warmup_steps,
        "lr_decay_steps": total_steps,
        "lr_decay_style": decay_style,
        "lr_decay_ratio": lr_decay_ratio,
        "wsd_decay_steps": wsd_decay_steps,
        "lr_wsd_decay_style": lr_wsd_decay_style,
        "override_opt_param_scheduler": override_opt_param_scheduler,
    }

    return LRSchedulersContainer(optimizers=optimizer, scheduler_kwargs=scheduler_kwargs)
