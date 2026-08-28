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
from typing import TYPE_CHECKING, Any, Dict, Iterator, List

from torch.optim.lr_scheduler import LambdaLR

from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer

if TYPE_CHECKING:
    from torch.optim import Optimizer

logger = logging.getLogger(__name__)



def get_constant_schedule_with_warmup(
    optimizer: "Optimizer",
    num_warmup_steps: int,
    init_lr: float,
    last_epoch: int = -1,
    lr_start: float = 0.0,
):
    """
    Creates a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.
    """

    def _lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            warmup_progress = current_step / max(1, num_warmup_steps)
            return (lr_start + (init_lr - lr_start) * warmup_progress) / init_lr

        return 1.0

    return LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: "Optimizer",
    num_warmup_steps: int,
    num_training_steps: int,
    init_lr: float,
    last_epoch: int = -1,
    min_lr: float = 1e-7,
    lr_start: float = 0.0,
):
    """
    Creates a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    """

    def _lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            warmup_progress = current_step / max(1, num_warmup_steps)
            return (lr_start + (init_lr - lr_start) * warmup_progress) / init_lr

        min_lr_ratio = min_lr / init_lr if init_lr != 0.0 else 0.0
        return max(
            min_lr_ratio,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: "Optimizer",
    num_warmup_steps: int,
    num_training_steps: int,
    init_lr: float,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    lr_decay_ratio: float = 1.0,
    min_lr: float = 1e-7,
    lr_start: float = 0.0,
):
    """
    Creates a schedule with a learning rate that decreases following the values of the cosine function between
    the initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0
    and the initial lr set in the optimizer.
    """

    def lr_lambda(current_step: int):
        lr_decay_steps = int(num_training_steps * lr_decay_ratio)
        if current_step < num_warmup_steps:
            warmup_progress = current_step / max(1, num_warmup_steps)
            return (lr_start + (init_lr - lr_start) * warmup_progress) / init_lr

        min_lr_ratio = min_lr / init_lr if init_lr != 0.0 else 0.0
        if current_step > lr_decay_steps:
            return min_lr_ratio

        progress = float(current_step - num_warmup_steps) / float(max(1, lr_decay_steps - num_warmup_steps))
        assert 0 <= progress <= 1
        factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        factor = factor * (1 - min_lr_ratio) + min_lr_ratio
        return max(0, factor)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class LRSchedulersContainer:
    """Container for multiple learning rate schedulers.

    Each scheduler is keyed by the same name as its corresponding sub-optimizer
    in ``ChainedOptimizer.optimizers_dict`` (e.g. ``"muon"``, ``"adamw"``).
    This ensures that ``state_dict`` / ``load_state_dict`` are robust to
    insertion-order differences between the save and load environments.
    """
    # , **scheduler_kwargs,, scheduler_kwargs: Dict[str, Any]
    def __init__(self, optimizers: ChainedOptimizer, scheduler) -> None:
        self._names: List[str] = list(optimizers.optimizers_dict.keys())
        self._schedulers_by_name = {}
        for name, opt in optimizers.optimizers_dict.items():
            self._schedulers_by_name[name] = scheduler(
                optimizer=opt
            )
        self.schedulers = [self._schedulers_by_name[name] for name in self._names]

    def __iter__(self) -> Iterator:
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
