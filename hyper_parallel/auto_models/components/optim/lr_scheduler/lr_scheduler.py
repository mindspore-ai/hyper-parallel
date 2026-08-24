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

from typing import Any

from hyper_parallel.core.optimizer import get_hyper_lr_scheduler


class MultiLRScheduler:
    """Build one learning-rate scheduler for each child optimizer."""

    def __init__(
        self,
        optimizer: Any,
        lr_decay_style: str,
        train_iters: int,
        lr_config: dict,
    ) -> None:
        """Initialize schedulers for every child optimizer.

        Args:
            optimizer: Optimizer or collection of optimizers to schedule.
            lr_decay_style: One of ``constant``, ``linear`` or ``cosine``.
            train_iters: Total number of training iterations.
            lr_config: Mapping with warmup/decay settings such as
                ``lr_warmup_ratio``, ``lr_warmup_steps``, ``lr_start``,
                ``lr_decay_ratio`` and ``min_lr``.
        """
        self.config = lr_config
        self.optimizer = optimizer
        self.lr_decay_style = lr_decay_style
        self.train_iters = train_iters

        lr_warmup_ratio = self.config.get('lr_warmup_ratio', 0.0)
        lr_warmup_steps = self.config.get('lr_warmup_steps')
        if lr_warmup_steps is None:
            lr_warmup_steps = int(train_iters * lr_warmup_ratio)
        lr_start = self.config.get('lr_start', 0.0)
        lr_decay_ratio = self.config.get('lr_decay_ratio', 1.0)
        min_lr = self.config.get('min_lr', self.config.get('lr_min', 1e-7))
        initial_lrs = {
            float(param_group.get("initial_lr", param_group["lr"]))
            for child_optimizer in self.optimizer.optimizers_dict.values()
            for param_group in child_optimizer.param_groups
        }
        if not initial_lrs:
            raise ValueError("MultiLRScheduler requires at least one optimizer parameter group")
        if len(initial_lrs) != 1:
            raise ValueError(
                "MultiLRScheduler on master requires one shared initial learning rate; "
                f"got {sorted(initial_lrs)}. Per-optimizer absolute minimum learning rates "
                "must be migrated separately."
            )

        self.lr_scheduler = get_hyper_lr_scheduler(
            optimizer=self.optimizer,
            total_steps=train_iters,
            warmup_steps=lr_warmup_steps,
            warmup_ratio=lr_warmup_ratio,
            decay_style=self.lr_decay_style,
            lr=initial_lrs.pop(),
            lr_min=min_lr,
            lr_start=lr_start,
            lr_decay_ratio=lr_decay_ratio,
        )

    def get_lr_scheduler(self) -> Any:
        """Return the scheduler container used by the trainer."""
        return self.lr_scheduler
