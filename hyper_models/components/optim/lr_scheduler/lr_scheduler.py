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

from hyper_parallel.core.optimizer.lr_scheduler import LRSchedulersContainer, get_constant_schedule_with_warmup, \
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


class MultiLRScheduler:
    """Build one learning-rate scheduler for each child optimizer."""

    def __init__(self, optimizer, lr_decay_style, train_steps, lr_config):
        self.config = lr_config
        self.optimizer = optimizer
        self.lr_decay_style = lr_decay_style
        self.train_step = train_steps

        lr_warmup_ratio = self.config.get('lr_warmup_ratio', 0.0)
        lr_warmup_steps = self.config.get('lr_warmup_steps')
        if lr_warmup_steps is None:
            lr_warmup_steps = int(train_steps * lr_warmup_ratio)
        lr_start = self.config.get('lr_start', 0.0)
        lr_decay_ratio = self.config.get('lr_decay_ratio', 1.0)
        min_lr = self.config.get('min_lr', self.config.get('lr_min', 1e-7))

        def build_scheduler(optimizer):
            init_lr = optimizer.param_groups[0]["lr"]
            if self.lr_decay_style == "constant":
                return get_constant_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=lr_warmup_steps,
                    init_lr=init_lr,
                    lr_start=lr_start,
                )
            if self.lr_decay_style == "linear":
                return get_linear_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=lr_warmup_steps,
                    num_training_steps=train_steps,
                    init_lr=init_lr,
                    min_lr=min_lr,
                    lr_start=lr_start,
                )
            if self.lr_decay_style == "cosine":
                return get_cosine_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=lr_warmup_steps,
                    num_training_steps=train_steps,
                    init_lr=init_lr,
                    lr_decay_ratio=lr_decay_ratio,
                    min_lr=min_lr,
                    lr_start=lr_start,
                )
            raise ValueError(f"Unsupported lr_decay_style: {self.lr_decay_style!r}")

        self.lr_scheduler = LRSchedulersContainer(
            optimizers=self.optimizer,
            scheduler=build_scheduler,
        )

    def get_lr_scheduler(self):
        """Return the scheduler container used by the trainer."""
        return self.lr_scheduler
