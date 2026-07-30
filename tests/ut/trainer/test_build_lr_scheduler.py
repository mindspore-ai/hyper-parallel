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
"""Unit tests for the direct cosine-scheduler function target."""

import unittest

import torch
from torch import nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    SequentialLR,
)

from hyper_models.components.optim import cosine_with_warmup
from hyper_models.trainer.config import Target


SCHEDULER_TARGET = (
    "hyper_models.components.optim.lr_scheduler.lr_scheduler.cosine_with_warmup"
)


def _optimizer(lr: float = 1.0) -> torch.optim.Optimizer:
    return torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=lr)


class TestCosineWithWarmupTarget(unittest.TestCase):
    """The YAML function target returns one runtime scheduler."""

    def test_zero_warmup_returns_single_cosine_scheduler(self):
        scheduler = cosine_with_warmup(
            optimizer=_optimizer(),
            train_steps=10,
            lr_warmup_steps=0,
            min_lr=0.1,
        )

        self.assertIsInstance(scheduler, LRScheduler)
        self.assertIsInstance(scheduler, CosineAnnealingLR)
        self.assertNotIsInstance(scheduler, list)
        self.assertEqual(scheduler.T_max, 10)
        self.assertEqual(scheduler.eta_min, 0.1)

    def test_positive_warmup_returns_sequential_scheduler(self):
        scheduler = cosine_with_warmup(
            optimizer=_optimizer(),
            train_steps=10,
            lr_warmup_steps=2,
            init_lr=0.1,
            max_lr=1.0,
            min_lr=0.01,
        )

        self.assertIsInstance(scheduler, LRScheduler)
        self.assertIsInstance(scheduler, SequentialLR)
        self.assertNotIsInstance(scheduler, list)
        self.assertEqual(scheduler._milestones, [2])

    def test_target_build_accepts_trainer_owned_runtime_arguments(self):
        optimizer = _optimizer()
        target = Target(
            cosine_with_warmup,
            target_path=SCHEDULER_TARGET,
            lr_warmup_steps=0,
        )

        scheduler = target.build(optimizer=optimizer, train_steps=4)

        self.assertIsInstance(scheduler, CosineAnnealingLR)
        self.assertIs(scheduler.optimizer, optimizer)
        self.assertEqual(scheduler.T_max, 4)

    def test_invalid_step_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, r"train_steps must be at least 1"):
            cosine_with_warmup(optimizer=_optimizer(), train_steps=0)
        with self.assertRaisesRegex(
            ValueError,
            r"lr_warmup_steps must not be negative",
        ):
            cosine_with_warmup(
                optimizer=_optimizer(),
                train_steps=10,
                lr_warmup_steps=-1,
            )


if __name__ == "__main__":
    unittest.main()
