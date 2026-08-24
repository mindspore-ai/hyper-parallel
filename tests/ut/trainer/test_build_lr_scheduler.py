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
"""Unit tests for the multi-optimizer learning-rate scheduler target."""

import unittest

import torch
from torch import nn

from hyper_parallel.auto_models.components.optim import MultiLRScheduler
from hyper_parallel.core.optimizer import ChainedOptimizer


class _Model(nn.Module):
    """Small model with one parameter for each optimizer child."""

    def __init__(self) -> None:
        super().__init__()
        self.adamw_parameter = nn.Parameter(torch.ones(1))
        self.muon_parameter = nn.Parameter(torch.ones(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return inputs unchanged; the test only needs registered parameters."""
        return inputs


def _build_scheduler(style: str = "cosine"):
    """Build a two-child optimizer and the configured scheduler container."""
    model = _Model()
    optimizer = ChainedOptimizer(
        model,
        optimizers={
            "adamw": torch.optim.SGD([model.adamw_parameter], lr=1e-4),
            "muon": torch.optim.SGD([model.muon_parameter], lr=1e-3),
        },
        flatten=True,
    )
    scheduler = MultiLRScheduler(
        optimizer=optimizer,
        lr_decay_style=style,
        train_steps=4,
        lr_config={"lr_warmup_steps": 1, "min_lr": 0.0},
    ).get_lr_scheduler()
    return optimizer, scheduler


class TestMultiLRSchedulerTarget(unittest.TestCase):
    """The YAML class target builds one scheduler per chained child."""

    def test_builds_scheduler_for_each_child(self):
        _, scheduler = _build_scheduler()

        scheduler_state = scheduler.state_dict()
        self.assertEqual(set(scheduler_state), {"adamw", "muon"})
        self.assertEqual(len(scheduler.get_last_lr()), 2)
        self.assertEqual(scheduler_state["adamw"]["base_lrs"], [1e-4])
        self.assertEqual(scheduler_state["muon"]["base_lrs"], [1e-3])

    def test_supports_constant_linear_and_cosine(self):
        for style in ("constant", "linear", "cosine"):
            _, scheduler = _build_scheduler(style)
            self.assertIsNotNone(scheduler)


if __name__ == "__main__":
    unittest.main()
