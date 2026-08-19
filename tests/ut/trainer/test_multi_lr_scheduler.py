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
"""Tests for scheduling both children of a chained optimizer."""

import pytest
import torch
from torch import nn

from hyper_models.components.optim import MultiLRScheduler
from hyper_parallel.core.optimizer import ChainedOptimizer


class _TwoChildModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adamw_parameter = nn.Parameter(torch.ones(1))
        self.muon_parameter = nn.Parameter(torch.ones(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Scale inputs by both optimizer-owned parameters."""
        return inputs * self.adamw_parameter * self.muon_parameter


def _build_runtime(style: str, min_lr: float = 0.0):
    """Build a two-child optimizer and its scheduler container."""
    model = _TwoChildModel()
    optimizer = ChainedOptimizer(
        model,
        optimizers={
            "adamw": torch.optim.AdamW(
                [model.adamw_parameter], lr=1e-4, foreach=False, fused=False
            ),
            "muon": torch.optim.AdamW(
                [model.muon_parameter], lr=1e-3, foreach=False, fused=False
            ),
        },
        flatten=True,
    )
    scheduler = MultiLRScheduler(
        optimizer=optimizer,
        lr_decay_style=style,
        train_steps=4,
        lr_config={"lr_warmup_steps": 0, "min_lr": min_lr},
    ).get_lr_scheduler()
    return optimizer, scheduler


def _step(optimizer: ChainedOptimizer, scheduler) -> None:
    for parameter in optimizer.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()


@pytest.mark.parametrize("style", ["linear", "cosine"])
def test_scheduler_uses_each_childs_own_initial_lr(style: str) -> None:
    """
    Feature: Learning-rate scheduling for chained optimizers.
    Description: Advance AdamW and Muon child schedulers with distinct peak learning rates.
    Expectation: Both rates decay while preserving their initial ten-to-one ratio.
    """
    optimizer, scheduler = _build_runtime(style)

    _step(optimizer, scheduler)

    adamw_lr = optimizer.optimizers_dict["adamw"].param_groups[0]["lr"]
    muon_lr = optimizer.optimizers_dict["muon"].param_groups[0]["lr"]
    assert adamw_lr > 0.0
    assert muon_lr > 0.0
    assert muon_lr / adamw_lr == pytest.approx(10.0)


def test_scalar_min_lr_is_the_final_lr_of_each_child() -> None:
    """
    Feature: Shared minimum learning rate for chained optimizers.
    Description: Advance both child schedulers through the complete cosine schedule.
    Expectation: Each child ends at the configured absolute minimum learning rate.
    """
    optimizer, scheduler = _build_runtime("cosine", min_lr=1e-6)

    for _ in range(4):
        _step(optimizer, scheduler)

    assert optimizer.optimizers_dict["adamw"].param_groups[0]["lr"] == pytest.approx(1e-6)
    assert optimizer.optimizers_dict["muon"].param_groups[0]["lr"] == pytest.approx(1e-6)
