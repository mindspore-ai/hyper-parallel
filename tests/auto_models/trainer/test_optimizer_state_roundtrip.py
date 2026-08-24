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
"""State round-trip tests for composite optimizers and LR schedulers."""

import torch
from torch import nn

from hyper_parallel.auto_models.components.optim import MultiLRScheduler
from hyper_parallel.core.optimizer import ChainedOptimizer


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        """Create a small model whose parameters can be split across two optimizers."""
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a scalar loss used to populate optimizer state."""
        return self.norm(self.linear(inputs)).sum()


def _build_runtime(model: _TinyModel):
    """Build a chained optimizer and matching scheduler container."""
    matrix_optimizer = torch.optim.AdamW([model.linear.weight], lr=1e-3, foreach=False)
    fallback_optimizer = torch.optim.AdamW(
        [model.linear.bias, model.norm.weight, model.norm.bias],
        lr=1e-4,
        foreach=False,
    )
    optimizer = ChainedOptimizer(
        model,
        optimizers={"muon": matrix_optimizer, "adamw": fallback_optimizer},
        flatten=True,
    )
    scheduler = MultiLRScheduler(
        optimizer=optimizer,
        lr_decay_style="cosine",
        train_steps=4,
        lr_config={"lr_warmup_steps": 1, "lr": 1e-3, "lr_min": 0.0},
    ).get_lr_scheduler()
    return optimizer, scheduler


def test_optimizer_and_scheduler_state_round_trip() -> None:
    """
    Feature: Optimizer and scheduler state restoration.
    Description: Advance both runtimes, save state, and load it into new instances.
    Expectation: Scheduler and optimizer states retain their saved values and keys.
    """
    model = _TinyModel()
    optimizer, scheduler = _build_runtime(model)

    model(torch.randn(2, 4)).backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()

    restored_model = _TinyModel()
    restored_optimizer, restored_scheduler = _build_runtime(restored_model)
    restored_optimizer.load_state_dict(optimizer_state)
    restored_scheduler.load_state_dict(scheduler_state)

    assert restored_scheduler.state_dict() == scheduler_state
    restored_optimizer_state = restored_optimizer.state_dict()
    assert optimizer_state.keys() <= restored_optimizer_state.keys()
    for key, expected in optimizer_state.items():
        actual = restored_optimizer_state[key]
        if isinstance(expected, torch.Tensor):
            assert torch.equal(actual, expected)
        else:
            assert actual == expected
