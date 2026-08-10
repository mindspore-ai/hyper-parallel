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
"""CPU tests for model-backed values and the optional Critic worker."""

from types import SimpleNamespace
from typing import Any

import torch

from rl.algorithm import build_algorithm
from rl.contracts import ExperienceBatch
from rl.roles import CriticModel, attach_value_head
from rl.roles import CriticManager


class _TinyQwenLike(torch.nn.Module):
    """Minimal Qwen-shaped model exposing final normalized hidden states."""

    def __init__(self) -> None:
        """Initialize the tiny embedding, norm, and language-model head."""
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embed = torch.nn.Embedding(8, 4)
        self.model = torch.nn.Module()
        self.model.norm = torch.nn.LayerNorm(4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Any = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Return language-model logits for the input token IDs."""
        del attention_mask, kwargs
        hidden = self.model.norm(self.embed(input_ids))
        return {"logits": self.lm_head(hidden)}


def _experience(critic: CriticModel) -> ExperienceBatch:
    sequences = torch.tensor([[1, 2], [1, 3]])
    attention_mask = torch.ones_like(sequences, dtype=torch.bool)
    action_mask = torch.tensor([[False, True], [False, True]])
    with torch.no_grad():
        values = critic.sequence_values(sequences, attention_mask).detach()
    return ExperienceBatch(
        trajectories=(),
        sequences=sequences,
        attention_mask=attention_mask,
        action_mask=action_mask,
        rewards=torch.ones(2),
        old_log_probs=torch.zeros((2, 1)),
        responses=("", ""),
        generation_seconds=0.0,
        advantages=torch.ones((2, 1)),
        values=values,
        returns=torch.ones((2, 1)),
        reference_log_probs=torch.zeros((2, 1)),
    )


def test_qwen_value_capability_uses_hidden_states_and_scalar_head() -> None:
    module = attach_value_head(_TinyQwenLike(), "qwen3_5")
    critic = CriticModel(module)
    values = critic.sequence_values(
        torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.bool)
    )
    assert values.shape == (1, 2)
    values.sum().backward()
    assert module.value_head.weight.grad is not None
    assert module.embed.weight.grad is not None


def test_critic_manager_updates_only_when_recipe_requires_it() -> None:
    """Update the optional Critic for a Recipe that explicitly requires it."""
    critic = CriticModel(attach_value_head(_TinyQwenLike(), "qwen3_5"))
    algorithm = build_algorithm({"name": "ppo", "loss_aggregation": "token-mean"})
    optimizer = torch.optim.SGD(critic.parameters(), lr=0.1)
    before = critic.module.value_head.weight.detach().clone()
    manager = CriticManager(
        critic=critic,
        algorithm=algorithm,
        optimizer=optimizer,
        lr_scheduler=None,
        device=torch.device("cpu"),
        dp_group_info=None,
        dp_size=1,
        micro_batch_size=1,
        response_mini_batch_size=2,
        update_epochs=1,
        max_grad_norm=10.0,
    )
    metrics = manager.update(_experience(critic))
    assert metrics.optimizer_steps == 1
    assert metrics.value_loss >= 0.0
    assert not torch.equal(before, critic.module.value_head.weight.detach())
