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
"""CPU unit tests for Policy forward-computation micro-batching."""
# Local test doubles are not public APIs; the suite intentionally uses Torch CPU tensors.
# pylint: disable=forbidden-backend-import,missing-public-docstring

from types import MethodType
from typing import Any

import pytest
import torch

from rl.algorithm import build_algorithm
from rl.dataset.contracts import ExperienceBatch
from rl.roles.policy.actor import Actor
from rl.roles.policy.critic import Critic


def _experience() -> ExperienceBatch:
    """Build five ordered rows so the final forward-computation micro-batch is partial."""
    sequences = torch.arange(20, dtype=torch.long).reshape(5, 4)
    return ExperienceBatch(
        trajectories=(),
        sequences=sequences,
        attention_mask=torch.ones_like(sequences, dtype=torch.bool),
        action_mask=torch.tensor([[False, True, True, True]] * 5),
        rewards=torch.arange(5, dtype=torch.float32),
        old_log_probs=torch.zeros((5, 3)),
        responses=("0", "1", "2", "3", "4"),
        generation_seconds=0.0,
    )


@pytest.mark.parametrize("role_name", ["actor", "critic"])
def test_policy_compute_micro_batches_preserve_order_and_mode(role_name: str) -> None:
    """Actor and Critic computation concatenates partial chunks and restores training mode."""
    experience = _experience()
    calls: list[list[int]] = []
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    if role_name == "actor":
        role: Any = Actor(
            model,
            build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"}),
            micro_batch_size=2,
            optimizer=optimizer,
            device=torch.device("cpu"),
        )

        def sequence_output(self: Actor, sequences: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            del self, attention_mask
            calls.append(sequences[:, 0].tolist())
            return sequences[:, 1:].float()

        role.sequence_log_probs = MethodType(sequence_output, role)
        invoke = role.compute_log_probs
        expected = experience.sequences[:, 1:].float()
    else:
        role = Critic(
            model,
            build_algorithm({"name": "ppo", "loss_aggregation": "token-mean"}),
            optimizer,
            None,
            torch.device("cpu"),
            None,
            1,
            2,
            5,
            1,
            1.0,
        )

        def sequence_output(self: Critic, sequences: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            del self, attention_mask
            calls.append(sequences[:, 0].tolist())
            return sequences[:, :-1].float()

        role.sequence_values = MethodType(sequence_output, role)
        invoke = role.compute_values
        expected = experience.sequences[:, :-1].float()

    role.train(True)
    actual = invoke(experience)

    torch.testing.assert_close(actual, expected)
    assert calls == [[0, 4], [8, 12], [16]]
    assert role.training
    assert not actual.requires_grad
