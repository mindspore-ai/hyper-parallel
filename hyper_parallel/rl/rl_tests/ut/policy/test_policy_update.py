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
"""CPU unit tests for Actor and Critic update orchestration."""
# Local test doubles are not public APIs; the suite intentionally uses Torch CPU tensors.
# pylint: disable=forbidden-backend-import,missing-public-docstring

from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
import math
from types import MethodType, SimpleNamespace
from typing import Any

import pytest
import torch

import rl.roles.policy.actor as actor_module
import rl.roles.policy.critic as critic_module
from rl.algorithm import build_algorithm
from rl.algorithm.loss import CriticLossOutput
from rl.dataset.contracts import ExperienceBatch
from rl.roles.policy.actor import Actor
from rl.roles.policy.critic import Critic
from rl.utils.monitoring.metrics import ActorMicroBatchMetrics


def _actor_experience() -> ExperienceBatch:
    """Build four responses with four local action tokens."""
    sequences = torch.arange(16, dtype=torch.long).reshape(4, 4)
    action_mask = torch.tensor(
        [
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
            [False, True, False, False],
        ]
    )
    return ExperienceBatch(
        trajectories=(),
        sequences=sequences,
        attention_mask=torch.ones_like(sequences, dtype=torch.bool),
        action_mask=action_mask,
        rewards=torch.ones(4),
        old_log_probs=torch.zeros((4, 3)),
        responses=("0", "1", "2", "3"),
        generation_seconds=0.0,
        advantages=torch.ones((4, 3)),
        reference_log_probs=torch.zeros((4, 3)),
    )


def test_actor_update_follows_token_and_optimizer_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Actor update uses global action tokens and one synchronized step per mini-batch."""
    events: list[str] = []
    experience = _actor_experience()

    class TinyPolicy(torch.nn.Module):
        """Expose two trainable logits with a closed-form policy gradient."""

        def __init__(self) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(2))

        def forward(self, input_ids: torch.Tensor, **_kwargs: Any) -> dict[str, torch.Tensor]:
            return {"logits": self.logits.expand(*input_ids.shape, 2)}

    model = TinyPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=False, fused=False)
    direct_actor = Actor(
        model, build_algorithm({"name": "grpo", "kl_coef": 0.0, "loss_aggregation": "token-mean"}),
        micro_batch_size=2, optimizer=optimizer,
        device=torch.device("cpu"), response_mini_batch_size=4,
    )
    batch = replace(
        experience,
        sequences=torch.tensor([[0, 1, 1, 1]] * 4),
        old_log_probs=torch.full((4, 3), -math.log(2)),
        reference_log_probs=torch.full((4, 3), -math.log(2)),
    )
    # Each action contributes [0.5, -0.5]; the global mean spans four actions.
    direct_metrics = direct_actor.forward_backward(batch, 0, 2, global_tokens=4)
    torch.testing.assert_close(model.logits.grad, torch.tensor([0.25, -0.25]))
    torch.testing.assert_close(direct_metrics.total_loss_sum, torch.tensor(-2.0))
    optimizer.zero_grad(set_to_none=True)
    gradients = []
    hook = optimizer.register_step_pre_hook(
        lambda *_args: gradients.append(model.logits.grad.detach().clone())
    )
    with monkeypatch.context() as cpu:
        cpu.setattr(actor_module, "hsdp_sync_stream", lambda: None)
        cpu.setattr(actor_module, "SkipDTensorDispatch", nullcontext)
        cpu.setattr(actor_module, "clip_grad_norm_", torch.nn.utils.clip_grad_norm_)
        updated = direct_actor.update(batch)
    hook.remove()
    assert len(gradients) == 1, f"Expected one optimizer step, got {len(gradients)}"
    torch.testing.assert_close(gradients[0], torch.tensor([0.5, -0.5]))
    torch.testing.assert_close(model.state_dict()["logits"], torch.tensor([-0.05, 0.05]))
    assert model.logits.grad is None, "Actor update must clear gradients after optimizer.step"
    assert updated.optimizer_steps == 1 and updated.valid_tokens == 4
    assert updated.total_loss == pytest.approx(-1.0)
    assert updated.gradient_norm == pytest.approx(math.sqrt(0.5))

    class RecordingOptimizer:
        """Record zero and step calls while exposing a standard param group."""

        param_groups = [{"lr": 0.01}]

        @staticmethod
        def zero_grad(*, set_to_none: bool) -> None:
            assert set_to_none
            events.append("zero")

        @staticmethod
        def step() -> None:
            events.append("step")

    class RecordingContext(AbstractContextManager[None]):
        """Record entry around the optimizer's local dispatch path."""

        def __enter__(self) -> None:
            events.append("dispatch-enter")

        def __exit__(self, *exc_info: Any) -> None:
            del exc_info
            events.append("dispatch-exit")

    scheduler_steps: list[str] = []
    actor = Actor(
        torch.nn.Linear(2, 2),
        build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"}),
        micro_batch_size=2,
        optimizer=RecordingOptimizer(),
        lr_scheduler=SimpleNamespace(step=lambda: scheduler_steps.append("scheduler")),
        device=torch.device("cpu"),
        dp_group_info="dp-group",
        dp_size=2,
        response_mini_batch_size=4,
        update_epochs=2,
    )
    global_token_inputs: list[int] = []

    def all_reduce(tensor: torch.Tensor, group: Any) -> None:
        assert group == "dp-group"
        if tensor.numel() == 1:
            global_token_inputs.append(int(tensor.item()))
        tensor.mul_(2)

    slices: list[tuple[int, int, int]] = []

    def forward_backward(
        self: Actor,
        experience: ExperienceBatch,
        start: int,
        end: int,
        *,
        global_tokens: int,
    ) -> ActorMicroBatchMetrics:
        del self, experience
        slices.append((start, end, global_tokens))
        values = [torch.tensor(1.0) for _ in range(6)]
        return ActorMicroBatchMetrics(*values)

    sync_flags: list[bool] = []

    def record_sync(is_last: bool) -> None:
        """Record whether one micro-batch enables gradient synchronization."""
        sync_flags.append(is_last)

    actor.forward_backward = MethodType(forward_backward, actor)
    actor._set_gradient_sync = record_sync  # pylint: disable=protected-access
    monkeypatch.setattr(actor_module.platform, "all_reduce", all_reduce)
    monkeypatch.setattr(actor_module, "hsdp_sync_stream", lambda: events.append("stream-sync"))
    monkeypatch.setattr(
        actor_module,
        "clip_grad_norm_",
        lambda *_args, **_kwargs: events.append("clip") or torch.tensor(2.0),
    )
    monkeypatch.setattr(actor_module, "SkipDTensorDispatch", RecordingContext)

    metrics = actor.update(experience)

    assert slices == [(0, 2, 8), (2, 4, 8), (0, 2, 8), (2, 4, 8)]
    assert sync_flags == [False, True, False, True]
    assert global_token_inputs == [4, 4]
    assert events == [
        "zero",
        "stream-sync",
        "clip",
        "dispatch-enter",
        "step",
        "dispatch-exit",
        "zero",
        "zero",
        "stream-sync",
        "clip",
        "dispatch-enter",
        "step",
        "dispatch-exit",
        "zero",
    ]
    assert scheduler_steps == ["scheduler"]
    assert metrics.valid_tokens == 16
    assert metrics.optimizer_steps == 2
    assert metrics.gradient_norm == 2.0


def test_critic_values_and_update_follow_policy_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Critic aligns next-token values and updates once over two micro-batches."""

    class ValueModel(torch.nn.Module):
        """Expose one trainable scalar in token-aligned value predictions."""

        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            use_cache: bool,
        ) -> dict[str, torch.Tensor]:
            del attention_mask
            assert not use_cache
            return {"values": input_ids.float() + self.bias}

    class FakePPO:
        """Record the value tensors supplied by Critic without re-testing PPO math."""

        name = "ppo"
        requirements = SimpleNamespace(roles=SimpleNamespace(critic=True))

        def __init__(self) -> None:
            self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        def compute_critic_loss(
            self,
            current_values: torch.Tensor,
            old_values: torch.Tensor,
            returns: torch.Tensor,
            action_mask: torch.Tensor,
        ) -> CriticLossOutput:
            self.calls.append((old_values.clone(), returns.clone(), action_mask.clone()))
            loss_sum = ((current_values - returns).square() * action_mask).sum()
            return CriticLossOutput(loss_sum, action_mask.sum().detach())

    class CPUOptimizer:
        """Apply one scalar CPU update without torch-npu optimizer dispatch."""

        def __init__(self, parameter: torch.nn.Parameter) -> None:
            self.parameter = parameter
            self.param_groups = [{"lr": 0.01}]

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none
            self.parameter.grad = None

        def step(self) -> None:
            assert self.parameter.grad is not None
            with torch.no_grad():
                self.parameter.add_(self.parameter.grad, alpha=-0.01)

    model = ValueModel()
    algorithm = FakePPO()
    optimizer = CPUOptimizer(model.bias)
    scheduler_steps: list[str] = []
    critic = Critic(
        model,
        algorithm,
        optimizer,
        SimpleNamespace(step=lambda: scheduler_steps.append("scheduler")),
        torch.device("cpu"),
        None,
        1,
        1,
        2,
        1,
        1.0,
    )
    sequences = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    action_mask = torch.tensor([[False, True, True, False], [False, True, False, True]])
    experience = ExperienceBatch(
        trajectories=(),
        sequences=sequences,
        attention_mask=torch.ones_like(sequences, dtype=torch.bool),
        action_mask=action_mask,
        rewards=torch.tensor([1.0, 1.0]),
        old_log_probs=None,
        responses=("a", "b"),
        generation_seconds=0.0,
        values=torch.zeros((2, 3)),
        returns=torch.ones((2, 3)),
    )
    monkeypatch.setattr(critic_module, "hsdp_sync_stream", lambda: None)
    monkeypatch.setattr(critic_module, "SkipDTensorDispatch", nullcontext)
    monkeypatch.setattr(critic_module, "clip_grad_norm_", lambda *_args, **_kwargs: torch.tensor(1.0))

    aligned = critic.sequence_values(sequences, experience.attention_mask)
    original_bias = model.state_dict()["bias"].clone()
    metrics = critic.update(experience)

    torch.testing.assert_close(aligned, sequences[:, :-1].float())
    assert len(algorithm.calls) == 2
    torch.testing.assert_close(algorithm.calls[0][0], torch.zeros((1, 3)))
    torch.testing.assert_close(algorithm.calls[0][1], torch.ones((1, 3)))
    assert torch.equal(algorithm.calls[0][2], experience.loss_action_mask[:1])
    assert metrics.optimizer_steps == 1
    assert metrics.valid_tokens == 4
    assert scheduler_steps == ["scheduler"]
    assert not torch.equal(model.state_dict()["bias"], original_bias)
