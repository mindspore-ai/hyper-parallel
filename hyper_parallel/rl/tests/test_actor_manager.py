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
"""CPU tests for algorithm-neutral mini-batch updates."""

import torch

from rl.algorithm.grpo import GRPOAlgorithm, GRPOConfig
from rl.contracts import ExperienceBatch
from rl.dataset import ExperienceBuilder
from rl.roles import ActorManager, ActorModel


class _TinyCausalLM(torch.nn.Module):
    """Return one trainable categorical distribution at every token position."""

    def __init__(self) -> None:
        """Initialize one trainable vocabulary-logit vector."""
        super().__init__()
        self.token_logits = torch.nn.Parameter(torch.zeros(3))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> dict[str, torch.Tensor]:
        """Expand the trainable logits to the requested causal-LM shape."""
        del attention_mask, use_cache
        batch_size, sequence_length = input_ids.shape
        logits = self.token_logits.reshape(1, 1, -1).expand(batch_size, sequence_length, -1)
        return {"logits": logits}


class _CountingScheduler:
    """Count scheduler steps without changing the optimizer learning rate."""

    def __init__(self) -> None:
        """Initialize the scheduler step counter."""
        self.steps = 0

    def step(self) -> None:
        """Record one outer rollout scheduler step."""
        self.steps += 1


def _build_manager(response_mini_batch_size: int) -> tuple[ActorManager, ActorModel, _CountingScheduler]:
    """Build a deterministic CPU actor/reference pair for update tests."""
    actor = ActorModel(_TinyCausalLM())
    reference = ActorModel(_TinyCausalLM())
    reference.load_state_dict(actor.state_dict())
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.2, foreach=False, fused=False)
    scheduler = _CountingScheduler()
    manager = ActorManager(
        actor=actor,
        algorithm=GRPOAlgorithm(GRPOConfig(kl_coef=0.0)),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=torch.device("cpu"),
        dp_group_info=None,
        dp_size=1,
        micro_batch_size=1,
        response_mini_batch_size=response_mini_batch_size,
        policy_update_epochs=1,
        max_grad_norm=10.0,
    )
    manager.experience_builder = ExperienceBuilder(
        algorithm=manager.algorithm,
        reference=reference,
        micro_batch_size=1,
    )
    return manager, actor, scheduler


def _update_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one mixed-reward response group with distinct chosen tokens."""
    sequences = torch.tensor(
        [
            [2, 0],
            [2, 0],
            [2, 1],
            [2, 1],
        ],
        dtype=torch.long,
    )
    response_mask = torch.ones((4, 1), dtype=torch.bool)
    rewards = torch.tensor([0.0, 0.0, 1.0, 1.0])
    return sequences, response_mask, rewards


def _experience(
    sequences: torch.Tensor,
    response_mask: torch.Tensor,
    rewards: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> ExperienceBatch:
    action_mask = torch.cat((torch.zeros_like(response_mask), response_mask), dim=-1)
    return ExperienceBatch(
        trajectories=(),
        sequences=sequences,
        attention_mask=torch.ones_like(sequences, dtype=torch.bool),
        action_mask=action_mask,
        rewards=rewards,
        old_log_probs=old_log_probs,
        responses=("",) * sequences.shape[0],
        generation_seconds=0.0,
    )


def test_second_mini_batch_uses_updated_actor() -> None:
    """Verify an optimizer step separates current and frozen old log-probabilities."""
    manager, actor, scheduler = _build_manager(response_mini_batch_size=2)
    initial_logits = actor.module.token_logits.detach().clone()
    sequences, response_mask, rewards = _update_inputs()
    with torch.no_grad():
        old_log_probs = actor.sequence_log_probs(
            sequences,
            torch.ones_like(sequences),
        ).detach()

    rollout = _experience(sequences, response_mask, rewards, old_log_probs)
    metrics = manager.update(manager.experience_builder.build(rollout))

    assert metrics.optimizer_steps == 2, (
        f"Expected two mini-batch updates, got {metrics.optimizer_steps}"
    )
    assert metrics.old_current_log_ratio_abs > 0.0, (
        "The second mini-batch must observe current log-probabilities from the updated actor"
    )
    assert metrics.old_policy_kl > 0.0, "Current/old-policy KL must become positive after the first step"
    assert not torch.equal(actor.module.token_logits.detach(), initial_logits), (
        "The actor parameters must change after the mini-batch updates"
    )
    assert scheduler.steps == 1, f"The scheduler must advance once per rollout batch, got {scheduler.steps}"


def test_single_full_mini_batch_keeps_current_equal_to_old() -> None:
    """Verify one full-batch optimizer update has no pre-update current/old drift."""
    manager, actor, scheduler = _build_manager(response_mini_batch_size=4)
    sequences, response_mask, rewards = _update_inputs()
    with torch.no_grad():
        old_log_probs = actor.sequence_log_probs(
            sequences,
            torch.ones_like(sequences),
        ).detach()

    rollout = _experience(sequences, response_mask, rewards, old_log_probs)
    metrics = manager.update(manager.experience_builder.build(rollout))

    assert metrics.optimizer_steps == 1, (
        f"Expected one full-batch optimizer update, got {metrics.optimizer_steps}"
    )
    assert metrics.old_current_log_ratio_abs == 0.0, (
        f"Current and old must match before the only optimizer step, got {metrics.old_current_log_ratio_abs}"
    )
    assert metrics.old_policy_kl == 0.0, (
        f"Expected zero pre-update old-policy KL, got {metrics.old_policy_kl}"
    )
    assert scheduler.steps == 1, f"The scheduler must advance once per rollout batch, got {scheduler.steps}"
