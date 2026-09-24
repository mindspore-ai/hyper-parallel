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
"""Canonical trajectory batching and model-free target preparation."""
from dataclasses import replace
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist

from rl.algorithm.loss import RLAlgorithm
from rl.dataset.contracts import ExperienceBatch, Trajectory
from rl.roles.rollout.base import GenerationSettings


def _detached(value: Optional[Any]) -> Optional[Any]:
    """Detach an optional algorithm tensor."""
    return None if value is None else value.detach()


def build_experience_batch(
    trajectories: tuple[Trajectory, ...],
    generation_seconds: float,
    settings: GenerationSettings,
    metadata: Mapping[str, Any],
) -> ExperienceBatch:
    """Pad canonical trajectories into the shared rollout batch contract."""
    if not trajectories:
        raise ValueError("At least one trajectory is required")
    sequences, attention_mask, action_mask, old_log_probs = _allocate_experience_tensors(trajectories, settings)
    for row, trajectory in enumerate(trajectories):
        length = int(trajectory.token_ids.numel())
        sequences[row, :length] = trajectory.token_ids
        attention_mask[row, :length] = trajectory.attention_mask
        action_mask[row, :length] = trajectory.action_mask
        if old_log_probs is not None:
            old_log_probs[row, : length - 1] = trajectory.rollout_log_probs
    rewards = torch.tensor(
        [trajectory.reward for trajectory in trajectories],
        dtype=torch.float32,
        device=sequences.device,
    )
    responses = tuple(
        "\n".join(turn.content for turn in trajectory.turns if turn.role == "assistant")
        for trajectory in trajectories
    )
    worker_versions = {trajectory.worker_policy_version for trajectory in trajectories}
    if len(worker_versions) != 1:
        raise ValueError(
            "Trajectories must carry one consistent worker policy version"
        )
    batch_metadata = dict(metadata)
    batch_metadata["generated_action_tokens"] = int(action_mask.flatten().sum(dim=0).item())
    return ExperienceBatch(
        trajectories=trajectories,
        sequences=sequences,
        attention_mask=attention_mask,
        action_mask=action_mask,
        rewards=rewards,
        old_log_probs=old_log_probs,
        responses=responses,
        generation_seconds=generation_seconds,
        worker_policy_version=worker_versions.pop(),
        metadata=batch_metadata,
    )


def _allocate_experience_tensors(
    trajectories: tuple[Trajectory, ...], settings: GenerationSettings,
) -> tuple[Any, Any, Any, Optional[Any]]:
    """Allocate padded token, mask and optional log-probability tensors."""
    max_length = max(int(trajectory.token_ids.numel()) for trajectory in trajectories)
    first = trajectories[0].token_ids
    sequences = first.new_full(
        (len(trajectories), max_length), settings.pad_token_id
    )
    attention_mask = first.new_zeros(
        (len(trajectories), max_length), dtype=torch.bool
    )
    action_mask = attention_mask.clone()
    any_log_probs = any(
        trajectory.rollout_log_probs is not None for trajectory in trajectories
    )
    collect_log_probs = settings.collect_log_probs or any_log_probs
    if collect_log_probs and not all(
        trajectory.rollout_log_probs is not None for trajectory in trajectories
    ):
        raise ValueError("Trajectories must consistently provide rollout log-probabilities")
    old_log_probs = None
    if collect_log_probs:
        old_log_probs = torch.zeros(
            (len(trajectories), max_length - 1),
            dtype=torch.float32,
            device=first.device,
        )
    return sequences, attention_mask, action_mask, old_log_probs


class ExperiencePreparer:
    """Combine completed role outputs into an immutable training batch."""

    def __init__(self, algorithm: RLAlgorithm, dp_group_info: Optional[Any] = None) -> None:
        """Initialize target construction for one algorithm recipe."""
        self.algorithm = algorithm
        self.dp_group_info = dp_group_info

    def prepare(
        self,
        rollout: ExperienceBatch,
        *,
        reference_log_probs: Optional[Any] = None,
        values: Optional[Any] = None,
        bootstrap_values: Optional[Any] = None,
    ) -> ExperienceBatch:
        """Validate role outputs and build algorithm-specific training targets."""
        requirements = self.algorithm.requirements.data
        required_inputs = (
            (requirements.rollout_log_probs, rollout.old_log_probs, "rollout log-probabilities"),
            (requirements.reference_log_probs, reference_log_probs, "reference log-probabilities"),
            (requirements.values, values, "critic values"),
        )
        for required, value, label in required_inputs:
            if required and value is None:
                raise ValueError(f"Algorithm '{self.algorithm.name}' requires {label}")
        detached_values = _detached(values)
        group_ids = (
            tuple(trajectory.group_id for trajectory in rollout.trajectories)
            if rollout.trajectories
            else None
        )
        targets = self.algorithm.build_targets(
            rewards=rollout.rewards,
            action_mask=rollout.loss_action_mask,
            group_ids=group_ids,
            values=detached_values,
            **({"bootstrap_values": bootstrap_values} if bootstrap_values is not None else {}),
        )
        if requirements.returns and targets.returns is None:
            raise RuntimeError(
                f"Algorithm '{self.algorithm.name}' declared returns but did not build them"
            )
        advantages = targets.advantages.detach()
        algorithm_config = getattr(self.algorithm, "config", None)
        if requirements.values and getattr(algorithm_config, "normalize_advantages", False):
            advantages = normalize_advantages(
                advantages, rollout.loss_action_mask, self.dp_group_info,
                epsilon=algorithm_config.advantage_epsilon,
            )
        return replace(
            rollout,
            reference_log_probs=_detached(reference_log_probs),
            values=detached_values,
            bootstrap_values=_detached(bootstrap_values),
            advantages=advantages,
            returns=_detached(targets.returns),
        )


def normalize_advantages(advantages: Any, mask: Any, group_info: Optional[Any] = None,
                         *, epsilon: float = 1.0e-6) -> Any:
    """Whiten using all valid action tokens in the DP batch, excluding TP duplicates."""
    selected = advantages.float().masked_select(mask.bool())
    statistics = torch.cat((
        selected.sum().reshape(1), selected.square().sum().reshape(1),
        selected.new_tensor([float(selected.numel())]),
    ))
    if group_info is not None and group_info.rank_size > 1:
        dist.all_reduce(statistics, group=group_info.group)
    total, square_sum, count = statistics.unbind()
    if count.item() == 0:
        raise ValueError("PPO batch has no valid action tokens")
    mean = total / count
    variance = (square_sum / count - mean.square()).clamp(min=0)
    return ((advantages - mean) / (variance.sqrt() + epsilon)) * mask


def get_bootstrap_values(rollout: ExperienceBatch, full_values: Any) -> Any:
    """Use terminal zero values; require complete context for nonterminal truncations."""
    result = full_values.new_zeros(full_values.shape[0])
    for row, trajectory in enumerate(rollout.trajectories):
        if trajectory.done:
            continue
        if not trajectory.truncated or not trajectory.metadata.get("bootstrap_context_complete", False):
            raise ValueError("PPO requires a terminal trajectory or an explicitly complete bootstrap context")
        length = int(rollout.attention_mask[row].sum().item())
        result[row] = full_values[row, length - 1]
    return result.detach()
