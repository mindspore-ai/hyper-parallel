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
from rl.algorithm.loss import RLAlgorithm
from rl.dataset.contracts import ExperienceBatch, Trajectory
from rl.roles.rollout.base import GenerationSettings
from hyper_parallel import get_platform
platform = get_platform()
def build_experience_batch(
    trajectories: tuple[Trajectory, ...],
    generation_seconds: float,
    settings: GenerationSettings,
    metadata: Mapping[str, Any],
) -> ExperienceBatch:
    """Pad canonical trajectories into the shared rollout batch contract."""
    if not trajectories:
        raise ValueError("At least one trajectory is required")
    max_length = max(int(trajectory.token_ids.numel()) for trajectory in trajectories)
    first = trajectories[0].token_ids
    sequences = first.new_full(
        (len(trajectories), max_length), settings.pad_token_id
    )
    attention_mask = first.new_zeros(
        (len(trajectories), max_length), dtype=platform.tensor_dtype.bool
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
        old_log_probs = platform.zeros(
            (len(trajectories), max_length - 1),
            dtype=platform.tensor_dtype.float32,
            device=first.device,
        )
    for row, trajectory in enumerate(trajectories):
        length = int(trajectory.token_ids.numel())
        sequences[row, :length] = trajectory.token_ids
        attention_mask[row, :length] = trajectory.attention_mask
        action_mask[row, :length] = trajectory.action_mask
        if old_log_probs is not None:
            old_log_probs[row, : length - 1] = trajectory.rollout_log_probs
    rewards = platform.tensor(
        [trajectory.reward for trajectory in trajectories],
        dtype=platform.tensor_dtype.float32,
        device=first.device,
    )
    responses = tuple(
        "\n".join(turn.content for turn in trajectory.turns if turn.role == "assistant")
        for trajectory in trajectories
    )
    worker_versions = {trajectory.worker_policy_version for trajectory in trajectories}
    worker_fingerprints = {
        trajectory.worker_policy_fingerprint for trajectory in trajectories
    }
    if len(worker_versions) != 1 or len(worker_fingerprints) != 1:
        raise ValueError(
            "Trajectories must carry one consistent worker policy version and fingerprint"
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
        worker_policy_fingerprint=worker_fingerprints.pop(),
        metadata=batch_metadata,
    )
class ExperiencePreparer:
    """Combine completed role outputs into an immutable training batch."""
    def __init__(self, algorithm: RLAlgorithm) -> None:
        """Initialize target construction for one algorithm recipe."""
        self.algorithm = algorithm
    def prepare(
        self,
        rollout: ExperienceBatch,
        *,
        reference_log_probs: Optional[Any] = None,
        values: Optional[Any] = None,
    ) -> ExperienceBatch:
        """Validate role outputs and build algorithm-specific training targets."""
        requirements = self.algorithm.requirements
        if requirements.data.rollout_log_probs and rollout.old_log_probs is None:
            raise ValueError(
                f"Algorithm '{self.algorithm.name}' requires rollout log-probabilities"
            )
        if requirements.data.reference_log_probs and reference_log_probs is None:
            raise ValueError(
                f"Algorithm '{self.algorithm.name}' requires reference log-probabilities"
            )
        if requirements.data.values and values is None:
            raise ValueError(
                f"Algorithm '{self.algorithm.name}' requires critic values"
            )
        detached_values = None if values is None else values.detach()
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
        )
        if requirements.data.returns and targets.returns is None:
            raise RuntimeError(
                f"Algorithm '{self.algorithm.name}' declared returns but did not build them"
            )
        return replace(
            rollout,
            reference_log_probs=(
                None if reference_log_probs is None else reference_log_probs.detach()
            ),
            values=detached_values,
            advantages=targets.advantages.detach(),
            returns=None if targets.returns is None else targets.returns.detach(),
        )
