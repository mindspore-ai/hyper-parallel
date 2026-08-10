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
"""Canonical trajectory batching and requirements-driven target preparation."""

from dataclasses import replace
from typing import Any, Mapping, Optional, Protocol

from rl.algorithm.base import RLAlgorithm
from rl.contracts import ExperienceBatch, Trajectory
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
    batch_metadata = dict(metadata)
    batch_metadata["generated_action_tokens"] = int(action_mask.sum().item())
    return ExperienceBatch(
        trajectories=trajectories,
        sequences=sequences,
        attention_mask=attention_mask,
        action_mask=action_mask,
        rewards=rewards,
        old_log_probs=old_log_probs,
        responses=responses,
        generation_seconds=generation_seconds,
        metadata=batch_metadata,
    )


class LogProbabilityModel(Protocol):
    """Minimal frozen-policy capability required by experience preparation."""

    training: bool

    def eval(self) -> Any:
        """Switch the model to evaluation mode."""

    def train(self, mode: bool = True) -> Any:
        """Set the model training mode and return the model."""

    def sequence_log_probs(self, sequences: Any, attention_mask: Any) -> Any:
        """Return next-token log-probabilities for each sequence."""


class ValueModel(Protocol):
    """Minimal Critic capability required by experience preparation."""

    training: bool

    def eval(self) -> Any:
        """Switch the model to evaluation mode."""

    def train(self, mode: bool = True) -> Any:
        """Set the model training mode and return the model."""

    def sequence_values(self, sequences: Any, attention_mask: Any) -> Any:
        """Return next-token values for each sequence."""


class ExperienceBuilder:
    """Populate only the data fields declared by a complete algorithm Recipe."""

    def __init__(
        self,
        algorithm: RLAlgorithm,
        reference: Optional[LogProbabilityModel] = None,
        critic: Optional[ValueModel] = None,
        micro_batch_size: int = 1,
    ) -> None:
        """Initialize requirements-driven frozen-model inference."""
        if micro_batch_size <= 0:
            raise ValueError("ExperienceBuilder micro_batch_size must be positive")
        if algorithm.requirements.roles.reference and reference is None:
            raise ValueError(f"Algorithm '{algorithm.name}' requires a reference model")
        if algorithm.requirements.roles.critic and critic is None:
            raise ValueError(f"Algorithm '{algorithm.name}' requires a Critic")
        self.algorithm = algorithm
        self.reference = reference
        self.critic = critic
        self.micro_batch_size = micro_batch_size

    def _chunked_inference(self, model: Any, method_name: str, experience: ExperienceBatch) -> Any:
        """Run frozen model inference in response-sized micro-batches."""
        was_training = model.training
        model.eval()
        chunks = []
        try:
            with platform.no_grad():
                method = getattr(model, method_name)
                for start in range(0, experience.sequences.shape[0], self.micro_batch_size):
                    end = min(start + self.micro_batch_size, experience.sequences.shape[0])
                    chunks.append(
                        method(
                            experience.sequences[start:end],
                            experience.attention_mask[start:end],
                        )
                    )
        finally:
            model.train(was_training)
        return platform.cat(chunks, dim=0).detach()

    def build(self, rollout: ExperienceBatch) -> ExperienceBatch:
        """Return an immutable batch with Recipe-required targets populated."""
        requirements = self.algorithm.requirements
        if requirements.data.rollout_log_probs and rollout.old_log_probs is None:
            raise ValueError(
                f"Algorithm '{self.algorithm.name}' requires rollout log-probabilities"
            )
        reference_log_probs = rollout.reference_log_probs
        if requirements.data.reference_log_probs:
            if self.reference is None:
                raise RuntimeError(
                    f"Algorithm '{self.algorithm.name}' requires reference log-probabilities"
                )
            reference_log_probs = self._chunked_inference(
                self.reference, "sequence_log_probs", rollout
            )
        values = rollout.values
        if requirements.data.values:
            if self.critic is None:
                raise RuntimeError(f"Algorithm '{self.algorithm.name}' requires values")
            values = self._chunked_inference(self.critic, "sequence_values", rollout)
        group_ids = (
            tuple(trajectory.group_id for trajectory in rollout.trajectories)
            if rollout.trajectories
            else None
        )
        targets = self.algorithm.build_targets(
            rewards=rollout.rewards,
            action_mask=rollout.loss_action_mask,
            group_ids=group_ids,
            values=values,
        )
        if requirements.data.returns and targets.returns is None:
            raise RuntimeError(
                f"Algorithm '{self.algorithm.name}' declared returns but did not build them"
            )
        return replace(
            rollout,
            reference_log_probs=reference_log_probs,
            values=values,
            advantages=targets.advantages.detach(),
            returns=None if targets.returns is None else targets.returns.detach(),
        )
