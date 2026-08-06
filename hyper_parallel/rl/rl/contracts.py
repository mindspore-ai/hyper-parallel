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
"""Stable token-first contracts shared across the Hyper-RL runtime."""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class Message:
    """One conversational message before tokenization."""

    role: Literal["system", "user", "assistant", "tool", "environment"]
    content: str


@dataclass(frozen=True)
class PromptRecord:
    """Stable input record understood by rollout and environments."""

    prompt_id: str
    messages: tuple[Message, ...]
    ground_truth: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Turn:
    """One token-aligned observation or action in a trajectory."""

    role: Literal["system", "user", "assistant", "tool", "environment"]
    content: str
    token_start: int
    token_end: int
    trainable: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trajectory:
    """Canonical output of both one-shot and multi-turn rollout."""

    trajectory_id: str
    prompt_id: str
    group_id: Optional[str]
    policy_version: int
    turns: tuple[Turn, ...]
    token_ids: Any
    attention_mask: Any
    action_mask: Any
    rollout_log_probs: Optional[Any]
    reward: float
    reward_components: dict[str, float]
    done: bool
    truncated: bool
    terminal_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fail early when token-aligned fields drift apart."""
        token_count = int(self.token_ids.numel())
        if int(self.attention_mask.numel()) != token_count:
            raise ValueError("trajectory attention_mask must align with token_ids")
        if int(self.action_mask.numel()) != token_count:
            raise ValueError("trajectory action_mask must align with token_ids")
        if self.rollout_log_probs is not None and int(self.rollout_log_probs.numel()) != token_count - 1:
            raise ValueError(
                "trajectory rollout_log_probs must align with next-token positions"
            )


@dataclass(frozen=True)
class ExperienceBatch:
    """Padded tensor batch consumed by an algorithm and the actor worker.

    ``action_mask`` spans the complete sequence. Optional algorithm-specific
    fields are populated only when their requirements request them.
    """

    trajectories: tuple[Trajectory, ...]
    sequences: Any
    attention_mask: Any
    action_mask: Any
    rewards: Any
    old_log_probs: Optional[Any]
    responses: tuple[str, ...]
    generation_seconds: float
    advantages: Optional[Any] = None
    returns: Optional[Any] = None
    values: Optional[Any] = None
    reference_log_probs: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the shared token dimensions before algorithm code runs."""
        if self.sequences.ndim != 2:
            raise ValueError("experience sequences must be rank two")
        if tuple(self.attention_mask.shape) != tuple(self.sequences.shape):
            raise ValueError("experience attention_mask must align with sequences")
        if tuple(self.action_mask.shape) != tuple(self.sequences.shape):
            raise ValueError("experience action_mask must align with sequences")
        if tuple(self.rewards.shape) != (self.sequences.shape[0],):
            raise ValueError("experience rewards must contain one value per sequence")
        log_prob_shape = (self.sequences.shape[0], self.sequences.shape[1] - 1)
        if self.old_log_probs is not None and tuple(self.old_log_probs.shape) != log_prob_shape:
            raise ValueError(
                "experience old_log_probs must align with next-token positions"
            )
        for field_name in ("advantages", "returns", "values", "reference_log_probs"):
            value = getattr(self, field_name)
            if value is not None and tuple(value.shape) != log_prob_shape:
                raise ValueError(
                    f"experience {field_name} must align with next-token positions"
                )
        if bool(self.action_mask[:, 0].any().item()):
            raise ValueError("the first sequence token cannot be a trainable action")

    @property
    def loss_action_mask(self) -> Any:
        """Align full-sequence action positions with next-token log-probabilities."""
        return self.action_mask[:, 1:]
