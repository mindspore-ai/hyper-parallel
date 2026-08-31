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
"""Stable, business-neutral contracts shared by Agentic RL layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional


class InteractionMode(str, Enum):
    """Supported episode scheduling modes."""

    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"

    @classmethod
    def parse(cls, value: "InteractionMode | str") -> "InteractionMode":
        """Normalize a configured interaction mode."""
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "interaction_mode must be 'single_turn' or 'multi_turn'"
            ) from error


class TerminationReason(str, Enum):
    """Business-neutral reasons for ending an episode."""

    RUNNING = "running"
    COMPLETED = "completed"
    MAX_TURNS = "max_turns"
    ENVIRONMENT_TRUNCATED = "environment_truncated"
    CONTEXT_LIMIT = "context_limit"


@dataclass(frozen=True)
class Observation:
    """One non-trainable environment observation for the next model call."""

    content: str
    token_ids: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentAction:
    """One raw model response whose tokens participate in policy training."""

    content: str
    token_ids: Any
    rollout_log_probs: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)


Action = AgentAction


@dataclass(frozen=True)
class ToolCall:
    """One validated function request parsed from an agent action."""

    call_id: str
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate identifiers before dispatching user-owned code."""
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError("Tool call_id must be non-empty")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Tool name must be non-empty")
        if not isinstance(self.arguments, dict):
            raise ValueError("Tool arguments must be a mapping")


@dataclass(frozen=True)
class ToolResult:
    """Normalized result returned after one tool call."""

    call_id: str
    name: str
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardResult:
    """Episode or turn reward with named components and optional evidence."""

    value: float
    components: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize values and detach mutable caller mappings."""
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(
            self,
            "components",
            {str(name): float(value) for name, value in self.components.items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


ObservationEncoder = Callable[[str, str, dict[str, Any]], Observation]


@dataclass(frozen=True)
class EpisodeContext:
    """Stable configuration and services shared by one agent episode."""

    prompt: Any
    policy_version: int
    sample_index: int
    max_turns: int
    interaction_mode: InteractionMode | str | None = None
    settings: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    observation_encoder: Optional[ObservationEncoder] = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate stable episode identity, limits, and scheduling mode."""
        if self.policy_version < 0:
            raise ValueError("Episode policy_version must be non-negative")
        if self.sample_index < 0:
            raise ValueError("Episode sample_index must be non-negative")
        if self.max_turns <= 0:
            raise ValueError("Episode max_turns must be positive")
        mode = self.interaction_mode
        if mode is None:
            mode = (
                InteractionMode.SINGLE_TURN
                if self.max_turns == 1
                else InteractionMode.MULTI_TURN
            )
        mode = InteractionMode.parse(mode)
        if mode is InteractionMode.SINGLE_TURN and self.max_turns != 1:
            raise ValueError("single_turn interaction requires max_turns=1")
        object.__setattr__(self, "interaction_mode", mode)
        object.__setattr__(self, "settings", dict(self.settings))

    def encode_observation(
        self,
        content: str,
        *,
        role: str = "environment",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Observation:
        """Encode exact incremental environment text for the next turn."""
        if self.observation_encoder is None:
            raise RuntimeError("This episode does not provide an observation encoder")
        return self.observation_encoder(content, role, dict(metadata or {}))

    @property
    def prompt_id(self) -> str:
        """Expose the legacy PromptRecord identity during environment migration."""
        return self.prompt.prompt_id

    @property
    def messages(self) -> Any:
        """Expose legacy prompt messages during environment migration."""
        return self.prompt.messages

    @property
    def ground_truth(self) -> Any:
        """Expose the legacy task target during environment migration."""
        return self.prompt.ground_truth

    @property
    def metadata(self) -> dict[str, Any]:
        """Expose legacy prompt metadata during environment migration."""
        return self.prompt.metadata


@dataclass(frozen=True)
class TurnContext:
    """Read-only episode state exposed while applying one action."""

    episode: EpisodeContext
    turn_index: int
    cumulative_reward: float

    def __post_init__(self) -> None:
        """Validate that the action lies within the configured episode limit."""
        if self.turn_index < 0 or self.turn_index >= self.episode.max_turns:
            raise ValueError(
                "Turn index must lie within the episode limit: "
                f"turn_index={self.turn_index}, max_turns={self.episode.max_turns}"
            )

    @property
    def remaining_turns(self) -> int:
        """Return the number of model actions available after this turn."""
        return self.episode.max_turns - self.turn_index - 1

    @property
    def is_last_turn(self) -> bool:
        """Return whether the runner will truncate after the current action."""
        return self.remaining_turns == 0


@dataclass(frozen=True)
class TurnResult:
    """Environment result controlling reward and the next interaction step."""

    observation: Observation
    reward: float
    done: bool
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    termination_reason: Optional[TerminationReason | str] = None

    def __post_init__(self) -> None:
        """Normalize reward and optional termination reason."""
        object.__setattr__(self, "reward", float(self.reward))
        reason = self.termination_reason
        if reason is not None and not isinstance(reason, TerminationReason):
            try:
                reason = TerminationReason(reason)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Unsupported termination reason: {reason!r}") from error
        object.__setattr__(self, "termination_reason", reason)
        object.__setattr__(self, "info", dict(self.info))

    @property
    def reward_result(self) -> RewardResult:
        """Return the normalized structured reward for this transition."""
        components = self.info.get("reward_components", {"environment": self.reward})
        return RewardResult(self.reward, components, self.info)


Transition = TurnResult


@dataclass(frozen=True)
class EpisodeResult:
    """Business-neutral summary produced after one complete episode."""

    episode_id: str
    reward: RewardResult
    turn_results: tuple[TurnResult, ...]
    done: bool
    truncated: bool
    termination_reason: TerminationReason
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detach mutable episode metadata from runner-owned state."""
        object.__setattr__(self, "metadata", dict(self.metadata))
