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
"""Stateful token-first agent episode consumed by the batched runner."""

import inspect
from typing import Any, Optional

from rl.agentic.core.types import (
    Action,
    EpisodeResult,
    EpisodeContext,
    Observation,
    ObservationEncoder,
    RewardResult,
    TerminationReason,
    Transition,
    TurnContext,
)
from rl.agentic.envs.base import Environment
from rl.dataset.contracts import PromptRecord, Trajectory, Turn

from hyper_parallel import get_platform

platform = get_platform()

_TOOL_COUNTER_NAMES = ("tool_call_count", "tool_success_count", "tool_error_count")
_RESERVED_TRANSITION_INFO = frozenset(("reward_components", *_TOOL_COUNTER_NAMES))


class AgentSession:
    """Own one Environment and accumulate its token-exact trajectory."""

    def __init__(
        self,
        prompt: PromptRecord,
        environment: Environment,
        policy_version: int,
        sample_index: int,
        max_turns: int,
        max_observation_tokens: Optional[int] = None,
        max_episode_tokens: Optional[int] = None,
        observation_encoder: Optional[ObservationEncoder] = None,
        episode_context: Optional[EpisodeContext] = None,
    ) -> None:
        """Initialize state for one prompt, sample, and policy version."""
        if max_observation_tokens is not None and max_observation_tokens < 0:
            raise ValueError("max_observation_tokens must be non-negative")
        if max_episode_tokens is not None and max_episode_tokens <= 0:
            raise ValueError("max_episode_tokens must be positive when configured")
        if episode_context is None:
            episode_context = EpisodeContext(
                prompt=prompt,
                policy_version=policy_version,
                sample_index=sample_index,
                max_turns=max_turns,
                observation_encoder=observation_encoder,
            )
        elif (
            episode_context.prompt.prompt_id != prompt.prompt_id
            or episode_context.policy_version != policy_version
            or episode_context.sample_index != sample_index
            or episode_context.max_turns != max_turns
        ):
            raise ValueError(
                "AgentSession arguments must match the provided EpisodeContext"
            )
        self.prompt = episode_context.prompt
        self.environment = environment
        step_method = getattr(environment, "step", None)
        try:
            inspect.signature(step_method).bind(None, None)
        except (TypeError, ValueError):
            self._step_accepts_context = step_method is None
        else:
            self._step_accepts_context = True
        self.policy_version = policy_version
        self.sample_index = sample_index
        self.max_turns = max_turns
        self.max_observation_tokens = max_observation_tokens
        self.max_episode_tokens = max_episode_tokens
        self.episode_context = episode_context
        self._record_encoded_action = getattr(
            episode_context.observation_encoder,
            "record_action",
            None,
        )
        # These token-aligned lists form the eventual training trajectory:
        # observation masks are 0, assistant-action masks are 1, and rollout
        # log-probabilities occupy the same token positions as their actions.
        self.turns: list[Turn] = []
        self._token_parts: list[Any] = []
        self._token_count = 0
        self._cached_token_ids: Optional[Any] = None
        self._action_mask_parts: list[Any] = []
        self._log_prob_parts: list[Any] = []
        self._collecting_log_probs: Optional[bool] = None
        self._worker_policy_version: Optional[int] = None
        self._worker_policy_fingerprint: Optional[str] = None
        # Rewards and termination belong to the whole episode. Per-turn values
        # are retained for diagnostics and future credit-assignment strategies.
        self.reward = 0.0
        self.reward_components: dict[str, float] = {}
        self.turn_rewards: list[float] = []
        self.turn_results: list[Transition] = []
        self.turn_infos: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}
        self.done = False
        self.truncated = False
        self.terminal_reason = TerminationReason.RUNNING
        self.action_contents: list[str] = []
        self._started = False
        self._closed = False

    @property
    def active(self) -> bool:
        """Return whether the episode can accept another action."""
        return self._started and not self.done and not self.truncated

    @property
    def token_ids(self) -> Any:
        """Return all observation and action tokens accumulated so far."""
        if not self._token_parts:
            raise RuntimeError("AgentSession has not been started")
        if self._cached_token_ids is None:
            self._cached_token_ids = platform.cat(self._token_parts)
        return self._cached_token_ids

    @property
    def turn_count(self) -> int:
        """Return the number of model actions already applied."""
        return len(self.action_contents)

    @property
    def turn_context(self) -> TurnContext:
        """Build the environment context for the next model action."""
        if not self.active:
            raise RuntimeError("Cannot build a turn context for an inactive AgentSession")
        return TurnContext(
            episode=self.episode_context,
            turn_index=self.turn_count,
            cumulative_reward=self.reward,
        )

    @property
    def remaining_token_budget(self) -> Optional[int]:
        """Return remaining episode tokens, or ``None`` when unbounded."""
        if self.max_episode_tokens is None:
            return None
        return self.max_episode_tokens - self._token_count

    async def start(self) -> None:
        """Reset the environment and append its initial observation."""
        if self._started:
            raise RuntimeError("AgentSession.start() may be called only once")
        self._started = True
        observation = await self.environment.reset(self.episode_context)
        if observation.token_ids.numel() == 0:
            raise ValueError("The initial environment observation must not be empty")
        self._append_observation(observation, initial=True)

    def _prepare_observation_tokens(self, observation: Observation, initial: bool) -> Any:
        """Validate one observation and move it to the trajectory device."""
        token_ids = observation.token_ids
        if token_ids.ndim != 1:
            raise ValueError("Observation token_ids must be rank one")
        if not initial and self._token_parts and token_ids.device != self._token_parts[0].device:
            token_ids = token_ids.to(self._token_parts[0].device)
        if (
            not initial
            and self.max_observation_tokens is not None
            and token_ids.numel() > self.max_observation_tokens
        ):
            raise ValueError(
                "Environment observation exceeds agentic.max_observation_tokens: "
                f"tokens={token_ids.numel()}, limit={self.max_observation_tokens}"
            )
        return token_ids

    def _observation_span(
        self,
        token_ids: Any,
        initial: bool,
        terminal: bool,
    ) -> Optional[tuple[int, int]]:
        """Reserve a token span or apply the configured context-limit outcome."""
        start = self._token_count
        end = start + int(token_ids.numel())
        if self.max_episode_tokens is None or end <= self.max_episode_tokens:
            return start, end
        if initial:
            raise ValueError(
                "Initial observation exceeds agentic.max_episode_tokens: "
                f"tokens={end}, limit={self.max_episode_tokens}"
            )
        self.metadata["dropped_observation_tokens"] = int(token_ids.numel())
        if not terminal:
            self.truncated = True
            self.terminal_reason = TerminationReason.CONTEXT_LIMIT
        return None

    def _append_observation_turn(
        self,
        observation: Observation,
        token_ids: Any,
        start: int,
        end: int,
    ) -> None:
        """Append one validated observation to token-aligned trajectory state."""
        self._token_parts.append(token_ids)
        self._token_count = end
        self._cached_token_ids = None
        self._action_mask_parts.append(
            token_ids.new_zeros(token_ids.shape, dtype=platform.tensor_dtype.bool)
        )
        self._log_prob_parts.append(
            token_ids.new_zeros(token_ids.shape, dtype=platform.tensor_dtype.float32)
        )
        role = str(observation.metadata.get("role", "environment"))
        if role not in {"system", "user", "tool", "environment"}:
            raise ValueError(f"Unsupported observation role: {role}")
        turn_metadata = {
            str(name): value
            for name, value in observation.metadata.items()
            if name != "role"
        }
        self.turns.append(Turn(role, observation.content, start, end, False, metadata=turn_metadata))

    def _append_observation(
        self,
        observation: Observation,
        initial: bool = False,
        terminal: bool = False,
    ) -> bool:
        """Append one non-trainable observation when it fits the episode budget."""
        token_ids = self._prepare_observation_tokens(observation, initial)
        span = self._observation_span(token_ids, initial, terminal)
        if span is None:
            return False
        self._append_observation_turn(observation, token_ids, *span)
        return True

    def _append_action(self, action: Action) -> None:
        """Append one trainable model action and its optional log-probabilities."""
        if action.token_ids.ndim != 1:
            raise ValueError("Action token_ids must be a rank-one tensor")
        has_log_probs = action.rollout_log_probs is not None
        if self._collecting_log_probs is None:
            self._collecting_log_probs = has_log_probs
        elif self._collecting_log_probs != has_log_probs:
            raise ValueError(
                "Every action must consistently provide rollout_log_probs or omit them"
            )
        if has_log_probs and action.rollout_log_probs.numel() != action.token_ids.numel():
            raise ValueError("Action rollout_log_probs must align with action token_ids")
        if self._record_encoded_action is not None:
            self._record_encoded_action(action)
        start = self._token_count
        end = start + int(action.token_ids.numel())
        self._token_parts.append(action.token_ids)
        self._token_count = end
        self._cached_token_ids = None
        # Every assistant span is trainable, including actions from later turns.
        self._action_mask_parts.append(
            action.token_ids.new_ones(
                action.token_ids.shape,
                dtype=platform.tensor_dtype.bool,
            )
        )
        self._log_prob_parts.append(
            action.rollout_log_probs.float()
            if has_log_probs
            else action.token_ids.new_zeros(
                action.token_ids.shape,
                dtype=platform.tensor_dtype.float32,
            )
        )
        self.turns.append(
            Turn(
                "assistant",
                action.content,
                start,
                end,
                True,
                metadata=dict(action.metadata),
            )
        )
        self.action_contents.append(action.content)

    def record_worker_policy_identity(
        self,
        version: Optional[int],
        fingerprint: Optional[str],
    ) -> None:
        """Record one stable worker-owned identity across every generated turn."""
        if (version is None) != (fingerprint is None):
            raise ValueError("Worker policy version and fingerprint must be provided together")
        if version is None:
            return
        identity = (int(version), str(fingerprint))
        current = (self._worker_policy_version, self._worker_policy_fingerprint)
        if self._worker_policy_version is not None and identity != current:
            raise RuntimeError(
                "AgentSession observed multiple rollout policy identities: "
                f"current={current}, received={identity}"
            )
        self._worker_policy_version, self._worker_policy_fingerprint = identity

    def _record_transition(self, transition: Transition) -> None:
        """Accumulate reward and structured transition diagnostics."""
        turn_reward = float(transition.reward)
        self.reward += turn_reward
        self.turn_rewards.append(turn_reward)
        self.turn_infos.append(dict(transition.info))
        for counter_name in _TOOL_COUNTER_NAMES:
            self.metadata[counter_name] = int(self.metadata.get(counter_name, 0)) + int(
                transition.info.get(counter_name, 0)
            )
        components = transition.info.get("reward_components")
        if components is None:
            components = {"environment": turn_reward}
        for name, value in components.items():
            component_name = str(name)
            self.reward_components[component_name] = (
                self.reward_components.get(component_name, 0.0) + float(value)
            )
        self.metadata.update(
            {
                str(name): value
                for name, value in transition.info.items()
                if name not in _RESERVED_TRANSITION_INFO
            }
        )

    def _update_terminal_reason(self) -> None:
        """Set the stable terminal reason after applying a transition."""
        transition = self.turn_results[-1]
        if transition.termination_reason is not None:
            self.terminal_reason = transition.termination_reason
        elif self.done:
            self.terminal_reason = TerminationReason.COMPLETED
        elif self.truncated and self.terminal_reason is not TerminationReason.CONTEXT_LIMIT:
            self.terminal_reason = TerminationReason.ENVIRONMENT_TRUNCATED

    async def apply(self, action: Action) -> None:
        """Apply one action and accumulate its resulting transition."""
        if not self.active:
            raise RuntimeError("Cannot apply an action to an inactive AgentSession")
        if self.turn_count >= self.max_turns:
            raise RuntimeError("AgentSession cannot exceed its configured max_turns")
        context = self.turn_context
        self._append_action(action)
        # Environment.step is the round boundary: it turns the latest inference
        # output into reward, terminal state, and possibly the next model input.
        transition = (
            await self.environment.step(action, context)
            if self._step_accepts_context
            else await self.environment.step(action)
        )
        self.turn_results.append(transition)
        self._record_transition(transition)
        self.done = bool(transition.done)
        self.truncated = bool(transition.truncated)
        self._append_observation(
            transition.observation,
            terminal=self.done or self.truncated,
        )
        self._update_terminal_reason()

    def finish_max_turns(self) -> None:
        """Truncate an active session after the configured turn limit."""
        if self.active:
            self.truncated = True
            self.terminal_reason = TerminationReason.MAX_TURNS

    def finish_context_limit(self) -> None:
        """Truncate an active session before a generation that cannot fit."""
        if self.active:
            self.truncated = True
            self.terminal_reason = TerminationReason.CONTEXT_LIMIT

    async def close(self) -> None:
        """Close the environment at most once."""
        if not self._closed:
            await self.environment.close()
            self._closed = True

    def result(self) -> EpisodeResult:
        """Return the standard episode summary without training-plane tensors."""
        if not self._started:
            raise RuntimeError("Cannot build an EpisodeResult before reset")
        return EpisodeResult(
            episode_id=f"{self.prompt.prompt_id}:{self.policy_version}:{self.sample_index}",
            reward=RewardResult(
                self.reward,
                self.reward_components,
                {"turn_rewards": tuple(self.turn_rewards)},
            ),
            turn_results=tuple(self.turn_results),
            done=self.done,
            truncated=self.truncated,
            termination_reason=self.terminal_reason,
            metadata=dict(self.metadata),
        )

    def build(self) -> Trajectory:
        """Build the immutable token-aligned trajectory for this episode."""
        if not self._started:
            raise RuntimeError("Cannot build an AgentSession before reset")
        token_ids = self.token_ids
        action_mask = platform.cat(self._action_mask_parts)
        token_log_probs = platform.cat(self._log_prob_parts)
        # Log-probabilities predict token[t + 1], so remove the first token slot
        # while action_mask remains aligned with the unshifted full sequence.
        rollout_log_probs = (
            token_log_probs[1:] if self._collecting_log_probs else None
        )
        return Trajectory(
            trajectory_id=(
                f"{self.prompt.prompt_id}:{self.policy_version}:{self.sample_index}"
            ),
            prompt_id=self.prompt.prompt_id,
            group_id=self.prompt.prompt_id,
            policy_version=self.policy_version,
            turns=tuple(self.turns),
            token_ids=token_ids,
            attention_mask=token_ids.new_ones(
                token_ids.shape,
                dtype=platform.tensor_dtype.bool,
            ),
            action_mask=action_mask,
            rollout_log_probs=rollout_log_probs,
            reward=self.reward,
            reward_components=dict(self.reward_components),
            done=self.done,
            truncated=self.truncated,
            terminal_reason=self.terminal_reason.value,
            worker_policy_version=self._worker_policy_version,
            worker_policy_fingerprint=self._worker_policy_fingerprint,
            metadata={
                **self.metadata,
                "sample_index": self.sample_index,
                "num_actions": len(self.action_contents),
                "turn_rewards": tuple(self.turn_rewards),
                "turn_infos": tuple(dict(info) for info in self.turn_infos),
            },
        )
