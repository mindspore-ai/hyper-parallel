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
from typing import Any, Optional
from rl.agentic.base import Action, Environment, Observation
from rl.dataset.contracts import PromptRecord, Trajectory, Turn
from hyper_parallel import get_platform
platform = get_platform()
class AgentSession:
    """Own one Environment and accumulate its token-exact trajectory."""
    def __init__(
        self,
        prompt: PromptRecord,
        environment: Environment,
        policy_version: int,
        sample_index: int,
        max_observation_tokens: Optional[int] = None,
    ) -> None:
        """Initialize state for one prompt, sample, and policy version."""
        if max_observation_tokens is not None and max_observation_tokens < 0:
            raise ValueError("max_observation_tokens must be non-negative")
        self.prompt = prompt
        self.environment = environment
        self.policy_version = policy_version
        self.sample_index = sample_index
        self.max_observation_tokens = max_observation_tokens
        self.turns: list[Turn] = []
        self._token_parts: list[Any] = []
        self._action_mask_parts: list[Any] = []
        self._log_prob_parts: list[Any] = []
        self._collecting_log_probs: Optional[bool] = None
        self._worker_policy_version: Optional[int] = None
        self._worker_policy_fingerprint: Optional[str] = None
        self.reward = 0.0
        self.reward_components: dict[str, float] = {}
        self.metadata: dict[str, Any] = {}
        self.done = False
        self.truncated = False
        self.terminal_reason = "running"
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
        return platform.cat(self._token_parts)
    async def start(self) -> None:
        """Reset the environment and append its initial observation."""
        if self._started:
            raise RuntimeError("AgentSession.start() may be called only once")
        self._started = True
        observation = await self.environment.reset(self.prompt)
        if observation.token_ids.numel() == 0:
            raise ValueError("The initial environment observation must not be empty")
        self._append_observation(observation, initial=True)
    def _append_observation(
        self,
        observation: Observation,
        initial: bool = False,
    ) -> None:
        """Append one non-trainable environment observation to the trajectory."""
        if observation.token_ids.ndim != 1:
            raise ValueError("Observation token_ids must be rank one")
        if (
            not initial
            and self.max_observation_tokens is not None
            and observation.token_ids.numel() > self.max_observation_tokens
        ):
            raise ValueError(
                "Environment observation exceeds agentic.max_observation_tokens: "
                f"tokens={observation.token_ids.numel()}, "
                f"limit={self.max_observation_tokens}"
            )
        start = sum(int(part.numel()) for part in self._token_parts)
        end = start + int(observation.token_ids.numel())
        self._token_parts.append(observation.token_ids)
        self._action_mask_parts.append(
            observation.token_ids.new_zeros(
                observation.token_ids.shape,
                dtype=platform.tensor_dtype.bool,
            )
        )
        self._log_prob_parts.append(
            observation.token_ids.new_zeros(
                observation.token_ids.shape,
                dtype=platform.tensor_dtype.float32,
            )
        )
        role = str(observation.metadata.get("role", "environment"))
        if role not in {"system", "user", "tool", "environment"}:
            raise ValueError(f"Unsupported observation role: {role}")
        self.turns.append(Turn(role, observation.content, start, end, False))
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
        start = sum(int(part.numel()) for part in self._token_parts)
        end = start + int(action.token_ids.numel())
        self._token_parts.append(action.token_ids)
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
        self.turns.append(Turn("assistant", action.content, start, end, True))
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
    async def apply(self, action: Action) -> None:
        """Apply one action and accumulate its resulting transition."""
        if not self.active:
            raise RuntimeError("Cannot apply an action to an inactive AgentSession")
        self._append_action(action)
        transition = await self.environment.step(action)
        self.reward += float(transition.reward)
        components = transition.info.get("reward_components")
        if components is None:
            components = {"environment": float(transition.reward)}
        for name, value in components.items():
            self.reward_components[str(name)] = (
                self.reward_components.get(str(name), 0.0) + float(value)
            )
        self.metadata.update(
            {
                str(name): value
                for name, value in transition.info.items()
                if name != "reward_components"
            }
        )
        self._append_observation(transition.observation)
        self.done = bool(transition.done)
        self.truncated = bool(transition.truncated)
        if self.done:
            self.terminal_reason = "done"
        elif self.truncated:
            self.terminal_reason = "environment_truncated"
    def finish_max_turns(self) -> None:
        """Truncate an active session after the configured turn limit."""
        if self.active:
            self.truncated = True
            self.terminal_reason = "max_turns"
    async def close(self) -> None:
        """Close the environment at most once."""
        if not self._closed:
            await self.environment.close()
            self._closed = True
    def build(self) -> Trajectory:
        """Build the immutable token-aligned trajectory for this episode."""
        if not self._started:
            raise RuntimeError("Cannot build an AgentSession before reset")
        token_ids = self.token_ids
        action_mask = platform.cat(self._action_mask_parts)
        token_log_probs = platform.cat(self._log_prob_parts)
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
            terminal_reason=self.terminal_reason,
            worker_policy_version=self._worker_policy_version,
            worker_policy_fingerprint=self._worker_policy_fingerprint,
            metadata={**self.metadata, "num_actions": len(self.action_contents)},
        )
