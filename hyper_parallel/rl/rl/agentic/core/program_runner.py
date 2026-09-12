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
"""User-owned agent control flow with the canonical HyperParallel-RL trajectory output."""

import asyncio
import time
from typing import Any, Callable, Optional, Protocol, Sequence

from hyper_parallel import get_platform
from rl.dataset.contracts import ExperienceBatch, PromptRecord, Trajectory, Turn
from rl.dataset.batch_builder import build_experience_batch
from rl.roles.rollout.base import GenerationSettings


platform = get_platform()


class AgentProgram(Protocol):
    """One user-defined episode, including its own tools and model calls."""

    async def run(self) -> Trajectory:
        """Execute one user-owned episode and return its trajectory."""

AgentProgramFactory = Callable[[PromptRecord, int, int], AgentProgram]


class ProgramAgentRunner:
    """Run user-owned agent programs and enforce only the data-plane contract.

    The factory receives ``(prompt, policy_version, sample_index)``.  User code
    owns the semantic loop and may call an external inference service, tools,
    or a sandbox.  HyperParallel-RL owns validation, batching, and downstream learning.
    """

    def __init__(
        self,
        program_factory: AgentProgramFactory,
        num_samples: int,
        settings: GenerationSettings,
        engine: Optional[Any] = None,
    ) -> None:
        """Initialize the runner with a program factory and sampling policy."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.program_factory = program_factory
        self.num_samples = num_samples
        self.settings = settings
        self.engine = engine

    async def _run(
        self,
        prompt_records: Sequence[PromptRecord],
        policy_version: int,
    ) -> tuple[Trajectory, ...]:
        """Run every sampled user program concurrently on one event loop."""
        programs = [
            self.program_factory(prompt, policy_version, sample_index)
            for prompt in prompt_records
            for sample_index in range(self.num_samples)
        ]
        return tuple(await asyncio.gather(*(program.run() for program in programs)))

    def rollout(
        self,
        prompt_records: Sequence[PromptRecord],
        policy_version: int,
    ) -> ExperienceBatch:
        """Run all user programs and batch their validated trajectories."""
        if not prompt_records:
            raise ValueError("ProgramAgentRunner requires at least one PromptRecord")
        started = time.perf_counter()
        trajectories = None
        local_error = None
        is_owner = bool(getattr(self.engine, "is_request_owner", True))
        if is_owner:
            try:
                trajectories = asyncio.run(self._run(prompt_records, policy_version))
            except Exception as error:  # pylint: disable=W0718
                local_error = error
        synchronize_error = getattr(self.engine, "synchronize_error", None)
        if callable(synchronize_error):
            synchronize_error(local_error, "agent program rollout")
        elif local_error is not None:
            raise local_error
        synchronize_payload = getattr(self.engine, "synchronize_agent_payload", None)
        if callable(synchronize_payload):
            payload = None if trajectories is None else self._serialize_trajectories(trajectories)
            payload = synchronize_payload(payload)
            trajectories = self._deserialize_trajectories(payload, prompt_records)
        if trajectories is None:
            raise RuntimeError("AgentProgram rollout produced no trajectories")
        allowed_prompt_ids = {prompt.prompt_id for prompt in prompt_records}
        for trajectory in trajectories:
            if trajectory.prompt_id not in allowed_prompt_ids:
                raise ValueError("AgentProgram returned a trajectory for an unknown prompt")
            if trajectory.policy_version != policy_version:
                raise ValueError(
                    "AgentProgram trajectory policy_version does not match the requested snapshot"
                )
        return build_experience_batch(
            trajectories=trajectories,
            generation_seconds=time.perf_counter() - started,
            settings=self.settings,
            metadata={"runner": "program"},
        )

    @staticmethod
    def _serialize_trajectories(trajectories: tuple[Trajectory, ...]) -> list[dict[str, Any]]:
        """Convert device tensors to an object-collective-safe payload."""
        return [
            {
                "trajectory_id": item.trajectory_id,
                "prompt_id": item.prompt_id,
                "group_id": item.group_id,
                "policy_version": item.policy_version,
                "turns": [
                    {
                        "role": turn.role,
                        "content": turn.content,
                        "token_start": turn.token_start,
                        "token_end": turn.token_end,
                        "trainable": turn.trainable,
                        "metadata": turn.metadata,
                    }
                    for turn in item.turns
                ],
                "token_ids": item.token_ids.detach().cpu().tolist(),
                "attention_mask": item.attention_mask.detach().cpu().tolist(),
                "action_mask": item.action_mask.detach().cpu().tolist(),
                "rollout_log_probs": (
                    None
                    if item.rollout_log_probs is None
                    else item.rollout_log_probs.detach().cpu().tolist()
                ),
                "reward": item.reward,
                "reward_components": item.reward_components,
                "done": item.done,
                "truncated": item.truncated,
                "terminal_reason": item.terminal_reason,
                "metadata": item.metadata,
                "worker_policy_version": item.worker_policy_version,
                "worker_policy_fingerprint": item.worker_policy_fingerprint,
            }
            for item in trajectories
        ]

    @staticmethod
    def _deserialize_trajectories(
        payload: Any,
        prompt_records: Sequence[PromptRecord],
    ) -> tuple[Trajectory, ...]:
        """Restore synchronized trajectory payloads on each Trainer TP sibling."""
        if not isinstance(payload, list):
            raise ValueError("Synchronized agent trajectory payload must be a list")
        prompts = {prompt.prompt_id: prompt for prompt in prompt_records}
        trajectories = []
        for item in payload:
            if not isinstance(item, dict) or item.get("prompt_id") not in prompts:
                raise ValueError("Synchronized agent trajectory references an unknown prompt")
            prototype = prompts[item["prompt_id"]].metadata.get("input_ids")
            if prototype is None or not platform.is_tensor(prototype):
                raise ValueError("Synchronized agent trajectory requires prompt input_ids")
            logprobs = item.get("rollout_log_probs")
            trajectories.append(
                Trajectory(
                    trajectory_id=str(item["trajectory_id"]),
                    prompt_id=str(item["prompt_id"]),
                    group_id=item.get("group_id"),
                    policy_version=int(item["policy_version"]),
                    turns=tuple(Turn(**turn) for turn in item["turns"]),
                    token_ids=prototype.new_tensor(item["token_ids"]),
                    attention_mask=prototype.new_tensor(item["attention_mask"]),
                    action_mask=prototype.new_tensor(
                        item["action_mask"],
                        dtype=platform.tensor_dtype.bool,
                    ),
                    rollout_log_probs=(
                        None
                        if logprobs is None
                        else prototype.new_tensor(logprobs, dtype=platform.tensor_dtype.float32)
                    ),
                    reward=float(item["reward"]),
                    reward_components=dict(item["reward_components"]),
                    done=bool(item["done"]),
                    truncated=bool(item["truncated"]),
                    terminal_reason=str(item["terminal_reason"]),
                    metadata=dict(item["metadata"]),
                    worker_policy_version=item.get("worker_policy_version"),
                    worker_policy_fingerprint=item.get("worker_policy_fingerprint"),
                )
            )
        return tuple(trajectories)
