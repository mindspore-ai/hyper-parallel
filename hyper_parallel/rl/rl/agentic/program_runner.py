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
"""User-owned agent control flow with the canonical Hyper-RL trajectory output."""

import asyncio
import time
from typing import Callable, Protocol, Sequence

from rl.contracts import ExperienceBatch, PromptRecord, Trajectory
from rl.dataset import build_experience_batch
from rl.roles.rollout.base import GenerationSettings


class AgentProgram(Protocol):
    """One user-defined episode, including its own tools and model calls."""

    async def run(self) -> Trajectory:
        """Execute one user-owned episode and return its trajectory."""


AgentProgramFactory = Callable[[PromptRecord, int, int], AgentProgram]


class ProgramAgentRunner:
    """Run user-owned agent programs and enforce only the data-plane contract.

    The factory receives ``(prompt, policy_version, sample_index)``.  User code
    owns the semantic loop and may call an external inference service, tools,
    or a sandbox.  Hyper-RL owns validation, batching, and downstream learning.
    """

    def __init__(
        self,
        program_factory: AgentProgramFactory,
        num_samples: int,
        settings: GenerationSettings,
    ) -> None:
        """Initialize the runner with a program factory and sampling policy."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.program_factory = program_factory
        self.num_samples = num_samples
        self.settings = settings

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
        trajectories = asyncio.run(self._run(prompt_records, policy_version))
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
