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
"""Single-turn code environment preserving the original policy action."""

from typing import Any
from uuid import uuid4

from rl.agentic.core.types import Action, EpisodeContext, Observation, TerminationReason, Transition, TurnContext
from rl.agentic.envs.environment import ENVIRONMENTS
from examples.code.client import SandboxFusionExecutor
from examples.code.judge import judge_stdio, validate_tests


class CodeEnvironment:
    """Judge a Python stdio candidate through an isolated execution service."""

    def __init__(self, context: EpisodeContext, executor: Any, runtime_version: str) -> None:
        """Own one candidate identity and its execution client."""
        if context.max_turns != 1:
            raise ValueError("CodeEnvironment requires single-turn episodes")
        validate_tests(context.prompt.ground_truth)
        metadata = context.prompt.metadata
        if metadata.get("task_type") != "code_stdio" or metadata.get("language") != "python":
            raise ValueError("CodeEnvironment requires Python code_stdio task metadata")
        self.episode = context
        self.executor = executor
        self.runtime_version = runtime_version
        self.candidate_id = f"{context.prompt.prompt_id}:{context.policy_version}:{context.sample_index}:{uuid4().hex}"
        self._stepped = False

    def _validate_context(self, context: EpisodeContext) -> None:
        if (context.prompt.prompt_id, context.policy_version, context.sample_index) != (
                self.episode.prompt.prompt_id, self.episode.policy_version, self.episode.sample_index):
            raise ValueError("Code environment context belongs to another candidate")

    async def reset(self, context: EpisodeContext) -> Observation:
        """Return the dataset prompt tokens without exposing private judge tests."""
        self._validate_context(context)
        tokens = context.prompt.metadata.get("input_ids")
        if tokens is None or tokens.ndim != 1 or tokens.numel() == 0:
            raise ValueError("Code prompt requires non-empty tokenized input_ids")
        return Observation(content=context.prompt.messages[-1].content, token_ids=tokens, metadata={"role": "user"})

    async def step(self, action: Action, context: TurnContext) -> Transition:
        """Score one completion and finish without modifying its sampled tokens."""
        self._validate_context(context.episode)
        if self._stepped or context.turn_index != 0:
            raise RuntimeError("CodeEnvironment accepts exactly one action")
        self._stepped = True
        reward = await judge_stdio(
            action.content, self.episode.prompt.ground_truth, self.executor,
            candidate_id=self.candidate_id, runtime_version=self.runtime_version,
        )
        finish_reason = action.metadata.get("finish_reason")
        truncated = finish_reason == "length"
        return Transition(
            observation=Observation(content="", token_ids=action.token_ids.new_empty((0,)),
                                    metadata={"role": "environment"}),
            reward=reward.value, done=True, truncated=truncated,
            termination_reason=TerminationReason.ENVIRONMENT_TRUNCATED if truncated else None,
            info={"reward_components": dict(reward.components), "finish_reason": finish_reason,
                  **dict(reward.metadata)},
        )

    async def close(self) -> None:
        """Close the client's transport; remote work follows the server timeout."""
        await self.executor.close()


def build_code_environment(context: EpisodeContext) -> CodeEnvironment:
    """Build the single-turn stdio environment from its explicit execution settings."""
    settings = context.settings.get("code", {})
    allowed = {"endpoint", "runtime_version", "run_timeout", "request_timeout"}
    if not isinstance(settings, dict) or set(settings) - allowed:
        raise ValueError("Unsupported agentic.code settings for the fixed SandboxFusion profile")
    runtime_version = settings.get("runtime_version")
    if not isinstance(runtime_version, str) or not runtime_version.strip():
        raise ValueError("agentic.code.runtime_version must identify the deployed execution image")
    endpoint = settings.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("agentic.code.endpoint is required")
    executor = SandboxFusionExecutor(endpoint, float(settings.get("run_timeout", 5.0)),
                                    float(settings.get("request_timeout", 300.0)))
    return CodeEnvironment(context, executor, runtime_version)


ENVIRONMENTS.register("code_stdio")(build_code_environment)
