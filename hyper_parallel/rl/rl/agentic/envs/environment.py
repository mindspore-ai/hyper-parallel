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
"""Environment registration, extension loading, and tool composition."""

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from rl.agentic.core.types import Action, EpisodeContext, Observation, Transition, TurnContext
from rl.agentic.envs.base import Environment
from rl.agentic.tools.protocol import InteractionProtocol, ParsedAction, ToolExecutorProtocol
from rl.registry import Registry


EnvironmentBuilder = Callable[[EpisodeContext], Environment]
RewardFunction = Callable[[str, Any], float]
ENVIRONMENTS = Registry[EnvironmentBuilder]("environment")


def load_agentic_module(module_path: str) -> ModuleType:
    """Load one user extension by import name or explicit Python file path."""
    if not isinstance(module_path, str) or not module_path.strip():
        raise ValueError("agentic.module_path must be a non-empty import or file path")
    if "/" not in module_path and "\\" not in module_path and not module_path.endswith(".py"):
        try:
            return importlib.import_module(module_path)
        except Exception as error:  # pylint: disable=W0718
            raise RuntimeError(f"Failed to import Agentic module: {module_path}") from error
    path = Path(module_path).expanduser().resolve()
    if path.suffix != ".py" or not path.is_file():
        raise ValueError(f"Agentic module must be an existing Python file: {path}")
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    module_name = f"_hyper_rl_agentic_{digest}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to create an import specification for: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # pylint: disable=W0718
        sys.modules.pop(module_name, None)
        raise RuntimeError(f"Failed to load Agentic module: {path}") from error
    return module


class ToolEnvironment:
    """Compose a response parser, tool executor, and terminal reward."""

    def __init__(
        self,
        context: EpisodeContext,
        protocol: InteractionProtocol,
        executor: ToolExecutorProtocol,
        reward_function: RewardFunction,
        invalid_action_reward: float = 0.0,
        tool_observation_role: str = "tool",
    ) -> None:
        """Bind task-independent interaction pieces to one episode."""
        if tool_observation_role not in {"tool", "environment", "user"}:
            raise ValueError("tool_observation_role must be tool, environment, or user")
        self.episode = context
        self.prompt = context.prompt
        self.protocol = protocol
        self.executor = executor
        self.reward_function = reward_function
        self.invalid_action_reward = float(invalid_action_reward)
        self.tool_observation_role = tool_observation_role

    def _validate_episode(self, context: EpisodeContext) -> None:
        """Ensure lifecycle calls cannot cross episode identities."""
        if (
            context.prompt.prompt_id != self.prompt.prompt_id
            or context.policy_version != self.episode.policy_version
            or context.sample_index != self.episode.sample_index
        ):
            raise ValueError("ToolEnvironment context does not match its registered episode")

    async def reset(self, context: EpisodeContext) -> Observation:
        """Return the dataset's exact tokenized prompt as initial input."""
        self._validate_episode(context)
        token_ids = self.prompt.metadata.get("input_ids")
        if token_ids is None or token_ids.ndim != 1 or token_ids.numel() == 0:
            raise ValueError("ToolEnvironment requires non-empty prompt input_ids")
        return Observation(
            content=self.prompt.messages[-1].content,
            token_ids=token_ids,
            metadata={"role": "user"},
        )

    def _invalid_action_transition(self, error: ValueError, context: TurnContext) -> Transition:
        """Return parser feedback that lets a multi-turn model self-correct."""
        message = str(error)
        feedback = self.protocol.format_error(message, context)
        return Transition(
            observation=context.episode.encode_observation(
                feedback,
                role="environment",
                metadata={"interaction_error": message},
            ),
            reward=self.invalid_action_reward,
            done=False,
            info={"interaction_error": message},
        )

    def _final_answer_transition(self, answer: str, action: Action) -> Transition:
        """Score a final answer and build a terminal transition."""
        reward = float(self.reward_function(answer, self.prompt))
        return Transition(
            observation=Observation(
                content="",
                token_ids=action.token_ids.new_empty((0,)),
                metadata={"role": "environment"},
            ),
            reward=reward,
            done=True,
            info={"final_answer": answer, "reward_components": {"task": reward}},
        )

    async def _tool_result_transition(
        self,
        parsed: ParsedAction,
        context: TurnContext,
    ) -> Transition:
        """Execute parsed calls and expose correlated results."""
        results = await self.executor.execute_many(parsed.tool_calls)
        feedback = self.protocol.format_tool_results(results, context)
        error_count = sum(int(result.is_error) for result in results)
        return Transition(
            observation=context.episode.encode_observation(
                feedback,
                role=self.tool_observation_role,
                metadata={
                    "tool_call_ids": tuple(result.call_id for result in results),
                    "tool_names": tuple(result.name for result in results),
                    "tool_errors": tuple(result.is_error for result in results),
                },
            ),
            reward=0.0,
            done=False,
            info={
                "tool_call_count": len(results),
                "tool_success_count": len(results) - error_count,
                "tool_error_count": error_count,
            },
        )

    async def step(self, action: Action, context: TurnContext) -> Transition:
        """Parse and dispatch one protocol-independent transition."""
        self._validate_episode(context.episode)
        try:
            parsed = self.protocol.parse_action(action, context)
        except ValueError as error:
            return self._invalid_action_transition(error, context)
        if parsed.final_answer is not None:
            return self._final_answer_transition(parsed.final_answer, action)
        return await self._tool_result_transition(parsed, context)

    async def close(self) -> None:
        """Close the episode-owned executor."""
        await self.executor.close()
