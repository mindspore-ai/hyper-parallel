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
"""Config-switchable single-turn and tool-assisted GSM8K agent example."""

import re
from typing import Any, Optional

from examples.agents.gsm8k.tools import build_calculator_registry
from rl.agentic.core.chat_template import CHAT_TEMPLATE_MESSAGES
from rl.agentic.core.types import (
    Action,
    EpisodeContext,
    InteractionMode,
    Observation,
    Transition,
    TurnContext,
)
from rl.agentic.envs.environment import ENVIRONMENTS, ToolEnvironment
from rl.agentic.tools import ToolExecutor
from rl.agentic.tools.protocol import INTERACTION_PROTOCOLS
from rl.dataset.contracts import PromptRecord


_ANSWER_PATTERN = re.compile(r"####\s*(\-?[0-9\.\,]+)")
_NUMERIC_PATTERN = re.compile(r"^\s*(\-?[0-9\.\,]+)\s*$")
_REWARD_WINDOW = 300
PROMPT_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
_MULTI_TURN_PROMPT = """Solve the math problem carefully. You may call the calculator.
For a tool call, emit JSON with tool_calls and calculator arguments.
When finished, emit JSON exactly as {"final_answer":"NUMBER"}."""


def normalize_answer(value: str) -> str:
    """Remove display-only characters from a numeric GSM8K answer."""
    if "####" in value:
        value = value.rsplit("####", maxsplit=1)[-1]
    return value.replace(",", "").replace("$", "").strip()


def extract_answer(solution: str) -> Optional[str]:
    """Extract a strict final answer from a response or tool final-answer value."""
    matches = _ANSWER_PATTERN.findall(solution[-_REWARD_WINDOW:])
    if matches:
        return normalize_answer(matches[-1])
    direct = _NUMERIC_PATTERN.fullmatch(solution)
    return None if direct is None else normalize_answer(direct.group(1))


def compute_gsm8k_reward(solution: str, ground_truth: str) -> float:
    """Return numeric exact-match reward for one response."""
    predicted = extract_answer(solution)
    return float(
        predicted is not None and predicted == normalize_answer(str(ground_truth))
    )


def _validate_context(expected: EpisodeContext, received: EpisodeContext) -> None:
    """Reject lifecycle calls crossing episode identities."""
    if (
        expected.prompt.prompt_id != received.prompt.prompt_id
        or expected.policy_version != received.policy_version
        or expected.sample_index != received.sample_index
    ):
        raise ValueError("Environment context does not match its registered episode")


class GSM8KSingleTurnEnvironment:
    """Score one generated reasoning response and finish immediately."""

    def __init__(self, context: EpisodeContext) -> None:
        """Bind one prompt to a stateless single-turn environment."""
        self.episode = context
        self.prompt = context.prompt
        self._stepped = False

    async def reset(self, context: EpisodeContext) -> Observation:
        """Return the dataset's exact tokenized prompt."""
        _validate_context(self.episode, context)
        token_ids = self.prompt.metadata.get("input_ids")
        if token_ids is None or token_ids.ndim != 1 or token_ids.numel() == 0:
            raise ValueError("GSM8K PromptRecord.metadata.input_ids must be a non-empty tensor")
        return Observation(
            content=self.prompt.messages[-1].content,
            token_ids=token_ids,
            metadata={"role": "user"},
        )

    async def step(self, action: Action, context: TurnContext) -> Transition:
        """Score the first action with the shared GSM8K reward."""
        if self._stepped:
            raise RuntimeError("GSM8K single-turn environment accepts exactly one action")
        if context.turn_index != 0:
            raise ValueError("GSM8K single-turn environment requires turn zero")
        _validate_context(self.episode, context.episode)
        self._stepped = True
        reward = compute_gsm8k_reward(action.content, str(self.prompt.ground_truth))
        return Transition(
            observation=Observation(
                content="",
                token_ids=action.token_ids.new_empty((0,)),
                metadata={"role": "environment"},
            ),
            reward=reward,
            done=True,
            info={
                "reward_components": {"correctness": reward},
                "extracted_answer": extract_answer(action.content),
            },
        )

    async def close(self) -> None:
        """Close the stateless environment."""
        return None


class GSM8KMultiTurnEnvironment(ToolEnvironment):
    """Expose calculator feedback before scoring the final answer."""

    async def reset(self, context: EpisodeContext) -> Observation:
        """Render multi-turn instructions and the original question."""
        self._validate_episode(context)
        question = self.prompt.messages[-1].content
        return context.encode_observation(
            f"{_MULTI_TURN_PROMPT}\n\nQuestion:\n{question}",
            role="system",
            metadata={
                CHAT_TEMPLATE_MESSAGES: (
                    {"role": "system", "content": _MULTI_TURN_PROMPT},
                    {"role": "user", "content": question},
                ),
                "tools": ("calculator",),
            },
        )


def build_gsm8k_environment(context: EpisodeContext) -> Any:
    """Build single or multi-turn GSM8K behavior from one generic mode field."""
    if context.interaction_mode is InteractionMode.SINGLE_TURN:
        return GSM8KSingleTurnEnvironment(context)
    protocol_name = str(context.settings.get("protocol", "json_function_call"))
    protocol = INTERACTION_PROTOCOLS.build(protocol_name)
    timeout_value = context.settings.get("tool_timeout_seconds", 5.0)

    def score(answer: str, prompt: PromptRecord) -> float:
        """Apply the same correctness reward used by single-turn mode."""
        return compute_gsm8k_reward(answer, str(prompt.ground_truth))

    return GSM8KMultiTurnEnvironment(
        context=context,
        protocol=protocol,
        executor=ToolExecutor(
            build_calculator_registry(),
            timeout_seconds=None if timeout_value is None else float(timeout_value),
            max_concurrency=int(context.settings.get("tool_max_concurrency", 2)),
            max_calls_per_turn=int(context.settings.get("tool_max_calls_per_turn", 2)),
        ),
        reward_function=score,
        invalid_action_reward=float(context.settings.get("invalid_action_reward", -0.05)),
        tool_observation_role="tool",
    )


ENVIRONMENTS.register("gsm8k_tools")(build_gsm8k_environment)
