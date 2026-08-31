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
"""Pluggable action/observation protocols for reusable agent interactions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence

from rl.agentic.core.types import Action, ToolCall, ToolResult, TurnContext
from rl.registry import Registry


@dataclass(frozen=True)
class ParsedAction:
    """Protocol-neutral interpretation of one model action.

    Exactly one branch is present: ``tool_calls`` continues interaction, while
    ``final_answer`` terminates the episode and is passed to the task scorer.
    """

    tool_calls: tuple[ToolCall, ...] = ()
    final_answer: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject ambiguous actions at the protocol boundary."""
        has_calls = bool(self.tool_calls)
        has_answer = self.final_answer is not None
        if has_calls == has_answer:
            raise ValueError("ParsedAction must contain tool_calls or one final_answer")


class InteractionProtocol(Protocol):
    """Translate between model-specific syntax and Hyper-RL interaction data.

    Implement this protocol to support another interaction convention without
    changing AgentRunner, AgentSession, ToolEnvironment, or the trainer.
    """

    def parse_action(self, action: Action, context: TurnContext) -> ParsedAction:
        """Parse one raw model action into tool calls or a final answer."""

    def format_tool_results(
        self,
        results: Sequence[ToolResult],
        context: TurnContext,
    ) -> str:
        """Render tool results as exact incremental next-turn model input."""

    def format_error(self, message: str, context: TurnContext) -> str:
        """Render recoverable parser feedback for the next model turn."""


ResponseParser = InteractionProtocol


class ToolExecutorProtocol(Protocol):
    """Executor contract consumed by tool-capable environments."""

    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute one normalized tool call."""

    async def execute_many(self, calls: tuple[ToolCall, ...]) -> tuple[ToolResult, ...]:
        """Execute multiple calls while preserving request order."""

    async def close(self) -> None:
        """Release executor resources and reject future calls."""


ProtocolBuilder = Callable[[], InteractionProtocol]
INTERACTION_PROTOCOLS = Registry[ProtocolBuilder]("interaction protocol")


class JsonFunctionCallProtocol:
    """Strict JSON function-call protocol with OpenAI-style call fields.

    Accepted tool action example::

        {"tool_calls": [{"id": "call-1", "function": {
            "name": "calculator", "arguments": "{\\"expression\\": \\"1+1\\"}"}}]}

    Accepted terminal action example::

        {"final_answer": "2"}

    The class is an included reference protocol. Model-specific chat templates
    can implement :class:`InteractionProtocol` and reuse the same environment.
    """

    @staticmethod
    def _arguments(value: Any) -> dict[str, Any]:
        """Normalize object or JSON-string arguments into one mapping."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError(f"Tool arguments are not valid JSON: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValueError("Tool arguments must be a JSON object")
        return dict(value)

    @classmethod
    def _tool_call(cls, value: Any, index: int) -> ToolCall:
        """Normalize one simplified or OpenAI-compatible function call."""
        if not isinstance(value, dict):
            raise ValueError("Each tool_calls entry must be a JSON object")
        function = value.get("function", value)
        if not isinstance(function, dict):
            raise ValueError("Tool call function must be a JSON object")
        call_id = value.get("id", value.get("call_id", f"call-{index}"))
        name = function.get("name", "")
        if not isinstance(call_id, str) or not isinstance(name, str):
            raise ValueError("Tool call id and function name must be strings")
        arguments = cls._arguments(function.get("arguments", {}))
        return ToolCall(call_id=call_id, name=name, arguments=arguments)

    def parse_action(self, action: Action, context: TurnContext) -> ParsedAction:
        """Parse strict JSON while preserving the original action in trajectory."""
        del context
        try:
            payload = json.loads(action.content)
        except json.JSONDecodeError as error:
            raise ValueError(f"Agent action is not valid JSON: {error.msg}") from error
        if not isinstance(payload, dict):
            raise ValueError("Agent action must be a JSON object")
        if "final_answer" in payload:
            answer = payload["final_answer"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("final_answer must be a non-empty string")
            return ParsedAction(final_answer=answer.strip())
        raw_calls = payload.get("tool_calls")
        if not isinstance(raw_calls, list) or not raw_calls:
            raise ValueError("Agent action must contain non-empty tool_calls or final_answer")
        calls = tuple(
            self._tool_call(value, index)
            for index, value in enumerate(raw_calls)
        )
        return ParsedAction(tool_calls=calls)

    def format_tool_results(
        self,
        results: Sequence[ToolResult],
        context: TurnContext,
    ) -> str:
        """Serialize correlated tool results for the next model turn."""
        del context
        payload = {
            "tool_results": [
                {
                    "tool_call_id": result.call_id,
                    "name": result.name,
                    "content": result.content,
                    "is_error": result.is_error,
                }
                for result in results
            ]
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def format_error(self, message: str, context: TurnContext) -> str:
        """Return a structured protocol error so the model can self-correct."""
        del context
        return json.dumps(
            {"interaction_error": message},
            ensure_ascii=False,
            separators=(",", ":"),
        )


class OpenAIToolCallProtocol(JsonFunctionCallProtocol):
    """OpenAI-compatible assistant/tool message protocol.

    Generated actions may be either the assistant message object itself or the
    compact payload accepted by :class:`JsonFunctionCallProtocol`. Tool
    observations are emitted as OpenAI ``role=tool`` message objects so another
    generation turn can correlate every result with its call id.
    """

    def parse_action(self, action: Action, context: TurnContext) -> ParsedAction:
        """Parse an OpenAI assistant message or a compact action payload."""
        try:
            payload = json.loads(action.content)
        except json.JSONDecodeError:
            return super().parse_action(action, context)
        if not isinstance(payload, dict):
            return super().parse_action(action, context)
        if payload.get("role") == "assistant":
            tool_calls = payload.get("tool_calls")
            if tool_calls:
                compact = json.dumps({"tool_calls": tool_calls}, ensure_ascii=False)
                normalized = Action(compact, action.token_ids, action.rollout_log_probs, action.metadata)
                return super().parse_action(normalized, context)
            content = payload.get("content")
            if isinstance(content, str) and content.strip():
                return ParsedAction(final_answer=content.strip())
            raise ValueError("OpenAI assistant message must contain tool_calls or non-empty content")
        return super().parse_action(action, context)

    def format_tool_results(
        self,
        results: Sequence[ToolResult],
        context: TurnContext,
    ) -> str:
        """Serialize tool results as newline-delimited OpenAI tool messages."""
        del context
        messages = (
            {
                "role": "tool",
                "tool_call_id": result.call_id,
                "name": result.name,
                "content": result.content,
                "is_error": result.is_error,
            }
            for result in results
        )
        return "\n".join(
            json.dumps(message, ensure_ascii=False, separators=(",", ":"))
            for message in messages
        )

    def format_error(self, message: str, context: TurnContext) -> str:
        """Return a recoverable OpenAI-style tool observation."""
        del context
        return json.dumps(
            {
                "role": "tool",
                "tool_call_id": "protocol-error",
                "name": "interaction_protocol",
                "content": message,
                "is_error": True,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


@INTERACTION_PROTOCOLS.register("json_function_call")
def build_json_function_call_protocol() -> JsonFunctionCallProtocol:
    """Build the included strict JSON function-call protocol."""
    return JsonFunctionCallProtocol()


@INTERACTION_PROTOCOLS.register("openai_tool_call")
def build_openai_tool_call_protocol() -> OpenAIToolCallProtocol:
    """Build the OpenAI-compatible tool-calling protocol."""
    return OpenAIToolCallProtocol()
