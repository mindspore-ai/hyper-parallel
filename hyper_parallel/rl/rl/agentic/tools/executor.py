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
"""Bounded asynchronous execution for registered agent tools."""

import asyncio
import contextvars
import inspect
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from rl.agentic.core.types import Action, ToolCall, ToolResult, TurnContext
from rl.registry import Registry


_DEFAULT_MAX_CONCURRENCY = 4
_DEFAULT_MAX_CALLS_PER_TURN = 16
ToolHandler = Callable[..., Any]
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class Tool:
    """One callable exposed to an interaction protocol."""

    name: str
    handler: ToolHandler
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate stable public tool metadata."""
        if _TOOL_NAME.fullmatch(self.name) is None:
            raise ValueError("Tool name must contain 1-64 letters, digits, underscores, or dashes")
        if not callable(self.handler):
            raise ValueError(f"Tool handler must be callable: {self.name}")
        if not isinstance(self.parameters, Mapping):
            raise ValueError(f"Tool parameters schema must be a mapping: {self.name}")
        object.__setattr__(self, "parameters", dict(self.parameters))


class ToolRegistry:
    """Per-environment registry that avoids global tool side effects."""

    def __init__(self) -> None:
        """Create an empty episode-local registry."""
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        *,
        description: str = "",
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> Callable[[ToolHandler], ToolHandler]:
        """Register a handler and return it unchanged for decorator use."""

        def decorator(handler: ToolHandler) -> ToolHandler:
            """Bind the decorated handler to validated tool metadata."""
            tool = Tool(name, handler, description, dict(parameters or {}))
            if tool.name in self._tools:
                raise ValueError(f"Tool is already registered: {tool.name}")
            self._tools[tool.name] = tool
            return handler

        return decorator

    def get(self, name: str) -> Tool:
        """Return a named tool or report the local choices."""
        try:
            return self._tools[name]
        except KeyError as error:
            raise ValueError(f"Unknown tool '{name}'; available={sorted(self._tools)}") from error

    @property
    def names(self) -> tuple[str, ...]:
        """Return tool names in deterministic order."""
        return tuple(sorted(self._tools))


@dataclass(frozen=True)
class ParsedAction:
    """Protocol-neutral tool calls or final answer parsed from one action."""

    tool_calls: tuple[ToolCall, ...] = ()
    final_answer: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject ambiguous actions at the protocol boundary."""
        if bool(self.tool_calls) == (self.final_answer is not None):
            raise ValueError("ParsedAction must contain tool_calls or one final_answer")


class InteractionProtocol(Protocol):
    """Translate model syntax to tool calls and format observations."""

    def parse_action(self, action: Action, context: TurnContext) -> ParsedAction:
        """Parse one raw model action into tool calls or a final answer."""

    def format_tool_results(self, results: Sequence[ToolResult], context: TurnContext) -> str:
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
    """Strict JSON function-call protocol with OpenAI-style call fields."""

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
        return ToolCall(call_id, name, cls._arguments(function.get("arguments", {})))

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
        return ParsedAction(
            tool_calls=tuple(self._tool_call(value, index) for index, value in enumerate(raw_calls))
        )

    def format_tool_results(self, results: Sequence[ToolResult], context: TurnContext) -> str:
        """Serialize correlated tool results for the next model turn."""
        del context
        payload = {"tool_results": [
            {
                "tool_call_id": result.call_id,
                "name": result.name,
                "content": result.content,
                "is_error": result.is_error,
            }
            for result in results
        ]}
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def format_error(self, message: str, context: TurnContext) -> str:
        """Return structured feedback so the model can self-correct."""
        del context
        return json.dumps({"interaction_error": message}, ensure_ascii=False, separators=(",", ":"))


class OpenAIToolCallProtocol(JsonFunctionCallProtocol):
    """OpenAI-compatible assistant/tool message protocol."""

    def parse_action(self, action: Action, context: TurnContext) -> ParsedAction:
        """Parse an OpenAI assistant message or a compact action payload."""
        try:
            payload = json.loads(action.content)
        except json.JSONDecodeError:
            return super().parse_action(action, context)
        if not isinstance(payload, dict) or payload.get("role") != "assistant":
            return super().parse_action(action, context)
        tool_calls = payload.get("tool_calls")
        if tool_calls:
            compact = json.dumps({"tool_calls": tool_calls}, ensure_ascii=False)
            normalized = Action(compact, action.token_ids, action.rollout_log_probs, action.metadata)
            return super().parse_action(normalized, context)
        content = payload.get("content")
        if isinstance(content, str) and content.strip():
            return ParsedAction(final_answer=content.strip())
        raise ValueError("OpenAI assistant message must contain tool_calls or non-empty content")

    def format_tool_results(self, results: Sequence[ToolResult], context: TurnContext) -> str:
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
            json.dumps(message, ensure_ascii=False, separators=(",", ":")) for message in messages
        )

    def format_error(self, message: str, context: TurnContext) -> str:
        """Return a recoverable OpenAI-style tool observation."""
        del context
        payload = {
            "role": "tool",
            "tool_call_id": "protocol-error",
            "name": "interaction_protocol",
            "content": message,
            "is_error": True,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@INTERACTION_PROTOCOLS.register("json_function_call")
def build_json_function_call_protocol() -> JsonFunctionCallProtocol:
    """Build the included strict JSON function-call protocol."""
    return JsonFunctionCallProtocol()


@INTERACTION_PROTOCOLS.register("openai_tool_call")
def build_openai_tool_call_protocol() -> OpenAIToolCallProtocol:
    """Build the OpenAI-compatible tool-calling protocol."""
    return OpenAIToolCallProtocol()


class ToolExecutor:
    """Execute registered sync or async tools and return model-visible errors."""

    def __init__(
        self,
        registry: ToolRegistry,
        timeout_seconds: Optional[float] = None,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        max_calls_per_turn: int = _DEFAULT_MAX_CALLS_PER_TURN,
    ) -> None:
        """Initialize execution policy for one environment.

        Args:
            registry: User-owned tools available to the episode.
            timeout_seconds: Optional positive timeout applied to each call.
            max_concurrency: Maximum calls that may be executing concurrently.
            max_calls_per_turn: Maximum calls accepted from one model action.
        """
        if timeout_seconds is not None:
            if isinstance(timeout_seconds, bool) or not isinstance(
                timeout_seconds, (int, float)
            ):
                raise ValueError("Tool timeout_seconds must be numeric or null")
            if timeout_seconds <= 0:
                raise ValueError("Tool timeout_seconds must be positive when configured")
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
            raise ValueError("Tool max_concurrency must be an integer")
        if max_concurrency <= 0:
            raise ValueError("Tool max_concurrency must be positive")
        if isinstance(max_calls_per_turn, bool) or not isinstance(max_calls_per_turn, int):
            raise ValueError("Tool max_calls_per_turn must be an integer")
        if max_calls_per_turn <= 0:
            raise ValueError("Tool max_calls_per_turn must be positive")
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self.max_concurrency = max_concurrency
        self.max_calls_per_turn = max_calls_per_turn
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._capacity: Optional[asyncio.Semaphore] = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._closed = False

    @staticmethod
    def _serialize(value: Any) -> str:
        """Convert common structured values into stable observation text."""
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list, tuple, int, float, bool)) or value is None:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return str(value)

    @staticmethod
    def _matches_json_type(value: Any, expected: str) -> bool:
        """Return whether a value satisfies one supported JSON Schema type."""
        predicates = {
            "array": lambda item: isinstance(item, (list, tuple)),
            "boolean": lambda item: isinstance(item, bool),
            "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "null": lambda item: item is None,
            "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
            "object": lambda item: isinstance(item, Mapping),
            "string": lambda item: isinstance(item, str),
        }
        predicate = predicates.get(expected)
        if predicate is None:
            raise ValueError(f"Unsupported tool schema type: {expected}")
        return predicate(value)

    @classmethod
    def _validate_arguments(cls, tool: Tool, arguments: dict[str, Any]) -> None:
        """Validate call shape and the supported JSON Schema object subset."""
        try:
            inspect.signature(tool.handler).bind(**arguments)
        except TypeError as error:
            raise ValueError(
                f"Invalid arguments for tool '{tool.name}': {error}"
            ) from error
        schema = tool.parameters
        if not schema:
            return
        if schema.get("type", "object") != "object":
            raise ValueError(f"Tool '{tool.name}' parameters schema must describe an object")
        required = schema.get("required", ())
        if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
            raise ValueError(f"Tool '{tool.name}' schema required must be a sequence")
        missing = [name for name in required if name not in arguments]
        if missing:
            raise ValueError(f"Invalid arguments for tool '{tool.name}': missing {missing}")
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise ValueError(f"Tool '{tool.name}' schema properties must be a mapping")
        for name, value in arguments.items():
            property_schema = properties.get(name)
            if property_schema is None:
                if schema.get("additionalProperties", True) is False:
                    raise ValueError(
                        f"Invalid arguments for tool '{tool.name}': unexpected '{name}'"
                    )
                continue
            if not isinstance(property_schema, Mapping):
                raise ValueError(f"Tool '{tool.name}' property schema must be a mapping")
            expected = property_schema.get("type")
            if expected is not None and (
                not isinstance(expected, str)
                or not cls._matches_json_type(value, expected)
            ):
                raise ValueError(
                    f"Invalid arguments for tool '{tool.name}': '{name}' must be {expected}"
                )

    @staticmethod
    def _set_thread_result(
        future: asyncio.Future[Any],
        value: Any,
        error: Optional[Exception],
    ) -> None:
        """Complete a sync-tool Future unless its event loop discarded it."""
        if future.done():
            return
        if error is None:
            future.set_result(value)
        else:
            future.set_exception(error)

    async def _invoke_sync(self, tool: Tool, arguments: dict[str, Any]) -> Any:
        """Run a synchronous handler in one isolated daemon worker thread."""
        event_loop = asyncio.get_running_loop()
        future = event_loop.create_future()
        context = contextvars.copy_context()

        def invoke() -> None:
            """Execute the handler and safely publish its result to the loop."""
            value = None
            error = None
            try:
                value = context.run(tool.handler, **arguments)
            except BaseException as invocation_error:  # pylint: disable=W0718
                if isinstance(invocation_error, Exception):
                    error = invocation_error
                else:
                    error = RuntimeError(
                        f"Tool raised {type(invocation_error).__name__}"
                    )
            try:
                event_loop.call_soon_threadsafe(
                    self._set_thread_result,
                    future,
                    value,
                    error,
                )
            except RuntimeError:
                # The owning rollout loop may already be closed after shutdown.
                pass

        worker = threading.Thread(
            target=invoke,
            name=f"hyper-rl-tool-{tool.name}",
            daemon=True,
        )
        worker.start()
        result = await future
        if inspect.isawaitable(result):
            return await result
        return result

    async def _invoke(self, tool: Tool, arguments: dict[str, Any]) -> Any:
        """Run synchronous handlers off-loop and await asynchronous handlers."""
        self._validate_arguments(tool, arguments)
        if inspect.iscoroutinefunction(tool.handler):
            return await tool.handler(**arguments)
        return await self._invoke_sync(tool, arguments)

    def _execution_capacity(self) -> asyncio.Semaphore:
        """Return a semaphore bound to this executor's single event loop."""
        event_loop = asyncio.get_running_loop()
        if self._event_loop is None:
            self._event_loop = event_loop
            self._capacity = asyncio.Semaphore(self.max_concurrency)
        elif self._event_loop is not event_loop:
            raise RuntimeError("ToolExecutor cannot be shared across event loops")
        if self._capacity is None:
            raise RuntimeError("ToolExecutor capacity was not initialized")
        return self._capacity

    @staticmethod
    def _remaining_timeout(deadline: Optional[float]) -> Optional[float]:
        """Return the time left for one call, including queueing delay."""
        if deadline is None:
            return None
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError
        return remaining

    async def _acquire_capacity(
        self,
        capacity: asyncio.Semaphore,
        deadline: Optional[float],
    ) -> None:
        """Acquire one physical execution slot within the call deadline."""
        remaining = self._remaining_timeout(deadline)
        if remaining is None:
            await capacity.acquire()
        else:
            await asyncio.wait_for(capacity.acquire(), timeout=remaining)

    def _retain_timed_out_sync_task(
        self,
        task: asyncio.Task[Any],
        capacity: asyncio.Semaphore,
    ) -> None:
        """Keep a timed-out sync call bounded until its worker really exits."""
        self._background_tasks.add(task)

        def release(completed: asyncio.Task[Any]) -> None:
            """Release capacity after a timed-out worker actually terminates."""
            self._background_tasks.discard(completed)
            capacity.release()
            try:
                completed.result()
            except BaseException:  # pylint: disable=W0718
                # The model already received the timeout result.
                pass

        task.add_done_callback(release)

    async def _execute_tool(self, tool: Tool, arguments: dict[str, Any]) -> Any:
        """Execute one tool under the configured deadline and capacity bound."""
        event_loop = asyncio.get_running_loop()
        deadline = (
            None
            if self.timeout_seconds is None
            else event_loop.time() + self.timeout_seconds
        )
        capacity = self._execution_capacity()
        await self._acquire_capacity(capacity, deadline)
        release_capacity = True
        try:
            remaining = self._remaining_timeout(deadline)
            invocation = asyncio.create_task(self._invoke(tool, arguments))
            if inspect.iscoroutinefunction(tool.handler):
                if remaining is None:
                    return await invocation
                return await asyncio.wait_for(invocation, timeout=remaining)
            try:
                if remaining is None:
                    return await asyncio.shield(invocation)
                return await asyncio.wait_for(
                    asyncio.shield(invocation),
                    timeout=remaining,
                )
            except BaseException:
                if not invocation.done():
                    self._retain_timed_out_sync_task(invocation, capacity)
                    release_capacity = False
                else:
                    try:
                        invocation.result()
                    except BaseException:  # pylint: disable=W0718
                        pass
                raise
        finally:
            if release_capacity:
                capacity.release()

    def _error_result(self, call: ToolCall, error: Exception) -> ToolResult:
        """Build one stable model-visible execution error."""
        if isinstance(error, (TimeoutError, asyncio.TimeoutError)) and self.timeout_seconds is not None:
            content = (
                f"ToolTimeoutError: tool '{call.name}' exceeded "
                f"{self.timeout_seconds:g} seconds"
            )
        else:
            content = f"{type(error).__name__}: {error}"
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            content=content,
            is_error=True,
            metadata={"error_type": type(error).__name__},
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute one call without turning recoverable tool errors into rollout failure."""
        try:
            if self._closed:
                raise RuntimeError("ToolExecutor is closed")
            tool = self.registry.get(call.name)
            value = await self._execute_tool(tool, call.arguments)
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                content=self._serialize(value),
            )
        except Exception as error:  # pylint: disable=W0718
            return self._error_result(call, error)

    async def execute_many(self, calls: tuple[ToolCall, ...]) -> tuple[ToolResult, ...]:
        """Execute a bounded call set concurrently while preserving request order."""
        if len(calls) > self.max_calls_per_turn:
            message = (
                f"ToolCallLimitError: received {len(calls)} calls; "
                f"maximum is {self.max_calls_per_turn}"
            )
            return tuple(
                ToolResult(
                    call.call_id,
                    call.name,
                    message,
                    is_error=True,
                    metadata={"error_type": "ToolCallLimitError"},
                )
                for call in calls
            )
        return tuple(await asyncio.gather(*(self.execute(call) for call in calls)))

    async def close(self) -> None:
        """Reject new calls without waiting for already timed-out sync handlers."""
        self._closed = True
