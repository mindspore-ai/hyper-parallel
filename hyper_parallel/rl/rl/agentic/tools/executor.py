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
import threading
from typing import Any, Mapping, Optional, Sequence

from rl.agentic.core.types import ToolCall, ToolResult
from rl.agentic.tools.registry import Tool, ToolRegistry


_DEFAULT_MAX_CONCURRENCY = 4
_DEFAULT_MAX_CALLS_PER_TURN = 16


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
        if isinstance(error, TimeoutError) and self.timeout_seconds is not None:
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
