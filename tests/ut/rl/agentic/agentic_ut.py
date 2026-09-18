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
"""Agentic unit tests and the mandatory 80 percent coverage gates.

Run from the repository root with::

    python tests/ut/rl/agentic/agentic_ut.py
"""
# White-box regression tests intentionally exercise internal state and lifecycle hooks.
# pylint: disable=protected-access

from __future__ import annotations

import asyncio
import importlib
import io
import json
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from importlib.util import find_spec
from types import ModuleType, SimpleNamespace
from typing import Any
from urllib.error import HTTPError, URLError

import pytest
import torch

_MINIMUM_COVERAGE = 80.0
_HERE = Path(__file__).resolve().parent
_RL_SOURCE_ROOT = Path(find_spec("hyper_parallel").origin).parent / "rl"
_AGENTIC_ROOT = _RL_SOURCE_ROOT / "rl" / "agentic"
_AGENTIC_TESTS = (Path(__file__).resolve(),)


def _modules() -> SimpleNamespace:
    """Import production modules after the standalone coverage collector starts."""
    names = {
        "agentic": "rl.agentic",
        "chat": "rl.agentic.core.chat_template",
        "types": "rl.agentic.core.types",
        "environment": "rl.agentic.envs.environment",
        "executor": "rl.agentic.tools.executor",
        "protocol": "rl.agentic.tools.executor",
        "registry": "rl.agentic.tools.executor",
        "mcp": "rl.agentic.mcp_server",
        "codex_harness": "rl.agentic.codex.harness",
        "deepseek_harness": "rl.agentic.ds_harness.harness",
    }
    return SimpleNamespace(
        **{name: importlib.import_module(module_name) for name, module_name in names.items()}
    )


def _prompt(prompt_id: str = "prompt-1") -> Any:
    """Build the smallest tokenized prompt accepted by Agentic contracts."""
    contracts = importlib.import_module("rl.dataset.contracts")
    return contracts.PromptRecord(
        prompt_id,
        (contracts.Message("user", "What is 1 + 1?"),),
        ground_truth="2",
        metadata={"input_ids": torch.tensor([10, 11])},
    )


def _reward_module(monkeypatch: pytest.MonkeyPatch, value: Any = 1.0) -> str:
    """Install one importable reward callback for Harness constructors."""
    module = ModuleType("agentic_ut_reward")
    module.score = lambda _answer, _prompt: value
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return f"{module.__name__}:score"


def _episode(modules: SimpleNamespace, **overrides: Any) -> Any:
    """Build a valid episode context with an observable encoder."""
    values = {
        "prompt": _prompt(),
        "policy_version": 2,
        "sample_index": 1,
        "max_turns": 2,
        "observation_encoder": lambda content, role, metadata: modules.types.Observation(
            content, torch.tensor([99]), {"role": role, **metadata}
        ),
    }
    values.update(overrides)
    return modules.types.EpisodeContext(**values)


def test_public_exports_and_value_contracts() -> None:
    """Public lazy exports and immutable value objects retain their contracts."""
    modules = _modules()
    assert "AgentRunner" in dir(modules.agentic)
    assert modules.agentic.AgentRunner.__name__ == "AgentRunner"
    assert (
        importlib.import_module("rl.agentic.envs.base").Environment
        is modules.environment.Environment
    )
    assert (
        importlib.import_module("rl.agentic.tools.registry").ToolRegistry
        is modules.executor.ToolRegistry
    )
    assert (
        importlib.import_module("rl.agentic.tools.protocol").InteractionProtocol
        is modules.executor.InteractionProtocol
    )
    assert importlib.import_module("rl.agentic.codex.trajectory").build_codex_trajectory
    assert importlib.import_module("rl.agentic.ds_harness.trajectory").build_deepseek_trajectory
    with pytest.raises(AttributeError):
        getattr(modules.agentic, "missing_export")

    assert modules.types.InteractionMode.parse("single_turn").value == "single_turn"
    assert modules.types.InteractionMode.parse(modules.types.InteractionMode.MULTI_TURN).value == "multi_turn"
    with pytest.raises(ValueError, match="interaction_mode"):
        modules.types.InteractionMode.parse("invalid")

    for values, message in (
        ({"policy_version": -1}, "policy_version"),
        ({"sample_index": -1}, "sample_index"),
        ({"max_turns": 0}, "max_turns"),
        ({"max_turns": 2, "interaction_mode": "single_turn"}, "single_turn"),
    ):
        with pytest.raises(ValueError, match=message):
            _episode(modules, **values)

    context = _episode(modules)
    assert context.prompt_id == "prompt-1"
    assert context.messages[-1].content == "What is 1 + 1?"
    assert context.ground_truth == "2"
    assert context.metadata["input_ids"].tolist() == [10, 11]
    assert context.encode_observation("result", metadata={"ok": True}).metadata["ok"]
    with pytest.raises(RuntimeError, match="observation encoder"):
        _episode(modules, observation_encoder=None).encode_observation("x")

    for index in (-1, 2):
        with pytest.raises(ValueError, match="Turn index"):
            modules.types.TurnContext(context, index, 0.0)
    turn = modules.types.TurnContext(context, 1, 0.5)
    assert turn.is_last_turn and turn.remaining_turns == 0

    transition = modules.types.TurnResult(
        modules.types.Observation("", torch.tensor([])),
        1,
        True,
        info={"reward_components": {"answer": 1}},
        termination_reason="completed",
    )
    assert transition.reward_result.components == {"answer": 1.0}
    with pytest.raises(ValueError, match="Unsupported termination"):
        modules.types.TurnResult(transition.observation, 0, False, termination_reason="bad")

    for call_id, name, arguments in (("", "tool", {}), ("id", "", {}), ("id", "tool", [])):
        with pytest.raises(ValueError):
            modules.types.ToolCall(call_id, name, arguments)


class _TemplateTokenizer:
    """Small chat-template tokenizer whose suffix depends on observation only."""

    def apply_chat_template(self, messages: Any, **_kwargs: Any) -> dict[str, list[int]]:
        result = [1]
        for message in messages:
            content = str(message["content"])
            result.extend((sum(map(ord, content)), 20 if message["role"] == "assistant" else 30))
        result.append(40)
        return {"input_ids": result}


def test_chat_template_encoder_lifecycle_and_validation() -> None:
    """Template encoding validates initial messages and action/observation order."""
    modules = _modules()
    with pytest.raises(ValueError, match="apply_chat_template"):
        modules.chat.TokenizerChatTemplateEncoder(object())
    encoder = modules.chat.TokenizerChatTemplateEncoder(_TemplateTokenizer())
    assert encoder._template_role("environment") == "user"
    for invalid in ("text", [], ["text"], [{"content": "x"}]):
        with pytest.raises(ValueError):
            encoder._normalize_initial_messages(invalid)

    initial = encoder(
        "question",
        "user",
        {modules.chat.CHAT_TEMPLATE_MESSAGES: [{"role": "system", "content": "s"},
                                               {"role": "user", "content": "question"}]},
    )
    assert initial.token_ids.ndim == 1 and initial.metadata["role"] == "user"
    with pytest.raises(ValueError, match="initial observation"):
        encoder("duplicate", "user", {modules.chat.CHAT_TEMPLATE_MESSAGES: [{"role": "user"}]})

    action = modules.types.AgentAction("answer", torch.tensor([5, 6]))
    encoder.record_action(action)
    with pytest.raises(RuntimeError, match="another action"):
        encoder.record_action(action)
    feedback = encoder("tool output", "environment", {})
    assert feedback.token_ids.numel() > 0 and feedback.metadata["role"] == "environment"

    fresh = modules.chat.TokenizerChatTemplateEncoder(_TemplateTokenizer())
    with pytest.raises(RuntimeError, match="before the initial"):
        fresh.record_action(action)
    with pytest.raises(RuntimeError, match="before recording"):
        fresh._render_observation_delta("x", "user")

    class BadTokenizer:
        def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"input_ids": None}

    with pytest.raises(ValueError, match="did not return"):
        modules.chat.TokenizerChatTemplateEncoder(BadTokenizer())("x", "user", {})

    class ResultTokenizer:
        def __init__(self, result: Any):
            self.result = result

        def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> Any:
            return self.result

    assert modules.chat.TokenizerChatTemplateEncoder(ResultTokenizer(torch.tensor([1])))._render([]) == [1]
    assert modules.chat.TokenizerChatTemplateEncoder(ResultTokenizer([[1, 2]]))._render([]) == [1, 2]
    with pytest.raises(ValueError, match="one token sequence"):
        modules.chat.TokenizerChatTemplateEncoder(ResultTokenizer([[1], [2]]))._render([])
    with pytest.raises(ValueError, match="flat integer"):
        modules.chat.TokenizerChatTemplateEncoder(ResultTokenizer(["bad"]))._render([])


def test_registry_protocol_and_schema_validation() -> None:
    """Tool registration, parsing, rendering, and schema errors are deterministic."""
    modules = _modules()
    for name, handler, parameters in (("bad name", lambda: None, {}), ("ok", 1, {}), ("ok", lambda: None, [])):
        with pytest.raises(ValueError):
            modules.registry.Tool(name, handler, parameters=parameters)

    registry = modules.registry.ToolRegistry()

    @registry.register("sum", description="add", parameters={"type": "object"})
    def add(left: int, right: int = 0) -> int:
        return left + right

    assert registry.names == ("sum",) and registry.get("sum").handler is add
    with pytest.raises(ValueError, match="already registered"):
        registry.register("sum")(lambda: None)
    with pytest.raises(ValueError, match="Unknown tool"):
        registry.get("missing")

    protocol = modules.protocol.JsonFunctionCallProtocol()
    context = modules.types.TurnContext(_episode(modules), 0, 0.0)
    action = modules.types.AgentAction(
        json.dumps({"tool_calls": [{"id": "c1", "function": {"name": "sum", "arguments": "{\"left\":2}"}}]}),
        torch.tensor([1]),
    )
    parsed = protocol.parse_action(action, context)
    assert parsed.tool_calls[0].arguments == {"left": 2}
    assert json.loads(protocol.format_tool_results(
        (modules.types.ToolResult("c1", "sum", "2"),), context
    ))["tool_results"][0]["content"] == "2"
    assert "interaction_error" in protocol.format_error("bad", context)
    assert protocol.parse_action(
        modules.types.AgentAction('{"final_answer":" 2 "}', torch.tensor([1])), context
    ).final_answer == "2"

    invalid_actions = ("not-json", "[]", "{}", '{"final_answer":""}', '{"tool_calls":[1]}')
    for content in invalid_actions:
        with pytest.raises(ValueError):
            protocol.parse_action(modules.types.AgentAction(content, torch.tensor([1])), context)
    for arguments in ("bad", "[]"):
        with pytest.raises(ValueError):
            protocol._arguments(arguments)
    with pytest.raises(ValueError):
        modules.protocol.ParsedAction()
    with pytest.raises(ValueError):
        modules.protocol.ParsedAction((modules.types.ToolCall("c", "sum", {}),), "answer")

    openai = modules.protocol.OpenAIToolCallProtocol()
    tool_message = {"role": "assistant", "tool_calls": [{"function": {"name": "sum", "arguments": {"left": 1}}}]}
    assert openai.parse_action(
        modules.types.AgentAction(json.dumps(tool_message), torch.tensor([1])), context
    ).tool_calls[0].name == "sum"
    assert openai.parse_action(
        modules.types.AgentAction('{"role":"assistant","content":" done "}', torch.tensor([1])), context
    ).final_answer == "done"
    with pytest.raises(ValueError, match="assistant message"):
        openai.parse_action(modules.types.AgentAction('{"role":"assistant"}', torch.tensor([1])), context)
    rendered = openai.format_tool_results((modules.types.ToolResult("c", "sum", "3"),), context)
    assert json.loads(rendered)["role"] == "tool"
    assert json.loads(openai.format_error("bad", context))["is_error"]
    assert isinstance(modules.protocol.build_json_function_call_protocol(), modules.protocol.JsonFunctionCallProtocol)
    assert isinstance(modules.protocol.build_openai_tool_call_protocol(), modules.protocol.OpenAIToolCallProtocol)


def test_tool_executor_success_errors_limits_and_timeout() -> None:
    """The executor handles sync/async tools, schema errors, limits, and closure."""
    modules = _modules()
    registry = modules.registry.ToolRegistry()
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    @registry.register("sync", parameters=schema)
    def sync_tool(value: int) -> dict[str, int]:
        return {"value": value}

    @registry.register("async")
    async def async_tool() -> str:
        await asyncio.sleep(0)
        return "async-result"

    @registry.register("failure")
    def failure() -> None:
        raise LookupError("failed")

    for kwargs in (
        {"timeout_seconds": True}, {"timeout_seconds": 0}, {"max_concurrency": True},
        {"max_concurrency": 0}, {"max_calls_per_turn": True}, {"max_calls_per_turn": 0},
    ):
        with pytest.raises(ValueError):
            modules.executor.ToolExecutor(registry, **kwargs)

    async def scenario() -> None:
        executor = modules.executor.ToolExecutor(registry, max_calls_per_turn=1)
        result = await executor.execute(modules.types.ToolCall("1", "sync", {"value": 3}))
        assert result.content == '{"value":3}' and not result.is_error
        assert (await executor.execute(modules.types.ToolCall("2", "async", {}))).content == "async-result"
        assert (await executor.execute(modules.types.ToolCall("3", "failure", {}))).is_error
        assert (await executor.execute(modules.types.ToolCall("4", "missing", {}))).is_error
        invalid = await executor.execute(modules.types.ToolCall("5", "sync", {"value": "bad"}))
        assert invalid.metadata["error_type"] == "ValueError"
        limited = await executor.execute_many((modules.types.ToolCall("6", "async", {}),
                                               modules.types.ToolCall("7", "async", {})))
        assert all(item.metadata["error_type"] == "ToolCallLimitError" for item in limited)
        await executor.close()
        assert (await executor.execute(modules.types.ToolCall("8", "async", {}))).is_error

        timed = modules.executor.ToolExecutor(registry, timeout_seconds=0.001)

        @registry.register("slow")
        async def slow() -> None:
            await asyncio.sleep(0.05)

        timeout_result = await timed.execute(modules.types.ToolCall("9", "slow", {}))
        assert "ToolTimeoutError" in timeout_result.content

    asyncio.run(scenario())

    tool = registry.get("sync")
    for bad_schema in (
        {"type": "array"}, {"required": "value"}, {"properties": []},
        {"properties": {"value": []}}, {"properties": {"value": {"type": "unsupported"}}},
    ):
        malformed = modules.registry.Tool("schema", tool.handler, parameters=bad_schema)
        with pytest.raises(ValueError):
            modules.executor.ToolExecutor._validate_arguments(malformed, {"value": 1})


def test_extension_loading_environment_and_mcp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Extension modules, ToolEnvironment, and MCP dispatch preserve tool semantics."""
    modules = _modules()
    with pytest.raises(ValueError):
        modules.environment.load_agentic_module("")
    assert modules.environment.load_agentic_module("json") is json
    with pytest.raises(RuntimeError, match="Failed to import"):
        modules.environment.load_agentic_module("missing_agentic_ut_module")
    extension = tmp_path / "extension.py"
    extension.write_text("VALUE = 7\n", encoding="utf-8")
    loaded = modules.environment.load_agentic_module(str(extension))
    assert loaded.VALUE == 7 and modules.environment.load_agentic_module(str(extension)) is loaded
    bad = tmp_path / "bad.py"
    bad.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Failed to load"):
        modules.environment.load_agentic_module(str(bad))
    with pytest.raises(ValueError, match="existing Python file"):
        modules.environment.load_agentic_module(str(tmp_path / "none.py"))

    registry = modules.registry.ToolRegistry()

    @registry.register("echo", description="echo text", parameters={"type": "object"})
    def echo(text: str = "ok") -> str:
        return text

    class Executor:
        closed = False

        async def execute_many(self, calls: Any) -> tuple[Any, ...]:
            return tuple(modules.types.ToolResult(call.call_id, call.name, "ok") for call in calls)

        async def close(self) -> None:
            self.closed = True

    protocol = modules.protocol.JsonFunctionCallProtocol()
    context = _episode(modules)
    executor = Executor()
    environment = modules.environment.ToolEnvironment(
        context, protocol, executor, lambda answer, _prompt: answer == "2"
    )

    async def environment_scenario() -> None:
        initial = await environment.reset(context)
        assert initial.token_ids.tolist() == [10, 11]
        turn = modules.types.TurnContext(context, 0, 0)
        invalid = await environment.step(modules.types.AgentAction("bad", torch.tensor([1])), turn)
        assert not invalid.done and "interaction_error" in invalid.info
        tool_action = modules.types.AgentAction(
            '{"tool_calls":[{"function":{"name":"echo","arguments":{}}}]}', torch.tensor([1])
        )
        tool_result = await environment.step(tool_action, turn)
        assert tool_result.info["tool_success_count"] == 1
        final = await environment.step(modules.types.AgentAction('{"final_answer":"2"}', torch.tensor([1])), turn)
        assert final.done and final.reward == 1.0
        await environment.close()
        assert executor.closed

    asyncio.run(environment_scenario())
    with pytest.raises(ValueError, match="tool_observation_role"):
        modules.environment.ToolEnvironment(context, protocol, executor, lambda *_args: 0, tool_observation_role="bad")
    with pytest.raises(ValueError, match="does not match"):
        environment._validate_episode(_episode(modules, sample_index=0))

    factory_module = ModuleType("agentic_ut_factory")
    factory_module.build = lambda settings: registry if settings == {"x": 1} else None
    monkeypatch.setitem(sys.modules, factory_module.__name__, factory_module)
    assert modules.mcp._load_registry("agentic_ut_factory:build", {"x": 1}) is registry
    for factory in ("invalid", "agentic_ut_factory:missing"):
        with pytest.raises(ValueError):
            modules.mcp._load_registry(factory, {})

    async def mcp_scenario() -> None:
        initialized = await modules.mcp._dispatch(registry, {"method": "initialize", "params": {}})
        assert initialized["protocolVersion"] == "2024-11-05"
        assert (await modules.mcp._dispatch(registry, {"method": "tools/list"}))["tools"][0]["name"] == "echo"
        called = await modules.mcp._dispatch(
            registry, {"method": "tools/call", "params": {"name": "echo", "arguments": {"text": "hi"}}}
        )
        assert called["content"][0]["text"] == "hi"
        failed = await modules.mcp._dispatch(
            registry, {"method": "tools/call", "params": {"name": "missing", "arguments": {}}}
        )
        assert failed["isError"]
        assert await modules.mcp._dispatch(registry, {"method": "ping"}) == {}
        assert await modules.mcp._dispatch(registry, {"method": "notifications/initialized"}) is None
        with pytest.raises(ValueError, match="Unsupported MCP"):
            await modules.mcp._dispatch(registry, {"method": "unknown"})

    asyncio.run(mcp_scenario())


def test_mcp_stdio_server_and_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stdio loop emits JSON-RPC results/errors and main validates settings."""
    modules = _modules()
    registry = modules.registry.ToolRegistry()

    @registry.register("async_echo")
    async def async_echo(text: str) -> dict[str, str]:
        return {"text": text}

    lines = b"".join(
        (
            b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"test"}}\n',
            b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n',
            b'{"jsonrpc":"2.0","id":2,"method":"tools/call",'
            b'"params":{"name":"async_echo","arguments":{"text":"hi"}}}\n',
            b'{"jsonrpc":"2.0","id":3,"method":"unsupported"}\n',
            b'[]\n',
        )
    )
    stdin = SimpleNamespace(buffer=io.BytesIO(lines))
    stdout = io.StringIO()
    monkeypatch.setattr(modules.mcp.sys, "stdin", stdin)
    monkeypatch.setattr(modules.mcp.sys, "stdout", stdout)
    asyncio.run(modules.mcp._serve(registry))
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert responses[0]["result"]["protocolVersion"] == "test"
    assert responses[1]["result"]["content"][0]["text"] == '{"text": "hi"}'
    assert responses[2]["error"]["code"] == -32603
    assert responses[3]["id"] is None

    factory = ModuleType("agentic_ut_main_factory")
    factory.build = lambda _settings: registry
    monkeypatch.setitem(sys.modules, factory.__name__, factory)
    served = []

    async def serve(selected: Any) -> None:
        served.append(selected)

    monkeypatch.setattr(modules.mcp, "_serve", serve)
    monkeypatch.setattr(
        modules.mcp.sys,
        "argv",
        ["mcp_server", "--factory", "agentic_ut_main_factory:build", "--settings-json", '{"x":1}'],
    )
    modules.mcp.main()
    assert served == [registry]
    monkeypatch.setattr(
        modules.mcp.sys,
        "argv",
        ["mcp_server", "--factory", "agentic_ut_main_factory:build", "--settings-json", "[]"],
    )
    with pytest.raises(ValueError, match="decode to an object"):
        modules.mcp.main()


class _HTTPResponse:
    """Expose a deterministic in-memory HTTP response for gateway tests."""
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self) -> "_HTTPResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


@pytest.mark.parametrize("harness_name", ["codex_harness", "deepseek_harness"])
def test_harness_http_helpers_and_runtime(
    harness_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both Harness HTTP boundaries normalize responses and runtime identity."""
    modules = _modules()
    harness = getattr(modules, harness_name)
    shared = importlib.import_module("rl.agentic.core.program_runner")
    monkeypatch.setattr(
        shared.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _HTTPResponse(b'{"ok":true}'),
    )
    assert harness._http_json("GET", "http://local", None, 1)["ok"]
    monkeypatch.setattr(
        shared.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _HTTPResponse(b"[]"),
    )
    with pytest.raises(RuntimeError, match="non-object"):
        harness._http_json("GET", "http://local", None, 1)
    monkeypatch.setattr(
        shared.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _HTTPResponse(b"bad"),
    )
    with pytest.raises(RuntimeError, match="invalid JSON"):
        harness._http_json("GET", "http://local", None, 1)
    monkeypatch.setattr(
        shared.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("down")),
    )
    with pytest.raises(RuntimeError, match="request failed"):
        harness._http_json("GET", "http://local", None, 1)

    events = []

    class Engine:
        """Record inference and lifecycle calls without starting a backend."""
        def __init__(self) -> None:
            self.errors = []

        @property
        def inference_base_url(self) -> str:
            events.append("backend")
            return "http://backend"

        @property
        def inference_model_name(self) -> str:
            events.append("model")
            return "qwen3"

        def synchronize_error(self, error: Any, operation: str) -> None:
            self.errors.append((error, operation))
            events.append("synchronize")

    engine = Engine()
    runtime_class = harness.CodexRuntime if harness_name == "codex_harness" else harness.DeepSeekRuntime
    gateway_class = harness.CodexGateway if harness_name == "codex_harness" else harness.DeepSeekGateway

    class Gateway:
        def __init__(self, **kwargs: Any):
            events.append(kwargs)

        def start(self) -> None:
            events.append("start")

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(harness, gateway_class.__name__, Gateway)
    monkeypatch.setattr(shared.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(shared.dist, "barrier", lambda: events.append("barrier"))
    runtime = runtime_class(engine, {})
    runtime.ensure_started()
    assert "start" in events and engine.errors[-1][0] is None
    assert events[:2] == ["backend", "model"]
    assert events.index("synchronize") < events.index("barrier")
    runtime.bind_episode_version(3)
    assert runtime.episode_version == 3
    runtime.clear_episode_version()
    with pytest.raises(RuntimeError, match="has not been established"):
        _ = runtime.episode_version
    with pytest.raises(ValueError, match="valid policy version"):
        runtime.bind_episode_version(-1)
    runtime.close()
    assert "close" in events

    events.clear()
    monkeypatch.setattr(shared.dist, "get_rank", lambda: 1)
    runtime_class(engine, {}).ensure_started()
    assert events == ["backend", "model", "synchronize", "barrier"]


@pytest.mark.parametrize("gateway_module", ["rl.agentic.codex.gateway", "rl.agentic.ds_harness.gateway"])
def test_gateway_session_state_lifecycle(
    gateway_module: str, tmp_path: Path
) -> None:
    """Gateway state validates, records, and releases policy-bound sessions."""
    gateway = importlib.import_module(gateway_module)
    state = gateway._State("http://backend/", "qwen3", 2.0)
    label = "DeepSeek" if "ds_harness" in gateway_module else "Codex"
    payload = {
        "policy_version": 3,
        "artifact_dir": str(tmp_path),
        "max_completions": 2,
        "generation": {"max_tokens": 10},
    }
    if label == "DeepSeek":
        payload["generation"]["reasoning_effort"] = "off"
    state.register("session", payload)
    assert state.get("session").policy_version == 3
    record = {"request": {}, "response": {}}
    state.save_completion("session", record)
    assert record["ordinal"] == 0
    event_lines = (tmp_path / "gateway-events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(event_lines) == 2
    with pytest.raises(ValueError, match="already exists"):
        state.register("session", payload)
    state.remove("session")
    with pytest.raises(ValueError, match=f"Unknown {label} session"):
        state.get("session")
    state.event("missing", "ignored", {})

    invalid_payloads = (
        ("", payload),
        ("other", {**payload, "policy_version": "invalid"}),
        ("other", {**payload, "max_completions": 0}),
        ("other", {**payload, "generation": []}),
        ("other", {**payload, "generation": {"unknown": True}}),
    )
    for session_id, invalid_payload in invalid_payloads:
        with pytest.raises(ValueError):
            state.register(session_id, invalid_payload)
    if label == "DeepSeek":
        invalid = {**payload, "generation": {"reasoning_effort": "medium"}}
        with pytest.raises(ValueError, match="reasoning_effort"):
            state.register("other", invalid)


def test_codex_gateway_reserves_final_completion_for_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bounded Codex session keeps its final response free of tool calls."""
    gateway = importlib.import_module("rl.agentic.codex.gateway")
    state = gateway._State("http://backend", "qwen3", 2.0)
    state.register("session", {"policy_version": 0, "artifact_dir": str(tmp_path),
                               "max_completions": 2, "generation": {}})
    requests = []

    def backend(unused_handler: Any, payload: dict[str, Any]) -> dict[str, Any]:
        del unused_handler
        requests.append(payload)
        return {"id": "response", "choices": [{"message": {"content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    monkeypatch.setattr(gateway._Handler, "_bearer_token", lambda _handler: "session")
    monkeypatch.setattr(gateway._Handler, "_backend_request", backend)
    monkeypatch.setattr(gateway._Handler, "_json", lambda *_args: None)
    handler = object.__new__(gateway._Handler)
    handler.server = SimpleNamespace(state=state)
    request = {"input": "calculate", "tools": [{"type": "local_shell", "description": "shell"}]}
    handler._proxy_responses(request)
    handler._proxy_responses(request)
    assert [payload["tool_choice"] for payload in requests] == ["auto", "none"]
    with pytest.raises(ValueError, match="max_completions"):
        handler._proxy_responses(request)


@pytest.mark.parametrize("gateway_module", ["rl.agentic.codex.gateway", "rl.agentic.ds_harness.gateway"])
def test_gateway_http_routes_and_lifecycle(gateway_module: str) -> None:
    """Both gateways expose deterministic health, session, and error routes."""
    gateway = importlib.import_module(gateway_module)
    is_deepseek = "ds_harness" in gateway_module
    gateway_class = gateway.DeepSeekGateway if is_deepseek else gateway.CodexGateway

    with pytest.raises(ValueError, match="host"):
        gateway_class("", 0, "http://backend", "qwen3", 1)
    with pytest.raises(ValueError, match="port"):
        gateway_class("127.0.0.1", 65536, "http://backend", "qwen3", 1)
    unopened = gateway_class("127.0.0.1", 0, "http://backend", "qwen3", 1)
    unopened.close()

    server = gateway_class("127.0.0.1", 0, "http://backend", "qwen3", 1)
    server.start()
    with pytest.raises(RuntimeError, match="already running"):
        server.start()
    host, port = server.address
    base_url = f"http://{host}:{port}"

    def request(
        method: str,
        path: str,
        payload: Any = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            base_url + path,
            data=data,
            headers={"Content-Type": "application/json", **(headers or {})},
            method=method,
        )
        try:
            response = urllib.request.urlopen(http_request, timeout=2)
        except HTTPError as error:
            return error.code, json.loads(error.read())
        with response:
            return response.status, json.loads(response.read())

    try:
        assert request("GET", "/healthz")[0] == 200
        assert request("GET", "/unknown")[0] == 404
        assert request("GET", "/internal/sessions/missing")[0] == 404
        assert request("POST", "/unknown", {})[0] == 404
        assert request("POST", "/internal/sessions", {})[0] == 400
        assert request("POST", "/internal/sessions", None)[0] == 400

        generation = {"max_tokens": 2}
        if is_deepseek:
            generation["reasoning_effort"] = "off"
        registration = {
            "session_id": "session",
            "policy_version": 2,
            "max_completions": 1,
            "generation": generation,
        }
        assert request("POST", "/internal/sessions", registration)[0] == 201
        assert request("GET", "/internal/sessions/session")[1]["policy_version"] == 2
        proxy_path = "/v1/chat/completions" if is_deepseek else "/v1/responses"
        assert request("POST", proxy_path, {"messages": []} if is_deepseek else {"input": "q"})[0] == 502
        assert request("DELETE", "/unknown")[0] == 404
        assert request("DELETE", "/internal/sessions/missing")[0] == 404
        assert request("DELETE", "/internal/sessions/session")[0] == 200
    finally:
        server.close()


def test_agent_session_guards_and_terminal_paths() -> None:
    """AgentSession validates token alignment, budgets, identity, and closure."""
    modules = _modules()
    session_module = importlib.import_module("rl.agentic.core.session")

    class Environment:
        """Provide controlled observations and transitions for lifecycle assertions."""
        def __init__(self, initial: Any = None, transition: Any = None):
            self.initial = initial or modules.types.Observation("question", torch.tensor([1]), {"role": "user"})
            self.transition = transition or modules.types.Transition(
                modules.types.Observation("done", torch.tensor([3]), {"role": "environment"}),
                1.0,
                True,
            )
            self.closed = 0

        async def reset(self, unused_context: Any) -> Any:
            del unused_context
            return self.initial

        async def step(self, unused_action: Any, unused_context: Any) -> Any:
            del unused_action, unused_context
            return self.transition

        async def close(self) -> None:
            self.closed += 1

    def make(environment: Any = None, **kwargs: Any) -> Any:
        return session_module.AgentSession(
            _prompt(), environment or Environment(), 2, 0, 2, **kwargs
        )

    for kwargs in ({"max_observation_tokens": -1}, {"max_episode_tokens": 0}):
        with pytest.raises(ValueError):
            make(**kwargs)
    with pytest.raises(ValueError, match="must match"):
        session_module.AgentSession(_prompt("other"), Environment(), 2, 0, 2, episode_context=_episode(modules))

    untouched = make()
    with pytest.raises(RuntimeError, match="not been started"):
        _ = untouched.token_ids
    with pytest.raises(RuntimeError, match="before reset"):
        untouched.result()
    with pytest.raises(RuntimeError, match="before reset"):
        untouched.build()
    with pytest.raises(RuntimeError, match="inactive"):
        _ = untouched.turn_context
    with pytest.raises(RuntimeError, match="inactive"):
        asyncio.run(untouched.apply(modules.types.Action("x", torch.tensor([2]))))
    assert untouched.remaining_token_budget is None

    empty = make(Environment(initial=modules.types.Observation("", torch.tensor([], dtype=torch.long))))
    with pytest.raises(ValueError, match="must not be empty"):
        asyncio.run(empty.start())
    ranked = make(Environment(initial=modules.types.Observation("bad", torch.tensor([[1]]))))
    with pytest.raises(ValueError, match="rank one"):
        asyncio.run(ranked.start())
    oversized = make(
        Environment(initial=modules.types.Observation("large", torch.tensor([1, 2]))),
        max_episode_tokens=1,
    )
    with pytest.raises(ValueError, match="Initial observation exceeds"):
        asyncio.run(oversized.start())

    invalid_role = make(Environment(initial=modules.types.Observation("x", torch.tensor([1]), {"role": "assistant"})))
    with pytest.raises(ValueError, match="Unsupported observation role"):
        asyncio.run(invalid_role.start())

    session = make(max_observation_tokens=1, max_episode_tokens=4)
    asyncio.run(session.start())
    assert session.remaining_token_budget == 3
    with pytest.raises(RuntimeError, match="only once"):
        asyncio.run(session.start())
    with pytest.raises(ValueError, match="rank-one"):
        session._append_action(modules.types.Action("bad", torch.tensor([[2]])))
    alignment = make()
    with pytest.raises(ValueError, match="align"):
        alignment._append_action(
            modules.types.Action("bad", torch.tensor([2]), torch.tensor([-0.1, -0.2]))
        )
    session._append_action(modules.types.Action("first", torch.tensor([2])))
    with pytest.raises(ValueError, match="consistently"):
        session._append_action(
            modules.types.Action("second", torch.tensor([2]), torch.tensor([-0.1]))
        )

    session.record_worker_policy_version(None)
    session.record_worker_policy_version(2)
    with pytest.raises(RuntimeError, match="multiple"):
        session.record_worker_policy_version(3)

    _assert_session_terminal_paths(make, Environment, modules)


def test_runner_boundary_validation() -> None:
    """AgentRunner rejects invalid limits, seeds, masks, and stale policies."""
    _modules()
    runner_module = importlib.import_module("rl.agentic.core.runner")
    rollout = importlib.import_module("rl.roles.rollout.base")
    settings = rollout.GenerationSettings(
        max_new_tokens=2,
        temperature=0,
        top_p=1,
        top_k=0,
        do_sample=False,
        pad_token_id=0,
        eos_token_id=2,
    )
    assert isinstance(runner_module._canonical_row_seed(1, "named", 0, 2), int)
    with pytest.raises(ValueError, match="non-negative"):
        runner_module._canonical_row_seed(1, "-1", 0, 2)
    for kwargs in (
        {"num_samples": 0, "max_turns": 1, "max_observation_tokens": 0},
        {"num_samples": 1, "max_turns": 0, "max_observation_tokens": 0},
        {"num_samples": 1, "max_turns": 1, "max_observation_tokens": -1},
        {"num_samples": 1, "max_turns": 1, "max_observation_tokens": 0, "max_episode_tokens": 0},
    ):
        with pytest.raises(ValueError):
            runner_module.AgentRunner(object(), object(), "env", settings=settings, **kwargs)
    runner = runner_module.AgentRunner(object(), object(), "env", 1, 1, 0, settings)
    with pytest.raises(ValueError, match="align"):
        runner._response_mask(torch.tensor([[1, 2]]), torch.tensor([[True]]))
    assert runner._response_mask(torch.tensor([[1, 2, 3]]), None).tolist() == [[True, True, False]]
    with pytest.raises(ValueError, match="at least one"):
        runner._populate_sessions([], [], 0)


def test_runner_complete_batched_rollout_and_helper_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AgentRunner drives active and dummy rows and validates engine boundaries."""
    modules = _modules()
    runner_module = importlib.import_module("rl.agentic.core.runner")
    rollout = importlib.import_module("rl.roles.rollout.base")

    class Tokenizer:
        def __call__(self, **_kwargs: Any) -> dict[str, Any]:
            return {"input_ids": torch.tensor([[41]])}

        def batch_decode(self, rows: list[list[int]], **_kwargs: Any) -> list[str]:
            return [f"answer-{row[0]}" for row in rows]

        def decode(self, row: list[int], **_kwargs: Any) -> str:
            return f"fallback-{row[0]}"

    class Environment:
        """Provide controlled observations and transitions for lifecycle assertions."""
        def __init__(self, context: Any):
            self.context = context
            self.closed = False

        async def reset(self, unused_context: Any) -> Any:
            del unused_context
            return modules.types.Observation(
                "question", torch.tensor([10, 11]), {"role": "user"}
            )

        async def step(self, unused_action: Any, turn: Any) -> Any:
            del unused_action
            done = self.context.sample_index == 0 or turn.turn_index == 1
            return modules.types.Transition(
                modules.types.Observation(
                    "feedback", torch.tensor([30 + turn.turn_index]), {"role": "environment"}
                ),
                1.0 if done else 0.0,
                done,
                termination_reason="completed" if done else None,
            )

        async def close(self) -> None:
            self.closed = True

    class Engine:
        """Record inference and lifecycle calls without starting a backend."""
        policy_version = 2

        def __init__(self) -> None:
            self.requests = []
            self.synchronized = []

        def generate(self, request: Any) -> Any:
            self.requests.append(request)
            response = torch.tensor([[5, 2], [6, 2]])
            return rollout.GenerationResult(
                torch.cat((request.input_ids, response), dim=1),
                torch.full(response.shape, -0.1),
                0.25,
                worker_policy_version=2,
            )

        def synchronize_error(self, error: Any, operation: str) -> None:
            self.synchronized.append((error, operation))
            if error is not None:
                raise error

    engine = Engine()
    settings = rollout.GenerationSettings(
        max_new_tokens=4,
        temperature=0,
        top_p=1,
        top_k=0,
        do_sample=False,
        pad_token_id=0,
        eos_token_id=2,
        collect_log_probs=True,
        seed=7,
    )
    runner = runner_module.AgentRunner(engine, Tokenizer(), "mock", 2, 2, 2, settings)
    monkeypatch.setattr(
        runner_module.ENVIRONMENTS,
        "build",
        lambda name, context: Environment(context) if name == "mock" else None,
    )
    captured = {}

    def build_batch(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(runner_module, "build_experience_batch", build_batch)
    batch = runner.rollout([_prompt()], 2)
    assert len(batch.trajectories) == 2
    assert batch.generation_seconds == 0.5
    assert batch.metadata["interaction_mode"] == "multi_turn"
    assert len(engine.requests) == 2 and engine.requests[0].row_seeds is not None
    assert batch.trajectories[0].done and batch.trajectories[1].done

    _assert_runner_helper_boundaries(runner)


def test_program_runner_owner_sibling_and_payload_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Program runner synchronizes owner results and validates sibling payloads."""
    program_runner = importlib.import_module("rl.agentic.core.program_runner")
    contracts = importlib.import_module("rl.dataset.contracts")
    rollout = importlib.import_module("rl.roles.rollout.base")
    settings = rollout.GenerationSettings(
        max_new_tokens=2,
        temperature=0,
        top_p=1,
        top_k=0,
        do_sample=False,
        pad_token_id=0,
        eos_token_id=2,
    )

    def trajectory(prompt_id: str = "prompt-1", policy_version: int = 2, with_logprobs: bool = True) -> Any:
        return contracts.Trajectory(
            trajectory_id="trajectory",
            prompt_id=prompt_id,
            group_id=prompt_id,
            policy_version=policy_version,
            turns=(contracts.Turn("user", "q", 0, 1, False), contracts.Turn("assistant", "a", 1, 2, True)),
            token_ids=torch.tensor([1, 2]),
            attention_mask=torch.tensor([True, True]),
            action_mask=torch.tensor([False, True]),
            rollout_log_probs=torch.tensor([-0.1]) if with_logprobs else None,
            reward=1,
            reward_components={"answer": 1},
            done=True,
            truncated=False,
            terminal_reason="completed",
            worker_policy_version=policy_version,
        )

    class Program:
        def __init__(self, result: Any):
            self.result = result

        async def run(self) -> Any:
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    with pytest.raises(ValueError, match="positive"):
        program_runner.ProgramAgentRunner(lambda *_args: Program(None), 0, settings)
    runner = program_runner.ProgramAgentRunner(lambda *_args: Program(trajectory()), 1, settings)
    with pytest.raises(ValueError, match="at least one"):
        runner.rollout([], 2)

    monkeypatch.setattr(program_runner, "build_experience_batch", lambda **kwargs: kwargs["trajectories"])
    assert runner.rollout([_prompt()], 2)[0].prompt_id == "prompt-1"

    failing = program_runner.ProgramAgentRunner(
        lambda *_args: Program(RuntimeError("failed")), 1, settings
    )
    with pytest.raises(RuntimeError, match="failed"):
        failing.rollout([_prompt()], 2)

    payload = runner._serialize_trajectories((trajectory(with_logprobs=False),))
    sibling_engine = SimpleNamespace(
        is_request_owner=False,
        synchronize_error=lambda error, _operation: error,
        synchronize_agent_payload=lambda _payload: payload,
    )
    sibling = program_runner.ProgramAgentRunner(lambda *_args: Program(None), 1, settings, sibling_engine)
    assert sibling.rollout([_prompt()], 2)[0].rollout_log_probs is None

    no_result = program_runner.ProgramAgentRunner(
        lambda *_args: Program(None), 1, settings, SimpleNamespace(is_request_owner=False)
    )
    with pytest.raises(RuntimeError, match="no trajectories"):
        no_result.rollout([_prompt()], 2)

    for bad in (trajectory("unknown"), trajectory(policy_version=3)):
        invalid = program_runner.ProgramAgentRunner(lambda *_args, result=bad: Program(result), 1, settings)
        with pytest.raises(ValueError):
            invalid.rollout([_prompt()], 2)

    with pytest.raises(ValueError, match="must be a list"):
        runner._deserialize_trajectories({}, [_prompt()])
    with pytest.raises(ValueError, match="unknown prompt"):
        runner._deserialize_trajectories([{}], [_prompt()])
    prompt_without_tokens = _prompt()
    prompt_without_tokens.metadata.pop("input_ids")
    with pytest.raises(ValueError, match="input_ids"):
        runner._deserialize_trajectories(payload, [prompt_without_tokens])
    restored = runner._deserialize_trajectories(
        runner._serialize_trajectories((trajectory(),)), [_prompt()]
    )
    assert restored[0].rollout_log_probs.tolist() == pytest.approx([-0.1])


def _program_config(tmp_path: Path, reward_callable: str) -> dict[str, Any]:
    return {
        "reward_callable": reward_callable,
        "max_turns": 2,
        "max_new_tokens": 16,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 0,
        "seed": 7,
        "session_root": str(tmp_path),
        "request_timeout": 1,
        "timeout_seconds": 1,
    }


def test_codex_program_configuration_and_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Codex program creates isolated state and consumes successful JSONL output."""
    modules = _modules()
    reward = _reward_module(monkeypatch)
    config = _program_config(tmp_path, reward)
    config["mcp_servers"] = [{"name": "tools", "command": "python", "args": ["-m", "server"],
                              "env": {"A": "B"}, "required": True}]
    program = modules.codex_harness.CodexAgentProgram(
        _prompt(), 2, 1, "http://127.0.0.1:8200/", config, 99
    )
    settings = program._generation_settings()
    assert settings["top_k"] == -1 and "seed" in settings
    artifact, workspace, home = program._prepare_directories("session")
    assert artifact.is_dir() and workspace.is_dir() and home.is_dir()
    program._write_codex_config(home, "session")
    assert "mcp_servers" in (home / "config.toml").read_text(encoding="utf-8")
    assert json.loads((home / "auth.json").read_text(encoding="utf-8"))["OPENAI_API_KEY"] == "session"

    class Stream:
        def __init__(self, lines: list[bytes]):
            self.lines = iter(lines)

        async def readline(self) -> bytes:
            return next(self.lines, b"")

    class Process:
        returncode = 0
        stdout = Stream([b'{"type":"item.completed","item":{"type":"agent_message","text":"2"}}\n'])
        stderr = Stream([b"diagnostic\n"])

        async def wait(self) -> int:
            return self.returncode

    async def create_process(*_args: Any, **_kwargs: Any) -> Process:
        return Process()

    monkeypatch.setattr(modules.codex_harness.asyncio, "create_subprocess_exec", create_process)
    answer, code, diagnostics = asyncio.run(program._run_codex("session", artifact, workspace, home))
    assert (answer, code, diagnostics) == ("2", 0, [])
    assert (artifact / "codex-events.jsonl").is_file()

    version_process = Process()
    version_process.stdout = None
    version_process.stderr = None

    async def communicate() -> tuple[bytes, bytes]:
        return b"codex-cli 0.152.1", b""

    version_process.communicate = communicate
    monkeypatch.setattr(
        modules.codex_harness.asyncio,
        "create_subprocess_exec",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=version_process),
    )
    asyncio.run(program._validate_version())


def test_deepseek_program_configuration_and_sdk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DeepSeek program pins settings and records a complete SDK result."""
    modules = _modules()
    reward = _reward_module(monkeypatch)
    config = _program_config(tmp_path, reward)
    program = modules.deepseek_harness.DeepSeekAgentProgram(
        _prompt(), 2, 1, "http://127.0.0.1:8300/v1/", "http://127.0.0.1:8300/", config, 99
    )
    assert program._generation_settings()["reasoning_effort"] == "off"
    artifact, workspace, sessions = program._prepare_directories("session")
    assert workspace.is_dir() and sessions.is_dir()

    @dataclass
    class Event:
        kind: str

    result = SimpleNamespace(
        events=[Event("message")],
        session_id="session",
        final_response="2",
        finish_reason="completed",
        notifications=[{"ok": True}],
    )

    class Harness:
        """Capture harness configuration without launching the external program."""
        def __init__(self, sdk_config: Any):
            self.sdk_config = sdk_config

        def __enter__(self) -> "Harness":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def run(self, instruction: str, session_id: str) -> Any:
            assert instruction and session_id == "session"
            return result

    class Config:
        def __init__(self, **kwargs: Any):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(modules.deepseek_harness, "_load_sdk", lambda _version: (Harness, Config))
    answer, reason, events = program._run_harness("session", artifact, workspace, sessions)
    assert answer == "2" and reason == "completed" and events == [{"kind": "message"}]
    assert (artifact / "deepseek-events.jsonl").is_file()
    assert "reasoningEffort" in (artifact / "dsh-home" / "settings.yaml").read_text(encoding="utf-8")
    assert json.loads((artifact / "deepseek-result.json").read_text(encoding="utf-8"))["final_response"] == "2"
    assert modules.deepseek_harness._json_value((Event("x"),)) == [{"kind": "x"}]


def test_codex_responses_protocol_complete_translation() -> None:
    """Responses translation preserves reasoning, messages, shell, and namespace tools."""
    protocol = importlib.import_module("rl.agentic.codex.protocol")
    translator = protocol.CodexResponsesProtocol()
    namespace = {
        "type": "namespace",
        "name": "math.tools",
        "tools": [{"type": "function", "name": "add", "description": "add"}],
    }
    tools = [namespace, {"type": "local_shell", "description": "shell"}]
    converted, aliases, reverse = translator._tools(tools)
    alias = aliases[("math.tools", "add")]
    assert len(converted) == 2 and reverse[alias] == ("math.tools", "add")

    encrypted = protocol._encrypt_reasoning("think")
    items = [
        {"type": "reasoning", "encrypted_content": encrypted},
        {"type": "function_call", "id": "f1", "name": "add", "namespace": "math.tools", "arguments": {"x": 1}},
        {"type": "function_call_output", "call_id": "f1", "output": "1"},
        {"type": "local_shell_call", "id": "s1", "action": {"commands": ["echo 1"], "timeout_ms": 5}},
        {"type": "local_shell_call_output", "call_id": "s1", "output": [{"type": "text", "text": "1"}]},
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "rule"}]},
        {"type": "input_text", "text": "question"},
    ]
    request = translator.transform_request(
        {
            "model": "policy",
            "instructions": "system",
            "input": items,
            "tools": tools,
            "tool_choice": {"type": "function", "namespace": "math.tools", "name": "add"},
            "max_output_tokens": 20,
            "temperature": 0.5,
            "reasoning": {"effort": "low"},
        },
        "qwen3",
    )
    assert request["model"] == "qwen3" and request["tool_choice"]["function"]["name"] == alias
    assert request["chat_template_kwargs"] == {"enable_thinking": True}

    response = translator.transform_response(
        {
            "id": "response",
            "choices": [{"message": {
                "content": [{"type": "text", "text": "done"}],
                "reasoning_content": "think",
                "tool_calls": [
                    {"id": "s2", "function": {"name": "shell", "arguments": '{"cmd":"pwd"}'}},
                    {"id": "f2", "function": {"name": alias, "arguments": "{}"}},
                ],
            }}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3},
        },
        {"model": "policy", "tools": tools},
    )
    assert response["usage"]["total_tokens"] == 7
    assert {item["type"] for item in response["output"]} == {
        "reasoning", "message", "local_shell_call", "function_call"
    }
    events = list(translator.stream_events(response))
    assert events[0]["type"] == "response.created" and events[-1]["type"] == "response.completed"
    assert any(event["type"] == "response.output_text.delta" for event in events)
    assert any(event["type"] == "response.function_call_arguments.done" for event in events)

    assert protocol._content_text(None) == ""
    assert protocol._content_text(3) == "3"
    assert protocol._reasoning_text({"summary": [{"text": "summary"}]}) == "summary"
    assert protocol._reasoning_text({"encrypted_content": "hyper-rl:bad"}) == ""
    with pytest.raises(ValueError):
        protocol._shell_arguments({"commands": []})
    with pytest.raises(ValueError):
        protocol._shell_action("bad")
    with pytest.raises(ValueError):
        translator.transform_request({"input": 1}, "qwen3")
    with pytest.raises(ValueError):
        translator._input_messages([{"type": "unknown"}], {})
    with pytest.raises(ValueError):
        translator._tool_choice({"type": "unknown"}, {})


def test_codex_responses_protocol_rejects_lossy_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Codex translation rejects every malformed message and tool boundary."""
    protocol = importlib.import_module("rl.agentic.codex.protocol")
    translator = protocol.CodexResponsesProtocol()

    assert protocol._reasoning_text({"content": [{"text": "reason"}]}) == "reason"
    assert protocol._reasoning_text({}) == ""
    assert protocol._reasoning_text(
        {"encrypted_content": protocol._encrypt_reasoning("private")}
    ) == "private"
    assert json.loads(protocol._shell_arguments({"commands": ["pwd"], "timeout_ms": 3}))["timeout_ms"] == 3
    for action in (None, {"commands": "pwd"}, {"commands": [1]}):
        with pytest.raises(ValueError):
            protocol._shell_arguments(action)
    with pytest.raises(ValueError, match="valid JSON"):
        protocol._shell_action("{")
    with pytest.raises(ValueError, match="omitted its command"):
        protocol._shell_action({})
    assert protocol._shell_action({"command": "pwd", "timeout_ms": 4})["timeout_ms"] == 4

    with pytest.raises(ValueError, match="JSON object"):
        translator.transform_request([], "qwen3")
    plain = translator.transform_request({"input": "hello", "tools": None}, "qwen3")
    assert plain["messages"] == [{"role": "user", "content": "hello"}]
    for response in ({}, {"choices": [1]}, {"choices": [{}]}):
        with pytest.raises(ValueError):
            translator.transform_response(response, {})
    with pytest.raises(ValueError, match="tool_calls"):
        translator.transform_response(
            {"choices": [{"message": {"tool_calls": {}}}]}, {}
        )
    empty = translator.transform_response(
        {"choices": [{"message": {}}], "usage": []}, {"model": "policy"}
    )
    assert empty["output"] == [] and empty["usage"]["total_tokens"] == 0

    invalid_items = (
        [1],
        [{"type": "message", "role": "tool"}],
        [{"type": "function_call_output"}],
    )
    for items in invalid_items:
        with pytest.raises(ValueError):
            translator._input_messages(items, {})
    reasoning_tail = translator._input_messages(
        [{"type": "reasoning", "summary": [{"text": "tail"}]}], {}
    )
    assert reasoning_tail[0]["reasoning"] == "tail"

    for item in (
        {"type": "function_call", "name": "tool"},
        {"type": "function_call", "id": "id"},
        {"type": "function_call", "id": "id", "name": "tool", "namespace": "ns"},
        {"type": "local_shell_call", "id": "id", "namespace": "ns", "action": {"commands": ["pwd"]}},
    ):
        with pytest.raises(ValueError):
            translator._chat_tool_call(
                item,
                local_shell=item["type"] == "local_shell_call",
                namespace_aliases={},
            )
    call = translator._chat_tool_call(
        {"id": "id", "name": "tool", "arguments": {"value": 1}}
    )
    assert json.loads(call["function"]["arguments"])["value"] == 1

    invalid_tools = (
        {},
        [1],
        [{"type": "namespace", "tools": [{}]}],
        [{"type": "namespace", "name": "ns", "tools": []}],
        [{"type": "namespace", "name": "ns", "tools": [1]}],
        [{"type": "namespace", "name": "ns", "tools": [{"type": "shell"}]}],
        [{"type": "namespace", "name": "ns", "tools": [{"type": "function"}]}],
        [{"type": "unsupported", "name": "x"}],
        [{"type": "function"}],
        [{"name": "same"}, {"name": "same"}],
    )
    for tools in invalid_tools:
        with pytest.raises(ValueError):
            translator._tools(tools)
    with pytest.raises(ValueError, match="duplicate tool"):
        translator._tools([
            {"type": "namespace", "name": "ns", "tools": [{"name": "x"}, {"name": "x"}]}
        ])
    monkeypatch.setattr(protocol.CodexResponsesProtocol, "_namespace_alias", lambda *_args: "collision")
    with pytest.raises(ValueError, match="duplicate name"):
        translator._tools([
            {"type": "namespace", "name": "ns", "tools": [{"name": "x"}, {"name": "y"}]}
        ])
    converted, _, _ = translator._tools([{"name": "strict", "strict": True}])
    assert converted[0]["function"]["strict"] is True

    assert translator._tool_choice({"type": "shell"}, {})["function"]["name"] == "shell"
    with pytest.raises(ValueError):
        translator._tool_choice(1, {})
    with pytest.raises(ValueError, match="unknown namespace"):
        translator._tool_choice({"type": "function", "namespace": "ns", "name": "x"}, {})
    with pytest.raises(ValueError, match="invalid tool call"):
        translator._response_tool_call([], {})
    shell = translator._response_tool_call(
        {"id": "id", "function": {"name": "shell", "arguments": '{"cmd":"pwd"}'}}, {}
    )
    assert shell["type"] == "local_shell_call"


def test_deepseek_protocol_reasoning_and_stream_validation() -> None:
    """DeepSeek protocol maps reasoning modes and validates streamed responses."""
    protocol = importlib.import_module("rl.agentic.ds_harness.protocol").DeepSeekChatProtocol()
    disabled = protocol.transform_request(
        {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "off"},
        "qwen3",
    )
    enabled = protocol.transform_request(
        {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "high"},
        "qwen3",
    )
    assert disabled["chat_template_kwargs"]["enable_thinking"] is False
    assert enabled["chat_template_kwargs"]["enable_thinking"] is True
    with pytest.raises(ValueError, match="non-empty messages"):
        protocol.transform_request({"messages": []}, "qwen3")
    with pytest.raises(ValueError, match="Unsupported DeepSeek"):
        protocol.transform_request(
            {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "medium"},
            "qwen3",
        )
    with pytest.raises(ValueError, match="first choice"):
        list(protocol.stream_events({"choices": []}))
    events = list(protocol.stream_events({
        "model": "qwen3",
        "choices": [{"message": {"content": "done", "tool_calls": []}, "finish_reason": None}],
        "usage": {"completion_tokens": 1},
    }))
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["usage"]["completion_tokens"] == 1
    with pytest.raises(ValueError, match="JSON object"):
        protocol.transform_request([], "qwen3")
    for effort in ("off", "high"):
        with pytest.raises(ValueError, match="chat_template_kwargs"):
            protocol.transform_request(
                {"messages": [{}], "reasoning_effort": effort, "chat_template_kwargs": []},
                "qwen3",
            )
    assert "chat_template_kwargs" not in protocol.transform_request({"messages": [{}]}, "qwen3")
    with pytest.raises(ValueError, match="assistant message"):
        list(protocol.stream_events({"choices": [{}]}))
    without_usage = list(protocol.stream_events({"choices": [{"message": {}}], "usage": []}))
    assert "usage" not in without_usage[-1]


@pytest.mark.parametrize(
    "module_name,label",
    [
        ("rl.agentic.codex.harness", "Codex"),
        ("rl.agentic.ds_harness.harness", "DeepSeek"),
    ],
)
def test_harness_trajectory_validation_branches(module_name: str, label: str) -> None:
    """Token-exact trajectory builders reject incomplete and rewritten traces."""
    trajectory = importlib.import_module(module_name)
    shared = importlib.import_module("rl.agentic.core.program_runner")

    def completion(
        prompt_ids: list[int],
        response_ids: list[int],
        *,
        finish_reason: str = "stop",
    ) -> dict[str, Any]:
        return {
            "ordinal": 0,
            "original_request": {},
            "request": {"messages": []},
            "response": {
                "prompt_token_ids": prompt_ids,
                "choices": [{
                    "message": {},
                    "finish_reason": finish_reason,
                    "token_ids": response_ids,
                    "logprobs": {"content": [
                        {"token_id": token_id, "logprob": -0.1}
                        for token_id in response_ids
                    ]},
                }],
            },
        }

    for value in (None, [], [object()]):
        with pytest.raises(ValueError):
            shared._int_list(value, label, "tokens")
    invalid_records = (
        {},
        {"request": {}, "response": {}},
        {"request": {}, "response": {"choices": [1]}},
    )
    for record in invalid_records:
        with pytest.raises(ValueError):
            shared._trace(record, label)

    fallback = completion([1], [2])
    choice = fallback["response"]["choices"][0]
    choice.pop("token_ids")
    assert shared._trace(fallback, label)["response_ids"] == [2]

    misaligned = completion([1], [2])
    misaligned["response"]["choices"][0]["logprobs"]["content"] = []
    with pytest.raises(ValueError, match="align"):
        shared._trace(misaligned, label)
    incomplete = completion([1], [2])
    incomplete["response"]["choices"][0]["logprobs"]["content"] = [{}]
    with pytest.raises(ValueError, match="incomplete"):
        shared._trace(incomplete, label)
    mismatched = completion([1], [2])
    mismatched["response"]["choices"][0]["logprobs"]["content"][0]["token_id"] = 3
    with pytest.raises(ValueError, match="differ"):
        shared._trace(mismatched, label)

    assert shared._end_of_turn_id([], 9, label) == 9
    with pytest.raises(ValueError, match="end-of-turn"):
        shared._end_of_turn_id(
            [{"response_ids": [2], "finish_reason": "length"}], None, label
        )
    assert shared._end_of_turn_id(
        [{"response_ids": [2], "finish_reason": "stop"}], None, label
    ) == 2
    max_rewrite = 0 if label == "Codex" else 16
    with pytest.raises(ValueError, match="end-of-turn boundary"):
        shared._interstitial([1, 3], [1], [2], 2, label, max_rewrite)
    assert shared._interstitial(
        [1, 2, 3], [1], [2], 2, label, max_rewrite
    ) == [3]
    assert shared._interstitial(
        [1, 2, 3], [1], [9], 2, label, max_rewrite
    ) == [2, 3]
    if label == "Codex":
        with pytest.raises(ValueError, match="rewrote"):
            shared._interstitial([9, 2], [1], [2], 2, label, max_rewrite)
    else:
        with pytest.raises(ValueError, match="rewrote"):
            shared._interstitial(
                [99], list(range(20)), [2], 2, label, max_rewrite
            )

    builder = trajectory.build_codex_trajectory if label == "Codex" else trajectory.build_deepseek_trajectory
    build_args = {
        "prompt": _prompt(),
        "policy_version": 2,
        "sample_index": 0,
        "reward": 1,
        "reward_components": {"answer": 1},
    }
    with pytest.raises(ValueError, match="at least one"):
        builder(completion_records=[], **build_args)
    with pytest.raises(ValueError, match="max_episode_tokens"):
        builder(
            completion_records=[completion([10, 11], [2])],
            max_episode_tokens=1,
            **build_args,
        )
    missing_tokens_prompt = _prompt()
    missing_tokens_prompt.metadata.pop("input_ids")
    with pytest.raises(ValueError, match="input_ids"):
        builder(
            completion_records=[completion([10, 11], [2])],
            **{**build_args, "prompt": missing_tokens_prompt},
        )


@pytest.mark.parametrize("harness_name", ["codex_harness", "deepseek_harness"])
def test_program_run_lifecycle(
    harness_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both programs register, score, convert, and delete one isolated session."""
    modules = _modules()
    harness = getattr(modules, harness_name)
    reward = _reward_module(monkeypatch, modules.types.RewardResult(1, {"answer": 1}))
    config = _program_config(tmp_path, reward)
    requests = []
    captured = {
        "policy_version": 2,
        "completions": [{"response": {"choices": []}}],
    }

    def http_json(method: str, url: str, payload: Any, timeout: float) -> dict[str, Any]:
        requests.append((method, url, payload, timeout))
        return captured if method == "GET" else {}

    monkeypatch.setattr(harness, "_http_json", http_json)
    trajectory = SimpleNamespace(reward=1.0)
    if harness_name == "codex_harness":
        program = harness.CodexAgentProgram(
            _prompt(), 2, 0, "http://gateway", config, 99
        )
        monkeypatch.setattr(program, "_write_codex_config", lambda *_args: None)

        async def validate() -> None:
            return None

        async def execute(*_args: Any) -> tuple[str, int, list[Any]]:
            return "2", 0, []

        monkeypatch.setattr(program, "_validate_version", validate)
        monkeypatch.setattr(program, "_run_codex", execute)
        monkeypatch.setattr(harness, "build_codex_trajectory", lambda **_kwargs: trajectory)
    else:
        program = harness.DeepSeekAgentProgram(
            _prompt(), 2, 0, "http://gateway/v1", "http://gateway", config, 99
        )
        monkeypatch.setattr(program, "_run_harness", lambda *_args: ("2", "completed", []))
        monkeypatch.setattr(program, "_capture_contract_error", lambda _captured: None)
        monkeypatch.setattr(harness, "build_deepseek_trajectory", lambda **_kwargs: trajectory)

    assert asyncio.run(program.run()) is trajectory
    assert [request[0] for request in requests] == ["POST", "GET", "DELETE"]
    if harness_name == "codex_harness":
        requests.clear()

        async def fail(*_args: Any) -> tuple[str, int, list[Any]]:
            raise RuntimeError("synthetic Codex failure")

        monkeypatch.setattr(program, "_run_codex", fail)
        with pytest.raises(RuntimeError, match="synthetic Codex failure"):
            asyncio.run(program.run())
        assert [request[0] for request in requests] == ["POST", "DELETE"]


def test_program_factories_verify_policy_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Codex and DeepSeek factories reject stale requested policy versions."""
    modules = _modules()
    reward = _reward_module(monkeypatch)
    for harness_name in ("codex_harness", "deepseek_harness"):
        harness = getattr(modules, harness_name)
        runtime = SimpleNamespace(
            config={"reward_callable": reward},
            episode_version=4,
            gateway_url="http://gateway/v1",
            admin_url="http://gateway",
        )
        factory_class = (
            harness.CodexProgramFactory
            if harness_name == "codex_harness"
            else harness.DeepSeekProgramFactory
        )
        factory = factory_class(runtime, 99, _program_config(Path("."), reward))
        with pytest.raises(RuntimeError, match="does not match"):
            factory(_prompt(), 3, 0)
        program = factory(_prompt(), 4, 0)
        assert program.policy_version == 4


def main() -> int:
    """Run Agentic tests and enforce separate line and branch thresholds."""
    try:
        # Coverage is required only by the standalone reporting entry point.
        from coverage import Coverage  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise RuntimeError(
            "agentic_ut requires coverage.py; install with 'python -m pip install coverage>=7,<8'"
        ) from error

    # Direct execution needs both the RL source root and the current checkout.
    for source_root in (_RL_SOURCE_ROOT, _HERE.parents[3]):
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))  # pylint: disable=sys-path-mutation
    coverage = Coverage(source=[str(_AGENTIC_ROOT)], branch=True, data_file=None)
    with tempfile.TemporaryDirectory(prefix=".agentic-ut-", dir=_HERE) as directory:
        coverage.start()
        status = pytest.main([
            "-q",
            "-p",
            "no:cacheprovider",
            f"--basetemp={Path(directory) / 'pytest'}",
            *(str(path) for path in _AGENTIC_TESTS),
        ])
        coverage.stop()
        coverage.report(show_missing=True)
        report_path = Path(directory) / "coverage.json"
        coverage.json_report(outfile=str(report_path))
        totals = json.loads(report_path.read_text(encoding="utf-8"))["totals"]
    line_coverage = float(totals["percent_statements_covered"])
    branch_coverage = float(totals["percent_branches_covered"])
    print(f"Agentic line coverage: {line_coverage:.2f}% (required: {_MINIMUM_COVERAGE:.2f}%)")
    print(f"Agentic branch coverage: {branch_coverage:.2f}% (required: {_MINIMUM_COVERAGE:.2f}%)")
    if status != pytest.ExitCode.OK:
        return int(status)
    thresholds_met = line_coverage >= _MINIMUM_COVERAGE and branch_coverage >= _MINIMUM_COVERAGE
    return 0 if thresholds_met else 1


if __name__ == "__main__":
    raise SystemExit(main())


def _assert_session_terminal_paths(make, environment_type, modules):
    """Exercise observation overflow, terminal reasons and idempotent closure."""
    active = make(
        environment_type(
            transition=modules.types.Transition(
                modules.types.Observation("too long", torch.tensor([3, 4])),
                0.5,
                False,
                info={"note": "kept", "tool_call_count": 1},
            )
        ),
        max_observation_tokens=1,
    )
    asyncio.run(active.start())
    with pytest.raises(ValueError, match="max_observation_tokens"):
        asyncio.run(active.apply(modules.types.Action("answer", torch.tensor([2]))))

    limited = make(
        environment_type(
            transition=modules.types.Transition(
                modules.types.Observation("overflow", torch.tensor([3, 4])), 0.25, False
            )
        ),
        max_episode_tokens=2,
    )
    asyncio.run(limited.start())
    asyncio.run(limited.apply(modules.types.Action("answer", torch.tensor([2]))))
    assert limited.truncated and limited.metadata["dropped_observation_tokens"] == 2
    assert limited.terminal_reason is modules.types.TerminationReason.CONTEXT_LIMIT

    terminal = make(
        environment_type(
            transition=modules.types.Transition(
                modules.types.Observation("overflow", torch.tensor([3, 4])),
                0.5,
                True,
                termination_reason="completed",
            )
        ),
        max_episode_tokens=2,
    )
    asyncio.run(terminal.start())
    asyncio.run(terminal.apply(modules.types.Action("answer", torch.tensor([2]))))
    assert terminal.done and not terminal.truncated
    assert terminal.result().reward.value == 0.5

    truncated = make(
        environment_type(
            transition=modules.types.Transition(
                modules.types.Observation("end", torch.tensor([3])), 0, False, truncated=True
            )
        )
    )
    asyncio.run(truncated.start())
    asyncio.run(truncated.apply(modules.types.Action("answer", torch.tensor([2]))))
    assert truncated.terminal_reason is modules.types.TerminationReason.ENVIRONMENT_TRUNCATED

    max_turns = make()
    asyncio.run(max_turns.start())
    max_turns.finish_max_turns()
    assert max_turns.terminal_reason is modules.types.TerminationReason.MAX_TURNS
    max_turns.finish_context_limit()
    max_turns.finish_max_turns()
    asyncio.run(max_turns.close())
    asyncio.run(max_turns.close())
    assert max_turns.environment.closed == 1

    over_turns = make()
    asyncio.run(over_turns.start())
    over_turns.action_contents.extend(["a", "b"])
    with pytest.raises(RuntimeError, match="max_turns"):
        asyncio.run(over_turns.apply(modules.types.Action("answer", torch.tensor([2]))))


def _assert_runner_helper_boundaries(runner):
    """Check masks, token budgets, malformed generations and propagated errors."""
    response = torch.tensor([[1, 2, 3]])
    explicit = torch.tensor([[True, True, True]])
    ignored = replace(runner.settings, ignore_eos=True)
    runner.settings = ignored
    assert runner._response_mask(response, None).all()
    assert runner._response_mask(response, explicit).all()
    runner.settings = replace(ignored, ignore_eos=False, eos_token_ids=(2, 3))
    assert runner._response_mask(response, explicit).tolist() == [[True, True, False]]

    class Session:
        """Represent session activity and remaining budgets for runner tests."""
        def __init__(self, active: bool, remaining: int | None, turn_count: int):
            self.active = active
            self.remaining_token_budget = remaining
            self.turn_count = turn_count
            self.prompt = SimpleNamespace(prompt_id="budget")
            self.max_episode_tokens = 2
            self.finished = False

        def finish_context_limit(self) -> None:
            self.active = False
            self.finished = True

    no_active = Session(False, 0, 1)
    assert runner._settings_for_turn([no_active]).max_new_tokens == 1
    unlimited = Session(True, None, 0)
    assert runner._settings_for_turn([unlimited]) is runner.settings
    finite = Session(True, 2, 0)
    assert runner._settings_for_turn([finite]).max_new_tokens == 2
    exhausted = Session(True, 0, 1)
    assert runner._settings_for_turn([exhausted]).max_new_tokens == 1
    assert exhausted.finished
    with pytest.raises(ValueError, match="no token budget"):
        runner._settings_for_turn([Session(True, 0, 0)])

    valid = SimpleNamespace(
        sequences=torch.tensor([[1, 2]]),
        rollout_log_probs=torch.tensor([[-0.1]]),
        response_mask=None,
    )
    for sequences in (torch.tensor([1, 2]), torch.tensor([[1, 2], [1, 2]]), torch.tensor([[1]])):
        with pytest.raises(ValueError, match="sequences must have shape"):
            runner._validate_generation_result(
                SimpleNamespace(sequences=sequences, rollout_log_probs=None, response_mask=None),
                1,
                2,
            )
    with pytest.raises(ValueError, match="log-probabilities"):
        runner._validate_generation_result(
            SimpleNamespace(sequences=valid.sequences, rollout_log_probs=None, response_mask=None),
            1,
            1,
        )
    with pytest.raises(ValueError, match="align"):
        runner._validate_generation_result(
            SimpleNamespace(
                sequences=valid.sequences,
                rollout_log_probs=torch.tensor([[-0.1, -0.2]]),
                response_mask=None,
            ),
            1,
            1,
        )

    runner.engine = object()
    runner._synchronize_error(None, "none")
    with pytest.raises(LookupError, match="local"):
        runner._synchronize_error(LookupError("local"), "failure")

    async def failure() -> None:
        raise ValueError("bad session")

    with pytest.raises(RuntimeError, match="session operation failed"):
        asyncio.run(runner._gather_session_operations("session operation", failure()))
