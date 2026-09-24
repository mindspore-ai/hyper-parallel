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
"""CPU contracts for tool attribution, gateway admission and Responses tools."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import threading
from types import SimpleNamespace
from typing import Any
import urllib.error
import urllib.request

import pytest

from rl.agentic.codex.gateway import CodexGateway, _Handler, _State
from rl.agentic.codex.protocol import CodexResponsesProtocol
from rl.agentic.ds_harness import gateway as ds_gateway
from rl.tool_protocol import inspect_tool_response, validate_trainability




def _response(raw: str, parsed: list) -> dict:
    """Build immutable engine/parser evidence with explicit sampled token identity."""
    return {"id": "completion", "model": "test", "created": 1, "prompt_token_ids": [9], "choices": [{
        "index": 0, "token_ids": [1, 2], "finish_reason": "tool_calls" if parsed else "stop",
        "message": {"role": "assistant", "content": raw if not parsed else None, "tool_calls": parsed},
        "logprobs": {"content": [{"token": "a", "logprob": -0.1}, {"token": "b", "logprob": -0.2}]},
    }], "hyper_tool_protocol": [{"parser_input": raw, "engine_text": raw, "decoded_tokens": raw,
                                  "token_ids": [1, 2], "parser_result": {"tool_calls": parsed}}]}


def test_additional_tools_share_namespace_mapping() -> None:
    """Inline tool declarations use the same reversible aliases as standard tools."""
    protocol = CodexResponsesProtocol()
    body = {"tools": [], "input": [{"type": "additional_tools", "tools": [{
        "type": "namespace", "name": "extra", "tools": [{"type": "function", "name": "lookup",
                                                          "parameters": {"type": "object"}}],
    }]}, {"type": "message", "role": "user", "content": "question"}]}
    request = protocol.transform_request(body, "test")
    assert request["messages"] == [{"role": "user", "content": "question"}]
    name = request["tools"][0]["function"]["name"]
    response = _response("", [{"id": "call1", "type": "function", "function": {"name": name, "arguments": "{}"}}])
    converted = protocol.transform_response(response, body)
    assert converted["output"][0]["namespace"] == "extra"
    assert converted["output"][0]["name"] == "lookup"
    body["input"][0]["tools"] = {}
    with pytest.raises(ValueError, match="additional_tools"):
        protocol.transform_request(body, "test")


@pytest.mark.parametrize("failure,origin,trainable", [
    ("valid", None, True), ("json", "model", True), ("schema", "model", True),
    ("parser", "infrastructure", False), ("missing", "unknown", False),
    ("parsed_without_evidence", "unknown", False), ("tokens", "unknown", False),
])
def test_tool_failure_requires_matching_immutable_evidence(failure: str, origin: Any, trainable: bool) -> None:
    """Model format errors remain trainable; missing or conflicting parser evidence does not."""
    raw = '<tool_call>{"name":"lookup","arguments":{"q":"hello"}}</tool_call>'
    parsed = [{"function": {"name": "lookup", "arguments": '{"q":"hello"}'}}]
    if failure == "json":
        raw, parsed = '<tool_call>{"name":</tool_call>', []
    elif failure == "schema":
        raw, parsed = '<tool_call>{"name":"lookup","arguments":[]}</tool_call>', []
    response = _response(raw, parsed)
    if failure == "parser":
        response["choices"][0]["message"]["tool_calls"] = []
    elif failure == "missing":
        response.pop("hyper_tool_protocol")
        response["choices"][0]["message"] = {"content": raw}
    elif failure == "parsed_without_evidence":
        response.pop("hyper_tool_protocol")
    elif failure == "tokens":
        response["hyper_tool_protocol"][0]["token_ids"] = [3]
    original = deepcopy(response)
    outcome = inspect_tool_response(response)
    assert outcome["failure_origin"] == origin and outcome["trainable"] is trainable
    assert response == original
    trajectory = SimpleNamespace(metadata={"gateway_records": [{"metadata": outcome}]})
    if trainable:
        validate_trainability([trajectory])
    else:
        with pytest.raises(ValueError, match="Untrainable"):
            validate_trainability([SimpleNamespace(metadata={}), trajectory])


def test_gateway_reservations_and_session_drain() -> None:
    """Closing waits for the full handler and cannot race a pending completion."""
    state = _State("http://unused", "test", 2)
    state.register("s", {"policy_version": 0, "max_completions": 1})
    session = state.begin_request("s")
    state.reserve_completion("s")
    with pytest.raises(ValueError, match="max_completions"):
        state.reserve_completion("s")
    state.release_completion(session)
    state.reserve_completion("s")
    state.release_completion(session)
    entered = threading.Event()

    def remove() -> None:
        """Signal that a concurrent close has begun, then wait for the owner."""
        entered.set()
        state.remove("s")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(remove)
        assert entered.wait(1)
        with state.idle:
            assert state.idle.wait_for(lambda: session.closing, timeout=1)
        assert not future.done()
        with pytest.raises(ValueError, match="closing"):
            state.begin_request("s")
        state.finish_request(session)
        future.result(timeout=2)
    assert "s" not in state.sessions


def test_terminal_snapshot_seals_admission_and_keeps_inflight_completion() -> None:
    """A terminal GET cannot omit a late completion or admit work before the following DELETE."""
    state = _State("http://unused", "test", 2)
    state.register("s", {"policy_version": 0, "max_completions": 2})
    session = state.begin_request("s")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(state.snapshot, "s")
        with state.idle:
            assert state.idle.wait_for(lambda: session.closing, timeout=1)
        assert not future.done()
        with pytest.raises(ValueError, match="closing"):
            state.begin_request("s")
        # Already admitted handlers may finish their bounded call budget after sealing.
        state.reserve_completion("s")
        with state.lock:
            state.save_completion("s", {"response": "inflight result"})
            state.release_completion(session)
        state.finish_request(session)
        captured = future.result(timeout=2)
    assert captured["completions"] == [{"response": "inflight result", "ordinal": 0}]
    with pytest.raises(ValueError, match="closing"):
        state.begin_request("s")
    state.remove("s")


@pytest.mark.parametrize("origins", [("model", "infrastructure"), ("infrastructure", "model"), ("unknown", "model")])
@pytest.mark.parametrize("state_type", [_State, ds_gateway._State])
def test_untrainable_failure_cannot_be_overwritten(origins: tuple, state_type: Any) -> None:
    """Either completion order preserves errors from calls that had no captured response."""
    state = state_type("http://unused", "test", 2)
    state.register("s", {"policy_version": 0, "max_completions": 2})
    session = state.get("s")
    for origin in origins:
        state.record_failure(session, {"failure_origin": origin, "failure_reason": "failure",
                                       "trainable": origin == "model"})
    assert session.failure["trainable"] is False
    assert session.failure["failure_origin"] in ("infrastructure", "unknown")


def test_deepseek_budget_is_atomic_and_failure_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two same-session calls cannot bypass the model budget or conceal its exact origin."""
    state = ds_gateway._State("http://unused", "test", 2)
    state.register("s", {"policy_version": 0, "max_completions": 1})
    handler = object.__new__(ds_gateway._Handler)
    handler.server = SimpleNamespace(state=state)
    handler.headers = {"x-deepseek-harness-session-id": "s"}
    entered, release = threading.Event(), threading.Event()
    calls = []

    def backend(unused_handler: Any, payload: dict) -> dict:
        """Keep the first backend request in flight while the second waits."""
        del unused_handler
        calls.append(payload)
        entered.set()
        assert release.wait(2)
        return {"choices": [{"message": {"content": "answer"}}]}

    monkeypatch.setattr(ds_gateway._Handler, "_backend_request", backend)
    monkeypatch.setattr(ds_gateway._Handler, "_json", lambda *_args, **_kwargs: None)
    body = {"messages": [{"role": "user", "content": "question"}]}
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(handler._proxy_chat, body)
        assert entered.wait(1)
        second = pool.submit(handler._proxy_chat, body)
        release.set()
        first.result(timeout=2)
        with pytest.raises(ValueError, match="max_completions"):
            second.result(timeout=2)
    captured = state.snapshot("s")
    assert len(calls) == len(captured["completions"]) == 1
    assert captured["failure"] == {
        "failure_origin": "model", "failure_reason": "call_budget_exhausted", "trainable": True,
    }
    with pytest.raises(ValueError, match="sealed"):
        handler._proxy_chat(body)


@pytest.mark.parametrize("failure", ["backend", "protocol"])
def test_deepseek_transport_and_protocol_errors_are_not_model_failures(
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """Captured terminal failure records untrainable errors even without a completion."""
    state = ds_gateway._State("http://unused", "test", 2)
    state.register("s", {"policy_version": 0, "max_completions": 1})
    handler = object.__new__(ds_gateway._Handler)
    handler.server = SimpleNamespace(state=state)
    handler.headers = {"x-deepseek-harness-session-id": "s"}

    def fail_backend(unused_handler: Any, unused_payload: dict) -> None:
        """Represent a failed remote execution with no sampled action."""
        del unused_handler, unused_payload
        raise RuntimeError("backend failed")

    monkeypatch.setattr(ds_gateway._Handler, "_backend_request", fail_backend)
    body = {"messages": [{"role": "user", "content": "question"}]} if failure == "backend" else {}
    with pytest.raises((ValueError, RuntimeError)):
        handler._proxy_chat(body)
    captured = state.snapshot("s")
    assert not captured["completions"]
    assert captured["failure"]["failure_origin"] == "infrastructure"
    assert captured["failure"]["trainable"] is False


@pytest.mark.parametrize("failure,origin", [("json", "model"), ("parser", "infrastructure"), ("evidence", "unknown")])
def test_deepseek_records_tool_attribution_without_resampling(
    monkeypatch: pytest.MonkeyPatch, failure: str, origin: str,
) -> None:
    """DeepSeek retains rejected raw actions and terminates with the same evidence classification."""
    state = ds_gateway._State("http://unused", "test", 2)
    state.register("s", {"policy_version": 0, "max_completions": 3})
    handler = object.__new__(ds_gateway._Handler)
    handler.server = SimpleNamespace(state=state)
    handler.headers = {"x-deepseek-harness-session-id": "s"}
    raw = '<tool_call>{"name":</tool_call>' if failure == "json" else (
        '<tool_call>{"name":"lookup","arguments":{}}</tool_call>')
    response = _response(raw, [])
    if failure == "evidence":
        response.pop("hyper_tool_protocol")
    monkeypatch.setattr(ds_gateway._Handler, "_backend_request", lambda *_args: deepcopy(response))
    with pytest.raises(RuntimeError):
        handler._proxy_chat({"messages": [{"role": "user", "content": "question"}]})
    captured = state.snapshot("s")
    assert len(captured["completions"]) == 1
    assert captured["failure"]["failure_origin"] == origin
    assert captured["failure"]["trainable"] is (origin == "model")
    record = captured["completions"][0]
    assert record["metadata"]["failure_origin"] == origin
    assert record["response"] == response


def test_backend_admission_settles_and_releases_failed_slots() -> None:
    """Multiple sessions share bounded backend admission and recover failed reservations."""
    state = _State("http://unused", "test", 2, max_inflight_requests=1)
    state.register("a", {"policy_version": 0, "max_completions": 2})
    state.register("b", {"policy_version": 0, "max_completions": 2})
    entered, release = threading.Event(), threading.Event()
    calls = []

    def backend(payload: dict) -> dict:
        """Block the first admitted request while another waits for its slot."""
        calls.append(payload["session"])
        entered.set()
        assert release.wait(2)
        if payload.get("fail"):
            raise RuntimeError("backend failed")
        return _response("answer", [])

    handler = SimpleNamespace(server=SimpleNamespace(state=state), _backend_request=backend)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_Handler._model_completion, handler, "a", {}, {"session": "a", "fail": True})
        assert entered.wait(1)
        second = pool.submit(_Handler._model_completion, handler, "b", {}, {"session": "b"})
        assert calls == ["a"]
        release.set()
        with pytest.raises(RuntimeError, match="backend failed"):
            first.result(timeout=2)
        second.result(timeout=2)
    assert state.get("a").reserved_completions == state.get("b").reserved_completions == 0
    assert len(state.get("b").completions) == 1
    state.reserve_completion("a")
    state.release_completion(state.get("a"))


def test_backend_admission_timeout_releases_reservation() -> None:
    """A queued request has a deadline and never consumes a model-call slot on timeout."""
    state = _State("http://unused", "test", 0.01)
    state.register("s", {"policy_version": 0, "max_completions": 1})
    handler = SimpleNamespace(server=SimpleNamespace(state=state))
    with state.backend_slots:
        with pytest.raises(RuntimeError, match="backend admission"):
            _Handler._model_completion(handler, "s", {}, {})
    assert state.get("s").reserved_completions == 0
    assert not state.get("s").completions


def test_gateway_keeps_rejected_calls_and_reports_terminal_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real HTTP gateway records every model call and preserves its terminal attribution."""
    response = _response('<tool_call>{"name":</tool_call>', [])
    monkeypatch.setattr(_Handler, "_backend_request", lambda _self, _payload: deepcopy(response))
    gateway = CodexGateway("127.0.0.1", 0, "http://unused", "test", 2)
    gateway.start()
    address = f"http://127.0.0.1:{gateway.address[1]}"

    def request(path: str, body: dict) -> dict:
        """Issue one authenticated local HTTP request."""
        req = urllib.request.Request(address + path, json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer s", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as result:
            return json.load(result)

    try:
        request("/internal/sessions", {"session_id": "s", "policy_version": 0, "max_completions": 2})
        with pytest.raises(urllib.error.HTTPError) as raised:
            request("/v1/responses", {"input": "question"})
        error = json.loads(raised.value.read())["error"]
        assert error["failure_origin"] == "model" and error["trainable"] is True
        with urllib.request.urlopen(address + "/internal/sessions/s", timeout=3) as result:
            captured = json.load(result)
        assert captured["failure"]["failure_reason"] == "tool_format_budget_exhausted"
        assert [record["ordinal"] for record in captured["completions"]] == [0, 1]
        assert all(record["response"]["choices"][0]["token_ids"] == [1, 2] for record in captured["completions"])
        assert "tool_choice" not in captured["completions"][-1]["request"]
    finally:
        gateway.close()
