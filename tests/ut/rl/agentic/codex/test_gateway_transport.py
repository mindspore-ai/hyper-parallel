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
"""Management isolation and bounded transport contracts for external agent gateways."""

from io import BytesIO
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
import urllib.error
import urllib.request

import pytest

from rl.agentic.codex import gateway as codex
from rl.agentic.codex.cli_contract import ISOLATED_COMPACT_PROMPT
from tests.ut.rl.agentic.codex.gateway_fixture import (
    gateway_http as _http, repository_request as _repository_request,
)


def test_legacy_text_gateway_does_not_gain_implicit_payload_limits() -> None:
    """Only explicitly configured limits constrain existing custom Codex programs."""
    gateway = codex.CodexGateway("127.0.0.1", 0, "http://unused", "test", 2)
    try:
        assert gateway._server.max_request_bytes is None
        assert gateway._server.max_response_bytes is None
        assert not gateway._server.require_admin_auth
    finally:
        gateway.close()
    handler = _handler()
    handler.server.max_request_bytes = None
    handler.server.max_response_bytes = None
    payload = b"x" * 100000
    assert handler._read_body(BytesIO(payload), None) == payload


def test_management_auth_precedes_body_and_cleanup_blocks_late_registration() -> None:
    """A candidate cannot inspect, release, register, or fail sessions using its model token."""
    gateway = codex.CodexGateway("127.0.0.1", 0, "http://unused", "test", 2, admin_token="test-controller")
    gateway.start()
    try:
        for method, path in [("GET", "/internal/sessions/s"), ("DELETE", "/internal/sessions/s"),
                             ("POST", "/internal/sessions"), ("POST", "/internal/sessions/s/failure")]:
            for token in (None, "s"):
                status, body = _http(gateway, method, path, token, b"invalid JSON")
                assert status == 401
                assert gateway.admin_token not in json.dumps(body)
        payload = json.dumps({"session_id": "s", "policy_version": 2, "max_completions": 2}).encode()
        assert _http(gateway, "POST", "/internal/sessions", gateway.admin_token, payload)[0] == 201
        assert _http(gateway, "GET", "/internal/sessions/s", "s")[0] == 401
        assert _http(gateway, "POST", "/internal/sessions/s/failure", gateway.admin_token, b"{}")[0] == 200
        status, snapshot = _http(gateway, "GET", "/internal/sessions/s", gateway.admin_token)
        assert status == 200 and snapshot["failure"]["trainable"] is False
        for session_id in ("s", "pending"):
            for _ in range(2):
                assert _http(gateway, "DELETE", "/internal/sessions/" + session_id, gateway.admin_token)[0] == 200
            late = json.dumps({"session_id": session_id, "policy_version": 2, "max_completions": 2}).encode()
            assert _http(gateway, "POST", "/internal/sessions", gateway.admin_token, late)[0] == 400
    finally:
        gateway.close()


@pytest.mark.parametrize("body", [b"{invalid", b"x" * 257])
def test_bad_model_body_invalidates_known_session(body: bytes) -> None:
    """Rejected bodies are infrastructure failures, never silently trainable empty traces."""
    gateway = codex.CodexGateway("127.0.0.1", 0, "http://unused", "test", 2,
                           max_request_bytes=256, admin_token="test-controller")
    gateway.start()
    try:
        payload = json.dumps({"session_id": "s", "policy_version": 1, "max_completions": 1}).encode()
        assert _http(gateway, "POST", "/internal/sessions", gateway.admin_token, payload)[0] == 201
        assert _http(gateway, "POST", "/v1/responses", "s", body)[0] == 502
        _, snapshot = _http(gateway, "GET", "/internal/sessions/s", gateway.admin_token)
        assert snapshot["completions"] == []
        assert snapshot["failure"]["failure_origin"] == "infrastructure"
        assert snapshot["failure"]["trainable"] is False
    finally:
        gateway.close()


def _handler() -> Any:
    """Construct the real request processing path with a controlled client transport."""
    handler = object.__new__(codex._Handler)
    state = codex._State("http://unused", "test", 2)
    state.register("s", {"policy_version": 3, "max_completions": 2})
    handler.server = SimpleNamespace(state=state, max_request_bytes=1024, max_response_bytes=4096)
    handler.headers = {"Authorization": "Bearer s"}
    handler.send_response = lambda *_args: None
    handler.send_header = lambda *_args: None
    handler.end_headers = lambda: None
    return handler


def test_router_capture_is_private_to_the_record_and_json_snapshot() -> None:
    """The Agent response carries no route payload while the training snapshot retains it once."""
    handler = _handler()
    handler.server.router_capture_spec = {"layer_ids": [2], "top_k": 2, "num_experts": 4}
    response = {"prompt_token_ids": [1, 2], "choices": [{"index": 0, "token_ids": [3, 9],
                "finish_reason": "stop", "message": {"role": "assistant", "content": "answer"},
                "routed_experts": "private-base64-npy"}]}
    handler._backend_request = lambda _request: response
    delivered, unused_outcome, record = handler._model_completion("s", {"input": "question"}, {})
    assert "routed_experts" not in delivered["choices"][0]
    assert "routed_experts" not in record["response"]["choices"][0]
    assert record["routed_experts"] == "private-base64-npy"
    assert record["router_capture_spec"] == handler.server.router_capture_spec
    snapshot = json.loads(json.dumps(handler.server.state.snapshot("s")))
    assert snapshot["completions"][0]["routed_experts"] == "private-base64-npy"
    assert json.dumps(snapshot).count("private-base64-npy") == 1


def _sampled_repository_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    """Tie one parsed tool action to immutable sampled token and parser evidence."""
    raw = '<tool_call>{"name":"exec_command","arguments":' + json.dumps(arguments) + '}</tool_call>'
    parsed = [{"id": "call-1", "type": "function", "function": {
        "name": "exec_command", "arguments": json.dumps(arguments),
    }}]
    return {"id": "sampled-tool", "model": "test", "prompt_token_ids": [9, 10],
            "choices": [{"index": 0, "token_ids": [11, 12], "finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": None, "tool_calls": parsed},
                         "logprobs": {"content": [{"token_id": 11, "logprob": -0.2},
                                                   {"token_id": 12, "logprob": -0.3}]}}],
            "hyper_tool_protocol": [{"parser_input": raw, "engine_text": raw, "decoded_tokens": raw,
                                     "token_ids": [11, 12], "parser_result": {"tool_calls": parsed}}]}


def test_repository_rejects_undeclared_stdin_and_resamples_with_cmd_heredoc_feedback() -> None:
    """A parser-verified rejected action receives cmd heredoc feedback before resampling."""
    handler = _handler()
    handler.server.state.get("s").isolated_cli = True
    handler.wfile = BytesIO()
    bad = _sampled_repository_tool({"cmd": "cat",
                                    "stdin": "*** Begin Patch\nsecret patch\n*** End Patch"})
    good = {"id": "final", "model": "test", "choices": [{"index": 0,
            "message": {"role": "assistant", "content": "Need a cmd heredoc."}, "finish_reason": "stop"}]}
    with patch.object(handler, "_backend_request", side_effect=[bad, good]) as backend:
        handler._proxy_responses(_repository_request())
    records = handler.server.state.snapshot("s")["completions"]
    assert backend.call_count == 2
    assert records[0]["response"] == bad
    assert records[0]["response"]["choices"][0]["token_ids"] == [11, 12]
    assert records[0]["metadata"]["failure_origin"] == "model"
    assert records[0]["metadata"]["failure_reason"] == "undeclared_tool_argument"
    assert records[0]["metadata"]["trainable"] is True
    assert records[0]["metadata"]["invalid_arguments"] == ["stdin"]
    assert records[1]["request"]["messages"][-2]["content"] == bad["hyper_tool_protocol"][0]["engine_text"]
    feedback = records[1]["request"]["messages"][-1]["content"]
    assert "stdin" in feedback and "cmd shell command" in feedback
    assert "heredoc" in feedback and "checked_patch.py" not in feedback
    assert "secret patch" not in feedback
    assert records[1]["metadata"]["failure_origin"] is None


@pytest.mark.parametrize("arguments,reason", [({}, "missing_tool_argument"),
                                                  ({"cmd": False}, "invalid_tool_argument_type")])
def test_repository_validates_required_and_typed_tool_arguments(arguments: dict, reason: str) -> None:
    """The forwarded closed schema also rejects missing or mistyped declared fields."""
    handler = _handler()
    session = handler.server.state.get("s")
    session.isolated_cli = True
    session.max_completions = 1
    bad = _sampled_repository_tool(arguments)
    with patch.object(handler, "_backend_request", return_value=bad), pytest.raises(codex.ToolProtocolFailure):
        handler._proxy_responses(_repository_request())
    snapshot = handler.server.state.snapshot("s")
    assert snapshot["completions"][0]["response"] == bad
    assert snapshot["completions"][0]["metadata"]["failure_reason"] == reason
    assert snapshot["failure"]["failure_reason"] == "tool_format_budget_exhausted"
    assert snapshot["failure"]["trainable"] is True


@pytest.mark.parametrize("repository", [False, True])
def test_repository_schema_validation_preserves_text_and_untrusted_evidence(repository: bool) -> None:
    """Text stays unchanged; missing token identity cannot become a model format error."""
    handler = _handler()
    handler.server.state.get("s").isolated_cli = repository
    handler.wfile = BytesIO()
    sampled = _sampled_repository_tool({"cmd": "true", "stdin": "ignored"})
    if repository:
        sampled["hyper_tool_protocol"][0]["token_ids"] = [99]
    with patch.object(handler, "_backend_request", return_value=sampled) as backend:
        if repository:
            with pytest.raises(codex.ToolProtocolFailure):
                handler._proxy_responses(_repository_request())
        else:
            handler._proxy_responses(_repository_request())
    assert backend.call_count == 1
    snapshot = handler.server.state.snapshot("s")
    metadata = snapshot["completions"][0]["metadata"]
    if repository:
        assert metadata["failure_origin"] == "unknown" and metadata["trainable"] is False
    else:
        assert metadata["failure_origin"] is None and metadata["trainable"] is True


@pytest.mark.parametrize("repository", [False, True])
def test_repository_context_corrected_only_for_model_and_preserves_original(repository: bool) -> None:
    """The real request keeps evidence while the backend sees only truthful Docker permissions."""
    handler = _handler()
    handler.server.state.get("s").isolated_cli = repository
    handler.wfile = BytesIO()
    request = _repository_request()
    original = json.loads(json.dumps(request))
    raw = {"id": "c", "model": "test", "choices": [{"index": 0,
           "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}]}
    with patch.object(handler, "_backend_request", return_value=raw) as backend:
        handler._proxy_responses(request)
    record = handler.server.state.snapshot("s")["completions"][0]
    assert request == original == record["original_request"]
    assert backend.call_args.args[0] == record["request"]
    system = record["request"]["messages"][0]["content"]
    environment = record["request"]["messages"][1]["content"]
    if repository:
        assert "network=none" in system and "No Docker socket or NPU" in system
        assert "a separate trusted grader" in system
        assert "Network access is enabled" not in system
        assert 'type="candidate-container"' in environment
        assert 'type="unrestricted"' not in environment
        original_exec = record["original_request"]["tools"][0]["parameters"]["properties"]
        visible_exec = record["request"]["tools"][0]["function"]
        assert "sandbox_permissions" in original_exec and "justification" in original_exec
        assert "sandbox_permissions" not in visible_exec["parameters"]["properties"]
        assert "justification" not in visible_exec["parameters"]["properties"]
        assert "prefix_rule" not in visible_exec["parameters"]["properties"]
        assert "checked_patch.py" not in visible_exec["description"]
    else:
        assert "Network access is enabled" in system
        assert 'type="unrestricted"' in environment
        assert "sandbox_permissions" in record["request"]["tools"][0]["function"]["parameters"]["properties"]
    assert record["request"]["messages"][-1]["content"] == "question"


@pytest.mark.parametrize("invalid", ["missing", "permissions", "environment"])
def test_repository_context_rejects_unrecognized_cli_blocks(invalid: str) -> None:
    """An unrecognized CLI injection is an infrastructure failure before sampling."""
    handler = _handler()
    handler.server.state.get("s").isolated_cli = True
    request = _repository_request()
    if invalid == "missing":
        request["input"].pop(0)
    elif invalid == "permissions":
        request["input"][0]["content"][0]["text"] += " unknown"
    else:
        request["input"][1]["content"][0]["text"] = request["input"][1]["content"][0]["text"].replace(
            'type="unrestricted"', 'type="unknown"'
        )
    with patch.object(handler, "_backend_request") as backend, pytest.raises(ValueError, match="Codex"):
        handler._proxy_responses(request)
    backend.assert_not_called()
    snapshot = handler.server.state.snapshot("s")
    assert snapshot["completions"] == []
    assert snapshot["failure"]["failure_origin"] == "infrastructure"
    assert snapshot["failure"]["trainable"] is False


def test_repository_context_rejects_extra_cli_tool() -> None:
    """CLI tool drift is an infrastructure error before the backend sees a request."""
    handler = _handler()
    handler.server.state.get("s").isolated_cli = True
    request = _repository_request()
    request["tools"].append({"type": "function", "name": "extra_tool", "description": "extra",
                             "parameters": {"type": "object", "properties": {}}})
    with patch.object(handler, "_backend_request") as backend, pytest.raises(ValueError, match="tool set"):
        handler._proxy_responses(request)
    backend.assert_not_called()
    snapshot = handler.server.state.snapshot("s")
    assert snapshot["completions"] == []
    assert snapshot["failure"]["failure_origin"] == "infrastructure"
    assert snapshot["failure"]["trainable"] is False


def test_repository_compaction_has_no_tools_and_preserves_sampled_request() -> None:
    """A pinned CLI summary request may run without tools while retaining truthful context."""
    handler = _handler()
    handler.server.state.get("s").isolated_cli = True
    handler.wfile = BytesIO()
    request = _repository_request()
    request["client_metadata"]["x-codex-turn-metadata"] = json.dumps({"request_kind": "compaction"})
    request["tools"] = []
    request["input"][-1]["content"] = [{"type": "input_text", "text": ISOLATED_COMPACT_PROMPT}]
    original = json.loads(json.dumps(request))
    raw = {"id": "summary", "model": "test", "choices": [{"index": 0,
           "message": {"role": "assistant", "content": "State summary"}, "finish_reason": "stop"}]}
    with patch.object(handler, "_backend_request", return_value=raw) as backend:
        handler._proxy_responses(request)
    record = handler.server.state.snapshot("s")["completions"][0]
    assert request == original == record["original_request"]
    assert backend.call_args.args[0] == record["request"]
    assert record["request"].get("tools") in (None, [])
    assert record["request"]["max_tokens"] == 1024
    assert "network=none" in record["request"]["messages"][0]["content"]
    assert 'type="candidate-container"' in record["request"]["messages"][1]["content"]


@pytest.mark.parametrize("limit", [None, 6000])
def test_repository_observation_limit_counts_characters_and_preserves_evidence(limit: int | None) -> None:
    """The optional reference budget affects model input, never the original tool evidence."""
    handler = _handler()
    session = handler.server.state.get("s")
    session.isolated_cli = True
    session.max_observation_chars = limit
    handler.wfile = BytesIO()
    request = _repository_request()
    output = "观察" * 4000
    request["input"].extend([
        {"type": "function_call", "call_id": "previous", "name": "exec_command", "arguments": '{"cmd":"pwd"}'},
        {"type": "function_call_output", "call_id": "previous", "output": output},
    ])
    original = json.loads(json.dumps(request))
    raw = _sampled_repository_tool({"cmd": "pwd"})
    with patch.object(handler, "_backend_request", return_value=raw) as backend:
        handler._proxy_responses(request)
    forwarded = backend.call_args.args[0]
    observation = next(message for message in forwarded["messages"] if message.get("role") == "tool")
    if limit is None:
        assert observation["content"] == output
    else:
        assert len(observation["content"]) == limit
        assert observation["content"].startswith(output[:100])
        assert observation["content"].endswith(output[-100:])
        assert "observation truncated" in observation["content"]
    assert request == original
    record = handler.server.state.snapshot("s")["completions"][0]
    assert record["original_request"] == original
    assert record["request"] == forwarded


def test_repository_commentary_and_tool_share_one_assistant_turn() -> None:
    """Responses output items must not introduce extra Chat turn boundaries."""
    handler = _handler()
    handler.server.state.get("s").isolated_cli = True
    handler.wfile = BytesIO()
    request = _repository_request()
    request["input"].extend([
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Read source."}]},
        {"type": "function_call", "call_id": "previous", "name": "exec_command", "arguments": '{"cmd":"pwd"}'},
        {"type": "function_call_output", "call_id": "previous", "output": "/workspace"},
    ])
    original = json.loads(json.dumps(request))
    with patch.object(handler, "_backend_request", return_value=_sampled_repository_tool({"cmd": "pwd"})) as backend:
        handler._proxy_responses(request)
    messages = backend.call_args.args[0]["messages"]
    assistants = [message for message in messages if message["role"] == "assistant"]
    assert len(assistants) == 1
    assert assistants[0]["content"] == "Read source."
    assert assistants[0]["tool_calls"][0]["id"] == "previous"
    assert request == original


@pytest.mark.parametrize("finish_reason,content", [("length", "incomplete"), ("stop", "")])
def test_repository_compaction_rejects_incomplete_summary(finish_reason: str, content: str) -> None:
    """Never replace CLI history with a truncated or empty summary."""
    handler = _handler()
    handler.server.state.get("s").isolated_cli = True
    request = _repository_request()
    request["client_metadata"]["x-codex-turn-metadata"] = json.dumps({"request_kind": "compaction"})
    request["tools"] = []
    request["input"][-1]["content"] = [{"type": "input_text", "text": ISOLATED_COMPACT_PROMPT}]
    raw = {"id": "summary", "model": "test", "prompt_token_ids": [10],
           "choices": [{"index": 0, "token_ids": [11], "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason}]}
    with patch.object(handler, "_backend_request", return_value=raw), pytest.raises(codex.ToolProtocolFailure):
        handler._proxy_responses(request)
    snapshot = handler.server.state.snapshot("s")
    assert len(snapshot["completions"]) == 1
    assert snapshot["completions"][0]["response"] == raw
    assert snapshot["failure"]["failure_origin"] == "infrastructure"
    assert snapshot["failure"]["trainable"] is False


@pytest.mark.parametrize("invalid", ["kind", "tools", "prompt"])
def test_repository_compaction_rejects_contract_drift(invalid: str) -> None:
    """A no-tool request is accepted only as the pinned CLI's summary operation."""
    handler = _handler()
    handler.server.state.get("s").isolated_cli = True
    request = _repository_request()
    request["client_metadata"]["x-codex-turn-metadata"] = json.dumps({"request_kind": "compaction"})
    request["tools"] = []
    request["input"][-1]["content"] = [{"type": "input_text", "text": ISOLATED_COMPACT_PROMPT}]
    if invalid == "kind":
        request["client_metadata"]["x-codex-turn-metadata"] = json.dumps({"request_kind": "turn"})
    elif invalid == "tools":
        request["tools"] = _repository_request()["tools"]
    else:
        request["input"][-1]["content"][0]["text"] += " changed"
    with patch.object(handler, "_backend_request") as backend, pytest.raises(ValueError, match="Codex"):
        handler._proxy_responses(request)
    backend.assert_not_called()
    snapshot = handler.server.state.snapshot("s")
    assert snapshot["completions"] == []
    assert snapshot["failure"]["failure_origin"] == "infrastructure"


@pytest.mark.parametrize("failure", [BrokenPipeError, ConnectionResetError])
def test_sse_disconnect_retains_completion_and_rejects_episode(failure: type) -> None:
    """A recorded sample cannot be trained after failure to deliver its SSE stream."""
    handler = _handler()
    raw = {"id": "c", "model": "test", "choices": [{"index": 0,
           "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}]}
    stream = SimpleNamespace(write=lambda _data: (_ for _ in ()).throw(failure("disconnect")), flush=lambda: None)
    handler.wfile = stream
    handler._backend_request = lambda _payload: raw
    with pytest.raises(failure):
        handler._proxy_responses({"input": "question", "stream": True})
    snapshot = handler.server.state.snapshot("s")
    assert snapshot["completions"][0]["response"] == raw
    assert snapshot["failure"]["failure_origin"] == "infrastructure"
    assert snapshot["failure"]["trainable"] is False


def test_backend_limits_cover_success_and_http_error() -> None:
    """Neither ordinary nor error responses can bypass the backend byte limit."""
    handler = _handler()
    handler.server.max_response_bytes = 16
    for response in (BytesIO(b"x" * 17), urllib.error.HTTPError(
            "http://unused", 500, "error", {}, BytesIO(b"x" * 17))):
        if isinstance(response, urllib.error.HTTPError):
            mocked = patch.object(codex.urllib.request, "urlopen", side_effect=response)
        else:
            mocked = patch.object(codex.urllib.request, "urlopen", return_value=response)
        with mocked, pytest.raises(ValueError, match="max_response_bytes"):
            handler._backend_request({})
    assert handler._read_backend(BytesIO(b"x" * 16)) == b"x" * 16
    with patch.object(codex.urllib.request, "urlopen", return_value=BytesIO(b'{"choices":[]}')):
        assert handler._backend_request({}) == {"choices": []}


def test_transport_settings_validate_before_binding() -> None:
    """Bad byte limits and empty explicit credentials fail at construction."""
    for settings in ({"admin_token": ""}, {"max_request_bytes": 0}, {"max_response_bytes": True}):
        with pytest.raises(ValueError):
            codex.CodexGateway("127.0.0.1", 0, "http://unused", "test", 2, **settings)


def test_response_limit_preserves_evidence_and_fails_delivery() -> None:
    """The byte cap rejects an oversized client response without removing its raw completion."""
    handler = _handler()
    handler.server.max_response_bytes = 16
    handler.wfile = BytesIO()
    raw = {"id": "c", "model": "test", "choices": [{"index": 0,
           "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}]}
    handler._backend_request = lambda _payload: raw
    for streaming in (False, True):
        state = codex._State("http://unused", "test", 2)
        state.register("s", {"policy_version": 3, "max_completions": 2})
        handler.server.state = state
        with pytest.raises(ValueError, match="max_response_bytes"):
            handler._proxy_responses({"input": "question", "stream": streaming})
        captured = state.snapshot("s")
        assert captured["completions"][0]["response"] == raw
        assert captured["failure"]["trainable"] is False
    assert handler.wfile.getvalue() == b""


def test_slow_drip_read_obeys_total_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Frequent tiny chunks cannot renew either backend or client-body deadlines indefinitely."""
    handler = _handler()
    for length in (None, 100):
        ticks = iter([0.0, 0.0, 1.0, 1.0, 2.1])
        monkeypatch.setattr(codex.time, "monotonic", lambda ticks=ticks: next(ticks))
        stream = SimpleNamespace(read1=lambda _size: b"x")
        with pytest.raises(TimeoutError, match="request_timeout"):
            handler._read_body(stream, 1024, length)


@pytest.mark.parametrize("repository", [False, True])
def test_completed_text_and_repository_budgets_are_structured(repository: bool) -> None:
    """Fully captured call limits terminate both runners without an infrastructure failure."""
    gateway = codex.CodexGateway("127.0.0.1", 0, "http://unused", "test", 2, admin_token="test-controller")
    raw = {"id": "c", "model": "test", "choices": [{"index": 0,
           "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}]}
    gateway.start()
    try:
        registration = json.dumps({"session_id": "s", "policy_version": 3, "max_completions": 1,
                                   "isolated_cli": repository}).encode()
        assert _http(gateway, "POST", "/internal/sessions", gateway.admin_token, registration)[0] == 201
        request = json.dumps(_repository_request() if repository else {"input": "question"}).encode()
        with patch.object(codex._Handler, "_backend_request", return_value=raw) as backend:
            assert _http(gateway, "POST", "/v1/responses", "s", request)[0] == 200
            status, result = _http(gateway, "POST", "/v1/responses", "s", request)
            assert status == 409
            assert backend.call_count == 1
            assert result["error"]["type"] == "completion_budget_exhausted"
            assert result["termination"] == {"reason": "max_completions", "limit": 1,
                                             "completed": 1, "policy_version": 3}
            assert _http(gateway, "POST", "/v1/responses", "s", request) == (status, result)
        _, snapshot = _http(gateway, "GET", "/internal/sessions/s", gateway.admin_token)
        assert len(snapshot["completions"]) == 1
        assert snapshot["failure"] is None
        assert snapshot["termination"] == result["termination"]
        assert _http(gateway, "POST", "/v1/responses", "s", request)[0] == 409
    finally:
        gateway.close()


def test_repository_budget_requires_completed_reservations_and_real_boolean_opt_in() -> None:
    """Reserved work is not a completed sample; failure always takes precedence over a budget."""
    state = codex._State("http://unused", "test", 2)
    for value in (1, "true", None):
        with pytest.raises(ValueError, match="isolated_cli"):
            state.register("invalid", {"policy_version": 3, "max_completions": 1, "isolated_cli": value})
    state.register("s", {"policy_version": 3, "max_completions": 1, "isolated_cli": True})
    session = state.reserve_completion("s")
    with pytest.raises(ValueError, match="max_completions"):
        state.reserve_completion("s")
    assert session.termination is None
    state.save_completion("s", {"response": {}})
    state.release_completion(session)
    with pytest.raises(codex.CompletionBudgetExhausted):
        state.reserve_completion("s")
    state.record_failure(session, {"failure_origin": "infrastructure", "trainable": False})
    with pytest.raises(ValueError, match="failed"):
        state.reserve_completion("s")
    assert state.snapshot("s")["failure"]["trainable"] is False


def test_budget_response_delivery_failure_invalidates_the_terminal_session() -> None:
    """A typed budget cannot conceal a broken delivery to the candidate CLI."""
    handler = _handler()
    state = handler.server.state
    session = state.get("s")
    session.isolated_cli = True
    session.completions = [{"ordinal": 0}, {"ordinal": 1}]
    handler.wfile = SimpleNamespace(write=lambda _data: (_ for _ in ()).throw(BrokenPipeError("disconnect")))
    with pytest.raises(BrokenPipeError):
        handler._proxy_responses(_repository_request())
    snapshot = state.snapshot("s")
    assert snapshot["termination"]["reason"] == "max_completions"
    assert snapshot["failure"]["failure_origin"] == "infrastructure"
    assert snapshot["failure"]["trainable"] is False
    assert session.inflight_requests == 0


@pytest.mark.parametrize("recovers", [False, True])
def test_repository_format_resampling_keeps_its_existing_terminal_attribution(recovers: bool) -> None:
    """Format-budget failure stays model zero; recovered calls may later reach normal task budget."""
    handler = _handler()
    state = handler.server.state
    session = state.get("s")
    session.isolated_cli = True
    session.max_completions = 2 if recovers else 1
    handler.wfile = BytesIO()
    raw = {"id": "c", "model": "test", "choices": [{"index": 0,
           "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}],
           "hyper_tool_protocol": [{"engine_text": "malformed tool"}]}
    handler._backend_request = lambda _payload: raw
    malformed = {"failure_origin": "model", "failure_reason": "invalid_json", "trainable": True}
    outcomes = [malformed, {"failure_origin": None, "trainable": True}] if recovers else [malformed]
    with patch.object(codex, "inspect_tool_response", side_effect=outcomes):
        if recovers:
            handler._proxy_responses(_repository_request())
            handler._proxy_responses(_repository_request("continue"))
            assert session.termination["reason"] == "max_completions"
            assert session.failure is None
        else:
            with pytest.raises(codex.ToolProtocolFailure):
                handler._proxy_responses(_repository_request())
            assert session.termination is None
            assert session.failure["failure_reason"] == "tool_format_budget_exhausted"
    assert len(session.completions) == session.max_completions
