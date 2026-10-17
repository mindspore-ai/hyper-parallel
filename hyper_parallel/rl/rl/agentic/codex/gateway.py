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
"""Codex Responses recorder with strict sampled-token evidence and isolated repository sessions."""

from __future__ import annotations

import http.client
import json
import logging
import secrets
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import parse_qs, unquote, urlparse

from rl.agentic.codex.protocol import CodexResponsesProtocol
from rl.agentic.codex.cli_contract import prepare_isolated_request, tool_argument_failure
from rl.tool_protocol import inspect_tool_response


logger = logging.getLogger(__name__)


class ToolProtocolFailure(RuntimeError):
    """Carry an evidence-based terminal outcome without inventing a zero reward."""

    def __init__(self, outcome: dict[str, Any]) -> None:
        """Retain the classified terminal outcome for the harness."""
        super().__init__(outcome["failure_reason"])
        self.outcome = outcome


class CompletionBudgetExhausted(RuntimeError):
    """Identify a fully recorded call budget without inventing an infrastructure failure."""

    def __init__(self, termination: dict[str, Any]) -> None:
        """Retain the controller-owned terminal evidence for the HTTP response."""
        super().__init__(f"Repository session exhausted its {termination['limit']} model calls")
        self.termination = dict(termination)


@dataclass
class _Session:
    """Mutable state for one gateway session and its sampled completions."""
    policy_version: int
    artifact_dir: Optional[Path]
    max_completions: int
    generation: dict[str, Any]
    instructions_override: Optional[str] = None
    enable_thinking: Optional[bool] = None
    compact_harness_context: bool = False
    exec_command_yield_time_ms: Optional[int] = None
    isolated_cli: bool = False
    max_observation_chars: Optional[int] = None
    completions: list[dict[str, Any]] = field(default_factory=list)
    reserved_completions: int = 0
    inflight_requests: int = 0
    closing: bool = False
    failure: Optional[dict[str, Any]] = None
    termination: Optional[dict[str, Any]] = None


def _completion_outcome(session: _Session, response: dict, transformed: dict, request_kind: str) -> dict:
    """Validate executable calls and compaction summaries against the session contract."""
    outcome = inspect_tool_response(response)
    if (session.isolated_cli and request_kind == "turn" and outcome["trainable"]
            and outcome["failure_origin"] is None):
        try:
            tool_failure = tool_argument_failure(response, transformed)
        except (KeyError, TypeError, ValueError):
            outcome = {"failure_origin": "infrastructure", "trainable": False,
                       "failure_reason": "Verified tool call shape differs from parser evidence"}
        else:
            if tool_failure is not None:
                outcome = tool_failure
    if request_kind == "compaction":
        choices = response.get("choices", [])
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        summary = message.get("content") if isinstance(message, dict) else None
        if choice.get("finish_reason") != "stop" or not isinstance(summary, str) or not summary.strip():
            outcome = {"failure_origin": "infrastructure", "trainable": False,
                       "failure_reason": "Codex compaction did not return a complete non-empty summary"}
    return outcome


class _State:
    """Own registered gateway sessions and captured completion events."""
    def __init__(self, backend_url: str, model_name: str, request_timeout: float,
                 max_inflight_requests: int = 1) -> None:
        """Initialize shared admission limits and trace storage."""
        if (isinstance(max_inflight_requests, bool) or not isinstance(max_inflight_requests, int)
                or max_inflight_requests <= 0):
            raise ValueError("Codex gateway max_inflight_requests must be a positive integer")
        parsed_backend = urlparse(backend_url)
        if (parsed_backend.scheme not in {"http", "https"} or not parsed_backend.hostname
                or parsed_backend.username or parsed_backend.password
                or parsed_backend.query or parsed_backend.fragment):
            raise ValueError("Backend URL must be an HTTP origin without credentials, query or fragment")
        self.backend_url = backend_url.rstrip("/")
        self.model_name = model_name
        self.request_timeout = request_timeout
        self.protocol = CodexResponsesProtocol()
        self.lock = threading.RLock()
        self.idle = threading.Condition(self.lock)
        self.backend_slots = threading.BoundedSemaphore(max_inflight_requests)
        self.sessions: dict[str, _Session] = {}
        self.released_sessions: set[str] = set()
        self.closing = False

    @staticmethod
    def _generation_settings(payload: dict[str, Any]) -> dict[str, Any]:
        """Validate generation and CLI context settings independently of registration."""
        generation = payload.get("generation", {})
        if not isinstance(generation, Mapping):
            raise ValueError("Codex session generation settings must be a mapping")
        allowed_generation = {
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "ignore_eos",
        }
        unknown_generation = set(generation) - allowed_generation
        if unknown_generation:
            raise ValueError(
                "Unknown Codex generation settings: "
                f"{sorted(unknown_generation)}"
            )
        instructions_override = payload.get("instructions_override")
        if instructions_override is not None and (
            not isinstance(instructions_override, str) or not instructions_override
        ):
            raise ValueError("Codex session instructions_override must be non-empty text or null")
        enable_thinking = payload.get("enable_thinking")
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise ValueError("Codex session enable_thinking must be a boolean or null")
        compact_harness_context = payload.get("compact_harness_context", False)
        if not isinstance(compact_harness_context, bool):
            raise ValueError("Codex session compact_harness_context must be a boolean")
        command_yield = payload.get("exec_command_yield_time_ms")
        if command_yield is not None and (
            not isinstance(command_yield, int) or isinstance(command_yield, bool) or command_yield <= 0
        ):
            raise ValueError("Codex session exec_command_yield_time_ms must be positive or null")
        return {"generation": dict(generation), "instructions_override": instructions_override,
                "enable_thinking": enable_thinking, "compact_harness_context": compact_harness_context,
                "exec_command_yield_time_ms": command_yield}

    def _new_session(self, payload: dict[str, Any], artifact_dir: Path | None) -> _Session:
        """Validate sampling identity and isolated CLI limits before publishing a session."""
        policy_version = payload.get("policy_version")
        if not isinstance(policy_version, int) or isinstance(policy_version, bool) or policy_version < 0:
            raise ValueError("Training sessions require a non-negative policy version")
        max_completions = payload.get("max_completions")
        if not isinstance(max_completions, int) or isinstance(max_completions, bool) or max_completions <= 0:
            raise ValueError("Codex session requires positive max_completions")
        isolated_cli = payload.get("isolated_cli", False)
        if not isinstance(isolated_cli, bool):
            raise ValueError("Codex session isolated_cli must be a boolean")
        observation_limit = payload.get("max_observation_chars")
        if observation_limit is not None and (
                not isolated_cli or isinstance(observation_limit, bool)
                or not isinstance(observation_limit, int) or observation_limit <= 0):
            raise ValueError("max_observation_chars requires an isolated session and a positive integer")
        settings = self._generation_settings(payload)
        session = _Session(
            policy_version=policy_version,
            artifact_dir=artifact_dir,
            max_completions=max_completions,
            **settings,
            isolated_cli=isolated_cli,
            max_observation_chars=observation_limit,
        )
        return session

    def register(self, session_id: str, payload: dict[str, Any]) -> None:
        """Validate and register a new gateway session and its artifact directory."""
        if not session_id:
            raise ValueError("Codex session ID must be non-empty")
        artifact_value = payload.get("artifact_dir")
        artifact_dir = Path(artifact_value).resolve() if isinstance(artifact_value, str) else None
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
        session = self._new_session(payload, artifact_dir)
        with self.lock:
            if self.closing:
                raise ValueError("Codex gateway is closing")
            if session_id in self.released_sessions:
                raise ValueError("Session was already released")
            if session_id in self.sessions:
                raise ValueError(f"Codex session already exists: {session_id}")
            self.sessions[session_id] = session
        self.event(session_id, "session.registered", payload)

    def get(self, session_id: str) -> _Session:
        """Resolve an existing session without inventing new lifecycle state."""
        with self.lock:
            try:
                return self.sessions[session_id]
            except KeyError as error:
                raise ValueError(f"Unknown Codex session: {session_id}") from error

    def remove(self, session_id: str) -> None:
        """Release one completed in-memory trace while retaining its artifacts."""
        with self.idle:
            self.released_sessions.add(session_id)
            if session_id not in self.sessions:
                return
            session = self.get(session_id)
            session.closing = True
            if not self.idle.wait_for(lambda: session.inflight_requests == 0, timeout=self.request_timeout):
                raise RuntimeError(f"Timed out draining Codex session: {session_id}")
            self.event(session_id, "session.released", {})
            self.sessions.pop(session_id, None)

    def snapshot(self, session_id: str) -> dict[str, Any]:
        """Seal admission and return complete terminal evidence after existing handlers settle."""
        with self.idle:
            session = self.get(session_id)
            session.closing = True
            if not self.idle.wait_for(lambda: session.inflight_requests == 0, timeout=self.request_timeout):
                raise RuntimeError(f"Timed out reading in-flight Codex session: {session_id}")
            result = {"policy_version": session.policy_version, "completions": list(session.completions),
                      "failure": session.failure}
            if session.termination is not None:
                result["termination"] = dict(session.termination)
            return result

    def observe(self, session_id: str) -> dict[str, Any]:
        """Read captured evidence without sealing admission for citation repair."""
        with self.lock:
            session = self.get(session_id)
            return {"policy_version": session.policy_version, "completions": list(session.completions),
                    "failure": session.failure}

    def record_failure(self, session: _Session, outcome: dict[str, Any]) -> None:
        """Never let a concurrent model failure hide a prior untrainable request failure."""
        with self.lock:
            if session.failure is None or (session.failure.get("trainable") is True
                                           and outcome.get("trainable") is not True):
                session.failure = dict(outcome)

    def begin_request(self, session_id: str) -> _Session:
        """Keep a session alive through its backend work and client response."""
        with self.idle:
            session = self.get(session_id)
            if self.closing or session.failure is not None or (session.closing and session.termination is None):
                raise ValueError(f"Codex session is closing or failed: {session_id}")
            session.inflight_requests += 1
            return session

    def finish_request(self, session: _Session) -> None:
        """Wake a draining session after this HTTP request has fully settled."""
        with self.idle:
            session.inflight_requests -= 1
            self.idle.notify_all()

    def drain(self) -> None:
        """Reject admission and wait for all owned HTTP handlers before shutdown."""
        with self.idle:
            self.closing = True
            for session in self.sessions.values():
                session.closing = True
            if not self.idle.wait_for(lambda: all(session.inflight_requests == 0 for session in self.sessions.values()),
                                      timeout=self.request_timeout):
                raise RuntimeError("Timed out draining Codex gateway requests")

    def reserve_completion(self, session_id: str) -> _Session:
        """Atomically reserve one model call, including calls waiting for admission."""
        with self.idle:
            session = self.get(session_id)
            if session.failure is not None or (session.closing and session.inflight_requests == 0):
                raise ValueError(f"Codex session is closing or failed: {session_id}")
            if session.termination is not None:
                raise CompletionBudgetExhausted(session.termination)
            if (len(session.completions) == session.max_completions
                    and session.reserved_completions == 0):
                session.termination = {"reason": "max_completions", "limit": session.max_completions,
                                       "completed": len(session.completions), "policy_version": session.policy_version}
                self.event(session_id, "session.budget_exhausted", session.termination)
                raise CompletionBudgetExhausted(session.termination)
            if len(session.completions) + session.reserved_completions >= session.max_completions:
                raise ValueError(f"Codex session exceeded max_completions={session.max_completions}")
            session.reserved_completions += 1
            return session

    def release_completion(self, session: _Session) -> None:
        """Release a reserved slot after the call was recorded or failed."""
        with self.idle:
            if session.reserved_completions <= 0:
                raise RuntimeError("Codex completion reservation underflow")
            session.reserved_completions -= 1
            self.idle.notify_all()

    def save_completion(self, session_id: str, record: dict[str, Any]) -> None:
        """Append one raw completion and assign its stable trace ordinal."""
        session = self.get(session_id)
        with self.lock:
            record["ordinal"] = len(session.completions)
            session.completions.append(record)
        self.event(session_id, "completion.recorded", record)

    def mark_protocol_recovery(
        self, session_id: str, records: list[dict[str, Any]],
        recovery_ordinal: int, valid_tool_call: bool,
    ) -> None:
        """Link rejected sampled actions to the first subsequent valid action."""
        with self.lock:
            for record in records:
                record["metadata"].update(
                    recovered_with_valid_action=True,
                    recovered_with_valid_tool_call=valid_tool_call,
                    recovery_call_index=recovery_ordinal,
                )
        self.event(session_id, "tool_protocol.recovered", {
            "recovery_call_index": recovery_ordinal,
            "valid_tool_call": valid_tool_call,
            "event_ids": [record["metadata"]["protocol_event_id"] for record in records],
        })

    def event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
        """Append a diagnostic event when the session has artifact storage."""
        try:
            session = self.get(session_id)
        except ValueError:
            return
        if session.artifact_dir is None:
            return
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "session_id": session_id,
            "payload": payload,
        }
        with self.lock:
            event_path = session.artifact_dir / "gateway-events.jsonl"
            with event_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")


class _Handler(BaseHTTPRequestHandler):
    """Translate HTTP gateway requests into versioned backend completions."""
    server: "_GatewayServer"

    def setup(self) -> None:
        """Bound disconnected client I/O as well as backend calls during shutdown."""
        super().setup()
        self.connection.settimeout(self.server.state.request_timeout)

    def _handle_get(self) -> None:
        """Serve health and captured-session inspection."""
        path = urlparse(self.path).path
        if path.startswith("/internal/") and not self._authorize_admin():
            return
        if path == "/healthz":
            self._json(HTTPStatus.OK, {"status": "ok"})
            return
        prefix = "/internal/sessions/"
        if path.startswith(prefix):
            session_id = unquote(path[len(prefix):])
            try:
                live = parse_qs(urlparse(self.path).query).get("live") == ["1"]
                snapshot = (self.server.state.observe(session_id) if live
                            else self.server.state.snapshot(session_id))
            except ValueError as error:
                self._error(HTTPStatus.NOT_FOUND, str(error))
                return
            except RuntimeError as error:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(error))
                return
            try:
                self._json(HTTPStatus.OK, snapshot)
            except ValueError as error:
                self._error(HTTPStatus.BAD_GATEWAY, str(error))
            return
        self._error(HTTPStatus.NOT_FOUND, "Unknown gateway route")

    def _handle_post(self) -> None:
        """Register a session or proxy one Responses request."""
        path = urlparse(self.path).path
        if path.startswith("/internal/") and not self._authorize_admin():
            return
        if path.startswith("/internal/sessions/") and path.endswith("/failure"):
            try:
                self._request_json()
                session_id = unquote(path[len("/internal/sessions/"):-len("/failure")])
                session = self.server.state.get(session_id)
                self.server.state.record_failure(session, {
                    "failure_origin": "infrastructure", "trainable": False,
                    "failure_reason": "Candidate relay transport failed",
                })
            except (ValueError, OSError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
                return
            self._json(HTTPStatus.OK, {"recorded": True})
            return
        if path == "/internal/sessions":
            try:
                body = self._request_json()
            except (ValueError, OSError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
                return
            session_id = body.pop("session_id", None)
            try:
                if body.get("isolated_cli") is True and not self.server.require_admin_auth:
                    raise ValueError("Repository sessions require authenticated management")
                self.server.state.register(str(session_id or ""), body)
            except (KeyError, TypeError, ValueError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
                return
            self._json(HTTPStatus.CREATED, {"session_id": session_id})
            return
        if path.rstrip("/") not in {"/responses", "/v1/responses"}:
            self._error(HTTPStatus.NOT_FOUND, "Unknown gateway route")
            return
        try:
            self._proxy_responses()
        except ToolProtocolFailure as error:
            self._json(HTTPStatus.BAD_GATEWAY, {"error": {
                "type": "tool_protocol_error", "message": str(error), **error.outcome,
            }})
        except (RuntimeError, ValueError, OSError) as error:
            logger.exception("Codex gateway request failed")
            self._error(HTTPStatus.BAD_GATEWAY, str(error))

    def _handle_delete(self) -> None:
        """Release a captured session after its trajectory is materialized."""
        path = urlparse(self.path).path
        if path.startswith("/internal/") and not self._authorize_admin():
            return
        prefix = "/internal/sessions/"
        if not path.startswith(prefix):
            self._error(HTTPStatus.NOT_FOUND, "Unknown gateway route")
            return
        session_id = unquote(path[len(prefix):])
        try:
            if not self.server.require_admin_auth:
                self.server.state.get(session_id)
            self.server.state.remove(session_id)
        except ValueError as error:
            self._error(HTTPStatus.NOT_FOUND, str(error))
            return
        except RuntimeError as error:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(error))
            return
        self._json(HTTPStatus.OK, {"released": session_id})

    def _proxy_responses(self, original: dict[str, Any] | None = None) -> None:
        """Translate a Responses request and capture its backend completion."""
        session_id = self._bearer_token()
        session = self.server.state.begin_request(session_id)
        try:
            if original is None:
                original = self._request_json()
            self._respond_to_session(session_id, session, original)
        except CompletionBudgetExhausted as error:
            try:
                self._json(HTTPStatus.CONFLICT, {
                    "error": {"type": "completion_budget_exhausted", "message": str(error)},
                    "termination": error.termination,
                })
            except (RuntimeError, ValueError, OSError) as delivery_error:
                self.server.state.record_failure(session, {
                    "failure_origin": "infrastructure", "failure_reason": str(delivery_error), "trainable": False,
                })
                raise
        except ToolProtocolFailure as error:
            self.server.state.record_failure(session, error.outcome)
            raise
        except (RuntimeError, ValueError, OSError) as error:
            self.server.state.record_failure(session, {
                "failure_origin": "infrastructure", "failure_reason": str(error), "trainable": False,
            })
            raise
        finally:
            self.server.state.finish_request(session)


    def _model_completion(self, session_id: str, original: dict, transformed: dict,
                          request_kind: str = "turn") -> tuple[dict, dict, dict]:
        """Admit one call and atomically convert its reservation into captured evidence."""
        state = self.server.state
        session = state.reserve_completion(session_id)
        record = None
        try:
            if not state.backend_slots.acquire(timeout=state.request_timeout):
                raise RuntimeError("Timed out waiting for Codex backend admission")
            try:
                response = self._backend_request(transformed)
            finally:
                state.backend_slots.release()
            outcome = _completion_outcome(session, response, transformed, request_kind)
            route_payload = None
            for index, choice in enumerate(response.get("choices", [])):
                if isinstance(choice, dict):
                    payload = choice.pop("routed_experts", None)
                    if index == 0:
                        route_payload = payload
            will_deliver_feedback = (
                outcome.get("failure_origin") == "model"
                and len(session.completions) + 1 < session.max_completions
            )
            record = {"timestamp": time.time(), "original_request": original, "request": transformed,
                      "response": response, "metadata": {"policy_version": session.policy_version,
                                                          "session_id": session_id,
                                                          "protocol_event_id": f"{session_id}:{uuid.uuid4().hex}",
                                                          "feedback_delivered": will_deliver_feedback,
                                                          "recovered_with_valid_action": False,
                                                          "recovered_with_valid_tool_call": False,
                                                          "recovery_call_index": None, **outcome}}
            capture_spec = getattr(self.server, "router_capture_spec", None)
            if capture_spec is not None:
                record["router_capture_spec"] = capture_spec
                record["routed_experts"] = route_payload
            return response, outcome, record
        finally:
            with state.lock:
                try:
                    if record is not None:
                        state.save_completion(session_id, record)
                finally:
                    state.release_completion(session)

    def _respond_to_session(self, session_id: str, session: _Session, original: dict) -> None:
        """Keep rejected model actions intact while allowing explicitly budgeted resampling."""
        transformed = self.server.state.protocol.transform_request(
            original,
            self.server.state.model_name,
            instructions_override=session.instructions_override,
            enable_thinking=session.enable_thinking,
            compact_harness_context=session.compact_harness_context,
        )
        request_kind = prepare_isolated_request(original, transformed) if session.isolated_cli else "turn"
        if session.max_observation_chars is not None:
            for message in transformed["messages"]:
                if message.get("role") == "tool" and isinstance(message.get("content"), str):
                    content, limit = message["content"], session.max_observation_chars
                    if len(content) > limit:
                        marker = "\n... (observation truncated) ...\n"[:limit]
                        head = (limit - len(marker)) // 2
                        tail = limit - len(marker) - head
                        message["content"] = content[:head] + marker + (content[-tail:] if tail else "")
        transformed.update(session.generation)
        if session.isolated_cli and request_kind == "compaction":
            transformed["max_tokens"] = min(1024, transformed.get("max_tokens", 1024))
        pending_protocol_errors: list[dict[str, Any]] = []
        while True:
            response, outcome, record = self._model_completion(session_id, original, transformed, request_kind)
            if outcome["failure_origin"] is None and pending_protocol_errors:
                message = response.get("choices", [{}])[0].get("message", {})
                self.server.state.mark_protocol_recovery(
                    session_id, pending_protocol_errors, int(record["ordinal"]),
                    bool(message.get("tool_calls")),
                )
                pending_protocol_errors.clear()
            if not outcome["trainable"]:
                raise ToolProtocolFailure(dict(outcome, model_calls=len(session.completions)))
            if outcome["failure_origin"] != "model":
                break
            pending_protocol_errors.append(record)
            if len(session.completions) >= session.max_completions:
                raise ToolProtocolFailure(dict(outcome, failure_reason="tool_format_budget_exhausted",
                                               model_calls=len(session.completions)))
            feedback = outcome.get("tool_feedback")
            transformed = {**transformed, "messages": [*transformed["messages"],
                {"role": "assistant", "content": response["hyper_tool_protocol"][0]["engine_text"]},
                {"role": "user", "content": feedback or
                 "Tool call rejected: invalid JSON/schema. Nothing was executed. "
                 "Generate a new valid tool call or answer. Use strict JSON escaping; "
                 "a backslash before an apostrophe is not a valid JSON escape."},
            ]}
        result = self.server.state.protocol.transform_response(
            response, original, exec_command_yield_time_ms=session.exec_command_yield_time_ms,
        )
        if bool(original.get("stream")):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Model-Calls", str(len(session.completions)))
            self.end_headers()
            self._write_sse_events(result)
            return
        self._json(HTTPStatus.OK, result, model_calls=len(session.completions))

    def _write_sse_events(self, result: dict[str, Any]) -> None:
        """Write one bounded Responses stream; delivery errors invalidate the episode."""
        deadline = time.monotonic() + self.server.state.request_timeout
        total = 0
        for event in self.server.state.protocol.stream_events(result):
            if time.monotonic() >= deadline:
                raise TimeoutError("Gateway SSE write exceeded request_timeout")
            payload = json.dumps(event, ensure_ascii=False).encode("utf-8")
            frame = b"event: " + str(event["type"]).encode("utf-8") + b"\n" + b"data: " + payload + b"\n\n"
            total += len(frame)
            if self.server.max_response_bytes is not None and total > self.server.max_response_bytes:
                raise ValueError("Gateway SSE response exceeds max_response_bytes")
            self.wfile.write(frame)
            self.wfile.flush()

    def _backend_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON chat request to the configured inference backend."""
        data = json.dumps(payload).encode("utf-8")
        if self.server.max_request_bytes is not None and len(data) > self.server.max_request_bytes:
            raise ValueError("Backend request exceeds max_request_bytes")
        headers = {"Content-Type": "application/json"}
        request = urllib.request.Request(
            f"{self.server.state.backend_url}/v1/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            timeout = self.server.state.request_timeout
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = self._read_backend(response)
        except urllib.error.HTTPError as error:
            detail = self._read_backend(error).decode("utf-8", errors="replace")[:4096]
            raise RuntimeError(
                f"vLLM chat completion failed with HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"vLLM chat completion request failed: {error.reason}") from error
        except http.client.RemoteDisconnected as error:
            raise RuntimeError(
                "vLLM chat completion connection closed before a response"
            ) from error
        try:
            decoded = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("vLLM chat completion returned invalid JSON") from error
        if not isinstance(decoded, dict):
            raise RuntimeError("vLLM chat completion returned a non-object response")
        return decoded

    def _authorize_admin(self) -> bool:
        """Authenticate management requests before reading bodies or touching session state."""
        if not self.server.require_admin_auth:
            return True
        supplied = self.headers.get("Authorization", "")
        if secrets.compare_digest(supplied.encode(), ("Bearer " + self.server.admin_token).encode()):
            return True
        self._error(HTTPStatus.UNAUTHORIZED, "Management authentication required")
        return False

    def _read_body(self, stream: Any, limit: int | None, length: int | None = None) -> bytes:
        """Bound total read time as well as bytes, including clients that continually drip data."""
        deadline = time.monotonic() + self.server.state.request_timeout
        chunks = []
        total = 0
        while length is None or total < length:
            if time.monotonic() >= deadline:
                raise TimeoutError("Gateway body read exceeded request_timeout")
            size = 65536
            if limit is not None:
                size = min(size, limit + 1 - total)
            if length is not None:
                size = min(size, length - total)
            chunk = stream.read1(size)
            if time.monotonic() >= deadline:
                raise TimeoutError("Gateway body read exceeded request_timeout")
            if not chunk:
                break
            total += len(chunk)
            if limit is not None and total > limit:
                raise ValueError("Gateway body exceeds max_response_bytes")
            chunks.append(chunk)
        if length is not None and total != length:
            raise ValueError("Request body ended before Content-Length")
        return b"".join(chunks)

    def _read_backend(self, response: Any) -> bytes:
        """Bound successful and error backend bodies without silently truncating evidence."""
        return self._read_body(response, self.server.max_response_bytes)

    def _request_json(self) -> dict[str, Any]:
        """Read and validate the HTTP request body as a JSON object."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid Content-Length") from error
        if length <= 0:
            raise ValueError("Request body must be non-empty")
        if self.server.max_request_bytes is not None and length > self.server.max_request_bytes:
            raise ValueError("Gateway request exceeds max_request_bytes")
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Transfer-Encoding is unsupported")
        encoded = self._read_body(self.rfile, self.server.max_request_bytes, length)
        try:
            body = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Request body must be valid JSON") from error
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object")
        return body

    def _bearer_token(self) -> str:
        authorization = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not authorization.startswith(prefix) or not authorization[len(prefix):]:
            raise ValueError("Codex gateway requires a session bearer token")
        return authorization[len(prefix):]

    def _json(self, status: HTTPStatus, payload: dict[str, Any], model_calls: Optional[int] = None) -> None:
        """Send one bounded JSON response with optional model-call accounting."""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if self.server.max_response_bytes is not None and len(body) > self.server.max_response_bytes:
            if status < HTTPStatus.BAD_REQUEST:
                raise ValueError("Gateway response exceeds max_response_bytes")
            body = b""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if model_calls is not None:
            self.send_header("X-Model-Calls", str(model_calls))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": {"message": message.replace(self.server.admin_token, "[redacted]")[:2048],
                                      "type": "gateway_error"}})

    # BaseHTTPRequestHandler's first argument is positional; avoid shadowing the format builtin.
    def log_message(self, format_string: str, *args: Any) -> None:  # pylint: disable=arguments-differ
        """Write HTTP diagnostics through the gateway logger."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Codex gateway: %s", format_string % args if args else format_string)

    # BaseHTTPRequestHandler dispatches by these exact attribute names.
    do_GET = _handle_get  # pylint: disable=invalid-name
    do_POST = _handle_post  # pylint: disable=invalid-name
    do_DELETE = _handle_delete  # pylint: disable=invalid-name


class _GatewayServer(ThreadingHTTPServer):
    """Serve authenticated controller requests and session-bound model requests."""
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: _State, admin_token: str | None,
                 max_request_bytes: int | None, max_response_bytes: int | None,
                 router_capture_spec: Mapping[str, Any] | None = None) -> None:
        """Bind one HTTP server to its shared gateway state."""
        self.state = state
        if admin_token is not None and (not isinstance(admin_token, str) or not admin_token.strip()):
            raise ValueError("Gateway admin_token must be a non-empty string")
        for name, value in (("max_request_bytes", max_request_bytes), ("max_response_bytes", max_response_bytes)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"Gateway {name} must be a positive integer")
        self.admin_token = admin_token or secrets.token_urlsafe(32)
        self.require_admin_auth = admin_token is not None
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes
        self.router_capture_spec = None if router_capture_spec is None else dict(router_capture_spec)
        super().__init__(address, _Handler)


class CodexGateway:
    """Own the one protocol adapter used by every Codex episode on a node."""

    def __init__(
        self,
        host: str,
        port: int,
        backend_url: str,
        model_name: str,
        request_timeout: float,
        max_inflight_requests: int = 1,
        admin_token: str | None = None,
        max_request_bytes: int | None = None,
        max_response_bytes: int | None = None,
        router_capture_spec: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize an unstarted gateway."""
        if not host:
            raise ValueError("Codex gateway host must be non-empty")
        if not 0 <= port < 65536:
            raise ValueError("Codex gateway port must be in [0, 65535]")
        self._server = _GatewayServer(
            (host, port),
            _State(backend_url, model_name, request_timeout, max_inflight_requests),
            admin_token, max_request_bytes, max_response_bytes, router_capture_spec,
        )
        self._thread: Optional[threading.Thread] = None

    @property
    def admin_token(self) -> str:
        """Return the controller-only management credential; never pass it to candidates."""
        return self._server.admin_token

    @property
    def address(self) -> tuple[str, int]:
        """Return the bound gateway address."""
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        """Start request serving on a daemon thread."""
        if self._thread is not None:
            raise RuntimeError("Codex gateway is already running")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="hyper-rl-codex-gateway",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Stop request serving and release the listening socket."""
        if self._thread is None:
            self._server.server_close()
            return
        self._server.shutdown()
        try:
            self._server.state.drain()
        finally:
            self._server.server_close()
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError("Codex gateway thread did not stop")
        self._thread = None
