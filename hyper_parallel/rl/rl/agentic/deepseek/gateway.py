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
"""Independent DeepSeek Chat Completions recorder for the shared vLLM server."""

from __future__ import annotations

import http.client
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from rl.agentic.deepseek.protocol import DeepSeekChatProtocol

logger = logging.getLogger(__name__)


@dataclass
class _Session:
    policy_version: int
    policy_fingerprint: str
    artifact_dir: Path | None
    max_completions: int
    generation: dict[str, Any]
    completions: list[dict[str, Any]] = field(default_factory=list)


class _State:
    def __init__(
        self, backend_url: str, model_name: str, request_timeout: float
    ) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.model_name = model_name
        self.request_timeout = request_timeout
        self.protocol = DeepSeekChatProtocol()
        self.lock = threading.RLock()
        self.sessions: dict[str, _Session] = {}

    def register(self, session_id: str, payload: dict[str, Any]) -> None:
        if not session_id:
            raise ValueError("DeepSeek session ID must be non-empty")
        fingerprint = payload.get("policy_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("DeepSeek session requires a policy fingerprint")
        artifact_value = payload.get("artifact_dir")
        artifact_dir = (
            Path(artifact_value).resolve() if isinstance(artifact_value, str) else None
        )
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
        max_completions = int(payload.get("max_completions", 0))
        if max_completions <= 0:
            raise ValueError("DeepSeek session requires positive max_completions")
        generation = payload.get("generation", {})
        if not isinstance(generation, Mapping):
            raise ValueError("DeepSeek session generation settings must be a mapping")
        allowed = {
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "ignore_eos",
            "reasoning_effort",
        }
        unknown = set(generation) - allowed
        if unknown:
            raise ValueError(f"Unknown DeepSeek generation settings: {sorted(unknown)}")
        reasoning_effort = generation.get("reasoning_effort")
        if reasoning_effort not in {None, "off", "high", "max"}:
            raise ValueError(
                "DeepSeek reasoning_effort must be 'off', 'high', or 'max'"
            )
        session = _Session(
            policy_version=int(payload["policy_version"]),
            policy_fingerprint=fingerprint,
            artifact_dir=artifact_dir,
            max_completions=max_completions,
            generation=dict(generation),
        )
        with self.lock:
            if session_id in self.sessions:
                raise ValueError(f"DeepSeek session already exists: {session_id}")
            self.sessions[session_id] = session
        self.event(session_id, "session.registered", payload)

    def get(self, session_id: str) -> _Session:
        with self.lock:
            try:
                return self.sessions[session_id]
            except KeyError as error:
                raise ValueError(f"Unknown DeepSeek session: {session_id}") from error

    def remove(self, session_id: str) -> None:
        self.get(session_id)
        self.event(session_id, "session.released", {})
        with self.lock:
            self.sessions.pop(session_id, None)

    def save_completion(self, session_id: str, record: dict[str, Any]) -> None:
        session = self.get(session_id)
        with self.lock:
            record["ordinal"] = len(session.completions)
            session.completions.append(record)
        self.event(session_id, "completion.recorded", record)

    def event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
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
        with (
            self.lock,
            (session.artifact_dir / "gateway-events.jsonl").open(
                "a", encoding="utf-8"
            ) as stream,
        ):
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")


class _Handler(BaseHTTPRequestHandler):
    server: _GatewayServer

    def do_GET(self) -> None:  # pylint: disable=C0103
        """Serve health and captured-session inspection."""
        path = urlparse(self.path).path
        if path == "/healthz":
            self._json(HTTPStatus.OK, {"status": "ok"})
            return
        prefix = "/internal/sessions/"
        if path.startswith(prefix):
            session_id = unquote(path[len(prefix) :])
            try:
                session = self.server.state.get(session_id)
            except ValueError as error:
                self._error(HTTPStatus.NOT_FOUND, str(error))
                return
            self._json(
                HTTPStatus.OK,
                {
                    "policy_version": session.policy_version,
                    "policy_fingerprint": session.policy_fingerprint,
                    "completions": session.completions,
                },
            )
            return
        self._error(HTTPStatus.NOT_FOUND, "Unknown DeepSeek gateway route")

    def do_POST(self) -> None:  # pylint: disable=C0103
        """Register a session or proxy one DeepSeek chat request."""
        try:
            body = self._request_json()
        except ValueError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
            return
        path = urlparse(self.path).path
        if path == "/internal/sessions":
            session_id = body.pop("session_id", None)
            try:
                self.server.state.register(str(session_id or ""), body)
            except (KeyError, TypeError, ValueError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
                return
            self._json(HTTPStatus.CREATED, {"session_id": session_id})
            return
        if path.rstrip("/") not in {"/chat/completions", "/v1/chat/completions"}:
            self._error(HTTPStatus.NOT_FOUND, "Unknown DeepSeek gateway route")
            return
        try:
            self._proxy_chat(body)
        except (RuntimeError, ValueError) as error:
            logger.exception("DeepSeek gateway request failed")
            self._error(HTTPStatus.BAD_GATEWAY, str(error))

    def do_DELETE(self) -> None:  # pylint: disable=C0103
        """Release one captured session while retaining its artifacts."""
        path = urlparse(self.path).path
        prefix = "/internal/sessions/"
        if not path.startswith(prefix):
            self._error(HTTPStatus.NOT_FOUND, "Unknown DeepSeek gateway route")
            return
        session_id = unquote(path[len(prefix) :])
        try:
            self.server.state.remove(session_id)
        except ValueError as error:
            self._error(HTTPStatus.NOT_FOUND, str(error))
            return
        self._json(HTTPStatus.OK, {"released": session_id})

    def _proxy_chat(self, original: dict[str, Any]) -> None:
        session_id = self.headers.get("x-deepseek-harness-session-id", "")
        if not session_id:
            session_id = self._bearer_token()
        session = self.server.state.get(session_id)
        if len(session.completions) >= session.max_completions:
            raise ValueError(
                f"DeepSeek session exceeded max_completions={session.max_completions}"
            )
        protocol_request = dict(original)
        reasoning_effort = session.generation.get("reasoning_effort")
        if reasoning_effort is not None:
            protocol_request["reasoning_effort"] = reasoning_effort
        transformed = self.server.state.protocol.transform_request(
            protocol_request, self.server.state.model_name
        )
        transformed.update(
            {
                name: value
                for name, value in session.generation.items()
                if name != "reasoning_effort"
            }
        )
        response = self._backend_request(transformed)
        self.server.state.save_completion(
            session_id,
            {
                "timestamp": time.time(),
                "original_request": original,
                "request": transformed,
                "response": response,
                "metadata": {
                    "policy_version": session.policy_version,
                    "policy_fingerprint": session.policy_fingerprint,
                },
            },
        )
        if bool(original.get("stream")):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self._write_sse(response)
            return
        self._json(HTTPStatus.OK, response)

    def _write_sse(self, response: dict[str, Any]) -> None:
        try:
            for event in self.server.state.protocol.stream_events(response):
                payload = json.dumps(event, ensure_ascii=False).encode("utf-8")
                self.wfile.write(b"data: " + payload + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("DeepSeek Harness disconnected before SSE completion")

    def _backend_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.server.state.backend_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.server.state.request_timeout
            ) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM chat completion failed with HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"vLLM chat completion request failed: {error.reason}"
            ) from error
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

    def _request_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid Content-Length") from error
        if length <= 0:
            raise ValueError("Request body must be non-empty")
        try:
            body = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Request body must be valid JSON") from error
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object")
        return body

    def _bearer_token(self) -> str:
        authorization = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not authorization.startswith(prefix) or not authorization[len(prefix) :]:
            raise ValueError("DeepSeek gateway requires a session identity")
        return authorization[len(prefix) :]

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": {"message": message, "type": "gateway_error"}})

    def log_message(self, format_string: str, *args: Any) -> None:
        logger.debug("DeepSeek gateway: " + format_string, *args)


class _GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: _State) -> None:
        self.state = state
        super().__init__(address, _Handler)


class DeepSeekGateway:
    """Own the protocol adapter used only by DeepSeek Harness episodes."""

    def __init__(
        self,
        host: str,
        port: int,
        backend_url: str,
        model_name: str,
        request_timeout: float,
    ) -> None:
        """Initialize an unstarted DeepSeek gateway."""
        if not host:
            raise ValueError("DeepSeek gateway host must be non-empty")
        if not 0 <= port < 65536:
            raise ValueError("DeepSeek gateway port must be in [0, 65535]")
        self._server = _GatewayServer(
            (host, port), _State(backend_url, model_name, request_timeout)
        )
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        """Return the bound gateway address."""
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        """Start request serving on a daemon thread."""
        if self._thread is not None:
            raise RuntimeError("DeepSeek gateway is already running")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="hyper-rl-deepseek-gateway",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Stop request serving and release the listening socket."""
        if self._thread is None:
            self._server.server_close()
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError("DeepSeek gateway thread did not stop")
        self._thread = None
