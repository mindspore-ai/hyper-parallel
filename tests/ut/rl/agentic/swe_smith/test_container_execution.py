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
"""Real local HTTP and process contracts for the networkless candidate relay."""

import asyncio
from asyncio.subprocess import Process
import base64
from importlib.util import find_spec
import json
import os
from pathlib import Path
import socket
import struct
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

import pytest

from rl.agentic.envs.docker_workspace import DockerWorkspace, acquire_workspace_slot
from examples.swe_smith import container_execution


def _local_endpoint(monkeypatch: pytest.MonkeyPatch) -> list:
    """Run the exact endpoint source in a local subprocess instead of Docker exec."""
    spawn = asyncio.create_subprocess_exec
    failures = []

    async def local_spawn(*argv: str, **kwargs: Any) -> Process:
        """Launch the unchanged endpoint under the active test interpreter."""
        script = argv.index("-c")
        return await spawn(sys.executable, "-I", "-u", *argv[script:], **kwargs)

    monkeypatch.setattr(container_execution.asyncio, "create_subprocess_exec", local_spawn)
    monkeypatch.setattr(container_execution, "request_gateway_json", lambda *args: failures.append(args) or {})
    return failures


async def _request(url: str, path: str, body: bytes = b"{}") -> bytes:
    """Build a session-bound relay request for the transport contract tests."""
    parsed = urlparse(url)
    reader, writer = await asyncio.open_connection(parsed.hostname, parsed.port)
    try:
        writer.write((f"POST {path} HTTP/1.0\r\nHost: attacker.invalid\r\n"
                      f"Authorization: Bearer other-session\r\nContent-Length: {len(body)}\r\n\r\n").encode()
                     + body)
        await writer.drain()
        return await asyncio.wait_for(reader.read(), 3)
    finally:
        writer.close()
        await writer.wait_closed()


def test_relay_binds_route_and_session_and_denies_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Actual HTTP requests cannot override the controller target or session credential."""
    failures = _local_endpoint(monkeypatch)

    async def scenario() -> None:
        """Exercise the contract with bounded real processes and sockets."""
        requests = []

        async def upstream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            """Serve one controlled Gateway-shaped HTTP response."""
            headers = await reader.readuntil(b"\r\n\r\n")
            length = int(next(line.split(b":", 1)[1] for line in headers.split(b"\r\n")
                              if line.lower().startswith(b"content-length:")))
            requests.append((headers, await reader.readexactly(length)))
            writer.write(b"HTTP/1.0 200 OK\r\nContent-Type: text/event-stream\r\n"
                         b"Content-Length: 12\r\n\r\ndata: done\n\n")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(upstream, "127.0.0.1", 0)
        address = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/fixed"
        workspace = DockerWorkspace("task-image", docker_host="ssh://user@127.0.0.1:2233")
        relay = container_execution.CandidateRelay(workspace, address, "bound-session",
                                          request_timeout=2, max_request_bytes=1024, max_response_bytes=1024)
        try:
            endpoint = await relay.start()
            forbidden = await _request(endpoint, "/internal/sessions")
            assert b" 403 " in forbidden.split(b"\r\n")[0]
            body = json.dumps({"input": [], "url": "http://attacker.invalid"}).encode()
            response = await _request(endpoint, "/v1/responses", body)
            assert response.endswith(b"data: done\n\n")
            assert len(requests) == 1
            headers, actual = requests[0]
            assert headers.startswith(b"POST /fixed/responses HTTP/1.0\r\n")
            assert b"Authorization: Bearer bound-session\r\n" in headers
            assert b"other-session" not in headers and b"attacker.invalid" not in headers
            assert actual == body
        finally:
            await relay.close()
            server.close()
            await server.wait_closed()
        assert relay._process.returncode == 0
        assert not failures

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["response_limit", "incomplete_response", "downstream_disconnect"])
def test_relay_transport_failure_invalidates_bound_session(
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """Truncation, oversized responses and failed local delivery never appear as model successes."""
    failures = _local_endpoint(monkeypatch)

    async def scenario() -> None:
        """Exercise the contract with bounded real processes and sockets."""
        sampled = asyncio.Event()
        release = asyncio.Event()

        async def upstream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            """Serve one controlled Gateway-shaped HTTP response."""
            await reader.readuntil(b"\r\n\r\n")
            await reader.readexactly(2)
            sampled.set()
            await release.wait()
            length = 2048 if failure == "response_limit" else 100
            payload = b"x" * (length if failure == "downstream_disconnect" else 1)
            writer.write(f"HTTP/1.0 200 OK\r\nContent-Length: {length}\r\n\r\n".encode() + payload)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(upstream, "127.0.0.1", 0)
        address = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        workspace = DockerWorkspace("task-image", docker_host="ssh://user@127.0.0.1:2233")
        relay = container_execution.CandidateRelay(workspace, address, "bound-session",
                                          request_timeout=2, max_request_bytes=1024, max_response_bytes=1024,
                                          admin_url=address, admin_token="controller-secret")
        request = None
        try:
            endpoint = await relay.start()
            if failure == "downstream_disconnect":
                client = socket.socket()
                client.connect(("127.0.0.1", urlparse(endpoint).port))
                client.sendall(b"POST /responses HTTP/1.0\r\nContent-Length: 2\r\n\r\n{}")
                await asyncio.wait_for(sampled.wait(), 2)
                client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                client.close()
            else:
                request = asyncio.create_task(_request(endpoint, "/responses"))
                await asyncio.wait_for(sampled.wait(), 2)
            release.set()
            await asyncio.wait_for(relay._pump_task, 3)
            with pytest.raises(container_execution.RelayError):
                await relay.close()
            assert len(failures) == 1
            assert failures[0][1:3] == ("POST", f"{address}/internal/sessions/bound-session/failure")
            assert failures[0][-1] == "controller-secret"
            assert relay._process.returncode is not None
            assert not relay.budget_exhausted.is_set()
            assert relay.budget_termination is None
        finally:
            release.set()
            if request is not None:
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)
            server.close()
            await server.wait_closed()
            if relay._process.returncode is None:
                relay._process.kill()
                await relay._process.wait()

    asyncio.run(scenario())


@pytest.mark.parametrize("ack", [b'{"delivered":true}\n', b'{"delivered":false}\n', b'{}\n', b''])
def test_budget_signal_requires_successful_delivery_ack(ack: bytes) -> None:
    """A controller budget response cannot stop the CLI before its delivery is acknowledged."""
    async def scenario() -> None:
        """Pause the response acknowledgement to inspect both sides of the signal boundary."""
        termination = {"reason": "max_completions", "limit": 2, "completed": 2, "policy_version": 3}
        reply = {"status": 409, "content_type": "application/json", "body": base64.b64encode(json.dumps({
            "error": {"type": "completion_budget_exhausted"}, "termination": termination,
        }).encode()).decode()}
        workspace = DockerWorkspace("task-image", docker_host="ssh://user@127.0.0.1:2233")
        relay = container_execution.CandidateRelay(workspace, "http://127.0.0.1:9", "bound")
        waiting, release = asyncio.Event(), asyncio.Event()
        lines = iter(enumerate((b'{"body":"e30="}\n', ack, b'')))

        async def readline() -> bytes:
            """Hold the acknowledgement after the controller has forwarded its response."""
            index, line = next(lines)
            if index == 1:
                waiting.set()
                await release.wait()
            return line

        relay._exchange = AsyncMock(return_value=reply)
        relay._process = SimpleNamespace(stdout=SimpleNamespace(readline=readline),
                                         stdin=SimpleNamespace(write=MagicMock(), drain=AsyncMock()))
        relay._closing = True
        task = asyncio.create_task(relay._pump())
        await waiting.wait()
        assert not relay.budget_exhausted.is_set()
        assert relay.budget_termination is None
        release.set()
        await asyncio.wait_for(task, 2)
        if ack == b'{"delivered":true}\n':
            assert relay.budget_exhausted.is_set()
            assert relay.budget_termination == termination
            assert relay._failure is None
        else:
            assert not relay.budget_exhausted.is_set()
            assert relay.budget_termination is None
            assert isinstance(relay._failure, container_execution.RelayError)

    asyncio.run(scenario())


@pytest.mark.parametrize("late_frame", [b'', b'{"body":"e30="}\n'])
def test_confirmed_budget_finishes_before_cli_retry_or_close(late_frame: bytes) -> None:
    """After the terminal response is delivered, CLI retries cannot reopen a completed relay."""
    async def scenario() -> None:
        """Deliver one real terminal frame before an EOF or a late retry."""
        termination = {"reason": "max_completions", "limit": 2, "completed": 2, "policy_version": 1}
        reply = {"status": 409, "body": base64.b64encode(json.dumps({
            "error": {"type": "completion_budget_exhausted"}, "termination": termination,
        }).encode()).decode()}
        workspace = DockerWorkspace("task-image")
        relay = container_execution.CandidateRelay(workspace, "http://127.0.0.1:9", "bound")
        frames = iter([b'{"body":"e30="}\n', b'{"delivered":true}\n', late_frame])
        relay._process = SimpleNamespace(stdout=SimpleNamespace(readline=AsyncMock(side_effect=lambda: next(frames))),
                                         stdin=SimpleNamespace(write=MagicMock(), drain=AsyncMock()))
        relay._exchange = AsyncMock(side_effect=[reply, container_execution.RelayError("late retry")])
        await relay._pump()
        assert relay.budget_exhausted.is_set()
        assert relay._failure is None
        relay._exchange.assert_awaited_once()

    asyncio.run(scenario())


@pytest.mark.parametrize("status,override", [
    (200, {}), (409, {"error": {"type": "other"}}), (409, {"termination": {}}),
    (409, {"termination": {"reason": "max_completions", "limit": 2, "completed": 1, "policy_version": 3}}),
    (409, {"termination": {"reason": "max_completions", "limit": True, "completed": 1, "policy_version": 3}}),
])
def test_budget_signal_rejects_model_body_and_invalid_gateway_terminal(status: int, override: dict) -> None:
    """A model-generated lookalike JSON body or malformed control response is not a budget signal."""
    body = {"error": {"type": "completion_budget_exhausted"},
            "termination": {"reason": "max_completions", "limit": 2, "completed": 2, "policy_version": 3},
            **override}
    reply = {"status": status, "body": base64.b64encode(json.dumps(body).encode()).decode()}
    assert container_execution._budget_termination(reply) is None


def test_workspace_slot_cross_process_and_cancel_release(tmp_path: Path) -> None:
    """A separate process cannot acquire an occupied slot; cancelled owners release it."""
    script = """
import asyncio
import sys
from pathlib import Path
from rl.agentic.envs.docker_workspace import DockerWorkspace, acquire_workspace_slot
loaded = [name for name in ('torch', 'transformers') if name in sys.modules]
assert not loaded, f'Expected a framework-free workspace import, got {loaded}'
async def run():
    try:
        async with acquire_workspace_slot(Path(sys.argv[1]), 1, 0.15):
            print('acquired', flush=True)
    except RuntimeError:
        print('occupied', flush=True)
asyncio.run(run())
"""

    async def child() -> bytes:
        """Attempt the shared lease from an independent Python process."""
        environment = os.environ.copy()
        source_root = str(Path(find_spec("rl").origin).resolve().parent.parent)
        # Pytest configures its own import path; wheel-based CI children need the same RL source root.
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, environment.get("PYTHONPATH"))))
        process = await asyncio.create_subprocess_exec(sys.executable, "-c", script, str(tmp_path),
                                                      stdout=asyncio.subprocess.PIPE,
                                                      stderr=asyncio.subprocess.PIPE, env=environment)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
        finally:
            if process.returncode is None:
                process.kill()
                await process.communicate()
        assert process.returncode == 0, stderr.decode()
        return stdout.strip()

    async def scenario() -> None:
        """Exercise the contract with bounded real processes and sockets."""
        entered = asyncio.Event()

        async def hold() -> None:
            """Retain the lease until its owner is cancelled."""
            async with acquire_workspace_slot(tmp_path, 1, 1):
                entered.set()
                await asyncio.Event().wait()

        owner = asyncio.create_task(hold())
        await entered.wait()
        try:
            assert await child() == b"occupied"
        finally:
            owner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await owner
        assert await child() == b"acquired"

    asyncio.run(scenario())


def test_relay_rejects_oversized_candidate_before_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Oversized local requests terminate the session without opening an upstream connection."""
    failures = _local_endpoint(monkeypatch)

    async def scenario() -> None:
        """Exercise the contract with bounded real processes and sockets."""
        workspace = DockerWorkspace("task-image", docker_host="ssh://user@127.0.0.1:2233")
        relay = container_execution.CandidateRelay(workspace, "http://127.0.0.1:9", "bound",
                                          request_timeout=2, max_request_bytes=4, max_response_bytes=1024,
                                          admin_token="controller-secret")
        endpoint = await relay.start()
        assert await _request(endpoint, "/responses", b"12345") == b""
        await asyncio.wait_for(relay._pump_task, 3)
        with pytest.raises(container_execution.RelayError):
            await relay.close()
        assert len(failures) == 1
        assert relay._process.returncode is not None

    asyncio.run(scenario())
