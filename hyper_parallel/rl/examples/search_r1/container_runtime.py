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
"""Per-episode container handoff and session-scoped Unix-socket Responses relay."""

from __future__ import annotations

import argparse
import asyncio
import http.client
import json
import logging
import os
from pathlib import Path
import socketserver
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

from rl.agentic.codex.container_execution import CandidateRelay, run_container_codex
from rl.agentic.envs.docker_workspace import DockerWorkspace


_LOGGER = logging.getLogger(__name__)


class SearchBudgetExceeded(ValueError):
    """A terminal local policy outcome, not a retryable upstream failure."""


def is_context_overflow(status: int, body: bytes) -> bool:
    """Recognize explicit length rejections only; all other server errors remain fatal."""
    message = body.decode("utf-8", errors="replace").lower()
    if status == 502:
        try:
            error = json.loads(body)["error"]
            message = error["message"].lower()
            if error.get("type") != "gateway_error" or not message.startswith(
                "vllm chat completion failed with http 400:"
            ):
                return False
        except (ValueError, KeyError, TypeError, AttributeError):
            return False
    elif status != 400:
        return False
    return any(marker in message for marker in (
        "maximum context length", "longer than the maximum model length",
        "exceeds the model's maximum context length",
    ))


class SearchFeedback:
    """Record citation observations and reserve the final answer call."""

    def __init__(self, artifact: Path, max_calls: int = 10) -> None:
        """Initialize episode-local observations and the model-call limit."""
        if max_calls < 2:
            raise ValueError("Search needs at least two model calls")
        self.artifact = artifact
        self.max_calls = max_calls
        self.calls = 0
        self.seen = set()
        self.visible_outputs = []

    def _record_observations(self, items: list[dict]) -> None:
        """Keep tool observations for citation checks without interpreting shell results."""
        for item in items:
            if item.get("type") != "function_call_output":
                continue
            key = item.get("call_id") or json.dumps(item, sort_keys=True)
            if key in self.seen:
                continue
            self.seen.add(key)
            output = item.get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            self.visible_outputs.append(output)
            (self.artifact / "search-observations.json").write_text(
                json.dumps(self.visible_outputs, ensure_ascii=False)
            )

    def prepare(self, body: bytes) -> bytes:
        """Annotate new observations, reserving the final call for an answer."""
        if self.calls >= self.max_calls:
            self.stop("budget_exhausted")
        payload = json.loads(body)
        items = payload.get("input", [])
        if not isinstance(items, list):
            raise ValueError("Search-R1 requires explicit Responses input history")
        self._record_observations(items)
        notes = []
        self.calls += 1
        if self.calls == self.max_calls:
            payload.update(tools=[], tool_choice="none")
            notes.append("FINAL MODEL CALL: do not call tools. Answer with verified REF citations, "
                         "or ANSWER: UNKNOWN followed by SOURCES: NONE if evidence is insufficient.")
        if notes:
            payload["input"] = items + [{"type": "message", "role": "user",
                                         "content": "Search execution feedback:\n" + "\n".join(notes)}]
        (self.artifact / "search-last-request.json").write_text(json.dumps(payload, ensure_ascii=False))
        with (self.artifact / "search-feedback.jsonl").open("a") as stream:
            stream.write(json.dumps({"call": self.calls, "notes": notes}, ensure_ascii=False) + "\n")
        return json.dumps(payload, ensure_ascii=False).encode()

    def stop(self, reason: str, origin: str = "model") -> None:
        """Persist a classified failure without inventing an assistant action."""
        self.record_terminal(reason, origin)
        raise SearchBudgetExceeded(reason)

    def record_terminal(self, reason: str, origin: str, **details: object) -> None:
        """Persist the single terminal contract consumed by rollout scoring."""
        record = {
            "reason": reason,
            "calls": self.calls,
            "failure_origin": origin,
            "failure_reason": reason,
            "trainable": origin == "model",
            **details,
        }
        (self.artifact / "search-terminal.json").write_text(json.dumps(record))


CANDIDATE_IMAGE = "hyper-rl-codex:0.152.1-arm64"

IMAGE = "swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.23.0.post1-code-agent-arm64"


PROBE = r'''
import http.client, json, os, pathlib, socket, shutil
from urllib.parse import urlparse
checks = {"non_root": os.geteuid() != 0}
checks["search_tools_available"] = all(shutil.which(name) for name in ("rg", "sed", "cat", "codex"))
corpus = pathlib.Path("/work/articles")
document = next(path for path in corpus.iterdir() if path.is_file())
checks["own_corpus_readable"] = bool(document.read_bytes())
for name in ("/data/test.parquet", "/data/train.parquet", "/models/Qwen3-4B/config.json",
             "/repo/hyper_parallel/rl/train_rl.py", "/results/private-canary.txt",
             "/var/run/docker.sock", "/proc/1/root/data/test.parquet"):
    checks["hidden:" + name] = not pathlib.Path(name).exists()
try:
    with document.open("a"):
        pass
    checks["corpus_readonly"] = False
except OSError:
    checks["corpus_readonly"] = True
try:
    pathlib.Path("/outside-workspace").touch()
    checks["root_readonly"] = False
except OSError:
    checks["root_readonly"] = True
try:
    with socket.create_connection(("1.1.1.1", 443), timeout=1):
        checks["external_network_blocked"] = False
except OSError:
    checks["external_network_blocked"] = True
for label, method, path, token in (
    ("administration_blocked", "GET", "/internal/sessions/anything", os.environ["OPENAI_API_KEY"]),
    ("cross_session_blocked", "POST", "/responses", "another-session"),
):
    endpoint = urlparse(os.environ["HYPER_CODEX_ENDPOINT"])
    connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=3)
    connection.request(method, path, b"{}", {"Authorization": "Bearer " + token})
    response = connection.getresponse()
    checks[label] = response.status == 403
    response.read()
    connection.close()
print(json.dumps(checks))
raise SystemExit(0 if all(checks.values()) else 1)
'''


def docker(arguments: list, **kwargs: object) -> subprocess.CompletedProcess:
    """Execute Docker without a shell so model arguments cannot affect the host."""
    return subprocess.run(["docker", *arguments], text=True, check=True, **kwargs)


def episode_workspace(request: dict, output: Path, image: str, prefix: str) -> DockerWorkspace:
    """Mount only this candidate's files into the shared read-only workspace backend."""
    def host_path(value: str) -> Path:
        """Resolve a controller-created artifact path within this run."""
        relative = Path(value).relative_to("/results")
        resolved = (output / relative).resolve(strict=True)
        if not resolved.is_relative_to(output.resolve()):
            raise ValueError("Episode mount escapes the result directory")
        return resolved

    workspace = host_path(request["workspace"])
    home = host_path(request["codex_home"])
    return DockerWorkspace(
        image, name_prefix=prefix, user=f"{os.getuid()}:{os.getgid()}", read_only=True, runtime="runc",
        memory="2g", cpus=2, pids_limit=128,
        mounts=((workspace, "/work", False), (workspace / "articles", "/work/articles", True),
                (home, "/codex-home", False)),
    )


async def _execute_episode(request_file: Path, output: Path, image: str, prefix: str,
                           stop: threading.Event | None) -> None:
    """Run one isolated episode and persist its result and transport evidence."""
    request = json.loads(request_file.read_text(encoding="utf-8"))
    artifact = request_file.parent
    workspace = episode_workspace(request, output, image, prefix)
    socket_path = (output / Path(request["bridge"]).relative_to("/results") / "responses.sock").resolve()
    if not socket_path.is_relative_to(output.resolve()):
        raise ValueError("Episode bridge escapes the result directory")
    terminal_file = "/tmp/codex-terminal"
    relay = CandidateRelay(
        workspace, "http://localhost", request["session_id"], unix_socket=socket_path,
        require_token=True, terminal_header="X-Search-Terminal", terminal_file=terminal_file,
        python_executable="python3",
    )
    timing = {"queued_seconds": max(0.0, time.time() - request_file.stat().st_mtime)}
    started = time.perf_counter()
    result = None
    try:
        phase = time.perf_counter()
        await workspace.start()
        timing["container_start_seconds"] = time.perf_counter() - phase
        (artifact / "candidate-container.json").write_text(json.dumps({"name": workspace.name}))
        phase = time.perf_counter()
        endpoint = await relay.start()
        timing["relay_start_seconds"] = time.perf_counter() - phase
        environment = {
            "HOME": "/codex-home", "CODEX_HOME": "/codex-home", "SHELL": "/bin/bash",
            "OPENAI_API_KEY": request["session_id"], "HYPER_CODEX_ENDPOINT": endpoint,
            "PATH": "/opt/codex/bin:/opt/codex/codex-path:/usr/local/bin:/usr/bin:/bin",
        }
        phase = time.perf_counter()
        probe = await workspace.exec(["python3", "-I", "-c", PROBE], cwd="/work", env=environment)
        (artifact / "isolation.json").write_text(probe.stdout)
        (artifact / "isolation-stderr.log").write_text(probe.stderr)
        if probe.returncode:
            raise RuntimeError(f"Isolation probe failed; see {artifact / 'isolation-stderr.log'}")
        timing["isolation_seconds"] = time.perf_counter() - phase
        command = list(request["command"])
        for index, value in enumerate(command):
            if value.startswith("model_providers.hyper_rl.base_url="):
                command[index] = f"model_providers.hyper_rl.base_url={json.dumps(endpoint)}"
        phase = time.perf_counter()
        # The controller owns its stdout files; the host broker must write separate raw evidence.
        execution_artifacts = artifact / "container"
        execution_artifacts.mkdir(exist_ok=True)
        execution = asyncio.create_task(run_container_codex(
            workspace, command, execution_artifacts, environment=environment, timeout=850, cwd="/work",
            terminal_file=terminal_file, python_executable="python3",
        ))
        try:
            while not execution.done():
                relay.raise_if_failed()
                if stop is not None and stop.is_set():
                    raise RuntimeError("Episode cancelled by host shutdown")
                await asyncio.wait((execution,), timeout=0.2)
            result = await execution
        finally:
            if not execution.done():
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
        timing["cli_seconds"] = time.perf_counter() - phase
        phase = time.perf_counter()
        await relay.close()
        timing["relay_close_seconds"] = time.perf_counter() - phase
        inspected = await workspace.inspect()
        (artifact / "container-boundary.json").write_text(json.dumps({
            key: inspected[key] for key in ("HostConfig", "Mounts", "State")
        }, indent=2))
    except BaseException as error:
        pending = artifact / "execution-result.tmp"
        pending.write_text(json.dumps({"returncode": 1, "error": str(error)}))
        pending.replace(artifact / "execution-result.json")
        raise
    finally:
        primary_error = sys.exc_info()[1]
        phase = time.perf_counter()
        try:
            await relay.close(abort=primary_error is not None)
        except Exception:  # pylint: disable=broad-exception-caught
            if primary_error is None:
                raise
            _LOGGER.exception("Candidate relay cleanup failed after an execution error")
        finally:
            try:
                await workspace.close()
            except Exception:  # pylint: disable=broad-exception-caught
                if primary_error is None:
                    raise
                _LOGGER.exception("Candidate workspace cleanup failed after an execution error")
            timing["cleanup_seconds"] = time.perf_counter() - phase
            timing["total_seconds"] = time.perf_counter() - started
            (artifact / "execution-timing.json").write_text(json.dumps(timing, indent=2))
    pending = artifact / "execution-result.tmp"
    pending.write_text(json.dumps({"returncode": result.returncode}))
    pending.replace(artifact / "execution-result.json")


def execute_episode(request_file: Path, output: Path, image: str, prefix: str,
                    stop: threading.Event | None = None) -> None:
    """Run one candidate using the common workspace, relay and CLI execution lifecycle."""
    if stop is not None and stop.is_set():
        return
    asyncio.run(_execute_episode(request_file, output, image, prefix, stop))


_BRIDGES = []
_BRIDGES_LOCK = threading.Lock()


class UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """Serve current-session requests on a filesystem socket."""

    daemon_threads = True


class RelayHandler(BaseHTTPRequestHandler):
    """Reject administrative routes and other sessions before forwarding."""

    def _forward_response(self, body: bytes, expected: str, policy) -> None:
        """Relay one bounded response and preserve terminal classification."""
        connection = self.server.connection_factory()
        try:
            connection.request("POST", "/responses", body, {
                "Content-Type": "application/json", "Authorization": expected,
            })
            response = connection.getresponse()
            content = response.read()
            if policy is not None and response.getheader("X-Model-Calls"):
                policy.calls = int(response.getheader("X-Model-Calls"))
            terminal_reason = response.getheader("X-Search-Terminal")
            if policy is not None and response.status >= 400:
                policy.calls -= 1
                terminal_reason = (
                    "context_exhausted" if is_context_overflow(response.status, content) else "upstream_error"
                )
                classification = {"failure_origin": "infrastructure",
                                  "failure_reason": terminal_reason, "trainable": False}
                try:
                    error = json.loads(content).get("error", {})
                    if isinstance(error, dict) and error.get("type") == "tool_protocol_error":
                        classification = {key: error[key] for key in classification}
                        terminal_reason = classification["failure_reason"]
                        policy.calls = int(error["model_calls"])
                except (ValueError, KeyError, AttributeError):
                    pass
                policy.record_terminal(
                    classification["failure_reason"],
                    classification["failure_origin"],
                    http_status=response.status,
                )
            if response.getheader("X-Search-Terminal"):
                event = getattr(self.server, "terminal_event", None)
                if event is not None:
                    event.set()
            self.send_response(response.status)
            if terminal_reason:
                self.send_header("X-Search-Terminal", terminal_reason)
            self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        finally:
            connection.close()

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Forward only bounded Responses requests with the pinned bearer token."""
        expected = "Bearer " + self.server.session_id
        if self.path not in {"/responses", "/v1/responses"} or self.headers.get("Authorization") != expected:
            self.send_error(403)
            return
        policy = getattr(self.server, "search_policy", None)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4 * 1024 * 1024:
                self.send_error(413)
                return
            body = self.rfile.read(length)
            if policy is not None and (policy.artifact / "search-terminal.json").exists():
                reason = json.loads((policy.artifact / "search-terminal.json").read_text())["reason"]
                self._terminal_response(reason)
                return
            if policy is not None:
                try:
                    body = policy.prepare(body)
                except SearchBudgetExceeded as error:
                    self._terminal_response(str(error), 409)
                    return
            self._forward_response(body, expected, policy)
        except (OSError, ValueError, http.client.HTTPException) as error:
            if policy is not None:
                policy.record_terminal(
                    "upstream_error", "infrastructure", error=str(error)
                )
            self._terminal_response("upstream_error")

    def _terminal_response(self, reason: str, status: int = 502) -> None:
        """Stop retries on infrastructure failure; never fabricate a task reward."""
        content = json.dumps({"error": reason}).encode()
        event = getattr(self.server, "terminal_event", None)
        if event is not None:
            event.set()
        self.send_response(status)
        self.send_header("X-Search-Terminal", reason)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # pylint: disable=invalid-name
        """Deny all inspection routes from the agent side."""
        self.send_error(403)

    def do_DELETE(self) -> None:  # pylint: disable=invalid-name
        """Deny session administration from the agent side."""
        self.send_error(403)

    def log_message(self, format: str, *args: object) -> None:  # pylint: disable=redefined-builtin
        """Keep transport chatter out of Codex JSONL output."""


def close_bridges(session_id: str | None = None) -> None:
    """Release completed episode relay threads."""
    with _BRIDGES_LOCK:
        selected = [(server, thread) for server, thread in _BRIDGES
                    if session_id is None or server.session_id == session_id]
        for pair in selected:
            _BRIDGES.remove(pair)
    for server, thread in selected:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request_container_execution(
    command: list[str], workspace_dir: Path, codex_home: Path, environment: dict[str, str]
) -> tuple[list[str], Path, dict[str, str]]:
    """Submit one trusted launch request to the host-side Docker broker.

    Only the current workspace, Codex state, and session socket are mounted
    into the child. Dataset labels and gateway administration stay outside.
    """
    if os.environ.get("HYPER_SEARCH_CONTAINER_BROKER") != "1":
        raise RuntimeError(
            "Search-R1 requires a host container broker; use examples/search_r1/launcher.py"
        )
    upstream = urlparse(environment.get("HYPER_CODEX_GATEWAY_URL", "http://127.0.0.1:8200"))
    if upstream.scheme != "http" or upstream.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Search-R1 supports only a loopback HTTP gateway in the trusted worker")
    max_calls = int(environment.get("HYPER_CODEX_MAX_CALLS", os.environ.get("PHASE_MAX_CALLS", "10")))
    if max_calls < 3:
        raise ValueError("Search-R1 requires at least three calls including citation repair")
    artifact_dir = workspace_dir.parent
    bridge_dir = artifact_dir / "bridge"
    bridge_dir.mkdir()
    socket_path = bridge_dir / "responses.sock"
    server = UnixServer(str(socket_path), RelayHandler)
    server.session_id = environment["OPENAI_API_KEY"]
    server.connection_factory = lambda: http.client.HTTPConnection(upstream.hostname, upstream.port or 80, timeout=600)
    if (workspace_dir / "articles").is_dir():
        server.search_policy = SearchFeedback(artifact_dir, max_calls - 1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with _BRIDGES_LOCK:
        _BRIDGES.append((server, thread))
    uid = int(os.environ["PHASE_UID"])
    gid = int(os.environ["PHASE_GID"])
    os.chown(artifact_dir, uid, gid)
    for directory in (workspace_dir, codex_home, bridge_dir):
        for path in [directory, *directory.rglob("*")]:
            if path.is_symlink():
                raise ValueError("Episode mounts must not contain symlinks")
            os.chown(path, uid, gid)
    request = {
        "command": command[:2] + ["-c", 'model_providers.hyper_rl.base_url="http://127.0.0.1:8200"'] + command[2:],
        "workspace": str(workspace_dir), "codex_home": str(codex_home),
        "bridge": str(bridge_dir), "session_id": server.session_id,
    }
    pending = artifact_dir / "execution-request.tmp"
    pending.write_text(json.dumps(request), encoding="utf-8")
    pending.replace(artifact_dir / "execution-request.json")
    return [sys.executable, str(Path(__file__).resolve()), "--wait", str(artifact_dir)], workspace_dir, environment


def main() -> None:
    """Wait for the host broker and forward its retained execution evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", type=Path)
    args = parser.parse_args()
    if args.wait:
        result_path = args.wait / "execution-result.json"
        deadline = time.monotonic() + 900
        while not result_path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Host container broker did not finish the episode")
            time.sleep(0.2)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for name, stream in (("codex-events.jsonl", sys.stdout), ("codex-stderr.log", sys.stderr)):
            path = args.wait / "container" / name
            if path.exists():
                stream.write(path.read_text(encoding="utf-8"))
        if result.get("error"):
            sys.stderr.write("Candidate execution failed: " + result["error"] + "\n")
        raise SystemExit(result["returncode"])
    parser.error("Only --wait is supported; candidates use the shared Codex executor")


if __name__ == "__main__":
    main()
