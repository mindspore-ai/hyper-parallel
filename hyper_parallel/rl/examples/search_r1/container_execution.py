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
import http.client
import json
import os
import re
import shlex
from pathlib import Path
import socket
import socketserver
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

class SearchBudgetExceeded(ValueError):
    """A terminal local policy outcome, not a retryable upstream failure."""


def _simple_shell_command(command: str) -> tuple[list[str], bool]:
    """Tokenize shell punctuation while preserving punctuation inside quotes."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return [], False
    controls = {"|", "||", "&", "&&", ";", ";;", "<", "<<", ">", ">>"}
    simple = bool(tokens) and not any(token in controls for token in tokens)
    # Command substitution remains out of scope; quoted pipes are already handled by shlex.
    if "`" in command or "$(" in command or "\n" in command:
        simple = False
    return tokens, simple


def classify_command_result(command: str, output: str) -> dict:
    """Conservatively classify explicit execution status, not arbitrary shell semantics."""
    match = re.search(r"^(?:Process exited with code|exit code[:=]?)\s*(\d+)\s*$", output, re.I | re.M)
    if not match or int(match[1]) == 0:
        return {"failure_origin": None, "failure_reason": None, "trainable": True}
    code = int(match[1])
    args, simple = _simple_shell_command(command)
    program = args[0] if args else ""
    if code in (126, 127, 137, 143) or any(marker in output.lower() for marker in (
        "input/output error", "cannot allocate memory", "transport endpoint", "no space left on device",
    )):
        return {"failure_origin": "infrastructure", "failure_reason": "execution_environment", "trainable": False}
    if simple and program == "rg" and code == 1:
        return {"failure_origin": "model", "failure_reason": "no_match", "trainable": True}
    lowered = output.lower()
    if code == 2 and "bash: -c:" in lowered and any(
        marker in lowered for marker in ("syntax error", "unexpected eof")
    ):
        return {"failure_origin": "model", "failure_reason": "invalid_command", "trainable": True}
    if simple and program in ("rg", "sed", "cat", "head", "tail", "ls") and any(
        marker in lowered for marker in ("unrecognized", "invalid option", "regex parse error", "no such file")
    ):
        return {"failure_origin": "model", "failure_reason": "invalid_command", "trainable": True}
    # Anything not proven to be a model action is conservatively infrastructure.
    # This keeps the runtime decision binary and prevents uncertain failures from
    # entering reward normalization.
    return {"failure_origin": "infrastructure", "failure_reason": "execution_environment", "trainable": False}


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
    """Append control feedback before forwarding; leave tool outputs unchanged."""

    def __init__(self, artifact: Path, max_calls: int = 10) -> None:
        if max_calls < 2:
            raise ValueError("Search needs at least two model calls")
        self.artifact = artifact
        self.max_calls = max_calls
        self.calls = 0
        self.seen = set()
        self.failed_commands = {}
        self.visible_outputs = []
        self.recovery_note = ""
        self.consecutive_failures = 0
        self.commands = {}
        self.workdirs = {}

    def _process_inputs(self, items: list[dict]) -> list[str]:
        """Record new tool observations and build recovery feedback."""
        notes = []
        commands = self.commands
        for item in items:
            if item.get("type") == "function_call":
                try:
                    arguments = json.loads(item.get("arguments", "{}"))
                    commands[item.get("call_id")] = arguments.get("cmd", arguments.get("command", ""))
                    self.workdirs[item.get("call_id")] = arguments.get("workdir", "/work")
                except (ValueError, AttributeError):
                    continue
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
            command = commands.get(item.get("call_id"), "")
            classification = classify_command_result(command, output)
            with (self.artifact / "tool-outcomes.jsonl").open("a") as stream:
                stream.write(json.dumps({"call_id": key, "command": command,
                    "workdir": self.workdirs.get(key), **classification}) + "\n")
            if not classification["trainable"]:
                self.stop(classification["failure_reason"], classification["failure_origin"])
            if classification["failure_reason"]:
                self.consecutive_failures += 1
                if command:
                    identity = (self.workdirs.get(key), command)
                    count = self.failed_commands.get(identity, 0) + 1
                    self.failed_commands[identity] = count
                    if count >= 3:
                        self.stop("repeated_tool_failure")
                self.recovery_note = self.recovery(command)
                if classification["failure_reason"] == "no_match":
                    self.recovery_note = ("Search completed with no matches (rg exit 1), not an environment failure. "
                                          + self.recovery_note)
            elif re.search(r"(?:exited with code|exit code[:=]?)\s*0\b", output, re.I):
                self.consecutive_failures = 0
                self.recovery_note = ""
            if "No such file" in output:
                notes.append("The article title is NOT a filename. Use rg '^TITLE:' articles to obtain a real path.")
            if "truncated" in output.lower() or "truncation" in output.lower():
                notes.append("Output was truncated. Read a smaller range; omitted text is not evidence.")
        return notes

    def prepare(self, body: bytes) -> bytes:
        """Annotate new observations, reserving the final call for an answer."""
        if self.calls >= self.max_calls:
            self.stop("budget_exhausted")
        payload = json.loads(body)
        items = payload.get("input", [])
        if not isinstance(items, list):
            raise ValueError("Search-R1 requires explicit Responses input history")
        notes = self._process_inputs(items)
        self.calls += 1
        if self.recovery_note:
            notes.append(self.recovery_note)
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

    def recovery(self, command: str) -> str:
        """Suggest a command derived only from the failed query, never from gold labels."""
        fallback = "rg '^TITLE:' articles"
        try:
            arguments = shlex.split(command)
        except ValueError:
            arguments = []
        if self.consecutive_failures == 1 and "--" in arguments:
            query_index = arguments.index("--") + 1
            if query_index < len(arguments):
                words = re.findall(r"[^\W\d_]+", arguments[query_index], re.UNICODE)
                if words:
                    fallback = "rg -i -F -m 5 -- " + shlex.quote(words[0]) + " articles"
        return ("Previous command failed. Do not repeat it. Next use exec_command with cmd="
                + json.dumps(fallback) + ". Execute this yourself; no search has been performed for you. "
                "Use returned filenames, not article titles, for the following read.")

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


IMAGE = "hyper-parallel/hyper-rl:v0.22.1rc1-unified-arm64"
PROBE = r'''
import importlib.util, json, os, pathlib, socket, shutil
spec = importlib.util.spec_from_file_location("relay", "/runtime/relay.py")
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)
checks = {}
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
    connection = relay.UnixConnection("/bridge/responses.sock", timeout=3)
    connection.request(method, path, b"{}", {"Authorization": "Bearer " + token})
    response = connection.getresponse()
    checks[label] = response.status == 403
    response.read()
    connection.close()
print(json.dumps(checks))
raise SystemExit(0 if all(checks.values()) else 1)
'''


def docker(arguments: list, **kwargs) -> subprocess.CompletedProcess:
    """Execute Docker without a shell so model arguments cannot affect the host."""
    return subprocess.run(["docker", *arguments], text=True, check=True, **kwargs)


def episode_options(request: dict, output: Path, image: str, name: str) -> list:
    """Build an unprivileged, networkless container exposing one episode only."""
    def host_path(value: str) -> Path:
        relative = Path(value).relative_to("/results")
        resolved = (output / relative).resolve()
        if not resolved.is_relative_to(output.resolve()):
            raise ValueError("Episode mount escapes the result directory")
        return resolved

    workspace = host_path(request["workspace"])
    codex_home = host_path(request["codex_home"])
    bridge = host_path(request["bridge"])
    script = Path(__file__).with_name("container_execution.py").resolve()
    options = [
        "--name", name, "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--user", f"{os.getuid()}:{os.getgid()}",
        "--pids-limit", "128", "--memory", "2g", "--cpus", "2",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
        "-v", f"{workspace}:/work:rw", "-v", f"{codex_home}:/codex-home:rw",
        "-v", f"{bridge}:/bridge:ro", "-v", f"{script}:/runtime/relay.py:ro",
        "-e", "HOME=/codex-home", "-e", "CODEX_HOME=/codex-home", "-e", "PYTHONPATH=",
        "-e", f"OPENAI_API_KEY={request['session_id']}", "-w", "/work", "--entrypoint", "python",
    ]
    for directory in ("/home", "/root", "/workspace", "/data", "/models", "/results"):
        options.extend(("--tmpfs", f"{directory}:rw,nosuid,nodev,size=1m,mode=755"))
    corpus_name = "articles" if (workspace / "articles").is_dir() else "corpus"
    options.extend(("-v", f"{workspace / corpus_name}:/work/{corpus_name}:ro", image))
    return options


def execute_episode(request_file: Path, output: Path, image: str, prefix: str,
                    stop: threading.Event | None = None) -> None:
    """Probe the exact container boundary, execute Codex, and retain its events."""
    request = json.loads(request_file.read_text())
    artifact = request_file.parent
    name = prefix + "-agent"
    options = episode_options(request, output, image, name)
    if stop is not None and stop.is_set():
        return
    command = ["run", *options, "/runtime/relay.py", "--", *request["command"]]
    try:
        probe = subprocess.run(["docker", "run", "--rm", *options, "-c", PROBE],
                               capture_output=True, text=True, timeout=30, check=False)
        (artifact / "isolation.json").write_text(probe.stdout)
        (artifact / "isolation-stderr.log").write_text(probe.stderr)
        if probe.returncode:
            raise RuntimeError(f"Isolation probe failed; see {artifact / 'isolation-stderr.log'}")
        if stop is not None and stop.is_set():
            return
        with (artifact / "container-events.jsonl").open("w") as stdout:
            with (artifact / "container-stderr.log").open("w") as stderr:
                with subprocess.Popen(["docker", *command], stdout=stdout, stderr=stderr) as process:
                    deadline = time.monotonic() + 850
                    try:
                        while process.poll() is None:
                            if stop is not None and stop.is_set():
                                raise RuntimeError("Episode cancelled by host shutdown")
                            if time.monotonic() >= deadline:
                                raise TimeoutError("Episode execution deadline exceeded")
                            time.sleep(0.2)
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=5)
        inspected = json.loads(docker(["inspect", name], capture_output=True).stdout)[0]
        (artifact / "container-boundary.json").write_text(json.dumps({
            "HostConfig": inspected["HostConfig"], "Mounts": inspected["Mounts"],
        }, indent=2))
        result = {"returncode": process.returncode}
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
    pending = artifact / "execution-result.tmp"
    pending.write_text(json.dumps(result))
    pending.replace(artifact / "execution-result.json")


_BRIDGES = []
_BRIDGES_LOCK = threading.Lock()


class UnixConnection(http.client.HTTPConnection):
    """Send HTTP over the only socket exposed to an agent container."""

    def connect(self) -> None:
        """Connect without granting the container a network interface."""
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.host)


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
    """Run the private client relay or await a host-broker completion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.wait:
        result_path = args.wait / "execution-result.json"
        deadline = time.monotonic() + 900
        while not result_path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Host container broker did not finish the episode")
            time.sleep(0.2)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        sys.stdout.write((args.wait / "container-events.jsonl").read_text(encoding="utf-8"))
        sys.stderr.write((args.wait / "container-stderr.log").read_text(encoding="utf-8"))
        raise SystemExit(result["returncode"])
    server = ThreadingHTTPServer(("127.0.0.1", 8200), RelayHandler)
    server.session_id = os.environ["OPENAI_API_KEY"]
    server.connection_factory = lambda: UnixConnection("/bridge/responses.sock", timeout=600)
    server.terminal_event = threading.Event()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        with subprocess.Popen(command, start_new_session=True) as process:
            while process.poll() is None:
                if server.terminal_event.wait(0.05):
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise SystemExit(75)
            raise SystemExit(process.returncode)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
