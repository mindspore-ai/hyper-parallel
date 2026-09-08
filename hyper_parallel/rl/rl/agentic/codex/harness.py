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
"""Codex 0.152.1 process harness and Hyper-RL AgentProgram implementation."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional
from urllib.parse import urlparse

from hyper_parallel import get_platform
from rl.agentic.codex.gateway import CodexGateway
from rl.agentic.codex.trajectory import build_codex_trajectory
from rl.agentic.core.types import RewardResult
from rl.dataset.contracts import PromptRecord, Trajectory


platform = get_platform()
DEFAULT_CODEX_VERSION = "0.152.1"
RewardCallable = Callable[[str, PromptRecord], float | RewardResult]
_MODEL_METADATA_FALLBACK = re.compile(
    r"^Model metadata for `[^`]+` not found\. Defaulting to fallback metadata;"
)


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _load_reward_callable(value: Any) -> RewardCallable:
    if not isinstance(value, str) or ":" not in value:
        raise ValueError("agentic.codex.reward_callable must use 'module:function' syntax")
    module_name, attribute = value.rsplit(":", 1)
    callback = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(callback):
        raise ValueError(f"Codex reward callable is not callable: {value}")
    return callback


def _classify_codex_events(
    lines: Iterable[str],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract the final answer and separate recoverable from fatal events."""
    final_answer = ""
    diagnostics: list[dict[str, Any]] = []
    failed_events: list[dict[str, Any]] = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        item = event.get("item")
        if (
            event.get("type") == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "agent_message"
        ):
            final_answer = str(item.get("text", ""))
        if event.get("type") in {"turn.failed", "error"}:
            failed_events.append(event)
            continue
        if not (
            event.get("type") == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "error"
        ):
            continue
        message = str(item.get("message", ""))
        if _MODEL_METADATA_FALLBACK.search(message):
            diagnostics.append(event)
        else:
            failed_events.append(event)
    return final_answer, diagnostics, failed_events


def _http_json(
    method: str,
    url: str,
    payload: Optional[Mapping[str, Any]],
    timeout: float,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Codex gateway HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Codex gateway request failed: {error.reason}") from error
    try:
        decoded = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Codex gateway returned invalid JSON") from error
    if not isinstance(decoded, dict):
        raise RuntimeError("Codex gateway returned a non-object response")
    return decoded


class CodexRuntime:
    """Own one node-level gateway while Hyper-RL continues to own vLLM."""

    def __init__(self, engine: Any, config: Mapping[str, Any]) -> None:
        """Bind the runtime to the existing shared rollout engine."""
        self.engine = engine
        self.config = dict(config)
        self._gateway: Optional[CodexGateway] = None
        self._episode_identity: Optional[tuple[int, str]] = None
        host = str(self.config.get("gateway_host", "127.0.0.1"))
        public_host = str(self.config.get("gateway_public_host", host))
        port = int(self.config.get("gateway_port", 8200))
        self.gateway_url = f"http://{public_host}:{port}"

    def ensure_started(self) -> None:
        """Materialize vLLM, then start exactly one protocol gateway."""
        backend_url = self.engine.inference_base_url
        local_error = None
        if platform.get_rank() == 0 and self._gateway is None:
            try:
                self._gateway = CodexGateway(
                    host=str(self.config.get("gateway_host", "127.0.0.1")),
                    port=int(self.config.get("gateway_port", 8200)),
                    backend_url=backend_url,
                    model_name=self.engine.inference_model_name,
                    request_timeout=float(self.config.get("request_timeout", 600.0)),
                )
                self._gateway.start()
            except Exception as error:  # pylint: disable=W0718
                local_error = error
        self.engine.synchronize_error(local_error, "Codex gateway startup")
        platform.barrier()

    def close(self) -> None:
        """Stop the gateway on its sole owning rank."""
        if self._gateway is not None:
            self._gateway.close()
            self._gateway = None

    def bind_episode_identity(self, identity: tuple[int, str]) -> None:
        """Publish an all-rank-verified identity to request-owner factories."""
        version, fingerprint = identity
        if version < 0 or not fingerprint:
            raise ValueError("Codex episode requires a valid policy identity")
        self._episode_identity = (int(version), str(fingerprint))

    def clear_episode_identity(self) -> None:
        """Prevent a completed rollout identity from leaking into the next step."""
        self._episode_identity = None

    @property
    def episode_identity(self) -> tuple[int, str]:
        """Return the identity established collectively before harness launch."""
        if self._episode_identity is None:
            raise RuntimeError("Codex episode policy identity has not been established")
        return self._episode_identity


class CodexAgentProgram:
    """Run one complete Codex tool loop and return one immutable trajectory."""

    def __init__(
        self,
        prompt: PromptRecord,
        policy_version: int,
        sample_index: int,
        policy_fingerprint: str,
        gateway_url: str,
        config: Mapping[str, Any],
        end_of_turn_token_id: Optional[int],
    ) -> None:
        """Capture the episode identity and immutable harness settings."""
        self.prompt = prompt
        self.policy_version = policy_version
        self.sample_index = sample_index
        self.policy_fingerprint = policy_fingerprint
        self.gateway_url = gateway_url.rstrip("/")
        self.config = dict(config)
        self.end_of_turn_token_id = end_of_turn_token_id
        self.reward_callable = _load_reward_callable(self.config.get("reward_callable"))

    async def run(self) -> Trajectory:
        """Run Codex, fetch the captured network trace, score it, and convert it."""
        session_id = uuid.uuid4().hex
        artifact_dir, workspace_dir, codex_home = self._prepare_directories(session_id)
        await asyncio.to_thread(
            _http_json,
            "POST",
            f"{self.gateway_url}/internal/sessions",
            {
                "session_id": session_id,
                "policy_version": self.policy_version,
                "policy_fingerprint": self.policy_fingerprint,
                "artifact_dir": str(artifact_dir),
                "max_completions": int(self.config["max_turns"]),
                "generation": self._generation_settings(),
            },
            float(self.config.get("request_timeout", 600.0)),
        )
        self._write_codex_config(codex_home, session_id)
        await self._validate_version()
        started = time.perf_counter()
        final_answer, return_code, diagnostic_events = await self._run_codex(
            session_id,
            artifact_dir,
            workspace_dir,
            codex_home,
        )
        if return_code != 0:
            raise RuntimeError(f"Codex exited with status {return_code}; see {artifact_dir}")
        captured = await asyncio.to_thread(
            _http_json,
            "GET",
            f"{self.gateway_url}/internal/sessions/{session_id}",
            None,
            float(self.config.get("request_timeout", 600.0)),
        )
        if captured.get("policy_version") != self.policy_version:
            raise RuntimeError("Codex gateway returned a different policy version")
        if captured.get("policy_fingerprint") != self.policy_fingerprint:
            raise RuntimeError("Codex gateway returned a different policy fingerprint")
        reward_result = self.reward_callable(final_answer, self.prompt)
        if not isinstance(reward_result, RewardResult):
            reward_value = float(reward_result)
            reward_result = RewardResult(reward_value, {"outcome": reward_value})
        trajectory = build_codex_trajectory(
            prompt=self.prompt,
            policy_version=self.policy_version,
            policy_fingerprint=self.policy_fingerprint,
            sample_index=self.sample_index,
            completion_records=captured.get("completions", []),
            reward=reward_result.value,
            reward_components=reward_result.components,
            end_of_turn_token_id=self.end_of_turn_token_id,
            max_episode_tokens=(
                None
                if self.config.get("max_episode_tokens") is None
                else int(self.config["max_episode_tokens"])
            ),
            metadata={
                "codex_version": str(self.config.get("version", DEFAULT_CODEX_VERSION)),
                "codex_session_id": session_id,
                "artifact_dir": str(artifact_dir),
                "workspace_dir": str(workspace_dir),
                "generation_seconds": time.perf_counter() - started,
                "final_answer": final_answer,
                "codex_diagnostics": diagnostic_events,
                **dict(reward_result.metadata),
            },
        )
        await asyncio.to_thread(
            _http_json,
            "DELETE",
            f"{self.gateway_url}/internal/sessions/{session_id}",
            None,
            float(self.config.get("request_timeout", 600.0)),
        )
        return trajectory

    def _generation_settings(self) -> dict[str, Any]:
        """Pin every hidden Codex model call to the rollout sampling contract."""
        settings = {
            "max_tokens": int(self.config["max_new_tokens"]),
            "temperature": float(self.config["temperature"]),
            "top_p": float(self.config["top_p"]),
            "top_k": (
                int(self.config["top_k"])
                if int(self.config["top_k"]) > 0
                else -1
            ),
        }
        seed = self.config.get("seed")
        if seed is not None:
            identity = f"{self.prompt.prompt_id}:{self.policy_version}:{self.sample_index}"
            offset = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")
            settings["seed"] = (int(seed) + offset) % (2**31)
        return settings

    def _prepare_directories(self, session_id: str) -> tuple[Path, Path, Path]:
        root = Path(
            str(self.config.get("session_root", "/tmp/hyper-rl-codex"))
        ).expanduser().resolve()
        artifact_dir = root / f"{self.prompt.prompt_id}-{self.sample_index}-{session_id}"
        workspace_dir = artifact_dir / "workspace"
        codex_home = artifact_dir / ".codex"
        artifact_dir.mkdir(parents=True, exist_ok=False)
        template_value = self.config.get("workspace_template")
        if template_value:
            template = Path(str(template_value)).expanduser().resolve()
            if not template.is_dir():
                raise ValueError(f"Codex workspace template does not exist: {template}")
            shutil.copytree(template, workspace_dir)
        else:
            workspace_dir.mkdir()
        codex_home.mkdir()
        return artifact_dir, workspace_dir, codex_home

    def _write_codex_config(self, codex_home: Path, session_id: str) -> None:
        lines = [
            'model_provider = "hyper_rl"',
            'web_search = "disabled"',
            "",
            "[model_providers.hyper_rl]",
            'name = "Hyper-RL local policy"',
            f"base_url = {_toml_string(self.gateway_url)}",
            'env_key = "OPENAI_API_KEY"',
            'wire_api = "responses"',
            "requires_openai_auth = true",
            "supports_websockets = false",
            "",
            "[features]",
            "apps = false",
            "plugins = false",
            "remote_plugin = false",
            "multi_agent = false",
            "multi_agent_v2 = false",
            "browser_use = false",
            "computer_use = false",
            "image_generation = false",
            "",
            "[analytics]",
            "enabled = false",
            "",
            "[feedback]",
            "enabled = false",
            "",
            "[otel]",
            'exporter = "none"',
        ]
        for server in self.config.get("mcp_servers", []):
            if not isinstance(server, Mapping):
                raise ValueError("Every Codex MCP server configuration must be a mapping")
            name = server.get("name")
            command = server.get("command")
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(command, str)
                or not command
            ):
                raise ValueError("Codex MCP server requires non-empty name and command")
            lines.extend(
                (
                    "",
                    f"[mcp_servers.{_toml_string(name)}]",
                    f"command = {_toml_string(command)}",
                )
            )
            arguments = server.get("args", [])
            if not isinstance(arguments, list) or not all(
                isinstance(item, str) for item in arguments
            ):
                raise ValueError("Codex MCP server args must be a string list")
            encoded_arguments = ", ".join(_toml_string(item) for item in arguments)
            lines.append(f"args = [{encoded_arguments}]")
            required = "true" if bool(server.get("required", True)) else "false"
            lines.append(f"required = {required}")
            explicit_environment = server.get("env", {})
            if not isinstance(explicit_environment, Mapping) or not all(
                isinstance(key, str)
                and key
                and isinstance(value, str)
                for key, value in explicit_environment.items()
            ):
                raise ValueError("Codex MCP server env must map non-empty names to strings")
            inherited_names = server.get("inherit_env", [])
            if not isinstance(inherited_names, list) or not all(
                isinstance(item, str) and item for item in inherited_names
            ):
                raise ValueError(
                    "Codex MCP server inherit_env must be a list of non-empty names"
                )
            environment = dict(explicit_environment)
            for variable_name in inherited_names:
                if variable_name not in os.environ:
                    raise ValueError(
                        "Codex MCP server requires an unset inherited environment "
                        f"variable: {variable_name}"
                    )
                environment.setdefault(variable_name, os.environ[variable_name])
            if environment:
                lines.extend(("", f"[mcp_servers.{_toml_string(name)}.env]"))
                for variable_name in sorted(environment):
                    lines.append(
                        f"{_toml_string(variable_name)} = "
                        f"{_toml_string(environment[variable_name])}"
                    )
        (codex_home / "config.toml").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        (codex_home / "auth.json").write_text(
            json.dumps({"OPENAI_API_KEY": session_id}) + "\n",
            encoding="utf-8",
        )

    async def _validate_version(self) -> None:
        executable = str(self.config.get("executable", "codex"))
        process = await asyncio.create_subprocess_exec(
            executable,
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        installed = (stdout or stderr).decode("utf-8", errors="replace")
        expected = str(self.config.get("version", DEFAULT_CODEX_VERSION))
        matches = re.search(rf"(?<!\d){re.escape(expected)}(?!\d)", installed)
        if process.returncode != 0 or matches is None:
            raise RuntimeError(
                f"Codex version mismatch: expected {expected}, got {installed.strip()!r}"
            )

    async def _run_codex(
        self,
        session_id: str,
        artifact_dir: Path,
        workspace_dir: Path,
        codex_home: Path,
    ) -> tuple[str, int, list[dict[str, Any]]]:
        executable = str(self.config.get("executable", "codex"))
        prompt_text = self.prompt.messages[-1].content
        template = str(self.config.get("instruction_template", "{prompt}"))
        instruction = template.format(prompt=prompt_text)
        command = [
            executable,
            "exec",
        ]
        sandbox = str(self.config.get("sandbox", "danger-full-access"))
        if sandbox == "danger-full-access":
            command.append("--dangerously-bypass-approvals-and-sandbox")
        elif sandbox == "workspace-write":
            command.extend(("--sandbox", sandbox, "--approve-for-me"))
        else:
            raise ValueError(f"Unsupported automated Codex sandbox: {sandbox}")
        command.extend(
            (
                "--strict-config",
                "--skip-git-repo-check",
                "--model",
                str(self.config.get("model", "policy")),
                "--json",
            )
        )
        reasoning_effort = self.config.get("reasoning_effort")
        if reasoning_effort is not None:
            command.extend(("-c", f'model_reasoning_effort="{reasoning_effort}"'))
        command.extend(("--", instruction))
        environment = os.environ.copy()
        for name in tuple(environment):
            if name.startswith(("CODEX_", "OPENAI_")):
                environment.pop(name)
        gateway_host = urlparse(self.gateway_url).hostname
        no_proxy = [environment.get("NO_PROXY", environment.get("no_proxy", ""))]
        no_proxy.extend(("127.0.0.1", "localhost"))
        if gateway_host:
            no_proxy.append(gateway_host)
        environment.update(
            {
                "CODEX_HOME": str(codex_home),
                "CODEX_API_KEY": session_id,
                "OPENAI_API_KEY": session_id,
                "NO_PROXY": ",".join(filter(None, no_proxy)),
                "no_proxy": ",".join(filter(None, no_proxy)),
            }
        )
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=workspace_dir,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        async def drain_stdout() -> None:
            if process.stdout is None:
                return
            with (artifact_dir / "codex-events.jsonl").open("w", encoding="utf-8") as stream:
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                    stdout_lines.append(text)
                    stream.write(text + "\n")
                    stream.flush()

        async def drain_stderr() -> None:
            if process.stderr is None:
                return
            with (artifact_dir / "codex-stderr.log").open("w", encoding="utf-8") as stream:
                while True:
                    line = await process.stderr.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace")
                    stderr_lines.append(text.rstrip("\r\n"))
                    stream.write(text)
                    stream.flush()

        timeout = float(self.config.get("timeout_seconds", 1800.0))
        stdout_task = asyncio.create_task(drain_stdout())
        stderr_task = asyncio.create_task(drain_stderr())
        try:
            await asyncio.wait_for(process.wait(), timeout)
        except asyncio.TimeoutError as error:
            process.kill()
            await process.wait()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            raise RuntimeError(f"Codex episode timed out after {timeout} seconds") from error
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
                await process.wait()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            raise
        await asyncio.gather(stdout_task, stderr_task)

        stderr_tail = "\n".join(stderr_lines[-20:]).strip()
        artifact_hint = f"artifacts: {artifact_dir}"
        return_code = int(process.returncode or 0)
        if return_code != 0:
            detail = f"; stderr tail:\n{stderr_tail}" if stderr_tail else ""
            raise RuntimeError(
                f"Codex exited with status {return_code}{detail}; {artifact_hint}"
            )

        final_answer, diagnostic_events, failed_events = _classify_codex_events(
            stdout_lines
        )
        if failed_events:
            detail = f"; stderr tail:\n{stderr_tail}" if stderr_tail else ""
            raise RuntimeError(
                "Codex reported a failed event: "
                f"{failed_events[-1]}{detail}; {artifact_hint}"
            )
        if not final_answer:
            event_tail = "\n".join(stdout_lines[-10:]).strip()
            details = []
            if event_tail:
                details.append(f"event tail:\n{event_tail}")
            if stderr_tail:
                details.append(f"stderr tail:\n{stderr_tail}")
            suffix = f"; {'; '.join(details)}" if details else ""
            raise RuntimeError(
                "Codex JSONL did not contain a final agent message"
                f"{suffix}; {artifact_hint}"
            )
        return final_answer, return_code, diagnostic_events


class CodexProgramFactory:
    """Create policy-identity-bound Codex programs for ProgramAgentRunner."""

    def __init__(
        self,
        runtime: CodexRuntime,
        end_of_turn_token_id: Optional[int],
        generation_config: Mapping[str, Any],
    ) -> None:
        """Bind the shared runtime and chat-template boundary token."""
        self.runtime = runtime
        self.end_of_turn_token_id = end_of_turn_token_id
        self.config = {**runtime.config, **dict(generation_config)}

    def __call__(
        self,
        prompt: PromptRecord,
        policy_version: int,
        sample_index: int,
    ) -> CodexAgentProgram:
        """Construct one episode after verifying the served policy identity."""
        served_version, fingerprint = self.runtime.episode_identity
        if served_version != policy_version:
            raise RuntimeError(
                "Codex requested policy version does not match the served policy: "
                f"requested={policy_version}, served={served_version}"
            )
        return CodexAgentProgram(
            prompt=prompt,
            policy_version=policy_version,
            sample_index=sample_index,
            policy_fingerprint=fingerprint,
            gateway_url=self.runtime.gateway_url,
            config=self.config,
            end_of_turn_token_id=self.end_of_turn_token_id,
        )
