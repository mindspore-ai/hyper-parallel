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
import json
import logging
import os
import re
import secrets
import shutil
import signal
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from urllib.parse import urlparse

from rl.agentic.codex.gateway import CodexGateway
from rl.agentic.core.program_runner import (
    AgentProgram,
    GatewayTransportError,
    HarnessProgramFactory,
    HarnessRuntime,
    build_harness_call_trajectories,
    build_harness_trajectory,
    harness_generation_settings,
    load_reward_callable,
    request_gateway_json,
)
from rl.agentic.core.types import RewardResult
from rl.dataset.contracts import PromptRecord, Trajectory


_LOGGER = logging.getLogger(__name__)
DEFAULT_CODEX_VERSION = "0.152.1"
RewardCallable = Callable[[str, PromptRecord], float | RewardResult]
WorkspaceCallable = Callable[[PromptRecord, Path], None]
ExecutionCallable = Callable[[list[str], Path, Path, dict[str, str]], tuple[list[str], Path, dict[str, str]]]
_MODEL_METADATA_FALLBACK = re.compile(
    r"^Model metadata for `[^`]+` not found\. Defaulting to fallback metadata;"
)
_COMPACTION_WARNING = (
    "Heads up: Long threads and multiple compactions can cause the model to be less accurate. "
    "Start a new thread when possible to keep threads small and targeted."
)
_BUDGET_RESPONSE = re.compile(
    r"unexpected status 409 Conflict: Repository session exhausted its ([1-9][0-9]*) model calls, "
    r"url: (https?://\S+/responses)"
)


class _CompletionBudgetExit(RuntimeError):
    """Identify the exact Gateway budget response that caused the CLI to stop."""

    def __init__(self, limit: int, url: str) -> None:
        """Keep the reported limit and endpoint for validation against the sealed capture."""
        super().__init__("Codex stopped after a structured completion budget response")
        self.limit = limit
        self.url = url


class _MissingFinalMessage(RuntimeError):
    """A successful CLI process ended without a final answer, pending capture validation."""

    def __init__(self, message: str, turn_completed: bool) -> None:
        """Keep the CLI terminal evidence separate from the missing-answer diagnostic."""
        super().__init__(message)
        self.turn_completed = turn_completed


class _CodexExecutionFailure(RuntimeError):
    """A timeout or abnormal process exit cannot be replaced by a sampled model outcome."""


def build_codex_trajectory(
    *,
    prompt: PromptRecord,
    policy_version: int,
    sample_index: int,
    completion_records: Sequence[Mapping[str, Any]],
    reward: float,
    reward_components: Mapping[str, float],
    end_of_turn_token_id: int | None = None,
    max_episode_tokens: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Trajectory:
    """Merge every captured Codex completion into one token-exact trajectory."""
    return build_harness_trajectory(
        label="Codex",
        runner_name="codex",
        prompt=prompt,
        policy_version=policy_version,
        sample_index=sample_index,
        completion_records=completion_records,
        reward=reward,
        reward_components=reward_components,
        end_of_turn_token_id=end_of_turn_token_id,
        max_episode_tokens=max_episode_tokens,
        metadata=metadata,
    )


def build_codex_call_trajectories(
    *,
    prompt: PromptRecord,
    policy_version: int,
    sample_index: int,
    completion_records: Sequence[Mapping[str, Any]],
    reward: float,
    reward_components: Mapping[str, float],
    max_episode_tokens: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Trajectory, ...]:
    """Keep every Codex action trainable under its real rollout prompt."""
    return build_harness_call_trajectories(
        label="Codex",
        runner_name="codex",
        prompt=prompt,
        policy_version=policy_version,
        sample_index=sample_index,
        completion_records=completion_records,
        reward=reward,
        reward_components=reward_components,
        max_episode_tokens=max_episode_tokens,
        metadata=metadata,
        status_resolver=_codex_call_status,
    )


def _codex_call_status(traces: Sequence[Mapping[str, Any]],
                       metadata: Mapping[str, Any]) -> tuple[bool, str] | None:
    """Preserve the verified model terminal or budget status for each call."""
    del traces
    if metadata.get("failure_reason") == "context_budget_exhausted":
        return True, "max_context_tokens"
    if metadata.get("failure_reason") == "output_budget_exhausted":
        return True, "max_output_tokens"
    if metadata.get("failure_reason") == "reasoning_only_eos":
        return False, "reasoning_only_eos"
    if metadata.get("repository_budget_truncated") is True or metadata.get("completion_budget_truncated") is True:
        return True, "max_completions"
    return None


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _load_reward_callable(value: Any) -> RewardCallable:
    return load_reward_callable(value, "agentic.codex.reward_callable", "Codex")


def _load_workspace_callable(value: Any) -> WorkspaceCallable | None:
    if value is None:
        return None
    return load_reward_callable(value, "agentic.codex.workspace_callable", "Codex workspace")


def _load_execution_callable(value: Any) -> ExecutionCallable | None:
    if value is None:
        return None
    return load_reward_callable(value, "agentic.codex.execution_callable", "Codex execution")


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
        if _MODEL_METADATA_FALLBACK.search(message) or message == _COMPACTION_WARNING:
            diagnostics.append(event)
        else:
            failed_events.append(event)
    return final_answer, diagnostics, failed_events


def _has_completed_codex_turn(output: str) -> bool:
    """Require the CLI terminal event even when a repository summary is absent."""
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "turn.completed":
            return True
    return False


def _http_json(
    method: str,
    url: str,
    payload: Optional[Mapping[str, Any]],
    timeout: float,
    admin_token: str | None = None,
    *, max_response_bytes: int | None = None,
) -> dict[str, Any]:
    """Send a bounded gateway request with optional controller authentication."""
    if admin_token is not None and max_response_bytes is None:
        max_response_bytes = 32 * 1024 * 1024
    return request_gateway_json("Codex", method, url, payload, timeout,
                                admin_token=admin_token, max_response_bytes=max_response_bytes)


class CodexRuntime(HarnessRuntime):
    """Own one node-level gateway while Hyper-RL continues to own vLLM."""

    def __init__(self, engine: Any, config: Mapping[str, Any]) -> None:
        """Bind the runtime to the existing shared rollout engine."""
        gateway_options = {
            "max_inflight_requests": int(config.get("max_inflight_requests", 1)),
            "policy_gate": bool(config.get("overlap_preparation", False)),
        }
        if getattr(engine, "asynchronous", False) is True:
            gateway_options["async_training"] = True
        capture_spec = getattr(engine, "router_capture_spec", None)
        if capture_spec is not None:
            gateway_options["router_capture_spec"] = asdict(capture_spec)
        for name in ("max_request_bytes", "max_response_bytes"):
            if name in config:
                gateway_options[name] = int(config[name])
        if config.get("admin_auth", False):
            gateway_options["admin_token"] = secrets.token_urlsafe(32)
        super().__init__(
            engine, config, CodexGateway, "Codex", 8200,
            gateway_options=gateway_options,
        )

    def ensure_preparation_started(self) -> None:
        """Bring up the gated CPU endpoint while model startup or training is still pending."""
        self._ensure_gateway(self.engine.configured_inference_base_url, self.engine.inference_model_name)

    def set_admission_policy(self, version: int | None) -> None:
        """Change admission on the gateway owner after the main-thread version barrier."""
        if self._gateway is not None:
            self._gateway.set_policy(version)


class CodexAgentProgram:
    """Run one Codex tool loop and return policy-aligned call trajectories."""

    def __init__(
        self,
        prompt: PromptRecord,
        policy_version: int,
        sample_index: int,
        gateway_url: str,
        config: Mapping[str, Any],
        end_of_turn_token_id: Optional[int],
        admin_url: str | None = None,
        admin_token: str | None = None,
    ) -> None:
        """Capture the episode identity and immutable harness settings."""
        if not isinstance(policy_version, int) or isinstance(policy_version, bool) or policy_version < 0:
            raise ValueError("Training requires a non-negative policy version")
        self.prompt = prompt
        self.policy_version = policy_version
        self.sample_index = sample_index
        self.gateway_url = gateway_url.rstrip("/")
        self.admin_url = (admin_url or gateway_url).rstrip("/")
        self.admin_token = admin_token
        self.config = dict(config)
        self.end_of_turn_token_id = end_of_turn_token_id
        self.reward_callable = _load_reward_callable(self.config.get("reward_callable"))
        self.workspace_callable = _load_workspace_callable(self.config.get("workspace_callable"))
        self.execution_callable = _load_execution_callable(self.config.get("execution_callable"))

    async def run(self) -> tuple[Trajectory, ...]:
        """Run Codex, freeze and score its output, then construct call trajectories."""
        started_at = time.time()
        session_id = uuid.uuid4().hex
        artifact_dir, workspace_dir, codex_home = self._prepare_directories(session_id)
        _performance_event(artifact_dir, "program.prepared", started_at=started_at,
                           prompt_id=self.prompt.prompt_id, sample_index=self.sample_index,
                           policy_version=self.policy_version)
        timeout = float(self.config.get("request_timeout", 600.0))
        try:
            await self._admin_request("POST", "/internal/sessions", {
                "session_id": session_id,
                "policy_version": self.policy_version,
                "artifact_dir": str(artifact_dir),
                "max_completions": int(self.config["max_turns"]),
                "generation": self._generation_settings(),
                **self._session_settings(),
                "instructions_override": self.config.get("system_instructions"),
                "enable_thinking": self.config.get("enable_thinking"),
                "compact_harness_context": self.config.get("compact_harness_context", False),
                "preserve_environment_context": self.config.get("preserve_environment_context", False),
                "allowed_tools": self.config.get("allowed_tools"),
                "exec_command_yield_time_ms": self.config.get("exec_command_yield_time_ms"),
            }, timeout)
            _performance_event(artifact_dir, "session.registered")
            return await self._run_registered(
                session_id, artifact_dir, workspace_dir, codex_home, timeout
            )
        except BaseException as error:
            _performance_event(artifact_dir, "program.failed", error_type=type(error).__name__, error=str(error))
            _LOGGER.exception("Codex program failed: session=%s prompt=%s sample=%d",
                              session_id, self.prompt.prompt_id, self.sample_index)
            raise
        finally:
            original_error = sys.exc_info()[1]
            try:
                await self._admin_request("DELETE", f"/internal/sessions/{session_id}", None, timeout)
            except Exception:  # pylint: disable=broad-exception-caught
                if original_error is None:
                    raise
                _LOGGER.exception("Codex session cleanup failed while preserving original failure")
            finally:
                _performance_event(artifact_dir, "program.finished")

    def _session_settings(self) -> dict[str, Any]:
        """Allow a trusted program to opt into additional gateway protocol checks."""
        return {}

    async def _admin_request(self, method: str, path: str, payload: Mapping | None,
                             timeout: float) -> dict[str, Any]:
        """Settle an in-flight controller request before cancellation cleanup."""
        retryable = (method == "POST" and path == "/internal/sessions") or method == "DELETE"
        if method == "POST" and path == "/internal/sessions":
            # Registration is a small local control request, not a model-generation deadline.
            timeout = min(timeout, 30.0)
        for attempt in range(3 if retryable else 1):
            try:
                return await self._admin_request_once(method, path, payload, timeout)
            except GatewayTransportError:
                if not retryable or attempt == 2:
                    raise
                await asyncio.sleep(0.1 * 2**attempt)
        raise RuntimeError("Management request retry loop exhausted")

    async def _admin_request_once(self, method: str, path: str, payload: Mapping | None,
                                  timeout: float) -> dict[str, Any]:
        """Await the result of one HTTP operation before retry or cancellation cleanup."""
        options = {}
        arguments = (method, self.admin_url + path, payload, timeout)
        if self.admin_token is not None:
            arguments = (*arguments, self.admin_token)
            options = {
                "max_response_bytes": int(self.config.get("max_response_bytes", 32 * 1024 * 1024)),
            }
        request = asyncio.create_task(asyncio.to_thread(
            _http_json, *arguments, **options,
        ))
        try:
            return await asyncio.shield(request)
        except asyncio.CancelledError:
            await asyncio.gather(request, return_exceptions=True)
            raise

    def _captured_failure(self, captured: Mapping[str, Any],
                          execution_error: Exception | None) -> RewardResult | None:
        """Return zero only for a verified trainable model failure."""
        if captured.get("policy_version") != self.policy_version:
            raise RuntimeError("Codex gateway returned a different policy version")
        failure = captured.get("failure")
        if failure is None:
            return None
        if (not isinstance(failure, Mapping) or failure.get("failure_origin") != "model"
                or failure.get("trainable") is not True):
            raise RuntimeError(f"Codex gateway reported an untrainable failure: {failure}") from execution_error
        return RewardResult(0.0, {"outcome": 0.0}, dict(failure))


    def _captured_budget(self, captured: Mapping[str, Any]) -> Optional[dict[str, Any]]:
        """Accept only a completed, failure-free budget matching this episode's identity."""
        termination = captured.get("termination")
        if termination is None:
            return None
        records = captured.get("completions", [])
        if (not isinstance(termination, Mapping)
                or set(termination) != {"reason", "limit", "completed", "policy_version"}
                or termination.get("reason") != "max_completions"
                or any(not isinstance(termination.get(key), int) or isinstance(termination.get(key), bool)
                       for key in ("limit", "completed", "policy_version"))
                or termination["limit"] != int(self.config["max_turns"])
                or termination["completed"] != termination["limit"]
                or termination["policy_version"] != self.policy_version
                or captured.get("failure") is not None
                or not isinstance(records, list) or len(records) != termination["limit"]
                or sorted(record.get("ordinal", -1) for record in records) != list(range(len(records)))):
            raise RuntimeError("Codex budget termination differs from the completed captured session")
        return dict(termination)


    def _missing_final_record(self, captured: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Select the last completion only from a contiguous, trainable captured session."""
        records = captured.get("completions", [])
        if (captured.get("failure") is not None or not isinstance(records, list) or not records
                or len(records) > int(self.config["max_turns"])
                or any(not isinstance(record, Mapping) or record.get("ordinal") != index
                       or isinstance(record.get("ordinal"), bool)
                       or record.get("metadata", {}).get("trainable") is not True
                       for index, record in enumerate(records))):
            return None
        return records[-1]

    @staticmethod
    def _reasoning_only_choice(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], list, int] | None:
        """Require a bounded token response containing reasoning but no answer or tool calls."""
        limit = record.get("request", {}).get("max_tokens")
        choices = record.get("response", {}).get("choices", [])
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
            return None
        choice = choices[0]
        message = choice.get("message", {})
        tokens = choice.get("token_ids")
        if (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
                or not isinstance(tokens, list) or not tokens or len(tokens) > limit
                or not isinstance(message, Mapping) or message.get("content") or message.get("tool_calls")
                or not isinstance(message.get("reasoning_content", message.get("reasoning")), str)
                or not message.get("reasoning_content", message.get("reasoning"))):
            return None
        return choice, tokens, limit

    def _captured_missing_final(self, captured: Mapping[str, Any],
                                execution_error: Exception | None) -> RewardResult | None:
        """Recognize a clean reasoning-only turn ending at the token cap or sampled EOS."""
        if not isinstance(execution_error, _MissingFinalMessage) or not execution_error.turn_completed:
            return None
        record = self._missing_final_record(captured)
        if record is None:
            return None
        completion = self._reasoning_only_choice(record)
        if completion is None:
            return None
        choice, tokens, limit = completion
        if (choice.get("finish_reason") == "stop" and self.end_of_turn_token_id is not None
                and tokens[-1] == self.end_of_turn_token_id):
            return RewardResult(0.0, {"outcome": 0.0}, {
                "failure_origin": "model", "failure_reason": "reasoning_only_eos", "trainable": True,
            })
        if choice.get("finish_reason") != "length" or len(tokens) != limit:
            return None
        return RewardResult(0.0, {"outcome": 0.0}, {
            "failure_origin": "model", "failure_reason": "output_budget_exhausted", "trainable": True,
            "output_budget": {"limit": limit, "completed": len(tokens), "ordinal": record["ordinal"]},
        })

    def resolve_execution_outcome(self, captured: Mapping[str, Any], execution_error: Exception | None,
                                  final_answer: str) -> RewardResult | None:
        """Resolve shared model terminals; reject unexplained exits before task-specific scoring.

        Args:
            captured: Gateway capture for this policy version, including all completed calls.
            execution_error: Original CLI exception, if any; unknown failures are never inferred as budgets.
            final_answer: Actual CLI final text, without synthesizing an answer for truncated output.

        Returns:
            Zero reward and terminal metadata for verified model failure, or None for normal task scoring.
        """
        if isinstance(execution_error, _CodexExecutionFailure):
            raise execution_error
        reward_result = self._captured_failure(captured, execution_error)
        budget = self._captured_budget(captured)
        if reward_result is None and not final_answer:
            reward_result = self._captured_missing_final(captured, execution_error)
        if reward_result is None and budget is not None and not final_answer:
            if execution_error is not None:
                if not isinstance(execution_error, _CompletionBudgetExit):
                    raise execution_error
                if (execution_error.limit, execution_error.url) != (
                    budget["limit"], self.gateway_url + "/responses",
                ):
                    raise execution_error
            reward_result = RewardResult(0.0, {"outcome": 0.0}, {
                "completion_budget_truncated": True, "budget_termination": budget,
            })
        if reward_result is None and execution_error is not None:
            raise execution_error
        return reward_result


    async def _run_registered(
        self,
        session_id: str,
        artifact_dir: Path,
        workspace_dir: Path,
        codex_home: Path,
        timeout: float,
    ) -> tuple[Trajectory, ...]:
        """Execute and materialize one already registered Codex session."""
        self._write_codex_config(codex_home, session_id)
        _performance_event(artifact_dir, "version_check.started")
        await self._validate_version()
        _performance_event(artifact_dir, "version_check.finished")
        started = time.perf_counter()
        execution_error = None
        final_answer, diagnostics = "", []
        try:
            final_answer, return_code, diagnostics = await self._run_codex(
                session_id, artifact_dir, workspace_dir, codex_home
            )
            if return_code != 0:
                raise RuntimeError(f"Codex exited with status {return_code}; see {artifact_dir}")
        except RuntimeError as error:
            execution_error = error
        captured = await self._admin_request("GET", f"/internal/sessions/{session_id}", None, timeout)
        _performance_event(artifact_dir, "capture.finished")
        reward_result = self.resolve_execution_outcome(captured, execution_error, final_answer)
        if reward_result is None:
            reward_result = self.reward_callable(final_answer, self.prompt)
            if not isinstance(reward_result, RewardResult):
                reward_value = float(reward_result)
                reward_result = RewardResult(reward_value, {"outcome": reward_value})
        _performance_event(artifact_dir, "trajectory_build.started")
        rows = self._build_rows(session_id, artifact_dir, workspace_dir, started, final_answer,
                                diagnostics, captured, reward_result)
        _performance_event(artifact_dir, "trajectory_build.finished", rows=len(rows), reward=reward_result.value)
        return rows

    def _build_rows(self, session_id: str, artifact_dir: Path, workspace_dir: Path, started: float,
                    final_answer: str, diagnostics: list, captured: Mapping,
                    reward_result: RewardResult) -> tuple[Trajectory, ...]:
        return build_codex_call_trajectories(
            prompt=self.prompt,
            policy_version=self.policy_version,
            sample_index=self.sample_index,
            completion_records=captured.get("completions", []),
            reward=reward_result.value,
            reward_components=reward_result.components,
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
                "codex_diagnostics": diagnostics,
                **dict(reward_result.metadata),
            },
        )

    def _generation_settings(self) -> dict[str, Any]:
        """Pin every hidden Codex model call to the rollout sampling contract."""
        return harness_generation_settings(
            self.config, self.prompt.prompt_id, self.policy_version, self.sample_index
        )

    def _prepare_directories(self, session_id: str) -> tuple[Path, Path, Path]:
        """Create isolated workspace and runtime directories for this session."""
        root = Path(
            str(self.config.get("session_root", "/tmp/hyper-rl-codex"))
        ).expanduser().resolve()
        # Logical IDs can contain separators; only generated session IDs enter paths.
        artifact_dir = root / f"{session_id}-{self.sample_index}"
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
        if self.workspace_callable is not None:
            self.workspace_callable(self.prompt, workspace_dir)
        codex_home.mkdir()
        return artifact_dir, workspace_dir, codex_home

    def _write_codex_config(self, codex_home: Path, session_id: str, *, gateway_url: str | None = None,
                            runtime_home: Path | None = None) -> None:
        """Write the isolated provider and tool configuration for this session."""
        lines = ['model_provider = "hyper_rl"', 'web_search = "disabled"']
        instructions = self.config.get("model_instructions")
        if instructions is not None:
            if not isinstance(instructions, str) or not instructions.strip():
                raise ValueError("agentic.codex.model_instructions must be a non-empty string when configured")
            filename = "model-instructions.md"
            (codex_home / filename).write_text(instructions, encoding="utf-8")
            instructions_path = (runtime_home or codex_home.resolve()) / filename
            lines.append(f"model_instructions_file = {_toml_string(str(instructions_path))}")
        lines.extend([
            "",
            "[model_providers.hyper_rl]",
            'name = "Hyper-RL local policy"',
            f"base_url = {_toml_string(gateway_url or self.gateway_url)}",
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
        ])
        lines.extend([
            "",
            "[analytics]",
            "enabled = false",
            "",
            "[feedback]",
            "enabled = false",
            "",
            "[otel]",
            'exporter = "none"',
        ])
        for server in self.config.get("mcp_servers", []):
            if not isinstance(server, Mapping):
                raise ValueError("Every Codex MCP server configuration must be a mapping")
            name = server.get("name")
            command = server.get("command")
            valid_name = isinstance(name, str) and bool(name)
            valid_command = isinstance(command, str) and bool(command)
            if not valid_name or not valid_command:
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
            environment = _mcp_environment(server)
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
        """Reject a runtime executable whose version differs from the pinned version."""
        executable = str(self.config.get("executable", "codex"))
        process = await asyncio.create_subprocess_exec(
            executable,
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 30.0)
        finally:
            await _stop_process_group(process)
        installed = (stdout or stderr).decode("utf-8", errors="replace")
        expected = str(self.config.get("version", DEFAULT_CODEX_VERSION))
        matches = re.search(rf"(?<!\d){re.escape(expected)}(?!\d)", installed)
        if process.returncode != 0 or matches is None:
            raise RuntimeError(
                f"Codex version mismatch: expected {expected}, got {installed.strip()!r}"
            )

    def _codex_command(self):
        """Build the pinned noninteractive command from the episode configuration."""
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
            command.extend(("--sandbox", sandbox))
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
        return command


    async def _run_codex(
        self,
        session_id: str,
        artifact_dir: Path,
        workspace_dir: Path,
        codex_home: Path,
    ) -> tuple[str, int, list[dict[str, Any]]]:
        """Run the configured program with isolated state and bounded execution time."""
        command = self._codex_command()
        environment = _codex_environment(self.gateway_url, codex_home, session_id)
        environment["HYPER_CODEX_GATEWAY_URL"] = self.gateway_url
        environment["HYPER_CODEX_MAX_CALLS"] = str(self.config.get("max_turns", 1))
        cwd = workspace_dir
        if self.execution_callable is not None:
            command, cwd, environment = self.execution_callable(
                command, workspace_dir, codex_home, environment
            )
        _performance_event(artifact_dir, "cli_spawn.started")
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        _performance_event(artifact_dir, "cli_spawn.finished", pid=process.pid)
        timeout = float(self.config.get("timeout_seconds", 1800.0))
        stdout_lines, stderr_lines = await _collect_codex_output(process, artifact_dir, timeout)
        _performance_event(artifact_dir, "cli_collect.finished", return_code=process.returncode)
        return_code = int(process.returncode or 0)
        final_answer, diagnostic_events = _interpret_codex_output(
            stdout_lines, stderr_lines, return_code, artifact_dir,
        )
        return final_answer, return_code, diagnostic_events


async def _wait_for_process_exit(process: asyncio.subprocess.Process) -> None:
    """Observe the harness exit even when a tool descendant still owns its output pipes."""
    while process.returncode is None:
        await asyncio.sleep(0.05)


async def _stop_process_group(process: asyncio.subprocess.Process) -> None:
    """Reap the harness and its tool descendants on success, failure or cancellation."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), 5.0)
    except asyncio.TimeoutError:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def _drain_codex_stream(
    reader: Any, path: Path, lines: list[str], *, normalize_newlines: bool,
) -> None:
    """Persist one child stream while retaining its decoded lines for diagnostics."""
    if reader is None:
        return
    with path.open("w", encoding="utf-8") as stream:
        while True:
            line = await reader.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace")
            text = decoded.rstrip("\r\n")
            if normalize_newlines:
                _performance_event(path.parent, "cli_stdout.received", line_index=len(lines))
            lines.append(text)
            stream.write(text + "\n" if normalize_newlines else decoded)
            stream.flush()


def _performance_event(artifact_dir: Path, event: str, **fields: Any) -> None:
    """Record receipt times separately so the original CLI and model payloads remain unchanged."""
    record = {"event": event, "timestamp": time.time(), "monotonic_ns": time.perf_counter_ns(), **fields}
    with (artifact_dir / "performance.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


async def _collect_codex_output(
    process: Any, artifact_dir: Path, timeout: float,
) -> tuple[list[str], list[str]]:
    """Drain both child pipes and preserve timeout and cancellation cleanup."""
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    stdout_task = asyncio.create_task(
        _drain_codex_stream(process.stdout, artifact_dir / "codex-events.jsonl", stdout_lines,
                            normalize_newlines=True)
    )
    stderr_task = asyncio.create_task(
        _drain_codex_stream(process.stderr, artifact_dir / "codex-stderr.log", stderr_lines,
                            normalize_newlines=False)
    )
    try:
        await asyncio.wait_for(_wait_for_process_exit(process), timeout)
    except asyncio.TimeoutError as error:
        raise _CodexExecutionFailure(f"Codex episode timed out after {timeout} seconds") from error
    finally:
        # Descendants may keep stdout open after the harness exits.
        await _stop_process_group(process)
        drain_results = await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
    for result in drain_results:
        if isinstance(result, BaseException):
            raise RuntimeError("Codex output capture failed") from result
    return stdout_lines, stderr_lines


def _cli_budget_response(event: Mapping[str, Any]) -> Optional[tuple[int, str]]:
    """Recognize only the pinned CLI's exact budget response, including its known retries."""
    message = event.get("message")
    if event.get("type") == "turn.failed":
        error = event.get("error")
        message = error.get("message") if isinstance(error, Mapping) else None
    elif event.get("type") != "error":
        return None
    if not isinstance(message, str):
        return None
    if message.startswith("Reconnecting..."):
        retry = re.fullmatch(r"Reconnecting\.\.\. [1-5]/5 \((.+)\)", message)
        if retry is None:
            return None
        message = retry[1]
    match = _BUDGET_RESPONSE.fullmatch(message)
    return None if match is None else (int(match[1]), match[2])


def _interpret_codex_output(
    stdout_lines: list[str], stderr_lines: list[str], return_code: int, artifact_dir: Path,
) -> tuple[str, list[dict[str, Any]]]:
    """Validate child status and return its final answer and diagnostic events."""
    stderr_tail = "\n".join(stderr_lines[-20:]).strip()
    artifact_hint = f"artifacts: {artifact_dir}"
    final_answer, diagnostic_events, failed_events = _classify_codex_events(stdout_lines)
    if return_code in (0, 1) and failed_events and failed_events[-1].get("type") == "turn.failed":
        budgets = [_cli_budget_response(event) for event in failed_events]
        if budgets[0] is not None and all(value == budgets[0] for value in budgets):
            raise _CompletionBudgetExit(*budgets[0])
    if return_code != 0:
        detail = f"; stderr tail:\n{stderr_tail}" if stderr_tail else ""
        error_type = RuntimeError if return_code == 1 else _CodexExecutionFailure
        raise error_type(f"Codex exited with status {return_code}{detail}; {artifact_hint}")
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
        raise _MissingFinalMessage(
            "Codex JSONL did not contain a final agent message"
            f"{suffix}; {artifact_hint}",
            turn_completed=_has_completed_codex_turn("\n".join(stdout_lines)),
        )
    return final_answer, diagnostic_events


class CodexProgramFactory(HarnessProgramFactory):
    """Create policy-identity-bound Codex programs for ProgramAgentRunner."""

    program_type = CodexAgentProgram
    label = "Codex"
    include_admin_url = True

    def create_prepared(self, prompt: PromptRecord, policy_version: int, sample_index: int) -> AgentProgram:
        """Create a future-version CPU session; its gateway must enforce policy admission."""
        if not self.runtime.config.get("overlap_preparation", False):
            raise RuntimeError("Codex preparation requires version-gated overlap")
        return self._build_program(prompt, policy_version, sample_index)

    def __init__(
        self, runtime: CodexRuntime, end_of_turn_token_id: int | None, generation_config: Mapping[str, Any]
    ) -> None:
        """Keep the mainline custom program override and authenticated repository runtime."""
        super().__init__(runtime, end_of_turn_token_id, generation_config)
        program = self.config.get("program_callable")
        if program is not None:
            self.program_type = load_reward_callable(
                program, "agentic.codex.program_callable", "Codex program"
            )


def _mcp_environment(server):
    """Resolve explicitly configured and inherited tool-server variables."""
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
    return environment


def _codex_environment(gateway_url, codex_home, session_id):
    """Build isolated child credentials and loopback proxy exclusions."""
    environment = os.environ.copy()
    # portable-pty 0.9.0 otherwise uses non-thread-safe getpwuid during parallel PTY creation.
    environment.setdefault("SHELL", "/bin/bash")
    for name in tuple(environment):
        if name.startswith(("CODEX_", "OPENAI_")):
            environment.pop(name)
    gateway_host = urlparse(gateway_url).hostname
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
    return environment
