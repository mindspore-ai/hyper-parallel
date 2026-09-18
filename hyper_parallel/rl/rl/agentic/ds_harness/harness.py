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
"""DeepSeek Harness SDK process boundary and Hyper-RL AgentProgram."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import json
import os
import shutil
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rl.agentic.core.program_runner import (
    HarnessProgramFactory,
    HarnessRuntime,
    build_harness_trajectory,
    harness_generation_settings,
    load_reward_callable,
    request_gateway_json,
)
from rl.agentic.core.types import RewardResult
from rl.agentic.ds_harness.gateway import DeepSeekGateway
from rl.dataset.contracts import PromptRecord, Trajectory
DEFAULT_DEEPSEEK_HARNESS_VERSION = "0.1.1rc1"
RewardCallable = Callable[[str, PromptRecord], float | RewardResult]
_MAX_GENERATION_PREFIX_REWRITE = 16


def _trajectory_status(
    traces: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> tuple[bool, str]:
    """Infer completion status and stop reason from captured traces."""
    finish_reasons = {str(trace.get("finish_reason") or "") for trace in traces}
    if finish_reasons & {"length", "max_tokens"}:
        return True, "max_tokens"
    harness_reason = str(metadata.get("finish_reason") or "")
    if harness_reason in {"error", "aborted"}:
        return True, f"harness_{harness_reason}"
    return False, "completed"


def build_deepseek_trajectory(
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
    """Merge every captured DeepSeek completion into one token-exact trajectory."""
    return build_harness_trajectory(
        label="DeepSeek",
        runner_name="deepseek",
        prompt=prompt,
        policy_version=policy_version,
        sample_index=sample_index,
        completion_records=completion_records,
        reward=reward,
        reward_components=reward_components,
        end_of_turn_token_id=end_of_turn_token_id,
        max_episode_tokens=max_episode_tokens,
        metadata=metadata,
        max_prefix_rewrite=_MAX_GENERATION_PREFIX_REWRITE,
        tool_history_field="messages",
        status_resolver=_trajectory_status,
    )


def _load_reward_callable(value: Any) -> RewardCallable:
    return load_reward_callable(value, "agentic.deepseek.reward_callable", "DeepSeek")


def _load_sdk(expected_version: str) -> tuple[Any, Any]:
    """Load the optional SDK only when the DeepSeek runner is selected."""
    try:
        installed = importlib.metadata.version("deepseek-harness-sdk")
        sdk = importlib.import_module("deepseek_harness")
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise RuntimeError(
            f"The DeepSeek runner requires deepseek-harness-sdk=={expected_version}"
        ) from error
    if installed != expected_version:
        raise RuntimeError(
            "DeepSeek Harness SDK version mismatch: "
            f"expected {expected_version}, got {installed}"
        )
    return sdk.DeepSeekHarness, sdk.DeepSeekHarnessConfig


def _http_json(
    method: str,
    url: str,
    payload: Mapping[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    return request_gateway_json("DeepSeek", method, url, payload, timeout)


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class DeepSeekRuntime(HarnessRuntime):
    """Own one DeepSeek-only gateway while Hyper-RL continues to own vLLM."""

    def __init__(self, engine: Any, config: Mapping[str, Any]) -> None:
        """Bind the independent runtime to the existing shared rollout engine."""
        super().__init__(engine, config, DeepSeekGateway, "DeepSeek", 8300, "/v1")


class DeepSeekAgentProgram:
    """Run one DeepSeek Harness episode and return one immutable trajectory."""

    def __init__(
        self,
        prompt: PromptRecord,
        policy_version: int,
        sample_index: int,
        gateway_url: str,
        admin_url: str,
        config: Mapping[str, Any],
        end_of_turn_token_id: int | None,
    ) -> None:
        """Capture the episode identity and immutable Harness settings."""
        self.prompt = prompt
        self.policy_version = policy_version
        self.sample_index = sample_index
        self.gateway_url = gateway_url.rstrip("/")
        self.admin_url = admin_url.rstrip("/")
        self.config = dict(config)
        self.end_of_turn_token_id = end_of_turn_token_id
        self.reward_callable = _load_reward_callable(self.config.get("reward_callable"))

    async def run(self) -> Trajectory:
        """Run the SDK, fetch exact network evidence, score, and convert it."""
        session_id = uuid.uuid4().hex
        artifact_dir, workspace_dir, session_root = self._prepare_directories(
            session_id
        )
        timeout = float(self.config.get("request_timeout", 600.0))
        registered = False
        try:
            await asyncio.to_thread(
                _http_json,
                "POST",
                f"{self.admin_url}/internal/sessions",
                {
                    "session_id": session_id,
                    "policy_version": self.policy_version,
                    "artifact_dir": str(artifact_dir),
                    "max_completions": int(self.config["max_turns"]),
                    "generation": self._generation_settings(),
                },
                timeout,
            )
            registered = True
            started = time.perf_counter()
            final_answer, finish_reason, events = await asyncio.to_thread(
                self._run_harness,
                session_id,
                artifact_dir,
                workspace_dir,
                session_root,
            )
            captured = await asyncio.to_thread(
                _http_json,
                "GET",
                f"{self.admin_url}/internal/sessions/{session_id}",
                None,
                timeout,
            )
            contract_error = self._capture_contract_error(captured)
            reward_result = self.reward_callable(final_answer, self.prompt)
            if not isinstance(reward_result, RewardResult):
                reward_value = float(reward_result)
                reward_result = RewardResult(reward_value, {"outcome": reward_value})
            if finish_reason in {"error", "aborted"}:
                contract_error = (
                    f"DeepSeek Harness ended with {finish_reason}"
                    if contract_error is None
                    else f"{contract_error}; Harness ended with {finish_reason}"
                )
            if contract_error is not None:
                reward_result = RewardResult(
                    0.0,
                    {**reward_result.components, "tool_contract": 0.0},
                    {**reward_result.metadata, "tool_contract_error": contract_error},
                )
            elif self.config.get("required_tool_calls") is not None:
                reward_result = RewardResult(
                    reward_result.value,
                    {**reward_result.components, "tool_contract": 1.0},
                    reward_result.metadata,
                )
            return build_deepseek_trajectory(
                prompt=self.prompt,
                policy_version=self.policy_version,
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
                    "deepseek_harness_version": str(
                        self.config.get("version", DEFAULT_DEEPSEEK_HARNESS_VERSION)
                    ),
                    "deepseek_session_id": session_id,
                    "artifact_dir": str(artifact_dir),
                    "workspace_dir": str(workspace_dir),
                    "generation_seconds": time.perf_counter() - started,
                    "final_answer": final_answer,
                    "finish_reason": finish_reason,
                    "deepseek_events": events,
                    **dict(reward_result.metadata),
                },
            )
        finally:
            if registered:
                await asyncio.to_thread(
                    _http_json,
                    "DELETE",
                    f"{self.admin_url}/internal/sessions/{session_id}",
                    None,
                    timeout,
                )

    def _validate_capture(self, captured: Mapping[str, Any]) -> None:
        """Validate captured policy identity and required tool-call evidence."""
        if captured.get("policy_version") != self.policy_version:
            raise RuntimeError("DeepSeek gateway returned a different policy version")
        expected_tool_calls = self.config.get("required_tool_calls")
        expected_tool_name = self.config.get("required_tool_name")
        if expected_tool_calls is None and expected_tool_name is None:
            return
        actual_tool_calls, tool_names = _captured_tool_calls(captured)
        if (
            expected_tool_calls is not None
            and actual_tool_calls != int(expected_tool_calls)
        ):
            raise RuntimeError(
                "DeepSeek episode violated its tool-call contract: "
                f"expected={int(expected_tool_calls)}, actual={actual_tool_calls}"
            )
        if expected_tool_name is not None and any(
            name != expected_tool_name for name in tool_names
        ):
            raise RuntimeError(
                "DeepSeek episode called an unexpected tool: "
                f"expected={expected_tool_name!r}, actual={tool_names}"
            )

    def _capture_contract_error(self, captured: Mapping[str, Any]) -> str | None:
        """Keep stochastic tool mistakes as zero-reward RL evidence."""
        try:
            self._validate_capture(captured)
        except RuntimeError as error:
            message = str(error)
            if "tool-call contract" not in message and "unexpected tool" not in message:
                raise
            return message
        return None

    def _generation_settings(self) -> dict[str, Any]:
        return harness_generation_settings(
            self.config,
            self.prompt.prompt_id,
            self.policy_version,
            self.sample_index,
            reasoning_effort=str(self.config.get("reasoning_effort", "off")),
        )

    def _prepare_directories(self, session_id: str) -> tuple[Path, Path, Path]:
        """Create isolated workspace and runtime directories for this session."""
        root = (
            Path(str(self.config.get("session_root", "/tmp/hyper-rl-deepseek")))
            .expanduser()
            .resolve()
        )
        artifact_dir = (
            root / f"{self.prompt.prompt_id}-{self.sample_index}-{session_id}"
        )
        workspace_dir = artifact_dir / "workspace"
        session_root = artifact_dir / "sessions"
        artifact_dir.mkdir(parents=True, exist_ok=False)
        template_value = self.config.get("workspace_template")
        if template_value:
            template = Path(str(template_value)).expanduser().resolve()
            if not template.is_dir():
                raise ValueError(
                    f"DeepSeek workspace template does not exist: {template}"
                )
            shutil.copytree(template, workspace_dir)
        else:
            workspace_dir.mkdir()
        session_root.mkdir()
        return artifact_dir, workspace_dir, session_root

    def _write_dsh_settings(self, artifact_dir: Path) -> Path:
        """Pin per-episode model behavior without reading ambient user settings."""
        dsh_home = artifact_dir / "dsh-home"
        dsh_home.mkdir()
        settings = {
            "agent-default-model": {
                "provider": str(self.config.get("provider", "deepseek-official")),
                "model": str(self.config.get("model", "policy")),
                "reasoningEffort": str(
                    self.config.get("reasoning_effort", "off")
                ),
            }
        }
        (dsh_home / "settings.yaml").write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return dsh_home

    def _run_harness(
        self,
        session_id: str,
        artifact_dir: Path,
        workspace_dir: Path,
        session_root: Path,
    ) -> tuple[str, str | None, list[dict[str, Any]]]:
        """Run the external harness and retain its output for trajectory validation."""
        expected = str(self.config.get("version", DEFAULT_DEEPSEEK_HARNESS_VERSION))
        harness_class, config_class = _load_sdk(expected)
        dsh_home = self._write_dsh_settings(artifact_dir)
        gateway_host = urlparse(self.gateway_url).hostname
        no_proxy = [os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))]
        no_proxy.extend(("127.0.0.1", "localhost"))
        if gateway_host:
            no_proxy.append(gateway_host)
        environment = {
            "DSH_HOME": str(dsh_home),
            "DSH_TELEMETRY_DISABLED": "1",
            "NO_PROXY": ",".join(filter(None, no_proxy)),
            "no_proxy": ",".join(filter(None, no_proxy)),
        }
        runtime_bin = self.config.get("runtime_bin")
        sdk_config = config_class(
            provider=str(self.config.get("provider", "deepseek-official")),
            model=str(self.config.get("model", "policy")),
            max_tokens=int(self.config["max_new_tokens"]),
            cwd=str(workspace_dir),
            runtime_cwd=str(workspace_dir),
            session_root=str(session_root),
            env=environment,
            runtime_bin=None if runtime_bin is None else str(runtime_bin),
            request_timeout_seconds=float(self.config.get("timeout_seconds", 1800.0)),
            shutdown_timeout_seconds=float(
                self.config.get("shutdown_timeout_seconds", 5.0)
            ),
            base_url=self.gateway_url,
            api_key=session_id,
        )
        prompt_text = self.prompt.messages[-1].content
        instruction = str(self.config.get("instruction_template", "{prompt}")).format(
            prompt=prompt_text
        )
        with harness_class(sdk_config) as harness:
            result = harness.run(instruction, session_id=session_id)
        events = [_json_value(event) for event in result.events]
        with (artifact_dir / "deepseek-events.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for event in events:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        summary = {
            "session_id": str(result.session_id),
            "final_response": str(result.final_response),
            "finish_reason": result.finish_reason,
            "notifications": [_json_value(item) for item in result.notifications],
        }
        (artifact_dir / "deepseek-result.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        final_answer = (
            "" if result.final_response is None else str(result.final_response)
        )
        if not final_answer and result.finish_reason not in {"error", "aborted"}:
            raise RuntimeError(
                f"DeepSeek Harness did not return a final response; see {artifact_dir}"
            )
        return final_answer, result.finish_reason, events


class DeepSeekProgramFactory(HarnessProgramFactory):
    """Create policy-identity-bound DeepSeek programs for ProgramAgentRunner."""

    program_type = DeepSeekAgentProgram
    label = "DeepSeek"
    include_admin_url = True


def _captured_tool_calls(captured):
    """Count captured tool calls while preserving malformed-record handling."""
    completions = captured.get("completions", [])
    actual_tool_calls = 0
    tool_names: list[str] = []
    if isinstance(completions, list):
        for completion in completions:
            if not isinstance(completion, Mapping):
                continue
            response = completion.get("response")
            choices = (
                response.get("choices")
                if isinstance(response, Mapping)
                else None
            )
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            message = (
                choice.get("message") if isinstance(choice, Mapping) else None
            )
            tool_calls = (
                message.get("tool_calls")
                if isinstance(message, Mapping)
                else None
            )
            if isinstance(tool_calls, list):
                actual_tool_calls += len(tool_calls)
                for tool_call in tool_calls:
                    function = (
                        tool_call.get("function")
                        if isinstance(tool_call, Mapping)
                        else None
                    )
                    name = (
                        function.get("name")
                        if isinstance(function, Mapping)
                        else None
                    )
                    tool_names.append(str(name or ""))
    return actual_tool_calls, tool_names
