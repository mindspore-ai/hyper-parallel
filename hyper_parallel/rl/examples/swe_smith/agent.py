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
"""SWE-Smith episode: prepare, execute Codex, freeze, grade, and build training rows."""

import asyncio
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any

from rl.agentic.codex.harness import (
    CodexAgentProgram, DEFAULT_CODEX_VERSION, _classify_codex_events,
    _has_completed_codex_turn, _toml_string,
)
from rl.agentic.codex.gateway import CompletionBudgetExhausted
from rl.agentic.codex.cli_contract import ISOLATED_COMPACT_PROMPT
from rl.agentic.core.types import RewardResult
from rl.agentic.envs.docker_workspace import (
    DockerWorkspace, InvalidSubmissionError, acquire_workspace_slot,
)
from rl.dataset.contracts import Trajectory
from rl.agentic.codex.container_execution import CandidateRelay, run_container_codex
from examples.swe_smith.reward import SWESmithTask


_LOGGER = logging.getLogger(__name__)


class _ExecutionError(RuntimeError):
    """CLI failure whose attribution must be checked against captured sampling evidence."""


class SWESmithProgram(CodexAgentProgram):
    """Own repository execution without introducing task branches into the shared harness."""

    def _session_settings(self) -> dict[str, Any]:
        return {"isolated_cli": True, "max_observation_chars": self.config.get("max_observation_chars")}

    async def _run_registered(self, session_id, artifact_dir, workspace_dir, codex_home, timeout):
        """Keep one lease through candidate execution and independent grading."""
        if not self.admin_token:
            raise ValueError("SWE-Smith requires controller management authentication")
        if self.config.get("mcp_servers") or self.config.get("workspace_template"):
            raise ValueError("SWE-Smith uses isolated container tools, not host templates or MCP")
        settings = self.config["workspace"]
        allowed = {"memory", "cpus", "pids_limit", "max_concurrent", "output_limit_bytes", "run_id", "docker_host"}
        if set(settings) - allowed:
            raise ValueError("SWE-Smith workspace contains unsupported settings")
        capacity = settings.get("max_concurrent", 1)
        output_limit = settings.get("output_limit_bytes", 4 * 1024 * 1024)
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in (capacity, output_limit)):
            raise ValueError("SWE-Smith workspace budgets must be positive integers")
        async with acquire_workspace_slot(artifact_dir.parent / ".workspace-slots", capacity, timeout):
            return await self._run_repository(session_id, artifact_dir, codex_home, timeout)

    def _write_codex_config(self, codex_home: Path, session_id: str, **kwargs: Any) -> None:
        """Apply the fixed isolated CLI profile after writing the shared provider settings."""
        if self.config.get("mcp_servers"):
            raise ValueError("SWE-Smith uses isolated container tools, not MCP servers")
        super()._write_codex_config(codex_home, session_id, **kwargs)
        context = self.config.get("model_context_window")
        if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
            raise ValueError("SWE-Smith requires a positive model_context_window")
        path = codex_home / "config.toml"
        prefix = (f"model_context_window = {context}\n"
                  "tool_output_token_limit = 10000\n"
                  f"compact_prompt = {_toml_string(ISOLATED_COMPACT_PROMPT)}\n")
        compact_limit = self.config.get("model_auto_compact_token_limit")
        if compact_limit is not None:
            if (isinstance(compact_limit, bool) or not isinstance(compact_limit, int)
                    or not 0 < compact_limit <= context):
                raise ValueError("model_auto_compact_token_limit must fit the model context window")
            prefix += f"model_auto_compact_token_limit = {compact_limit}\n"
        features = ("[features]\nview_image = false\ngoals = false\n"
                    "default_mode_request_user_input = false\n")
        suffix = ("\n[tools.experimental_request_user_input]\nenabled = false\n"
                  "\n[skills]\ninclude_instructions = false\n\n[skills.bundled]\nenabled = false\n")
        path.write_text(prefix + path.read_text().replace("[features]\n", features) + suffix)

    async def _run_repository(self, session_id: str, artifact_dir: Path,
                              codex_home: Path, timeout: float) -> tuple[Trajectory, ...]:
        """Run isolated repository tools and score only a frozen submission."""
        settings = dict(self.config["workspace"])
        settings.pop("max_concurrent", None)
        settings.pop("output_limit_bytes", None)
        run_id = settings.pop("run_id", self.config.get("run_id", session_id))
        task_config = dict(self.config["task"])
        task_config["run_id"] = run_id
        task_config["docker_host"] = settings.get("docker_host")
        task_config["artifact_dir"] = str(artifact_dir)
        task = SWESmithTask(self.prompt, task_config)
        settings["image"] = task.workspace_image
        workspace = DockerWorkspace(**settings, network="none", name_prefix="hyper-repository-candidate",
                                    run_id=task_config["run_id"])
        relay = CandidateRelay(workspace, self.gateway_url, session_id, timeout,
                               int(self.config.get("max_request_bytes", 8 * 1024 * 1024)),
                               int(self.config.get("max_response_bytes", 32 * 1024 * 1024)),
                               admin_url=self.admin_url, admin_token=self.admin_token)
        started = time.perf_counter()
        try:
            await workspace.start()
            await task.prepare(workspace)
            model_url = await relay.start()
            self._write_codex_config(codex_home, session_id, gateway_url=model_url,
                                     runtime_home=Path("/opt/hyper-codex-home"))
            await workspace.copy_in(codex_home, "/opt/hyper-codex-home")
            await self._validate_container_version(workspace)
            execution_error = None
            final_answer, diagnostics = "", []
            budget_termination = None
            try:
                final_answer, diagnostics, budget_termination = await self._run_repository_cli(
                    workspace, relay, session_id, artifact_dir, model_url,
                )
            except _ExecutionError as error:
                execution_error = error
            # Transport failure takes precedence even when the model also exhausted its budget.
            await relay.close()
            captured = await self._admin_request("GET", f"/internal/sessions/{session_id}", None, timeout)
            failure_reward = self._captured_failure(captured, execution_error)
            if budget_termination is not None:
                if (failure_reward is not None or captured.get("termination") != budget_termination
                        or budget_termination.get("policy_version") != self.policy_version
                        or budget_termination.get("limit") != int(self.config["max_turns"])
                        or len(captured.get("completions", [])) != budget_termination["limit"]):
                    raise RuntimeError("Repository budget termination differs from the captured session")
            await workspace.stop()
            if failure_reward is not None:
                reward = failure_reward
            else:
                if execution_error is not None:
                    raise execution_error
                reward = await self._grade_submission(workspace, task, artifact_dir)
            if budget_termination is not None:
                reward = RewardResult(reward.value, reward.components,
                                      {**reward.metadata, "budget_termination": budget_termination,
                                       "repository_budget_truncated": True})
            return self._build_rows(session_id, artifact_dir, Path("/workspace"), started,
                                    final_answer, diagnostics, captured, reward)
        finally:
            original_error = sys.exc_info()[1]
            cleanup = asyncio.create_task(self._close_repository(relay, workspace))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await asyncio.gather(cleanup, return_exceptions=True)
                raise
            except Exception:  # pylint: disable=broad-exception-caught
                if original_error is None:
                    raise
                _LOGGER.exception("Repository cleanup failed while preserving original error")

    async def _grade_submission(self, workspace: DockerWorkspace, task: Any, artifact_dir: Path) -> RewardResult:
        """Grade the stopped candidate's frozen artifact, preserving infrastructure failures."""
        try:
            archive = await workspace.export(artifact_dir / "submission.tar", **task.artifact_limits)
        except InvalidSubmissionError as error:
            return RewardResult(0.0, {"success": 0.0},
                                {"status": "invalid_submission", "reason": str(error)})
        reward = await self.reward_callable(archive, task)
        if not isinstance(reward, RewardResult):
            raise TypeError("Repository task.evaluate must return RewardResult")
        return reward

    @staticmethod
    async def _close_repository(relay: CandidateRelay, workspace: DockerWorkspace) -> None:
        try:
            await relay.close()
        finally:
            await workspace.close()

    async def _run_repository_cli(self, workspace: DockerWorkspace, relay: CandidateRelay,
                                  session_id: str, artifact_dir: Path, model_url: str) -> tuple[str, list, dict | None]:
        """Drain a confirmed budget response before actively stopping the candidate CLI."""
        expected_stop = asyncio.Event()
        expected_budget = {}
        execution = asyncio.create_task(self._run_container_codex(
            workspace, session_id, artifact_dir, expected_stop=expected_stop, expected_budget=expected_budget,
        ))
        budget = asyncio.create_task(relay.budget_exhausted.wait())
        try:
            await asyncio.wait((execution, budget), return_when=asyncio.FIRST_COMPLETED)
            if execution.done():
                answer, diagnostics = await execution
                return answer, diagnostics, relay.budget_termination if relay.budget_exhausted.is_set() else None
            await relay.close()
            if execution.done():
                answer, diagnostics = await execution
                return answer, diagnostics, relay.budget_termination
            expected_budget.update(termination=relay.budget_termination, url=model_url.rstrip("/") + "/responses")
            expected_stop.set()
            await workspace.stop()
            answer, diagnostics = await execution
            return answer, diagnostics, relay.budget_termination
        finally:
            for task in (budget, execution):
                if not task.done():
                    task.cancel()
            await asyncio.gather(budget, execution, return_exceptions=True)

    async def _validate_container_version(self, workspace: DockerWorkspace) -> None:
        result = await workspace.exec([str(self.config.get("executable", "codex")), "--version"],
                                      timeout=30, output_limit_bytes=4096)
        expected = str(self.config.get("version", DEFAULT_CODEX_VERSION))
        installed = result.stdout or result.stderr
        if result.returncode or re.search(rf"(?<!\d){re.escape(expected)}(?!\d)", installed) is None:
            raise RuntimeError(f"Container Codex version mismatch: expected {expected}, got {installed.strip()!r}")

    async def _run_container_codex(self, workspace: DockerWorkspace, session_id: str,
                                   artifact_dir: Path, *, expected_stop: asyncio.Event | None = None,
                                   expected_budget: dict | None = None,
                                   ) -> tuple[str, list[dict[str, Any]]]:
        """Run the pinned CLI in the candidate and retain bounded output and termination evidence."""
        environment = {"HOME": "/opt/hyper-codex-home", "CODEX_HOME": "/opt/hyper-codex-home",
                       "OPENAI_API_KEY": session_id, "PYTHONDONTWRITEBYTECODE": "1",
                       "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
        result = await run_container_codex(
            workspace, self._codex_command(), artifact_dir, environment=environment,
            timeout=float(self.config.get("timeout_seconds", 1800)),
            output_limit_bytes=int(self.config["workspace"].get("output_limit_bytes", 4 * 1024 * 1024)),
            controller_stop=expected_stop,
        )
        final_answer, diagnostics, failures = _classify_codex_events(result.stdout.splitlines())
        if expected_stop is not None and expected_stop.is_set() and result.returncode in (137, 143):
            if expected_budget:
                message = str(CompletionBudgetExhausted(expected_budget["termination"]))
                pattern = (r"Reconnecting\.\.\. [1-5]/5 \(unexpected status 409 Conflict: "
                           + re.escape(message) + r", url: " + re.escape(expected_budget["url"]) + r"\)")
                retry_events = [event for event in failures if event.get("type") == "error"
                                and re.fullmatch(pattern, str(event.get("message", "")))]
                diagnostics.extend(retry_events)
                failures = [event for event in failures if event not in retry_events]
            if not failures:
                return final_answer, diagnostics
        if result.returncode or failures or not _has_completed_codex_turn(result.stdout):
            raise _ExecutionError(f"Container Codex did not complete cleanly; see {artifact_dir}")
        return final_answer, diagnostics
