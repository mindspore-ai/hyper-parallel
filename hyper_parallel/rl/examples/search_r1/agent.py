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
"""Search task task outcomes and one recorded citation repair call."""

import asyncio
import json
import re
from pathlib import Path
import urllib.request

from rl.agentic.codex.harness import CodexAgentProgram, build_codex_call_trajectories
from rl.tool_diagnostics import summarize_tool_protocol
from rl.tool_protocol import InfrastructureRolloutError
from rl.agentic.core.program_runner import audit_training_rows, request_gateway_json
from examples.search_r1.container_execution import close_bridges


TASK_FAILURES = {"budget_exhausted", "repeated_tool_failure", "tool_format_budget_exhausted"}


def repair_request(payload: dict, answer: str, errors: list[str]) -> dict:
    """Request a new model action without rewriting the previous generated answer."""
    result = dict(payload)
    result.update(stream=False, tools=[], tool_choice="none")
    result.pop("previous_response_id", None)
    result["input"] = list(payload["input"]) + [
        {"type": "message", "role": "assistant", "content": answer},
        {"type": "message", "role": "user", "content": "One final citation correction, no tools. Errors: "
         + json.dumps(errors, ensure_ascii=False)
         + ". Copy only REF references already visible in the tool observations. "
         "Return ANSWER and SOURCES lines. Do not invent evidence; use UNKNOWN/NONE if insufficient."},
    ]
    return result


def validate_citations(answer: str, workspace: Path, visible_outputs: list[str]) -> dict:
    """Check exact candidate references and visibility, not semantic entailment."""
    lines = answer.strip().splitlines()
    errors = []
    if len(lines) != 2 or not lines[0].startswith("ANSWER: ") or not lines[1].startswith("SOURCES: "):
        return {"valid": False, "errors": ["expected_two_line_format"]}
    if lines == ["ANSWER: UNKNOWN", "SOURCES: NONE"]:
        return {"valid": False, "errors": ["insufficient_evidence"]}
    if not lines[0][len("ANSWER: "):].strip():
        return {"valid": False, "errors": ["empty_answer"]}
    candidates = set()
    for article in (workspace / "articles").glob("*.txt"):
        candidates.update(re.findall(r"^\[REF: (.+:\d+)\]", article.read_text(), re.MULTILINE))
    visible = "\n".join(visible_outputs)
    sources = [source.strip() for source in lines[1][len("SOURCES: "):].split(";")]
    for source in sources:
        if source not in candidates:
            errors.append("unknown_reference:" + source)
        elif f"[REF: {source}]" not in visible:
            errors.append("unseen_reference:" + source)
    return {"valid": not errors, "errors": errors, "sources": sources}


class SearchR1Program(CodexAgentProgram):
    """Keep task-failure actions with zero reward; infrastructure errors still raise."""

    async def _run_registered(self, session_id, artifact_dir, workspace_dir, codex_home, timeout):
        """Clean up only this episode's relay, including cancellation and errors."""
        try:
            return await self._run_search(session_id, artifact_dir, workspace_dir, codex_home, timeout)
        except Exception as error:
            terminal_path = artifact_dir / "search-terminal.json"
            terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else {}
            execution_path = artifact_dir / "execution-result.json"
            container_failed = (execution_path.exists()
                                and json.loads(execution_path.read_text()).get("returncode") not in (0, 75))
            captured = {"completions": []}
            try:
                captured = await self._capture_raw(session_id, timeout)
            except Exception:  # pylint: disable=broad-exception-caught
                # Preserve the primary exception; missing evidence stays unknown.
                pass
            completion_failures = [
                record.get("metadata", {}) for record in captured.get("completions", [])
                if record.get("metadata", {}).get("trainable") is not True
            ]
            origin = "infrastructure"
            reason = "container_exit" if container_failed else type(error).__name__
            if completion_failures and not container_failed:
                origin = completion_failures[-1].get("failure_origin", origin)
                reason = completion_failures[-1].get("failure_reason", reason)
            failed_outcome = {
                "codex_session_id": session_id,
                "failure_origin": origin if container_failed else terminal.get("failure_origin", origin),
                "failure_reason": reason if container_failed else terminal.get("failure_reason", reason),
                "trainable": False, "error": str(error), "harness_completed": False,
                "citation_valid": False, "task_success": False, "reward_evaluated": False,
                "reward": None, "training_rows_emitted": 0,
            }
            (artifact_dir / "search-outcome.json").write_text(json.dumps(failed_outcome, ensure_ascii=False))
            summary = summarize_tool_protocol(captured.get("completions", []), failed_outcome)
            (artifact_dir / "tool-error-summary.json").write_text(json.dumps(summary, indent=2))
            raise InfrastructureRolloutError(
                f"Search-R1 infrastructure failure ({failed_outcome['failure_reason']}): {error}"
            ) from error
        finally:
            await asyncio.to_thread(close_bridges, session_id)

    async def _execute_search_process(self, session_id, artifact_dir, workspace_dir, codex_home):
        """Run Codex and classify a trainable task terminal."""
        self._write_codex_config(codex_home, session_id)
        await self._validate_version()
        answer = ""
        outcome = "completed"
        diagnostics = []
        try:
            answer, return_code, diagnostics = await self._run_codex(
                session_id, artifact_dir, workspace_dir, codex_home
            )
            if return_code != 0:
                raise RuntimeError(f"Codex exited with status {return_code}; see {artifact_dir}")
        except RuntimeError:
            terminal_path = artifact_dir / "search-terminal.json"
            if not terminal_path.exists():
                raise
            terminal = json.loads(terminal_path.read_text())
            outcome = terminal.get("reason")
            if (outcome not in TASK_FAILURES or terminal.get("failure_origin") != "model"
                    or terminal.get("trainable") is not True):
                raise
        # Codex may return normally after receiving an HTTP error; process success is not rollout success.
        terminal_path = artifact_dir / "search-terminal.json"
        if terminal_path.exists():
            terminal = json.loads(terminal_path.read_text())
            if (terminal.get("reason") not in TASK_FAILURES or terminal.get("failure_origin") != "model"
                    or terminal.get("trainable") is not True):
                raise RuntimeError("Untrainable search terminal: " + str(terminal))
            outcome = terminal["reason"]
        execution_path = artifact_dir / "execution-result.json"
        if execution_path.exists() and json.loads(execution_path.read_text()).get("returncode") not in (0, 75):
            raise RuntimeError("Agent container failed independently of the task outcome")
        return answer, outcome, diagnostics

    async def _check_search_citations(self, session_id, artifact_dir, workspace_dir, timeout, answer, outcome):
        """Validate visible citations and make at most one correction call."""
        captured = await self._capture(session_id, timeout)
        observations_path = artifact_dir / "search-observations.json"
        observations = json.loads(observations_path.read_text()) if observations_path.exists() else []
        citation = validate_citations(answer, workspace_dir, observations)
        repaired = False
        if outcome == "completed" and not citation["valid"]:
            if len(captured["completions"]) < int(self.config["max_turns"]):
                previous = answer
                answer = await asyncio.to_thread(self._repair, session_id, artifact_dir, answer, citation, timeout)
                repaired = True
                (artifact_dir / "citation-repair.json").write_text(json.dumps({
                    "original_answer": previous, "repaired_answer": answer,
                }, ensure_ascii=False))
                captured = await self._capture(session_id, timeout)
                citation = validate_citations(answer, workspace_dir, observations)
            if not citation["valid"]:
                outcome = "invalid_citation"
        if not captured.get("completions"):
            raise RuntimeError("No sampled actions available; this is not a trainable task failure")
        return captured, citation, answer, outcome, repaired

    async def _run_search(self, session_id, artifact_dir, workspace_dir, codex_home, timeout):
        """Run one isolated search episode and record its trainable trajectories."""
        answer, outcome, diagnostics = await self._execute_search_process(
            session_id, artifact_dir, workspace_dir, codex_home
        )
        captured, citation, answer, outcome, repaired = await self._check_search_citations(
            session_id, artifact_dir, workspace_dir, timeout, answer, outcome
        )
        scored = self.reward_callable(answer, self.prompt) if outcome == "completed" else None
        reward = scored.value if scored else 0.0
        metadata = {
            "phase": self.prompt.metadata.get("phase", "training"),
            "artifact_dir": str(artifact_dir), "workspace_dir": str(workspace_dir),
            "final_answer": answer, "task_outcome": outcome, "citation_check": citation,
            "citation_repaired": repaired, "codex_diagnostics": diagnostics,
            "codex_session_id": session_id, "failure_reward": 0.0,
            "failure_origin": None if outcome == "completed" else "model",
            "failure_reason": None if outcome == "completed" else outcome, "trainable": True,
            "harness_completed": True, "citation_valid": bool(citation["valid"]),
            "task_success": bool(scored is not None and scored.value > 0),
            "reward_evaluated": scored is not None, "reward": reward,
            "training_rows_emitted": len(captured["completions"]),
        }
        summary = summarize_tool_protocol(captured["completions"], metadata)
        metadata["tool_protocol_metrics"] = {key: value for key, value in summary.items() if key != "events"}
        (artifact_dir / "search-outcome.json").write_text(json.dumps(metadata, ensure_ascii=False))
        (artifact_dir / "tool-error-summary.json").write_text(json.dumps(summary, indent=2))
        trajectories = build_codex_call_trajectories(
            prompt=self.prompt, policy_version=self.policy_version, sample_index=self.sample_index,
            completion_records=captured["completions"], reward=reward,
            reward_components=scored.components if scored else {"task_failure": 1.0},
            max_episode_tokens=self.config.get("max_episode_tokens"), metadata=metadata,
        )
        audit = audit_training_rows(trajectories)
        (artifact_dir / "training-audit.json").write_text(json.dumps(audit))
        return trajectories

    async def _capture(self, session_id: str, timeout: float) -> dict:
        """Read gateway records and reject untrainable failures."""
        captured = await self._capture_raw(session_id, timeout)
        failure = captured.get("failure")
        if failure is not None and (
            failure.get("trainable") is not True or failure.get("failure_origin") != "model"
        ):
            raise InfrastructureRolloutError("Gateway reported an untrainable failure: " + str(failure))
        for record in captured.get("completions", []):
            metadata = record.get("metadata", {})
            if metadata.get("trainable", True) is not True:
                raise InfrastructureRolloutError("Untrainable gateway completion: " + str(metadata))
        return captured

    async def _capture_raw(self, session_id: str, timeout: float) -> dict:
        """Fetch immutable records; callers decide whether failures are trainable."""
        captured = await asyncio.to_thread(
            request_gateway_json, "Codex", "GET",
            self.gateway_url + "/internal/sessions/" + session_id + "?live=1", None, timeout
        )
        if captured.get("policy_version") != self.policy_version:
            raise RuntimeError("Search-R1 policy version mismatch")
        return captured

    def _repair(self, session_id: str, artifact: Path, answer: str, citation: dict, timeout: float) -> str:
        """Request one citation correction without changing the sampled answer."""
        payload = repair_request(
            json.loads((artifact / "search-last-request.json").read_text()), answer, citation["errors"]
        )
        request = urllib.request.Request(
            self.gateway_url + "/responses", json.dumps(payload).encode(),
            {"Content-Type": "application/json", "Authorization": "Bearer " + session_id}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        return "\n".join(
            block["text"] for item in result.get("output", []) if item.get("type") == "message"
            for block in item.get("content", []) if block.get("type") == "output_text"
        )
