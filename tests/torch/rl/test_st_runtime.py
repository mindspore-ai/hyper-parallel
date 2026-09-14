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
"""CPU checks for ST failure reporting, resume phases, and bounded container cleanup."""

import json
from pathlib import Path
import subprocess
from unittest.mock import MagicMock

import pytest

from tests.torch.rl import st_runtime as runtime
from tests.torch.rl import test_rl_st as system_entry


def test_level0_entry_requires_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    """A selected gate case must fail rather than skip when its model is missing."""
    monkeypatch.delenv("RL_ST_MODEL", raising=False)
    monkeypatch.setenv("RL_ST_REQUIRED", "0")
    with pytest.raises(pytest.fail.Exception, match="Set RL_ST_MODEL"):
        system_entry.test_rl_system(runtime.Case("required-gate"), monkeypatch)


def test_command_supports_separate_test_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mount the installed package when the CI test archive contains no production code."""
    tests_root = tmp_path / "testcases"
    package_root = tmp_path / "site-packages" / "hyper_parallel"
    monkeypatch.setattr(runtime, "ROOT", tests_root)
    monkeypatch.setattr(runtime, "EXAMPLES", package_root / "rl" / "examples")
    args = runtime.command(runtime.Case("split-archive"), tmp_path, tmp_path, tmp_path,
                           "runtime-image", [0, 1], 1, "rl-st-owned", master_port=29500)
    mounts = [args[index + 1] for index, value in enumerate(args) if value == "-v"]
    assert f"{tests_root}:/repo:ro" in mounts
    assert f"{package_root}:/repo/hyper_parallel:ro" in mounts


@pytest.mark.parametrize("required", ["0", "1"])
def test_missing_resources_skip_locally_but_fail_required_jobs(
    monkeypatch: pytest.MonkeyPatch, required: str,
) -> None:
    """Missing model selection must never turn a required CI case into a green skip."""
    monkeypatch.delenv("RL_ST_MODEL", raising=False)
    monkeypatch.setenv("RL_ST_REQUIRED", required)
    expected = pytest.fail.Exception if required == "1" else pytest.skip.Exception
    with pytest.raises(expected, match="Set RL_ST_MODEL"):
        runtime.resources(runtime.Case("missing"))


@pytest.mark.parametrize("failure", ["training", "evidence", "none"])
def test_resume_reports_failure_without_starting_the_next_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """A failed phase is recorded durably and cannot be combined with a later success."""
    monkeypatch.setenv("RL_ST_RESULT_ROOT", str(tmp_path))
    monkeypatch.setattr(runtime, "resources", lambda _case: (tmp_path, tmp_path, "image", [0, 1]))
    monkeypatch.setattr(runtime, "available_port", MagicMock(side_effect=range(20000, 20006)))
    monkeypatch.setattr(runtime.socket, "socket", MagicMock())
    execute = MagicMock(side_effect=RuntimeError("training failed") if failure == "training" else None)
    validate = MagicMock(side_effect=AssertionError("evidence failed") if failure == "evidence" else None)
    monkeypatch.setattr(runtime, "execute", execute)
    monkeypatch.setattr(runtime, "validate_phase", validate)
    case = runtime.Case("checkpoint-resume", resume=True)
    if failure == "none":
        runtime.run_case(case)
    else:
        with pytest.raises((RuntimeError, AssertionError), match=f"{failure} failed"):
            runtime.run_case(case)
    reports = list(tmp_path.glob("*/result.json"))
    assert len(reports) == 1, f"Expected one durable report, got={reports}"
    report = json.loads(reports[0].read_text())
    expected_status = "passed" if failure == "none" else "failed"
    assert report["status"] == expected_status, f"Expected status={expected_status}, got={report}"
    expected_phases = 2 if failure == "none" else 1
    assert len(report["phases"]) == expected_phases, f"Expected phases={expected_phases}, got={report}"
    assert execute.call_count == expected_phases, (
        f"Expected launches={expected_phases}, got={execute.call_count}"
    )
    if failure != "none":
        assert report["error"] == f"{failure} failed", f"Unexpected failure report={report}"
    else:
        names = [phase["command"][phase["command"].index("--name") + 1] for phase in report["phases"]]
        assert len(set(names)) == 2, f"Resume must use fresh containers, got={names}"
        assert (reports[0].parent / "phase-2.yaml").is_file(), f"Missing resume YAML beside={reports[0]}"


@pytest.mark.parametrize("outcome", ["success", "exit-error", "timeout"])
def test_execute_cleans_only_its_owned_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    """Success, nonzero exit, and timeout must all reach exact-name container cleanup."""
    process = MagicMock()
    process.__enter__.return_value = process
    process.wait.side_effect = subprocess.TimeoutExpired("docker", 1) if outcome == "timeout" else None
    process.wait.return_value = 1 if outcome == "exit-error" else 0
    process.poll.return_value = 0
    monkeypatch.setattr(runtime.subprocess, "Popen", MagicMock(return_value=process))
    commands = []

    def run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        """Record exact Docker targets without accessing a real daemon."""
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(runtime.subprocess, "run", run)
    if outcome == "timeout":
        with pytest.raises(subprocess.TimeoutExpired):
            runtime.execute(["docker", "run"], tmp_path, 1, "rl-st-owned", 1)
    elif outcome == "exit-error":
        with pytest.raises(AssertionError, match="Training exited 1"):
            runtime.execute(["docker", "run"], tmp_path, 1, "rl-st-owned", 1)
    else:
        runtime.execute(["docker", "run"], tmp_path, 1, "rl-st-owned", 1)
    expected = [
        ["docker", "rm", "-f", "rl-st-owned"],
        ["docker", "ps", "-aq", "--filter", "name=^rl-st-owned$"],
    ]
    assert commands == expected, f"Expected scoped cleanup={expected}, got={commands}"
