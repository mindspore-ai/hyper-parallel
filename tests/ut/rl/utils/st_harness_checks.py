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
"""CPU contracts for ST recipes, isolated runs and acceptance evidence."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from hyper_parallel.rl.tests.st import st_evidence, st_runtime as runtime
from hyper_parallel.rl.tests.st.st_cases import CASES, Case, prepare_config
from hyper_parallel.rl.tests.st.st_evidence import metrics, validate_phase, validate_sessions
from hyper_parallel.rl.tests.st.st_runtime import ROOT, command
from rl.algorithm import build_algorithm
from rl.config.limits import resolve_vllm_automatic_limits
from rl.config.validate import validate_config
from tests.ut.rl.trainer.test_config_runtime import _complete_config


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_recipe_matches_case(case: Case, tmp_path: Path) -> None:
    """Generated workloads preserve the required topology and production entry."""
    devices = list(range(case.cards))
    config = prepare_config(case, 1, (8100, 8200), devices)
    assert config["algorithm"]["name"] == case.algorithm
    assert config["train"]["accelerator"]["tp"] == case.tp
    assert config["rollout"]["vllm"]["weight_sync"]["strategy"] == case.strategy
    assert config["model"]["weights_path"] == "/model"
    if config["evaluation"]["enabled"]:
        assert config["train"]["checkpoint"]["save_final"], "Evaluation needs a checkpoint boundary"
        assert config["evaluation"]["max_samples"] == 8
        assert config["evaluation"]["max_new_tokens"] == 512
    assert config["agentic"].get("runner", "internal") == case.runner
    assert config["train"]["accelerator"]["ep"] == 1
    if case.runner == "codex":
        instruction = config["agentic"]["codex"]["instruction_template"]
        assert "never call any tool again" in instruction
        assert "even to retry or verify" in instruction
        assert "login=false" in instruction
    if case.runner == "deepseek":
        instruction = config["agentic"]["deepseek"]["instruction_template"]
        assert "run_in_background" in instruction
        assert "job_output, job_list, or job_kill" in instruction
    args = command(case, tmp_path, tmp_path, tmp_path, "image", devices, 1, "test-name", master_port=29500)
    assert "RL_ST_MASTER_PORT=29500" in args, f"Missing isolated rendezvous port: command={args}"
    assert "PYTEST_ADDOPTS=-o log_cli=true -o log_cli_level=INFO -p no:cacheprovider" in args, (
        f"Worker INFO metrics must survive successful pytest execution: command={args}"
    )
    assert f"{ROOT}:/repo:ro" in args
    assert "_launch.py" in args[-1] and args[-1].endswith(str(case.world)), (
        f"Unexpected distributed launcher: command={args[-1]}"
    )
    if case.resume:
        resumed = prepare_config(case, 2, (8100, 8200), devices)
        checkpoint_step = 2 if case.name == "reward-model-resume" else 1
        assert resumed["train"]["checkpoint"]["load_path"] == f"/results/checkpoints/step_{checkpoint_step}"
        expected_steps = 4 if case.name == "reward-model-resume" else 3 if case.algorithm == "ppo" else 2
        assert resumed["train"]["max_steps"] == expected_steps


def test_all_st_recipes_pass_production_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every ST topology and resume phase must remain accepted by the real config validator."""
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: SimpleNamespace(type="npu"))
    validated_recipe = _complete_config(tmp_path)
    for case in CASES:
        devices = list(range(case.cards))
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", ",".join(map(str, devices[:case.world])))
        for phase in range(1, 3 if case.resume else 2):
            config = prepare_config(case, phase, (8100, 8200), devices)
            config["model"].update(validated_recipe["model"])
            if "reward_model" in config:
                config["reward_model"]["model_path"] = validated_recipe["model"]["weights_path"]
            for field in ("train_path", "test_path"):
                config["data"][field] = validated_recipe["data"][field]
            validate_config(resolve_vllm_automatic_limits(config), build_algorithm(config["algorithm"]))


def _evidence(output: Path, case: Case | None = None, phase: int = 1) -> Case:
    """Write one complete synthetic acceptance record, never a real ST result."""
    case = case or Case("synthetic", strategy="full_gather")
    lines = []
    steps = (phase,) if case.resume else (1, 2)
    if case.resume and case.algorithm == "ppo" and phase == 2:
        steps = (2, 3)
    for step in steps:
        row = {"train/global_step": step, "policy/version": step, "train/valid_tokens": 2,
               "train/optimizer_steps": 1, "rollout/generated_tokens": 2,
               "train/total_loss": 0.1, "train/gradient_norm": 1,
               f"weight_sync/last_{case.strategy}": 1,
               "critic/valid_tokens": 2, "critic/optimizer_steps": 1, "critic/gradient_norm": 1,
               "weight_sync/streaming_bucket_count": 2, "weight_sync/streaming_acked_buckets": 2,
               "weight_sync/streaming_released_buckets": 2, "weight_sync/streaming_max_inflight_buckets": 1,
               "weight_sync/streaming_max_gathered_bytes": 128, "weight_sync/streaming_max_packed_bytes": 128}
        if case.algorithm == "gspo":
            row = {key: value for key, value in row.items() if not key.startswith("critic/")}
            row.update({"train/valid_sequences": 1, "rollout/sequence_count": 1,
                        "train/sequence_clip_fraction": 0.5, "train/kl_loss": 0})
        if case.name.startswith("reward-model-"):
            row.update({"reward/request_count": 1, "reward/scored_sequences": 1, "reward/retry_count": 0})
            if step == steps[-1]:
                row.update({"validation/total": 8, "validation/reward_mean": 0.5})
        lines.append(f"INFO | step={step} | " + ", ".join(f"{key}={value}" for key, value in row.items()))
    (output / f"phase-{phase}.log").write_text("\n".join(lines))
    return case


def test_reward_model_evidence_rejects_missing_scores_and_fake_accuracy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only scored, request-backed RM runs may pass the CPU evidence gate."""
    case = Case("reward-model-synthetic", strategy="full_gather", algorithm="gspo")
    _evidence(tmp_path, case)
    monkeypatch.setattr(st_evidence, "_validate_reward_service_evidence", lambda *_args: None)
    monkeypatch.setattr(st_evidence, "_validate_checkpoint", lambda *_args: None)
    validate_phase(tmp_path, case, 1)
    log = tmp_path / "phase-1.log"
    original = log.read_text()
    for field, value, error in (
        ("reward/request_count", 0, "requests"),
        ("reward/scored_sequences", 0, "duplicated"),
        ("reward/accuracy", 0.5, "mislabeled"),
    ):
        log.write_text(original + f"\nINFO | step=1 | {field}={value}")
        with pytest.raises(AssertionError, match=error):
            validate_phase(tmp_path, case, 1)


@pytest.mark.parametrize("damage", ["none", "stale-version", "zero-gradient", "wrong-strategy"])
def test_evidence_accepts_complete_run_and_rejects_false_pass(tmp_path: Path, damage: str) -> None:
    """Reject stale versions, wrong strategies, and runs without learning."""
    case = _evidence(tmp_path)
    log = tmp_path / "phase-1.log"
    if damage == "stale-version":
        log.write_text(log.read_text().replace("policy/version=2", "policy/version=1"))
    elif damage == "zero-gradient":
        log.write_text(log.read_text().replace("train/gradient_norm=1", "train/gradient_norm=0"))
    elif damage == "wrong-strategy":
        log.write_text(
            log.read_text().replace(
                "weight_sync/last_full_gather=1",
                "weight_sync/last_full_gather=0",
            )
        )
    if damage == "none":
        validate_phase(tmp_path, case, 1)
        assert len(metrics(tmp_path / "phase-1.log")) == 2
    else:
        with pytest.raises(AssertionError):
            validate_phase(tmp_path, case, 1)


@pytest.mark.parametrize("damage", ["none", "no-release", "no-tool-feedback", "logprob-length", "nan-logprob",
                                   "positive-logprob", "boolean-logprob", "missing-logprob"])
@pytest.mark.parametrize("logprob_format", ["completion", "chat"])
def test_agent_requires_roundtrip_and_release(tmp_path: Path, damage: str, logprob_format: str) -> None:
    """Require the full tool interaction and release, not just a registration."""
    trace = tmp_path / "gateway-events.jsonl"
    events = [{"type": "session.registered", "session_id": "s",
               "payload": {"policy_version": 0}}]
    for ordinal in range(2):
        messages = [{"role": "user", "content": "question"}]
        if ordinal and damage != "no-tool-feedback":
            messages.append({"role": "tool", "content": "answer"})
        values = {
            "logprob-length": [], "nan-logprob": [float("nan")], "positive-logprob": [0.5],
            "boolean-logprob": [False], "missing-logprob": [None],
        }.get(damage, [-0.5])
        logprobs = {"token_logprobs": values}
        if logprob_format == "chat":
            logprobs = {"content": [{} if value is None else {"logprob": value} for value in values]}
        events.append({
            "type": "completion.recorded", "session_id": "s",
            "payload": {"ordinal": ordinal, "request": {"messages": messages},
                        "response": {"prompt_token_ids": [1], "choices": [{
                            "token_ids": [2], "logprobs": logprobs,
                            "message": {"tool_calls": [{"id": "call"}] if ordinal == 0 else []},
                        }]}},
        })
    if damage != "no-release":
        events.append({"type": "session.released", "session_id": "s", "payload": {}})
    trace.write_text("\n".join(map(json.dumps, events)))
    if damage == "none":
        validate_sessions(tmp_path, (0,))
    else:
        with pytest.raises(AssertionError):
            validate_sessions(tmp_path, (0,))


@pytest.mark.parametrize("damage", ["none", "runtime", "hf-config", "hf-architecture", "hf-tokenizer",
                                   "hf-shard", "hf-index"])
@pytest.mark.parametrize("scenario", ["checkpoint-resume", "ppo-checkpoint-resume"])
def test_checkpoint_uses_current_distributed_format(tmp_path: Path, damage: str, scenario: str) -> None:
    """Checkpoint evidence requires resumable state and a complete final HF export."""
    case = Case(scenario, resume=True,
                algorithm="ppo" if scenario.startswith("ppo") else "grpo")
    _evidence(tmp_path, case)
    step = 1
    checkpoint = tmp_path / "checkpoints/step_1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint_complete.json").write_text(
        json.dumps({"step": step, "world_size": 2, "critic": case.algorithm == "ppo"}))
    (checkpoint / "extra_state.json").write_text(json.dumps({"global_step": step}))
    (checkpoint / "_rank0_.safetensors").write_bytes(b"model")
    for rank in range(2):
        local = checkpoint / f"rank_{rank}"
        local.mkdir()
        (local / f"{rank}.metadata").write_bytes(b"metadata")
        if damage != "runtime":
            (local / f"_rank{rank}_.bytes").write_bytes(b"state")
    export = checkpoint / "hf"
    export.mkdir()
    architecture = "HSDPQwen3ForCausalLM" if damage == "hf-architecture" else "Qwen3ForCausalLM"
    (export / "config.json").write_text(json.dumps({"model_type": "qwen3", "architectures": [architecture]}))
    (export / "tokenizer_config.json").write_text("{}")
    (export / "tokenizer.json").write_text("{}")
    (export / "model.safetensors").write_bytes(b"weights")
    if damage == "hf-config":
        (export / "config.json").unlink()
    elif damage == "hf-tokenizer":
        (export / "tokenizer.json").unlink()
    elif damage == "hf-shard":
        (export / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.weight": "missing.safetensors"}}))
    elif damage == "hf-index":
        (export / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    if damage != "none":
        with pytest.raises(AssertionError):
            validate_phase(tmp_path, case, 1)
    else:
        validate_phase(tmp_path, case, 1)


@pytest.mark.parametrize("phase", [1, 2])
def test_ppo_resume_rejects_actor_only_checkpoint(tmp_path: Path, phase: int) -> None:
    """Fresh-container resume requires an explicit Critic completion marker in both phases."""
    case = Case("ppo-checkpoint-resume", strategy="full_gather", algorithm="ppo", resume=True)
    _evidence(tmp_path, case, phase)
    step = 1 if phase == 1 else 3
    checkpoint = tmp_path / f"checkpoints/step_{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint_complete.json").write_text(json.dumps({"step": step, "world_size": 2}))
    with pytest.raises(AssertionError, match="does not include Critic"):
        validate_phase(tmp_path, case, phase)


def test_command_supports_separate_test_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mount the installed package when the CI test archive contains no production code."""
    tests_root = tmp_path / "testcases"
    package_root = tmp_path / "site-packages" / "hyper_parallel"
    monkeypatch.setattr(runtime, "ROOT", tests_root)
    monkeypatch.setattr(runtime, "EXAMPLES", package_root / "rl" / "examples")
    args = runtime.command(Case("split-archive"), tmp_path, tmp_path, tmp_path,
                           "runtime-image", [0, 1], 1, "rl-st-owned", master_port=29500)
    mounts = [args[index + 1] for index, value in enumerate(args) if value == "-v"]
    assert f"{tests_root}:/repo:ro" in mounts
    assert f"{package_root}:/repo/hyper_parallel:ro" in mounts
    assert f"{runtime.HERE}:/repo/hyper_parallel/rl/tests/st:ro" in mounts
    assert "/repo/hyper_parallel/rl/tests/st/_launch.py" in args[-1]


@pytest.mark.parametrize("required", ["0", "1"])
def test_missing_resources_skip_locally_but_fail_required_jobs(
    monkeypatch: pytest.MonkeyPatch, required: str,
) -> None:
    """Missing model selection must never turn a required CI case into a green skip."""
    monkeypatch.delenv("RL_ST_MODEL", raising=False)
    monkeypatch.setenv("RL_ST_REQUIRED", required)
    expected = pytest.fail.Exception if required == "1" else pytest.skip.Exception
    with pytest.raises(expected, match="Set RL_ST_MODEL"):
        runtime.resources(Case("missing"))


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
    case = Case("checkpoint-resume", resume=True)
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
            runtime.execute(["docker", "run"], tmp_path, 1, "rl-st-owned", 1, "test-image")
    elif outcome == "exit-error":
        with pytest.raises(AssertionError, match="Training exited 1"):
            runtime.execute(["docker", "run"], tmp_path, 1, "rl-st-owned", 1, "test-image")
    else:
        runtime.execute(["docker", "run"], tmp_path, 1, "rl-st-owned", 1, "test-image")
    expected = [
        ["docker", "rm", "-f", "rl-st-owned"],
        ["docker", "ps", "-aq", "--filter", "name=^rl-st-owned$"],
        ["docker", "run", "--rm", "--user", "0:0", "-v", f"{tmp_path.resolve()}:/results",
         "--entrypoint", "chown", "test-image", "-R", f"{os.getuid()}:{os.getgid()}", "/results"],
    ]
    assert commands == expected, f"Expected scoped cleanup={expected}, got={commands}"
