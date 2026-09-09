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
"""CPU self-checks for ST configuration, evidence rejection and launcher isolation."""

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from st_evidence import metrics, tensor, validate_phase, validate_sessions
from st_runtime import CASES, ROOT, Case, prepare_config, command


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_recipe_matches_case(case: Case, tmp_path: Path) -> None:
    """Generated workloads preserve the required topology and production entry."""
    devices = list(range(case.cards))
    config = prepare_config(case, 1, (8100, 8200), devices)
    assert config["train"]["accelerator"]["tp"] == case.tp
    assert config["rollout"]["vllm"]["weight_sync"]["strategy"] == case.strategy
    assert config["model"]["weights_path"] == "/model"
    assert config["consistency"]["enabled"] is case.exact
    if config["evaluation"]["enabled"]:
        assert config["train"]["checkpoint"]["save_final"], "Evaluation needs a checkpoint boundary"
    assert config["agentic"].get("runner", "internal") == case.runner
    assert config["train"]["accelerator"]["ep"] == (4 if case.family != "qwen3" else 1)
    if case.family != "qwen3":
        assert config["agentic"]["environment"] == "moe_update_control"
        assert config["agentic"]["apply_chat_template"] is True
        assert config["rollout"]["max_new_tokens"] == 48
        assert config["rollout"]["temperature"] == 0.0
        assert config["train"]["prompt_batch_size"] == 1
        assert config["train"]["response_mini_batch_size"] == 4
    if case.runner == "codex":
        instruction = config["agentic"]["codex"]["instruction_template"]
        assert "never call any tool again" in instruction
        assert "even to retry or verify" in instruction
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
        assert resumed["train"]["checkpoint"]["load_path"] == "/results/checkpoints/step_1"
        assert resumed["train"]["max_steps"] == 2


def _tensor(values: list, dtype: str) -> dict:
    fmt = {"int64": "q", "int32": "i", "float32": "f"}[dtype]
    raw = b"".join(struct.pack("<" + fmt, value) for row in values for value in row)
    return {"values": values, "shape": [len(values), len(values[0])],
            "dtype": dtype, "sha256": hashlib.sha256(raw).hexdigest()}


def _evidence(output: Path) -> Case:
    """Write one complete synthetic acceptance record, never a real ST result."""
    case = Case("synthetic", strategy="direct_reshard")
    (output / "rollouts").mkdir()
    (output / "manifests").mkdir()
    lines = []
    for step in (1, 2):
        row = {"train/global_step": step, "policy/version": step, "train/valid_tokens": 2,
               "train/optimizer_steps": 1, "rollout/generated_tokens": 2,
               "train/total_loss": 0.1, "train/gradient_norm": 1, "policy/fingerprint_changed": 1,
               "weight_sync/completed_direct_reshard": 1, "weight_sync/fallback_count": 0}
        lines.append(f"INFO | step={step} | " + ", ".join(f"{key}={value}" for key, value in row.items()))
        for rank in range(2):
            artifact = {"oracle_run_id": output.name, "trainer_rank": rank,
                        "policy_version": step - 1, "worker_policy_version": step - 1,
                        "worker_policy_fingerprint": f"identity-{step-1}",
                        "sequences": _tensor([[1, 2, 3]], "int64"),
                        "attention_mask": _tensor([[1, 1, 1]], "int32"),
                        "action_mask": _tensor([[0, 1, 1]], "int32"),
                        "raw_logprobs": _tensor([[-0.5, -0.25]], "float32")}
            (output / "rollouts" / f"direct_reshard-policy{step-1}-rank{rank}.json").write_text(json.dumps(artifact))
            tensors = {"weight": {"num_bytes": 4, "sha256": str(step) * 64, "shape": [1], "dtype": "float32"}}
            digest = hashlib.sha256(json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            manifest = {"oracle_run_id": output.name, "policy_version": step,
                        "dp_rank": rank, "dp_size": 2, "tp_rank": 0, "tp_size": 1,
                        "tensors": tensors, "total_bytes": 4, "parameter_count": 1,
                        "manifest_sha256": digest, "model_type": "qwen3"}
            (output / "manifests" / f"direct_reshard-version{step}-dp{rank}.json").write_text(json.dumps(manifest))
    (output / "phase-1.log").write_text("\n".join(lines))
    return case


@pytest.mark.parametrize("damage", ["none", "missing-worker", "stale-version", "zero-gradient", "bad-digest"])
def test_evidence_accepts_complete_run_and_rejects_false_pass(tmp_path: Path, damage: str) -> None:
    """Reject incomplete publication, stale generation, no learning and damaged data."""
    case = _evidence(tmp_path)
    artifact_path = tmp_path / "rollouts/direct_reshard-policy1-rank0.json"
    if damage == "missing-worker":
        (tmp_path / "manifests/direct_reshard-version2-dp0.json").unlink()
    elif damage in ("stale-version", "bad-digest"):
        record = json.loads(artifact_path.read_text())
        if damage == "stale-version":
            record["worker_policy_version"] = 0
        else:
            record["sequences"]["sha256"] = "0" * 64
        artifact_path.write_text(json.dumps(record))
    elif damage == "zero-gradient":
        log = tmp_path / "phase-1.log"
        log.write_text(log.read_text().replace("train/gradient_norm=1", "train/gradient_norm=0"))
    if damage == "none":
        validate_phase(tmp_path, case, 1)
        assert len(metrics(tmp_path / "phase-1.log")) == 2
        assert tensor(_tensor([[1]], "int64")) == [[1]]
    else:
        with pytest.raises(AssertionError):
            validate_phase(tmp_path, case, 1)


@pytest.mark.parametrize("damage", ["none", "no-release", "no-tool-feedback"])
def test_agent_requires_roundtrip_and_release(tmp_path: Path, damage: str) -> None:
    """Require the full tool interaction and release, not just a registration."""
    trace = tmp_path / "gateway-events.jsonl"
    events = [{"type": "session.registered", "session_id": "s",
               "payload": {"policy_version": 0, "policy_fingerprint": "p"}}]
    for ordinal in range(2):
        messages = [{"role": "user", "content": "question"}]
        if ordinal and damage != "no-tool-feedback":
            messages.append({"role": "tool", "content": "answer"})
        events.append({
            "type": "completion.recorded", "session_id": "s",
            "payload": {"ordinal": ordinal, "request": {"messages": messages},
                        "response": {"prompt_token_ids": [1], "choices": [{
                            "token_ids": [2], "logprobs": {"token_logprobs": [-0.5]},
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


@pytest.mark.parametrize("missing_state", [False, True])
def test_checkpoint_uses_current_hyperparallel_format(tmp_path: Path, missing_state: bool) -> None:
    """Checkpoint evidence requires the shipped bytes/metadata and model files."""
    _evidence(tmp_path)
    log = tmp_path / "phase-1.log"
    log.write_text(log.read_text().splitlines()[0])
    checkpoint = tmp_path / "checkpoints/step_1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint_complete.json").write_text(json.dumps({"step": 1, "world_size": 2}))
    (checkpoint / "extra_state.json").write_text(json.dumps({"global_step": 1}))
    (checkpoint / "_rank0_.safetensors").write_bytes(b"model")
    for rank in range(2):
        local = checkpoint / f"rank_{rank}"
        local.mkdir()
        (local / f"{rank}.metadata").write_bytes(b"metadata")
        if not missing_state:
            (local / f"_rank{rank}_.bytes").write_bytes(b"state")
    case = Case("checkpoint-resume", strategy="direct_reshard", resume=True)
    if missing_state:
        with pytest.raises(AssertionError, match="checkpoint bytes"):
            validate_phase(tmp_path, case, 1)
    else:
        validate_phase(tmp_path, case, 1)


def test_launcher_is_framework_free() -> None:
    """The pytest launcher must remain importable without training backends."""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parent) + os.pathsep + str(ROOT))
    code = ("import sys; import test_rl_st; import _launch; "
            "assert not {'torch','torch_npu','hyper_parallel','mindspore'} & sys.modules.keys()")
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
