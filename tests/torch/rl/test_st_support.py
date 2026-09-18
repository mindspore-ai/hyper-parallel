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

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from tests.torch.rl import _launch as launch_module
from tests.torch.rl.st_evidence import metrics, validate_phase, validate_sessions
from tests.torch.rl.st_runtime import CASES, DEFAULT_RESULT_ROOT, ROOT, Case, prepare_config, command


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_recipe_matches_case(case: Case, tmp_path: Path) -> None:
    """Generated workloads preserve the required topology and production entry."""
    devices = list(range(case.cards))
    config = prepare_config(case, 1, (8100, 8200), devices)
    assert config["algorithm"]["name"] == case.algorithm
    assert config["train"]["accelerator"]["tp"] == case.tp
    assert config["rollout"]["vllm"]["weight_sync"]["strategy"] == case.strategy
    assert config["model"]["weights_path"] == "/model"
    assert config["consistency"]["enabled"] is case.exact
    assert config["rollout"]["vllm"]["batch_invariant"] is case.exact
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
        assert resumed["train"]["checkpoint"]["load_path"] == "/results/checkpoints/step_1"
        assert resumed["train"]["max_steps"] == (3 if case.algorithm == "ppo" else 2)


def test_default_results_use_rl_output() -> None:
    """All ST artifacts default to the single ignored RL output directory."""
    assert DEFAULT_RESULT_ROOT == ROOT / "hyper_parallel/rl/output"


def test_selected_acceptance_matrix() -> None:
    """Run only the approved six real-NPU acceptance scenarios."""
    assert tuple(case.name for case in CASES) == (
        "dense-tp2-consistency-full",
        "dense-tp2-direct",
        "checkpoint-resume",
        "codex-agent",
        "deepseek-agent",
        "ppo-tp1-full",
    )
    exact = next(case for case in CASES if case.exact)
    config = prepare_config(exact, 1, (8100, 8200), list(range(exact.cards)))
    vllm = config["rollout"]["vllm"]
    assert exact.tp == 2 and exact.algorithm == "grpo"
    assert vllm["model_implementation"] == "hyper"
    assert vllm["enable_prefix_caching"] and vllm["enable_chunked_prefill"]
    assert vllm["logprobs_mode"] == "raw_logprobs"


def _evidence(output: Path, case: Case | None = None, phase: int = 1) -> Case:
    """Write one complete synthetic acceptance record, never a real ST result."""
    case = case or Case("synthetic", strategy="direct_reshard")
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
               "training/pre_update_exact_valid": 1, "training/pre_update_exact_tokens": 2,
               "training/pre_update_mismatch_count": 0, "training/pre_update_max_abs_diff": 0,
               "training/pre_update_mean_abs_diff": 0,
               "training/post_update_old_policy_tokens": 2,
               "training/post_update_old_policy_mismatch_count": 1,
               "training/post_update_negative_control_valid": 1,
               "weight_sync/streaming_bucket_count": 2, "weight_sync/streaming_acked_buckets": 2,
               "weight_sync/streaming_released_buckets": 2, "weight_sync/streaming_max_inflight_buckets": 1,
               "weight_sync/streaming_max_gathered_bytes": 128, "weight_sync/streaming_max_packed_bytes": 128}
        lines.append(f"INFO | step={step} | " + ", ".join(f"{key}={value}" for key, value in row.items()))
    (output / f"phase-{phase}.log").write_text("\n".join(lines))
    return case


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
                "weight_sync/last_direct_reshard=1",
                "weight_sync/last_direct_reshard=0",
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
@pytest.mark.parametrize("scenario", ["checkpoint-resume", "ppo-checkpoint-resume", "dense-tp1-full"])
def test_checkpoint_uses_current_hyperparallel_format(tmp_path: Path, damage: str, scenario: str) -> None:
    """Checkpoint evidence requires resumable state and a complete final HF export."""
    case = Case(scenario, strategy="direct_reshard", resume=scenario != "dense-tp1-full",
                algorithm="ppo" if scenario.startswith("ppo") else "grpo")
    _evidence(tmp_path, case)
    step = 1 if case.resume else 2
    if not case.resume:
        with (tmp_path / "phase-1.log").open("a") as log:
            log.write("\nINFO | step=2 | validation/total=8, validation/correct=2, "
                      "validation/accuracy=0.25, validation/generated_tokens=32")
    checkpoint = tmp_path / f"checkpoints/step_{step}"
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


@pytest.mark.parametrize(
    ("case", "field", "value", "error"),
    [
        (Case("exact", strategy="direct_reshard", exact=True), "training/pre_update_exact_valid", 0, "successful"),
        (Case("exact", strategy="direct_reshard", exact=True), "training/pre_update_exact_tokens", 0, "Empty"),
        (Case("exact", strategy="direct_reshard", exact=True), "training/pre_update_mismatch_count", 1, "Consistency"),
        (Case("exact", strategy="direct_reshard", exact=True), "training/pre_update_max_abs_diff", 0.01, "Consistency"),
        (Case("exact", strategy="direct_reshard", exact=True),
         "training/pre_update_mean_abs_diff", 0.01, "Consistency"),
        (Case("exact", strategy="direct_reshard", exact=True), "training/post_update_old_policy_tokens", 0, "Invalid"),
        (Case("exact", strategy="direct_reshard", exact=True),
         "training/post_update_old_policy_mismatch_count", 3, "Invalid"),
        (Case("exact", strategy="direct_reshard", exact=True),
         "training/post_update_negative_control_valid", 0, "Inconsistent"),
        (Case("ppo", strategy="direct_reshard", algorithm="ppo"), "critic/gradient_norm", 0, "Critic update"),
        (Case("ppo", strategy="direct_reshard", algorithm="ppo"), "critic/optimizer_steps", 0, "Critic optimization"),
        (Case("ppo", strategy="direct_reshard", algorithm="ppo"), "critic/valid_tokens", 0, "Critic optimization"),
        (Case("full"), "weight_sync/streaming_acked_buckets", 1, "Unacknowledged"),
        (Case("full"), "weight_sync/streaming_released_buckets", 1, "Unacknowledged"),
        (Case("full"), "weight_sync/streaming_max_inflight_buckets", 2, "Unbounded"),
        (Case("full"), "weight_sync/streaming_max_packed_bytes", 64, "Invalid packed"),
        (Case("full"), "train/total_loss", float("nan"), "Non-finite"),
        (Case("full"), "train/valid_tokens", 0, "No train/valid_tokens"),
    ],
)
def test_feature_evidence_rejects_false_pass(
    tmp_path: Path, case: Case, field: str, value: float, error: str,
) -> None:
    """Damage one otherwise passing record to exercise each specialized acceptance gate."""
    _evidence(tmp_path, case)
    validate_phase(tmp_path, case, 1)
    log = tmp_path / "phase-1.log"
    with log.open("a") as stream:
        for step in (1, 2):
            stream.write(f"\nINFO | step={step} | {field}={value}")
    with pytest.raises(AssertionError, match=error):
        validate_phase(tmp_path, case, 1)


def test_consistency_requires_a_real_post_update_negative_control(tmp_path: Path) -> None:
    """A self-consistent zero-change diagnostic does not prove policy learning."""
    case = Case("exact", strategy="direct_reshard", exact=True)
    _evidence(tmp_path, case)
    log = tmp_path / "phase-1.log"
    log.write_text(log.read_text().replace("post_update_old_policy_mismatch_count=1",
                                          "post_update_old_policy_mismatch_count=0")
                   .replace("post_update_negative_control_valid=1", "post_update_negative_control_valid=0"))
    with pytest.raises(AssertionError, match="No post-update change"):
        validate_phase(tmp_path, case, 1)


@pytest.mark.parametrize("phase", [1, 2])
def test_ppo_resume_rejects_actor_only_checkpoint(tmp_path: Path, phase: int) -> None:
    """Fresh-container resume requires an explicit Critic completion marker in both phases."""
    case = Case("ppo-checkpoint-resume", strategy="direct_reshard", algorithm="ppo", resume=True)
    _evidence(tmp_path, case, phase)
    step = 1 if phase == 1 else 3
    checkpoint = tmp_path / f"checkpoints/step_{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint_complete.json").write_text(json.dumps({"step": step, "world_size": 2}))
    with pytest.raises(AssertionError, match="does not include Critic"):
        validate_phase(tmp_path, case, phase)


@pytest.mark.parametrize(
    "evaluation",
    [
        "validation/total=2, validation/correct=1, validation/accuracy=0.5, validation/generated_tokens=32",
        "validation/total=8, validation/correct=2, validation/accuracy=0.5, validation/generated_tokens=32",
        "validation/total=8, validation/correct=2, validation/accuracy=0.25, validation/generated_tokens=0",
    ],
)
def test_evaluation_rejects_partial_or_inconsistent_results(tmp_path: Path, evaluation: str) -> None:
    """Only the full configured evaluation at the final policy version can pass."""
    case = Case("dense-tp1-full")
    _evidence(tmp_path, case)
    with (tmp_path / "phase-1.log").open("a") as log:
        log.write(f"\nINFO | step=2 | {evaluation}")
    with pytest.raises(AssertionError, match="evaluation|Evaluation"):
        validate_phase(tmp_path, case, 1)


def test_launcher_is_framework_free() -> None:
    """The pytest launcher must remain importable without training backends."""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parent) + os.pathsep + str(ROOT))
    code = ("import sys; import test_rl_st; import _launch; "
            "assert not {'torch','torch_npu','hyper_parallel','mindspore'} & sys.modules.keys()")
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


def test_launcher_preserves_rank_logs_after_sigkill(tmp_path: Path) -> None:
    """A killed launcher leaves raw worker evidence on the mounted result directory."""
    config = tmp_path / "phase-1.yaml"
    config.write_text("config")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parent) + os.pathsep + str(ROOT),
               RL_ST_MASTER_PORT="29500")
    code = (
        "import os, signal, sys\n"
        "from pathlib import Path\n"
        "import _launch\n"
        "def stop_rank(*args, **kwargs):\n"
        "    Path('rank.log').write_text('last worker evidence')\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n"
        "_launch.torchrun_case = stop_rank\n"
        "sys.argv = [_launch.__file__, sys.argv[1], '2']\n"
        "_launch.main()\n"
    )
    result = subprocess.run([sys.executable, "-c", code, str(config)], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == -signal.SIGKILL, (
        f"Expected isolated child exit={-signal.SIGKILL}, got={result.returncode}; stderr={result.stderr}"
    )
    logs = list(tmp_path.glob("rl-st-phase-1-*/rank.log"))
    assert len(logs) == 1, f"Expected one surviving rank log, got={logs}"
    assert logs[0].read_text() == "last worker evidence", f"Missing interrupted worker evidence in={logs[0]}"


@pytest.mark.parametrize("failed", [False, True])
def test_launcher_removes_temporary_rank_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failed: bool,
) -> None:
    """Rank logs are streamed and their temporary directory is removed."""
    config = tmp_path / "phase-1.yaml"
    config.write_text("config")
    rank_directories = []

    def run_rank(*_args: object, **_kwargs: object) -> None:
        """Write the same log artifact shape as the distributed launcher."""
        rank_directory = Path.cwd()
        rank_directories.append(rank_directory)
        (rank_directory / "rank.log").write_text("rank output\n")
        if failed:
            raise RuntimeError("worker failed")

    original_cwd = Path.cwd()
    monkeypatch.setattr(launch_module, "torchrun_case", run_rank)
    monkeypatch.setattr(launch_module.sys, "argv", ["_launch.py", str(config), "2"])
    monkeypatch.setenv("RL_ST_MASTER_PORT", "29500")

    if failed:
        with pytest.raises(RuntimeError, match="worker failed"):
            launch_module.main()
    else:
        launch_module.main()

    assert Path.cwd() == original_cwd
    assert rank_directories and not rank_directories[0].exists()
    assert rank_directories[0].parent == config.parent, (
        f"Interrupted rank logs must live on the result mount={config.parent}, got={rank_directories[0]}"
    )
    assert not config.with_suffix("").exists()
    assert "rank output" in capsys.readouterr().out
