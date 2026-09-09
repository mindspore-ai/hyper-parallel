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
"""Run isolated system tests through PR1354's production train_rl.py entry point."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import time
import uuid

import pytest
import yaml

from st_evidence import validate_phase, validate_sessions

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
EXAMPLES = ROOT / "hyper_parallel/rl/examples"
BASE_IMAGE = "swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-arm64"


@dataclass(frozen=True)
class Case:
    """Describe a production recipe and its required hardware."""
    name: str
    tp: int = 1
    strategy: str = "full_gather"
    family: str = "qwen3"
    runner: str = "internal"
    disjoint: bool = False
    resume: bool = False
    exact: bool = False

    @property
    def world(self) -> int:
        """Trainer uses two FSDP shards and the requested TP degree."""
        return 2 * self.tp

    @property
    def cards(self) -> int:
        """Disjoint rollout consumes a separate device set."""
        return self.world * (2 if self.disjoint else 1)


CASES = (
    Case("dense-tp1-full", exact=True),
    Case("dense-tp2-direct", tp=2, strategy="direct_reshard", exact=True),
    Case("dense-disjoint-direct", tp=2, strategy="direct_reshard", disjoint=True),
    Case("checkpoint-resume", resume=True),
    Case("qwen3-moe-ep4", tp=2, strategy="direct_reshard", family="qwen3_moe"),
    Case("moonlight-ep4", tp=2, strategy="direct_reshard", family="deepseek_v3"),
    Case("codex-agent", runner="codex"),
    Case("deepseek-agent", runner="deepseek"),
)

AGENT_INSTRUCTIONS = {
    "codex": """/no_think
Solve the arithmetic word problem below.
You must first use the local shell tool exactly once to run Python for the calculation.
Do not return a final answer before observing the command output.
After observing the first command output, never call any tool again, even to retry or verify it.
Return only the final answer in the form "#### NUMBER" without an explanation.

Problem: {prompt}""",
    "deepseek": """Solve the arithmetic word problem below.
Immediately call the Bash tool exactly once and use Python for the calculation.
Run Bash in the foreground: omit run_in_background or set it to false.
Do not write analysis before the tool call.
Never use job_output, job_list, or job_kill.
After the Bash result, never call any tool again and return exactly "#### NUMBER".

Problem: {prompt}""",
}


def prepare_config(case: Case, phase: int, ports: tuple[int, int], devices: list[int]) -> dict:
    """Derive a bounded ST workload from the shipped production YAML."""
    if case.runner != "internal":
        recipe = EXAMPLES / f"agents/gsm8k/configs/{case.runner}_multi_turn.yaml"
    elif case.family != "qwen3":
        recipe = EXAMPLES / "configs/moonlight_16b_a3b_gsm8k_native_vllm.yaml"
    else:
        recipe = EXAMPLES / "configs/qwen3_4b_gsm8k_vllm_production.yaml"
    config = yaml.safe_load(recipe.read_text())
    config["model"].update(weights_path="/model", tokenizer_path="/model")
    if case.family == "qwen3_moe":
        config["model"].update(registry_name="qwen3_30b_a3b", name="qwen3_moe",
                               attention_implementation=None)
    config["consistency"]["enabled"] = case.exact
    is_moe = case.family != "qwen3"
    config["data"].update(train_path="/data/train.parquet", test_path="/data/test.parquet",
                          max_train_samples=4 if is_moe else 8, shuffle=False, num_workers=0)
    # Dense Qwen3 needs room to finish reasoning before the answer is scored.
    generation_budget = 48 if is_moe else (512 if case.runner == "internal" else 256)
    config["rollout"].update(num_return_sequences=4, max_new_tokens=generation_budget, seed=20260908)
    if is_moe:
        config["rollout"].update(temperature=0.0, top_p=1.0, top_k=0)
    vllm = config["rollout"]["vllm"]
    vllm.update(deployment="disjoint" if case.disjoint else "colocated",
                data_parallel_size=2, tensor_parallel_size=case.tp, port=ports[0],
                model_implementation="native" if case.runner != "internal" else "hyper",
                enable_expert_parallel=case.family != "qwen3", enforce_eager=True,
                max_num_seqs=4)
    vllm["weight_sync"].update(strategy=case.strategy, fallback_strategy="none", bucket_size_mb=128)
    if case.disjoint:
        vllm["visible_devices"] = ",".join(map(str, devices[case.world:]))
    train = config["train"]
    train.update(max_steps=1 if case.resume and phase == 1 else 2,
                 prompt_batch_size=1 if is_moe else 2, micro_batch_size=4 if is_moe else 2,
                 response_mini_batch_size=4 if is_moe else 8)
    train["accelerator"].update(dp_replicate=1, dp_shard=2, tp=case.tp,
                                 ep=4 if case.family != "qwen3" else 1,
                                 cpu_offload=True, activation_checkpoint="full")
    # Real rewards are preserved. Zero-advantage batches may occur; the validator
    # requires evidence of at least one actual update across the completed run.
    train["learning_gate"]["enabled"] = False
    if is_moe:
        config["agentic"].update(
            module_path="/repo/hyper_parallel/rl/rl_tests/st/_moe_control.py",
            environment="moe_update_control",
            interaction_mode="single_turn",
            max_turns=1,
            max_observation_tokens=0,
            apply_chat_template=True,
        )
    # Production evaluation is scheduled at checkpoint boundaries.
    train["checkpoint"].update(output_dir="/results/checkpoints", save_steps=0,
                                save_final=case.resume or case.name == "dense-tp1-full", verify_reload=False,
                                load_path="/results/checkpoints/step_1" if phase == 2 else None)
    config["evaluation"].update(enabled=case.name == "dense-tp1-full",
                                 batch_size=1, max_samples=2, max_new_tokens=64)
    config["logging"].update(backends=["console"], log_steps=1)
    config["logging"]["wandb"]["mode"] = "disabled"
    if case.runner != "internal":
        config["agentic"][case.runner].update(
            gateway_port=ports[1], session_root="/results/sessions",
            instruction_template=AGENT_INSTRUCTIONS[case.runner])
    return config


def available_port() -> int:
    """Obtain an unused host port; cases run serially to limit allocation races."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def command(case: Case, model: Path, data: Path, output: Path, image: str,
            devices: list[int], phase: int, name: str, master_port: int | None = None) -> list[str]:
    """Use a fresh container for each phase, including checkpoint resume."""
    env = {
        "ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, devices[:case.world])),
        "HYPER_PARALLEL_PLATFORM": "torch",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_ADDOPTS": "-o log_cli=true -o log_cli_level=INFO -p no:cacheprovider",
        "PYTHONPATH": "/repo/hyper_parallel/rl:/repo",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_HOST_IP": "127.0.0.1",
        "GLOO_SOCKET_IFNAME": "lo",
        "HYPER_RL_WEIGHT_ORACLE_RUN_ID": output.name,
        "HYPER_RL_WEIGHT_MANIFEST_DIR": "/results/manifests",
        "HYPER_RL_ROLLOUT_ARTIFACT_DIR": "/results/rollouts",
        "HYPER_RL_ROLLOUT_ARTIFACT_STRATEGY": case.strategy,
    }
    hccl_port = int(os.environ.get("RL_ST_HCCL_BASE_PORT", "62000"))
    if not 1024 <= hccl_port <= 65436:
        raise ValueError("RL_ST_HCCL_BASE_PORT must leave room for a 100-port range")
    env.update(HCCL_IF_BASE_PORT=str(hccl_port),
               HCCL_NPU_SOCKET_PORT_RANGE=f"{hccl_port}-{hccl_port + 99}")
    env["RL_ST_MASTER_PORT"] = str(master_port if master_port is not None else hccl_port - 1)
    args = ["docker", "run", "--rm", "--name", name, "--privileged",
            "--network=host", "--shm-size=64g"]
    for key, value in env.items():
        args += ["-e", f"{key}={value}"]
    mounts = [(ROOT, "/repo", True), (model, "/model", True), (data, "/data", True),
              (output, "/results", False)]
    for path in ("/usr/local/dcmi", "/usr/local/Ascend/driver", "/etc/ascend_install.info"):
        mounts.append((Path(path), path, True))
    for host, container, readonly in mounts:
        args += ["-v", f"{host}:{container}" + (":ro" if readonly else "")]
    launch = ["python", "/repo/hyper_parallel/rl/rl_tests/st/_launch.py",
              f"/results/phase-{phase}.yaml", str(case.world)]
    return args + ["-w", "/repo", image, "/bin/bash", "-lc",
                   "unset VLLM_PLUGINS; exec " + shlex.join(launch)]


def resources(case: Case) -> tuple[Path, Path, str, list[int]]:
    """Require explicitly selected assets; required CI never passes by skipping."""
    def missing(reason: str) -> None:
        """Expose unavailable resources as a skip locally or a failure in CI."""
        if os.environ.get("RL_ST_REQUIRED", "0") == "1":
            pytest.fail(reason)
        pytest.skip(reason)

    model_key = {"qwen3": "RL_ST_MODEL", "qwen3_moe": "RL_ST_MOE_MODEL",
                 "deepseek_v3": "RL_ST_MOONLIGHT_MODEL"}[case.family]
    for key in (model_key, "RL_ST_DATA", "RL_ST_DEVICES"):
        if not os.environ.get(key):
            missing(f"Set {key} to run {case.name}")
    model, data = Path(os.environ[model_key]).resolve(), Path(os.environ["RL_ST_DATA"]).resolve()
    for path in (model / "config.json", data / "train.parquet", data / "test.parquet"):
        if not path.is_file():
            missing(f"Required asset missing: {path}")
    identity = json.loads((model / "config.json").read_text())
    if identity.get("model_type") != case.family:
        pytest.fail(f"{model_key} has model_type={identity.get('model_type')}, expected {case.family}")
    devices = [int(value) for value in os.environ["RL_ST_DEVICES"].split(",")]
    if len(set(devices)) != len(devices) or any(value < 0 for value in devices):
        pytest.fail("RL_ST_DEVICES must contain unique non-negative physical IDs")
    if len(devices) < case.cards:
        missing(f"{case.name} needs {case.cards} devices")
    image = os.environ.get(f"RL_ST_{case.runner.upper()}_IMAGE", BASE_IMAGE)
    if case.runner != "internal" and f"RL_ST_{case.runner.upper()}_IMAGE" not in os.environ:
        missing(f"Set RL_ST_{case.runner.upper()}_IMAGE to the matching Agent runtime image")
    if not shutil.which("docker"):
        missing("Docker is required")
    for driver in ("/usr/local/dcmi", "/usr/local/Ascend/driver", "/etc/ascend_install.info"):
        if not Path(driver).exists():
            missing(f"Missing Ascend driver mount: {driver}")
    probe = subprocess.run(["docker", "image", "inspect", image],
                           capture_output=True, timeout=20, check=False)
    if probe.returncode:
        missing(f"Local image unavailable: {image}; {probe.stderr.decode(errors='replace')}")
    return model, data, image, devices[:case.cards]


def execute(args: list[str], output: Path, phase: int, name: str, timeout: int) -> None:
    """Bound a child process and clean up only its uniquely named ST container."""
    with (output / f"phase-{phase}.log").open("w") as log, subprocess.Popen(
        args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT
    ) as process:
        try:
            status = process.wait(timeout=timeout)
            if status:
                raise AssertionError(f"Training exited {status}; see {log.name}")
        finally:
            # Docker removes the container on success; this also handles timeout.
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30, check=False)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            remaining = subprocess.run(["docker", "ps", "-aq", "--filter", f"name=^{name}$"],
                                       capture_output=True, timeout=20, check=True)
            if remaining.stdout.strip():
                raise AssertionError(f"ST container cleanup incomplete: {name}")


def run_case(case: Case) -> Path:
    """Run real training and record a verdict only after checking its evidence."""
    model, data, image, devices = resources(case)
    default_timeout = "3600" if case.family != "qwen3" else "1800"
    timeout = int(os.environ.get("RL_ST_TIMEOUT", default_timeout))
    if timeout <= 0:
        raise ValueError("RL_ST_TIMEOUT must be positive")
    output = HERE / "results" / f"{case.name}-{uuid.uuid4().hex[:10]}"
    output.mkdir(parents=True)
    report = {"case": asdict(case), "status": "running", "devices": devices,
              "image": image, "model": str(model), "data": str(data), "phases": []}
    started = time.monotonic()
    try:
        for phase in range(1, 3 if case.resume else 2):
            ports = []
            while len(ports) < 3:
                port = available_port()
                if port not in ports:
                    ports.append(port)
            config = prepare_config(case, phase, (ports[0], ports[1]), devices)
            (output / f"phase-{phase}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            name = f"rl-st-{uuid.uuid4().hex[:12]}"
            args = command(case, model, data, output, image, devices, phase, name, master_port=ports[2])
            report["phases"].append({"command": args, "ports": ports})
            (output / "result.json").write_text(json.dumps(report, indent=2))
            execute(args, output, phase, name, timeout)
            validate_phase(output, case, phase)
            for port in ports:
                with socket.socket() as listener:
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    listener.bind(("127.0.0.1", port))
        if case.runner != "internal":
            validate_sessions(output / "sessions", (0, 1))
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(report, indent=2))
    return output
