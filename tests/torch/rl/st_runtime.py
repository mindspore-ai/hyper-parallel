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
"""Run isolated system tests through the current production train_rl.py entry point."""

from __future__ import annotations

from dataclasses import asdict
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

from tests.common import rl_st_cases
from tests.torch.rl.st_evidence import validate_phase, validate_sessions

Case = rl_st_cases.Case
CASES = rl_st_cases.CASES
AGENT_INSTRUCTIONS = rl_st_cases.AGENT_INSTRUCTIONS
EXAMPLES = rl_st_cases.EXAMPLES
prepare_config = rl_st_cases.prepare_config

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE_IMAGE = "swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-unified-arm64"
DEFAULT_RESULT_ROOT = ROOT / "hyper_parallel/rl/output"


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
    # CI deploys the test archive separately from the installed HyperParallel wheel.
    mounts = [(ROOT, "/repo", True), (EXAMPLES.parent.parent, "/repo/hyper_parallel", True),
              (model, "/model", True), (data, "/data", True),
              (output, "/results", False)]
    for path in ("/usr/local/dcmi", "/usr/local/Ascend/driver", "/etc/ascend_install.info"):
        mounts.append((Path(path), path, True))
    for host, container, readonly in mounts:
        args += ["-v", f"{host}:{container}" + (":ro" if readonly else "")]
    launch = ["python", "/repo/tests/torch/rl/_launch.py",
              f"/results/phase-{phase}.yaml", str(case.world)]
    return args + ["-w", "/repo", image, "/bin/bash", "-lc",
                   "set -e; unset VLLM_PLUGINS; "
                   "bash /repo/hyper_parallel/rl/docker/install_runtime.sh; "
                   "exec " + shlex.join(launch)]


def resources(case: Case) -> tuple[Path, Path, str, list[int]]:
    """Require explicitly selected assets; required CI never passes by skipping."""
    def missing(reason: str) -> None:
        """Expose unavailable resources as a skip locally or a failure in CI."""
        if os.environ.get("RL_ST_REQUIRED", "0") == "1":
            pytest.fail(reason)
        pytest.skip(reason)

    model_key = "RL_ST_MODEL"
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
    timeout = int(os.environ.get("RL_ST_TIMEOUT", "1800"))
    if timeout <= 0:
        raise ValueError("RL_ST_TIMEOUT must be positive")
    output = Path(os.environ.get("RL_ST_RESULT_ROOT", str(DEFAULT_RESULT_ROOT))) / (
        f"{case.name}-{uuid.uuid4().hex[:10]}"
    )
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
