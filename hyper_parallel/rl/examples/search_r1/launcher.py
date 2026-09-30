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
"""Host-owned Search-R1 training: trusted trainer and isolated sibling agents."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid

# Direct script execution puts search_r1/, not the RL root, on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __package__:
    from .container_execution import IMAGE, docker, execute_episode
else:
    from container_execution import IMAGE, docker, execute_episode


TRAINER_HCCL_ENV = {"HCCL_IF_BASE_PORT": "62800", "HCCL_NPU_SOCKET_PORT_RANGE": "62800-62900"}


def training_command(steps: int, resume: bool = False) -> list[str]:
    """Use the real trainer and search task program, not the frozen comparison worker."""
    command = [
        "python", "-m", "torch.distributed.run", "--nnodes=1", "--node_rank=0",
        "--master_addr=127.0.0.1", "--master_port=29500", "--nproc_per_node=2", "train_rl.py",
        "examples/search_r1/configs/qwen3_4b_search_r1.yaml",
        f"--train.max_steps={steps}", "--data.train_path=/data/train.parquet",
        "--data.test_path=/data/test.parquet", "--agentic.codex.session_root=/results/jobs",
    ]
    if resume:
        command.append("--train.checkpoint.load_path=/resume")
    return command


def verify_training(log: str, steps: int, strict: bool = False) -> None:
    """Require a final synchronized policy version and a real nonzero-gradient update."""
    final = [line for line in log.splitlines() if re.search(rf"\bstep={steps} \|", line)]
    if not final or not any(f"policy/version={steps}" in line and f"train/global_step={steps}" in line
                            for line in final):
        raise RuntimeError("Training did not reach the requested synchronized policy version")
    gradients = re.findall(r"train/gradient_norm=([0-9.eE+-]+)", log)
    updates = re.findall(r"train/optimizer_steps=([0-9]+)", log)
    if not any(float(value) > 0 for value in gradients) or not any(int(value) > 0 for value in updates):
        raise RuntimeError("Training finished without evidence of a nonzero-gradient optimizer update")
    if strict:
        _verify_strict_training(log, final)


def _verify_strict_training(log: str, final: list[str]) -> None:
    """Check every update and final evaluation against strict acceptance rules."""
    updates = [line for line in log.splitlines() if re.search(r"\bstep=\d+ \|", line)
               and "train/global_step=" in line]
    for line in updates:
        exact = re.search(r"training/pre_update_exact_valid=([0-9.eE+-]+)", line)
        mismatch = re.search(r"training/pre_update_mismatch_count=([0-9.eE+-]+)", line)
        if exact is None or float(exact[1]) != 1 or mismatch is None or float(mismatch[1]) != 0:
            raise RuntimeError("Missing or failed strict pre-update probability gate")
    validation = re.search(r"validation/accuracy=([0-9.eE+-]+).*?validation/total=([0-9.eE+-]+)", final[-1])
    if (validation is None or not 0 <= float(validation[1]) <= 1
            or float(validation[2]) <= 0 or "verified checkpoint reload" not in log):
        raise RuntimeError("Missing independent evaluation or checkpoint reload evidence")


def verify_artifacts(output: Path) -> dict:
    """Validate trainable episodes and prove infra failures never reached training."""
    # Keep the host/container broker importable with the standard library only.
    from rl.tool_diagnostics import aggregate_tool_protocol_summaries  # pylint: disable=import-outside-toplevel

    requests = list(output.glob("jobs/*/execution-request.json"))
    if not requests:
        raise RuntimeError("No isolated search episodes were executed")
    result = {"episodes": len(requests), "trainable_episodes": 0, "infrastructure_failures": 0,
              "model_calls": 0, "raw_prefix_mismatches": 0,
              "training_context_mismatches": 0, "citation_repairs": 0, "task_failures": 0,
              "training_episodes": 0, "evaluation_episodes": 0}
    protocol_summaries = []
    summaries_by_phase = {"training": [], "evaluation": [], "infrastructure": []}
    for request in requests:
        artifact = request.parent
        isolation = json.loads((artifact / "isolation.json").read_text())
        outcome = json.loads((artifact / "search-outcome.json").read_text())
        protocol_path = artifact / "tool-error-summary.json"
        if not protocol_path.is_file():
            raise RuntimeError(f"Missing tool protocol summary: {artifact}")
        protocol_summary = json.loads(protocol_path.read_text())
        protocol_summaries.append(protocol_summary)
        if not isolation or not all(value is True for value in isolation.values()):
            raise RuntimeError(f"Isolation failed: {artifact}")

        if outcome.get("trainable", True) is not True:
            summaries_by_phase["infrastructure"].append(protocol_summary)
            if (outcome.get("failure_origin") != "infrastructure"
                    or outcome.get("reward_evaluated") is not False
                    or outcome.get("reward") is not None
                    or outcome.get("training_rows_emitted") != 0):
                raise RuntimeError(f"Invalid untrainable infrastructure outcome: {artifact}")
            result["infrastructure_failures"] += 1
            continue

        audit_path = artifact / "training-audit.json"
        if not audit_path.is_file():
            raise RuntimeError(f"Missing training context audit: {artifact}")
        audit = json.loads(audit_path.read_text())
        phase = outcome.get("phase", "training")
        result[phase + "_episodes"] += 1
        result["trainable_episodes"] += 1
        summaries_by_phase.setdefault(phase, []).append(protocol_summary)
        if audit["training_context_mismatches"] != 0 or audit["model_calls"] < 1:
            raise RuntimeError(f"Training context audit failed: {artifact}")
        result["model_calls"] += audit["model_calls"]
        result["raw_prefix_mismatches"] += audit["raw_prefix_mismatches"]
        result["citation_repairs"] += int(outcome.get("citation_repaired", False))
        result["task_failures"] += int(outcome.get("task_outcome") != "completed")
    protocol = aggregate_tool_protocol_summaries(protocol_summaries)
    protocol["by_phase"] = {
        phase: aggregate_tool_protocol_summaries(items)
        for phase, items in summaries_by_phase.items() if items
    }
    if protocol["infra_reward_contamination_count"] != 0:
        raise RuntimeError("Infrastructure failures contaminated reward or training rows")
    if not protocol["complete"]:
        raise RuntimeError("At least one tool protocol summary is incomplete")
    (output / "tool-error-summary.json").write_text(json.dumps(protocol, indent=2))
    result["tool_protocol"] = protocol
    return result


def validate_data(data: Path, resume: Path | None = None) -> dict:
    """Verify immutable split hashes, and disallow changing data during resume."""
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest.get("split_overlap") != 0:
        raise ValueError("Dataset manifest must report disjoint train/test splits")
    for split in ("train", "test"):
        digest = hashlib.sha256()
        with (data / f"{split}.parquet").open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != manifest.get("sha256", {}).get(split):
            raise ValueError(f"Dataset {split} hash differs from preparation manifest")
    if resume is not None:
        previous = json.loads((resume.resolve().parents[1] / "data-source.json").read_text())
        if previous.get("manifest", {}).get("sha256") != manifest["sha256"]:
            raise ValueError("Resume must use the checkpoint's original train/test data")
    return manifest


def _interrupt_host(signum, frame) -> None:
    """Convert host termination into bounded cleanup of this run's containers."""
    del frame
    signal.signal(signum, signal.SIG_IGN)
    raise KeyboardInterrupt("Host training launcher terminated")


def _parse_training_args():
    """Parse and validate required devices, inputs, and resume checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True, help="Two explicitly reserved NPUs, e.g. 2,3")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--data", type=Path, required=True, help="Prepared HotpotQA dataset directory")
    parser.add_argument("--model", type=Path, default=Path("/home/mwl/ckpt/qwen3-4b"))
    parser.add_argument("--backend-privileged", action="store_true")
    parser.add_argument("--resume", type=Path, help="Completed distributed checkpoint directory")
    args = parser.parse_args()
    devices = args.devices.split(",")
    if len(devices) != 2 or len(set(devices)) != 2 or any(value not in map(str, range(8)) for value in devices):
        raise ValueError("Exactly two distinct Ascend device IDs are required")
    if args.steps < 1:
        raise ValueError("Steps must be positive")
    if args.resume and not (args.resume / "checkpoint_complete.json").is_file():
        raise ValueError("Resume requires a completed checkpoint")
    for path in (args.data / "train.parquet", args.data / "test.parquet", args.model / "config.json"):
        if not path.is_file():
            raise ValueError(f"Required input is missing: {path}")
    manifest = validate_data(args.data, args.resume)
    return args, devices, manifest


def _start_trainer(args, output: Path, devices: list[str]) -> tuple[str, str]:
    """Build and start the isolated trainer container."""
    repo = Path(__file__).resolve().parents[4]
    prefix = "search-r1-train-" + uuid.uuid4().hex[:10]
    worker = prefix + "-worker"
    launch = ["run", "-d", "--name", worker, "--hostname", worker, "--add-host", f"{worker}:127.0.0.1",
              "--network", "none", "--shm-size", "32g"]
    if args.backend_privileged:
        launch.append("--privileged")
    for device in devices:
        launch.extend(("--device", f"/dev/davinci{device}"))
    if args.resume:
        launch.extend(("-v", f"{args.resume.resolve()}:/resume:ro"))
    for device in ("davinci_manager", "devmm_svm", "hisi_hdc"):
        launch.extend(("--device", "/dev/" + device))
    for path in ("/usr/local/Ascend/driver/lib64", "/usr/local/dcmi", "/usr/local/bin/npu-smi",
                 "/usr/local/Ascend/driver/version.info", "/etc/ascend_install.info"):
        launch.extend(("-v", f"{path}:{path}:ro"))
    for source, target, mode in ((repo, "/repo", "rw"), (args.data.resolve(), "/data", "ro"),
                                  (args.model.resolve(), "/models/Qwen3-4B", "ro"), (output, "/results", "rw")):
        launch.extend(("-v", f"{source}:{target}:{mode}"))
    environment = {
        **TRAINER_HCCL_ENV,
        "ASCEND_RT_VISIBLE_DEVICES": args.devices, "HYPER_PARALLEL_PLATFORM": "torch",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_HOST_IP": "127.0.0.1", "GLOO_SOCKET_IFNAME": "lo",
        "HYPER_SEARCH_CONTAINER_BROKER": "1", "PHASE_UID": str(os.getuid()), "PHASE_GID": str(os.getgid()),
    }
    for name, value in environment.items():
        launch.extend(("-e", f"{name}={value}"))
    command = (
        "set -euo pipefail; export PYTHONPATH=/repo/hyper_parallel/rl:/repo:${PYTHONPATH:-}; "
        "python -c 'import socket; print(socket.gethostbyname(socket.gethostname()))'; "
        "git config --global --add safe.directory /repo; "
        "python -m pip install --no-deps --no-build-isolation -e /repo; "
        "bash examples/search_r1/scripts/install_runtime.sh; "
        + shlex.join(training_command(args.steps, resume=args.resume is not None)) + " 2>&1 | tee /results/train.log"
    )
    launch.extend(("-w", "/repo/hyper_parallel/rl", "--entrypoint", "/bin/bash", args.image, "-c", command))
    docker(launch, capture_output=True)
    return worker, prefix


def main() -> None:
    """Launch a two-device trainer and service concurrent episode requests on the host."""
    args, devices, manifest = _parse_training_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "data-source.json").write_text(json.dumps({
        "directory": str(args.data.resolve()), "max_train_samples": None,
        "manifest": manifest,
    }, indent=2))
    (output / "private-canary.txt").write_text("Agent containers must not read this file.\n")
    worker, prefix = _start_trainer(args, output, devices)
    futures = {}
    stop = threading.Event()
    executor = ThreadPoolExecutor(max_workers=8)
    previous_termination = signal.signal(signal.SIGTERM, _interrupt_host)
    try:
        deadline = time.monotonic() + max(3600, args.steps * 1200)
        while True:
            for request in output.glob("jobs/*/execution-request.json"):
                if request not in futures:
                    futures[request] = executor.submit(execute_episode, request, output, args.image,
                                                       prefix + "-" + request.parent.name[:12], stop)
            for future in futures.values():
                if future.done():
                    future.result()
            state = json.loads(docker(["inspect", worker, "--format", "{{json .State}}"], capture_output=True).stdout)
            if not state["Running"]:
                if state["ExitCode"] != 0:
                    raise RuntimeError(f"Trainer exited {state['ExitCode']}; see worker.log/train.log")
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Search-R1 training exceeded its overall deadline")
            time.sleep(1)
        verify_training((output / "train.log").read_text(), args.steps, strict=True)
        checkpoint = output / "checkpoints" / f"step_{args.steps}" / "checkpoint_complete.json"
        if not checkpoint.is_file() or json.loads(checkpoint.read_text()).get("step") != args.steps:
            raise RuntimeError("Final checkpoint completion marker is missing or stale")
        audit = verify_artifacts(output)
        (output / "training-acceptance.json").write_text(json.dumps({
            "steps": args.steps, "optimizer_gate": "passed", **audit,
        }))
    finally:
        stop.set()
        for future in futures.values():
            future.cancel()
        log = subprocess.run(["docker", "logs", worker], capture_output=True, text=True, check=False)
        (output / "worker.log").write_text(log.stdout + log.stderr)
        subprocess.run(["docker", "rm", "-f", worker], capture_output=True, check=False)
        executor.shutdown(wait=True, cancel_futures=True)
        for request in futures:
            name = prefix + "-" + request.parent.name[:12] + "-agent"
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
        signal.signal(signal.SIGTERM, previous_termination)


if __name__ == "__main__":
    main()
