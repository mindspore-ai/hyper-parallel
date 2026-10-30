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

"""Host-side NPU telemetry and contention guard for one experiment container."""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import time
from pathlib import Path


def cards(output: str) -> dict:
    """Parse per-card HBM and AICore readings from the host tool output."""
    result = {}
    current = None
    for line in output.split("Process id", 1)[0].split("\n"):
        match = re.match(r"\|\s*(\d+)\s+910B", line)
        if match:
            current = int(match[1])
        parts = line.split("|")
        if current is not None and len(parts) > 3 and re.search(r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:", parts[2]):
            numbers = [int(x) for x in re.findall(r"\d+", parts[3])]
            if len(numbers) == 5:
                result[str(current)] = {"aicore": numbers[0], "hbm_mb": numbers[3], "capacity_mb": numbers[4]}
    return result


def _process_stat(pid: str) -> tuple[str, str]:
    """Read state and start time without treating spaces or parentheses in comm as fields."""
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    return fields[0], fields[19]


def process_identity(pid: str, identity: str) -> dict:
    """Classify a live process conservatively, excluding exited tasks still listed by the driver.

    Args:
        pid: Host process identifier returned by the device tool.
        identity: Full container identity whose live workers belong to the experiment.

    Returns:
        Ownership and process lifetime evidence. PID reuse and unreadable live state remain unknown.
    """
    try:
        state, started = _process_stat(pid)
        if state in ("Z", "X"):
            return {"owner": "exited", "state": state, "start_ticks": started}
        cgroup = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8")
        final_state, final_started = _process_stat(pid)
        if started != final_started:
            return {"owner": "unknown", "reason": "pid_reused"}
        if final_state in ("Z", "X"):
            return {"owner": "exited", "state": final_state, "start_ticks": started}
        owner = "owned" if identity and identity in cgroup else "external"
        return {"owner": owner, "state": final_state, "start_ticks": started}
    except FileNotFoundError:
        return {"owner": "gone"}
    except (OSError, IndexError, ValueError) as error:
        return {"owner": "unknown", "reason": type(error).__name__}


def sample(identity: str = "") -> dict:
    """Treat query errors and unreadable live PIDs as unavailable hardware."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=15, check=False)
        output, code = result.stdout, result.returncode
    except (OSError, subprocess.SubprocessError) as error:
        return {"utc": now, "npu_smi": "", "npu_smi_exit_code": 124, "foreign_compute": {}, "error": str(error)}
    owners = {}
    for line in output.split("Process id", 1)[-1].splitlines():
        parts = line.split("|")
        if len(parts) <= 3 or not parts[2].strip().isdigit():
            continue
        pid = parts[2].strip()
        owners[pid] = process_identity(pid, identity)
    return {
        "utc": now,
        "npu_smi": output,
        "npu_smi_exit_code": code,
        "processes": owners,
        "foreign_compute": {
            pid: details["owner"] for pid, details in owners.items()
            if details["owner"] not in ("owned", "exited", "gone")
        },
    }


def idle(row: dict, min_free_gib: int = 27) -> bool:
    """Require eight readable cards, no external compute PIDs and sufficient free memory per card."""
    devices = cards(row["npu_smi"])
    return (
        row["npu_smi_exit_code"] == 0
        and not row["foreign_compute"]
        and len(devices) == 8
        and all(value["capacity_mb"] - value["hbm_mb"] >= min_free_gib * 1024 for value in devices.values())
    )


def read_failure(directory: Path) -> dict | None:
    """Read atomically published rank failures from the current training run."""
    for path in sorted(directory.glob("failure-rank-*.json")):
        return {"source": path.name, "record": json.loads(path.read_text(encoding="utf-8"))}
    return None


def stop_container(identity: str) -> None:
    """Stop exactly the recorded container, escalate to kill, and verify its exit.

    Docker operations have individual deadlines. Failure to confirm resource
    release raises an error instead of allowing the next experiment to start.
    """
    if re.fullmatch(r"[0-9a-f]{12,64}", identity) is None:
        raise ValueError("Expected the owned Docker container ID")
    try:
        subprocess.run(["docker", "stop", "-t", "10", identity], check=False, timeout=20, capture_output=True)
    except subprocess.TimeoutExpired:
        pass
    if _is_running(identity):
        subprocess.run(["docker", "kill", identity], check=True, timeout=10, capture_output=True)
    if _is_running(identity):
        raise RuntimeError(f"Owned container did not stop: {identity}")


def _is_running(identity: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", identity],
        check=True, timeout=10, capture_output=True, text=True,
    )
    state = result.stdout.strip()
    if state not in ("true", "false"):
        raise RuntimeError(f"Unexpected Docker running state: {state!r}")
    return state == "true"


def main() -> None:
    """Stop only the supplied experiment container when monitoring cannot ensure an uncontended run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    args = parser.parse_args()
    while not (args.case_dir / "stop_monitor").exists():
        row = sample(args.container_id)
        with (args.results / "device-monitor.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        reason = None
        if (args.results / "PAUSE").exists():
            reason = "user_pause"
        elif row["npu_smi_exit_code"] or row["foreign_compute"] or len(cards(row["npu_smi"])) != 8:
            reason = "external_occupancy_or_monitor_failure"
        failure = read_failure(args.case_dir)
        if failure is not None:
            reason = "training_failure"
            row["failure"] = failure
        if reason:
            (args.case_dir / "interruption.json").write_text(json.dumps({"reason": reason, "sample": row}, indent=2))
            stop_container(args.container_id)
            return
        time.sleep(1)


if __name__ == "__main__":
    main()
