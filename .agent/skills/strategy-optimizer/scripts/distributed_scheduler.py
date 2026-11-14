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
#!/usr/bin/env python3
"""
Distributed task scheduler for strategy validation.

Launches single-node torchrun benchmarks with generated Trainer configs.
Importing this module requires only the Python standard library.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

def allocate_port() -> int:
    """Choose an available local rendezvous port."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]



def _run_torchrun(cmd: list, env: dict, log_path: Path, timeout: int) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=Path(__file__).resolve().parents[4],
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    return process.returncode


def _torchrun_command(
    config_path: str, world_size: int, rendezvous: str, validation_steps: int, train_entrypoint: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={rendezvous}",
        f"--nproc_per_node={world_size}",
        "--module",
        train_entrypoint,
        config_path,
        f"--training.train_iters={validation_steps}",
        "--training.train_samples=null",
    ]


def _benchmark_outcome(
    returncode: int, log_path: Path, metrics_path: Path, validation_steps: int, elapsed: float,
) -> dict:
    if returncode == 0:
        print(f"[Scheduler] ✓ Validation completed in {elapsed:.1f}s")

        # Extract metrics from log
        metrics = extract_metrics_from_log(log_path, validation_steps)

        if "error" in metrics:
            return {"success": False, "error": metrics["error"], "metrics": metrics,
                    "log_path": str(log_path), "elapsed_time": elapsed}

        # Save metrics
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        return {
            "success": True,
            "metrics": metrics,
            "log_path": str(log_path),
            "elapsed_time": elapsed,
        }
    else:
        print(f"[Scheduler] ✗ Validation failed with exit code {returncode}")

        # Try to diagnose error
        error_msg = diagnose_error_from_log(log_path)

        return {
            "success": False,
            "error": error_msg,
            "log_path": str(log_path),
            "elapsed_time": elapsed,
        }


def launch_validation_benchmark(
    config_path: str,
    world_size: int,
    output_dir: str,
    master_port: Optional[int] = None,
    master_addr: str = "127.0.0.1",
    validation_steps: int = 50,
    timeout: int = 1800,  # 30 minutes
    train_entrypoint: str = "examples.training_demo.train_text",
) -> Dict[str, Any]:
    """
    Launch validation benchmark using torchrun.

    Args:
        config_path: Path to training config YAML
        world_size: Number of processes (devices)
        output_dir: Output directory for logs and metrics
        master_port: Master port (auto-allocated if None)
        master_addr: Master address
        validation_steps: Number of validation steps
        timeout: Timeout in seconds

    Returns:
        Dict containing:
            - success: bool
            - metrics: Dict (if success)
            - error: str (if failed)
            - log_path: str
    """
    # Prepare output directory
    if world_size < 1 or validation_steps < 1 or timeout <= 0:
        raise ValueError("world_size, validation_steps and timeout must be positive")
    config_path = str(Path(config_path).resolve())
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    log_path = output_path / "validation_run.log"
    metrics_path = output_path / "metrics.json"

    # Allocate port if needed
    if master_port is None:
        master_port = allocate_port()

    cmd = _torchrun_command(
        config_path, world_size, f"{master_addr}:{master_port}", validation_steps, train_entrypoint,
    )
    # Environment setup
    env = os.environ.copy()

    # Launch process
    print(f"[Scheduler] Launching validation benchmark...")
    print(f"[Scheduler] Config: {config_path}")
    print(f"[Scheduler] World size: {world_size}")
    print(f"[Scheduler] Output: {output_dir}")
    print(f"[Scheduler] Master: {master_addr}:{master_port}")

    start_time = time.time()

    try:
        returncode = _run_torchrun(cmd, env, log_path, timeout)

        elapsed = time.time() - start_time

        return _benchmark_outcome(returncode, log_path, metrics_path, validation_steps, elapsed)

    except subprocess.TimeoutExpired:
        print(f"[Scheduler] ✗ Validation timeout after {timeout}s")
        return {
            "success": False,
            "error": f"Timeout after {timeout}s",
            "log_path": str(log_path),
            "elapsed_time": timeout,
        }

    except Exception as e:
        print(f"[Scheduler] ✗ Unexpected error: {e}")
        return {
            "success": False,
            "error": str(e),
            "log_path": str(log_path),
            "elapsed_time": time.time() - start_time,
        }


def _metric_fields(line: str) -> dict:
    fields = {}
    for name, pattern in (
            ("throughput", r"tokens/s=([\d.eE+-]+)"),
            ("time", r"step_time=([\d.eE+-]+)s"),
            ("memory", r"memory/device_max_allocated_gb=([\d.eE+-]+)")):
        match = re.search(pattern, line)
        if match:
            fields[name] = float(match.group(1))
    return fields


def extract_metrics_from_log(log_path: Path, expected_steps: int = 50) -> Dict[str, Any]:
    """
    Extract performance metrics from validation log.

    Args:
        log_path: Path to log file
        expected_steps: Expected number of steps

    Returns:
        Dict with metrics:
            - throughput_tokens_per_sec
            - step_time_ms
            - memory_peak_gb
            - etc.
    """
    if expected_steps < 1:
        raise ValueError("expected_steps must be positive")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    records = {}
    for line in re.split(r"[\r\n]", text):
        fields = _metric_fields(line)
        if len(fields) != 3:
            continue
        if any(not math.isfinite(value) or value <= 0 for value in fields.values()):
            return {"error": "Nonpositive or nonfinite benchmark metrics"}
        rank = re.search(r"\brank[\s:=]*(\d+)", line, re.I)
        if rank and int(rank.group(1)) != 0:
            continue
        step = re.search(r"\bstep[\s:=]+(\d+)\b", line, re.I)
        if step is None:
            step = re.search(r"[|:]\s*(\d+)/\d+\b", line)
        if step:
            records[int(step.group(1))] = fields
    if len(records) < expected_steps:
        return {"error": f"Incomplete benchmark metrics: {len(records)} distinct steps, expected {expected_steps}"}
    ordered = [value for _, value in sorted(records.items())][-expected_steps:]
    warmup = min(10, len(ordered) // 5)
    measured = ordered[warmup:]
    throughput = [row["throughput"] for row in measured]
    times = [row["time"] * 1000 for row in measured]
    return {"throughput_tokens_per_sec": statistics.mean(throughput),
            "throughput_std": statistics.stdev(throughput) if len(throughput) > 1 else 0.0,
            "step_time_ms": statistics.mean(times),
            "step_time_std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
            "memory_peak_gb": max(row["memory"] for row in ordered),
            "total_steps": len(ordered), "measured_steps": len(measured), "warmup_steps": warmup}


def diagnose_error_from_log(log_path: Path) -> str:
    """
    Diagnose error from log file.

    Returns:
        Error message string
    """
    try:
        with open(log_path) as f:
            log_content = f.read()

        # Check for common errors
        if "out of memory" in log_content.lower() or "oom" in log_content.lower():
            return "OOM (Out of Memory)"

        if "fsdp" in log_content.lower() and "no_grad" in log_content.lower():
            return "FSDP Conflict (DP=1 issue)"

        if "address already in use" in log_content.lower():
            return "Port Conflict"

        if "timeout" in log_content.lower():
            return "Communication Timeout"

        if "assertion" in log_content.lower():
            return "Assertion Failed"

        # Extract last error line
        error_lines = [line for line in log_content.split('\n') if 'error' in line.lower()]
        if error_lines:
            return error_lines[-1][:200]  # Last 200 chars

        return "Unknown Error (check log)"

    except Exception as e:
        return f"Failed to read log: {e}"


def launch_batch_validation(
    configs: List[Dict[str, Any]],
    world_size: int,
    output_base_dir: str,
    parallel: bool = False,
    max_parallel: int = 3,
    validation_steps: int = 50,
    timeout: int = 1800,
    train_entrypoint: str = "examples.training_demo.train_text",
) -> List[Dict[str, Any]]:
    """
    Launch multiple validation benchmarks.

    Args:
        configs: List of {config_path, candidate_id}
        world_size: Number of processes per validation
        output_base_dir: Base output directory
        parallel: Run in parallel (requires separate devices)
        max_parallel: Max parallel jobs

    Returns:
        List of validation results
    """
    results = []

    if parallel:
        # Parallel execution (requires multi-node or device partitioning)
        print(f"[Scheduler] Launching {len(configs)} validations in parallel (max {max_parallel})")
        raise ValueError("Parallel benchmark jobs require an explicit device partition; use sequential mode")

    if not parallel:
        # Sequential execution
        print(f"[Scheduler] Launching {len(configs)} validations sequentially")

        for i, config in enumerate(configs, 1):
            print(f"\n[Scheduler] === Validation {i}/{len(configs)}: {config['candidate_id']} ===")

            output_dir = os.path.join(output_base_dir, config['candidate_id'])

            result = launch_validation_benchmark(
                config_path=config['config_path'],
                world_size=world_size,
                output_dir=output_dir,
                validation_steps=validation_steps,
                timeout=timeout,
                train_entrypoint=train_entrypoint,
            )

            result['candidate_id'] = config['candidate_id']
            results.append(result)

            # Print result
            if result['success']:
                metrics = result['metrics']
                print(f"[Scheduler] ✓ {config['candidate_id']}: "
                      f"{metrics['throughput_tokens_per_sec']:.0f} tokens/s, "
                      f"{metrics['memory_peak_gb']:.2f} GB")
            else:
                print(f"[Scheduler] ✗ {config['candidate_id']}: {result['error']}")

    return results


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch distributed validation")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--world-size", type=int, default=8, help="Number of devices")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--steps", type=int, default=50, help="Validation steps")

    args = parser.parse_args()

    result = launch_validation_benchmark(
        config_path=args.config,
        world_size=args.world_size,
        output_dir=args.output_dir,
        validation_steps=args.steps,
    )

    if result['success']:
        print("\n✓ Validation successful")
        print(f"Metrics: {result['metrics']}")
    else:
        print("\n✗ Validation failed")
        print(f"Error: {result['error']}")
        sys.exit(1)
