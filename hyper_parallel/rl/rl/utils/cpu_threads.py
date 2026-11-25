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
"""Resolve CPU thread budgets before torchrun, without importing Torch or the device runtime."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import posixpath
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
import re

from rl.config.loader import load_config

logger = logging.getLogger(__name__)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as error:
        logger.warning("Cannot inspect CPU limit %s: %s", path, error)
        return None


def _unescape_mount_path(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _cgroup_directories(proc_root: Path) -> list[tuple[Path, Path, bool]]:
    """Locate this process's CPU controller in host or private cgroup namespaces."""
    membership = {}
    for line in (_read_text(proc_root / "self/cgroup") or "").splitlines():
        _, controllers, path = line.split(":", 2)
        for controller in controllers.split(","):
            membership[controller] = PurePosixPath(posixpath.normpath(path))

    directories = []
    for line in (_read_text(proc_root / "self/mountinfo") or "").splitlines():
        before, after = line.split(" - ", 1)
        fields, filesystem = before.split(), after.split()
        unified = filesystem[0] == "cgroup2"
        if not unified and not (filesystem[0] == "cgroup" and "cpu" in filesystem[2].split(",")):
            continue
        member = membership.get("" if unified else "cpu")
        if member is None:
            continue
        mount = Path(_unescape_mount_path(fields[4]))
        root = PurePosixPath(_unescape_mount_path(fields[3]))
        if member == PurePosixPath("/"):
            # A private namespace can expose '/' as membership but retain the
            # host cgroup path as the mount root. Its quota lives at the mount.
            relative = PurePosixPath(".")
        else:
            try:
                relative = member.relative_to(root)
            except ValueError:
                continue
        directories.append((mount / relative, mount, unified))
    return directories


def _cpu_quota(proc_root: Path) -> float | None:
    """Use the tightest visible ancestor quota, not just the leaf's cpu.max."""
    limits = []
    for current, mount, unified in _cgroup_directories(proc_root):
        while True:
            if unified:
                value = _read_text(current / "cpu.max")
                if value is not None:
                    quota, period = value.split()
                    if quota != "max":
                        limits.append(int(quota) / int(period))
            else:
                quota = _read_text(current / "cpu.cfs_quota_us")
                period = _read_text(current / "cpu.cfs_period_us")
                if quota is not None and period is not None and int(quota) > 0:
                    limits.append(int(quota) / int(period))
            if current == mount:
                break
            current = current.parent
    return min(limits) if limits else None


def detect_cpu_resources(*, proc_root: Path = Path("/proc"),
                         cpu_root: Path = Path("/sys/devices/system/cpu")) -> dict:
    """Inspect affinity/cpuset, SMT topology and visible cgroup v1/v2 quotas.

    Run in the container's launcher before spawning workers. The affinity mask
    already reflects the kernel's cpuset restriction; these are allocated
    resources, not an estimate of momentarily idle CPUs on a shared host.
    """
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    siblings = [_read_text(cpu_root / f"cpu{cpu}/topology/thread_siblings_list") for cpu in cpus]
    physical = len(set(siblings)) if all(siblings) else len(cpus)
    quota = _cpu_quota(proc_root)
    available = min(physical, math.floor(quota)) if quota is not None else physical
    return {"affinity_cpus": len(cpus), "physical_cores": physical, "topology_available": all(siblings),
            "quota_cpus": quota, "available_cores": max(1, available)}


def _positive_threads(value: str, name: str) -> int:
    if not value.isdigit() or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def resolve_cpu_threads(resources: Mapping, local_world_size: int, *, cpu_offload: bool,
                        environment: Mapping[str, str]) -> dict:
    """Divide the launcher's shared CPU budget once across local training ranks.

    Explicit MKL/OMP values win (MKL has precedence, as in PyTorch). Otherwise
    CPU offload uses at most half the available cores, capped at eight threads
    per rank (and at least one). Device-only training keeps one thread. This
    does not bind workers to cores or tune NUMA.
    """
    if isinstance(local_world_size, bool) or not isinstance(local_world_size, int) or local_world_size < 1:
        raise ValueError("local_world_size must be a positive integer")
    cores = int(resources["available_cores"])
    reserve_value = environment.get("HYPER_RL_CPU_RESERVE", "auto")
    if reserve_value in ("", "auto"):
        reserved = math.ceil(cores / 2)
    elif reserve_value.isdigit():
        # A larger reserve can tighten the budget, but never bypass the 50% limit.
        reserved = max(math.ceil(cores / 2), int(reserve_value))
    else:
        raise ValueError("HYPER_RL_CPU_RESERVE must be auto or a nonnegative integer")

    # Explicit values must be resolved before torchrun inserts its own default
    # OMP_NUM_THREADS=1; after launch that default cannot be distinguished from
    # a user's deliberate request for a single thread.
    requested = {}
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = environment.get(name, "auto")
        if value not in ("", "auto"):
            requested[name] = _positive_threads(value, name)
    source = next((name for name in ("MKL_NUM_THREADS", "OMP_NUM_THREADS") if name in requested), "auto")
    budget = max(0, cores - reserved)
    automatic = max(1, min(8, budget // local_world_size)) if cpu_offload else 1
    threads = requested.get(source, automatic)
    plan = {**resources, "reserved_cores": reserved, "local_world_size": local_world_size,
            "cpu_offload": cpu_offload, "threads_per_rank": threads, "source": source,
            "training_threads": threads * local_world_size}
    if plan["training_threads"] > cores:
        logger.warning("Training threads exceed the detected CPU budget: %s", plan)
    return plan


def configure_inference_cpu_threads(environment: dict[str, str]) -> None:
    """Apply an optional inference-only limit to a copied server environment."""
    value = environment.get("HYPER_RL_VLLM_CPU_THREADS")
    if value is not None:
        threads = _positive_threads(value, "HYPER_RL_VLLM_CPU_THREADS")
        environment.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))


def main() -> None:
    """Print one shell-readable thread count and log its resource calculation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-world-size", required=True, type=int)
    parser.add_argument("--report", type=Path)
    args, overrides = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config = load_config(args.config, overrides)
    offload = config["train"]["accelerator"]["cpu_offload"]
    if not isinstance(offload, bool):
        raise ValueError("train.accelerator.cpu_offload must be a boolean")
    plan = resolve_cpu_threads(detect_cpu_resources(), args.local_world_size,
                               cpu_offload=offload, environment=os.environ)
    # Validate the independent inference budget before allocating model memory.
    configure_inference_cpu_threads(dict(os.environ))
    logger.info("CPU_THREAD_BUDGET %s", json.dumps(plan, sort_keys=True))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(plan["threads_per_rank"])


if __name__ == "__main__":
    main()
