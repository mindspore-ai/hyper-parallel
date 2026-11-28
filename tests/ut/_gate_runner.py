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
"""Run the full CPU gate in two pytest sessions sharing its coverage configuration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

import pytest


SUITE = Path(__file__).resolve().parent
SERIAL_PATHS = (
    "auto_models",
    "auto_parallel/hyper_offload/runtime/test_residency.py",
    "models/deepseek_v41/adapter/validation/test_cropped_model.py",
)
PHASE = "HYPER_PARALLEL_UT_PHASE"


def is_gate() -> bool:
    """Recognize Jenkins and its separately packaged tests inside the UT container."""
    return bool(os.environ.get("JENKINS_URL")) or SUITE.parent.parts[-2:] == ("testcases", "tests")


def _cpu_limits() -> dict:
    """Read the process affinity and quotas at each visible CPU controller ancestor."""
    status = Path("/proc/self/status")
    if not status.is_file():
        return {"system_cpu_count": os.cpu_count()}
    affinity = next(row.split()[1] for row in status.read_text(encoding="utf-8").splitlines()
                    if row.startswith("Cpus_allowed_list:"))
    ranges = [list(map(int, span.split("-"))) for span in affinity.split(",")]
    count = sum(span[-1] - span[0] + 1 for span in ranges)
    return {"affinity_count": count, "affinity": affinity, "quotas": _cgroup_quotas()}


def _cgroup_quotas() -> dict:
    """Locate the process CPU controllers in v1 or unified cgroup mounts."""
    groups = {}
    for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
        _, controllers, path = line.split(":", 2)
        groups.update(dict.fromkeys(controllers.split(","), Path(path)))
    quotas = {}
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields, spec = line.split(" - ", 1)
        filesystem, _, opts = spec.split()
        unified = filesystem == "cgroup2"
        if not unified and not (filesystem == "cgroup" and "cpu" in opts.split(",")):
            continue
        root, mount = map(Path, fields.split()[3:5])
        member = groups.get("" if unified else "cpu")
        if member is None:
            continue
        current = mount / member.relative_to(root) if member.is_relative_to(root) else mount
        quotas.update(_ancestor_quotas(current, mount, unified))
    return quotas


def _ancestor_quotas(current: Path, mount: Path, unified: bool) -> dict:
    """Read quota and period through the visible ancestors of a CPU controller."""
    quotas = {}
    while current.is_relative_to(mount):
        quota_file = current / ("cpu.max" if unified else "cpu.cfs_quota_us")
        if quota_file.is_file():
            raw = quota_file.read_text(encoding="utf-8").strip()
            if not unified:
                raw += " " + (current / "cpu.cfs_period_us").read_text(encoding="utf-8").strip()
            quotas[str(quota_file)] = raw
        if current == mount:
            break
        current = current.parent
    return quotas


def coverage_details() -> dict:
    """Report the coverage collector actually active in this pytest process."""
    try:
        cov = import_module("coverage").Coverage.current()
    except ModuleNotFoundError:
        cov = None
    if cov is None:
        return {"active": False}
    return {"active": True, "source": cov.get_option("run:source"), "branch": cov.get_option("run:branch"),
            "data_file": cov.get_option("run:data_file")}


def _merge_junit(paths: list[Path], output: Path) -> None:
    """Keep both phases in the JUnit document requested by the original command."""
    suites = ElementTree.Element("testsuites", name="pytest tests")
    for path in paths:
        document = ElementTree.parse(path).getroot()
        suites.extend(list(document) if document.tag == "testsuites" else [document])
    output.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.ElementTree(suites).write(output, encoding="utf-8", xml_declaration=True)


def _run_phases(config: pytest.Config, args: list[str], index: int, directory: Path) -> int:
    """Execute both file partitions in order and propagate test and runner failures."""
    junit = config.getoption("xmlpath", default=None)
    xml_paths = []
    result = 0
    started = time.monotonic()
    for phase in ("serial", "parallel"):
        child = args.copy()
        if phase == "serial":
            child[index:index + 1] = [str(SUITE / path) for path in SERIAL_PATHS]
        else:
            child.extend(["-n", "4", "--dist=loadfile", *(f"--ignore={SUITE / path}" for path in SERIAL_PATHS)])
        if phase == "parallel" and config.getoption("cov_source", default=None):
            child.append("--cov-append")
        if junit:
            xml_paths.append(directory / f"{phase}.xml")
            child.append(f"--junitxml={xml_paths[-1]}")
        print(f"UT_PHASE_START {phase}", flush=True)
        before = time.monotonic()
        completed = subprocess.run([sys.executable, "-m", "pytest", *child], env={**os.environ, PHASE: phase},
                                   check=False)
        print(f"UT_PHASE_END {phase} exit={completed.returncode} wall_s={time.monotonic() - before:.3f}", flush=True)
        if completed.returncode not in (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED):
            return completed.returncode if completed.returncode > 0 else 128 - completed.returncode
        result = max(result, completed.returncode)
    if junit:
        _merge_junit(xml_paths, Path(junit))
    print(f"UT_HYBRID_WALL {time.monotonic() - started:.3f}", flush=True)
    return result


def run_full_gate(config: pytest.Config) -> int | None:
    """Dispatch only an unsharded full-UT invocation in the Jenkins environment.

    Args:
        config: Parsed pytest configuration for this invocation.

    Returns:
        The two phases' combined exit code, or None for normal pytest execution.
    """
    if os.environ.get(PHASE) or not is_gate() or config.option.collectonly or len(config.args) != 1:
        return None
    if Path(config.args[0]).resolve() != SUITE or config.getoption("numprocesses", default=0):
        return None
    args = list(map(str, config.invocation_params.args))
    index = next((i for i, arg in enumerate(args) if not arg.startswith("-") and Path(arg).resolve() == SUITE), None)
    if index is None:
        return None
    print("UT_CPU", json.dumps(_cpu_limits()), flush=True)
    print("UT_COVERAGE controller", json.dumps(coverage_details()), flush=True)
    with TemporaryDirectory(prefix="hp-ut-reports-") as temporary:
        return _run_phases(config, args, index, Path(temporary))
