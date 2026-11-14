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
"""Call profiling's parser and add measured operator/shape evidence for the planner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


SKILLS = Path(__file__).resolve().parents[2]


def file_hash(path: Path) -> str:
    """Hash evidence without loading a large profiler export into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _add_kernel(groups: dict, row: dict, path: Path) -> None:
    name = row.get("Type") or row.get("Name") or "unknown"
    core = row.get("Accelerator Core") or row.get("Core Type")
    if core == "COMMUNICATION" or name.endswith("AicpuKernel"):
        return
    shape = row.get("Input Shapes") or row.get("InputShapes") or None
    dtype = row.get("Input Data Types") or row.get("Input Dtypes") or None
    layout = row.get("Input Formats") or None
    key = (name, core, shape, dtype, layout)
    duration = float(row["Duration(us)"])
    if not math.isfinite(duration) or duration < 0:
        raise ValueError(f"Invalid kernel duration in {path}: {duration}")
    slot = groups.setdefault(key, {"operator": name, "core": core, "input_shapes": shape,
                                   "input_dtypes": dtype, "input_formats": layout,
                                   "kernel_names": [], "step_ids": [],
                                   "calls": 0, "summed_device_us": 0.0})
    for field, observed in (("kernel_names", row.get("Name")), ("step_ids", row.get("Step Id"))):
        if observed and observed not in slot[field] and len(slot[field]) < 20:
            slot[field].append(observed)
    slot["calls"] += 1
    slot["summed_device_us"] += duration


def operator_shapes(path: Path, limit: int = 40) -> dict:
    """Aggregate actual input shapes, retaining missing columns as unknown.

    Kernel durations are sums across streams, never critical-path fractions.
    Name matching suggests a lookup direction only; it does not prove fallback.
    """
    if not path.is_file():
        return {"status": "UNVERIFIED", "reason": "kernel_details.csv absent", "hotspots": []}
    groups = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if "Duration(us)" not in headers or not ("Name" in headers or "Type" in headers):
            return {"status": "UNVERIFIED", "reason": "unsupported kernel CSV headers", "headers": headers}
        for row in reader:
            _add_kernel(groups, row, path)
    ordered = sorted(groups.values(), key=lambda value: value["summed_device_us"], reverse=True)
    for slot in ordered:
        slot["mean_device_us"] = slot["summed_device_us"] / slot["calls"]
    return {"status": "PASS" if ordered else "UNVERIFIED", "source": str(path),
            "sha256": file_hash(path), "group_count": len(ordered), "hotspots": ordered[:limit],
            "duration_scope": "sum across captured kernels/streams; NOT fraction of wall time",
            "shape_coverage": "known" if ordered and all(x["input_shapes"] for x in ordered) else "partial_or_unknown"}


def check_reports(reports: list[dict]) -> list[str]:
    """Require positive evidence; an empty parser check list is not a pass."""
    problems = []
    if not reports:
        return ["No rank reports"]
    for index, report in enumerate(reports):
        prefix = f"rank report {index}"
        trace = report.get("trace", {})
        if trace.get("steps", 0) < 2 or trace.get("context", {}).get("Stage", 0) <= 0:
            problems.append(f"{prefix}: at least two positive steady-state trace steps required")
        checks = report.get("checks", [])
        if not checks or any(check.get("passed") is not True for check in checks):
            problems.append(f"{prefix}: missing/failed profiling invariant checks")
        schedule = report.get("provenance", {}).get("schedule", {})
        if schedule.get("warmup", 0) < 1:
            problems.append(f"{prefix}: warmup not proven; use a capture with explicit warmup")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Collect facts and optional prior/memory evidence into one state JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prof-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--prior", type=Path, help="PSA output; preserved as a prior, not measured truth")
    parser.add_argument("--memory", type=Path, help="memory-analysis JSON; status retained verbatim")
    parser.add_argument("--capture-metadata", type=Path,
                        help="Profile worker's actual trial_token/source_digest; required for controller acceptance")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    report_path = args.out / "profiling.json"
    command = [sys.executable, str(SKILLS / "profiling/scripts/breakdown.py"),
               "--prof-dir", str(args.prof_dir.resolve()), "--out", str(args.out / "profiling.html"),
               "--json", str(report_path)]
    with (args.out / "parser.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    reports = json.loads(report_path.read_text()) if report_path.exists() else []
    problems = check_reports(reports)
    if completed.returncode != 0:
        problems.append(f"profiling parser exit {completed.returncode}")
    operators = [operator_shapes(Path(report["out_dir"]) / "kernel_details.csv") for report in reports]
    if any(item["status"] != "PASS" for item in operators) or not operators:
        problems.append("Operator evidence missing; do not select a fusion from absent shapes/timings")
    sources = {}
    for report in reports:
        directory = Path(report["out_dir"])
        paths = [directory / name for name in ("step_trace_time.csv", "op_statistic.csv",
                                               "operator_details.csv", "communication.json", "kernel_details.csv")]
        paths.extend(directory.parent.glob("profiler_info*.json"))
        paths.append(directory.parent / "profiler_metadata.json")
        sources.update({str(path): file_hash(path) for path in paths if path.is_file()})
    state = {"schema": "mfu-riser-state/v1", "valid": not problems, "problems": problems,
             "profiling": reports, "operators": operators,
             "source_sha256": sources,
             "prior": json.loads(args.prior.read_text()) if args.prior else None,
             "memory": json.loads(args.memory.read_text()) if args.memory else None,
             "notes": ["Optimizer is a subset of compute, not an additive fourth term.",
                       "No causal attribution or end-to-end speedup inferred from summed kernel durations."]}
    state["measurement_binding"] = json.loads(args.capture_metadata.read_text()) if args.capture_metadata else None
    if args.capture_metadata:
        state["capture_metadata_sha256"] = file_hash(args.capture_metadata)
    (args.out / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"state": str(args.out / "state.json"), "valid": state["valid"], "problems": problems}))
    return 0 if state["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
