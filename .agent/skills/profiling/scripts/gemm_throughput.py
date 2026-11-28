# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Estimate useful BF16 GEMM throughput from Ascend kernel shapes and durations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path


TASK_KEY = "model_id, stream_id, task_id, context_id, batch_id"


def _matrix_shape(value: str) -> tuple[int, int] | None:
    """Parse one unblocked two-dimensional matrix shape."""
    parts = [part.strip() for part in value.strip().strip('"').split(",")]
    if len(parts) != 2 or not all(part.isascii() and part.isdecimal() for part in parts):
        return None
    rows, columns = (int(part) for part in parts)
    return (rows, columns) if min(rows, columns) > 0 else None


def _gemm_shape(row: dict[str, str]) -> tuple[int, int, int] | None:
    """Recover M, N and a unique K from a pure BF16 MatMulV3 record."""
    if (row["Input Data Types"] or "").split(";") != ["DT_BF16", "DT_BF16"]:
        return None
    inputs = (row["Input Shapes"] or "").split(";")
    if len(inputs) != 2:
        return None
    left, right = map(_matrix_shape, inputs)
    output = _matrix_shape(row["Output Shapes"] or "")
    if left is None or right is None or output is None:
        return None
    m, n = output
    left_dims = {k for outer, k in (left, left[::-1]) if outer == m}
    right_dims = {k for k, outer in (right, right[::-1]) if outer == n}
    contractions = left_dims.intersection(right_dims)
    return (m, n, contractions.pop()) if len(contractions) == 1 else None


def _duration(value: str | int | float) -> float:
    """Return a finite positive duration or zero for an unusable entry."""
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return 0.0
    return duration if math.isfinite(duration) and duration > 0 else 0.0


def _record(groups: dict, dims: tuple[int, int, int] | None, duration_us: float) -> None:
    """Accumulate comparable logical work and accelerator duration."""
    if dims is None or not duration_us:
        return
    item = groups.setdefault(dims, {"count": 0, "duration_us": 0.0, "flops": 0})
    item["count"] += 1
    item["duration_us"] += duration_us
    item["flops"] += 2 * math.prod(dims)


def _report(groups: dict, all_records: int, all_time: float, peak_tflops: float) -> dict:
    """Render a scoped geometry estimate using the common report schema."""
    results = []
    for dims, item in sorted(groups.items(), key=lambda pair: pair[1]["duration_us"], reverse=True):
        achieved = item["flops"] / item["duration_us"] / 1e6
        results.append({"m_n_k": dims, **item, "shape_derived_tflops": achieved,
                        "percent_of_declared_peak": 100 * achieved / peak_tflops})
    volume = sum(item["flops"] for item in results)
    included_time = sum(item["duration_us"] for item in results)
    achieved = volume / included_time / 1e6
    return {
        "bf16_peak_tflops_die": peak_tflops,
        "all_matmulv3_count": all_records, "all_matmulv3_duration_us": all_time,
        "included_count": sum(item["count"] for item in results),
        "included_duration_us": included_time, "included_flops": volume,
        "shape_derived_tflops": achieved, "percent_of_declared_peak": 100 * achieved / peak_tflops,
        "groups": results,
        "scope": "Useful 2*M*N*K work over summed durations of recognized BF16 matrix kernels. "
                 "Kernel durations can overlap. This estimate excludes blocked shapes, hardware tile padding, "
                 "other operators and device FLOP counters.",
    }


def summarize(csv_path: Path, steps: set[int], peak_tflops: float) -> dict:
    """Report shape-derived GEMM rates, keeping skipped-kernel coverage explicit."""
    if not steps or min(steps) < 1 or not math.isfinite(peak_tflops) or peak_tflops <= 0:
        raise ValueError("Positive step IDs and a finite positive BF16 peak are required")
    requested_steps = {str(step) for step in steps}
    observed_steps: set[str] = set()
    all_records, all_time = 0, 0.0
    groups: dict[tuple[int, int, int], dict] = {}
    required = {"Step Id", "Type", "Duration(us)", "Input Shapes", "Input Data Types", "Output Shapes"}
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing profiling columns: {sorted(missing)}")
        for row in reader:
            step = row["Step Id"]
            if step not in requested_steps:
                continue
            observed_steps.add(step)
            if row["Type"] != "MatMulV3":
                continue
            duration = _duration(row["Duration(us)"])
            all_records += 1
            all_time += duration
            _record(groups, _gemm_shape(row), duration)
    if requested_steps - observed_steps:
        raise ValueError(f"Missing profiling steps: {sorted(requested_steps - observed_steps, key=int)}")
    if not groups:
        raise ValueError("No recognized BF16 matrix records in the requested steps")
    return {"steps": sorted(steps), **_report(groups, all_records, all_time, peak_tflops)}


def _native_database(root: Path, name: str) -> Path:
    """Require one database from one decoded rank capture."""
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} in the decoded rank capture")
    return matches[0]


def _file_sha256(path: Path) -> str:
    """Hash the source database without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_native(rank_profile: Path, peak_tflops: float) -> dict:
    """Estimate BF16 GEMM throughput across a decoded single-rank CANN capture."""
    if not math.isfinite(peak_tflops) or peak_tflops <= 0:
        raise ValueError("A finite positive BF16 peak is required")
    metadata = _native_database(rank_profile, "ge_info.db")
    tasks = _native_database(rank_profile, "ascend_task.db")
    captures = {
        next((parent for parent in path.parents if parent.name.startswith("PROF_")), rank_profile)
        for path in (metadata, tasks)
    }
    if len(captures) != 1:
        raise ValueError("Native metadata and device tasks come from different captures")
    shapes = defaultdict(set)
    with closing(sqlite3.connect(metadata.resolve().as_uri() + "?mode=ro", uri=True)) as database:
        if database.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("Incomplete native task metadata database")
        sql = f"SELECT {TASK_KEY}, op_type, input_data_types, input_shapes, output_shapes FROM TaskInfo"
        for *key, operator, dtypes, inputs, outputs in database.execute(sql):
            shapes[tuple(key)].add((operator, dtypes, inputs, outputs))
    all_records, all_time, skipped = 0, 0.0, 0
    skipped_time = 0.0
    groups = {}
    with closing(sqlite3.connect(tasks.resolve().as_uri() + "?mode=ro", uri=True)) as database:
        if database.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("Incomplete native device task database")
        sql = (f"SELECT {TASK_KEY}, duration FROM AscendTask WHERE start_time > 0 AND duration > 0 "
               "AND device_task_type IN ('AI_CORE', 'MIX_AIC')")
        for *key, value in database.execute(sql):
            duration = _duration(value) / 1e3
            records = shapes.get(tuple(key), ())
            if len(records) != 1:
                skipped += 1
                skipped_time += duration
                continue
            operator, dtypes, inputs, outputs = next(iter(records))
            if operator != "MatMulV3":
                continue
            all_records += 1
            all_time += duration
            row = {"Input Data Types": dtypes, "Input Shapes": inputs, "Output Shapes": outputs}
            _record(groups, _gemm_shape(row), duration)
    if not groups:
        raise ValueError("No recognized BF16 matrix records in the native rank capture")
    return {
        "steps": None,
        "capture_scope": "complete decoded rank capture",
        "unmatched_or_ambiguous_compute_tasks": skipped,
        "unmatched_or_ambiguous_compute_duration_us": skipped_time,
        "database_sha256": {path.name: _file_sha256(path) for path in (metadata, tasks)},
        **_report(groups, all_records, all_time, peak_tflops),
    }


def main() -> None:
    """Write the scoped throughput estimate as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?")
    parser.add_argument("--native-rank", type=Path, help="decoded CANN rank folder containing native SQLite files")
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--peak-tflops", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.csv) == bool(args.native_rank):
        parser.error("Supply a kernel CSV or one decoded --native-rank directory")
    if args.native_rank is not None:
        if args.steps:
            parser.error("Native tasks are scoped to the complete capture. Step selection requires a CSV.")
        result = summarize_native(args.native_rank, args.peak_tflops)
    else:
        if not args.steps:
            parser.error("CSV analysis requires --steps")
        result = summarize(args.csv, set(args.steps), args.peak_tflops)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
