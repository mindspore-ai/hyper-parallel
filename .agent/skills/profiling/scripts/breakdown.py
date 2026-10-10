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
"""Parse an Ascend profiling output set and break the step down into an HTML report.

Public-toolchain skeleton: it reads the ``op_summary`` CSV from an msprof /
torch_npu profiler output directory, classifies each operator into a compute
class (Cube / Vector / FA) or marks AI-CPU fallbacks, aggregates the shares,
and fills the HTML template. Communication / free / optimizer dimensions and
the parallel-axis mapping are rule-based (see
``references/breakdown-dimensions.md``); classifications needing an internal
口径 are left ``[待对齐]`` and reported as ``unclassified`` rather than
guessed.

Pure stdlib; it reads files the profiler already produced and writes HTML.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Operator-type substrings → compute class. Rule-based and documented; extend
# in references/breakdown-dimensions.md. [待对齐] for internal-specific types.
_COMPUTE_RULES = (
    ("FA", ("FlashAttention", "FusedAttention", "PromptFlashAttention", "IncreFlashAttention")),
    ("Cube", ("MatMul", "BatchMatMul", "Conv", "Gemm")),
    ("Vector", ("Add", "Mul", "Softmax", "LayerNorm", "RmsNorm", "Gelu", "ReduceSum", "Cast")),
)
_OPTYPE_KEYS = ("OP Type", "Op Type", "Type")
_DUR_KEYS = ("Task Duration(us)", "Task Duration", "Duration(us)", "aicore_time(us)")
_TASKTYPE_KEYS = ("Task Type", "Task_Type")


def _resolve(row: dict, keys: tuple[str, ...]) -> str | None:
    """First present, non-empty column among ``keys``."""
    for key in keys:
        if key in row and row[key] != "":
            return row[key]
    return None


def _classify(op_type: str, task_type: str | None) -> str:
    """Return the compute class for an operator, or 'AI_CPU' / 'unclassified'."""
    if task_type and "AI_CPU" in task_type.upper().replace(" ", "_"):
        return "AI_CPU"
    for name, needles in _COMPUTE_RULES:
        if any(needle.lower() in op_type.lower() for needle in needles):
            return name
    return "unclassified"


def _op_summary_csv(prof_dir: Path) -> Path | None:
    """Find the op_summary CSV in the profiler output tree."""
    matches = sorted(prof_dir.rglob("*op_summary*.csv"))
    return matches[-1] if matches else None


def breakdown_compute(prof_dir: Path) -> dict[str, float]:
    """Aggregate device operator time by compute class from op_summary."""
    csv_path = _op_summary_csv(prof_dir)
    if csv_path is None:
        return {}
    buckets: dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            op_type = _resolve(row, _OPTYPE_KEYS) or ""
            dur = _to_float(_resolve(row, _DUR_KEYS)) or 0.0
            cls = _classify(op_type, _resolve(row, _TASKTYPE_KEYS))
            buckets[cls] = buckets.get(cls, 0.0) + dur
    return buckets


def _to_float(value: str | None) -> float | None:
    """Best-effort float parse."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _render(buckets: dict[str, float], prof_dir: Path, template: Path) -> str:
    """Fill the HTML template with the compute breakdown rows."""
    total = sum(buckets.values()) or 1.0
    rows = "".join(
        f"<tr><td>{name}</td><td>{dur:.1f}</td><td>{dur / total * 100:.1f}%</td></tr>"
        for name, dur in sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)
    )
    html = template.read_text(encoding="utf-8")
    return (html.replace("<!--SOURCE-->", str(prof_dir))
                .replace("<!--COMPUTE_ROWS-->", rows or "<tr><td colspan=3>[待解析]</td></tr>"))


def main(argv: list[str] | None = None) -> int:
    """Parse the profiling output and write the breakdown HTML report."""
    parser = argparse.ArgumentParser(
        prog="python .agent/skills/profiling/scripts/breakdown.py",
        description="Break an Ascend profiling trace down into an HTML report.",
    )
    parser.add_argument("--prof-dir", required=True)
    parser.add_argument("--out", default="perf-breakdown.html")
    args = parser.parse_args(argv)

    prof_dir = Path(args.prof_dir).resolve()
    template = Path(__file__).resolve().parent.parent / "templates" / "perf-breakdown.html"
    buckets = breakdown_compute(prof_dir)
    Path(args.out).write_text(_render(buckets, prof_dir, template), encoding="utf-8")
    if not buckets:
        print(f"[breakdown] no op_summary parsed under {prof_dir}; report marked [待解析]", flush=True)
        return 1
    total = sum(buckets.values())
    summary = " ".join(f"{k}={v / total * 100:.0f}%" for k, v in sorted(buckets.items()))
    print(f"[breakdown] compute split: {summary}; report={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
