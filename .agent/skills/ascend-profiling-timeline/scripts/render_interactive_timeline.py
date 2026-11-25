#!/usr/bin/env python3
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
"""Render a portable, interactive zoomable operator timeline from Ascend profiler CSV exports.

Supports both ASCEND_PROFILER_OUTPUT (kernel_details.csv + step_trace_time.csv)
and CANN mindstudio_profiler_output (op_summary*.csv) automatically.
"""
import argparse
import base64
from collections import defaultdict
import csv
import gzip
from html import escape
import json
import math
from pathlib import Path
from typing import Optional


def number(row: dict, *names: str) -> Optional[float]:
    """Return the first finite nonnegative numeric field, or None."""
    for name in names:
        if name in row:
            try:
                value = float(row[name])
                if math.isfinite(value) and value >= 0:
                    return value
            except (TypeError, ValueError):
                pass
    return None


def percent(row: dict, name: str, scale: float = 100) -> Optional[float]:
    """Scale an available metric without filling missing observations."""
    value = number(row, name)
    return round(value * scale, 2) if value is not None else None


def classify(kind: str, core: str) -> int:
    """Map profiler task identities to display categories."""
    normalized = kind.lower().replace("_", "")
    if core == "COMMUNICATION" or normalized.startswith(("hcom", "hccl")):
        return 2
    if core == "AI_CPU":
        return 3
    if core in ("AI_CORE", "MIX_AIC"):
        return 0
    if core in ("AI_VECTOR_CORE", "MIX_AIV"):
        return 1
    return 4


def merge_intervals(iv: list) -> list:
    """Merge overlapping intervals into disjoint time coverage."""
    if not iv:
        return []
    iv.sort()
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def interval_len(iv: list) -> float:
    """Return the total duration of disjoint intervals."""
    return sum(e - s for s, e in iv)


def intersect_len(a: list, b: list) -> float:
    """Return the duration shared by two disjoint interval lists."""
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def locate_kernel_csv(input_dir: Path) -> Path:
    """Locate a supported kernel CSV in the selected input."""
    if input_dir.is_file() and input_dir.suffix == ".csv":
        return input_dir
    kd = input_dir / "kernel_details.csv"
    if kd.is_file():
        return kd
    for cand in sorted(input_dir.glob("op_summary*.csv")):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Neither kernel_details.csv nor op_summary*.csv found in {input_dir}")


def read_timeline(path: Path) -> tuple[dict, tuple]:
    """Parse task timings, available PMU metrics and interval coverage."""
    labels = []
    label_index = {}
    tasks = []
    weights = defaultdict(lambda: [[0.0, 0.0] for _ in range(4)])
    summary = defaultdict(lambda: [0, 0.0, 0.0])
    comp_iv, comm_iv = [], []
    origin = None

    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            start = number(row, "Start Time(us)", "Task Start Time(us)")
            duration = number(row, "Duration(us)", "Task Duration(us)")
            if start is None or duration is None:
                continue
            if origin is None:
                origin = start
            kind = (row.get("Type") or row.get("OP Type") or "未知").strip()
            name = (row.get("Name") or row.get("Op Name") or kind).strip()
            core = (row.get("Accelerator Core") or row.get("Task Type") or "").strip()
            group = classify(kind, core)
            key = (kind, name, core, group)
            if key not in label_index:
                label_index[key] = len(labels)
                labels.append(key)
            mac = percent(row, "aic_mac_ratio")
            vector = percent(row, "aiv_vec_ratio")
            cube = percent(row, "cube_utilization(%)", 1)
            mte2 = percent(row, "aic_mte2_ratio")
            stream = (row.get("Stream ID") or "").strip()
            aic = number(row, "aicore_time(us)") or 0.0
            aiv = number(row, "aiv_time(us)") or 0.0
            tasks.append(
                [
                    round(start - origin, 1),
                    round(duration, 1),
                    label_index[key],
                    stream,
                    mac if aic else None,
                    vector if aiv else None,
                    cube if aic else None,
                ]
            )
            if group < 2:
                comp_iv.append((start, start + duration))
            elif group == 2:
                comm_iv.append((start, start + duration))

            totals = summary[kind]
            totals[0] += 1
            totals[1] += duration
            totals[2] += aic
            for pair, (value, weight) in zip(weights[kind], ((cube, aic), (mac, aic), (vector, aiv), (mte2, aic))):
                if value is not None and weight:
                    pair[0] += value * weight
                    pair[1] += weight

    if not tasks:
        raise ValueError(f"no timed tasks in {path}")
    tasks.sort(key=lambda task: task[0])
    first = tasks[0][0]
    if first:
        for task in tasks:
            task[0] = round(task[0] - first, 1)
    span = max(row[0] + row[1] for row in tasks)

    all_aic = sum(row[2] for row in summary.values())
    table = []
    for kind, (count, duration, aic) in sorted(summary.items(), key=lambda pair: -pair[1][1]):
        metrics = [round(weighted / weight, 2) if weight else None for weighted, weight in weights[kind]]
        table.append(
            [
                kind,
                count,
                round(duration / 1000, 2),
                round(aic / 1000, 2),
                *metrics,
                round(100 * aic / all_aic, 2) if all_aic else None,
            ]
        )
    overall = []
    for metric in range(4):
        weighted = sum(group[metric][0] for group in weights.values())
        divisor = sum(group[metric][1] for group in weights.values())
        overall.append(weighted / divisor if divisor else None)

    comp_m = merge_intervals(comp_iv)
    comm_m = merge_intervals(comm_iv)
    comp_t = interval_len(comp_m)
    comm_t = interval_len(comm_m)
    ov_t = intersect_len(comp_m, comm_m)
    fallback_cov = (
        [max(0.0, comp_t - ov_t), max(0.0, comm_t - ov_t), ov_t, max(0.0, span - comp_t - max(0.0, comm_t - ov_t))],
        span,
        ["未标号"],
    )
    return {"ops": labels, "tasks": tasks, "table": table, "overall": overall, "span": span}, fallback_cov


def read_coverage(path: Path) -> tuple[list, float, list]:
    """Read step coverage, rejecting inconsistent or empty timing."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no step timing in {path}")
    fields = ["Computing", "Communication(Not Overlapped)", "Overlapped", "Free"]
    totals = [sum(number(row, field) or 0 for row in rows) for field in fields]
    compute, comm, overlap, free = totals
    if overlap > compute:
        raise ValueError("overlap exceeds compute coverage")
    durations = [compute - overlap, comm, overlap, free]
    total = sum(durations)
    if total <= 0:
        raise ValueError("step timing has zero covered time")
    return durations, total, [row.get("Step") or "未标号" for row in rows]


def main(argv: Optional[list[str]] = None) -> None:
    """Parse command-line inputs and write the profiling report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="Profiling directory containing kernel_details.csv or op_summary*.csv")
    ap.add_argument("output", type=Path, help="Output self-contained HTML file")
    ap.add_argument("--scope", required=True, help="Subtitle describing hardware/run/step scope")
    ap.add_argument("--sequence", default="Ascend NPU", help="Sequence / workload label for page title")
    ap.add_argument("--reference", default="", help="Optional reference note in method section")
    ap.add_argument(
        "--window-ms",
        default="",
        help="Optional interactive timeline slice START:END in ms (keeps full-step coverage & ranking table)",
    )
    ap.add_argument(
        "--min-duration-us",
        type=float,
        default=0.0,
        help="Optional minimum task duration (us) for interactive timeline payload",
    )
    args = ap.parse_args(argv)

    if not math.isfinite(args.min_duration_us) or args.min_duration_us < 0:
        raise ValueError("min-duration-us must be finite and nonnegative")
    kernel_csv = locate_kernel_csv(args.input)
    data, fallback_cov = read_timeline(kernel_csv)
    total_task_count = len(data["tasks"])
    slice_note = ""
    if args.window_ms or args.min_duration_us > 0:
        w_lo, w_hi = 0.0, data["span"]
        if args.window_ms:
            parts = args.window_ms.split(":")
            if len(parts) != 2:
                raise ValueError("window-ms must be START:END")
            w_lo, w_hi = (float(part) * 1000.0 for part in parts)
            if not (math.isfinite(w_lo) and math.isfinite(w_hi) and 0 <= w_lo < w_hi):
                raise ValueError("window-ms must have finite 0 <= START < END")
        kept = [t for t in data["tasks"] if (t[0] + t[1] >= w_lo and t[0] <= w_hi and t[1] >= args.min_duration_us)]
        if not kept:
            raise ValueError("No tasks in the requested timeline window/duration filter")
        if kept:
            base_t = kept[0][0]
            used_ops = {}
            new_ops = []
            new_tasks = []
            for t in kept:
                oid = t[2]
                if oid not in used_ops:
                    used_ops[oid] = len(new_ops)
                    new_ops.append(data["ops"][oid])
                new_tasks.append([round(t[0] - base_t, 1), t[1], used_ops[oid], t[3], t[4], t[5], t[6]])
            data["ops"] = new_ops
            data["tasks"] = new_tasks
            data["span"] = max(t[0] + t[1] for t in new_tasks)
            slice_note = (
                f"（时间线仅载入局部切片 {w_lo/1000:.0f}–{w_hi/1000:.0f} ms 共 {len(new_tasks):,} /"
                f" {total_task_count:,} 条任务，上方覆盖与下方算子表仍为全步统计）"
            )
    step_csv = (args.input if args.input.is_dir() else args.input.parent) / "step_trace_time.csv"
    if step_csv.is_file():
        coverage, total, steps = read_coverage(step_csv)
    else:
        coverage, total, steps = fallback_cov

    legend = ["纯计算", "纯通信", "计算与通信重叠", "Free"]
    colors = ["#2563eb", "#dc2626", "#16a34a", "#94a3b8"]
    boxes = "".join(
        f'<div style="--c:{color}"><b>{value/total:.1%}</b><span>{title} · {value/1e6:.3f} s</span></div>'
        for title, color, value in zip(legend, colors, coverage)
    )
    stack = "".join(
        f'<span style="width:{value/total:.6%};background:{color}"></span>' for color, value in zip(colors, coverage)
    )
    data["steps"] = steps
    overview = " · ".join(
        f"{label} {value:.2f}%"
        for label, value in zip(("Cube", "MAC", "Vector", "MTE2"), data["overall"])
        if value is not None
    )
    payload = base64.b64encode(
        gzip.compress(json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode(), mtime=0)
    ).decode()

    sparse_grad = next((row for row in data["table"] if row[0] == "SparseFlashMlaGrad"), None)
    attention = ""
    if sparse_grad:
        _, count, duration, _aic, _cube, mac, _vector, _mte2, share = sparse_grad
        pmu = f" · 加权 MAC {mac:.2f}%" if mac is not None else ""
        aic_share = f" · 占全部 AICore 累计时间 {share:.2f}%" if share is not None else ""
        attention = (
            f'<p class="spotlight"><b>稀疏注意力反向</b>　{count:,} 次 · 累计 {duration:,.2f} ms'
            f"{aic_share}{pmu}。可点击算子名对照该调用与通信的时间位置。</p>"
        )

    template = (Path(__file__).resolve().parents[1] / "resources" / "timeline.html").read_text(encoding="utf-8")
    values = {
        "TITLE": escape(args.sequence),
        "SCOPE": escape(args.scope),
        "REFERENCE": escape(args.reference),
        "COVERAGE": boxes,
        "STACK": stack,
        "STATS": (
            f"采样窗口 {total/1e6:.3f} s · {total_task_count:,} 条任务 · 通信重叠率"
            f" {coverage[2]/max(1,coverage[1]+coverage[2]):.1%}{slice_note}"
        ),
        "DATA": payload,
        "STEPS": escape("、".join(steps)),
        "UTILIZATION": overview or "无可用 PMU 读数",
        "ATTENTION": attention,
        "FIGURES": "",
    }
    for name, value in values.items():
        template = template.replace("@" + name + "@", value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template, encoding="utf-8")
    print(f"{args.output} · {len(data['tasks']):,} tasks · steps {'/'.join(steps)}")


if __name__ == "__main__":
    main()
