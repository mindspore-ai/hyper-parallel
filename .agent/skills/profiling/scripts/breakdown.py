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

Reads what ``torch_npu.profiler`` writes under ``<run>_ascend_pt/``:

* ``ASCEND_PROFILER_OUTPUT/step_trace_time.csv`` — the additive step split.
  ``Stage`` equals ``Computing + Communication(Not Overlapped) + Free``; the
  parser checks that identity and reports the drift rather than assuming it.
* ``ASCEND_PROFILER_OUTPUT/op_statistic.csv`` — device time per operator type
  with a ``Core Type`` column (``AI_CORE`` / ``AI_VECTOR_CORE`` / ``MIX_AIV`` /
  ``AI_CPU`` / ``COMMUNICATION``), used for the compute-class split and for
  AI-CPU fallback hotspots.
* ``ASCEND_PROFILER_OUTPUT/operator_details.csv`` — torch-level ranges;
  ``Optimizer.step#<opt>.step`` carries optimizer device time, one row per step.
* ``ASCEND_PROFILER_OUTPUT/communication.json`` — per-step collective and p2p
  detail, with a ``Total Op Info`` aggregate and per-link transit sizes.
* ``profiler_info_<rank>.json`` / ``profiler_metadata.json`` — CANN and
  torch_npu versions, the warmup/active schedule, and the parallel groups.

Two properties of the real output drive the parsing and are easy to get wrong:

* Optimizer time is a named range *inside* ``Computing``, not a fourth additive
  slice, so it is reported as a share of the step but kept out of the sum.
* HCCL launches an AI-CPU-side kernel per collective (``allreduceAicpuKernel``
  and friends). It runs on AI CPU but it is communication, not an operator that
  fell back to AI CPU, so it is excluded from the fallback hotspots.

Pure stdlib; it reads files the profiler already produced and writes HTML.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

OUTPUT_SUBDIR = "ASCEND_PROFILER_OUTPUT"

# `Core Type` (op_statistic) / `Accelerator Core` (kernel_details) value →
# reported class. Value set observed on CANN 9.0.0 + torch_npu 2.9.0; an
# unlisted value is reported verbatim instead of folded into a neighbour.
CORE_CLASS = {
    "AI_CORE": "Cube (AI_CORE)",
    "AI_VECTOR_CORE": "Vector (AI_VECTOR_CORE)",
    "MIX_AIV": "Mixed (MIX_AIV)",
    "COMMUNICATION": "通信 kernel",
}

# HCCL's per-collective AI-CPU launch kernel: AI CPU, but communication rather
# than an operator that fell back. Excluded from the fallback hotspot list.
COMM_AICPU_SUFFIX = "AicpuKernel"

# AI_CPU is reported as TWO rows, never one. A single "AI_CPU" row is read as
# operator fallback, and on a collective-heavy step it is mostly HCCL launch
# kernels: a 16-layer 32B crop showed AI_CPU as the largest class (6.08s,
# 53.8% of summed operator time) with an empty fallback table, and the
# author of this parser - with the exclusion rule documented ten lines above
# the constant - still read it as a fail-open fallback detector before
# checking. A label that misleads its own author is a defect in the label.
AI_CPU_COMM_LABEL = "AI_CPU（HCCL 启动核，属通信）"
AI_CPU_OP_LABEL = "AI_CPU（算子回退）"

# communication.json stores the per-step aggregate under this key alongside the
# individual operators; summing every key would double the total.
COMM_TOTAL_KEY = "Total Op Info"

# The three step_trace_time columns that add up to `Stage`.
STEP_PARTS = (("计算", "Computing"),
              ("通信（未掩盖）", "Communication(Not Overlapped)"),
              ("Free", "Free"))
# Columns kept as context: real, but outside the additive split.
STEP_CONTEXT = ("Communication", "Overlapped", "Bubble", "Preparing", "Stage")

# Per-step boundary-attribution slack on `Stage = 计算 + 通信(未掩盖) + Free`.
#
# The drift is where the profiler cut its step boundaries, not a statement
# about the framework, so the gate has to be set at "would this change a
# conclusion?" - and it was not. Six field profiles:
#
#   model / layout          drift    step     drift/step   old verdict
#   GLM-4 9B                 80us   2.576s     3.1e-05     PASS
#   SmolLM2 360M             54us   0.512s     1.05e-04    PASS
#   DeepSeek-V2-Lite MoE     38us   2.110s     1.8e-05     PASS
#   Qwen2.5-VL-3B           476us   8.837s     5.4e-05     PASS
#   Qwen2.5-32B crop, 1 node 188us  2.083s     9.0e-05     PASS
#   Qwen2.5-32B crop, 2 node 226us  2.142s     1.05e-04    FAIL
#
# Two facts kill the old `max(150us, step * 1e-4)` form. The last two rows are
# the same model and shape, 17% apart in drift, straddling the line - so the
# verdict was decided by run-to-run noise. And rows 2 and 6 have *identical*
# relative drift (1.05e-04) with opposite verdicts, because below ~1.5s of
# step time the absolute floor governs and above it the relative term does:
# the threshold tracks step time while the measured drift does not (2.08s and
# 2.11s steps drift 188us and 38us, a 5x spread). Meanwhile the largest drift
# in the set, 476us, passed with 18x slack.
#
# So: gate at 1e-3 of the step, which no field profile approaches (the widest
# is 1.05e-04, a 10x margin) and which a misattributed slice - the failure
# this check exists to catch, at >=1% of the step - still fails by 10x. Keep
# the absolute floor for fast-stepping models, where a few tens of us of
# rounding is a large fraction of a short step. Report the drift's share of
# the step and of the smallest slice every time, passing or failing, so a
# near-threshold result is visible as near-threshold instead of as a verdict.
IDENTITY_TOL_US = 150.0
IDENTITY_TOL_REL = 1e-3
CROSS_TOL_RATIO = 0.01     # slack on step_trace total vs its uncovered part


def find_output_dirs(prof_dir: Path) -> list[Path]:
    """Locate every ``ASCEND_PROFILER_OUTPUT`` directory under ``prof_dir``.

    Accepts the output directory itself, a single ``*_ascend_pt`` run directory,
    or a parent holding one run directory per rank.

    Args:
        prof_dir: Directory handed to ``--prof-dir``.

    Returns:
        Sorted output directories; empty when none is reachable.
    """
    if prof_dir.name == OUTPUT_SUBDIR:
        return [prof_dir]
    direct = prof_dir / OUTPUT_SUBDIR
    if direct.is_dir():
        return [direct]
    return sorted(path for path in prof_dir.rglob(OUTPUT_SUBDIR) if path.is_dir())


def read_rows(path: Path) -> list[dict]:
    """Read a CSV into dicts, or return an empty list when it is absent."""
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None) -> float:
    """Parse a profiler numeric cell, tolerating blanks and trailing tabs."""
    if value is None:
        return 0.0
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return 0.0


def parse_step_trace(out_dir: Path) -> dict:
    """Average the additive step split over the profiled steps.

    Args:
        out_dir: An ``ASCEND_PROFILER_OUTPUT`` directory.

    Returns:
        ``{"steps": n, "parts": {label: us}, "context": {column: us},
        "identity_drift_us": float}``; empty when the file is absent.
    """
    rows = read_rows(out_dir / "step_trace_time.csv")
    if not rows:
        return {}
    count = float(len(rows))
    parts = {label: sum(to_float(row.get(col)) for row in rows) / count
             for label, col in STEP_PARTS}
    context = {col: sum(to_float(row.get(col)) for row in rows) / count
               for col in STEP_CONTEXT}
    return {"steps": len(rows), "parts": parts, "context": context,
            "step_ids": [row.get("Step", "?") for row in rows],
            "identity_drift_us": abs(sum(parts.values()) - context.get("Stage", 0.0))}


def parse_optimizer(out_dir: Path) -> dict:
    """Read optimizer device time from the ``Optimizer.step#`` torch ranges.

    Each profiled step emits exactly one such row, and its
    ``Device Total Duration(us)`` is the whole optimizer range — unlike the
    operator rows, which would have to be summed by name and would then depend
    on telling ``Pow`` (a model op) from ``Pows`` (AdamW bias correction).

    Args:
        out_dir: An ``ASCEND_PROFILER_OUTPUT`` directory.

    Returns:
        ``{"name": str, "per_step_us": float, "rows": int}``; empty when no
        optimizer range was recorded.
    """
    hits = [row for row in read_rows(out_dir / "operator_details.csv")
            if row.get("Name", "").startswith("Optimizer.step#")]
    if not hits:
        return {}
    total = sum(to_float(row.get("Device Total Duration(us)")) for row in hits)
    return {"name": hits[0]["Name"], "rows": len(hits), "per_step_us": total / len(hits)}


def parse_compute(out_dir: Path) -> dict:
    """Split device operator time by core type and list AI-CPU fallbacks.

    Args:
        out_dir: An ``ASCEND_PROFILER_OUTPUT`` directory.

    Returns:
        ``{"classes": {label: us}, "fallbacks": [(op_type, us, share)],
        "total_us": float}``; empty when ``op_statistic.csv`` is absent.
    """
    rows = read_rows(out_dir / "op_statistic.csv")
    if not rows:
        return {}
    classes: dict[str, float] = {}
    fallbacks: list[tuple[str, float, float]] = []
    total = 0.0
    for row in rows:
        core = (row.get("Core Type") or "").strip().upper()
        op_type = (row.get("OP Type") or "").strip()
        dur = to_float(row.get("Total Time(us)"))
        is_comm_launch = op_type.endswith(COMM_AICPU_SUFFIX)
        if core == "AI_CPU":
            label = AI_CPU_COMM_LABEL if is_comm_launch else AI_CPU_OP_LABEL
        else:
            label = CORE_CLASS.get(core, core or "unclassified")
        classes[label] = classes.get(label, 0.0) + dur
        total += dur
        if core == "AI_CPU" and not is_comm_launch:
            fallbacks.append((op_type, dur, to_float(row.get("Ratio(%)"))))
    fallbacks.sort(key=lambda item: item[1], reverse=True)
    return {"classes": classes, "fallbacks": fallbacks, "total_us": total}


def _comm_time_info(record: dict) -> dict:
    """Pull the time block out of one communication.json entry."""
    return record.get("Communication Time Info", {}) if isinstance(record, dict) else {}


def parse_communication(out_dir: Path) -> dict:
    """Aggregate per-step collective totals and per-link transit volume.

    Args:
        out_dir: An ``ASCEND_PROFILER_OUTPUT`` directory.

    Returns:
        ``{"steps": n, "elapse_us": float, "wait_us": float, "ops": int,
        "links": {name: (transit_ms, size_mb)}, "groups": [hash]}``; empty when
        ``communication.json`` is absent.
    """
    path = out_dir / "communication.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    elapse, wait, ops, steps = 0.0, 0.0, 0, 0
    links: dict[str, list[float]] = {}
    groups: set[str] = set()
    for step in payload.values():
        steps += 1
        for kind in ("collective", "p2p"):
            entries = step.get(kind) or {}
            total = _comm_time_info(entries.get(COMM_TOTAL_KEY, {}))
            elapse += total.get("Elapse Time(ms)", 0.0) * 1000.0
            wait += total.get("Wait Time(ms)", 0.0) * 1000.0
            for name, record in entries.items():
                if name == COMM_TOTAL_KEY:
                    continue
                ops += 1
                groups.add(name.rsplit("@", 1)[-1] if "@" in name else "?")
                _accumulate_links(record, links)
    if not steps:
        return {}
    return {"steps": steps, "ops": ops, "elapse_us": elapse / steps,
            "wait_us": wait / steps, "groups": sorted(groups),
            "links": {k: tuple(v) for k, v in links.items() if v[1] > 0.0}}


def _accumulate_links(record: dict, links: dict[str, list[float]]) -> None:
    """Add one collective's per-link transit time and size into ``links``."""
    bandwidth = record.get("Communication Bandwidth Info", {}) if isinstance(record, dict) else {}
    for link, info in bandwidth.items():
        if not isinstance(info, dict):
            continue
        slot = links.setdefault(link, [0.0, 0.0])
        slot[0] += info.get("Transit Time(ms)", 0.0)
        slot[1] += info.get("Transit Size(MB)", 0.0)


def parse_provenance(run_dir: Path) -> dict:
    """Read versions, the profiler schedule and the parallel groups.

    Args:
        run_dir: The ``*_ascend_pt`` directory holding ``ASCEND_PROFILER_OUTPUT``.

    Returns:
        ``{"cann": str, "torch_npu": str, "rank": str, "schedule": dict,
        "level": str, "groups": {name: ranks}}``; missing fields are omitted.
    """
    out: dict = {}
    info_files = sorted(run_dir.glob("profiler_info*.json"))
    if info_files:
        with info_files[0].open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        config = info.get("config", {})
        common = config.get("common_config", {})
        out.update(cann=info.get("cann_version", "?"), rank=info.get("rank_id", "?"),
                   torch_npu=info.get("torch_npu_version", "?"),
                   schedule=common.get("schedule", {}),
                   level=config.get("experimental_config", {}).get("_profiler_level", "?"))
    meta = run_dir / "profiler_metadata.json"
    if meta.is_file():
        with meta.open("r", encoding="utf-8") as handle:
            groups = json.load(handle).get("parallel_group_info", {}) or {}
        out["groups"] = {g.get("group_name", key): g.get("global_ranks", [])
                         for key, g in groups.items()}
    return out


def run_checks(report: dict) -> list[tuple[str, bool, str]]:
    """Validate the parse against the profiler's own invariants.

    Returns a ``(check, passed, detail)`` row per invariant so the report states
    what held rather than implying it.
    """
    checks: list[tuple[str, bool, str]] = []
    trace, comm, opt = report.get("trace", {}), report.get("comm", {}), report.get("optimizer", {})
    prov = report.get("provenance", {})
    if trace:
        drift = trace["identity_drift_us"]
        stage = trace["context"].get("Stage", 0.0)
        tol = max(IDENTITY_TOL_US, stage * IDENTITY_TOL_REL)
        # The share of the step and of the smallest slice are reported whether
        # the check passes or fails: the drift is a measurement of boundary
        # attribution, and a bare verdict hides how close it ran to the line.
        smallest = min(trace["parts"].values()) if trace["parts"] else 0.0
        share_step = drift / stage * 100 if stage else float("nan")
        share_slice = drift / smallest * 100 if smallest else float("nan")
        checks.append(("Stage = 计算 + 通信(未掩盖) + Free", drift <= tol,
                       f"偏差 {drift:.3f}us（容差 {tol:.3f}us）"
                       f"；占步 {share_step:.4f}%，占最小片 {share_slice:.3f}%"))
        total = trace["context"].get("Communication")
        uncovered = trace["parts"].get("通信（未掩盖）")
        if total is not None and uncovered is not None:
            checks.append(("step_trace: 通信总量 >= 通信（未掩盖）",
                           total + total * CROSS_TOL_RATIO >= uncovered,
                           f"{total:.3f}us >= {uncovered:.3f}us"))
    if trace and comm:
        # The two files measure communication differently once compute/comm
        # overlap and multi-rank aggregation enter: a real 8-rank FSDP run
        # showed 1.77x between them with both parses correct. The row reports
        # the ratio instead of gating on an equality that only holds for
        # single-stream captures.
        ref = trace["context"].get("Communication", 0.0)
        ratio = comm["elapse_us"] / ref if ref > 0.0 else float("inf")
        checks.append(("跨文件口径比（communication.json / step_trace，报告项）", True,
                       f"{comm['elapse_us']:.3f}us / {ref:.3f}us = {ratio:.2f}"))
    warmup = (prov.get("schedule") or {}).get("warmup")
    if warmup is not None:
        checks.append(("采集跳过了 warmup", warmup > 0, f"warmup={warmup}"))
    active = (prov.get("schedule") or {}).get("active")
    if active is not None:
        checks.append(("稳态步数 >= 2（单步不可靠）", active >= 2, f"active={active}"))
    if trace and opt:
        steps = trace["steps"]
        rows = opt["rows"]
        per_step = rows // steps if steps and rows % steps == 0 else 0
        checks.append(("优化器区间数 = 步数的整数倍", steps > 0 and per_step >= 1,
                       f"{rows} 条 / {steps} 步"
                       + (f"（每步 {per_step} 个优化器区间）" if per_step else "")))
    return checks


def analyse(out_dir: Path) -> dict:
    """Parse one rank's output directory into the full breakdown record."""
    report = {
        "out_dir": out_dir,
        "trace": parse_step_trace(out_dir),
        "optimizer": parse_optimizer(out_dir),
        "compute": parse_compute(out_dir),
        "comm": parse_communication(out_dir),
        "provenance": parse_provenance(out_dir.parent),
    }
    report["checks"] = run_checks(report)
    return report


def _rows(cells: list[tuple]) -> str:
    """Render table rows from a list of cell tuples."""
    return "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in cells)


def _step_rows(report: dict) -> str:
    """Build the step-dimension table, with optimizer marked as a sub-range."""
    trace = report.get("trace") or {}
    if not trace:
        return '<tr><td colspan=4 class="todo">[待解析] 无 step_trace_time.csv</td></tr>'
    stage = trace["context"].get("Stage") or sum(trace["parts"].values()) or 1.0
    cells = [(label, f"{us:.1f}", f"{us / stage * 100:.1f}%", "加和项")
             for label, us in trace["parts"].items()]
    opt = report.get("optimizer") or {}
    cells.append((f"优化器（{opt.get('name', 'n/a')}）",
                  f"{opt['per_step_us']:.1f}" if opt else "[待解析]",
                  f"{opt['per_step_us'] / stage * 100:.1f}%" if opt else "—",
                  "计算的子集，<b>不参与加和</b>"))
    for col in STEP_CONTEXT:
        value = trace["context"].get(col, 0.0)
        cells.append((col, f"{value:.1f}", "—", "参考项，不在加和内"))
    return _rows(cells)


def _compute_rows(report: dict) -> str:
    """Build the compute-class table."""
    compute = report.get("compute") or {}
    if not compute:
        return '<tr><td colspan=3 class="todo">[待解析] 无 op_statistic.csv</td></tr>'
    total = compute["total_us"] or 1.0
    ordered = sorted(compute["classes"].items(), key=lambda kv: kv[1], reverse=True)
    cells = [(name, f"{us:.1f}", f"{us / total * 100:.1f}%") for name, us in ordered]
    # The denominator is summed operator duration across concurrent streams,
    # so it exceeds the wall-clock step and these shares are NOT comparable
    # with the step split above. Said in the table, because a reader comparing
    # "AI_CPU 53.8%" against "计算 80.3%" reaches a conclusion neither number
    # supports.
    cells.append(("（分母：设备算子时长按流叠加，大于墙钟步时，"
                  "不可与上方步内占比相比）", "", ""))
    return _rows(cells)


def _fallback_rows(report: dict) -> str:
    """Build the AI-CPU fallback hotspot table."""
    compute = report.get("compute") or {}
    hits = compute.get("fallbacks") or []
    if not hits:
        return "<tr><td colspan=3>无（已排除 HCCL 的 AI CPU launch kernel）</td></tr>"
    return _rows([(op, f"{us:.1f}", f"{ratio:.3f}%") for op, us, ratio in hits[:10]])


def _comm_rows(report: dict) -> str:
    """Build the communication table, including per-link bandwidth."""
    comm = report.get("comm") or {}
    if not comm:
        return '<tr><td colspan=3 class="todo">[待解析] 无 communication.json</td></tr>'
    elapse = comm["elapse_us"] or 1.0
    cells: list[tuple] = [
        ("Elapse（每步）", f"{comm['elapse_us']:.1f} us", f"{comm['ops']} 个通信算子/全采集"),
        ("Wait（每步）", f"{comm['wait_us']:.1f} us", f"占 Elapse {comm['wait_us'] / elapse * 100:.1f}%"),
        ("通信域", ", ".join(comm["groups"]) or "—", "communication.json 的 @ 后缀"),
    ]
    for link, (transit_ms, size_mb) in sorted(comm["links"].items()):
        # MB per ms is GB per s already (1 MB/ms = 1000 MB/s = 1 GB/s).
        rate = size_mb / transit_ms if transit_ms > 0 else 0.0
        cells.append((f"链路 {link}", f"{size_mb:.2f} MB / {transit_ms:.3f} ms", f"{rate:.1f} GB/s"))
    return _rows(cells)


def _check_rows(report: dict) -> str:
    """Build the invariant-check table."""
    checks = report.get("checks") or []
    if not checks:
        return "<tr><td colspan=3>无可执行的校验（输入过少）</td></tr>"
    return _rows([(name, "通过" if ok else '<span class="todo">不通过</span>', detail)
                  for name, ok, detail in checks])


def _provenance(report: dict) -> str:
    """One-line provenance string from profiler_info / profiler_metadata."""
    prov = report.get("provenance") or {}
    if not prov:
        return "[待解析] 无 profiler_info.json"
    sched = prov.get("schedule") or {}
    groups = "; ".join(f"{n}={r}" for n, r in (prov.get("groups") or {}).items()) or "—"
    return (f"CANN {prov.get('cann', '?')} · torch_npu {prov.get('torch_npu', '?')} · "
            f"rank {prov.get('rank', '?')} · {prov.get('level', '?')} · "
            f"schedule wait={sched.get('wait', '?')}/warmup={sched.get('warmup', '?')}/"
            f"active={sched.get('active', '?')} · 并行组 {groups}")


def _verdict(report: dict) -> str:
    """Name the single largest movable item, or say why none was named."""
    trace = report.get("trace") or {}
    compute = report.get("compute") or {}
    if not trace:
        return '<span class="todo">[待解析]</span>'
    stage = trace["context"].get("Stage") or sum(trace["parts"].values()) or 1.0
    items = [(label, us) for label, us in trace["parts"].items() if label != "计算"]
    fallbacks = compute.get("fallbacks") or []
    if fallbacks:
        op, us, ratio = fallbacks[0]
        items.append((f"AI_CPU 回退算子 {op}（占设备算子时间 {ratio:.1f}%）", us / max(trace["steps"], 1)))
    label, us = max(items, key=lambda kv: kv[1])
    return f"{label}，每步约 {us:.1f} us，占 Stage {us / stage * 100:.1f}%"


def render(report: dict, template: Path) -> str:
    """Fill the HTML template from a parsed breakdown record."""
    html = template.read_text(encoding="utf-8")
    for marker, value in (
            ("<!--SOURCE-->", str(report["out_dir"])),
            ("<!--PROVENANCE-->", _provenance(report)),
            ("<!--STEP_ROWS-->", _step_rows(report)),
            ("<!--COMPUTE_ROWS-->", _compute_rows(report)),
            ("<!--FALLBACK_ROWS-->", _fallback_rows(report)),
            ("<!--COMM_ROWS-->", _comm_rows(report)),
            ("<!--CHECK_ROWS-->", _check_rows(report)),
            ("<!--VERDICT-->", _verdict(report))):
        html = html.replace(marker, value)
    return html


def _jsonable(report: dict) -> dict:
    """Strip non-serializable fields for ``--json`` output."""
    out = {k: v for k, v in report.items() if k not in ("out_dir", "checks")}
    out["out_dir"] = str(report["out_dir"])
    out["checks"] = [{"check": n, "passed": ok, "detail": d} for n, ok, d in report["checks"]]
    return out


def _rank_label(report: dict) -> str:
    """Name the rank, or say so when the run was not distributed.

    A non-distributed run writes ``profiler_info.json`` with no ``rank_id`` key
    at all, while a distributed one writes ``profiler_info_<rank>.json``.
    """
    rank = (report.get("provenance") or {}).get("rank")
    return f"rank {rank}" if rank not in (None, "?") else "单卡（无 rank 记录）"


def _summarize(report: dict) -> str:
    """One console line per rank: the split plus any failed invariant."""
    trace = report.get("trace") or {}
    if not trace:
        return f"{report['out_dir']}: [待解析] 无 step_trace_time.csv"
    stage = trace["context"].get("Stage") or 1.0
    split = " ".join(f"{label}={us / stage * 100:.1f}%" for label, us in trace["parts"].items())
    failed = [name for name, ok, _ in report["checks"] if not ok]
    tail = f"; 校验未通过: {', '.join(failed)}" if failed else "; 校验全通过"
    return f"{_rank_label(report)}: {split}{tail}"


def main(argv: list[str] | None = None) -> int:
    """Parse the profiling output and write the breakdown HTML report."""
    parser = argparse.ArgumentParser(
        prog="python .agent/skills/profiling/scripts/breakdown.py",
        description="Break an Ascend profiling trace down into an HTML report.")
    parser.add_argument("--prof-dir", required=True,
                        help="ASCEND_PROFILER_OUTPUT, a *_ascend_pt run dir, or their parent")
    parser.add_argument("--out", default="perf-breakdown.html")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="also write the parsed record as JSON")
    args = parser.parse_args(argv)

    prof_dir = Path(args.prof_dir).resolve()
    out_dirs = find_output_dirs(prof_dir)
    if not out_dirs:
        print(f"[breakdown] no {OUTPUT_SUBDIR} under {prof_dir}", flush=True)
        return 1

    reports = [analyse(out_dir) for out_dir in out_dirs]
    template = Path(__file__).resolve().parent.parent / "templates" / "perf-breakdown.html"
    Path(args.out).write_text(render(reports[0], template), encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps([_jsonable(r) for r in reports], indent=2, ensure_ascii=False),
            encoding="utf-8")

    for report in reports:
        print(f"[breakdown] {_summarize(report)}", flush=True)
    print(f"[breakdown] report={args.out} ({len(reports)} rank(s) parsed)", flush=True)
    return 0 if all(ok for r in reports for _, ok, _ in r["checks"]) else 2


if __name__ == "__main__":
    sys.exit(main())
