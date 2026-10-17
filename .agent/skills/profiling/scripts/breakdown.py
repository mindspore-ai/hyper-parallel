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
    # Not labelled "回退": this class also holds HCCL's AI-CPU launch kernels.
    # Only the fallback table, which excludes them, makes that call.
    "AI_CPU": "AI_CPU",
    "COMMUNICATION": "通信 kernel",
}

# HCCL's per-collective AI-CPU launch kernel: AI CPU, but communication rather
# than an operator that fell back. Excluded from the fallback hotspot list.
COMM_AICPU_SUFFIX = "AicpuKernel"

# communication.json stores the per-step aggregate under this key alongside the
# individual operators; summing every key would double the total.
COMM_TOTAL_KEY = "Total Op Info"

# The three step_trace_time columns that add up to `Stage`.
STEP_PARTS = (("计算", "Computing"),
              ("通信（未掩盖）", "Communication(Not Overlapped)"),
              ("Free", "Free"))
# Columns kept as context: real, but outside the additive split.
STEP_CONTEXT = ("Communication", "Overlapped", "Bubble", "Preparing", "Stage")

IDENTITY_TOL_US = 1.0      # rounding slack on Stage == sum(parts)
CROSS_TOL_RATIO = 0.01     # 1% slack between communication.json and step_trace


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
        label = CORE_CLASS.get(core, core or "unclassified")
        classes[label] = classes.get(label, 0.0) + dur
        total += dur
        if core == "AI_CPU" and not op_type.endswith(COMM_AICPU_SUFFIX):
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
        checks.append(("Stage = 计算 + 通信(未掩盖) + Free", drift <= IDENTITY_TOL_US,
                       f"偏差 {drift:.3f}us（容差 {IDENTITY_TOL_US}us）"))
    if trace and comm:
        ref = trace["context"].get("Communication", 0.0)
        gap = abs(comm["elapse_us"] - ref)
        ok = ref <= 0.0 or gap <= ref * CROSS_TOL_RATIO
        checks.append(("communication.json Elapse = step_trace Communication", ok,
                       f"{comm['elapse_us']:.3f}us vs {ref:.3f}us，差 {gap:.3f}us"))
    warmup = (prov.get("schedule") or {}).get("warmup")
    if warmup is not None:
        checks.append(("采集跳过了 warmup", warmup > 0, f"warmup={warmup}"))
    active = (prov.get("schedule") or {}).get("active")
    if active is not None:
        checks.append(("稳态步数 >= 2（单步不可靠）", active >= 2, f"active={active}"))
    if trace and opt:
        checks.append(("优化器区间数 = 采集步数", opt["rows"] == trace["steps"],
                       f"{opt['rows']} 条 / {trace['steps']} 步"))
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
    return _rows([(name, f"{us:.1f}", f"{us / total * 100:.1f}%") for name, us in ordered])


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
