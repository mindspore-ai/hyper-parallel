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
"""Parse an NPU memory snapshot and break device memory down into an HTML report.

Written against snapshots taken with ``torch_npu.npu.memory._snapshot()`` on
torch_npu 2.9.0.post2 / Ascend 910. The snapshot is ``{"segments": [...],
"device_traces": [...]}``; each segment carries ``total_size`` /
``allocated_size`` / ``active_size`` and a ``blocks`` list whose entries have
``size`` / ``requested_size`` / ``state`` / ``frames``.

Three properties of the real data drive this parser:

* **A snapshot is one instant, not the peak.** A snapshot taken after
  ``optimizer.step()`` holds parameters, gradients and optimizer state but no
  activations at all; one taken mid-backward holds activations too and was
  still below ``max_memory_allocated``. The report therefore states the
  snapshot instant's total next to the recorded peak and flags the gap instead
  of letting a snapshot be read as the peak.
* **``frames[0]`` is useless for attribution.** The recorded stack is
  innermost-first and its innermost entries are the snapshot machinery itself
  (``torch_npu::unwind::unwind``, ``CapturedTraceback::gather``, ``??``), so
  every block would be attributed to the same frame. Attribution scans for the
  innermost Python frame outside ``site-packages`` for the call site, and uses
  the framework frames for the category.
* **Gradients carry no Python frame at all** — the autograd engine allocates
  them from C++ — so they are recognised by
  ``torch::autograd::AccumulateGrad`` among the native frames.

Pure stdlib. The snapshot is a file the recording already produced.
"""
from __future__ import annotations

import argparse
import json
import pickle  # nosec B403 - trusted local snapshot produced by our own run
import sys
from pathlib import Path

MIB = 1024 * 1024

# Innermost frames belonging to the capture machinery, not the allocation site.
FRAME_NOISE = ("unwind", "CapturedTraceback", "gather", "??")
# Allocator plumbing that every device allocation goes through; the frame that
# says something about the caller sits just outside it (for a gradient, that is
# `torch::autograd::AccumulateGrad::apply`).
NATIVE_PLUMBING = ("at::", "c10::")
# Installed-library path markers; a frame outside these is the user's own code.
SITE_MARKERS = ("site-packages", "dist-packages")
# Native symbol that marks a gradient buffer allocated by the autograd engine.
GRAD_SYMBOL = "AccumulateGrad"

ACTIVE_PREFIX = "active"
CATEGORY_OTHER = "其他用户张量"
CATEGORY_UNKNOWN = "unclassified"


def load_snapshot(path: Path) -> dict:
    """Load a snapshot from pickle or JSON.

    Args:
        path: Snapshot file written by the recording hook.

    Returns:
        The snapshot dict.
    """
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)  # nosec B301 - local, self-produced file
    except PermissionError as exc:
        # `_dump_snapshot` creates the file write-only (mode 0200); a dump taken
        # without `mem_hook`, which restores owner read, lands unreadable.
        raise PermissionError(
            f"{path} is not readable (torch_npu writes snapshots mode 0200); "
            f"run `chmod 600 {path}`, or dump through mem_hook which does it"
        ) from exc


def load_sidecar(path: Path) -> dict:
    """Read the allocator counters recorded beside the snapshot, if present.

    ``memory_reserved`` / ``memory_allocated`` / ``max_memory_allocated`` are
    live-process values that the snapshot itself does not carry; the recording
    hook writes them to ``<snapshot>.stats.json`` so the offline report can
    cross-check totals and name the peak.
    """
    sidecar = path.with_suffix(path.suffix + ".stats.json")
    if not sidecar.is_file():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _python_frames(frames: list | None) -> list[dict]:
    """Python frames only, in the recorded innermost-first order."""
    return [frame for frame in (frames or [])
            if isinstance(frame, dict) and str(frame.get("filename", "")).endswith(".py")]


def _named_native(frames: list | None) -> list[str]:
    """Native symbol names with the capture machinery filtered out."""
    out = []
    for frame in frames or []:
        name = frame.get("name") if isinstance(frame, dict) else None
        if name and not any(noise in name for noise in FRAME_NOISE):
            out.append(name)
    return out


def _format_frame(frame: dict) -> str:
    """``file.py:line func`` for one recorded frame."""
    filename = str(frame.get("filename", "?")).replace("\\", "/").rsplit("/", 1)[-1]
    return f"{filename}:{frame.get('line', '?')} {frame.get('name', '?')}"


def call_site(frames: list | None) -> str:
    """Attribute a block to the innermost frame that is the caller's own code.

    Falls back to the innermost framework Python frame, then to the innermost
    meaningful native symbol, then to ``unclassified``.
    """
    pythons = _python_frames(frames)
    for frame in pythons:
        if not any(marker in str(frame.get("filename", "")) for marker in SITE_MARKERS):
            return _format_frame(frame)
    if pythons:
        return _format_frame(pythons[0])
    native = _named_native(frames)
    meaningful = [name for name in native
                  if not any(name.startswith(prefix) for prefix in NATIVE_PLUMBING)]
    chosen = (meaningful or native or [CATEGORY_UNKNOWN])[0]
    return chosen.partition("(")[0][:80]


def categorize(frames: list | None) -> str:
    """Classify a block from its recorded frames.

    Rules, in order, each validated against a run whose composition was known:
    optimizer state (a ``torch/optim`` frame), parameter (the module ``.to()``
    conversion), gradient (``AccumulateGrad`` among the native frames, which is
    the only signal because gradients carry no Python frame), activation (the
    innermost own-code frame is inside a ``forward``).
    """
    pythons = _python_frames(frames)
    paths = " ".join(str(frame.get("filename", "")).replace("\\", "/") for frame in pythons)
    names = " ".join(str(frame.get("name", "")) for frame in pythons)
    if "/optim/" in paths:
        return "优化器状态"
    if "_module.py" in paths and ("convert" in names or " to" in f" {names}"):
        return "参数"
    if any(GRAD_SYMBOL in symbol for symbol in _named_native(frames)):
        return "梯度"
    own = [frame for frame in pythons
           if not any(marker in str(frame.get("filename", "")) for marker in SITE_MARKERS)]
    if own and "forward" in str(own[0].get("name", "")):
        return "激活"
    if own:
        return CATEGORY_OTHER
    return CATEGORY_UNKNOWN


def _active_blocks(snapshot: dict) -> list[tuple[int, dict]]:
    """Every active block as ``(size, block)``."""
    out = []
    for segment in snapshot.get("segments") or []:
        for block in segment.get("blocks") or []:
            if str(block.get("state", "")).startswith(ACTIVE_PREFIX):
                out.append((int(block.get("size", 0) or 0), block))
    return out


def summarize(snapshot: dict) -> dict:
    """Total reserved / allocated / fragmentation plus the per-category split.

    Segment totals are taken from the segment fields rather than re-summed, and
    the re-summed values are kept for the consistency checks.
    """
    segments = snapshot.get("segments") or []
    reserved = sum(int(seg.get("total_size", 0) or 0) for seg in segments)
    allocated = sum(int(seg.get("allocated_size", 0) or 0) for seg in segments)
    active = sum(int(seg.get("active_size", 0) or 0) for seg in segments)
    blocks = _active_blocks(snapshot)
    categories: dict[str, float] = {}
    rows = []
    for size, block in blocks:
        category = categorize(block.get("frames"))
        categories[category] = categories.get(category, 0.0) + size / MIB
        rows.append((size / MIB, category, call_site(block.get("frames"))))
    rows.sort(reverse=True)
    return {
        "segments": len(segments),
        "reserved_mib": reserved / MIB,
        "allocated_mib": allocated / MIB,
        "active_mib": active / MIB,
        "fragmentation_mib": max(0.0, (reserved - allocated) / MIB),
        "block_sum_mib": sum(size for size, _ in blocks) / MIB,
        "categories": categories,
        "top_blocks": rows[:20],
        "block_count": len(blocks),
    }


def run_checks(snapshot: dict, stats: dict, sidecar: dict) -> list[tuple[str, bool, str]]:
    """Validate the parse against the allocator's own invariants.

    The first two hold inside the snapshot and are always checked; the rest need
    the counters the recording hook wrote beside it.
    """
    checks: list[tuple[str, bool, str]] = []
    checks.append(_segment_identity(snapshot))
    checks.append(("Σ活跃块 = Σ段 allocated_size",
                   abs(stats["block_sum_mib"] - stats["allocated_mib"]) < 1e-6,
                   f"{stats['block_sum_mib']:.2f} vs {stats['allocated_mib']:.2f} MiB"))
    for key, label, value in (
            ("memory_reserved", "Σ段 total_size = memory_reserved()", stats["reserved_mib"]),
            ("memory_allocated", "Σ段 allocated_size = memory_allocated()",
             stats["allocated_mib"])):
        if key in sidecar:
            recorded = sidecar[key] / MIB
            checks.append((label, abs(recorded - value) < 1e-6,
                           f"{value:.2f} vs {recorded:.2f} MiB"))
    checks.append(_peak_check(stats, sidecar))
    return [check for check in checks if check is not None]


def _segment_identity(snapshot: dict) -> tuple[str, bool, str]:
    """Each segment's blocks must tile it exactly, and match its allocated_size."""
    bad = []
    for index, segment in enumerate(snapshot.get("segments") or []):
        blocks = segment.get("blocks") or []
        total = sum(int(block.get("size", 0) or 0) for block in blocks)
        live = sum(int(block.get("size", 0) or 0) for block in blocks
                   if str(block.get("state", "")).startswith(ACTIVE_PREFIX))
        if total != int(segment.get("total_size", 0) or 0):
            bad.append(f"seg{index} 块总和≠total_size")
        if live != int(segment.get("allocated_size", 0) or 0):
            bad.append(f"seg{index} 活跃块≠allocated_size")
    return ("逐段：Σ块 = total_size，Σ活跃块 = allocated_size", not bad,
            "全部符合" if not bad else "; ".join(bad[:4]))


def _peak_check(stats: dict, sidecar: dict) -> tuple[str, bool, str] | None:
    """State whether this snapshot is actually the peak, never assume it is."""
    if "max_memory_allocated" not in sidecar:
        return ("快照是否为峰值", False,
                "无 max_memory_allocated 记录，无法判断；快照仅代表取样瞬间")
    peak = sidecar["max_memory_allocated"] / MIB
    gap = peak - stats["allocated_mib"]
    near = abs(gap) <= peak * 0.02
    detail = (f"快照 {stats['allocated_mib']:.2f} MiB vs 峰值 {peak:.2f} MiB"
              f"，差 {gap:.2f} MiB")
    return ("快照是否为峰值（容差 2%）", near,
            detail + ("" if near else "——本报告的构成是取样瞬间的，不是峰值构成"))


def analyse(path: Path) -> dict:
    """Parse one snapshot into the full record."""
    snapshot = load_snapshot(path)
    sidecar = load_sidecar(path)
    stats = summarize(snapshot)
    return {"path": path, "stats": stats, "sidecar": sidecar,
            "checks": run_checks(snapshot, stats, sidecar)}


def compare(baseline: dict, later: dict) -> dict:
    """Diff two snapshots' category totals to expose growth across steps.

    A leak shows up as a category that keeps growing between steps, which a
    single snapshot cannot distinguish from a large one-off allocation.

    Args:
        baseline: ``summarize`` output for the earlier snapshot.
        later: ``summarize`` output for the later one.

    Returns:
        ``{"allocated_delta_mib": float, "reserved_delta_mib": float,
        "categories": {name: delta_mib}}``.
    """
    names = set(baseline["categories"]) | set(later["categories"])
    return {
        "allocated_delta_mib": later["allocated_mib"] - baseline["allocated_mib"],
        "reserved_delta_mib": later["reserved_mib"] - baseline["reserved_mib"],
        "categories": {name: later["categories"].get(name, 0.0)
                       - baseline["categories"].get(name, 0.0)
                       for name in sorted(names)},
    }


def _rows(cells: list[tuple]) -> str:
    """Render table rows from a list of cell tuples."""
    return "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in cells)


def _total_rows(report: dict) -> str:
    """Build the totals table."""
    stats = report["stats"]
    sidecar = report["sidecar"]
    peak = (f"{sidecar['max_memory_allocated'] / MIB:.1f}"
            if "max_memory_allocated" in sidecar else "[待解析]")
    return _rows([
        ("Reserved（Σ段 total_size）", f"{stats['reserved_mib']:.1f}", "分配器向设备保留"),
        ("Allocated（Σ段 allocated_size）", f"{stats['allocated_mib']:.1f}", "取样瞬间在用"),
        ("Active（Σ段 active_size）", f"{stats['active_mib']:.1f}",
         "含 pending-free；与 Allocated 相等表示无待释放块"),
        ("碎片（Reserved − Allocated）", f"{stats['fragmentation_mib']:.1f}",
         "保留但未在用"),
        ("峰值 max_memory_allocated", peak, "<b>与 Allocated 的差即快照未覆盖的部分</b>"),
    ])


def _category_rows(report: dict) -> str:
    """Build the per-category table."""
    categories = report["stats"]["categories"]
    if not categories:
        return '<tr><td colspan=3 class="todo">[待解析] 快照内无活跃块</td></tr>'
    total = sum(categories.values()) or 1.0
    ordered = sorted(categories.items(), key=lambda kv: kv[1], reverse=True)
    return _rows([(name, f"{mib:.1f}", f"{mib / total * 100:.1f}%") for name, mib in ordered])


def _block_rows(report: dict) -> str:
    """Build the largest-blocks table."""
    rows = report["stats"]["top_blocks"]
    if not rows:
        return '<tr><td colspan=3 class="todo">[待解析]</td></tr>'
    return _rows([(f"{mib:.2f}", category, site) for mib, category, site in rows])


def _check_rows(report: dict) -> str:
    """Build the invariant-check table."""
    return _rows([(name, "通过" if ok else '<span class="todo">不通过</span>', detail)
                  for name, ok, detail in report["checks"]])


def _verdict(report: dict) -> str:
    """Name the largest category and the largest single block."""
    stats = report["stats"]
    if not stats["categories"]:
        return '<span class="todo">[待解析]</span>'
    name, mib = max(stats["categories"].items(), key=lambda kv: kv[1])
    biggest = stats["top_blocks"][0]
    frag = stats["fragmentation_mib"]
    parts = [f"占比最大的类别是 <b>{name}</b>（{mib:.1f} MiB）",
             f"最大单块 {biggest[0]:.2f} MiB 来自 <code>{biggest[2]}</code>"]
    if frag > stats["reserved_mib"] * 0.1:
        parts.append(f"碎片 {frag:.1f} MiB 超过 Reserved 的 10%，值得先看分配器配置")
    return "；".join(parts)


def _compare_rows(report: dict) -> str:
    """Build the cross-snapshot growth table, or say none was requested."""
    diff = report.get("compare")
    if not diff:
        return ("<tr><td colspan=3>未比对（传 <code>--compare &lt;另一份快照&gt;</code> "
                "可查跨 step 增长，用于区分泄露与一次性峰值）</td></tr>")
    cells = [("Allocated 增量", f"{diff['allocated_delta_mib']:+.2f}", "两份快照之差"),
             ("Reserved 增量", f"{diff['reserved_delta_mib']:+.2f}", "分配器保留之差")]
    for name, delta in sorted(diff["categories"].items(), key=lambda kv: -abs(kv[1])):
        cells.append((f"类别 {name}", f"{delta:+.2f}",
                      "持续增长即疑似泄露" if delta > 0 else ""))
    return _rows(cells)


def render(report: dict, template: Path) -> str:
    """Fill the HTML template from a parsed snapshot record."""
    html = template.read_text(encoding="utf-8")
    stats = report["stats"]
    for marker, value in (
            ("<!--COMPARE_ROWS-->", _compare_rows(report)),
            ("<!--SOURCE-->", str(report["path"])),
            ("<!--SEGMENTS-->", f"{stats['segments']} 段 / {stats['block_count']} 活跃块"),
            ("<!--TOTAL_ROWS-->", _total_rows(report)),
            ("<!--CATEGORY_ROWS-->", _category_rows(report)),
            ("<!--BLOCK_ROWS-->", _block_rows(report)),
            ("<!--CHECK_ROWS-->", _check_rows(report)),
            ("<!--VERDICT-->", _verdict(report))):
        html = html.replace(marker, value)
    return html


def _summarize_line(report: dict) -> str:
    """One console line: the totals plus any failed invariant."""
    stats = report["stats"]
    failed = [name for name, ok, _ in report["checks"] if not ok]
    tail = f"; 校验未通过: {', '.join(failed)}" if failed else "; 校验全通过"
    return (f"reserved={stats['reserved_mib']:.0f}MiB "
            f"allocated={stats['allocated_mib']:.0f}MiB "
            f"frag={stats['fragmentation_mib']:.0f}MiB{tail}")


def main(argv: list[str] | None = None) -> int:
    """Parse the snapshot and write the memory-breakdown HTML report."""
    parser = argparse.ArgumentParser(
        prog="python .agent/skills/memory-analysis/scripts/breakdown.py",
        description="Break an NPU memory snapshot down into an HTML report.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--out", default="memory-breakdown.html")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="also write the parsed record as JSON")
    parser.add_argument("--compare", default=None,
                        help="an earlier snapshot to diff against, to expose growth "
                             "across steps (a leak) rather than a one-off peak")
    args = parser.parse_args(argv)

    path = Path(args.snapshot).resolve()
    if not path.is_file():
        print(f"[breakdown] snapshot not found: {path}", flush=True)
        return 1
    report = analyse(path)
    if args.compare:
        earlier = Path(args.compare).resolve()
        if not earlier.is_file():
            print(f"[breakdown] --compare snapshot not found: {earlier}", flush=True)
            return 1
        report["compare"] = compare(summarize(load_snapshot(earlier)), report["stats"])
        print(f"[breakdown] vs {earlier.name}: "
              f"allocated {report['compare']['allocated_delta_mib']:+.2f}MiB", flush=True)
    template = Path(__file__).resolve().parent.parent / "templates" / "memory-breakdown.html"
    Path(args.out).write_text(render(report, template), encoding="utf-8")
    if args.json_out:
        payload = {"path": str(path), "stats": report["stats"], "sidecar": report["sidecar"],
                   "compare": report.get("compare"),
                   "checks": [{"check": n, "passed": ok, "detail": d}
                              for n, ok, d in report["checks"]]}
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print(f"[breakdown] {_summarize_line(report)}; report={args.out}", flush=True)
    return 0 if all(ok for _, ok, _ in report["checks"]) else 2


if __name__ == "__main__":
    sys.exit(main())
