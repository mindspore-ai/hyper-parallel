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

Public-toolchain skeleton: it loads a snapshot (the public torch/torch_npu
``_snapshot`` shape: a dict with ``segments``), computes reserved vs active
totals and the fragmentation gap, lists the largest active blocks that
compose the peak, and fills the HTML template. Static/dynamic and phase
classification by call-stack is rule-based (see
``references/breakdown-dimensions.md``); parts needing an internal口径 are
left ``[待对齐]`` and reported as ``unclassified`` rather than guessed.

Pure stdlib. The snapshot is a file the recording already produced.
"""
from __future__ import annotations

import argparse
import json
import pickle  # nosec B403 - trusted local snapshot produced by our own run
import sys
from pathlib import Path

_MIB = 1024 * 1024


def _load_snapshot(path: Path) -> dict:
    """Load a snapshot from pickle or json."""
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("rb") as handle:
        return pickle.load(handle)  # nosec B301 - local, self-produced file


def _segments(snapshot: dict) -> list[dict]:
    """Return the segment list from a snapshot dict, defensively."""
    if isinstance(snapshot, dict):
        return snapshot.get("segments", []) or []
    return []


def summarize(snapshot: dict) -> dict:
    """Compute reserved / active / fragmentation and the largest active blocks."""
    reserved = 0
    active = 0
    blocks: list[tuple[int, str]] = []
    for segment in _segments(snapshot):
        reserved += int(segment.get("total_size", 0) or 0)
        for block in segment.get("blocks", []) or []:
            size = int(block.get("size", 0) or 0)
            if block.get("state") in ("active_allocated", "active", "active_pending_free"):
                active += size
                blocks.append((size, _origin(block)))
    blocks.sort(reverse=True)
    return {
        "reserved_mib": reserved / _MIB,
        "active_mib": active / _MIB,
        "fragmentation_mib": max(0.0, (reserved - active) / _MIB),
        "top_blocks": blocks[:15],
    }


def _origin(block: dict) -> str:
    """Best-effort attribution of a block from its recorded call frames."""
    frames = block.get("frames") or block.get("history") or []
    if frames and isinstance(frames, list):
        top = frames[0]
        if isinstance(top, dict):
            return top.get("name") or top.get("filename") or "<frame>"
    return "unclassified"


def _render(stats: dict, snapshot_path: Path, template: Path) -> str:
    """Fill the HTML template with the memory summary."""
    rows = "".join(
        f"<tr><td>{size / _MIB:.1f}</td><td>{origin}</td></tr>"
        for size, origin in stats["top_blocks"]
    )
    html = template.read_text(encoding="utf-8")
    return (html.replace("<!--SOURCE-->", str(snapshot_path))
                .replace("<!--RESERVED-->", f"{stats['reserved_mib']:.1f}")
                .replace("<!--ACTIVE-->", f"{stats['active_mib']:.1f}")
                .replace("<!--FRAG-->", f"{stats['fragmentation_mib']:.1f}")
                .replace("<!--TOP_BLOCKS-->", rows or "<tr><td colspan=2>[待解析]</td></tr>"))


def main(argv: list[str] | None = None) -> int:
    """Parse the snapshot and write the memory-breakdown HTML report."""
    parser = argparse.ArgumentParser(
        prog="python .agent/skills/memory-analysis/scripts/breakdown.py",
        description="Break an NPU memory snapshot down into an HTML report.",
    )
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--out", default="memory-breakdown.html")
    args = parser.parse_args(argv)

    snapshot_path = Path(args.snapshot).resolve()
    template = Path(__file__).resolve().parent.parent / "templates" / "memory-breakdown.html"
    stats = summarize(_load_snapshot(snapshot_path))
    Path(args.out).write_text(_render(stats, snapshot_path, template), encoding="utf-8")
    print(f"[breakdown] reserved={stats['reserved_mib']:.0f}MiB "
          f"active={stats['active_mib']:.0f}MiB frag={stats['fragmentation_mib']:.0f}MiB; "
          f"report={args.out}", flush=True)
    return 0 if stats["top_blocks"] else 1


if __name__ == "__main__":
    sys.exit(main())
