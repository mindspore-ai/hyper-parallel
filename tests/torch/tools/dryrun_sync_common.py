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
"""Dry-run vs real-training memory reconciliation (no test_ prefix: helpers).

The reconciliation compares the dry-run logical snapshot against the real
allocator snapshot of the same recipe:

1. Peak ratio: the logical peak must stay within a tolerance band of the real
   trace-level peak (allocator reserve excluded by replaying alloc/free events).
2. Site census: every large allocation call site observed in the real trace
   must appear in the dry-run trace with the same block sizes. This is the
   sensitive check — a divergence in trainer-loop replay (for example a missed
   reduce-scatter scheduling branch) shows up here as a named ``file:line``
   before it is visible in the peak.
"""
from __future__ import annotations

import glob
import pickle
from collections import Counter
from typing import Any

#: Blocks below this size are noise for divergence hunting.
LARGE_BLOCK_BYTES = 32 * 2**20

#: Logical peak must stay within this band of the real trace peak.
PEAK_RATIO_BAND = (0.80, 1.10)

#: Value-dependency replay owns these modules: their tensor sizes depend on
#: runtime values (document boundaries, routing loads) that dry-run replays
#: with balanced/synthetic equivalents, so per-site size census is expected to
#: differ there. The peak-band check still covers their aggregate. Everything
#: outside this list — FSDP, communication, optimizer — must census-match:
#: a mismatch there is a trainer-loop replay divergence.
VALUE_REPLAY_WAIVED_MODULES = (
    "components/modules/shared_compressed_dsa_attention.py",
    "components/modules/engram.py",
    "components/functional/",
)


def load_snapshot(path: str) -> dict[str, Any]:
    """Load one allocator-style snapshot pickle."""
    with open(path, "rb") as snapshot_file:
        return pickle.load(snapshot_file)


def latest_snapshot(directory: str) -> dict[str, Any]:
    """Load the newest snapshot pickle in a directory."""
    files = sorted(glob.glob(f"{directory}/*.pickle"))
    if not files:
        raise FileNotFoundError(f"no snapshot pickle under {directory}")
    return load_snapshot(files[-1])


def trace_peak_bytes(snapshot: dict[str, Any]) -> int:
    """Replay alloc/free events and return the trace-level peak."""
    peak = 0
    current = 0
    for trace in snapshot["device_traces"]:
        for event in trace:
            if event["action"] == "alloc":
                current += event["size"]
                peak = max(peak, current)
            elif event["action"] == "free_completed":
                current -= event["size"]
    return peak


def _allocation_site(event: dict[str, Any]) -> str:
    """Return the first project-owned python frame of one allocation.

    Line numbers are bucketed (//8) because the native allocator and the
    logical tracker capture a multi-line expression on slightly different
    lines; the bucket keeps the site discriminating without false negatives.
    """
    for frame in event.get("frames", ()):
        filename = frame.get("filename", "")
        if not filename.endswith(".py"):
            continue
        if "hyper_parallel" not in filename:
            continue
        if any(skip in filename for skip in ("tools/dryrun", "trainer/")):
            continue
        line_bucket = int(frame.get("line", 0)) // 8
        return f"{filename.split('hyper_parallel/')[-1]}:~{line_bucket * 8}"
    return "(anonymous)"


def large_allocation_census(snapshot: dict[str, Any]) -> Counter:
    """Count large allocations keyed by (site, size bucket).

    Sizes are bucketed to 1 MiB because the native allocator pads requests
    (for example 32 KiB NPU alignment) while the logical tracker records the
    exact storage bytes; the bucket absorbs padding without hiding a missing
    buffer class.
    """
    census: Counter = Counter()
    for trace in snapshot["device_traces"]:
        for event in trace:
            if event["action"] == "alloc" and event["size"] >= LARGE_BLOCK_BYTES:
                size_bucket = round(event["size"] / 2**20)
                census[(_allocation_site(event), size_bucket)] += 1
    return census


def reconcile(dry_snapshot: dict[str, Any], real_snapshot: dict[str, Any]) -> list[str]:
    """Return a list of human-readable divergences (empty when reconciled).

    The real allocator snapshot typically spans several optimizer steps
    (``memory.start_step``..``end_step``) while dry-run simulates exactly one,
    so call-site counts are compared per step: the real census is divided by
    the smallest ratio that makes the most common sites match. A genuine replay
    divergence breaks per-site proportionality and still fails.
    """
    problems: list[str] = []

    dry_peak = int(dry_snapshot["summary"]["peak_bytes"])
    real_peak = trace_peak_bytes(real_snapshot)
    ratio = dry_peak / max(real_peak, 1)
    low, high = PEAK_RATIO_BAND
    if not low <= ratio <= high:
        problems.append(
            f"peak ratio {ratio:.3f} outside [{low}, {high}]: "
            f"dry={dry_peak / 2**30:.3f} GiB real={real_peak / 2**30:.3f} GiB"
        )

    dry_census = large_allocation_census(dry_snapshot)
    real_census = large_allocation_census(real_snapshot)
    steps = _estimate_real_steps(dry_census, real_census)
    for (site, size_bucket), count in sorted(real_census.items(), key=lambda item: -item[0][1]):
        if site == "(anonymous)":
            continue
        if any(waived in site for waived in VALUE_REPLAY_WAIVED_MODULES):
            continue
        per_step = count / steps
        dry_count = dry_census.get((site, size_bucket), 0)
        if dry_count < per_step:
            problems.append(
                f"real trace has {per_step:g}x/step ~{size_bucket} MiB at {site}, "
                f"dry-run only {dry_count}x — trainer-loop replay divergence"
            )
    return problems


def _estimate_real_steps(dry_census: Counter, real_census: Counter) -> int:
    """Estimate how many optimizer steps the real snapshot window spans.

    Uses the modal real/dry count ratio over sites both traces share. Falls
    back to 1 (strictest comparison) when no site is shared.
    """
    ratios: Counter = Counter()
    for key, real_count in real_census.items():
        dry_count = dry_census.get(key, 0)
        if dry_count > 0 and real_count % dry_count == 0:
            ratios[real_count // dry_count] += 1
    if not ratios:
        return 1
    return max(ratios.most_common(1)[0][0], 1)
