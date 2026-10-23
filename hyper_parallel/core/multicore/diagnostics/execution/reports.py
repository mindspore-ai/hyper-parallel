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
"""Offline coverage checks for collected execution-monitor reports."""

from __future__ import annotations

from collections.abc import Collection
import json
from pathlib import Path
from typing import Any


def _identity_directories(root: Path, world_size: int) -> tuple[dict[int, list[Path]], list[str]]:
    """Group report directories by the independently expected rank identities."""
    seen: dict[int, list[Path]] = {}
    invalid = []
    for identity_file in root.glob("rank_*/identity.json"):
        identity = json.loads(identity_file.read_text(encoding="utf-8"))
        rank = identity["rank"]
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
            invalid.append(str(identity_file))
            continue
        seen.setdefault(rank, []).append(identity_file.parent)
        if identity["world_size"] != world_size or identity.get("schema_version") != 1:
            invalid.append(str(identity_file))
    return seen, invalid


def _event_summary(row: dict[str, Any], rank: int) -> dict[str, Any]:
    """Summarize Host observations without treating nested CPU costs as step latency."""
    events = row.get("events", [])
    plans = [event for event in events if event["kind"] == "plan"]
    routes = [event for event in events if event["kind"] == "route"]
    return {
        "rank": rank,
        "plan_build_calls": sum(not event["cache_hit"] for event in plans),
        "plan_build_cpu_ms": sum(event["build_cpu_ms"] for event in plans),
        "maximum_owner_skew": max((event["maximum_over_mean"] for event in routes), default=None),
        "local_token_counts": sorted({event["local_tokens"] for event in events if event["kind"] == "moe"}),
    }


def _step_coverage(
    step: int, seen: dict[int, list[Path]],
) -> tuple[list[int], list[str], list[str], list[dict[str, Any]]]:
    """Read one requested step without accepting an absent or truncated report."""
    ranks, invalid, truncated, observations = [], [], [], []
    for rank, directories in seen.items():
        for directory in directories:
            path = directory / f"step_{step:06d}.json"
            if not path.is_file():
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if (row.get("step") != step or row.get("schema_version") != 1
                    or not isinstance(row.get("events"), list)):
                invalid.append(str(path))
                continue
            ranks.append(rank)
            observations.append(_event_summary(row, rank))
            if row.get("dropped_events", 0):
                truncated.append(str(path))
    return sorted(ranks), invalid, truncated, observations


def _expected_steps(world_size: int, expected_steps: Collection[int]) -> set[int]:
    """Validate coverage requirements independently of the available files."""
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    steps = set(expected_steps)
    if not steps or any(not isinstance(step, int) or isinstance(step, bool) or step < 0 for step in steps):
        raise ValueError("expected_steps must contain non-negative integer steps")
    return steps


def summarize_execution_reports(
    root: str | Path, *, world_size: int, expected_steps: Collection[int],
) -> dict[str, Any]:
    """Check every expected rank and step, rejecting duplicated or truncated samples.

    Args:
        root: Collected output directory from all training nodes.
        world_size: Expected total distributed rank count, supplied independently.
        expected_steps: Required zero-based sampled training steps.

    Returns:
        JSON-compatible coverage and Host-event summaries. Incomplete reports remain inspectable.
    """
    steps = _expected_steps(world_size, expected_steps)
    seen, invalid = _identity_directories(Path(root), world_size)
    coverage = {}
    observations = {}
    truncated = []
    for step in sorted(steps):
        coverage[str(step)], step_invalid, step_truncated, observations[str(step)] = _step_coverage(step, seen)
        invalid.extend(step_invalid)
        truncated.extend(step_truncated)
    expected_ranks = set(range(world_size))
    duplicates = sorted(rank for rank, directories in seen.items() if len(directories) != 1)
    complete = (set(seen) == expected_ranks and not duplicates and not invalid and not truncated
                and all(ranks == list(range(world_size)) for ranks in coverage.values()))
    return {"schema_version": 1, "complete": complete, "ranks": sorted(seen), "steps": coverage,
            "observations": observations,
            "duplicate_ranks": duplicates, "invalid_reports": invalid, "truncated_reports": truncated,
            "scope": "Host observations and coverage; not numerical or performance acceptance."}
