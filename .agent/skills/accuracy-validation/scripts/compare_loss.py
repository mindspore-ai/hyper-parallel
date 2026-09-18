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
"""Compare aligned baseline and candidate optimizer-step loss trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TIERS = {
    "strict": {"median": 3e-4, "p95": 1e-3, "maximum": 2e-3, "bias": 2e-4},
    "standard": {"median": 2e-3, "p95": 5e-3, "maximum": 1e-2, "bias": 2e-3},
    "relaxed": {"median": 5e-3, "p95": 1e-2, "maximum": 2e-2, "bias": 5e-3},
}


def _read_records(path: Path) -> Iterable[dict]:
    """Read JSONL or CSV loss records.

    Args:
        path: Input file ending in ``.jsonl`` or ``.csv``.

    Returns:
        An iterable of record dictionaries.

    Raises:
        ValueError: If the input suffix is unsupported or a JSONL line is invalid.
    """
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        return records
    raise ValueError(f"unsupported loss file {path}; expected .jsonl or .csv")


def _load_series(path: Path, step_key: str, loss_key: str) -> Dict[int, float]:
    """Load a unique optimizer-step to loss mapping.

    Args:
        path: JSONL or CSV input path.
        step_key: Field containing the optimizer-step index.
        loss_key: Field containing the globally normalized loss.

    Returns:
        Mapping from optimizer step to loss.

    Raises:
        ValueError: If fields are missing, values are invalid, or steps repeat.
    """
    series: Dict[int, float] = {}
    for record_index, record in enumerate(_read_records(path), start=1):
        try:
            step = int(record[step_key])
            loss = float(record[loss_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"record {record_index} in {path} must contain numeric {step_key!r} and {loss_key!r}"
            ) from exc
        if step in series:
            raise ValueError(f"duplicate optimizer step {step} in {path}")
        series[step] = loss
    if not series:
        raise ValueError(f"no loss records found in {path}")
    return series


def _percentile(values: List[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for sorted numeric values."""
    if not values:
        raise ValueError("cannot calculate a percentile for an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def compare_series(
    baseline: Dict[int, float],
    candidate: Dict[int, float],
    tier: str,
    epsilon: float,
) -> Tuple[dict, bool]:
    """Compare aligned loss mappings against a predeclared tolerance tier.

    Args:
        baseline: Baseline optimizer-step losses.
        candidate: Candidate optimizer-step losses.
        tier: One of ``strict``, ``standard``, or ``relaxed``.
        epsilon: Minimum denominator used for relative errors.

    Returns:
        Tuple of the JSON-serializable report and overall pass flag.

    Raises:
        ValueError: If steps differ, fewer than 20 steps are present, or the tier is unknown.
    """
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose from {sorted(TIERS)}")
    baseline_steps = set(baseline)
    candidate_steps = set(candidate)
    if baseline_steps != candidate_steps:
        missing = sorted(baseline_steps - candidate_steps)
        extra = sorted(candidate_steps - baseline_steps)
        raise ValueError(f"optimizer-step mismatch: missing_candidate={missing}, extra_candidate={extra}")
    steps = sorted(baseline_steps)
    if len(steps) < 20:
        raise ValueError(f"trajectory validation requires at least 20 aligned steps, got {len(steps)}")

    deltas = [candidate[step] - baseline[step] for step in steps]
    relative_errors = [
        abs(delta) / max(abs(baseline[step]), epsilon)
        for step, delta in zip(steps, deltas)
    ]
    mean_reference = statistics.fmean(abs(baseline[step]) for step in steps)
    observed = {
        "median": statistics.median(relative_errors),
        "p95": _percentile(relative_errors, 0.95),
        "maximum": max(relative_errors),
        "bias": abs(statistics.fmean(deltas)) / max(mean_reference, epsilon),
    }
    limits = TIERS[tier]
    checks = {name: observed[name] <= limit for name, limit in limits.items()}
    worst_index = max(range(len(relative_errors)), key=relative_errors.__getitem__)
    report = {
        "tier": tier,
        "steps": len(steps),
        "first_step": steps[0],
        "last_step": steps[-1],
        "observed": observed,
        "limits": limits,
        "checks": checks,
        "delta_signs": {
            "positive": sum(delta > 0 for delta in deltas),
            "negative": sum(delta < 0 for delta in deltas),
            "zero": sum(delta == 0 for delta in deltas),
        },
        "worst_step": {
            "step": steps[worst_index],
            "baseline": baseline[steps[worst_index]],
            "candidate": candidate[steps[worst_index]],
            "delta": deltas[worst_index],
            "relative_error": relative_errors[worst_index],
        },
        "passed": all(checks.values()),
    }
    return report, report["passed"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--tier", choices=sorted(TIERS), required=True)
    parser.add_argument("--step-key", default="step")
    parser.add_argument("--loss-key", default="loss")
    parser.add_argument("--epsilon", default=1e-8, type=float)
    return parser.parse_args()


def main() -> int:
    """Run the loss comparison command."""
    args = _parse_args()
    try:
        baseline = _load_series(args.baseline, args.step_key, args.loss_key)
        candidate = _load_series(args.candidate, args.step_key, args.loss_key)
        report, passed = compare_series(baseline, candidate, args.tier, args.epsilon)
    except (OSError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
