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
"""Estimate gradient accumulation scaling from comparable, validated MFU Riser trials."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from riser_contract import positive, read_json, require, step_time_budget


def _count(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name}: positive integer required")
    return value


def _read_trial(directory: Path) -> tuple[dict, dict]:
    result = read_json(directory / "result.json")
    context = read_json(directory / "context.json")
    metrics = result.get("metrics")
    require(result.get("cleanup") == "PASS" and result.get("decision") != "crash" and metrics,
            f"Expected a completed trial with validated metrics: {directory}")
    require(result["id"] == directory.name and result["source_digest"] == context["source_digest"],
            f"Trial and source identity disagree: {directory}")
    config = context["config"]
    workload = dict(config["workload"])
    count = _count(workload.pop("gradient_accumulation_steps"), "gradient_accumulation_steps")
    batch = _count(workload.pop("global_batch_size"), "global_batch_size")
    _count(workload["micro_batch_size"], "micro_batch_size")
    _count(workload["sequence_length"], "sequence_length")
    require(batch % count == 0, "Global batch must scale with the number of accumulated microbatches")
    workload["global_batch_per_microbatch"] = batch // count
    samples = metrics["samples"]
    first, last = workload["window"]
    require(bool(samples) and [row["step"] for row in samples] == list(range(first, last + 1)),
            f"Incomplete timing window: {directory}")
    for row in samples:
        require(row["model_tokens"] == samples[0]["model_tokens"]
                and math.isclose(row["model_flops"], samples[0]["model_flops"], rel_tol=1e-9),
                f"Variable per-update token or FLOP work: {directory}")
    policy = config.get("performance_policy") or {}
    scope = {"source_digest": context["source_digest"], "workload": workload,
             "flops_accounting": policy.get("flops_accounting"),
             "validation_level": result.get("validation_level")}
    observation = {"microbatches": count, "mean_step_seconds": metrics["mean_step_seconds"],
                   "mfu_percent": metrics["mfu_percent"],
                   "tokens_per_second_per_device": metrics["tokens_per_second_per_device"],
                   "trial": str(directory)}
    return scope, observation


def estimate(observations: list[dict], probes: list[int], target_mfu_percent: float,
             max_relative_time_error: float) -> dict:
    """Fit an update-time model to comparable observations and replay independent points.

    Args:
        observations: Unprofiled observations with microbatch count, mean step time,
            MFU percent, and effective tokens per second per device. The caller must
            establish comparable source, workload, hardware, and timing scope.
        probes: Authorized microbatch counts to estimate, above two.
        target_mfu_percent: MFU objective under the observations' accounting convention.
        max_relative_time_error: Predeclared held-out timing error tolerance, as a fraction.

    Returns:
        Fit coefficients, held-out residuals, and provisional probe estimates.
    """
    target = positive(target_mfu_percent, "target_mfu_percent")
    tolerance = max_relative_time_error
    require(isinstance(tolerance, (float, int)) and not isinstance(tolerance, bool) and 0 <= tolerance < 1,
            "max_relative_time_error must be in [0, 1)")
    by_count = {_count(row["microbatches"], "microbatches"): row for row in observations}
    require(len(by_count) == len(observations) and {1, 2} <= by_count.keys(),
            "Provide unique observations including one and two microbatches")
    require(all(_count(count, "probe") > 2 for count in probes), "Probes must exceed two microbatches")
    seconds = {count: positive(row["mean_step_seconds"], "mean_step_seconds") for count, row in by_count.items()}
    per_microbatch_mfu_seconds = positive(by_count[1]["mfu_percent"], "mfu_percent") * seconds[1]
    per_microbatch_tokens = positive(by_count[1]["tokens_per_second_per_device"], "tokens_per_second_per_device")
    per_microbatch_tokens *= seconds[1]
    for count, row in sorted(by_count.items()):
        seconds_per_microbatch = seconds[count] / count
        require(math.isclose(positive(row["mfu_percent"], "mfu_percent") * seconds_per_microbatch,
                             per_microbatch_mfu_seconds, rel_tol=1e-6),
                f"Model work per microbatch changed at {count} microbatches")
        require(math.isclose(positive(row["tokens_per_second_per_device"], "tokens_per_second_per_device")
                             * seconds_per_microbatch, per_microbatch_tokens, rel_tol=1e-6),
                f"Effective tokens per microbatch changed at {count} microbatches")

    marginal = seconds[2] - seconds[1]
    fixed = seconds[1] - marginal
    require(fixed >= 0 and marginal > 0, "The two measurements do not support a fixed-cost amortization model")
    checks = []
    for count in sorted(by_count.keys() - {1, 2}):
        predicted = fixed + marginal * count
        error = (seconds[count] - predicted) / predicted
        checks.append({"microbatches": count, "predicted_mean_step_seconds": predicted,
                       "observed_mean_step_seconds": seconds[count],
                       "relative_time_error": error, "within_tolerance": abs(error) <= tolerance})
    estimates = []
    if all(row["within_tolerance"] for row in checks):
        for count in sorted(set(probes)):
            predicted = fixed + marginal * count
            estimated_mfu = per_microbatch_mfu_seconds * count / predicted
            target_seconds, savings = step_time_budget(predicted, estimated_mfu, target)
            estimates.append({"microbatches": count, "predicted_mean_step_seconds": predicted,
                              "estimated_mfu_percent": estimated_mfu,
                              "estimated_tokens_per_second_per_device": per_microbatch_tokens * count / predicted,
                              "target_step_budget_seconds": target_seconds,
                              "required_step_savings_seconds": savings})
    return {"schema": "mfu-riser-accumulation-estimate/v1", "fixed_seconds": fixed,
            "seconds_per_microbatch": marginal, "modeled_mfu_limit_percent": per_microbatch_mfu_seconds / marginal,
            "target_mfu_percent": target, "max_relative_time_error": tolerance, "observations": observations,
            "independent_checks": checks, "probe_estimates": estimates}


def main(argv: list[str] | None = None) -> int:
    """Read completed trial directories and print a planning estimate as JSON.

    Args:
        argv: Optional command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", nargs="+", type=Path, help="Completed MFU Riser trial directories")
    parser.add_argument("--probe", type=int, action="append", default=[], help="Accumulated microbatches to estimate")
    parser.add_argument("--target-mfu-percent", type=float, required=True)
    parser.add_argument("--max-relative-time-error", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        scopes, observations = zip(*(_read_trial(path) for path in args.trials))
        require(all(scope == scopes[0] for scope in scopes[1:]),
                "Calibration trials changed source, workload, hardware, or measurement scope beyond accumulation")
        report = estimate(list(observations), args.probe, args.target_mfu_percent, args.max_relative_time_error)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
