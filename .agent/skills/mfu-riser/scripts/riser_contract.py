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
"""Evidence contract for comparable training trials; no device or framework imports."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from collect_state import check_reports


CHECKS = ("real_weights", "all_ranks_complete", "finite_loss_grad", "input_equivalence",
          "numerical_equivalence", "independent_flops", "timing_boundary")
WORKLOAD_KEYS = ("model", "weights_id", "dataset_id", "optimizer", "precision", "routing",
                 "global_batch_size", "world_size", "nodes", "peak_tflops_per_device",
                 "flops_convention", "timing_scope", "train_steps", "window")
CAPACITY_FIELDS = {"sequence_length", "micro_batch_size", "global_batch_size", "gradient_accumulation_steps"}
PERFORMANCE_SCOPES = {"compute_fusion", "equivalent_implementation", "parallel_layout", "communication",
                      "activation_memory", "optimizer"}


def require(condition: bool, message: str) -> None:
    """Reject an invalid contract condition.

    Args:
        condition: Whether the contract condition holds.
        message: Diagnostic to raise when it fails.
    """
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    """Read JSON evidence.

    Args:
        path: Path to the UTF-8 JSON file.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def digest_json(value: object) -> str:
    """Hash a canonical JSON value with finite numbers.

    Args:
        value: JSON-serializable object.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def hash_file(path: Path) -> str:
    """Hash a file in bounded memory.

    Args:
        path: File to hash.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive(value: object, name: str) -> float:
    """Validate a finite positive number.

    Args:
        value: Numeric value from the contract.
        name: Field name used in diagnostics.
    """
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name}: number required")
    require(math.isfinite(value) and value > 0, f"{name}: finite positive number required")
    return float(value)


def _validate_workload(workload: dict, validation_level: str) -> None:
    for key in WORKLOAD_KEYS:
        require(key in workload and workload[key] not in (None, "", []), f"Missing workload.{key}")
    for key in ("world_size", "global_batch_size", "train_steps"):
        require(isinstance(workload[key], int) and not isinstance(workload[key], bool)
                and workload[key] > 0, f"Invalid {key}")
    positive(workload["peak_tflops_per_device"], "peak_tflops_per_device")
    window = workload["window"]
    require(isinstance(window, list) and len(window) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in window), "Invalid window")
    require(1 < window[0] <= window[1] <= workload["train_steps"], "Window must follow warmup and fit training")
    if validation_level == "acceptance":
        require(window[1] - window[0] >= 19, "Window must contain at least 20 steps for acceptance")


def _validate_performance_policy(policy: dict, workload: dict) -> None:
    require(isinstance(policy, dict), "performance_policy must be an object")
    require(policy.get("validation_level") in ("screening", "acceptance"), "Invalid validation_level")
    architecture = workload.get("model_config_sha256")
    require(isinstance(architecture, str) and re.fullmatch(r"[a-f0-9]{64}", architecture) is not None,
            "Performance policy requires workload.model_config_sha256")
    threshold = policy.get("low_mfu_threshold_percent")
    if threshold is not None:
        require(positive(threshold, "low_mfu_threshold_percent") <= 100, "Low MFU threshold exceeds 100%")
    tunable = policy.get("tunable_workload_fields", [])
    require(isinstance(tunable, list) and all(isinstance(key, str) and key in CAPACITY_FIELDS for key in tunable),
            "Invalid tunable_workload_fields")
    for key in CAPACITY_FIELDS.intersection(workload).union(tunable):
        value = workload.get(key)
        require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"Invalid workload.{key}")
    accounting = policy.get("flops_accounting")
    require(isinstance(accounting, dict) and accounting.get("method") in ("architecture_aware", "approximation"),
            "Specify an architecture-aware or audited approximation FLOPs method")
    if accounting["method"] == "approximation":
        tolerance = accounting.get("max_relative_error")
        require(isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool) and 0 <= tolerance < 1,
                "Approximation requires max_relative_error in [0, 1)")


def validate_config(config: dict) -> None:
    """Validate the workload, budget, and correctness commands before freezing them.

    Args:
        config: Proposed comparison-series configuration.
    """
    require(config.get("schema") == "mfu-riser/v1", "Unsupported run config schema")
    policy = config.get("performance_policy")
    if policy is not None:
        _validate_performance_policy(policy, config["workload"])
    _validate_workload(config["workload"], (policy or {}).get("validation_level", "acceptance"))
    for key in ("max_trials", "max_seconds", "command_timeout_seconds"):
        require(isinstance(config[key], int) and not isinstance(config[key], bool)
                and config[key] > 0, f"Invalid budget {key}")
    positive(config["target_mfu_percent"], "target_mfu_percent")
    noise = config["noise_fraction"]
    require(isinstance(noise, (int, float)) and 0 < noise < 0.25, "noise_fraction must be in (0, 0.25)")
    for phase in ("precheck", "benchmark", "profile", "validate", "cleanup"):
        require(phase in config["commands"], f"Missing command {phase}")
        command = config["commands"][phase]
        require(isinstance(command, list) and command and all(isinstance(x, str) and x for x in command),
                f"{phase} must be a nonempty argv list")


def series_transition(parent: dict, child: dict) -> dict:
    """Validate a transition to a linked capacity or acceptance series.

    Args:
        parent: Frozen configuration of the measured parent series.
        child: Configuration for the new series.
    """
    original, requested = parent.get("performance_policy"), child.get("performance_policy")
    require(original is not None and requested is not None, "Linked series require performance_policy in both runs")
    require(original["flops_accounting"] == requested["flops_accounting"], "Linked series changed FLOPs accounting")
    tunable = set(original.get("tunable_workload_fields", []))
    require(set(requested.get("tunable_workload_fields", [])) <= tunable, "Linked series expanded workload tuning")
    before, after = parent["workload"], child["workload"]
    changed = {key for key in before.keys() | after.keys()
               if key not in before or key not in after or before[key] != after[key]}
    if original["validation_level"] == requested["validation_level"]:
        require(bool(changed) and changed <= tunable, "Capacity series changed fields outside declared workload tuning")
        kind = "capacity"
    else:
        require(original["validation_level"] == "screening" and requested["validation_level"] == "acceptance",
                "Validation transition must advance from screening to acceptance")
        require(changed <= {"window", "train_steps"}, "Acceptance series must use the screened workload")
        kind = "acceptance"
    return {"kind": kind, "changed_workload_fields": sorted(changed)}


def evidence_path(root: Path, relative: str) -> Path:
    """Resolve a file inside the current trial.

    Args:
        root: Trial evidence directory.
        relative: Evidence path supplied by the validator.
    """
    require(isinstance(relative, str) and bool(relative), "Evidence path required")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), f"Missing/local evidence required: {relative}")
    return path


def _verified_evidence(root: Path, receipt: dict, name: str, artifacts: dict) -> Path:
    path = evidence_path(root, receipt.get("evidence"))
    digest = hash_file(path)
    require(digest == receipt.get("sha256"), f"Evidence hash mismatch: {name}")
    artifacts[str(path.relative_to(root))] = digest
    return path


def _audit_model_flops(path: Path, samples: list, context: dict) -> None:
    audit = read_json(path)
    config = context["config"]
    workload = config["workload"]
    accounting = config["performance_policy"]["flops_accounting"]
    require(isinstance(audit, dict) and audit.get("trial_token") == context["trial_token"]
            and audit.get("source_digest") == context["source_digest"], "FLOPs audit belongs to a different trial")
    require(audit.get("model_config_sha256") == workload["model_config_sha256"]
            and audit.get("convention") == workload["flops_convention"], "FLOPs audit changed model or convention")
    steps = audit.get("steps")
    require(isinstance(steps, list) and len(steps) == len(samples), "Incomplete FLOPs audit window")
    for measured, audited in zip(samples, steps):
        require(isinstance(audited, dict) and audited.get("step") == measured["step"]
                and audited.get("model_tokens") == measured["model_tokens"], "FLOPs audit step/token mismatch")
        components = audited.get("components")
        require(isinstance(components, dict) and bool(components), "Missing architecture FLOPs components")
        reference_flops = math.fsum(positive(value, name) for name, value in components.items())
        accounted = reference_flops
        if accounting["method"] == "approximation":
            accounted = positive(audited.get("approximate_model_flops"), "approximate_model_flops")
            relative_error = abs(accounted - reference_flops) / reference_flops
            require(relative_error <= accounting["max_relative_error"], "MFU approximation exceeds declared error")
        require(math.isclose(measured["model_flops"], accounted, rel_tol=1e-9), "Model FLOPs disagree with audit")


def _executed_work_metrics(receipt: dict, samples: list, peak: float, root: Path, artifacts: dict) -> dict:
    require(isinstance(receipt, dict) and receipt.get("method") in ("hardware_counters", "estimate"),
            "Specify executed work measurement method")
    coverage = receipt.get("coverage")
    require(isinstance(coverage, str) and bool(coverage.strip()), "Specify executed work coverage")
    _verified_evidence(root, receipt, "executed_work", artifacts)
    measurements = receipt.get("samples")
    require(isinstance(measurements, list) and bool(measurements), "Missing executed work samples")
    times = {row["step"]: row["time_s"] for row in samples}
    seen = set()
    flops = seconds = 0.0
    for row in measurements:
        require(isinstance(row, dict), "Invalid executed work sample")
        step = row.get("step")
        require(isinstance(step, int) and not isinstance(step, bool) and step in times and step not in seen,
                "Invalid or duplicate executed work step")
        seen.add(step)
        seconds += times[step]
        flops += positive(row.get("executed_flops"), "executed_flops")
    return {"hfu_percent": flops / (seconds * peak) * 100, "hfu_method": receipt["method"],
            "hfu_coverage": coverage, "hfu_steps": sorted(seen)}


def _validate_checks(data: dict, context: dict, root: Path) -> dict:
    config, plan = context["config"], context["plan"]
    policy = config.get("performance_policy")
    required = list(CHECKS)
    if policy is not None:
        require(data.get("validation_level") == policy["validation_level"], "Validation level changed")
        required.append("model_semantics")
        if policy["validation_level"] == "screening":
            required.remove("numerical_equivalence")
            required.append("implementation_correctness")
    if plan["actions"]:
        if policy is None or plan.get("operator_scope", "compute_fusion") == "compute_fusion":
            required += ["fusion_active", "fallback_correct"]
        else:
            required.append("candidate_active")
    artifacts = {}
    checks = data.get("checks", {})
    for key in required:
        check = checks.get(key, {})
        require(isinstance(check, dict) and check.get("passed") is True, f"Failed/unknown check: {key}")
        _verified_evidence(root, check, key, artifacts)
    return artifacts


def _validate_profile(data: dict, context: dict, root: Path, artifacts: dict) -> None:
    profile_path = evidence_path(root, data.get("profile_state"))
    profile = read_json(profile_path)
    require(profile.get("schema") == "mfu-riser-state/v1" and profile.get("valid") is True,
            "Missing/invalid profiling state")
    binding = profile.get("measurement_binding") or {}
    require(binding.get("trial_token") == context["trial_token"]
            and binding.get("source_digest") == context["source_digest"],
            "Profiling belongs to an old/different trial or has no worker binding")
    require(profile.get("profiling") and profile.get("operators"), "Empty profiling facts")
    require(not check_reports(profile["profiling"]), "Invalid nested profiling checks")
    require(all(row.get("status") == "PASS" and row.get("hotspots") for row in profile["operators"]),
            "No observed operator hotspots")
    artifacts[str(profile_path.relative_to(root))] = hash_file(profile_path)


def validate_measurement(data: dict, context: dict, root: Path, reference: dict | None) -> dict:
    """Validate all-rank evidence and recompute MFU from actual model work.

    The frozen project validator produces the named domain checks from raw logs.
    The controller verifies their integrity, identity, and arithmetic.

    Args:
        data: Measurement produced by the validator.
        context: Frozen trial context with the workload and source digest.
        root: Directory containing this trial's evidence.
        reference: Best metrics in this comparison series, if available.
    """
    config = context["config"]
    workload = config["workload"]
    require(data.get("schema") == "mfu-riser-measurement/v1", "Unsupported measurement schema")
    require(data.get("trial_token") == context["trial_token"], "Stale trial token")
    require(data.get("workload") == workload, "Workload/metric contract changed")
    require(data.get("profiler_during_benchmark") is False, "Benchmark must run without profiler")
    runtime = data.get("runtime", [])
    require(len(runtime) == workload["world_size"], "Incomplete runtime rank coverage")
    require(sorted(row["rank"] for row in runtime) == list(range(workload["world_size"])), "Duplicate/missing ranks")
    for row in runtime:
        require(row.get("source_digest") == context["source_digest"], "Worker measured stale/different source")
        require(row.get("steps_completed") == workload["train_steps"], "Incomplete training")
    artifacts = _validate_checks(data, context, root)
    _validate_profile(data, context, root, artifacts)
    samples = data["samples"]
    first, last = workload["window"]
    require([row["step"] for row in samples] == list(range(first, last + 1)), "Incomplete/duplicate metric window")
    peak = workload["world_size"] * workload["peak_tflops_per_device"] * 1e12
    duration = model_tokens = model_flops = 0.0
    for index, row in enumerate(samples):
        time_s = positive(row["time_s"], "time_s")
        tokens = positive(row["model_tokens"], "model_tokens")
        flops = positive(row["model_flops"], "model_flops")
        expected = flops / (time_s * peak) * 100
        require(math.isclose(positive(row["mfu_percent"], "mfu_percent"), expected, rel_tol=1e-6),
                "Logged MFU disagrees with model FLOPs/time/peak")
        if reference:
            before = reference["samples"][index]
            require(tokens == before["model_tokens"] and math.isclose(flops, before["model_flops"], rel_tol=1e-9),
                    "Model work/tokens changed within the comparison series")
        duration += time_s
        model_tokens += tokens
        model_flops += flops
    if config.get("performance_policy") is not None:
        audit = evidence_path(root, data["checks"]["independent_flops"]["evidence"])
        _audit_model_flops(audit, samples, context)
    tflops = model_flops / duration / workload["world_size"] / 1e12
    throughput = model_tokens / duration
    metrics = {"mfu_percent": tflops / workload["peak_tflops_per_device"] * 100, "tokens_per_second": throughput,
               "tokens_per_second_per_device": throughput / workload["world_size"], "model_tflops_per_device": tflops,
               "mean_step_seconds": duration / len(samples), "samples": samples, "artifacts": artifacts}
    if data.get("executed_work") is not None:
        metrics.update(_executed_work_metrics(data["executed_work"], samples, peak, root, artifacts))
    if data.get("allocation") is not None:
        allocation = data["allocation"]
        require(isinstance(allocation, dict), "Invalid allocation receipt")
        _verified_evidence(root, allocation, "allocation", artifacts)
        metrics["allocated_device_seconds"] = positive(allocation.get("device_seconds"), "allocated_device_seconds")
    return metrics


def decide(metrics: dict, best: dict | None, config: dict, kind: str) -> str:
    """Compare a trial with the retained result under the declared noise band.

    Args:
        metrics: Validated metrics for the trial.
        best: Retained result, or None before the baseline.
        config: Frozen configuration with the noise fraction.
        kind: Baseline, candidate, or confirmation.
    """
    if best is None:
        return "baseline"
    ratio = metrics["mfu_percent"] / best["metrics"]["mfu_percent"]
    if kind == "confirm":
        if abs(ratio - 1) <= config["noise_fraction"]:
            return "confirmed"
        return "unstable"
    if ratio > 1 + config["noise_fraction"]:
        return "keep"
    return "discard"
