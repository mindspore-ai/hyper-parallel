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
"""Comparators across validation-case evidence directories."""
# pylint: disable=forbidden-backend-import

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch

from hyper_parallel.tools.model_integration.schemas import tolerance_value


def _optional_tolerance(tolerance: Mapping[str, Any], name: str) -> Optional[float]:
    """Return one declared tolerance without inventing an absent zero bound."""
    value = tolerance.get(name)
    return None if value is None else float(value)


def _relative_error(reference: float, candidate: float) -> float:
    """Return reference-relative error with a finite zero-safe denominator."""
    return abs(candidate - reference) / max(abs(reference), 1.0e-30)


def _combine_tolerance_checks(
        checks: Iterable[bool],
        combination: Any,
        *,
        owner: str,
) -> bool:
    """Combine explicitly declared numerical checks with validated semantics."""
    if combination not in ("all", "any"):
        raise ValueError(
            f"{owner} tolerance combination must be 'all' or 'any', got "
            f"{combination!r}"
        )
    resolved_checks = tuple(checks)
    if not resolved_checks:
        raise ValueError(f"{owner} tolerance must declare at least one numerical bound")
    return all(resolved_checks) if combination == "all" else any(resolved_checks)


def _scalar_metric_delta(
        reference: float,
        candidate: float,
        tolerance: Mapping[str, Any],
        prefix: str,
) -> dict[str, Any]:
    """Compare one scalar using explicitly declared absolute/relative bounds."""
    max_abs = abs(candidate - reference)
    max_rel = _relative_error(reference, candidate)
    max_abs_limit = _optional_tolerance(tolerance, f"{prefix}_max_abs")
    max_rel_limit = _optional_tolerance(tolerance, f"{prefix}_max_rel")
    if max_abs_limit is None and max_rel_limit is None:
        # Preserve the schema-v1 contract: an omitted scalar policy means
        # exact absolute equality, rather than silently disabling the gate.
        max_abs_limit = 0.0
    checks = []
    if max_abs_limit is not None:
        checks.append(max_abs <= max_abs_limit)
    if max_rel_limit is not None:
        checks.append(max_rel <= max_rel_limit)
    combination = tolerance.get(
        f"{prefix}_combination",
        tolerance.get("combination", "all"),
    )
    return {
        "status": _combine_tolerance_checks(
            checks,
            combination,
            owner=f"{prefix} scalar",
        ),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "combination": combination,
    }


def compare_scalar_curves(
    baseline_rows: Iterable[Mapping[str, Any]],
    candidate_rows: Iterable[Mapping[str, Any]],
    tolerance: Mapping[str, Any],
    *,
    allow_candidate_subset: bool = False,
) -> dict[str, Any]:
    """Compare aligned loss/norm/LR rows without parsing Trainer text logs."""
    baseline = list(baseline_rows)
    candidate = list(candidate_rows)
    if not baseline or not candidate:
        return {
            "status": "BLOCKED",
            "reason": "structured scalar evidence is missing or empty",
        }
    if allow_candidate_subset:
        baseline_by_step = {row.get("step"): row for row in baseline}
        missing_steps = [row.get("step") for row in candidate if row.get("step") not in baseline_by_step]
        if missing_steps:
            return {
                "status": "FAIL",
                "reason": f"resume steps are absent from uninterrupted baseline: {missing_steps}",
            }
        baseline = [baseline_by_step[row.get("step")] for row in candidate]
    elif len(baseline) != len(candidate):
        return {
            "status": "FAIL",
            "reason": f"step count differs: baseline={len(baseline)}, candidate={len(candidate)}",
        }
    comparisons = []
    status = "PASS"
    for baseline_row, candidate_row in zip(baseline, candidate):
        baseline_loss = float(baseline_row["loss"])
        candidate_loss = float(candidate_row["loss"])
        baseline_norm = float(baseline_row["grad_norm_pre_clip"])
        candidate_norm = float(candidate_row["grad_norm_pre_clip"])
        loss_delta = _scalar_metric_delta(
            baseline_loss,
            candidate_loss,
            tolerance,
            "loss",
        )
        norm_delta = _scalar_metric_delta(
            baseline_norm,
            candidate_norm,
            tolerance,
            "norm",
        )
        baseline_input_hash = baseline_row.get("global_input_sha256")
        candidate_input_hash = candidate_row.get("global_input_sha256")
        input_equal = (
            baseline_input_hash is not None
            and candidate_input_hash is not None
            and baseline_input_hash == candidate_input_hash
        )
        baseline_lr = baseline_row.get("lr")
        candidate_lr = candidate_row.get("lr")
        learning_rate_equal = (
            baseline_lr is not None
            and candidate_lr is not None
            and baseline_lr == candidate_lr
        )
        baseline_step = baseline_row.get("step")
        candidate_step = candidate_row.get("step")
        step_equal = (
            baseline_step is not None
            and candidate_step is not None
            and baseline_step == candidate_step
        )
        baseline_post_norm = baseline_row.get("grad_norm_post_clip")
        candidate_post_norm = candidate_row.get("grad_norm_post_clip")
        post_norm_delta = None
        post_norm_relative_delta = None
        post_norm_equal = baseline_post_norm is None and candidate_post_norm is None
        if baseline_post_norm is not None and candidate_post_norm is not None:
            post_norm_comparison = _scalar_metric_delta(
                float(baseline_post_norm),
                float(candidate_post_norm),
                tolerance,
                "norm",
            )
            post_norm_delta = post_norm_comparison["max_abs"]
            post_norm_relative_delta = post_norm_comparison["max_rel"]
            post_norm_equal = post_norm_comparison["status"]
        finite = all(
            math.isfinite(float(value))
            for value in (
                baseline_row["loss"],
                candidate_row["loss"],
                baseline_row["grad_norm_pre_clip"],
                candidate_row["grad_norm_pre_clip"],
            )
        )
        numerical_pass = (
            loss_delta["status"]
            and norm_delta["status"]
            and post_norm_equal
            and learning_rate_equal
            and step_equal
        )
        if not finite:
            row_status = "FAIL"
        elif not input_equal:
            # Different logical inputs make a numerical A/B verdict invalid;
            # the topology is not proven wrong, but this experiment is blocked.
            row_status = "BLOCKED"
        else:
            row_status = "PASS" if numerical_pass else "FAIL"
        if row_status == "FAIL":
            status = "FAIL"
        elif row_status == "BLOCKED" and status == "PASS":
            status = "BLOCKED"
        comparisons.append(
            {
                "step": baseline_row.get("step"),
                "status": row_status,
                "loss_max_abs": loss_delta["max_abs"],
                "loss_max_rel": loss_delta["max_rel"],
                "loss_tolerance_combination": loss_delta["combination"],
                "norm_max_abs": norm_delta["max_abs"],
                "norm_max_rel": norm_delta["max_rel"],
                "norm_tolerance_combination": norm_delta["combination"],
                "post_clip_norm_max_abs": post_norm_delta,
                "post_clip_norm_max_rel": post_norm_relative_delta,
                "input_identity": input_equal,
                "learning_rate_equal": learning_rate_equal,
                "step_equal": step_equal,
                "finite": finite,
            }
        )
    return {"status": status, "steps": comparisons}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a structured JSON-lines evidence stream."""
    input_path = Path(path)
    if not input_path.is_file():
        return []
    return [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line]


def _nested_value(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for name in path.split("."):
        if not isinstance(current, Mapping) or name not in current:
            return None
        current = current[name]
    return current


def compare_preflight_configs(
        baseline_dir: str | Path,
        candidate_dir: str | Path,
) -> dict[str, Any]:
    """Prove that topology cases retain one training/precision contract."""
    baseline_path = Path(baseline_dir) / "trainer_config.resolved.json"
    candidate_path = Path(candidate_dir) / "trainer_config.resolved.json"
    if not baseline_path.is_file() or not candidate_path.is_file():
        return {"status": "BLOCKED", "reason": "resolved Trainer config evidence is missing"}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    invariant_paths = (
        "model_init_dtype",
        "training.global_batch_size",
        "training.max_grad_norm",
        "training.loss_aggregation",
        "training.seed",
        "fsdp_config.mix_precision.param_dtype",
        "fsdp_config.mix_precision.reduce_dtype",
        "fsdp_config.mix_precision.output_dtype",
        "fsdp_config.mix_precision.cast_forward_inputs",
        "optimizer.fp32_main_params",
    )
    mismatches = [
        {
            "path": path,
            "baseline": _nested_value(baseline, path),
            "candidate": _nested_value(candidate, path),
        }
        for path in invariant_paths
        if _nested_value(baseline, path) != _nested_value(candidate, path)
    ]
    return {
        "status": "FAIL" if mismatches else "PASS",
        "mismatches": mismatches,
    }


def _probe_rows(case_dir: Path) -> list[dict[str, Any]]:
    candidates = tuple(case_dir.glob("parameter_probes/rank*.jsonl"))
    if not candidates:
        candidates = tuple(case_dir.glob("**/parameter_probes/rank*.jsonl"))
    return [row for path in sorted(candidates) for row in load_jsonl(path)]


def _numeric_summary_delta(
        baseline: Mapping[str, Any],
        candidate: Mapping[str, Any],
        tolerance: Mapping[str, Any],
) -> dict[str, Any]:
    max_abs_limit = tolerance_value(tolerance, "max_abs", "atol")
    relative_l2_limit = tolerance_value(tolerance, "relative_l2", "rtol")
    tolerance_combination = tolerance.get("combination", "all")
    baseline_values = baseline.get("values")
    candidate_values = candidate.get("values")
    exact_values = baseline_values is not None and candidate_values is not None
    summary_policy_source = None
    if exact_values:
        baseline_tensor = torch.tensor(baseline_values, dtype=torch.float64)
        candidate_tensor = torch.tensor(candidate_values, dtype=torch.float64)
        difference = candidate_tensor - baseline_tensor
        max_abs = float(difference.abs().max()) if difference.numel() else 0.0
        denominator = max(
            float(torch.linalg.vector_norm(baseline_tensor)),  # pylint: disable=not-callable
            1.0e-30,
        )
        relative_l2 = (
            float(torch.linalg.vector_norm(difference))  # pylint: disable=not-callable
            / denominator
        )
        within_tolerance = _combine_tolerance_checks(
            (max_abs <= max_abs_limit, relative_l2 <= relative_l2_limit),
            tolerance_combination,
            owner="parameter",
        )
        comparison_mode = "exact_values"
        l2_norm_max_abs = None
        l2_norm_relative_error = None
        aggregate_deltas = None
    else:
        max_abs = None
        relative_l2 = None
        comparison_mode = "aggregate_summary"
        baseline_l2 = float(baseline.get("l2") or 0.0)
        candidate_l2 = float(candidate.get("l2") or 0.0)
        l2_norm_max_abs = abs(baseline_l2 - candidate_l2)
        l2_norm_relative_error = _relative_error(baseline_l2, candidate_l2)
        aggregate_deltas = {
            name: abs(float(baseline.get(name) or 0.0) - float(candidate.get(name) or 0.0))
            for name in ("sum", "min", "max")
        }
        summary_tolerance = tolerance.get("summary", {})
        if not isinstance(summary_tolerance, Mapping):
            raise ValueError("parameter summary tolerance must be a mapping")
        summary_relative_limit = _optional_tolerance(
            summary_tolerance,
            "l2_norm_relative",
        )
        summary_policy_source = "summary.l2_norm_relative"
        if summary_relative_limit is None:
            summary_relative_limit = relative_l2_limit
            summary_policy_source = "parameters.relative_l2 (compatibility fallback)"
        summary_abs_limit = _optional_tolerance(
            summary_tolerance,
            "l2_norm_max_abs",
        )
        summary_checks = []
        if summary_abs_limit is not None:
            summary_checks.append(l2_norm_max_abs <= summary_abs_limit)
        if summary_relative_limit is not None:
            summary_checks.append(l2_norm_relative_error <= summary_relative_limit)
        summary_combination = summary_tolerance.get("combination", "all")
        within_tolerance = _combine_tolerance_checks(
            summary_checks,
            summary_combination,
            owner="parameter summary",
        )
        tolerance_combination = summary_combination
    layout_equal = baseline.get("layout", {}).get("global_shape") == candidate.get(
        "layout", {}
    ).get("global_shape")
    finite = bool(baseline.get("finite")) and bool(candidate.get("finite"))
    status = "PASS" if finite and layout_equal and within_tolerance else "FAIL"
    result = {
        "status": status,
        "comparison_mode": comparison_mode,
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "l2_norm_max_abs": l2_norm_max_abs,
        "l2_norm_relative_error": l2_norm_relative_error,
        "aggregate_deltas": aggregate_deltas,
        "global_shape_equal": layout_equal,
        "finite": finite,
        "full_value_comparison": exact_values,
        "tolerance_combination": tolerance_combination,
    }
    if summary_policy_source is not None:
        result["summary_policy_source"] = summary_policy_source
    return result


def compare_parameter_probes(
        baseline_dir: str | Path,
        candidate_dir: str | Path,
        tolerance: Mapping[str, Any],
        *,
        allow_candidate_subset: bool = False,
        compare_optimizer_states: bool = True,
) -> dict[str, Any]:
    """Compare canonical parameter lifecycle summaries from two topologies.

    Optimizer internals such as Muon momentum may be topology-local across HSDP
    replica groups even when synchronized gradients and model updates are
    identical. Cross-topology cases therefore validate their live layouts and
    checkpoint recovery, but reserve numerical state comparison for matching
    topologies.
    """
    baseline_rows = _probe_rows(Path(baseline_dir))
    candidate_rows = _probe_rows(Path(candidate_dir))
    if not baseline_rows or not candidate_rows:
        return {
            "status": "BLOCKED",
            "reason": "strict parameter probe evidence is missing",
        }

    def first_rank(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
        """Index one canonical row per step, stage, and parameter."""
        indexed = {}
        for row in rows:
            key = (row.get("step"), row.get("stage"), row.get("name"))
            if key not in indexed or int(row.get("rank", 0)) < int(indexed[key].get("rank", 0)):
                indexed[key] = row
        return indexed

    baseline = first_rank(baseline_rows)
    candidate = first_rank(candidate_rows)

    def zero_optimizer_state(row: Optional[Mapping[str, Any]]) -> bool:
        """Treat a load-primed zero state as equivalent to an absent lazy state."""
        states = None if row is None else row.get("optimizer_state")
        return bool(states) and all(
            bool(summary.get("finite")) and float(summary.get("l2") or 0.0) == 0.0
            for summary in states.values()
        )

    if allow_candidate_subset:
        candidate_steps = {key[0] for key in candidate}
        expected_baseline = {key for key in baseline if key[0] in candidate_steps}
        unmatched = expected_baseline ^ set(candidate)
    else:
        unmatched = set(baseline) ^ set(candidate)
    zero_initialized = sorted(
        (
            key
            for key in unmatched
            if zero_optimizer_state(baseline.get(key))
            or zero_optimizer_state(candidate.get(key))
        ),
        key=repr,
    )
    missing = sorted(set(unmatched) - set(zero_initialized), key=repr)
    comparisons = []
    skipped_optimizer_states = 0
    for key in sorted(set(baseline).intersection(candidate), key=repr):
        baseline_row = baseline[key]
        candidate_row = candidate[key]
        if "optimizer_state" in baseline_row or "optimizer_state" in candidate_row:
            baseline_states = baseline_row.get("optimizer_state", {})
            candidate_states = candidate_row.get("optimizer_state", {})
            state_names = set(baseline_states) | set(candidate_states)
            if not compare_optimizer_states:
                skipped_optimizer_states += len(state_names)
                continue
            for state_name in sorted(state_names):
                if state_name not in baseline_states or state_name not in candidate_states:
                    comparisons.append(
                        {"key": (*key, state_name), "status": "FAIL", "reason": "state missing"}
                    )
                    continue
                comparison = _numeric_summary_delta(
                    baseline_states[state_name],
                    candidate_states[state_name],
                    tolerance,
                )
                comparison["key"] = (*key, state_name)
                comparisons.append(comparison)
            continue
        comparison = _numeric_summary_delta(baseline_row, candidate_row, tolerance)
        comparison["key"] = key
        if baseline_row.get("comparison") == "exact_hash":
            comparison["hash_equal"] = baseline_row.get("sha256") == candidate_row.get("sha256")
            if not comparison["hash_equal"]:
                comparison["status"] = "FAIL"
        baseline_replica = baseline_row.get("replica", {}).get("replicas_equal", True)
        candidate_replica = candidate_row.get("replica", {}).get("replicas_equal", True)
        comparison["replicas_equal"] = baseline_replica and candidate_replica
        if not comparison["replicas_equal"]:
            comparison["status"] = "FAIL"
        comparisons.append(comparison)
    status = "FAIL" if missing or any(row["status"] == "FAIL" for row in comparisons) else "PASS"
    return {
        "status": status,
        "missing": [repr(key) for key in missing],
        "zero_initialized_optimizer_states": [repr(key) for key in zero_initialized],
        "optimizer_state_numeric_comparison": (
            "enabled" if compare_optimizer_states else "skipped_cross_topology"
        ),
        "skipped_optimizer_states": skipped_optimizer_states,
        "values": comparisons,
    }


def summarize_performance(
        rows: Iterable[Mapping[str, Any]],
        warmup_steps: int = 1,
        memory_rows: Optional[Iterable[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Summarize steady-state wall time and throughput independently of precision."""
    values = list(rows)[warmup_steps:]
    step_times = sorted(float(row["step_time_seconds"]) for row in values)
    if not step_times:
        return {"status": "BLOCKED", "reason": "no steady-state performance steps"}
    p90_index = min(math.ceil(0.9 * len(step_times)) - 1, len(step_times) - 1)
    token_rates = [
        float(row["tokens_per_second"])
        for row in values
        if row.get("tokens_per_second") is not None
    ]
    sample_rates = [
        float(row["samples_per_second"])
        for row in values
        if row.get("samples_per_second") is not None
    ]
    allocator_rows = list(memory_rows) if memory_rows is not None else values
    peak_allocated = [
        int(row["peak_memory_allocated_bytes"])
        for row in allocator_rows
        if row.get("peak_memory_allocated_bytes") is not None
    ]
    peak_reserved = [
        int(row["peak_memory_reserved_bytes"])
        for row in allocator_rows
        if row.get("peak_memory_reserved_bytes") is not None
    ]
    return {
        "status": "MEASURED",
        "steady_steps": len(step_times),
        "step_time_p50_seconds": statistics.median(step_times),
        "step_time_p90_seconds": step_times[p90_index],
        "tokens_per_second_p50": statistics.median(token_rates) if token_rates else None,
        "samples_per_second_p50": statistics.median(sample_rates) if sample_rates else None,
        "peak_memory_allocated_bytes": max(peak_allocated) if peak_allocated else None,
        "peak_memory_reserved_bytes": max(peak_reserved) if peak_reserved else None,
    }


def compare_checkpoint_layouts(
        baseline_dir: str | Path,
        candidate_dir: str | Path,
        *,
        same_topology: bool,
) -> dict[str, Any]:
    """Compare DCP-bound logical layouts while allowing legal resharding."""
    baseline_files = sorted(Path(baseline_dir).glob("**/checkpoint/*_rank*.json"))
    candidate_files = sorted(Path(candidate_dir).glob("**/checkpoint/*_rank*.json"))
    if not baseline_files or not candidate_files:
        return {"status": "BLOCKED", "reason": "checkpoint layout evidence is missing"}

    def group_events(files: list[Path]) -> dict[str, list[Path]]:
        """Group every rank file by its versioned checkpoint event name."""
        grouped: dict[str, list[Path]] = {}
        for path in files:
            event_name = path.name.rsplit("_rank", 1)[0]
            grouped.setdefault(event_name, []).append(path)
        return grouped

    def latest_event(events: Mapping[str, list[Path]], preferred: str) -> str:
        """Choose the newest numbered event, retaining legacy-name support."""
        matching = [
            event_name
            for event_name in events
            if event_name == preferred or event_name.startswith(f"{preferred}_")
        ]
        if not matching:
            return sorted(events)[-1]

        def event_order(event_name: str) -> tuple[int, str]:
            """Sort numbered checkpoint events after legacy unnumbered events."""
            suffix = event_name.rsplit("_", 1)[-1]
            return (int(suffix), event_name) if suffix.isdigit() else (-1, event_name)

        return max(matching, key=event_order)

    def load_rank_layouts(files: list[Path]) -> list[dict[str, Any]]:
        """Load checkpoint tensor-layout maps for all selected ranks."""
        return [
            json.loads(path.read_text(encoding="utf-8")).get("tensor_layouts", {})
            for path in files
        ]

    baseline_events = group_events(baseline_files)
    candidate_events = group_events(candidate_files)
    candidate_event = latest_event(candidate_events, "after_load")
    checkpoint_identity = candidate_event.removeprefix("after_load").lstrip("_")
    baseline_event = (
        f"before_save_{checkpoint_identity}"
        if checkpoint_identity
        else latest_event(baseline_events, "before_save")
    )
    if baseline_event not in baseline_events:
        return {
            "status": "BLOCKED",
            "reason": "baseline checkpoint evidence for the restored checkpoint is missing",
            "baseline_event": baseline_event,
            "candidate_event": candidate_event,
        }
    baseline_ranks = load_rank_layouts(baseline_events[baseline_event])
    candidate_ranks = load_rank_layouts(candidate_events[candidate_event])
    baseline = baseline_ranks[0]
    candidate = candidate_ranks[0]
    baseline_only = set(baseline) - set(candidate)
    candidate_only = set(candidate) - set(baseline)
    initialized_optimizer_entries = sorted(
        name for name in candidate_only if name.startswith("optimizer.")
    )
    missing = sorted(
        baseline_only
        | {
            name
            for name in candidate_only
            if name not in initialized_optimizer_entries
        }
    )
    mismatches = []
    for name in sorted(set(baseline).intersection(candidate)):
        baseline_layout = baseline[name]
        candidate_layout = candidate[name]
        fields = ["global_shape"]
        if same_topology:
            fields.extend(("mesh_dim_names", "mesh_shape", "placements"))
        differences = {
            field: {
                "baseline": baseline_layout.get(field),
                "candidate": candidate_layout.get(field),
            }
            for field in fields
            if baseline_layout.get(field) != candidate_layout.get(field)
        }
        if differences:
            mismatches.append({"name": name, "differences": differences})
    rank_inconsistencies = []
    for side_name, rank_layouts in (
        ("baseline", baseline_ranks),
        ("candidate", candidate_ranks),
    ):
        all_names = set().union(*(set(layouts) for layouts in rank_layouts))
        for name in sorted(all_names):
            values = [layouts.get(name) for layouts in rank_layouts]
            global_shapes = {
                tuple(value.get("global_shape", ()))
                for value in values
                if value is not None
            }
            if any(value is None for value in values) or len(global_shapes) != 1:
                rank_inconsistencies.append(
                    {
                        "side": side_name,
                        "name": name,
                        "reason": "rank-local payloads disagree on key presence or global shape",
                    }
                )
    if same_topology and len(baseline_ranks) != len(candidate_ranks):
        rank_inconsistencies.append(
            {
                "side": "comparison",
                "reason": "same-topology checkpoint evidence has different rank counts",
                "baseline_ranks": len(baseline_ranks),
                "candidate_ranks": len(candidate_ranks),
            }
        )
    if same_topology and len(baseline_ranks) == len(candidate_ranks):
        for rank, (baseline_layouts, candidate_layouts) in enumerate(
                zip(baseline_ranks, candidate_ranks)
        ):
            for name in sorted(set(baseline_layouts).intersection(candidate_layouts)):
                if baseline_layouts[name].get("local_shape") != candidate_layouts[name].get(
                        "local_shape"
                ):
                    rank_inconsistencies.append(
                        {
                            "side": "comparison",
                            "rank": rank,
                            "name": name,
                            "reason": "same-topology local shard shape changed",
                        }
                    )
    return {
        "status": "FAIL" if missing or mismatches or rank_inconsistencies else "PASS",
        "missing": missing,
        "mismatches": mismatches,
        "rank_inconsistencies": rank_inconsistencies,
        "baseline_rank_count": len(baseline_ranks),
        "candidate_rank_count": len(candidate_ranks),
        "baseline_event": baseline_event,
        "candidate_event": candidate_event,
        "initialized_optimizer_entries": initialized_optimizer_entries,
    }


__all__ = [
    "compare_checkpoint_layouts",
    "compare_parameter_probes",
    "compare_preflight_configs",
    "compare_scalar_curves",
    "load_jsonl",
    "summarize_performance",
]
