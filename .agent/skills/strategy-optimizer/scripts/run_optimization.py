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
#!/usr/bin/env python3
"""Closed-loop parallel strategy optimization.

The executable path requires a real HyperParallel search config and a training
config template. ``--plan-only`` is available for generating candidates without
devices; planned candidates are never reported as validated.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from safety_checks import run_safety_checks
from distributed_scheduler import launch_batch_validation


def parse_baseline_config(config_str: str) -> Dict[str, int]:
    """Parse ``TP=2,CP=1,EP=2,DP=4`` into normalized lower-case keys."""
    config: Dict[str, int] = {}
    for pair in config_str.split(","):
        if "=" not in pair:
            raise ValueError(f"Invalid baseline component: {pair!r}")
        key, value = pair.strip().split("=", 1)
        config[key.strip().lower()] = int(value)
    for key in ("tp", "cp", "ep", "pp", "dp"):
        config.setdefault(key, 1)
    if any(value < 1 for value in config.values()):
        raise ValueError("Parallel degrees must be positive")
    return config


def _strategy_key(strategy: Dict[str, Any]) -> Tuple[str, ...]:
    keys = ("tp", "cp", "ep", "pp", "dp", "dp_shard", "edp_shard", "activation_checkpoint")
    return tuple(str(strategy.get(key, 1)) for key in keys)


def _fit_to_devices(strategy: Dict[str, Any], device_count: int) -> Optional[Dict[str, Any]]:
    """Return a candidate whose parallel degrees consume exactly all devices."""
    degrees = [strategy.get(key, 1) for key in ("tp", "cp", "ep", "pp", "dp")]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in degrees):
        return None
    if not isinstance(device_count, int) or device_count < 1:
        return None
    candidate = dict(zip(("tp", "cp", "ep", "pp", "dp"), degrees))
    non_dp = candidate["tp"] * candidate["cp"] * candidate["pp"]
    if non_dp < 1 or device_count % non_dp:
        return None
    candidate["dp"] = device_count // non_dp
    stage_size = device_count // candidate["pp"]
    if stage_size % candidate["ep"]:
        return None
    for key in ("activation_checkpoint", "dp_shard", "edp_shard"):
        if key in strategy:
            candidate[key] = strategy[key]
    domain = candidate["dp"] * candidate["cp"]
    candidate.setdefault("dp_shard", domain)
    candidate.setdefault("edp_shard", stage_size // candidate["ep"])
    if any(isinstance(candidate[key], bool) or not isinstance(candidate[key], int) or candidate[key] < 1
           for key in ("dp_shard", "edp_shard")):
        return None
    if domain % candidate["dp_shard"] or (stage_size // candidate["ep"]) % candidate["edp_shard"]:
        return None
    return candidate


def _apply_constraints(strategy: Dict[str, int], constraints: Dict[str, Any]) -> Dict[str, int]:
    result = dict(strategy)
    if constraints.get("max_tp"):
        result["tp"] = min(result["tp"], int(constraints["max_tp"]))
    if constraints.get("max_cp"):
        result["cp"] = min(result["cp"], int(constraints["max_cp"]))
    if constraints.get("min_dp"):
        result["dp"] = max(result["dp"], int(constraints["min_dp"]))
    if constraints.get("activation_checkpoint"):
        result["activation_checkpoint"] = constraints["activation_checkpoint"]
    if constraints.get("full_shard"):
        result.pop("dp_shard", None)
        result.pop("edp_shard", None)
    return result


def _heuristic_candidates(
        baseline_config: Dict[str, int],
        device_count: int,
        constraints: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Generate a small neighborhood used only for PSA alternatives/plan mode."""
    constraints = constraints or {}
    base = _apply_constraints(baseline_config, constraints)
    proposals = [("baseline_neighborhood", base, "Baseline strategy")]

    if base["cp"] > 1:
        candidate = dict(base)
        candidate["cp"] = 1
        proposals.append(("candidate_no_cp", candidate, "Remove CP communication"))
    if base["ep"] < device_count:
        candidate = dict(base)
        candidate["ep"] = min(device_count, base["ep"] * 2)
        proposals.append(("candidate_more_ep", candidate, "Increase EP degree"))
    if base["tp"] > 1:
        candidate = dict(base)
        candidate["tp"] = 1
        proposals.append(("candidate_no_tp", candidate, "Remove TP communication"))

    candidates: List[Dict[str, Any]] = []
    seen = set()
    for candidate_id, proposal, rationale in proposals:
        fitted = _fit_to_devices(proposal, device_count)
        if fitted is None:
            continue
        fitted = _apply_constraints(fitted, constraints)
        fitted = _fit_to_devices(fitted, device_count)
        if fitted is None or fitted["dp"] < constraints.get("min_dp", 1) or _strategy_key(fitted) in seen:
            continue
        seen.add(_strategy_key(fitted))
        candidates.append({
            "id": candidate_id,
            "strategy": fitted,
            "rationale": rationale,
            "source": "neighborhood",
        })
    return candidates[:3]


def _configure_search(search_config: Any, device_count: int, constraints: Dict[str, Any]) -> None:
    if constraints.get("memory_limit_gb"):
        search_config.constraint["memory_limit_gb"] = constraints["memory_limit_gb"]
    cluster = search_config.cluster_spec
    if int(cluster.get("num_nodes", 1)) != 1:
        raise ValueError("The executable scheduler supports one node; PSA num_nodes must be 1")
    total_devices = int(cluster.get("num_nodes", 1)) * int(cluster.get("cards_per_node", 1))
    if total_devices != device_count:
        raise ValueError("PSA cluster size must match --devices")
    for dimension in ("tp", "cp"):
        maximum = constraints.get("max_" + dimension)
        if maximum is None:
            continue
        key = {"tp": "tensor_parallel_degree", "cp": "context_parallel_degree"}[dimension]
        fixed = search_config.constraint.get("fixed_" + dimension + "_degree")
        if fixed and fixed > maximum:
            raise ValueError(f"PSA fixed {dimension} conflicts with refinement limit {maximum}")
        options = search_config.search_space.get(key, [2 ** exponent for exponent in range(16)])
        search_config.search_space[key] = [value for value in options if value <= maximum]
    if constraints.get("activation_checkpoint"):
        search_config.estimator["recompute_strategy"] = "auto"
        search_config.estimator["recompute_modes"] = [constraints["activation_checkpoint"]]


def generate_psa_candidates(
        model_info: Dict[str, Any],
        hardware_info: Dict[str, Any],
        baseline_config: Dict[str, int],
        psa_config: Optional[Path] = None,
        constraints: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run the repository PSA when configured, then add valid neighbors.

    The executable path requires ``--psa-config`` and invokes
    ``config_adapter.search_strategies``. Without it, this function is a
    planning fallback and marks its candidates as neighborhood estimates.
    """
    del model_info  # The model is fully described by psa_config for real runs.
    constraints = constraints or {}
    if psa_config is None:
        return _heuristic_candidates(baseline_config, hardware_info["device_count"], constraints)

    # The search runtime depends on optional Torch; planning works without it.
    from hyper_parallel.auto_parallel.config_adapter import (  # pylint: disable=import-outside-toplevel
        read_search_config,
        search_strategies,
        validate_strict,
    )

    search_config = read_search_config(str(psa_config))
    _configure_search(search_config, hardware_info["device_count"], constraints)
    validate_strict(search_config)
    psa_result = search_strategies(search_config)
    strategy = {
        "tp": int(psa_result.get("tp", 1)),
        "cp": int(psa_result.get("cp", 1)),
        "ep": int(psa_result.get("ep", 1)),
        "pp": int(psa_result.get("pp", 1)),
        "dp": int(psa_result.get("dp", 1)),
    }
    strategy.update({key: psa_result[key] for key in ("dp_shard", "activation_checkpoint") if key in psa_result})
    fitted = _fit_to_devices(strategy, hardware_info["device_count"])
    if fitted is None or fitted["dp"] != strategy["dp"]:
        raise ValueError(
            "PSA strategy is incompatible with Trainer world/EP/FSDP domains"
        )

    if (fitted["dp"] < constraints.get("min_dp", 1)
            or any(fitted[key] > constraints.get("max_" + key, fitted[key]) for key in ("tp", "cp"))):
        raise ValueError("PSA strategy violates the current refinement constraints")
    if constraints.get("full_shard"):
        fitted["dp_shard"] = fitted["dp"] * fitted["cp"]
        fitted["edp_shard"] = hardware_info["device_count"] // fitted["pp"] // fitted["ep"]
        psa_result["dp_shard"] = fitted["dp_shard"]
        psa_result["dp_replicate"] = 1
        # Full sharding changes the searched result; discard its stale cost estimate.
        psa_result.pop("score", None)
        psa_result.pop("memory_estimate_mb", None)

    candidates = [{
        "id": "psa_best",
        "strategy": fitted,
        "rationale": "HyperParallel config_adapter.search_strategies result",
        "source": "hyper_parallel.auto_parallel.config_adapter",
        "resolved_strategy": psa_result,
        "psa_config": str(psa_config),
        "estimate": {
            key: psa_result[key]
            for key in ("memory_estimate_mb", "score")
            if key in psa_result
        },
    }]
    for candidate in _heuristic_candidates(fitted, hardware_info["device_count"], constraints):
        if _strategy_key(candidate["strategy"]) != _strategy_key(fitted):
            candidate["source"] = "psa_neighborhood"
            candidates.append(candidate)
    return candidates[:3]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Training template must be a YAML mapping: {path}")
    return value


def write_candidate_config(
        output_dir: Path, candidate: Dict[str, Any], model_name: str,
        validation_steps: Optional[int], device_type: str, template: Optional[Path],
) -> Path:
    """Preserve a runnable Trainer template and patch only supported fields.

    With no template, emit a planning descriptor, never a runnable config.
    A PSA result is injected using the repository's official YAML writer.
    """
    del model_name, device_type
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{candidate['id']}.yaml"
    if template is None:
        path.write_text(yaml.safe_dump({"status": "PLANNED", "strategy": candidate["strategy"]}))
        return path
    payload = _load_yaml(template)
    if not isinstance(payload.get("model"), dict) or "_target_" not in payload["model"] or "optimizer" not in payload:
        raise ValueError("Use a runnable Trainer template with model._target_ and optimizer")
    if candidate.get("resolved_strategy"):
        # Only executable search/config injection requires the optional project runtime.
        from hyper_parallel.auto_parallel.config_adapter import read_search_config, write_resolved_yaml
        normalized = read_search_config(candidate["psa_config"])
        normalized.resolved_strategy = candidate["resolved_strategy"]
        write_resolved_yaml(normalized, str(template), str(path))
        payload = _load_yaml(path)
    strategy = candidate["strategy"]
    payload.pop("context", None)
    payload.setdefault("accelerator", {}).update({key + "_size": strategy[key] for key in ("tp", "cp", "ep", "pp")})
    payload.setdefault("fsdp_config", {}).update(dp_shard_size=strategy["dp_shard"],
                                               edp_shard_size=strategy["edp_shard"])
    if strategy.get("activation_checkpoint") and not candidate.get("resolved_strategy"):
        payload["activation_checkpoint"] = {"mode": strategy["activation_checkpoint"]}
    training = payload.setdefault("training", {})
    if validation_steps is not None:
        training.update(train_iters=validation_steps, train_samples=None)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _evaluate_results(
        results: Iterable[Dict[str, Any]],
        baseline_throughput: float,
        target_improvement: float,
        max_memory_gb: Optional[float],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
    evaluated = []
    for result in results:
        item = dict(result)
        metrics = item.get("metrics") or {}
        throughput = metrics.get("throughput_tokens_per_sec")
        memory = metrics.get("memory_peak_gb")
        step_time = metrics.get("step_time_ms")
        complete = all(isinstance(value, (float, int)) and not isinstance(value, bool)
                       and math.isfinite(value) and value > 0 for value in (throughput, memory, step_time))
        eligible = bool(item.get("success") and complete and (max_memory_gb is None or memory <= max_memory_gb))
        item["eligible"] = eligible
        target_met = eligible and throughput >= baseline_throughput * target_improvement
        item["target_met"] = target_met
        item["status"] = "PASS" if target_met else ("MEASURED" if item.get("success") else "FAIL")
        evaluated.append(item)

    measured = [item for item in evaluated if item.get("eligible")]
    best = max(
        measured,
        key=lambda item: item["metrics"].get("throughput_tokens_per_sec", float("-inf")),
        default=None,
    )
    return evaluated, best, any(item["target_met"] for item in evaluated)


def _refine_constraints(
        constraints: Dict[str, Any],
        results: Iterable[Dict[str, Any]],
        best: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Update constraints from measured failure modes for the next round."""
    refined = dict(constraints)
    errors = " ".join(str(item.get("error", "")) for item in results).lower()
    if "oom" in errors or "out of memory" in errors:
        refined["full_shard"] = True
        refined["activation_checkpoint"] = "full"
    if "communication" in errors or "timeout" in errors:
        refined["max_tp"] = max(1, int(refined.get("max_tp", 8)) // 2)
        refined["max_cp"] = max(1, int(refined.get("max_cp", 8)) // 2)
    if best and best.get("metrics", {}).get("memory_peak_gb"):
        refined.setdefault("observed_memory_gb", best["metrics"]["memory_peak_gb"])
    return refined


def _write_result(path: Path, result: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_args() -> tuple:
    parser = argparse.ArgumentParser(description="Closed-loop parallel strategy optimizer")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--devices", type=int, required=True, help="Number of devices")
    parser.add_argument("--baseline-throughput", type=float, required=True)
    parser.add_argument("--baseline-config", required=True, help="TP=2,CP=2,EP=2,DP=1")
    parser.add_argument("--psa-config", type=Path, help="HyperParallel auto_parallel search YAML")
    parser.add_argument("--config-template", type=Path, help="Runnable training YAML template")
    parser.add_argument("--output-dir", type=Path, default=Path("./optimizer_output"))
    parser.add_argument("--target-improvement", type=float, default=1.2)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--device-type", default="npu", choices=["npu", "cuda"])
    parser.add_argument("--train-entrypoint", default="examples.training_demo.train_text")
    parser.add_argument("--validation-steps", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-memory-gb", type=float)
    parser.add_argument("--plan-only", action="store_true", help="Generate candidates without validation runs")
    args = parser.parse_args()

    if args.max_iterations < 1 or args.max_iterations > 3:
        parser.error("--max-iterations must be between 1 and 3")
    if not args.plan_only and args.psa_config is None:
        parser.error("--psa-config is required for executable optimization; use --plan-only for planning")
    if not args.plan_only and args.config_template is None:
        parser.error("--config-template is required for executable validation")

    if (args.devices < 1 or args.validation_steps < 1 or args.timeout < 1
            or any(not math.isfinite(value) or value <= 0
                   for value in (args.baseline_throughput, args.target_improvement))):
        parser.error("devices, throughput, improvement, validation steps and timeout must be finite and positive")
    if args.max_memory_gb is not None and (not math.isfinite(args.max_memory_gb) or args.max_memory_gb <= 0):
        parser.error("--max-memory-gb must be positive")
    baseline_config = parse_baseline_config(args.baseline_config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty to prevent mixing evidence from different runs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args, baseline_config


def _prepare_iteration(
    args: argparse.Namespace, iteration: int, candidates: list, constraints: dict, candidate_by_id: dict,
) -> dict:
    for candidate in candidates:
        candidate["id"] = f"iteration_{iteration}_{candidate['id']}"
        candidate_by_id[candidate["id"]] = candidate
    iteration_record: Dict[str, Any] = {
        "iteration": iteration,
        "candidates": candidates,
        "constraints": dict(constraints),
    }
    config_items = []
    for candidate in candidates:
        config_path = write_candidate_config(
            args.output_dir / "configs" / f"iteration_{iteration}", candidate,
            args.model, args.validation_steps, args.device_type, args.config_template,
        )
        config_items.append({"candidate_id": candidate["id"], "config_path": str(config_path)})
    iteration_record["config_files"] = config_items
    return iteration_record


def _optimize(args: argparse.Namespace, hardware: dict, baseline_config: dict, result: dict) -> tuple:
    constraints: Dict[str, Any] = {"memory_limit_gb": args.max_memory_gb} if args.max_memory_gb else {}
    measured_keys = set()
    candidate_by_id = {}
    best_overall: Optional[Dict[str, Any]] = None

    for iteration in range(1, args.max_iterations + 1):
        candidates = generate_psa_candidates(
            {"name": args.model}, hardware, baseline_config,
            psa_config=args.psa_config, constraints=constraints,
        )
        candidates = [candidate for candidate in candidates
                      if _strategy_key(candidate["strategy"]) not in measured_keys]
        if not candidates:
            result["stop_reason"] = "No unmeasured candidates remain under current constraints"
            break
        iteration_record = _prepare_iteration(args, iteration, candidates, constraints, candidate_by_id)
        config_items = iteration_record["config_files"]

        if args.plan_only:
            iteration_record["validation"] = [
                {"candidate_id": item["candidate_id"], "status": "PLANNED"}
                for item in config_items
            ]
            result["iterations"].append(iteration_record)
            break

        validation = launch_batch_validation(
            config_items,
            world_size=args.devices,
            output_base_dir=str(args.output_dir / "validation" / f"iteration_{iteration}"),
            validation_steps=args.validation_steps,
            timeout=args.timeout,
            train_entrypoint=args.train_entrypoint,
        )
        measured_keys.update(_strategy_key(candidate["strategy"]) for candidate in candidates)
        evaluated, best, target_met = _evaluate_results(
            validation, args.baseline_throughput, args.target_improvement, args.max_memory_gb,
        )
        iteration_record["validation"] = evaluated
        result["iterations"].append(iteration_record)
        _write_result(args.output_dir / "optimization_result.json", result)
        if best and (best_overall is None or best["metrics"].get("throughput_tokens_per_sec", 0)
                     > best_overall["metrics"].get("throughput_tokens_per_sec", 0)):
            best_overall = best
        if target_met:
            break
        constraints = _refine_constraints(constraints, evaluated, best)

    return best_overall, candidate_by_id


def main() -> int:
    """Run the bounded strategy search and report its measured outcome."""
    args, baseline_config = _parse_args()
    hardware = {"device_count": args.devices, "device_type": args.device_type}

    if not args.plan_only:
        safe, message = run_safety_checks(
            device_type=args.device_type,
            world_size=args.devices,
            config_summary={"model": args.model, "world_size": args.devices},
        )
        if not safe:
            print(f"Safety checks failed: {message}", file=sys.stderr)
            return 1

    result: Dict[str, Any] = {
        "metadata": {
            "skill": "strategy-optimizer",
            "version": "2.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "hardware": hardware,
        },
        "baseline": {
            "config": baseline_config,
            "metrics": {"throughput_tokens_per_sec": args.baseline_throughput},
        },
        "iterations": [],
    }
    best_overall, candidate_by_id = _optimize(args, hardware, baseline_config, result)

    has_target = any(
        item.get("target_met")
        for iteration in result["iterations"]
        for item in iteration.get("validation", [])
    )
    final_status = "PLANNED" if args.plan_only else (
        "SUCCESS" if has_target else ("PARTIAL" if best_overall else "FAILED")
    )
    result["final_result"] = {
        "status": final_status,
        "total_iterations": len(result["iterations"]),
        "optimal_candidate": best_overall.get("candidate_id") if best_overall else None,
        "optimal_metrics": best_overall.get("metrics") if best_overall else None,
    }
    if best_overall and best_overall.get("log_path"):
        result["final_result"]["validation_log"] = best_overall["log_path"]
    if best_overall:
        selected = candidate_by_id[best_overall["candidate_id"]]
        production = write_candidate_config(args.output_dir, {**selected, "id": "production_config"},
                                            args.model, None, args.device_type, args.config_template)
        result["final_result"]["production_config"] = str(production)
        result["final_result"]["optimal_strategy"] = selected["strategy"]
    _write_result(args.output_dir / "optimization_result.json", result)
    report = ["# Strategy optimization", "", f"Status: {final_status}",
              f"Iterations: {len(result['iterations'])}", "",
              "```json", json.dumps(result["final_result"], indent=2, default=str), "```", ""]
    (args.output_dir / "optimization_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result["final_result"], indent=2, default=str))

    return 0 if final_status in ("SUCCESS", "PARTIAL", "PLANNED") else 1


if __name__ == "__main__":
    sys.exit(main())
