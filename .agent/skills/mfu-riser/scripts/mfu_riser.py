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
"""Bounded experiment controller for the agent-driven MFU Riser planning loop."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from os import killpg
from pathlib import Path
from signal import SIGKILL, SIGTERM
import subprocess
import tarfile
import time
import uuid

from riser_contract import (PERFORMANCE_SCOPES, decide, digest_json, hash_file, read_json, require,
                            series_transition, validate_config, validate_measurement)


def save(path: Path, value: dict) -> None:
    """Atomically replace a controller record.

    Args:
        path: Destination record.
        value: JSON object to persist.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def exclusive(root: Path) -> Iterator[None]:
    """Serialize transitions for one run.

    Args:
        root: Run directory that owns the lock.
    """
    with (root / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def source_manifest(workspace: Path, paths: list[str]) -> dict:
    """Hash the declared source and configuration files.

    Args:
        workspace: Resolved candidate workspace.
        paths: Relative source paths, excluding output directories.
    """
    require(bool(paths), "source_paths must cover model code, adapters, config and launcher")
    manifest = {}
    for item in paths:
        path = (workspace / item).resolve()
        require(path.is_relative_to(workspace) and path.exists(), f"Invalid source path: {item}")
        files = path.rglob("*") if path.is_dir() else [path]
        for filename in sorted(files):
            if not filename.is_file() or any(part in ("__pycache__", ".git") for part in filename.parts):
                continue
            require(filename.resolve().is_relative_to(workspace), "Source symlink escapes workspace")
            manifest[str(filename.relative_to(workspace))] = hash_file(filename)
    require(bool(manifest), "Empty source manifest")
    return manifest


def validator_hashes(config: dict) -> dict:
    """Hash the configured acceptance implementation.

    Args:
        config: Run configuration with absolute validator paths.
    """
    files = config.get("validator_files", [])
    require(bool(files), "validator_files must pin the acceptance scripts")
    require(all(Path(path).is_absolute() and Path(path).is_file() for path in files),
            "validator_files must be existing absolute files")
    return {path: hash_file(Path(path)) for path in files}


def initialize(root: Path, config: dict, parent_root: Path | None = None) -> dict:
    """Create a run with frozen workload, commands, gates, and budgets.

    Args:
        root: New run directory.
        config: Workload and execution contract.
        parent_root: Existing run to link for capacity tuning or acceptance.
    """
    validate_config(config)
    gates = validator_hashes(config)
    parent_series = None
    if parent_root is not None:
        parent_root = parent_root.resolve()
        with exclusive(parent_root):
            parent_config, parent_state = load_run(parent_root)
            parent_series = series_transition(parent_config, config)
            best = best_result(parent_root, parent_state)
            require(best is not None, "Linked series requires a validated parent result")
            parent_series.update(run_dir=str(parent_root), config_digest=parent_state["config_digest"],
                                 trial=best["id"], result_sha256=parent_state["result_hashes"][best["id"]],
                                 source_digest=best["source_digest"], actions=best["plan"]["actions"],
                                 operator_scope=best["plan"].get("operator_scope", "compute_fusion"))
    root.mkdir(parents=True, exist_ok=False)
    now = time.time()
    save(root / "config.json", config)
    state = {"schema": "mfu-riser-controller/v1", "config_digest": digest_json(config),
             "validators": gates, "started": now, "deadline": now + config["max_seconds"],
             "best": None, "baseline": None, "pending": None, "attempts": [],
             "result_hashes": {}, "target_confirmed": False}
    if parent_series is not None:
        state["parent_series"] = parent_series
    save(root / "controller.json", state)
    (root / "history.jsonl").touch()
    return state


def load_run(root: Path) -> tuple[dict, dict]:
    """Read and verify the frozen configuration and controller state.

    Args:
        root: Existing run directory.
    """
    config, state = read_json(root / "config.json"), read_json(root / "controller.json")
    require(digest_json(config) == state["config_digest"], "Run config changed; initialize a new run")
    require(validator_hashes(config) == state["validators"], "Acceptance scripts changed")
    return config, state


def best_result(root: Path, state: dict) -> dict | None:
    """Read the best validated result.

    Args:
        root: Existing run directory.
        state: Controller state with the retained trial and result digest.
    """
    if not state["best"]:
        return None
    path = root / "trials" / state["best"] / "result.json"
    require(hash_file(path) == state["result_hashes"][state["best"]], "Best result changed after validation")
    return read_json(path)


def stop_reason(config: dict, state: dict) -> str | None:
    """Return the stop condition, if any.

    Args:
        config: Frozen run configuration with trial limits.
        state: Controller state with elapsed time and attempt history.
    """
    if state["pending"]:
        return "pending_trial_requires_recovery"
    if state["target_confirmed"]:
        return "target_confirmed"
    if time.time() >= state["deadline"]:
        return "time_budget_exhausted"
    if len(state["attempts"]) >= config["max_trials"] + 1:
        return "trial_budget_exhausted"
    return None


def validate_plan(plan: dict, state: dict, best: dict | None, policy: dict | None = None) -> None:
    """Require a falsifiable proposal based on the current best.

    Args:
        plan: Trial kind, proposed actions, and evidence.
        state: Controller state for the comparison series.
        best: Retained trial, or None before the baseline.
        policy: Optional performance policy for the comparison series.
    """
    require(plan.get("kind") in ("baseline", "candidate", "confirm"), "Invalid trial kind")
    require(plan.get("parent") == state["best"], "Plan is based on a stale best result")
    require((plan["kind"] == "baseline") == (best is None), "Baseline must be the first trial only")
    require(isinstance(plan.get("actions"), list), "actions must list the proposed changes")
    if plan["kind"] == "candidate":
        for key in ("observation", "mechanism", "expected_effect", "falsifier", "compatibility", "actions"):
            require(bool(plan.get(key)), f"Missing proposal field: {key}")
        for key in ("evidence", "knowledge"):
            require(isinstance(plan.get(key), list) and plan[key], f"Missing proposal {key}")
            for value in plan[key]:
                require(Path(value).is_file(), f"Missing proposal evidence: {value}")
        scope = plan.get("operator_scope")
        if policy is None:
            require(scope == "compute_fusion", "Candidates without performance_policy must target compute fusion")
        else:
            require(isinstance(scope, str) and scope in PERFORMANCE_SCOPES, "Invalid performance candidate scope")
    if plan["kind"] == "confirm":
        require(plan["actions"] == best["plan"]["actions"], "Confirmation must repeat the same actions")
        scope = best["plan"].get("operator_scope", "compute_fusion")
        require(plan.get("operator_scope", scope) == scope, "Confirmation must repeat the same scope")


def command_env(context: dict, directory: Path) -> dict:
    """Expose trial identity to training and validation adapters.

    Args:
        context: Frozen trial context.
        directory: Trial directory containing context.json.
    """
    return {**os.environ, "RISER_TRIAL_DIR": str(directory), "RISER_TRIAL_TOKEN": context["trial_token"],
            "RISER_CONTEXT": str(directory / "context.json"), "RISER_SOURCE_DIGEST": context["source_digest"],
            "RISER_WORKSPACE": context["plan"]["workspace"]}


def execute(
    command: list[str], directory: Path, context: dict, phase: str, timeout: float,
    stage_seconds: dict[str, float],
) -> None:
    """Run argv and terminate the local process group on timeout.

    Args:
        command: Command arguments.
        directory: Trial directory for logs and process metadata.
        context: Frozen trial context.
        phase: Stage name.
        timeout: Maximum time in seconds for the command.
        stage_seconds: Output mapping for stage wall time.
    """
    started = time.monotonic()
    try:
        with (directory / f"{phase}.log").open("a") as log:
            with subprocess.Popen(command, cwd=context["plan"]["workspace"],
                                  env=command_env(context, directory), stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True) as process:
                save(directory / f"{phase}-process.json", {"pid": process.pid, "started": time.time(), "argv": command})
                try:
                    code = process.wait(timeout=max(0.01, timeout))
                except BaseException:
                    try:
                        killpg(process.pid, SIGTERM)
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        killpg(process.pid, SIGKILL)
                        process.wait()
                    except ProcessLookupError:
                        pass
                    raise
        require(code == 0, f"{phase} exited {code}; see {phase}.log")
    finally:
        stage_seconds[phase] = time.monotonic() - started


def complete_trial(root: Path, state: dict, result: dict) -> None:
    """Persist the outcome and advance the controller state.

    Args:
        root: Run directory.
        state: Current controller state.
        result: Validated result or failure record.
    """
    trial_id = result["id"]
    save(root / "trials" / trial_id / "result.json", result)
    state["result_hashes"][trial_id] = hash_file(root / "trials" / trial_id / "result.json")
    state["pending"] = None
    state["attempts"].append(trial_id)
    if result["decision"] in ("baseline", "keep"):
        state["best"] = trial_id
        if result["decision"] == "baseline":
            state["baseline"] = trial_id
    if result["decision"] == "confirmed":
        state["target_confirmed"] = result["target_met"]
    with (root / "history.jsonl").open("a") as handle:
        handle.write(json.dumps({key: result[key] for key in ("id", "decision", "reason", "finished_utc")}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    save(root / "controller.json", state)


def _prepare_trial(root: Path, config: dict, state: dict, plan: dict, best: dict | None) -> tuple[Path, dict]:
    workspace = Path(plan["workspace"]).resolve()
    require(workspace.is_dir(), "workspace must exist")
    plan = {**plan, "workspace": str(workspace)}
    manifest = source_manifest(workspace, plan["source_paths"])
    digest = digest_json(manifest)
    if plan["kind"] == "confirm":
        require(digest == best["source_digest"], "Confirmation source/config differs from best")
        plan["operator_scope"] = best["plan"].get("operator_scope", "compute_fusion")
    parent_series = state.get("parent_series") or {}
    if plan["kind"] == "baseline" and parent_series.get("kind") == "acceptance":
        require(digest == parent_series["source_digest"], "Acceptance baseline must replay the screened source")
        plan.update(actions=parent_series["actions"], operator_scope=parent_series["operator_scope"])
    trial_id = f"{len(state['attempts']):03d}-{uuid.uuid4().hex[:8]}"
    directory = root / "trials" / trial_id
    directory.mkdir(parents=True)
    context = {"config": config, "plan": plan, "trial_token": uuid.uuid4().hex,
               "source_digest": digest, "source_manifest": manifest,
               "reference_trial": str(root / "trials" / state["best"]) if best else None,
               "parent_series": state.get("parent_series")}
    context["proposal_evidence_sha256"] = {path: hash_file(Path(path))
                                          for key in ("evidence", "knowledge") for path in plan.get(key, [])}
    save(directory / "context.json", context)
    with tarfile.open(directory / "source.tar.gz", "w:gz") as archive:
        for name in manifest:
            archive.add(workspace / name, arcname=name, recursive=False)
    state["pending"] = trial_id
    save(root / "controller.json", state)
    return directory, context


def run_trial(root: Path, plan: dict) -> dict:
    """Execute one bounded iteration.

    Args:
        root: Run directory.
        plan: Trial proposal for the current best result.
    """
    with exclusive(root):
        config, state = load_run(root)
        policy = config.get("performance_policy")
        level = (policy or {}).get("validation_level", "acceptance")
        require(stop_reason(config, state) is None, f"Run stopped: {stop_reason(config, state)}")
        best = best_result(root, state)
        validate_plan(plan, state, best, policy)
        directory, context = _prepare_trial(root, config, state, plan, best)
        plan = context["plan"]
        result = {"id": directory.name, "plan": plan, "source_digest": context["source_digest"], "decision": "crash",
                  "reason": "", "metrics": None, "target_met": False, "validation_level": level,
                  "stage_seconds": {}}
        try:
            for phase in ("precheck", "benchmark", "profile", "validate"):
                remaining = state["deadline"] - time.time()
                require(remaining > 0, "Time budget exhausted during trial")
                execute(config["commands"][phase], directory, context, phase,
                        min(config["command_timeout_seconds"], remaining), result["stage_seconds"])
            require(source_manifest(Path(plan["workspace"]), plan["source_paths"]) == context["source_manifest"],
                    "Measured source changed during trial")
            require(validator_hashes(config) == state["validators"], "Acceptance scripts changed during trial")
            data = read_json(directory / "measurement.json")
            reference = best["metrics"] if best else None
            metrics = validate_measurement(data, context, directory, reference)
            result["metrics"] = metrics
            result["decision"] = decide(metrics, best, config, plan["kind"])
            result["reason"] = "Validated model work/time; compared against best with declared noise band"
            result["target_met"] = (level == "acceptance" and metrics["mfu_percent"] >= config["target_mfu_percent"]
                                    and best is not None
                                    and best["metrics"]["mfu_percent"] >= config["target_mfu_percent"])
        except (OSError, ValueError, KeyError, TypeError, OverflowError, subprocess.TimeoutExpired,
                KeyboardInterrupt) as exc:
            result["decision"] = "crash"
            result["reason"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                execute(config["commands"]["cleanup"], directory, context, "cleanup", 30, result["stage_seconds"])
                result["cleanup"] = "PASS"
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                result.update(decision="crash", cleanup="FAIL", metrics=None, reason=f"Cleanup failed: {exc}")
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if result["cleanup"] == "FAIL":
            save(directory / "result.json", result)
            return result
        complete_trial(root, state, result)
        return result


def recover(root: Path) -> dict:
    """Run job-scoped cleanup for an interrupted trial and record it as a crash.

    The adapter identifies its remote workers by the unique trial token.

    Args:
        root: Run directory with an interrupted trial.
    """
    with exclusive(root):
        config, state = load_run(root)
        require(state["pending"] is not None, "No interrupted trial")
        trial_id = state["pending"]
        directory = root / "trials" / trial_id
        context = read_json(directory / "context.json")
        result_path = directory / "result.json"
        stage_seconds = read_json(result_path).get("stage_seconds", {}) if result_path.exists() else {}
        execute(config["commands"]["cleanup"], directory, context, "recovery-cleanup", 30, stage_seconds)
        result = {"id": trial_id, "plan": context["plan"], "source_digest": context["source_digest"],
                  "decision": "crash", "reason": "Interrupted trial recovered; measurements not promoted",
                  "metrics": None, "target_met": False, "cleanup": "PASS", "stage_seconds": stage_seconds,
                  "finished_utc": datetime.now(timezone.utc).isoformat()}
        complete_trial(root, state, result)
        return result


def status(root: Path) -> dict:
    """Return the next transition and the best evidence pointer.

    Args:
        root: Existing run directory.
    """
    config, state = load_run(root)
    best = best_result(root, state)
    reason = stop_reason(config, state)
    policy = config.get("performance_policy") or {}
    level = policy.get("validation_level", "acceptance")
    best_mfu = best["metrics"]["mfu_percent"] if best else None
    next_action = "stop" if reason else "plan_from_best_profiling"
    if not reason and best is None:
        next_action = "baseline"
    elif not reason and best_mfu >= config["target_mfu_percent"]:
        next_action = "start_acceptance_series" if level == "screening" else "confirm_best"
    elif not reason and best_mfu < (policy.get("low_mfu_threshold_percent") or 0):
        next_action = "plan_from_critical_path"
    now = time.time()
    return {"next_action": next_action, "stop_reason": reason, "best": state["best"],
            "best_metrics": best["metrics"] if best else None,
            "attempts": state["attempts"], "seconds_left": max(0, state["deadline"] - now),
            "elapsed_seconds": now - state["started"], "validation_level": level,
            "required_throughput_multiplier": config["target_mfu_percent"] / best_mfu if best_mfu else None,
            "parent_series": state.get("parent_series")}


def report(root: Path) -> Path:
    """Generate a portable result index including discarded and failed trials.

    Args:
        root: Run directory containing the result records.
    """
    summary = status(root)
    state = read_json(root / "controller.json")
    lines = ["# MFU Riser experiment record", "", f"Next action: {summary['next_action']}",
             f"Stop reason: {summary['stop_reason']}", f"Best trial: {summary['best']}",
             f"Validation level: {summary['validation_level']}", f"Elapsed seconds: {summary['elapsed_seconds']:.1f}",
             "", "MFU uses fixed model work and measured training time. HFU reports executed work.", "",
             "| Trial | Decision | MFU % | HFU % | HFU method | tokens/s/device | Device seconds | Evidence |",
             "|---|---|---:|---:|---|---:|---:|---|"]
    stage_totals = {}
    for trial_id in state["attempts"]:
        value = read_json(root / "trials" / trial_id / "result.json")
        metrics = value["metrics"] or {}
        lines.append(f"| {trial_id} | {value['decision']} | {metrics.get('mfu_percent', '—')} | "
                     f"{metrics.get('hfu_percent', '—')} | {metrics.get('hfu_method', '—')} | "
                     f"{metrics.get('tokens_per_second_per_device', '—')} | "
                     f"{metrics.get('allocated_device_seconds', '—')} | [result](trials/{trial_id}/result.json) |")
        for phase, seconds in value.get("stage_seconds", {}).items():
            stage_totals[phase] = stage_totals.get(phase, 0.0) + seconds
    if stage_totals:
        lines += ["", "| Stage | Recorded wall seconds |", "|---|---:|"]
        lines += [f"| {phase} | {seconds:.3f} |" for phase, seconds in stage_totals.items()]
    path = root / "REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    """Dispatch a controller command.

    Args:
        argv: Optional argument list for programmatic invocation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "trial", "status", "report", "recover"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--parent-run", type=Path, help="Link a new capacity or acceptance series to an existing run")
    args = parser.parse_args(argv)
    root = args.run_dir.resolve()
    try:
        require(args.parent_run is None or args.command == "init", "--parent-run applies only to init")
        if args.command == "init":
            require(args.config is not None, "init requires --config")
            result = initialize(root, read_json(args.config), args.parent_run)
        elif args.command == "trial":
            require(args.plan is not None, "trial requires --plan")
            result = run_trial(root, read_json(args.plan))
        elif args.command == "recover":
            result = recover(root)
        elif args.command == "report":
            result = {"report": str(report(root))}
        else:
            result = status(root)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 2 if result.get("decision") == "crash" else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
