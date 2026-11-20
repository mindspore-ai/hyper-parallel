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
"""The deterministic half of the autoresearch loop.

One iteration: commit the agent's edit to the declared target files, run the
benchmark (and gate), parse the metric, decide ``keep`` / ``discard`` /
``crash`` against the best recorded result, append the bookkeeping row, and
either keep the experiment commit or restore the target files. Four rules
learned the hard way are enforced here rather than left to agent discipline:

* the benchmark only ever measures committed code (remote benchmarks sync
  by commit, so an uncommitted edit silently measures the previous state);
* positive-evidence markers (e.g. an "engaged" flag) are required in the
  output - their absence is a ``crash``, not a quiet pass;
* a failed gate is a ``crash`` regardless of how good the metric looks;
* discarded and crashed experiments stay in history - the target files are
  restored by a new commit, never by rewriting it.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import RunConfig  # sibling module; the entry script puts this directory on sys.path

RESULTS_HEADER = "timestamp\tcommit\tmetric\tstatus\tdescription\n"
CANDIDATES_FILE = "candidates.jsonl"
_STATUS_KEEP = "keep"
_STATUS_DISCARD = "discard"
_STATUS_CRASH = "crash"


@dataclass
class CommandResult:
    """Outcome of one shell command."""

    output: str
    missing: list[str]
    returncode: int | None
    seconds: float
    failure: str = ""

    @property
    def ok(self) -> bool:
        """Whether the command and its evidence passed."""
        return self.returncode == 0 and not self.missing and not self.failure


@dataclass
class IterationOutcome:
    """Everything a caller needs to report one iteration."""

    status: str
    metric: float | None
    best_before: float | None
    experiment_commit: str
    description: str
    detail: str

    @property
    def kept(self) -> bool:
        """Whether the experiment commit remains the new state."""
        return self.status == _STATUS_KEEP


def _git(repo: Path, *args: str) -> str:
    """Run one git command in ``repo`` and return stripped stdout."""
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, encoding="utf-8", errors="replace", check=True,
    )
    return done.stdout.strip()


def _run_shell(repo: Path, command: list[str], must_match: list[str]) -> CommandResult:
    """Run ``command`` (argv list) from the repo root; verify markers.

    Commands are argv lists by contract - shell constructs such as pipes or
    redirects belong inside the invoked script, never in the configuration.
    """
    # Decode explicitly: with bare text=True the locale codec applies (GBK on
    # Chinese Windows) and one UTF-8 byte in the output kills the reader thread.
    started = time.monotonic()
    try:
        done = subprocess.run(
            list(command), cwd=str(repo),
            capture_output=True, encoding="utf-8", errors="replace", check=False,
        )
    except OSError:
        return CommandResult("", [], None, time.monotonic() - started, "launch_failed")
    output = "\n".join((done.stdout, done.stderr))
    missing = [marker for marker in must_match if marker not in output]
    return CommandResult(output, missing, done.returncode, time.monotonic() - started)


def _job_failure(result: CommandResult, receipt: dict, expected_steps: int) -> str:
    """Reject incomplete windows, unintended paths and unsuccessful cleanup."""
    completed = receipt["completed_steps"]
    status = receipt["status"]
    if status == "futile" and completed >= expected_steps:
        return "futility_stop_after_window"
    if status != "passed":
        return status
    if completed != expected_steps:
        return "incomplete_window" if completed < expected_steps else "unexpected_window"
    if result.returncode != 0:
        return "exit_failed_after_window"
    if not receipt["engaged"]:
        return "path_not_engaged"
    return "missing_markers" if result.missing else ""


def _run_job(config: RunConfig, phase: str, jobs: list[dict]) -> CommandResult:
    """Run one gate or measurement and attach its validated evidence."""
    gate = phase == "gate"
    command = config.gate_cmd if gate else config.benchmark_cmd
    markers = config.gate_must_match if gate else config.bench_must_match
    result = _run_shell(config.repo_root, command, markers)
    protocol = config.training_protocol
    if protocol is None:
        return result
    expected_steps = protocol.gate_steps if gate else protocol.benchmark_steps
    job = {"phase": phase, "exit_code": result.returncode, "expected_steps": expected_steps,
           "seconds": round(result.seconds, 3), "missing_markers": result.missing, "receipt": None}
    if result.returncode is not None:
        try:
            receipt = protocol.parse_receipt(result.output, phase)
        except ValueError as exc:
            result.failure = "invalid_training_receipt"
            job["receipt_error"] = str(exc)
        else:
            result.failure = _job_failure(result, receipt, expected_steps)
            job["receipt"] = receipt
    job["failure"] = result.failure or None
    jobs.append(job)
    return result


def _benchmark_metric(config: RunConfig, jobs: list[dict], result: CommandResult) -> float | None:
    """Read a metric only from a passing benchmark and record a missing value."""
    if not result.ok:
        return None
    metric = config.extract_metric(result.output)
    if config.training_protocol is not None:
        jobs[-1]["metric"] = metric
        if metric is None:
            result.failure = "invalid_metric"
            jobs[-1]["failure"] = result.failure
    return metric


def _dirty_files(repo: Path) -> list[str]:
    """Repo-relative paths with uncommitted changes (staged or not)."""
    done = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "-z"],
        capture_output=True, encoding="utf-8", errors="replace", check=True,
    )
    # NUL-separated "XY <path>" records; rename records list the new path
    # first, which is exactly what the dirty-tree check needs.
    return [entry[3:] for entry in done.stdout.split("\0") if len(entry) > 3]


def _describe(description: str, aux: dict[str, str]) -> str:
    """Append captured auxiliary metrics to a row description."""
    if not aux:
        return description
    tail = " ".join(f"{name}={value}" for name, value in aux.items())
    return f"{description} [{tail}]"


def _append_result(config: RunConfig, commit: str, metric: float | None,
                   status: str, description: str) -> None:
    """Append one TSV row, creating the file with its header if needed."""
    path = config.results_path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(RESULTS_HEADER, encoding="utf-8")
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    value = "0" if metric is None else f"{metric:.4g}"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{stamp}\t{commit}\t{value}\t{status}\t{description}\n")


def best_recorded_metric(config: RunConfig) -> float | None:
    """Best ``keep`` metric in the results TSV, or ``None`` when absent."""
    path = config.results_path
    if not path.exists():
        return None
    best: float | None = None
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 4 or parts[3] != _STATUS_KEEP:
            continue
        try:
            value = float(parts[2])
        except ValueError:
            continue
        if best is None:
            best = value
        elif config.metric_lower_is_better:
            best = min(best, value)
        else:
            best = max(best, value)
    return best


def _bookkeeping_paths(config: RunConfig) -> list[str]:
    """Repo-relative run files the loop itself is allowed to commit."""
    run_rel = config.run_dir.relative_to(config.repo_root).as_posix()
    return [f"{run_rel}/{config.results_file}", f"{run_rel}/{config.log_file}",
            f"{run_rel}/ideas.md", f"{run_rel}/learnings.md", f"{run_rel}/{CANDIDATES_FILE}"]


def _append_receipt(config: RunConfig, receipt: dict) -> None:
    """Append a complete baseline or candidate receipt."""
    with (config.run_dir / CANDIDATES_FILE).open("a", encoding="utf-8", newline="\n") as ledger:
        ledger.write(json.dumps(receipt, ensure_ascii=False) + "\n")


def _check_training_baseline(config: RunConfig, *, starting: bool) -> None:
    """Require a fresh series and enforce its frozen protocol before training."""
    path = config.run_dir / CANDIDATES_FILE
    frozen = None
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("kind") == "baseline" and row["status"] == _STATUS_KEEP:
                frozen = row["training_protocol"]
                break
    protocol = asdict(config.training_protocol) if config.training_protocol is not None else None
    if frozen is not None and protocol != frozen:
        raise RuntimeError("training protocol differs from the accepted baseline; start a new run directory")
    if protocol is None:
        return
    if starting and (frozen is not None or best_recorded_metric(config) is not None):
        raise RuntimeError("training baseline requires a run directory without an accepted measurement")
    if not starting and frozen is None:
        raise RuntimeError("record a passing training baseline before measuring candidates")


def pending_retries(config: RunConfig) -> list[dict]:
    """Return deferred candidates triggered by later kept changes, oldest first."""
    path = config.run_dir / CANDIDATES_FILE
    if not path.exists():
        return []
    changed: set[str] = set()
    retried: set[str] = set()
    ready = []
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        if row.get("kind") == "baseline":
            continue
        if row["retry_of"]:
            retried.add(row["retry_of"])
        if row["status"] == _STATUS_KEEP:
            changed.update(row["changes"])
        elif row["commit"] not in retried and changed.intersection(row["revisit_on"]):
            ready.append(row)
    return list(reversed(ready))


def _commit_paths(repo: Path, paths: list[str], message: str) -> str:
    """Stage ``paths`` that exist and commit; return the new commit hash."""
    existing = [item for item in paths if (repo / item).exists()]
    _git(repo, "add", "--", *existing)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "--short", "HEAD")


def record_baseline(config: RunConfig, description: str = "baseline") -> IterationOutcome:
    """Benchmark the current commit and record it as the first ``keep``.

    Args:
        config: The run configuration.
        description: Row description for the TSV.

    Returns:
        The outcome; ``crash`` when the gate or benchmark fails.
    """
    dirty = [item for item in _dirty_files(config.repo_root) if item in config.target_files]
    if dirty:
        raise RuntimeError(f"target files have uncommitted changes: {dirty}")
    _check_training_baseline(config, starting=True)
    head = _git(config.repo_root, "rev-parse", "HEAD")
    jobs: list[dict] = []
    status, metric, detail, aux, stage = _run_gate_and_benchmark(config, None, jobs)
    if config.training_protocol is not None:
        _append_receipt(config, {
            "kind": "baseline", "commit": head, "description": description,
            "status": status, "stage": stage, "metric": metric, "detail": detail, "aux": aux,
            "training_protocol": asdict(config.training_protocol), "jobs": jobs,
        })
    head = _git(config.repo_root, "rev-parse", "--short", head)
    if metric is None:
        _append_result(config, "xxxxxxx", None, _STATUS_CRASH, f"{description} ({detail})")
        _commit_paths(config.repo_root, _bookkeeping_paths(config),
                      f"[autoresearch] crash: {description}")
        return IterationOutcome(status, None, None, head, description, detail)
    _append_result(config, head, metric, _STATUS_KEEP, _describe(description, aux))
    _commit_paths(config.repo_root, _bookkeeping_paths(config),
                  f"[autoresearch] baseline: metric={metric:.4g}")
    return IterationOutcome(_STATUS_KEEP, metric, None, head, description, "baseline recorded")


def _confirm_within_noise(config: RunConfig, best: float | None, metric: float,
                          bench: CommandResult, jobs: list[dict]) -> tuple[float | None, CommandResult, str, str]:
    """Re-run the benchmark when the first sample lands inside the noise band.

    A within-noise first sample is indistinguishable from measurement noise,
    so the decision is taken on the median of ``1 + noise_confirm_reruns``
    samples instead. Clear improvements and clear regressions skip the
    reruns; a rerun that fails its markers voids the whole measurement.

    Returns:
        ``(metric, bench, crash_detail, note)`` - ``metric`` is the decision
        value (``None`` on crash), ``bench`` the last successful run (its
        output feeds the auxiliary captures), ``note`` a suffix describing
        the confirmation for the bookkeeping row.
    """
    if (config.noise_confirm_reruns <= 0 or best is None
            or config.improved(metric, best) or not config.within_noise(metric, best)):
        return metric, bench, "", ""
    samples = [metric]
    for _ in range(config.noise_confirm_reruns):
        again = _run_job(config, "benchmark", jobs)
        value = _benchmark_metric(config, jobs, again)
        if value is None:
            detail = _failure_detail("noise-confirm rerun", again, "failed")
            return None, bench, detail, ""
        samples.append(value)
        bench = again
    return statistics.median(samples), bench, "", f" (median of {len(samples)} runs)"


def _failure_detail(phase: str, result: CommandResult, fallback: str) -> str:
    """Describe a structured failure or use the legacy marker diagnostic."""
    if result.failure:
        return f"{phase}: {result.failure}"
    if result.missing:
        return f"{phase} missing markers: {result.missing}"
    return f"{phase} {fallback}"


def _run_gate_and_benchmark(config: RunConfig, best: float | None,
                            jobs: list[dict]) -> tuple[str, float | None, str, dict[str, str], str]:
    """Gate, benchmark, noise-confirm, and decide against ``best``.

    Returns ``(status, metric, detail, aux, stage)``; the status stays ``crash``
    until a parsed metric passes the gate, the markers, and any
    noise-confirmation reruns.
    """
    if config.gate_cmd:
        gate = _run_job(config, "gate", jobs)
        if not gate.ok:
            detail = _failure_detail("gate", gate, "failed")
            return _STATUS_CRASH, None, detail, config.extract_aux(gate.output), "gate"
    bench = _run_job(config, "benchmark", jobs)
    metric = _benchmark_metric(config, jobs, bench)
    if metric is None:
        detail = _failure_detail("benchmark", bench, "failed or metric absent")
        return _STATUS_CRASH, None, detail, config.extract_aux(bench.output), "benchmark"
    metric, bench, crash_detail, note = _confirm_within_noise(config, best, metric, bench, jobs)
    if metric is None:
        return _STATUS_CRASH, None, crash_detail, {}, "noise_confirmation"
    aux = config.extract_aux(bench.output)
    if best is None or config.improved(metric, best):
        return _STATUS_KEEP, metric, f"metric {metric:.4g} vs best {best}{note}", aux, "decision"
    return (_STATUS_DISCARD, metric,
            f"metric {metric:.4g} did not beat best {best:.4g} beyond noise{note}", aux, "decision")


def iterate(config: RunConfig, description: str, *, changes: tuple[str, ...] = (),
            revisit_on: tuple[str, ...] = (), retry_of: str | None = None) -> IterationOutcome:
    """Run one full experiment iteration for the agent's pending edit.

    The target files must carry the (uncommitted) experimental edit and
    nothing else may be dirty. The edit is committed first - benchmarks
    measure commits, never working trees - then gated, benchmarked, decided
    against the best recorded metric, and either kept or restored.

    Args:
        config: The run configuration.
        description: Short experiment description for commits and the TSV.
        changes: Configuration scopes changed by this candidate.
        revisit_on: Accepted changes to these scopes can trigger a retry.
        retry_of: Full commit ID of a candidate currently due for retry.

    Returns:
        The structured outcome of the iteration.

    Raises:
        RuntimeError: The working tree has changes outside the target files,
            there is no pending edit, or the retry is not due.
    """
    repo = config.repo_root
    dirty = _dirty_files(repo)
    run_rel = config.run_dir.relative_to(repo).as_posix() + "/"
    offending = [item for item in dirty
                 if item not in config.target_files and not item.startswith(run_rel)]
    if offending:
        raise RuntimeError(f"uncommitted changes outside target files: {offending}")
    dirty_targets = [item for item in dirty if item in config.target_files]
    if not dirty_targets:
        raise RuntimeError("no pending edit in the target files; nothing to measure")
    if retry_of is not None and not any(row["commit"] == retry_of for row in pending_retries(config)):
        raise RuntimeError(f"candidate is not due for retry: {retry_of}")

    _check_training_baseline(config, starting=False)
    before = _git(repo, "rev-parse", "HEAD")
    _git(repo, "add", "--", *dirty_targets)
    _git(repo, "commit", "-m", f"[autoresearch] exp: {description}")
    experiment = _git(repo, "rev-parse", "--short", "HEAD")

    best = best_recorded_metric(config)
    jobs: list[dict] = []
    status, metric, detail, aux, stage = _run_gate_and_benchmark(config, best, jobs)

    row_commit = experiment if status == _STATUS_KEEP else "xxxxxxx"
    _append_result(config, row_commit, metric, status, _describe(description, aux if metric is not None else {}))
    receipt = {
        "parent": before,
        "commit": _git(repo, "rev-parse", "HEAD"),
        "description": description,
        "status": status,
        "stage": stage,
        "detail": detail,
        "metric": metric,
        "best_before": best,
        "gate": ("fail" if stage == "gate" else "pass") if config.gate_cmd else "in_benchmark",
        "required_markers": {"gate": config.gate_must_match, "benchmark": config.bench_must_match},
        "aux": aux,
        "changes": changes,
        "revisit_on": revisit_on,
        "retry_of": retry_of,
    }
    if config.training_protocol is not None:
        receipt["training_protocol"] = asdict(config.training_protocol)
        receipt["jobs"] = jobs
    _append_receipt(config, receipt)

    if status != _STATUS_KEEP:
        # Restore the target files with a forward commit: the failed state
        # stays in history for the audit trail, the tree returns to best.
        _git(repo, "checkout", before, "--", *config.target_files)
        _commit_paths(repo, config.target_files + _bookkeeping_paths(config),
                      f"[autoresearch] {status}: {description}")
    else:
        _commit_paths(repo, _bookkeeping_paths(config),
                      f"[autoresearch] keep: {description}")
    return IterationOutcome(status, metric, best, experiment, description, detail)


def run_status(config: RunConfig, tail: int = 5) -> str:
    """Human-readable summary of the run so far."""
    path = config.results_path
    if not path.exists():
        return "no results recorded yet"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines()[1:] if line]
    counts: dict[str, int] = {}
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 4:
            counts[parts[3]] = counts.get(parts[3], 0) + 1
    best = best_recorded_metric(config)
    summary = [f"experiments={len(lines)} " +
               " ".join(f"{key}={value}" for key, value in sorted(counts.items())),
               f"best_metric={best}"]
    summary.extend(lines[-tail:])
    summary.extend(
        f"retry_ready={row['commit']} ({row['description']})"
        for row in pending_retries(config)
    )
    return "\n".join(summary)
