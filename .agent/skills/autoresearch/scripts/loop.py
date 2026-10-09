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

import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import RunConfig  # sibling module; the entry script puts this directory on sys.path

RESULTS_HEADER = "timestamp\tcommit\tmetric\tstatus\tdescription\n"
_STATUS_KEEP = "keep"
_STATUS_DISCARD = "discard"
_STATUS_CRASH = "crash"


@dataclass
class CommandResult:
    """Outcome of one shell command."""

    ok: bool
    output: str
    missing: list[str]


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
    done = subprocess.run(
        list(command), cwd=str(repo),
        capture_output=True, encoding="utf-8", errors="replace", check=False,
    )
    output = done.stdout + done.stderr
    missing = [marker for marker in must_match if marker not in output]
    return CommandResult(ok=done.returncode == 0 and not missing,
                         output=output, missing=missing)


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
            f"{run_rel}/ideas.md", f"{run_rel}/learnings.md"]


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
        The outcome; ``crash`` when the benchmark or its markers fail.
    """
    dirty = [item for item in _dirty_files(config.repo_root) if item in config.target_files]
    if dirty:
        raise RuntimeError(f"target files have uncommitted changes: {dirty}")
    head = _git(config.repo_root, "rev-parse", "--short", "HEAD")
    bench = _run_shell(config.repo_root, config.benchmark_cmd, config.bench_must_match)
    metric = config.extract_metric(bench.output)
    if not bench.ok or metric is None:
        detail = f"missing markers: {bench.missing}" if bench.missing else "benchmark failed"
        _append_result(config, "xxxxxxx", None, _STATUS_CRASH, f"{description} ({detail})")
        _commit_paths(config.repo_root, _bookkeeping_paths(config),
                      f"[autoresearch] crash: {description}")
        return IterationOutcome(_STATUS_CRASH, None, None, head, description, detail)
    _append_result(config, head, metric, _STATUS_KEEP,
                   _describe(description, config.extract_aux(bench.output)))
    _commit_paths(config.repo_root, _bookkeeping_paths(config),
                  f"[autoresearch] baseline: metric={metric:.4g}")
    return IterationOutcome(_STATUS_KEEP, metric, None, head, description, "baseline recorded")


def _confirm_within_noise(config: RunConfig, best: float | None, metric: float,
                          bench: CommandResult) -> tuple[float | None, CommandResult, str, str]:
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
        again = _run_shell(config.repo_root, config.benchmark_cmd, config.bench_must_match)
        value = config.extract_metric(again.output)
        if not again.ok or value is None:
            detail = (f"noise-confirm rerun missing markers: {again.missing}"
                      if again.missing else "noise-confirm rerun failed")
            return None, bench, detail, ""
        samples.append(value)
        bench = again
    return statistics.median(samples), bench, "", f" (median of {len(samples)} runs)"


def _run_gate_and_benchmark(
        config: RunConfig, best: float | None,
) -> tuple[str, float | None, str, dict[str, str]]:
    """Gate, benchmark, noise-confirm, and decide against ``best``.

    Returns ``(status, metric, detail, aux)``; the status stays ``crash``
    until a parsed metric passes the gate, the markers, and any
    noise-confirmation reruns.
    """
    repo = config.repo_root
    if config.gate_cmd:
        gate = _run_shell(repo, config.gate_cmd, config.gate_must_match)
        if not gate.ok:
            detail = (f"gate missing markers: {gate.missing}"
                      if gate.missing else "gate failed")
            return _STATUS_CRASH, None, detail, {}
    bench = _run_shell(repo, config.benchmark_cmd, config.bench_must_match)
    metric = config.extract_metric(bench.output)
    if not bench.ok or metric is None:
        detail = (f"benchmark missing markers: {bench.missing}"
                  if bench.missing else "benchmark failed or metric absent")
        return _STATUS_CRASH, None, detail, {}
    metric, bench, crash_detail, note = _confirm_within_noise(config, best, metric, bench)
    if metric is None:
        return _STATUS_CRASH, None, crash_detail, {}
    aux = config.extract_aux(bench.output)
    if best is None or config.improved(metric, best):
        return _STATUS_KEEP, metric, f"metric {metric:.4g} vs best {best}{note}", aux
    return (_STATUS_DISCARD, metric,
            f"metric {metric:.4g} within noise of best {best:.4g}{note}", aux)


def iterate(config: RunConfig, description: str) -> IterationOutcome:
    """Run one full experiment iteration for the agent's pending edit.

    The target files must carry the (uncommitted) experimental edit and
    nothing else may be dirty. The edit is committed first - benchmarks
    measure commits, never working trees - then gated, benchmarked, decided
    against the best recorded metric, and either kept or restored.

    Args:
        config: The run configuration.
        description: Short experiment description for commits and the TSV.

    Returns:
        The structured outcome of the iteration.

    Raises:
        RuntimeError: The working tree has changes outside the target files,
            or there is no pending edit to measure.
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

    before = _git(repo, "rev-parse", "--short", "HEAD")
    _git(repo, "add", "--", *dirty_targets)
    _git(repo, "commit", "-m", f"[autoresearch] exp: {description}")
    experiment = _git(repo, "rev-parse", "--short", "HEAD")

    best = best_recorded_metric(config)
    status, metric, detail, aux = _run_gate_and_benchmark(config, best)

    row_commit = experiment if status == _STATUS_KEEP else "xxxxxxx"
    _append_result(config, row_commit, metric, status, _describe(description, aux))

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
    return "\n".join(summary)
