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
"""Run configuration for an autoresearch run directory.

A run directory holds ``run.json`` plus the living documents (ideas,
learnings, experiment log, results). Paths inside the configuration are
interpreted relative to the repository root so the benchmark command, the
target files, and the bookkeeping stay reproducible from any CWD.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_NAME = "run.json"
_METRIC_GROUP = "metric"
_AUX_GROUP = "value"
_AUX_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_RECEIPT_PREFIX = "RSI_TRAINING_RECEIPT "
_RECEIPT_STATUSES = {
    "passed", "oom", "numerical_mismatch", "incomplete", "cleanup_failed", "contention", "futile", "error",
}
_RECEIPT_FIELDS = ("workload_id", "phase", "status", "completed_steps", "engaged")


def _read_receipt(output: str) -> dict:
    """Decode one terminal receipt from the wrapper output."""
    lines = [line[len(_RECEIPT_PREFIX):] for line in output.splitlines() if line.startswith(_RECEIPT_PREFIX)]
    if len(lines) != 1:
        raise ValueError(f"expected one terminal training receipt, got {len(lines)}")
    try:
        raw = json.loads(lines[0])
    except (ValueError, RecursionError) as exc:
        raise ValueError("malformed terminal training receipt") from exc
    if not isinstance(raw, dict) or any(key not in raw for key in _RECEIPT_FIELDS):
        raise ValueError("training receipt is missing required fields")
    return raw


@dataclass(frozen=True)
class TrainingProtocol:
    """Frozen comparison identity and required completed training steps."""

    workload_id: str
    gate_steps: int
    benchmark_steps: int

    def parse_receipt(self, output: str, phase: str) -> dict:
        """Validate terminal evidence for a gate or benchmark invocation.

        Args:
            output: Combined command output.
            phase: Expected gate or benchmark phase.
        Returns:
            Validated runtime evidence, including a failure status if reported.
        Raises:
            ValueError: The receipt is missing, malformed or belongs to a different trial.
        """
        raw = _read_receipt(output)
        if raw["workload_id"] != self.workload_id or raw["phase"] != phase:
            raise ValueError("training receipt workload or phase does not match the frozen contract")
        if not isinstance(raw["status"], str) or raw["status"] not in _RECEIPT_STATUSES:
            raise ValueError("unknown training receipt status")
        if not _is_count(raw["completed_steps"], 0):
            raise ValueError("completed_steps must be a nonnegative integer")
        if not isinstance(raw["engaged"], bool):
            raise ValueError("engaged must be a boolean")
        receipt = {key: raw[key] for key in _RECEIPT_FIELDS}
        for key, minimum in (("allocation_retries", 0), ("first_error_step", 1)):
            value = raw.get(key)
            if value is not None and not _is_count(value, minimum):
                raise ValueError(f"{key} must be an integer >= {minimum} or null")
            receipt[key] = value
        return receipt


def _is_count(value: object, minimum: int) -> bool:
    """Recognize a JSON integer count, excluding JSON booleans."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


@dataclass
class RunConfig:
    """Validated contents of a run directory's ``run.json``.

    Attributes:
        run_dir: Absolute path of the run directory.
        repo_root: Absolute path of the enclosing git repository.
        target_files: Repo-relative files the agent is allowed to modify.
        benchmark_cmd: Argv list producing the metric line on stdout.
        gate_cmd: Optional argv list for a separate correctness gate;
            empty when the benchmark command already runs the gate.
        metric_pattern: Regex with a ``metric`` named group, applied to the
            combined benchmark output; the last match wins.
        metric_lower_is_better: Direction of improvement.
        noise_fraction: Relative change below which a result is ``discard``.
        noise_confirm_reruns: Extra benchmark runs when a result lands inside
            the noise band; the decision then uses the median of all samples.
            0 (the default) keeps the single-run behaviour.
        aux_patterns: Optional named regexes (each with a ``value`` group)
            captured from benchmark output and recorded as ``name=value``
            in the result row's description, e.g. a wall time or peak memory.
        bench_must_match: Literal substrings that must appear in benchmark
            output (e.g. an engagement marker); absence means ``crash``.
        gate_must_match: Literal substrings required in gate output.
        results_file: Run-relative TSV path (append-only bookkeeping).
        log_file: Run-relative experiment log path.
        training_protocol: Optional contract for structured training receipts.
    """

    run_dir: Path
    repo_root: Path
    target_files: list[str]
    benchmark_cmd: list[str]
    gate_cmd: list[str] = field(default_factory=list)
    metric_pattern: str = r"total_ms=(?P<metric>[0-9.]+)"
    metric_lower_is_better: bool = True
    noise_fraction: float = 0.02
    noise_confirm_reruns: int = 0
    aux_patterns: dict[str, str] = field(default_factory=dict)
    bench_must_match: list[str] = field(default_factory=list)
    gate_must_match: list[str] = field(default_factory=list)
    results_file: str = "results.tsv"
    log_file: str = "experiment_log.md"
    training_protocol: TrainingProtocol | None = None

    @property
    def results_path(self) -> Path:
        """Absolute path of the results TSV."""
        return self.run_dir / self.results_file

    @property
    def log_path(self) -> Path:
        """Absolute path of the experiment log."""
        return self.run_dir / self.log_file

    def extract_metric(self, output: str) -> float | None:
        """Return the last metric occurrence in ``output``, if any."""
        matches = list(re.finditer(self.metric_pattern, output))
        if not matches:
            return None
        try:
            metric = float(matches[-1].group(_METRIC_GROUP))
        except (TypeError, ValueError):
            return None
        return metric if math.isfinite(metric) else None

    def improved(self, candidate: float, best: float) -> bool:
        """Whether ``candidate`` beats ``best`` beyond the noise band."""
        if self.metric_lower_is_better:
            return candidate < best * (1.0 - self.noise_fraction)
        return candidate > best * (1.0 + self.noise_fraction)

    def within_noise(self, candidate: float, best: float) -> bool:
        """Whether ``candidate`` sits inside the noise band around ``best``."""
        return abs(candidate - best) <= abs(best) * self.noise_fraction

    def extract_aux(self, output: str) -> dict[str, str]:
        """Capture the last occurrence of each auxiliary pattern, if present."""
        captured: dict[str, str] = {}
        for name, pattern in self.aux_patterns.items():
            matches = list(re.finditer(pattern, output))
            if matches:
                captured[name] = matches[-1].group(_AUX_GROUP)
        return captured


def _require(condition: bool, message: str) -> None:
    """Raise ``ValueError`` with ``message`` unless ``condition`` holds."""
    if not condition:
        raise ValueError(f"autoresearch run.json: {message}")


def _load_training_protocol(raw: dict, gate_cmd: list) -> TrainingProtocol | None:
    """Validate the opt-in contract without changing legacy run configuration."""
    protocol = raw.get("training_protocol")
    if protocol is None:
        return None
    _require(isinstance(protocol, dict), "training_protocol must be an object")
    _require(set(protocol) == {"workload_id", "gate_steps", "benchmark_steps"},
             "training_protocol requires workload_id, gate_steps and benchmark_steps")
    _require(bool(gate_cmd), "training_protocol requires a separate gate_cmd")
    workload = protocol["workload_id"]
    _require(isinstance(workload, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", workload) is not None,
             "training_protocol.workload_id must be a simple identifier (1-128 characters)")
    for key in ("gate_steps", "benchmark_steps"):
        _require(_is_count(protocol[key], 1),
                 f"training_protocol.{key} must be a positive integer")
    return TrainingProtocol(**protocol)


def _load_aux_patterns(raw: dict) -> dict[str, str]:
    """Validate optional named evidence patterns."""
    aux_raw = raw.get("aux_patterns", {}) or {}
    _require(isinstance(aux_raw, dict), "aux_patterns must be a name->regex object")
    aux_patterns: dict[str, str] = {}
    for name, aux_pattern in aux_raw.items():
        _require(isinstance(name, str) and _AUX_NAME.match(name),
                 f"aux_patterns name is not a simple token: {name!r}")
        try:
            aux_compiled = re.compile(str(aux_pattern))
        except re.error as exc:
            raise ValueError(f"autoresearch run.json: bad aux pattern {name}: {exc}") from exc
        _require(_AUX_GROUP in aux_compiled.groupindex,
                 f"aux pattern {name} needs a (?P<value>...) group")
        aux_patterns[name] = str(aux_pattern)
    return aux_patterns


def load_run_config(run_dir: str | Path, repo_root: str | Path) -> RunConfig:
    """Load and validate ``run.json`` from ``run_dir``.

    Args:
        run_dir: The run directory containing ``run.json``.
        repo_root: Root of the git repository the run operates on.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: ``run.json`` is missing.
        ValueError: A field is missing, ill-typed, or unusable.
    """
    run_path = Path(run_dir).resolve()
    root = Path(repo_root).resolve()
    config_file = run_path / CONFIG_NAME
    if not config_file.is_file():
        raise FileNotFoundError(f"missing {config_file}")
    raw = json.loads(config_file.read_text(encoding="utf-8"))

    targets = raw.get("target_files")
    _require(isinstance(targets, list) and targets, "target_files must be a non-empty list")
    _require(all(isinstance(item, str) for item in targets), "target_files entries must be strings")
    for item in targets:
        _require((root / item).is_file(), f"target file not found: {item}")

    benchmark_cmd = raw.get("benchmark_cmd")
    _require(isinstance(benchmark_cmd, list) and benchmark_cmd,
             "benchmark_cmd must be a non-empty argv list")
    _require(all(isinstance(item, str) for item in benchmark_cmd),
             "benchmark_cmd entries must be strings")
    gate_cmd = raw.get("gate_cmd", []) or []
    _require(isinstance(gate_cmd, list), "gate_cmd must be an argv list")

    pattern = raw.get("metric_pattern", RunConfig.metric_pattern)
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"autoresearch run.json: bad metric_pattern: {exc}") from exc
    _require(_METRIC_GROUP in compiled.groupindex, "metric_pattern needs a (?P<metric>...) group")

    noise = float(raw.get("noise_fraction", RunConfig.noise_fraction))
    _require(0.0 <= noise < 1.0, "noise_fraction must be in [0, 1)")

    reruns = int(raw.get("noise_confirm_reruns", RunConfig.noise_confirm_reruns))
    _require(0 <= reruns <= 10, "noise_confirm_reruns must be in [0, 10]")

    return RunConfig(
        run_dir=run_path,
        repo_root=root,
        target_files=list(targets),
        benchmark_cmd=list(benchmark_cmd),
        gate_cmd=[str(item) for item in gate_cmd],
        metric_pattern=pattern,
        metric_lower_is_better=bool(raw.get("metric_lower_is_better", True)),
        noise_fraction=noise,
        noise_confirm_reruns=reruns,
        aux_patterns=_load_aux_patterns(raw),
        bench_must_match=[str(item) for item in raw.get("bench_must_match", [])],
        gate_must_match=[str(item) for item in raw.get("gate_must_match", [])],
        results_file=str(raw.get("results_file", RunConfig.results_file)),
        log_file=str(raw.get("log_file", RunConfig.log_file)),
        training_protocol=_load_training_protocol(raw, gate_cmd),
    )
