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
import re
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_NAME = "run.json"
_METRIC_GROUP = "metric"
_AUX_GROUP = "value"
_AUX_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


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
        return float(matches[-1].group(_METRIC_GROUP))

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
        aux_patterns=aux_patterns,
        bench_must_match=[str(item) for item in raw.get("bench_must_match", [])],
        gate_must_match=[str(item) for item in raw.get("gate_must_match", [])],
        results_file=str(raw.get("results_file", RunConfig.results_file)),
        log_file=str(raw.get("log_file", RunConfig.log_file)),
    )
