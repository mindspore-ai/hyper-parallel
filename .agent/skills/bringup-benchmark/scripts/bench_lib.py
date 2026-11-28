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
"""Core library of the bring-up asset benchmark (the agent-level ruler).

The benchmark measures *assets* — the written bring-up workflow under
``.agent/skills/hf-model-integration`` plus the generic kit — not models and
not the framework. It has two layers:

- **Deterministic layer**: every zero-device check from the integration kit,
  run mechanically over a fixed model catalog. Each cell carries the
  checker's *expected correct verdict* for that model category (a tied
  release is expected to trip the tied-save detector — that is the detector
  working). A cell is ``ok`` when the observed verdict equals the expectation,
  so an asset regression shows up as flipped cells, each attributable to a
  concrete catalog row. Zero variance by construction; its resolution is its
  coverage, and what it cannot see is reported, not hidden.
- **Agent layer**: clean-room trials (see the integration skill's
  ``references/cleanroom-protocol.md``) scored from machine-readable trial
  outputs (``trial_meta.json`` + ``gaps.jsonl``). The score is stochastic, so
  the benchmark carries a calibration step that turns repeat-run spread into
  a minimum detectable difference, and refuses A/B conclusions below it.

This skill is part of the adjudication layer: it measures the evolvable
asset set and therefore must never live inside it or be edited by an
improvement proposal. Its own changes go through operator review.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

CATALOG_SCHEMA = "bringup-bench/catalog/v1"
TRIAL_META_SCHEMA = "bringup-bench/trial-meta/v1"
GAP_CATEGORIES = ("wrong", "missing", "contradictory")
GATE_VERDICTS = ("pass", "fail", "env_stop", "blocked", "not_reached")
_CHECK_SCRIPTS = {
    "released_layout": "check_released_layout.py",
    "tied_save": "check_tied_save.py",
}
_EXIT_VERDICT = {0: "pass", 2: "fail"}
PATHS_ENV = "BRINGUP_BENCH_PATHS"


class BenchError(ValueError):
    """Raised for malformed catalogs, trials, or impossible requests."""


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

@dataclass
class CatalogModel:
    """One validation-set entry: a model category with expected verdicts."""

    model_id: str
    family: str
    category: str
    checks: dict[str, str]


def load_catalog(path: Path) -> list[CatalogModel]:
    """Parse and validate the catalog file; reject unknown checks early."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != CATALOG_SCHEMA:
        raise BenchError(f"{path}: expected schema {CATALOG_SCHEMA}")
    models = []
    for row in raw.get("models", []):
        missing = {"id", "family", "category", "checks"} - set(row)
        if missing:
            raise BenchError(f"{path}: model row missing {sorted(missing)}")
        for check, expect in row["checks"].items():
            if check not in _CHECK_SCRIPTS:
                raise BenchError(f"{path}: unknown check '{check}' on {row['id']}")
            if expect not in ("pass", "fail"):
                raise BenchError(f"{path}: expect must be pass|fail, got '{expect}'")
        models.append(CatalogModel(row["id"], row["family"], row["category"],
                                   dict(row["checks"])))
    if not models:
        raise BenchError(f"{path}: catalog has no models")
    return models


def resolve_model_dirs(models: list[CatalogModel],
                       paths_file: Optional[Path]) -> dict[str, Path]:
    """Map catalog ids to local model directories.

    Site paths never live in the repository: they come from a local paths
    file (``id: directory`` mapping) named on the command line or through
    the ``BRINGUP_BENCH_PATHS`` environment variable. There is no default.
    """
    source = paths_file or os.environ.get(PATHS_ENV)
    if not source:
        raise BenchError(
            f"no paths file: pass --paths or set {PATHS_ENV} "
            "(local id->model-dir mapping; never committed)")
    mapping = yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}
    resolved = {}
    for model in models:
        entry = mapping.get(model.model_id)
        if not entry:
            raise BenchError(f"paths file lacks '{model.model_id}'")
        resolved[model.model_id] = Path(entry)
    return resolved


# ---------------------------------------------------------------------------
# Deterministic layer: the check suite
# ---------------------------------------------------------------------------

def _default_runner(cmd: list[str], timeout: int) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          check=False)
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
    return proc.returncode, "\n".join(tail)


def run_suite(assets_root: Path, models: list[CatalogModel],
              model_dirs: dict[str, Path], timeout: int = 900,
              runner: Callable[[list[str], int], tuple[int, str]] = _default_runner,
              ) -> dict[str, Any]:
    """Run every (check x model) cell against one materialized asset snapshot.

    ``assets_root`` is the repository snapshot under measurement — the checks
    are taken from *it*, so two snapshots of different asset versions compare
    those versions, not the operator's working tree.
    """
    script_dir = (Path(assets_root) / ".agent" / "skills"
                  / "hf-model-integration" / "scripts")
    cells = []
    for model in models:
        for check, expect in sorted(model.checks.items()):
            script = script_dir / _CHECK_SCRIPTS[check]
            if not script.is_file():
                cells.append({"model": model.model_id, "check": check,
                              "verdict": "unavailable", "expect": expect,
                              "ok": False, "evidence": f"missing {script.name}"})
                continue
            cmd = [sys.executable, str(script),
                   "--model-dir", str(model_dirs[model.model_id])]
            code, tail = runner(cmd, timeout)
            verdict = _EXIT_VERDICT.get(code, "error" if code != 1 else "skipped")
            ok = verdict == expect if verdict in ("pass", "fail") else False
            cells.append({"model": model.model_id, "check": check,
                          "verdict": verdict, "expect": expect, "ok": ok,
                          "exit": code, "evidence": tail})
    covered = sorted({m.category for m in models})
    return {"schema": "bringup-bench/suite/v1",
            "assets_root": str(assets_root),
            "cells": cells,
            "ok_cells": sum(1 for c in cells if c["ok"]),
            "total_cells": len(cells),
            "covered_categories": covered}


def diff_suites(card_a: dict[str, Any], card_b: dict[str, Any]) -> list[dict[str, Any]]:
    """Cells whose ok-status differs between two scorecards (A -> B)."""
    def _key(cell: dict[str, Any]) -> tuple[str, str]:
        return (cell["model"], cell["check"])

    cells_a = {_key(c): c for c in card_a["cells"]}
    cells_b = {_key(c): c for c in card_b["cells"]}
    if set(cells_a) != set(cells_b):
        raise BenchError("scorecards cover different cells; compare like with like")
    flips = []
    for cell_key in sorted(cells_a):
        before, after = cells_a[cell_key], cells_b[cell_key]
        if (before["ok"], before["verdict"]) != (after["ok"], after["verdict"]):
            flips.append({"model": cell_key[0], "check": cell_key[1],
                          "before": before["verdict"], "after": after["verdict"],
                          "regression": before["ok"] and not after["ok"]})
    return flips


# ---------------------------------------------------------------------------
# Agent layer: trial scoring
# ---------------------------------------------------------------------------

@dataclass
class TrialScore:
    """Mechanically derived score of one clean-room trial."""

    trial_dir: str
    model_id: str
    assets_ref: str
    gaps_total: int
    gaps_by_category: dict[str, int]
    gates_passed: int
    gates_failed: int
    env_stops: int
    inventions: int
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Serialize for the scorecard file."""
        return {"schema": "bringup-bench/trial-score/v1", **self.__dict__}


def _read_gaps(path: Path) -> list[dict[str, Any]]:
    gaps = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            gap = json.loads(line)
        except json.JSONDecodeError as err:
            raise BenchError(f"{path}:{lineno}: not JSON ({err})") from err
        missing = {"id", "category", "stage", "indicted_file", "summary"} - set(gap)
        if missing:
            raise BenchError(f"{path}:{lineno}: gap missing {sorted(missing)}")
        if gap["category"] not in GAP_CATEGORIES:
            raise BenchError(f"{path}:{lineno}: category must be one of "
                             f"{GAP_CATEGORIES}, got '{gap['category']}'")
        gaps.append(gap)
    return gaps


def _normalized_gates(gates: Any, source: Path) -> list[dict[str, Any]]:
    """Accept both documented gate shapes and refuse everything else.

    Canonical form is a list of {name, verdict, evidence}; the compact form
    is a {name: verdict} mapping (the scalar score never reads evidence —
    the prose report carries it). Anything else is a contract violation.
    """
    if isinstance(gates, dict):
        gates = [{"name": name, "verdict": verdict}
                 for name, verdict in gates.items()]
    if not isinstance(gates, list):
        raise BenchError(f"{source}: gates must be a list or a name->verdict "
                         "mapping")
    for gate in gates:
        if not isinstance(gate, dict):
            raise BenchError(f"{source}: each gate must be an object, "
                             f"got {type(gate).__name__}")
        if gate.get("verdict") not in GATE_VERDICTS:
            raise BenchError(f"{source}: gate verdict must be one of "
                             f"{GATE_VERDICTS}, got '{gate.get('verdict')}'")
    return gates


def score_trial(trial_dir: Path) -> TrialScore:
    """Parse one trial directory (trial_meta.json + gaps.jsonl) into a score."""
    trial_dir = Path(trial_dir)
    meta_path = trial_dir / "trial_meta.json"
    gaps_path = trial_dir / "gaps.jsonl"
    if not meta_path.is_file() or not gaps_path.is_file():
        raise BenchError(f"{trial_dir}: needs trial_meta.json and gaps.jsonl")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema") != TRIAL_META_SCHEMA:
        raise BenchError(f"{meta_path}: expected schema {TRIAL_META_SCHEMA}")
    gates = _normalized_gates(meta.get("gates", []), meta_path)
    gaps = _read_gaps(gaps_path)
    by_category = {category: 0 for category in GAP_CATEGORIES}
    for gap in gaps:
        by_category[gap["category"]] += 1
    return TrialScore(
        trial_dir=str(trial_dir),
        model_id=meta["model_id"],
        assets_ref=meta["assets_ref"],
        gaps_total=len(gaps),
        gaps_by_category=by_category,
        gates_passed=sum(1 for g in gates if g["verdict"] == "pass"),
        gates_failed=sum(1 for g in gates if g["verdict"] == "fail"),
        env_stops=sum(1 for g in gates if g["verdict"] == "env_stop"),
        inventions=int(meta.get("inventions", 0)),
        extras={"agent_model": meta.get("agent_model", ""),
                "stages": meta.get("stages", {})})


# ---------------------------------------------------------------------------
# Calibration and A/B verdicts
# ---------------------------------------------------------------------------

def calibrate(scores: list[TrialScore]) -> dict[str, Any]:
    """Turn repeat-run spread into the benchmark's resolution.

    The minimum detectable difference is deliberately conservative: the
    larger of the observed spread and twice the sample standard deviation,
    never below one gap. A difference of means at or below it is reported
    as indistinguishable — "cannot see" is a result, not a failure.
    """
    if len(scores) < 2:
        raise BenchError("calibration needs at least two repeat trials")
    refs = {s.assets_ref for s in scores}
    models = {s.model_id for s in scores}
    if len(refs) > 1 or len(models) > 1:
        raise BenchError(f"repeats must share assets_ref and model, got "
                         f"{sorted(refs)} x {sorted(models)}")
    totals = [s.gaps_total for s in scores]
    spread = max(totals) - min(totals)
    sd = statistics.stdev(totals)
    mdd = max(1, spread, math.ceil(2 * sd))
    return {"schema": "bringup-bench/resolution/v1",
            "n": len(totals), "model_id": scores[0].model_id,
            "assets_ref": scores[0].assets_ref,
            "gaps": {"mean": statistics.mean(totals), "sd": sd,
                     "min": min(totals), "max": max(totals), "spread": spread},
            "min_detectable_gap_difference": mdd}


def compare_trials(group_a: list[TrialScore], group_b: list[TrialScore],
                   resolution: dict[str, Any]) -> dict[str, Any]:
    """Three-valued A/B verdict on gap counts, gated by the calibration."""
    if not group_a or not group_b:
        raise BenchError("both arms need at least one trial")
    mdd = resolution["min_detectable_gap_difference"]
    mean_a = statistics.mean(s.gaps_total for s in group_a)
    mean_b = statistics.mean(s.gaps_total for s in group_b)
    delta = mean_b - mean_a
    if abs(delta) <= mdd:
        verdict = "indistinguishable"
    else:
        verdict = "better" if delta < 0 else "worse"
    return {"schema": "bringup-bench/ab/v1",
            "arm_a": {"n": len(group_a), "mean_gaps": mean_a},
            "arm_b": {"n": len(group_b), "mean_gaps": mean_b},
            "delta_b_minus_a": delta,
            "min_detectable_gap_difference": mdd,
            "verdict": verdict}


# ---------------------------------------------------------------------------
# Asset snapshots and self-check injections
# ---------------------------------------------------------------------------

def snapshot_assets(repo_root: Path, ref: str, dest: Path,
                    inject: Optional[Path] = None) -> None:
    """Materialize a detached worktree of ``ref`` at ``dest``; optionally
    apply one injection patch (self-check: a deliberately damaged asset)."""
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise BenchError(f"{dest} exists and is not empty")
    added = subprocess.run(
        ["git", "-C", str(repo_root), "-c", "core.longpaths=true",
         "worktree", "add", "--detach", str(dest), ref],
        capture_output=True, text=True, check=False)
    if added.returncode != 0:
        raise BenchError(f"worktree add failed for {ref} -> {dest}: "
                         f"{added.stderr.strip().splitlines()[-1] if added.stderr else 'no stderr'}")
    if inject is not None:
        apply = subprocess.run(["git", "-C", str(dest), "apply", "--ignore-whitespace",
                                str(Path(inject).resolve())],
                               capture_output=True, text=True, check=False)
        if apply.returncode != 0:
            raise BenchError(f"injection {inject} did not apply: {apply.stderr.strip()}")


def selfcheck_aa(card_a: dict[str, Any], card_b: dict[str, Any]) -> dict[str, Any]:
    """Deterministic-layer A/A: two runs of the same ref must be identical."""
    flips = diff_suites(card_a, card_b)
    return {"schema": "bringup-bench/selfcheck-aa/v1",
            "identical": not flips, "flips": flips}


def selfcheck_injection(baseline: dict[str, Any], injected: dict[str, Any],
                        injection_name: str) -> dict[str, Any]:
    """An injected asset damage must flip at least one cell to be 'detected'.

    An undetected injection is itself a finding: it names a category the
    validation set is blind to, and goes into the report's
    "what this ruler cannot see" section rather than being dropped.
    """
    flips = diff_suites(baseline, injected)
    regressions = [f for f in flips if f["regression"]]
    return {"schema": "bringup-bench/selfcheck-injection/v1",
            "injection": injection_name,
            "detected": bool(regressions),
            "flips": flips}


_MASK_ROOTS = ("hyper_parallel/models", "tests/ut/auto_models/models", "examples")
_MASK_TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".sh", ".txt", ".rst"}


def _move_to_quarantine(path: Path, snapshot_root: Path, quarantine: Path) -> str:
    relative = path.relative_to(snapshot_root).as_posix()
    dest = quarantine / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    path.rename(dest)
    return relative


def _residual_mentions(snapshot_root: Path, token: str, limit: int = 200) -> list[str]:
    hits = []
    for path in snapshot_root.rglob("*"):
        if len(hits) >= limit:
            break
        if not path.is_file() or path.suffix.lower() not in _MASK_TEXT_SUFFIXES:
            continue
        try:
            if path.stat().st_size > 1_000_000:
                continue
            if token in path.read_text(encoding="utf-8", errors="ignore").lower():
                hits.append(path.relative_to(snapshot_root).as_posix())
        except OSError:
            continue
    return hits


def mask_model_family(snapshot_root: Path, family: str,
                      quarantine: Path) -> dict[str, Any]:
    """Quarantine a model family's own implementation from a snapshot.

    Supports the second-operator protocol's resource asymmetry: when the
    drawn model already has an in-repo implementation, the agent arm must
    not see it — generic assets stay, the family's model directories, family
    tests and family examples move to the quarantine directory (outside the
    snapshot), and a manifest records every move plus the remaining textual
    mentions, which are disclosed rather than deleted. The first operator
    reviews the manifest and runs an import smoke before handover.
    """
    snapshot_root = Path(snapshot_root)
    quarantine = Path(quarantine)
    token = family.strip().lower()
    if len(token) < 3:
        raise BenchError("family token must be at least 3 characters")
    if quarantine.resolve().is_relative_to(snapshot_root.resolve()):
        raise BenchError("quarantine directory must live outside the snapshot")
    moved = []
    for root in _MASK_ROOTS:
        base = snapshot_root / root
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if token in entry.name.lower():
                moved.append(_move_to_quarantine(entry, snapshot_root, quarantine))
    return {"schema": "bringup-bench/mask/v1",
            "family": token,
            "snapshot_root": str(snapshot_root),
            "quarantine": str(quarantine),
            "moved": moved,
            "residual_mentions": _residual_mentions(snapshot_root, token)}


def parse_trial_ref(name: str) -> str:
    """Normalize a trial directory name to its repeat index for reports."""
    match = re.search(r"(\d+)$", Path(name).name)
    return match.group(1) if match else Path(name).name
