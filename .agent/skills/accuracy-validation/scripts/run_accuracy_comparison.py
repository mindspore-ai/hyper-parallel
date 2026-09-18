#!/usr/bin/env python3
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
"""Execute a frozen baseline/candidate accuracy-comparison manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from string import Formatter
from typing import Dict, List, Mapping, Sequence, Tuple


SUPPORTED_TIERS = {"strict", "standard", "relaxed"}
SUBSTITUTION_KEYS = {
    "baseline_dir",
    "baseline_loss",
    "candidate_dir",
    "candidate_loss",
    "output_dir",
    "python",
    "repo_root",
}
REQUIRED_SECTIONS = (
    "reference",
    "candidate",
    "model",
    "data",
    "training",
    "parallel",
    "commands",
)


def _repo_root() -> Path:
    """Return the checkout root containing the accuracy skill."""
    return Path(__file__).resolve().parents[4]


def _load_manifest(path: Path) -> dict:
    """Load and validate a JSON manifest."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    missing = [section for section in REQUIRED_SECTIONS if section not in manifest]
    if missing:
        raise ValueError(f"manifest is missing required sections: {missing}")
    tier = manifest.get("tolerance_tier")
    if tier not in SUPPORTED_TIERS:
        raise ValueError(f"tolerance_tier must be one of {sorted(SUPPORTED_TIERS)}")
    if "REPLACE_ME" in json.dumps(manifest, sort_keys=True):
        raise ValueError("manifest still contains REPLACE_ME placeholders")
    commands = manifest["commands"]
    if not isinstance(commands, dict):
        raise ValueError("commands must be a JSON object")
    for phase in ("baseline", "candidate"):
        command = commands.get(phase)
        if not isinstance(command, list) or not command:
            raise ValueError(f"commands.{phase} must be a non-empty argument array")
        if not all(isinstance(argument, str) and argument for argument in command):
            raise ValueError(f"commands.{phase} arguments must be non-empty strings")
    return manifest


def _substitute(argument: str, values: Mapping[str, str]) -> str:
    """Expand one command argument using only declared runner substitutions."""
    for _, field_name, format_spec, conversion in Formatter().parse(argument):
        if field_name is None:
            continue
        if field_name not in SUBSTITUTION_KEYS or format_spec or conversion:
            allowed = ", ".join(sorted(SUBSTITUTION_KEYS))
            raise ValueError(f"unsupported command substitution {field_name!r}; allowed: {allowed}")
    try:
        expanded = argument.format_map(values)
    except KeyError as exc:
        missing = str(exc.args[0])
        allowed = ", ".join(sorted(SUBSTITUTION_KEYS))
        raise ValueError(f"unknown command substitution {missing!r}; allowed: {allowed}") from exc
    return expanded


def _expand_commands(manifest: dict, values: Mapping[str, str]) -> Dict[str, List[str]]:
    """Return shell-free expanded baseline and candidate commands."""
    return {
        phase: [_substitute(argument, values) for argument in manifest["commands"][phase]]
        for phase in ("baseline", "candidate")
    }


def _freeze_manifest(manifest: dict, destination: Path) -> None:
    """Write a manifest once, refusing to replace different frozen input."""
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        existing = destination.read_text(encoding="utf-8")
        if existing != serialized:
            raise ValueError(f"refusing to overwrite different frozen manifest: {destination}")
        return
    destination.write_text(serialized, encoding="utf-8")


def _run_phase(
    phase: str,
    command: Sequence[str],
    repo_root: Path,
    phase_dir: Path,
    loss_path: Path,
    manifest_path: Path,
) -> Tuple[int, Path]:
    """Execute one side and retain its combined process log."""
    phase_dir.mkdir(parents=True, exist_ok=True)
    log_path = phase_dir / "command.log"
    environment = os.environ.copy()
    environment.update(
        {
            "HP_VALIDATION_ROLE": phase,
            "HP_VALIDATION_OUTPUT": str(loss_path),
            "HP_VALIDATION_MANIFEST": str(manifest_path),
        }
    )
    try:
        result = subprocess.run(
            list(command),
            cwd=repo_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        log = result.stdout + result.stderr
        return_code = result.returncode
    except OSError as exc:
        log = f"failed to execute {command[0]!r}: {exc}\n"
        return_code = 2
    log_path.write_text(log, encoding="utf-8")
    if return_code == 0 and not loss_path.is_file():
        log_path.write_text(log + f"missing declared loss artifact: {loss_path}\n", encoding="utf-8")
        return_code = 2
    return return_code, log_path


def _run_comparison(
    repo_root: Path,
    baseline_loss: Path,
    candidate_loss: Path,
    tier: str,
    output_path: Path,
) -> int:
    """Execute the canonical loss comparator and persist its JSON output."""
    comparator = repo_root / ".agent/skills/accuracy-validation/scripts/compare_loss.py"
    command = [
        sys.executable,
        str(comparator),
        "--baseline",
        str(baseline_loss),
        "--candidate",
        str(candidate_loss),
        "--tier",
        tier,
    ]
    result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, check=False)
    output_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    return result.returncode


def _write_summary(output_dir: Path, status: str, phases: Mapping[str, dict]) -> None:
    """Write machine-readable and human-readable run summaries."""
    summary = {"status": status, "phases": phases}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = [
        "# Accuracy comparison summary",
        "",
        f"Overall status: **{status}**",
        "",
        "| Phase | Exit code | Artifact |",
        "| --- | ---: | --- |",
    ]
    for phase, result in phases.items():
        rows.append(f"| {phase} | {result['exit_code']} | `{result['artifact']}` |")
    (output_dir / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Execute the requested accuracy comparison."""
    args = _parse_args()
    repo_root = _repo_root()
    try:
        manifest = _load_manifest(args.manifest.resolve())
        output_dir = args.output.resolve()
        baseline_dir = output_dir / "baseline"
        candidate_dir = output_dir / "candidate"
        values = {
            "baseline_dir": str(baseline_dir),
            "baseline_loss": str(baseline_dir / "loss.jsonl"),
            "candidate_dir": str(candidate_dir),
            "candidate_loss": str(candidate_dir / "loss.jsonl"),
            "output_dir": str(output_dir),
            "python": sys.executable,
            "repo_root": str(repo_root),
        }
        commands = _expand_commands(manifest, values)
    except ValueError as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2

    if args.dry_run:
        print(json.dumps({"manifest": manifest, "commands": commands}, ensure_ascii=False, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_manifest = output_dir / "run_manifest.json"
    resolved_manifest = json.loads(json.dumps(manifest))
    resolved_manifest["commands"] = commands
    resolved_manifest["runner"] = {
        "python": sys.executable,
        "repo_root": str(repo_root),
    }
    phases: Dict[str, dict] = {}
    try:
        _freeze_manifest(resolved_manifest, frozen_manifest)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2

    for phase, phase_dir, loss_path in (
        ("baseline", baseline_dir, baseline_dir / "loss.jsonl"),
        ("candidate", candidate_dir, candidate_dir / "loss.jsonl"),
    ):
        return_code, log_path = _run_phase(
            phase,
            commands[phase],
            repo_root,
            phase_dir,
            loss_path,
            frozen_manifest,
        )
        phases[phase] = {"exit_code": return_code, "artifact": str(log_path)}
        if return_code != 0:
            _write_summary(output_dir, "ERROR", phases)
            return return_code

    comparison_path = output_dir / "compare_loss.json"
    return_code = _run_comparison(
        repo_root,
        baseline_dir / "loss.jsonl",
        candidate_dir / "loss.jsonl",
        manifest["tolerance_tier"],
        comparison_path,
    )
    phases["comparison"] = {"exit_code": return_code, "artifact": str(comparison_path)}
    status = "PASS" if return_code == 0 else "FAIL" if return_code == 1 else "ERROR"
    _write_summary(output_dir, status, phases)
    print(json.dumps({"status": status, "output": str(output_dir)}, ensure_ascii=False, indent=2))
    return return_code


if __name__ == "__main__":
    sys.exit(main())
