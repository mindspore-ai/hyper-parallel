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
"""Run pylint for pre-commit on staged Python files, against a frozen baseline.

The repository carries pre-existing pylint debt that predates the current rule
set. Blocking every commit on it would only push authors to add local
suppressions, so the debt is *grandfathered*: ``pylint_baseline.json`` records
how many findings of each message id each file already had, and a commit fails
only when a file exceeds its recorded count. Existing debt is frozen, new debt
is rejected, and the count can only ratchet down.

Regenerate the baseline (after deliberately fixing or accepting debt) with::

    python3 scripts/pre-commit/run_pylint.py --update-baseline
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, List

BASELINE_NAME = "pylint_baseline.json"
# Findings are keyed by (path, message id) and counted. Line numbers deliberately
# play no part: any edit above a finding shifts them and would spuriously fail.
Baseline = Dict[str, Dict[str, int]]


def _repo_root() -> Path:
    """Resolve the repository root from the script location."""
    return Path(__file__).resolve().parents[2]


def _baseline_path() -> Path:
    """Return the on-disk location of the frozen baseline."""
    return Path(__file__).resolve().parent / BASELINE_NAME


def _collect_python_files(argv: List[str]) -> List[str]:
    """Keep only existing Python files from the pre-commit file list."""
    repo_root = _repo_root()
    files: List[str] = []

    for file_name in argv:
        path = Path(file_name)
        if path.suffix != ".py":
            continue
        absolute_path = path if path.is_absolute() else repo_root / path
        if absolute_path.is_file():
            files.append(str(path))

    return files


def _all_python_files() -> List[str]:
    """List every git-tracked Python file, the universe the baseline covers."""
    repo_root = _repo_root()
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _run_pylint(files: List[str]) -> List[dict]:
    """Run pylint over ``files`` and return its findings as JSON records.

    Args:
        files: Repository-relative paths to lint.

    Returns:
        One dict per finding, as emitted by pylint's ``json`` reporter. An empty
        list when pylint produced no parsable output.

    Raises:
        RuntimeError: If pylint fails without producing JSON (a crash or a bad
            configuration, which must not be silently treated as "no findings").
    """
    repo_root = _repo_root()
    command = [
        "pylint",
        f"--rcfile={repo_root / '.pylintrc'}",
        "--output-format=json",
        *files,
    ]
    result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, check=False)
    text = result.stdout.strip()
    if not text:
        # Exit code 32 is a usage error; anything non-zero with no JSON is a real
        # failure rather than a clean run.
        if result.returncode not in (0,):
            raise RuntimeError(f"pylint produced no output (exit {result.returncode}):\n{result.stderr}")
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"could not parse pylint output: {exc}\n{text[:2000]}") from exc


def _tally(findings: List[dict]) -> Baseline:
    """Count findings per (path, message id)."""
    counts: Dict[str, Counter] = {}
    for item in findings:
        path = item.get("path", "")
        message_id = item.get("message-id", "")
        if not path or not message_id:
            continue
        counts.setdefault(path, Counter())[message_id] += 1
    return {path: dict(counter) for path, counter in counts.items()}


def _load_baseline() -> Baseline:
    """Read the frozen baseline, treating a missing file as "no debt"."""
    path = _baseline_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _regressions(current: Baseline, baseline: Baseline) -> List[tuple]:
    """List every (path, message id, count, budget) whose count exceeds the baseline."""
    problems: List[tuple] = []
    for path, counts in sorted(current.items()):
        allowed = baseline.get(path, {})
        for message_id, count in sorted(counts.items()):
            budget = allowed.get(message_id, 0)
            if count > budget:
                problems.append((path, message_id, count, budget))
    return problems


def _update_baseline() -> int:
    """Relint the whole repository and overwrite the frozen baseline."""
    files = _all_python_files()
    print(f"pylint: baselining {len(files)} tracked Python files (this takes a few minutes)...")
    findings = _run_pylint(files)
    baseline = _tally(findings)
    total = sum(sum(counts.values()) for counts in baseline.values())
    _baseline_path().write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"pylint: baseline written with {total} findings across {len(baseline)} files.")
    return 0


def main() -> int:
    """Lint the staged files and fail only on findings beyond the baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="relint the whole repository and rewrite the frozen baseline",
    )
    parser.add_argument("files", nargs="*", help="files supplied by pre-commit")
    args = parser.parse_args()

    if args.update_baseline:
        return _update_baseline()

    files = _collect_python_files(args.files)
    if not files:
        print("pylint skipped: no Python files to check.")
        return 0

    findings = _run_pylint(files)
    problems = _regressions(_tally(findings), _load_baseline())
    if not problems:
        print(f"pylint: {len(files)} file(s) at or below baseline.")
        return 0

    print("pylint: new findings beyond the frozen baseline:")
    for path, message_id, count, budget in problems:
        print(f"  {path}: {message_id} {count} > {budget} allowed")
    # Print only the findings for the (file, message id) pairs that regressed;
    # a file's grandfathered findings of other kinds are not the author's problem.
    regressed = {(path, message_id) for path, message_id, _, _ in problems}
    print("")
    for item in findings:
        if (item.get("path"), item.get("message-id")) in regressed:
            print(f"  {item['path']}:{item.get('line')}: [{item.get('message-id')}] {item.get('message')}")
    print("\nFix them, or run --update-baseline if the debt is deliberate and reviewed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
