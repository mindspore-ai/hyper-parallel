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
"""Run lizard for pre-commit on staged source files."""

from __future__ import annotations

import csv
import io
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List

LIZARD_CCN = 15
LIZARD_NLOC = 100
SOURCE_SUFFIXES = {".py", ".c", ".cc", ".cpp", ".h", ".hpp"}
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _repo_root() -> Path:
    """Resolve the repository root from the script location."""
    return Path(__file__).resolve().parents[2]


def _collect_source_files(argv: List[str]) -> List[str]:
    """Keep only existing lizard-supported source files from pre-commit input."""
    repo_root = _repo_root()
    files: List[str] = []

    for file_name in argv:
        path = Path(file_name)
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        absolute_path = path if path.is_absolute() else repo_root / path
        if absolute_path.is_file():
            files.append(str(path))

    return files


def main() -> int:
    """Run lizard against the provided file list."""
    files = _collect_source_files(sys.argv[1:])
    if not files:
        print("lizard skipped: no source files to check.")
        return 0

    repo_root = _repo_root()
    command = [
        "lizard",
        f"-T cyclomatic_complexity={LIZARD_CCN}",
        f"-T nloc={LIZARD_NLOC}",
        "-w",
    ] + files
    from_ref = os.environ.get("PRE_COMMIT_FROM_REF")
    to_ref = os.environ.get("PRE_COMMIT_TO_REF")
    if from_ref and to_ref:
        return _run_incremental(repo_root, files, from_ref, to_ref)
    result = subprocess.run(command, cwd=repo_root, check=False)
    return result.returncode


def _run_incremental(repo_root: Path, files: List[str], from_ref: str, to_ref: str) -> int:
    """Reject complex functions only when the pushed diff touches their body."""
    changed = _changed_line_ranges(repo_root, files, from_ref, to_ref)
    result = subprocess.run(
        ["lizard", "--csv"] + files,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        sys.stderr.write(result.stderr)
        return result.returncode
    failures = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) < 11:
            continue
        nloc, complexity = int(row[0]), int(row[1])
        filename, function_name = row[6], row[7]
        start, end = int(row[9]), int(row[10])
        if nloc <= LIZARD_NLOC and complexity <= LIZARD_CCN:
            continue
        if not _overlaps(changed.get(filename, ()), start, end):
            continue
        failures.append(
            f"{filename}:{start}: warning: {function_name} has {nloc} NLOC, "
            f"{complexity} CCN"
        )
    if failures:
        print("\n".join(failures))
        return 1
    return 0


def _changed_line_ranges(
    repo_root: Path, files: List[str], from_ref: str, to_ref: str
) -> dict[str, list[tuple[int, int]]]:
    """Return added/modified line ranges for each file in a pushed ref range."""
    result = subprocess.run(
        ["git", "diff", "--unified=0", from_ref, to_ref, "--"] + files,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    ranges: dict[str, list[tuple[int, int]]] = {}
    current = None
    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
            continue
        match = _HUNK_RE.match(line)
        if current is None or match is None:
            continue
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        if count:
            ranges.setdefault(current, []).append((start, start + count - 1))
    return ranges


def _overlaps(ranges: list[tuple[int, int]], start: int, end: int) -> bool:
    """Return whether a function intersects an added or modified line range."""
    return any(range_start <= end and start <= range_end for range_start, range_end in ranges)


if __name__ == "__main__":
    raise SystemExit(main())
