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
"""Run pylint for pre-commit on staged Python files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List


def _repo_root() -> Path:
    """Resolve the repository root from the script location."""
    return Path(__file__).resolve().parents[2]


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


def main() -> int:
    """Run pylint against the provided file list."""
    files = _collect_python_files(sys.argv[1:])
    if not files:
        print("pylint skipped: no Python files to check.")
        return 0

    repo_root = _repo_root()
    command = ["pylint", f"--rcfile={repo_root / '.pylintrc'}"] + files
    result = subprocess.run(command, cwd=repo_root, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
