# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Explicitly acquire or verify pinned native build dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCK_PATH = _REPO_ROOT / "scripts" / "native" / "config" / "dependencies.lock.json"
_DEFAULT_CACHE = _REPO_ROOT / "build" / "native" / "deps"
_DEPENDENCIES = ("shmem", "ops_nn", "ops_transformer")


def _parse_args() -> argparse.Namespace:
    """Parse the explicit dependency acquisition contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dependency",
        action="append",
        required=True,
        choices=_DEPENDENCIES,
        help="Dependency to prepare. Repeat the option to prepare several dependencies.",
    )
    parser.add_argument("--cache-dir", default=str(_DEFAULT_CACHE))
    parser.add_argument("--source-dir")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Acquire requested dependencies, then verify their locked identities."""
    args = _parse_args()
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = _REPO_ROOT / cache_dir
    if args.source_dir and len(args.dependency) != 1:
        raise ValueError("--source-dir can only be used with one --dependency value.")
    prepared: dict[str, Any] = {}
    for dependency in args.dependency:
        source_dir = Path(args.source_dir) if args.source_dir else cache_dir / dependency / "src"
        if not source_dir.is_absolute():
            source_dir = _REPO_ROOT / source_dir
        dependency_lock = _read_dependency_lock(dependency)
        if not source_dir.exists():
            if args.verify_only:
                raise ValueError(f"Dependency source directory does not exist: {source_dir}")
            _clone_dependency(dependency_lock, source_dir)
        try:
            metadata = verify_git_dependency(dependency_lock, source_dir, dependency_name=dependency)
        except ValueError:
            if args.verify_only or args.source_dir:
                raise
            print(f"Refreshing managed dependency cache: {source_dir}", file=sys.stderr)
            shutil.rmtree(source_dir)
            _clone_dependency(dependency_lock, source_dir)
            metadata = verify_git_dependency(dependency_lock, source_dir, dependency_name=dependency)
        prepared[dependency] = metadata
    print(json.dumps(prepared, sort_keys=True))
    return 0


def verify_git_dependency(
    dependency_lock: dict[str, Any],
    source_dir: str | Path,
    dependency_name: str | None = None,
) -> dict[str, Any]:
    """Verify a Git dependency is the exact tracked source locked by this repository.

    Args:
        dependency_lock: Dependency entry from ``dependencies.lock.json``.
        source_dir: Existing Git worktree to verify.
        dependency_name: Stable lock name; inferred from the repository when omitted.

    Returns:
        Verified source identity suitable for a run record.

    Raises:
        ValueError: If the source is missing, modified, or does not match the lock.
    """
    source_dir = Path(source_dir).resolve()
    if not (source_dir / ".git").exists():
        raise ValueError(f"Dependency source is not a Git worktree: {source_dir}")
    actual_commit = _git_output(source_dir, "rev-parse", "HEAD")
    actual_tree = _git_output(source_dir, "rev-parse", "HEAD^{tree}")
    tracked_changes = _git_output(source_dir, "status", "--porcelain", "--untracked-files=no")
    if actual_commit != dependency_lock["commit"]:
        raise ValueError(f"Dependency commit mismatch: expected={dependency_lock['commit']}, actual={actual_commit}")
    if actual_tree != dependency_lock["tree"]:
        raise ValueError(f"Dependency tree mismatch: expected={dependency_lock['tree']}, actual={actual_tree}")
    if tracked_changes:
        raise ValueError("Dependency source contains tracked modifications; upstream patching is not allowed.")
    repository_name = Path(dependency_lock["repository"]).stem
    archive_hash = _git_archive_sha256(
        source_dir,
        dependency_lock["commit"],
        f"{repository_name}-{dependency_lock['version']}/",
    )
    if archive_hash != dependency_lock["git_archive_tar_sha256"]:
        raise ValueError(
            "Dependency archive hash mismatch: "
            f"expected={dependency_lock['git_archive_tar_sha256']}, actual={archive_hash}"
        )
    return {
        "dependency": dependency_name or repository_name.replace("-", "_"),
        "version": dependency_lock["version"],
        "repository": dependency_lock["repository"],
        "source_dir": str(source_dir),
        "commit": actual_commit,
        "tree": actual_tree,
        "git_archive_tar_sha256": archive_hash,
        "patched": False,
    }


def _read_dependency_lock(dependency: str) -> dict[str, Any]:
    """Read one dependency entry from the repository lock."""
    lock = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    component_name = "symmetric_memory" if dependency == "shmem" else "multicore"
    try:
        return lock["components"][component_name][dependency]
    except KeyError as error:
        raise ValueError(f"Dependency is not present in the native lock: {dependency}") from error


def _clone_dependency(dependency_lock: dict[str, Any], source_dir: Path) -> None:
    """Populate one pinned dependency from its declared tag or commit ref."""
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    fetch_ref = dependency_lock.get("fetch_ref")
    if fetch_ref:
        _fetch_dependency_commit(dependency_lock, source_dir, fetch_ref)
        return
    result = subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            dependency_lock["version"],
            dependency_lock["repository"],
            str(source_dir),
        ],
        cwd=_REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone pinned dependency into {source_dir}")


def _fetch_dependency_commit(dependency_lock: dict[str, Any], source_dir: Path, fetch_ref: str) -> None:
    """Fetch and check out a pinned upstream commit ref."""
    source_dir.mkdir(parents=True)
    commands = (
        ("git", "init", str(source_dir)),
        ("git", "-C", str(source_dir), "remote", "add", "origin", dependency_lock["repository"]),
        ("git", "-C", str(source_dir), "fetch", "--depth", "1", "origin", fetch_ref),
        ("git", "-C", str(source_dir), "checkout", "--detach", dependency_lock["commit"]),
    )
    for command in commands:
        result = subprocess.run(command, cwd=_REPO_ROOT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to fetch pinned dependency into {source_dir}")


def _git_output(source_dir: Path, *args: str) -> str:
    """Run one read-only Git inspection and return stripped stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=source_dir,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"Git inspection failed ({' '.join(args)}): {result.stderr.strip()}")
    return result.stdout.strip()


def _git_archive_sha256(source_dir: Path, commit: str, archive_prefix: str) -> str:
    """Hash a deterministic uncompressed Git archive without creating a tarball."""
    with subprocess.Popen(
        ["git", "archive", "--format=tar", f"--prefix={archive_prefix}", commit],
        cwd=source_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        if process.stdout is None:
            raise RuntimeError("Cannot read git archive output.")
        digest = hashlib.sha256()
        for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
            digest.update(chunk)
        _, stderr = process.communicate()
        if process.returncode != 0:
            raise ValueError(f"Git archive failed: {stderr.decode(errors='replace').strip()}")
    return digest.hexdigest()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        sys.stderr.write(f"[HP-NATIVE-DEPENDENCY-ERROR] {error}\n")
        raise SystemExit(1) from error
