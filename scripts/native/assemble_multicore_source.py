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
"""Assemble the isolated HyperMegaMoe source closure from pinned kernel sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

from scripts.native.prepare_dependencies import verify_git_dependency


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCK_PATH = _REPO_ROOT / "scripts" / "native" / "config" / "dependencies.lock.json"
_MULTICORE_OPS = _REPO_ROOT / "hyper_parallel" / "core" / "multicore" / "ops"
_OPS_NN_PATHS = (
    "activation/swi_glu/op_kernel",
    "activation/swi_glu_grad/op_kernel",
)
_OPS_TRANSFORMER_PATHS = ("gmm/grouped_matmul/op_kernel",)
_HYPER_OPERATORS = ("hyper_mega_moe", "hyper_mega_moe_grad")


def _parse_args() -> argparse.Namespace:
    """Parse the isolated source assembly contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops-nn-source", required=True)
    parser.add_argument("--ops-transformer-source", required=True)
    parser.add_argument("--work-dir", required=True)
    return parser.parse_args()


def main() -> int:
    """Verify, export, adapt, and compose the selected kernel sources."""
    args = _parse_args()
    lock = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))["components"]["multicore"]
    ops_nn_source = Path(args.ops_nn_source).resolve()
    ops_transformer_source = Path(args.ops_transformer_source).resolve()
    work_dir = Path(args.work_dir).resolve()
    _validate_new_work_dir(work_dir, (ops_nn_source, ops_transformer_source))

    verify_git_dependency(lock["ops_nn"], ops_nn_source, dependency_name="ops_nn")
    verify_git_dependency(
        lock["ops_transformer"],
        ops_transformer_source,
        dependency_name="ops_transformer",
    )

    ops_nn_copy = work_dir / "adapter-inputs" / "ops-nn"
    transformer_copy = work_dir / "adapter-inputs" / "ops-transformer"
    _export_git_tree(
        ops_nn_source,
        ops_nn_copy,
        lock["ops_nn"]["commit"],
        _OPS_NN_PATHS,
    )
    _export_git_tree(
        ops_transformer_source,
        transformer_copy,
        lock["ops_transformer"]["commit"],
        _OPS_TRANSFORMER_PATHS,
    )
    _apply_locked_adapters(ops_nn_copy, lock["ops_nn"])
    _apply_locked_adapters(transformer_copy, lock["ops_transformer"])

    source_root = work_dir / "source"
    _compose_hyper_parallel_ops(source_root, ops_nn_copy, transformer_copy)
    _require_assembled_files(source_root)
    print(json.dumps({"source_root": str(source_root)}, sort_keys=True))
    return 0


def _validate_new_work_dir(work_dir: Path, inputs: tuple[Path, ...]) -> None:
    """Reject broad, existing, or source-overlapping output paths."""
    protected = {Path("/").resolve(), _REPO_ROOT.resolve(), _REPO_ROOT.parent.resolve(), *inputs}
    if work_dir in protected or any(work_dir == path.parent for path in inputs):
        raise ValueError(f"Refusing unsafe multicore work directory: {work_dir}")
    if work_dir.exists():
        raise ValueError(f"Multicore work directory must not already exist: {work_dir}")
    work_dir.mkdir(parents=True)


def _export_git_tree(
    source_root: Path,
    destination_root: Path,
    commit: str,
    relative_paths: tuple[str, ...],
) -> None:
    """Export selected committed files from one verified dependency revision."""
    if destination_root.exists():
        raise ValueError(f"Git export destination already exists: {destination_root}")
    destination_root.mkdir(parents=True)
    command = ["git", "archive", "--format=tar", commit, *relative_paths]
    with tempfile.TemporaryFile() as archive_file:
        result = subprocess.run(
            command,
            cwd=source_root,
            check=False,
            stdout=archive_file,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(
                f"Failed to export locked dependency {source_root} at {commit}: {result.stderr.strip()}"
            )
        archive_file.seek(0)
        with tarfile.open(fileobj=archive_file, mode="r:") as source:
            _validate_tar_members(source.getmembers())
            source.extractall(destination_root)


def _apply_locked_adapters(source_root: Path, dependency_lock: dict[str, Any]) -> None:
    """Verify and apply fusion adapters only to the isolated source copy."""
    for adapter in dependency_lock.get("patches", []):
        adapter_path = (_REPO_ROOT / adapter["path"]).resolve()
        actual_hash = _sha256(adapter_path)
        if actual_hash != adapter["sha256"]:
            raise ValueError(
                f"Adapter hash mismatch for {adapter_path}: expected={adapter['sha256']}, actual={actual_hash}"
            )
        _run_git_apply(source_root, adapter_path, check_only=True)
        _run_git_apply(source_root, adapter_path, check_only=False)
        _run_git_apply(source_root, adapter_path, check_only=True, reverse=True)


def _run_git_apply(
    source_root: Path,
    adapter_path: Path,
    check_only: bool,
    reverse: bool = False,
) -> None:
    """Run Git's patch parser against one isolated source copy."""
    command = ["git", "apply", "--ignore-space-change"]
    if reverse:
        command.append("--reverse")
    if check_only:
        command.append("--check")
    command.append(str(adapter_path))
    environment = os.environ.copy()
    environment["GIT_CEILING_DIRECTORIES"] = str(source_root.parent)
    result = subprocess.run(
        command,
        cwd=source_root,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        phase = "reverse check" if reverse else "check" if check_only else "apply"
        raise ValueError(f"Adapter {phase} failed for {adapter_path}: {result.stdout.strip()}")


def _compose_hyper_parallel_ops(
    source_root: Path,
    ops_nn_copy: Path,
    transformer_copy: Path,
) -> None:
    """Compose HP operator code with selected adapted upstream kernel sources."""
    for operator_name in _HYPER_OPERATORS:
        operator_root = source_root / operator_name
        shutil.copytree(_MULTICORE_OPS / operator_name, operator_root)
        shutil.copytree(_MULTICORE_OPS / "runtime", operator_root / "op_kernel" / "runtime")
        shutil.copytree(
            ops_nn_copy / "activation" / "swi_glu" / "op_kernel",
            operator_root / "op_kernel" / "swi_glu",
        )
        shutil.copytree(
            transformer_copy / "gmm" / "grouped_matmul" / "op_kernel",
            operator_root / "op_kernel" / "grouped_matmul",
        )
    shutil.copytree(
        ops_nn_copy / "activation" / "swi_glu_grad" / "op_kernel",
        source_root / "hyper_mega_moe_grad" / "op_kernel" / "swi_glu_grad",
    )


def _require_assembled_files(source_root: Path) -> None:
    """Reject incomplete source closures before invoking the CANN toolchain."""
    required_paths = (
        source_root / "hyper_mega_moe" / "op_host" / "hyper_mega_moe_def.cpp",
        source_root / "hyper_mega_moe" / "op_kernel" / "hyper_mega_moe.cpp",
        source_root / "hyper_mega_moe" / "op_kernel" / "swi_glu" / "swi_glu.cpp",
        source_root / "hyper_mega_moe" / "op_kernel" / "grouped_matmul" / "grouped_matmul.cpp",
        source_root / "hyper_mega_moe_grad" / "op_host" / "hyper_mega_moe_grad_def.cpp",
        source_root / "hyper_mega_moe_grad" / "op_kernel" / "hyper_mega_moe_grad.cpp",
        source_root / "hyper_mega_moe_grad" / "op_kernel" / "swi_glu_grad" / "swi_glu_grad.cpp",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise ValueError(f"Incomplete HyperMegaMoe source closure: {missing}")


def _validate_archive_names(names: Any) -> None:
    """Reject absolute or parent-traversing archive members."""
    for name in names:
        member = Path(name)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"Unsafe path in Git archive: {name}")


def _validate_tar_members(members: list[tarfile.TarInfo]) -> None:
    """Allow only safe regular files and directories before archive extraction."""
    _validate_archive_names(member.name for member in members)
    for member in members:
        if member.issym() or member.islnk():
            raise ValueError(
                f"Archive links are not allowed in native build inputs: {member.name} -> {member.linkname}"
            )
        if not member.isfile() and not member.isdir():
            raise ValueError(
                f"Archive special files are not allowed in native build inputs: {member.name}"
            )


def _sha256(path: Path) -> str:
    """Return the SHA256 digest for one fusion adapter."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        sys.stderr.write(f"[HP-MULTICORE-SOURCE-ERROR] {error}\n")
        raise SystemExit(1) from error
