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
"""Assemble a verified, isolated multicore vendor source tree."""

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
import zipfile

from scripts.native.prepare_dependencies import verify_git_dependency


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCK_PATH = _REPO_ROOT / "scripts" / "native" / "config" / "dependencies.lock.json"
_MULTICORE_OPS = _REPO_ROOT / "hyper_parallel" / "core" / "multicore" / "ops"
_OPS_NN_PATHS = (
    "cmake",
    "common",
    "control",
    "scripts",
    "conv/common/op_kernel",
    "matmul/common/cmct",
    "activation/common/op_api",
    "activation/swi_glu/op_kernel",
    "activation/swi_glu_grad/op_kernel",
    "index/common/op_api",
    "norm/common/op_api",
    "pooling/common/op_api",
    "build.sh",
    "CMakeLists.txt",
    "install_deps.sh",
    "version.cmake",
)
_OPS_TRANSFORMER_PATHS = ("gmm/grouped_matmul/op_kernel",)
_HYPER_OPERATORS = ("hyper_mega_moe", "hyper_mega_moe_grad")


def _parse_args() -> argparse.Namespace:
    """Parse the isolated source assembly contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops-nn-source", required=True)
    parser.add_argument("--ops-transformer-source", required=True)
    parser.add_argument("--cann-cmake-source", required=True)
    parser.add_argument("--opbase-source", required=True)
    parser.add_argument("--ops-tensor-source", required=True)
    parser.add_argument("--third-party-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    return parser.parse_args()


def main() -> int:
    """Verify inputs, apply adapters to copies, and assemble the build source."""
    args = _parse_args()
    lock = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))["components"]["multicore"]
    ops_nn_source = Path(args.ops_nn_source).resolve()
    ops_transformer_source = Path(args.ops_transformer_source).resolve()
    cann_cmake_source = Path(args.cann_cmake_source).resolve()
    opbase_source = Path(args.opbase_source).resolve()
    ops_tensor_source = Path(args.ops_tensor_source).resolve()
    third_party_dir = Path(args.third_party_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    _validate_new_work_dir(
        work_dir,
        (
            ops_nn_source,
            ops_transformer_source,
            cann_cmake_source,
            opbase_source,
            ops_tensor_source,
            third_party_dir,
        ),
    )

    verify_git_dependency(lock["ops_nn"], ops_nn_source, dependency_name="ops_nn")
    verify_git_dependency(
        lock["ops_transformer"],
        ops_transformer_source,
        dependency_name="ops_transformer",
    )
    verify_git_dependency(
        lock["cann_cmake"],
        cann_cmake_source,
        dependency_name="cann_cmake",
    )
    verify_git_dependency(lock["opbase"], opbase_source, dependency_name="opbase")
    verify_git_dependency(
        lock["ops_tensor"],
        ops_tensor_source,
        dependency_name="ops_tensor",
    )

    source_root = work_dir / "source"
    transformer_copy = work_dir / "adapter-inputs" / "ops-transformer"
    _export_git_tree(ops_nn_source, source_root, lock["ops_nn"]["commit"], _OPS_NN_PATHS)
    _export_git_tree(
        ops_transformer_source,
        transformer_copy,
        lock["ops_transformer"]["commit"],
        _OPS_TRANSFORMER_PATHS,
    )
    _apply_locked_adapters(source_root, lock["ops_nn"])
    _apply_locked_adapters(transformer_copy, lock["ops_transformer"])
    _copy_hyper_parallel_ops(source_root, transformer_copy)
    _stage_build_archives(
        third_party_dir,
        source_root / "third_party",
        work_dir / "archive-extract",
        lock["build_archives"],
    )
    _export_git_tree(
        cann_cmake_source,
        source_root / "third_party" / "cann-cmake",
        lock["cann_cmake"]["commit"],
    )
    _export_git_tree(
        opbase_source,
        source_root / "third_party" / "opbase",
        lock["opbase"]["commit"],
    )
    _export_git_tree(
        ops_tensor_source,
        source_root.parent / "ops-tensor",
        lock["ops_tensor"]["commit"],
    )

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
    relative_paths: tuple[str, ...] = (),
) -> None:
    """Export only committed files from one verified dependency revision."""
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
    """Verify and apply adapters only to the isolated source copy."""
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
    """Run Git's patch parser against a generated source copy."""
    # The locked ops-nn tree contains both LF and CRLF files. Ignore only
    # whitespace changes while the forward/reverse checks still require every
    # functional patch hunk to match the locked source.
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


def _copy_hyper_parallel_ops(source_root: Path, transformer_copy: Path) -> None:
    """Compose HP-owned operator/runtime code with the adapted upstream kernels."""
    category_root = source_root / "mega_moe"
    for operator_name in _HYPER_OPERATORS:
        operator_root = category_root / operator_name
        shutil.copytree(_MULTICORE_OPS / operator_name, operator_root)
        shutil.copytree(_MULTICORE_OPS / "runtime", operator_root / "op_kernel" / "runtime")
        shutil.copytree(
            source_root / "activation" / "swi_glu" / "op_kernel",
            operator_root / "op_kernel" / "swi_glu",
        )
        shutil.copytree(
            transformer_copy / "gmm" / "grouped_matmul" / "op_kernel",
            operator_root / "op_kernel" / "grouped_matmul",
        )
    shutil.copytree(
        source_root / "activation" / "swi_glu_grad" / "op_kernel",
        category_root / "hyper_mega_moe_grad" / "op_kernel" / "swi_glu_grad",
    )


def _stage_build_archives(
    cache_dir: Path,
    destination_root: Path,
    extraction_root: Path,
    archive_lock: dict[str, dict[str, str]],
) -> None:
    """Verify every CANN build archive and stage it without any build-time network access."""
    package_root = destination_root / "pkg"
    package_root.mkdir(parents=True)
    for name, archive in archive_lock.items():
        source = cache_dir / archive["file"]
        if not source.is_file():
            raise ValueError(
                f"Required multicore build archive is missing: {source}; fetch the locked URL explicitly first"
            )
        actual_hash = _sha256(source)
        if actual_hash != archive["sha256"]:
            raise ValueError(
                f"Build archive hash mismatch for {source}: expected={archive['sha256']}, actual={actual_hash}"
            )
        staging = archive["staging"]
        if staging == "archive":
            shutil.copy2(source, destination_root / archive["file"])
        elif staging.startswith("extract:"):
            destination_name = staging.split(":", maxsplit=1)[1]
            _extract_archive(source, destination_root / destination_name, extraction_root / name)
        else:
            raise ValueError(f"Unsupported build archive staging policy: {staging}")


def _extract_archive(archive: Path, destination: Path, temporary_root: Path) -> None:
    """Safely unpack one verified archive and flatten its optional top-level directory."""
    temporary_root.mkdir(parents=True)
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as source:
            _validate_tar_members(source.getmembers())
            source.extractall(temporary_root)
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            _validate_archive_names(source.namelist())
            source.extractall(temporary_root)
    else:
        raise ValueError(f"Unsupported build archive format: {archive}")
    entries = [path for path in temporary_root.iterdir() if path.name != "__MACOSX"]
    payload_root = entries[0] if len(entries) == 1 and entries[0].is_dir() else temporary_root
    shutil.copytree(payload_root, destination)


def _validate_archive_names(names: Any) -> None:
    """Reject absolute or parent-traversing archive members."""
    for name in names:
        member = Path(name)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"Unsafe path in build archive: {name}")


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
    """Return the SHA256 digest for one adapter file."""
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
