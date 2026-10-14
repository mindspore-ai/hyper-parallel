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
"""Passive, rank-local SHMEM loader evidence for diagnostic precision runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_ENVIRONMENT_KEYS = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "NODE_NAME", "BUILD_URL", "PATH", "PYTHONPATH",
    "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "LD_DEBUG", "LD_DEBUG_OUTPUT",
    "ASCEND_HOME_PATH", "ASCEND_INSTALL_PATH", "ASCEND_CUSTOM_OPP_PATH", "ASCEND_RT_VISIBLE_DEVICES",
    "HYPER_PARALLEL_SHMEM_HEAP_SIZE", "HP_SHMEM_LOADER_EVIDENCE_DIR",
    "HP_SHMEM_LOADER_WORKER_SOURCE", "HP_SHMEM_LOADER_LAUNCHER_SOURCE",
)
_SOURCE_PATHS = {
    "worker": "HP_SHMEM_LOADER_WORKER_SOURCE", "launcher": "HP_SHMEM_LOADER_LAUNCHER_SOURCE",
}
_SYMBOLS = (
    "rtDeviceGetBareTgid", "rtGetDeviceInfo", "rtSetIpcMemorySuperPodPid", "rtIpcSetMemoryName",
    "rtIpcDestroyMemoryName", "rtIpcOpenMemory", "rtIpcCloseMemory", "rtGetLogicDevIdByUserDevId",
)
_LIBRARY_NAMES = (
    "libascendcl", "libruntime", "libascend_dump", "libhccl", "libtorch_npu",
    "shmem", "hybm", "hyper_parallel", "cust_opapi",
)
_SNAPSHOTS: dict[int, list[dict[str, Any]]] = {}
_LIBRARIES: dict[tuple[Any, ...], dict[str, Any]] = {}
_IDENTITIES: dict[int, dict[str, Any]] = {}
_MARKER = "[SHMEM-LOADER-EVIDENCE] "


def _tool_environment() -> dict[str, str]:
    """Keep inspection subprocesses out of the worker's loader trace files."""
    environment = dict(os.environ, LC_ALL="C")
    environment.pop("LD_DEBUG", None)
    environment.pop("LD_DEBUG_OUTPUT", None)
    return environment


def _sha256(path: Path) -> str:
    """Hash a mapped library without loading it into the observer process."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity() -> dict[str, Any]:
    """Identify a real source checkout and retain explicit archive metadata errors."""
    root = Path(__file__).resolve().parents[2]
    result: dict[str, Any] = {
        "source_sha": None, "collector_sha256": _sha256(Path(__file__)),
    }
    for name, key in _SOURCE_PATHS.items():
        source = os.getenv(key)
        result[f"{name}_source"] = source
        result[f"{name}_sha256"] = None
        if source is None:
            result[f"{name}_sha256_error"] = "diagnostic source path is not configured"
        else:
            result[f"{name}_sha256"] = _sha256(Path(source))
    try:
        fields = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel", "HEAD"], cwd=root, text=True, stderr=subprocess.PIPE,
            env=_tool_environment(), timeout=10,
        ).splitlines()
        if len(fields) == 2 and Path(fields[0]).resolve() == root:
            result["source_sha"] = fields[1]
        else:
            result["source_sha_error"] = "source directory is not the Git checkout root"
    except (OSError, subprocess.SubprocessError) as error:
        result["source_sha_error"] = str(error)
    return result


def _mapped_libraries(text: str) -> list[dict[str, Any]]:
    """Retain actual CANN and private-library paths with their mapped inodes."""
    libraries: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = fields[5].removesuffix(" (deleted)")
        if ".so" not in Path(path).name:
            continue
        if not ("/Ascend/" in path or any(name in Path(path).name for name in _LIBRARY_NAMES)):
            continue
        library = libraries.setdefault(path, {"path": path, "inode": int(fields[4]), "mappings": []})
        library["mappings"].append(line)
    return [libraries[path] for path in sorted(libraries)]


def _readelf(path: Path, option: str) -> dict[str, Any]:
    """Inspect ELF metadata and report missing tools or failed commands explicitly."""
    executable = shutil.which("readelf")
    if executable is None:
        return {"error": "readelf is not installed"}
    try:
        result = subprocess.run(
            [executable, option, "--wide", str(path)], text=True, capture_output=True,
            check=False, timeout=15, env=_tool_environment(),
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": str(error)}
    if result.returncode:
        return {"error": result.stderr.strip(), "exit_code": result.returncode}
    return {"output": result.stdout}


def _symbol_definitions(text: str) -> dict[str, list[dict[str, Any]]]:
    """Separate defined Runtime symbols from undefined references and retain versions."""
    symbols: dict[str, list[dict[str, Any]]] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 8:
            continue
        name = fields[7].split("@", 1)[0]
        if name not in _SYMBOLS:
            continue
        symbols.setdefault(name, []).append({
            "defined": fields[6] != "UND", "binding": fields[4], "visibility": fields[5],
            "value": fields[1], "name": fields[7],
        })
    return symbols


def _library_identity(path: Path) -> dict[str, Any]:
    """Cache disk hashes, dependency names and Runtime exports by file identity."""
    resolved = path.resolve()
    stat = resolved.stat()
    key = (str(resolved), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _LIBRARIES:
        dynamic, symbols = _readelf(resolved, "--dynamic"), _readelf(resolved, "--dyn-syms")
        text = dynamic.get("output", "")
        _LIBRARIES[key] = {
            "real_path": str(resolved), "inode": stat.st_ino, "size": stat.st_size,
            "sha256": _sha256(resolved),
            "needed": re.findall(r"\(NEEDED\).*\[(.*?)\]", text),
            "search_paths": re.findall(r"\((?:RPATH|RUNPATH)\).*\[(.*?)\]", text),
            "symbols": _symbol_definitions(symbols.get("output", "")),
            "errors": [result["error"] for result in (dynamic, symbols) if "error" in result],
        }
        after = resolved.stat()
        _LIBRARIES[key]["file_changed_during_collection"] = (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ) != key[1:]
    return _LIBRARIES[key]


def _snapshot(stage: str) -> dict[str, Any]:
    """Observe mapped libraries and imported packages without importing frameworks."""
    libraries = _mapped_libraries(Path("/proc/self/maps").read_text(encoding="utf-8"))
    for library in libraries:
        try:
            identity = _library_identity(Path(library["path"]))
            library["identity"] = identity
            library["mapped_inode_matches_file"] = library["inode"] == identity["inode"]
        except OSError as error:
            library["error"] = str(error)
    modules = {}
    for name in ("torch", "torch_npu", "hyper_parallel", "hyper_parallel_shmem_torch"):
        module = sys.modules.get(name)
        if module is not None:
            modules[name] = {"file": getattr(module, "__file__", None),
                             "version": str(getattr(module, "__version__", "unknown"))}
    return {"stage": stage, "time_ns": time.time_ns(), "libraries": libraries, "modules": modules}


def capture_loader_snapshot(stage: str) -> None:
    """Persist one passive snapshot per stage without replacing the worker's failure.

    Args:
        stage: Stable checkpoint name shared by the master and PR workers.
    """
    directory = os.getenv("HP_SHMEM_LOADER_EVIDENCE_DIR")
    if not directory:
        return
    pid = os.getpid()
    snapshots = _SNAPSHOTS.setdefault(pid, [])
    if any(item["stage"] == stage for item in snapshots):
        return
    try:
        if pid not in _IDENTITIES:
            _IDENTITIES[pid] = {
                **_source_identity(), "pid": pid, "hostname": platform.node(),
                "machine": platform.machine(), "libc": platform.libc_ver(),
                "python": sys.version, "executable": str(Path(sys.executable).resolve()),
                "environment": {key: os.getenv(key) for key in _ENVIRONMENT_KEYS},
            }
        snapshots.append(_snapshot(stage))
        path = Path(directory) / f"rank-{os.getenv('RANK', 'unknown')}-pid-{pid}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"identity": _IDENTITIES[pid], "snapshots": snapshots}) + "\n", encoding="utf-8")
        temporary.replace(path)
    except Exception as error:  # pylint: disable=broad-exception-caught
        # Diagnostic failures are reported separately from the actual worker exception.
        print(_MARKER + json.dumps({"diagnostic_error": str(error), "stage": stage, "pid": pid}),
              file=sys.stderr, flush=True)


def _trace_excerpt(path: Path) -> dict[str, Any]:
    """Retain bounded lookup evidence while recording how much trace was omitted."""
    symbols, libraries = deque(maxlen=128), deque(maxlen=128)
    symbol_lines_total = library_lines_total = 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if any(name in line for name in _SYMBOLS):
                symbols.append(line.rstrip())
                symbol_lines_total += 1
            elif any(name in line for name in _LIBRARY_NAMES):
                libraries.append(line.rstrip())
                library_lines_total += 1
    return {"path": str(path), "symbol_lines": list(symbols), "library_lines": list(libraries),
            "symbol_lines_total": symbol_lines_total, "library_lines_total": library_lines_total}


def _report_evidence(directory: Path) -> None:
    """Replay rank evidence and relevant loader traces into the launcher output."""
    for path in sorted(directory.glob("rank-*.json")):
        print(_MARKER + path.read_text(encoding="utf-8").strip(), flush=True)
    for path in sorted(directory.glob("loader.*")):
        print(_MARKER + json.dumps({"loader_trace": _trace_excerpt(path)}), flush=True)
    if not list(directory.glob("rank-*.json")):
        print(_MARKER + json.dumps({"diagnostic_error": "no rank snapshots", "directory": str(directory)}), flush=True)
    if not list(directory.glob("loader.*")):
        print(_MARKER + json.dumps({"diagnostic_error": "no loader traces", "directory": str(directory)}), flush=True)


@contextmanager
def loader_evidence_session(
    monkeypatch: Any, *, export_on_success: bool = False,
    worker_source: str | Path | None = None, launcher_source: str | Path | None = None,
) -> Iterator[Path]:
    """Collect and replay evidence even when a precision worker succeeds.

    Args:
        monkeypatch: Pytest fixture that restores the launcher's environment.
        export_on_success: Emit a deliberate archive-triggering failure after successful workers.
            Reserved for a temporary diagnostic PR; ordinary precision runs leave this disabled.
        worker_source: Actual worker file whose identity is recorded in each rank.
        launcher_source: Actual launcher file, independent of the collector's location.

    Yields:
        Directory containing every rank snapshot and loader trace.
    """
    directory = Path.cwd() / "logs" / "shmem_loader" / f"{time.time_ns()}-{os.getpid()}"
    directory.mkdir(parents=True)
    monkeypatch.setenv("HP_SHMEM_LOADER_EVIDENCE_DIR", str(directory))
    monkeypatch.setenv("LD_DEBUG", "libs,scopes")
    monkeypatch.setenv("LD_DEBUG_OUTPUT", str(directory / "loader"))
    for name, source in (("worker", worker_source), ("launcher", launcher_source)):
        if source is None:
            monkeypatch.delenv(_SOURCE_PATHS[name], raising=False)
        else:
            monkeypatch.setenv(_SOURCE_PATHS[name], str(Path(source).resolve()))
    workers_succeeded = False
    try:
        yield directory
        workers_succeeded = True
    finally:
        try:
            result = {"precision_workers_succeeded": workers_succeeded,
                      "export_on_success": export_on_success, "directory": str(directory)}
            print(_MARKER + json.dumps(result), flush=True)
            _report_evidence(directory)
        except Exception as error:  # pylint: disable=broad-exception-caught
            # Evidence replay must preserve the original worker failure.
            print(_MARKER + json.dumps({"diagnostic_error": str(error), "directory": str(directory)}), flush=True)
    if export_on_success:
        raise RuntimeError("SHMEM_DIAGNOSTIC_EXPORT: precision workers passed; deliberate failure to archive evidence")
