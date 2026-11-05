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
"""Artifact-layout resolution and atomic writes for codegen bundles.

The artifact bundle lives in a ``generated/`` directory and holds the
generated modeling file, its diff against the original, and copied remote-code
dependencies. Two layout anchors exist: the model-anchored default
(``hyper_parallel/models/<model_type>/generated``, one bundle per model type)
and an explicit directory for isolated model-construction workflows. The bundle is
written as one atomic unit: every file is first staged into a temporary
*sibling* directory, then the whole directory is swapped into place, so a
half-written bundle can never be imported.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import errno
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Iterator

# Bundle directory name shared by every layout constructor.
_GENERATED_DIR = "generated"

_LOGGER = logging.getLogger(__name__)
_LOCK_INITIALIZATION_GRACE_S = 1.0


@dataclass
class ArtifactLayout:
    """Describe the paths belonging to one generated bundle."""

    artifact_dir: str
    modeling_path: str
    diff_path: str
    init_path: str

    @classmethod
    def build_default(
        cls,
        artifact_dir: str | None = None,
        model_name: str | None = None,
    ) -> "ArtifactLayout":
        """Create the layout for model-construction codegen.

        ``artifact_dir`` defaults to ``Path.cwd()/generated``.
        """
        resolved_dir = os.path.abspath(artifact_dir or os.path.join(os.getcwd(), _GENERATED_DIR))
        modeling_name = _default_modeling_name(model_name)
        return cls(
            artifact_dir=resolved_dir,
            modeling_path=os.path.join(resolved_dir, modeling_name),
            diff_path=os.path.join(resolved_dir, modeling_name + ".diff"),
            init_path=os.path.join(resolved_dir, "__init__.py"),
        )

    @classmethod
    def build_model_anchored(
        cls,
        model_type: str,
        model_name: str | None = None,
    ) -> "ArtifactLayout":
        """Create the model-anchored layout for one model type.

        ``artifact_dir`` is ``hyper_parallel/models/<model_type>/generated``
        — exactly one bundle per model type, colocated with the model's
        adapter code.  When ``models/<model_type>/`` does not exist yet, a
        warning is emitted and the directory is created on demand, so a
        model without a family directory still gets its own bundle home
        instead of silently falling back to another location.
        """
        # ``artifact.py`` lives in ``hyper_parallel/codegen/``; the models
        # tree is its sibling — package-relative, never cwd-dependent.
        models_root = Path(__file__).resolve().parents[1] / "models"
        family_root = models_root / model_type
        if not family_root.is_dir():
            _LOGGER.warning(
                "codegen: model type %r has no directory under "
                "hyper_parallel/models yet; the bundle will be created at "
                "%s (add an adapter package there to colocate it with the "
                "model's conversion code)",
                model_type,
                os.path.join(str(family_root), _GENERATED_DIR),
            )
        artifact_dir = os.path.join(str(family_root), _GENERATED_DIR)
        modeling_name = _default_modeling_name(model_name)
        return cls(
            artifact_dir=artifact_dir,
            modeling_path=os.path.join(artifact_dir, modeling_name),
            diff_path=os.path.join(artifact_dir, modeling_name + ".diff"),
            init_path=os.path.join(artifact_dir, "__init__.py"),
        )


def _default_modeling_name(model_name: str | None) -> str:
    base = model_name or "model"
    # '' / '-' are not valid Python identifiers in module names.
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in base)
    return f"modeling_{safe}_gen_npu.py"


def resolve_default_artifact_layout(
    artifact_dir: str | None = None,
    model_name: str | None = None,
) -> ArtifactLayout:
    """Resolve the default model-construction artifact layout."""
    return ArtifactLayout.build_default(artifact_dir, model_name)


def resolve_model_artifact_layout(
    model_type: str,
    model_name: str | None = None,
) -> "ArtifactLayout":
    """Resolve the model-anchored layout for ``model_type``.

    The bundle directory is always ``hyper_parallel/models/<model_type>/
    generated`` — a strict one-to-one mapping keyed by the model type, with
    no aliasing between model types.  When ``models/<model_type>/`` does not
    exist yet, ``ArtifactLayout.build_model_anchored`` warns and the directory is
    created on demand by :func:`ensure_artifact_dir`.
    """
    return ArtifactLayout.build_model_anchored(model_type, model_name=model_name)


def ensure_artifact_dir(layout: ArtifactLayout) -> None:
    """Create the artifact directory (recursive, no-op if present).

    Raises a readable error when the directory cannot be created: the
    model-anchored default lives inside the installed ``hyper_parallel``
    package, which is read-only under a non-editable install.
    """
    try:
        Path(layout.artifact_dir).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"codegen: cannot create artifact directory {layout.artifact_dir}: {exc}. "
            "The default bundle location lives inside the hyper_parallel package; "
            "for a non-editable install either reinstall with 'pip install -e .' "
            "or point 'model.codegen_artifact_dir' at a writable directory in the YAML."
        ) from exc


def write_bundle_atomic(layout: ArtifactLayout, files: dict[str, str]) -> None:
    """Write the whole bundle into place as one atomic directory switch.

    ``files`` maps filename -> content; keys must be plain file names (not
    absolute paths) so nothing bypasses the staging directory.

    The bundle is first staged into a temporary *sibling* of ``artifact_dir``
    (never inside it), then the whole directory is swapped in.  On Windows
    ``os.replace`` cannot replace a non-empty target directory, so the swap is
    three steps: move the old bundle aside, rename the staged dir into place,
    then delete the backup.  A reader on ``artifact_dir`` therefore sees either
    the old complete bundle or the new complete bundle — never a mix.
    """
    ensure_artifact_dir(layout)
    parent = os.path.dirname(layout.artifact_dir) or "."

    tmpdir = tempfile.mkdtemp(prefix=".codegen_tmp_", dir=parent)
    old_dir = None
    try:
        _stage_bundle(tmpdir, files)

        if os.path.exists(layout.artifact_dir):
            old_dir = tempfile.mkdtemp(prefix=".codegen_old_", dir=parent)
            os.rmdir(old_dir)  # reserve a name; must be absent for os.replace
            os.replace(layout.artifact_dir, old_dir)
        os.replace(tmpdir, layout.artifact_dir)
        if old_dir is not None:
            shutil.rmtree(old_dir, ignore_errors=True)
            old_dir = None
    finally:
        # If the staged dir never made it into place, put the old bundle back.
        if old_dir is not None and os.path.isdir(old_dir):
            if not os.path.exists(layout.artifact_dir):
                os.replace(old_dir, layout.artifact_dir)
            else:
                shutil.rmtree(old_dir, ignore_errors=True)
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


def _stage_bundle(tmpdir: str, files: dict[str, str]) -> None:
    """Write the complete newly emitted file set into ``tmpdir``."""
    normalized = dict(files)
    for name in normalized:
        _validate_bundle_filename(name)

    init_present = "__init__.py" in normalized
    if not init_present:
        normalized["__init__.py"] = ""

    for name, content in normalized.items():
        # newline="" disables the platform's default CRLF translation so the
        # staged bytes equal the emitted text bytes exactly.  On Windows the
        # default (newline=None) would turn every LF into CRLF and make bundle
        # bytes platform-dependent.
        with open(
            os.path.join(tmpdir, name), "w", encoding="utf-8", newline=""
        ) as handle:
            handle.write(content)

def _validate_bundle_filename(name: str) -> None:
    """Reject absolute, nested, and parent-relative bundle paths."""
    if (
        not name
        or os.path.isabs(name)
        or os.path.basename(name) != name
        or "/" in name
        or "\\" in name
        or name in {".", ".."}
    ):
        raise ValueError(f"bundle file keys must be plain names, got {name!r}")


def clean_temp_artifacts(layout: ArtifactLayout) -> None:
    """Remove stale ``.codegen_tmp_*`` / ``.codegen_old_*`` dirs.

    Staging now happens in the artifact dir's *parent*, so these leftovers are
    scanned there rather than inside the bundle.
    """
    parent = os.path.dirname(layout.artifact_dir) or "."
    if not os.path.isdir(parent):
        return
    for name in os.listdir(parent):
        if name.startswith(".codegen_tmp_") or name.startswith(".codegen_old_"):
            shutil.rmtree(os.path.join(parent, name), ignore_errors=True)


@contextmanager
def artifact_generation_lock(
    layout: ArtifactLayout,
    *,
    timeout_s: float = 600.0,
) -> Iterator[None]:
    """Serialize bundle publication and recover locks left by dead owners."""
    parent = os.path.dirname(layout.artifact_dir) or "."
    bundle_name = os.path.basename(layout.artifact_dir.rstrip(os.sep)) or "generated"
    lock_path = os.path.join(parent, f".{bundle_name}.codegen.lock")
    deadline = time.monotonic() + timeout_s
    descriptor = None
    owner_identity = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(descriptor, str(os.getpid()).encode("ascii"))
            except BaseException:
                os.close(descriptor)
                descriptor = None
                try:
                    os.unlink(lock_path)
                except FileNotFoundError:
                    pass
                raise
            owner_identity = _file_identity(lock_path)
        except FileExistsError:
            if _remove_stale_lock(lock_path):
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"codegen: timed out waiting for artifact lock {lock_path}"
                ) from None
            time.sleep(0.1)
    try:
        yield
    finally:
        os.close(descriptor)
        if owner_identity is not None and _file_identity(lock_path) == owner_identity:
            try:
                os.unlink(lock_path)
            except FileNotFoundError:
                pass


def _remove_stale_lock(lock_path: str) -> bool:
    """Remove ``lock_path`` when its recorded owner process is dead."""
    try:
        identity = _file_identity(lock_path)
        stat_result = os.stat(lock_path)
        with open(lock_path, "r", encoding="ascii") as handle:
            value = handle.read().strip()
    except (FileNotFoundError, OSError, UnicodeError):
        return False

    try:
        pid = int(value)
    except ValueError:
        # There is a tiny interval between O_EXCL creation and writing the PID.
        # Only reclaim malformed locks after that initialization window.
        age_s = max(0.0, time.time() - stat_result.st_mtime)
        stale = age_s >= _LOCK_INITIALIZATION_GRACE_S
    else:
        stale = not _pid_is_alive(pid)
    if not stale or _file_identity(lock_path) != identity:
        return False
    try:
        os.unlink(lock_path)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    _LOGGER.warning("codegen: removed stale artifact lock %s", lock_path)
    return True


def _pid_is_alive(pid: int) -> bool:
    """Return whether ``pid`` still identifies a live process."""
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_is_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        return True
    return True


def _windows_pid_is_alive(pid: int) -> bool:
    """Query a Windows process without using ``os.kill(pid, 0)``.

    Windows does not implement the POSIX signal-zero probe; using it may
    terminate the target process. ``OpenProcess`` plus ``GetExitCodeProcess``
    is read-only and reports ``STILL_ACTIVE`` for a live owner.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    open_process.restype = ctypes.c_void_p
    get_exit_code = kernel32.GetExitCodeProcess
    get_exit_code.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
    get_exit_code.restype = ctypes.c_int
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int
    process = open_process(0x1000, False, pid)
    if not process:
        # Access denied proves that the process exists. Other errors are
        # treated conservatively except ERROR_INVALID_PARAMETER (no such PID).
        return ctypes.get_last_error() != 87
    try:
        exit_code = ctypes.c_ulong()
        if not get_exit_code(process, ctypes.byref(exit_code)):
            return True
        return exit_code.value == 259  # STILL_ACTIVE
    finally:
        close_handle(process)


def _file_identity(path: str) -> tuple[int, int, int] | None:
    """Return enough stat identity to avoid unlinking a replaced lock file."""
    try:
        result = os.stat(path)
    except FileNotFoundError:
        return None
    return result.st_ino, result.st_mtime_ns, result.st_size


__all__ = [
    "ArtifactLayout",
    "artifact_generation_lock",
    "clean_temp_artifacts",
    "ensure_artifact_dir",
    "resolve_default_artifact_layout",
    "resolve_model_artifact_layout",
    "write_bundle_atomic",
]
