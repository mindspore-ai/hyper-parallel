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
generated modeling file, its diff against the original, and
``codegen_meta.json``.  CLI workflows can anchor it beside a YAML file; model
construction workflows can use the current working directory or an explicit
artifact directory.  The bundle is written as one atomic unit: every file is first staged into a
temporary *sibling* directory, then the whole directory is swapped into place,
so a half-written bundle can never be imported.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ArtifactLayout:
    """Describe the bundle path set for one YAML."""

    yaml_path: str
    artifact_dir: str
    modeling_path: str
    diff_path: str
    meta_path: str
    init_path: str

    @classmethod
    def build(cls, yaml_path: str, model_name: str | None = None) -> "ArtifactLayout":
        """Create the layout from a YAML path and optional model name.

        ``artifact_dir`` is ``<yaml_dir>/generated/``. Example directories
        contain one Codegen YAML, so another per-YAML directory is redundant.
        """
        yaml_path = os.path.abspath(yaml_path)
        artifact_dir = os.path.join(os.path.dirname(yaml_path), "generated")
        modeling_name = _default_modeling_name(model_name)
        return cls(
            yaml_path=yaml_path,
            artifact_dir=artifact_dir,
            modeling_path=os.path.join(artifact_dir, modeling_name),
            diff_path=os.path.join(artifact_dir, modeling_name + ".diff"),
            meta_path=os.path.join(artifact_dir, "codegen_meta.json"),
            init_path=os.path.join(artifact_dir, "__init__.py"),
        )

    @classmethod
    def build_default(
        cls,
        artifact_dir: str | None = None,
        model_name: str | None = None,
    ) -> "ArtifactLayout":
        """Create the layout for model-construction codegen.

        ``artifact_dir`` defaults to ``Path.cwd()/generated``.  ``yaml_path``
        is kept as an empty string for backward-compatible metadata shape; the
        signature is computed from the resolved config projection, not from a
        YAML file path.
        """
        resolved_dir = os.path.abspath(artifact_dir or os.path.join(os.getcwd(), "generated"))
        modeling_name = _default_modeling_name(model_name)
        return cls(
            yaml_path="",
            artifact_dir=resolved_dir,
            modeling_path=os.path.join(resolved_dir, modeling_name),
            diff_path=os.path.join(resolved_dir, modeling_name + ".diff"),
            meta_path=os.path.join(resolved_dir, "codegen_meta.json"),
            init_path=os.path.join(resolved_dir, "__init__.py"),
        )

    def require(self) -> "ArtifactLayout":
        """Validate the layout is fully populated and return self."""
        ensure_artifact_dir(self)
        return self


def _default_modeling_name(model_name: str | None) -> str:
    base = model_name or "model"
    # '' / '-' are not valid Python identifiers in module names.
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in base)
    return f"modeling_{safe}_gen_npu.py"


def resolve_artifact_layout(
    yaml_path: str, model_name: str | None = None
) -> ArtifactLayout:
    """Resolve the artifact layout for ``yaml_path``."""
    return ArtifactLayout.build(yaml_path, model_name)


def resolve_default_artifact_layout(
    artifact_dir: str | None = None,
    model_name: str | None = None,
) -> ArtifactLayout:
    """Resolve the default model-construction artifact layout."""
    return ArtifactLayout.build_default(artifact_dir, model_name)


def ensure_artifact_dir(layout: ArtifactLayout) -> None:
    """Create the artifact directory (recursive, no-op if present)."""
    Path(layout.artifact_dir).mkdir(parents=True, exist_ok=True)


def artifact_exists(layout: ArtifactLayout) -> bool:
    """True if both the meta and modeling files exist."""
    return os.path.isfile(layout.meta_path) and os.path.isfile(layout.modeling_path)


def write_bundle_atomic(
    layout: ArtifactLayout, files: dict[str, str], meta: object = None
) -> None:
    """Write the whole bundle into place as one atomic directory switch.

    ``files`` maps filename -> content; keys must be plain file names (not
    absolute paths) so nothing bypasses the staging directory.  ``meta``, when
    given, is hashed against the *staged* files and written
    ``codegen_meta.json`` LAST inside the staging directory, so meta being
    present AND parseable always means the rest of the bundle is already there.

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
        _stage_bundle(tmpdir, layout, files, has_meta=meta is not None)
        if meta is not None:
            _stage_meta(tmpdir, layout, meta)

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


def _stage_bundle(
    tmpdir: str,
    layout: ArtifactLayout,
    files: dict[str, str],
    *,
    has_meta: bool,
) -> None:
    """Copy any existing bundle files plus ``files`` into ``tmpdir``.

    Existing non-temp files are preserved so a partial update (e.g.
    ``copy_source_to_artifact`` adding ``source_*.py``) never drops the rest of
    the bundle; a key in ``files`` always wins over the preserved copy.  When
    ``has_meta`` is False the prior ``codegen_meta.json`` is also preserved —
    otherwise the caller is going to write a fresh one into the staging dir.
    """

    def _is_tmp_name(name: str) -> bool:
        return name.startswith(".codegen_tmp_") or name.startswith(".codegen_old_")

    normalized = dict(files)
    for name in normalized:
        _validate_bundle_filename(name)

    init_present = "__init__.py" in normalized
    if not has_meta and os.path.isdir(layout.artifact_dir):
        for name in os.listdir(layout.artifact_dir):
            if _is_tmp_name(name):
                continue
            if name == "codegen_meta.json":
                continue  # preserved below, after the loop
            if name in normalized:
                continue
            src = os.path.join(layout.artifact_dir, name)
            if not os.path.isfile(src):
                continue
            if name == "__init__.py":
                init_present = True
            shutil.copy2(src, os.path.join(tmpdir, name))

    if not init_present:
        normalized["__init__.py"] = ""

    for name, content in normalized.items():
        # newline="" disables the platform's default CRLF translation so the
        # staged bytes equal the emitted text bytes exactly.  On Windows the
        # default (newline=None) would turn every LF into CRLF, inflating each
        # file by one byte per line and breaking the drift/hash checks that
        # compare on-disk bytes against the re-emitted LF text.
        with open(
            os.path.join(tmpdir, name), "w", encoding="utf-8", newline=""
        ) as handle:
            handle.write(content)

    if not has_meta and os.path.isfile(layout.meta_path):
        shutil.copy2(
            layout.meta_path, os.path.join(tmpdir, os.path.basename(layout.meta_path))
        )


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


def _stage_meta(tmpdir: str, layout: ArtifactLayout, meta: object) -> None:
    """Hash the staged content files then write ``codegen_meta.json`` last."""
    from hyper_parallel.codegen.meta import (
        record_output_hashes,
        write_codegen_meta,
    )

    record_output_hashes(meta, layout, base_dir=tmpdir)
    meta_path = os.path.join(tmpdir, os.path.basename(layout.meta_path))
    write_codegen_meta(meta, meta_path)


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


__all__ = [
    "ArtifactLayout",
    "artifact_exists",
    "clean_temp_artifacts",
    "ensure_artifact_dir",
    "resolve_default_artifact_layout",
    "resolve_artifact_layout",
    "write_bundle_atomic",
]
