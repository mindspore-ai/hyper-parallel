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
"""Bundle emission: the entry the codegen manager calls to produce an artifact.

``emit_bundle`` is what the manager calls: it is invoked as
``emit_bundle(layout, meta)`` after the native plan has
been derived and frozen into ``meta``.  So this layer never derives anything —
it renders the already-frozen contract into a ``files`` dict
(filename -> content), and the manager stages + atomically swaps the whole
bundle (including ``codegen_meta.json``) in one directory switch.

The whole bundle is written as one atomic unit: ``emit_bundle``
returns the modeling file, its diff, and ``__init__.py`` as a dict, and
``write_bundle_atomic`` stages them alongside the meta in a temp sibling then
swaps the directory in.  A half-written bundle can never be imported.

``meta`` carries the frozen plan, resolved source, and parallel dimensions, so
passing separate copies would create a second source of truth.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from hyper_parallel.codegen.artifact import ArtifactLayout
from hyper_parallel.codegen.emit.diff import unified_diff
from hyper_parallel.codegen.emit.modeling import (
    copy_original_modeling,
    emit_modeling_file,
)

logger = logging.getLogger(__name__)


def emit_bundle(
    layout: ArtifactLayout,
    meta: Any,
    *,
    patch_plan: Any = None,
    snapshot: Any = None,
) -> dict[str, str]:
    """Render the artifact bundle for one frozen ``meta``.

    Manager-facing entry point.  Returns a ``filename -> content`` dict
    holding the generated modeling file, the unified diff against the original
    source, and ``__init__.py``; the caller (manager) stages and atomically
    swaps the bundle, then writes ``codegen_meta.json`` last inside the
    staging dir.

    ``meta.covered`` is prepared by the manager before emission.
    """
    if snapshot is not None:
        snapshot_payload = snapshot.to_dict()
        if getattr(meta, "snapshot_status", None) != snapshot_payload["status"]:
            raise ValueError("bundle meta and snapshot status do not match")
    if getattr(meta, "snapshot_status", None) == "not_required":
        logger.info("codegen: rendered audit-only bundle %s", layout.artifact_dir)
        return {"__init__.py": _strip_trailing_whitespace(emit_init_file(layout))}
    if patch_plan is None:
        from hyper_parallel.codegen.emit.patches import emission_patches_from_meta

        patch_plan = emission_patches_from_meta(meta)
    original_text = copy_original_modeling(meta.source)
    generated_text = _strip_trailing_whitespace(
        emit_modeling_file(meta, patch_plan=patch_plan)
    )

    modeling_name = os.path.basename(layout.modeling_path)
    diff_text = _strip_trailing_whitespace(
        emit_diff(original_text, generated_text, layout)
    )

    logger.info(
        "codegen: rendered bundle %s (modeling %s, %d plan boundaries)",
        layout.artifact_dir,
        modeling_name,
        len(getattr(meta, "param_plan", None) or {}),
    )
    files = {
        # Filename keys (not absolute paths): write_bundle_atomic stages them
        # in a temp sibling and swaps the whole directory in.
        modeling_name: generated_text,
        os.path.basename(layout.diff_path): diff_text,
        "__init__.py": _strip_trailing_whitespace(emit_init_file(layout)),
    }
    # A ``<remote>`` source is
    # shipped inside the checkpoint and uses single-dot relative imports
    # (``from .configuration_deepseek import ...``) that resolve against its
    # own package.  The loader compiles the generated file as the synthetic
    # package ``hyper_parallel_generated.<digest>``, so those siblings must be
    # IN the bundle and registered under that package.  Copy the transitive
    # closure of the source's relative imports here (original basenames), and
    # record what was copied so preflight/loader can tell a stale bundle.
    siblings: dict[str, str] = {}
    source_obj = meta.source
    source_module = (
        source_obj.get("module_name") if isinstance(source_obj, dict)
        else getattr(source_obj, "module_name", None)
    )
    if source_module == "<remote>":
        from hyper_parallel.codegen.source.resolver import resolve_remote_siblings

        resolved = resolve_remote_siblings(source_obj)
        for path in resolved.values():
            basename = os.path.basename(path)
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                siblings[basename] = handle.read()
        files.update(siblings)
        if siblings:
            meta.remote_siblings = sorted(siblings)
    return files


def _strip_trailing_whitespace(text: str) -> str:
    """Keep generated artifacts clean without changing their line structure."""
    return "".join(f"{line.rstrip()}\n" for line in text.splitlines())


def emit_init_file(layout: ArtifactLayout) -> str:
    """Render the bundle's ``__init__.py`` text.

    The bundle is an audit/export artifact, so this file carries no re-exports.
    """
    return (
        '"""Generated codegen bundle — DO NOT EDIT MANUALLY.\n'
        "\n"
        f"Generated from: {layout.yaml_path}\n"
        "Regenerate with ``python -m hyper_parallel.codegen generate``; the\n"
        "bundle is rebuilt whenever the snapshot signature changes.\n"
        '"""\n'
    )


def emit_diff(original_text: str, generated_text: str, layout: ArtifactLayout) -> str:
    """Render the unified diff from the original source to the generated file.

    Returned rather than written so ``emit_bundle`` can write the whole bundle
    in one atomic call.
    """
    return unified_diff(
        original_text,
        generated_text,
        fromfile=f"a/{_source_basename(layout)}",
        tofile=f"b/{os.path.basename(layout.modeling_path)}",
    )


def _source_basename(layout: ArtifactLayout) -> str:
    """Diff ``fromfile`` label: the original modeling filename."""
    name = os.path.basename(layout.modeling_path)
    if name.endswith("_gen_npu.py"):
        return name[: -len("_gen_npu.py")] + ".py"
    return name


__all__ = [
    "emit_bundle",
    "emit_diff",
    "emit_init_file",
    "emit_modeling_file",
]
