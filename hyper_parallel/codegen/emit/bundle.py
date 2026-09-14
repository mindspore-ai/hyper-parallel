# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Bundle emission: the entry the codegen manager calls to produce an artifact.

``emit_bundle`` is the manager's ``emit_fn``: it is invoked as
``emit_fn(layout, meta)`` (manager.ensure_codegen_artifact) after the plan has
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


def emit_bundle(layout: ArtifactLayout, meta: Any) -> dict[str, str]:
    """Render the artifact bundle for one frozen ``meta``.

    Manager-compatible ``emit_fn``.  Returns a ``filename -> content`` dict
    holding the generated modeling file, the unified diff against the original
    source, and ``__init__.py``; the caller (manager) stages and atomically
    swaps the bundle, then writes ``codegen_meta.json`` last inside the
    staging dir.

    ``meta.covered`` is prepared by the manager before emission.
    """
    original_text = copy_original_modeling(meta.source)
    generated_text = emit_modeling_file(meta)

    modeling_name = os.path.basename(layout.modeling_path)
    diff_text = emit_diff(original_text, generated_text, layout)

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
        "__init__.py": emit_init_file(layout),
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
    source_obj = meta.source if isinstance(meta.source, dict) else meta.source
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


def emit_init_file(layout: ArtifactLayout) -> str:
    """Render the bundle's ``__init__.py`` text.

    The bundle is imported by file location (loader.import_generated_module),
    not as a package, so this file carries no re-exports — it only marks the
    directory and records what the bundle is for a reader who opens it.
    """
    return (
        '"""Generated Hyper codegen bundle — DO NOT EDIT MANUALLY.\n'
        "\n"
        f"Generated from: {layout.yaml_path}\n"
        "Regenerate by re-running training with ``codegen: true``; the bundle\n"
        "is rebuilt whenever the codegen signature changes.\n"
        '"""\n'
    )


def emit_modeling_file_text(meta: Any) -> str:
    """Render the generated modeling file text (see ``emit.modeling``)."""
    return emit_modeling_file(meta)


def emit_diff(original_text: str, generated_text: str, layout: ArtifactLayout) -> str:
    """Render the unified diff from the original source to the generated file.

    Returned rather than written so ``emit_bundle`` can write the whole bundle
    in one atomic call; ``emit.diff.write_diff`` is the standalone writer.
    """
    return unified_diff(
        original_text,
        generated_text,
        fromfile=f"a/{_source_basename(layout)}",
        tofile=f"b/{os.path.basename(layout.modeling_path)}",
    )


def format_generated_filename(model_name: str | None) -> str:
    """Return the generated modeling filename for ``model_name``.

    Delegates to the artifact layout's own naming so the emitted filename can
    never drift from what ``resolve_artifact_layout`` records in the layout and
    what ``loader.find_generated_modeling_file`` scans for
    (``modeling_*_gen_npu.py``).
    """
    from hyper_parallel.codegen.artifact import _default_modeling_name

    return _default_modeling_name(model_name)


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
    "format_generated_filename",
]
