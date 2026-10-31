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

import ast
import logging
import os
import re
from typing import Any

from hyper_parallel.codegen.artifact import ArtifactLayout
from hyper_parallel.codegen.check.structure import verify_generated_source_structure
from hyper_parallel.codegen.emit.diff import unified_diff
from hyper_parallel.codegen.emit.modeling import (
    copy_original_modeling,
    emit_modeling_file,
    verify_recorded_patch_bundle,
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
    generated_text = _make_primary_executable(
        generated_text,
        architecture=getattr(meta, "model_class", None),
        artifact_dir=layout.artifact_dir,
    )
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
    from hyper_parallel.codegen.source.resolver import resolve_remote_siblings

    resolved = resolve_remote_siblings(source_obj)
    if resolved:
        bundled_siblings: dict[str, str] = {}
        original_siblings: dict[str, tuple[str, str]] = {}
        sibling_diffs: dict[str, str] = {}
        for path in resolved.values():
            basename = os.path.basename(path)
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                original_sibling = handle.read()
            generated_sibling = _strip_trailing_whitespace(
                _materialize_sibling(
                    original_sibling,
                    meta,
                    patch_plan,
                    source_path=path,
                )
            )
            if generated_sibling != _strip_trailing_whitespace(original_sibling):
                output_name = _generated_sibling_name(basename)
            else:
                output_name = basename
            bundled_siblings[basename] = output_name
            siblings[output_name] = generated_sibling
            original_siblings[output_name] = (basename, original_sibling)
        if bundled_siblings:
            files[modeling_name] = _rewrite_sibling_imports(files[modeling_name], bundled_siblings)
            files[os.path.basename(layout.diff_path)] = _strip_trailing_whitespace(
                emit_diff(original_text, files[modeling_name], layout)
            )
            siblings = {
                name: _rewrite_sibling_imports(text, bundled_siblings)
                for name, text in siblings.items()
            }
            for output_name, rendered_sibling in siblings.items():
                basename, original_sibling = original_siblings[output_name]
                if rendered_sibling == _strip_trailing_whitespace(original_sibling):
                    continue
                sibling_diffs[f"{output_name}.diff"] = _strip_trailing_whitespace(
                    unified_diff(
                        original_sibling,
                        rendered_sibling,
                        fromfile=f"a/{basename}",
                        tofile=f"b/{output_name}",
                    )
                )
        files.update(siblings)
        files.update(sibling_diffs)
        if siblings:
            meta.remote_siblings = sorted(siblings)
    verify_generated_source_structure(files)
    verify_recorded_patch_bundle(files, patch_plan=patch_plan)
    return files


def _make_primary_executable(
    source_text: str,
    *,
    architecture: str | None,
    artifact_dir: str,
) -> str:
    """Add direct-file package setup and the standard training footer."""
    if not architecture:
        return source_text
    entry_class = _resolve_entry_class(source_text, architecture)
    family = os.path.basename(os.path.dirname(os.path.normpath(artifact_dir)))
    package = f"hyper_parallel.models.{family}.generated"
    tree = ast.parse(source_text)
    insertion_line = 0
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(
            tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        insertion_line = tree.body[0].end_lineno or 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insertion_line = max(insertion_line, node.end_lineno or 0)
    lines = source_text.splitlines()
    bootstrap = [
        "",
        "# Direct execution keeps bundled relative imports in their package.",
        "if __name__ == \"__main__\" and __package__ is None:",
        f"    __package__ = {package!r}",
        "",
    ]
    lines[insertion_line:insertion_line] = bootstrap
    footer = [
        "",
        "",
        "if __name__ == \"__main__\":",
        "    from hyper_parallel.models._generated import run_generated_training",
        "",
        "    run_generated_training(",
        f"        globals(), {architecture!r}, model_class_name={entry_class!r},",
        "    )",
    ]
    return "\n".join([*lines, *footer])


def _resolve_entry_class(source_text: str, architecture: str) -> str:
    """Resolve the emitted entry class without model-family name checks."""
    classes = [
        node.name
        for node in ast.parse(source_text).body
        if isinstance(node, ast.ClassDef)
    ]
    if architecture in classes:
        return architecture
    task_suffixes = (
        "ForConditionalGeneration",
        "ForImageTextToText",
        "ForSequenceClassification",
        "ForCausalLM",
    )
    suffix = next(
        (candidate for candidate in task_suffixes if architecture.endswith(candidate)),
        None,
    )
    matches = [name for name in classes if suffix is not None and name.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(
        "codegen: generated primary source does not expose a unique training "
        f"entry for architecture {architecture!r}; candidates={matches!r}"
    )


def _materialize_sibling(
    source_text: str,
    meta: Any,
    patch_plan: Any,
    *,
    source_path: str,
) -> str:
    """Apply source-owned strategy and recorded patches to one sibling."""
    from hyper_parallel.codegen.emit.recorded import materialize_recorded_forwards
    from hyper_parallel.codegen.inline.pipeline import (
        render_inline_strategies_for_source_unit,
    )

    source = getattr(meta, "source", None)
    architecture = (
        source.get("architecture")
        if isinstance(source, dict)
        else getattr(source, "architecture", None)
    )
    model_type = getattr(meta, "model_class", None) or architecture
    rendered = render_inline_strategies_for_source_unit(
        source_text,
        meta,
        model_type=model_type,
        patch_plan=patch_plan,
    )
    return materialize_recorded_forwards(
        rendered,
        meta,
        patch_plan=patch_plan,
        source_path=source_path,
    )


def _generated_sibling_name(basename: str) -> str:
    stem, extension = os.path.splitext(basename)
    return f"{stem}_gen_npu{extension}"


def _rewrite_sibling_imports(
    source_text: str,
    bundled_siblings: dict[str, str],
) -> str:
    """Retarget sibling imports to source units shipped in the bundle."""
    rendered = source_text
    for original, generated in sorted(bundled_siblings.items()):
        original_module = os.path.splitext(original)[0]
        generated_module = os.path.splitext(generated)[0]
        rendered = re.sub(
            rf"(?m)^(\s*from\s+\.+){re.escape(original_module)}(\s+import\s+)",
            rf"\g<1>{generated_module}\g<2>",
            rendered,
        )
        rendered = re.sub(
            rf"(?m)^(\s*from\s+)"
            rf"transformers_modules(?:\.[A-Za-z_][A-Za-z0-9_]*)+\."
            rf"{re.escape(original_module)}(\s+import\s+)",
            rf"\g<1>.{generated_module}\g<2>",
            rendered,
        )
    return rendered


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
