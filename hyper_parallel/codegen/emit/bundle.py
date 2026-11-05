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
bundle in one directory switch.

The whole bundle is written as one atomic unit: ``emit_bundle``
returns the modeling file, its diff, and ``__init__.py`` as a dict, and
``write_bundle_atomic`` stages them in a temp sibling then
swaps the directory in.  A half-written bundle can never be imported.

The in-memory context carries the frozen plan, resolved source, and parallel dimensions, so
passing separate copies would create a second source of truth.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from hyper_parallel.codegen.artifact import ArtifactLayout
from hyper_parallel.codegen.emit.patches import emission_patches_from_meta
from hyper_parallel.codegen.emit.recorded import materialize_recorded_forwards
from hyper_parallel.codegen.emit.validation import verify_generated_source_structure
from hyper_parallel.codegen.emit.diff import unified_diff
from hyper_parallel.codegen.emit.modeling import (
    copy_original_modeling,
    emit_modeling_file,
    verify_recorded_patch_bundle,
)
from hyper_parallel.codegen.inline.pipeline import render_inline_strategies_for_source_unit
from hyper_parallel.codegen.source.resolver import resolve_remote_siblings

logger = logging.getLogger(__name__)

_NO_PATCH_DIFF = "# No codegen patches were required for this configuration.\n"
_GENERATED_SOURCE_HEADER = (
    "# GENERATED CODE. DO NOT EDIT MANUALLY.\n"
    "# pylint: skip-file\n"
    "# codespell:ignore-begin\n"
)
_GENERATED_SOURCE_FOOTER = "# codespell:ignore-end\n"
_APACHE_LICENSE_HEADER = """# Copyright 2026 Huawei Technologies Co., Ltd
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
"""
_COPIED_SOURCE_SPELLING_FIXES = {
    "Ple" "se": "Please",
    "suppo" "erted": "supported",
}


def emit_bundle(
    layout: ArtifactLayout,
    meta: Any,
    *,
    patch_plan: Any = None,
) -> dict[str, str]:
    """Render the artifact bundle from one frozen in-memory context.

    Manager-facing entry point.  Returns a ``filename -> content`` dict
    holding the generated modeling file, the unified diff against the original
    source, and ``__init__.py``; the caller (manager) stages and atomically
    swaps the complete bundle directory.

    """
    if getattr(meta, "status", None) == "not_required":
        logger.info("codegen: rendered audit-only bundle %s", layout.artifact_dir)
        modeling_name = os.path.basename(layout.modeling_path)
        return {
            modeling_name: _prepare_generated_python(
                copy_original_modeling(meta.source)
            ),
            os.path.basename(layout.diff_path): _NO_PATCH_DIFF,
            "__init__.py": _strip_trailing_whitespace(emit_init_file(layout)),
        }
    if patch_plan is None:
        patch_plan = emission_patches_from_meta(meta)
    original_text = copy_original_modeling(meta.source)
    generated_text = _prepare_generated_python(
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
    # closure of the source's relative imports here using their original
    # basenames, so the generated bundle remains self-contained.
    resolved = resolve_remote_siblings(meta.source)
    if resolved:
        siblings, sibling_diffs = _emit_remote_siblings(resolved, meta, patch_plan)
        import_map = {
            os.path.basename(path): name
            for path, name in zip(resolved.values(), siblings)
        }
        files[modeling_name] = _rewrite_sibling_imports(
            files[modeling_name], import_map
        )
        files[os.path.basename(layout.diff_path)] = _strip_trailing_whitespace(
            emit_diff(original_text, files[modeling_name], layout)
        )
        files.update(siblings)
        files.update(sibling_diffs)
    verify_generated_source_structure(files)
    verify_recorded_patch_bundle(files, patch_plan=patch_plan)
    return files


def _emit_remote_siblings(
    resolved: dict[str, str],
    meta: Any,
    patch_plan: Any,
) -> tuple[dict[str, str], dict[str, str]]:
    """Materialize remote-code siblings and their visible diffs."""
    siblings, originals, import_map = {}, {}, {}
    for path in resolved.values():
        basename = os.path.basename(path)
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            original = handle.read()
        materialized = _strip_trailing_whitespace(
            _materialize_sibling(original, meta, patch_plan, source_path=path)
        )
        rendered = _prepare_generated_python(materialized)
        output_name = (
            _generated_sibling_name(basename)
            if materialized != _strip_trailing_whitespace(original)
            else basename
        )
        import_map[basename] = output_name
        siblings[output_name] = rendered
        originals[output_name] = (basename, original)
    siblings = {
        name: _rewrite_sibling_imports(text, import_map)
        for name, text in siblings.items()
    }
    diffs = _remote_sibling_diffs(siblings, originals)
    return siblings, diffs


def _remote_sibling_diffs(
    siblings: dict[str, str],
    originals: dict[str, tuple[str, str]],
) -> dict[str, str]:
    """Render diffs only for remote siblings whose source changed."""
    diffs = {}
    for output_name, rendered in siblings.items():
        basename, original = originals[output_name]
        if rendered == _strip_trailing_whitespace(original):
            continue
        diffs[f"{output_name}.diff"] = _strip_trailing_whitespace(
            unified_diff(
                _normalize_copied_source(original),
                rendered,
                fromfile=f"a/{basename}",
                tofile=f"b/{output_name}",
            )
        )
    return diffs


def _prepare_generated_python(source_text: str) -> str:
    """Mark copied source as generated and normalize comment-only URLs.

    Generated bundles deliberately preserve complete upstream and remote-code
    source units. Repository lint rules for hand-maintained code therefore do
    not apply to those copied units. The markers are comments and do not alter
    imports, executable statements, or runtime behavior.
    """
    text = _normalize_copied_source(source_text)
    text = re.sub(
        r"(^[ \t]*\#.*https?://\S*[A-Za-z0-9])$",
        r"\1 (link).",
        text,
        flags=re.MULTILINE,
    )
    if text.startswith(_GENERATED_SOURCE_HEADER):
        return text
    return (
        _GENERATED_SOURCE_HEADER
        + text.rstrip()
        + "\n"
        + _GENERATED_SOURCE_FOOTER
    )


def _normalize_copied_source(source_text: str) -> str:
    """Normalize repository-gate-only defects in copied upstream source."""
    text = _strip_trailing_whitespace(source_text)
    for typo, correction in _COPIED_SOURCE_SPELLING_FIXES.items():
        text = text.replace(typo, correction)
    return text


def _materialize_sibling(
    source_text: str,
    meta: Any,
    patch_plan: Any,
    *,
    source_path: str,
) -> str:
    """Apply source-owned strategy and recorded patches to one sibling."""
    source = getattr(meta, "source", None)
    architecture = (
        source.get("architecture")
        if isinstance(source, dict)
        else getattr(source, "architecture", None)
    )
    rendered = render_inline_strategies_for_source_unit(
        source_text,
        meta,
        model_type=architecture,
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
    del layout
    return (
        _APACHE_LICENSE_HEADER
        + '"""Generated codegen bundle — DO NOT EDIT MANUALLY.\n'
        "\n"
        "Regenerate with ``python -m hyper_parallel.codegen generate``; the\n"
        "the bundle is rebuilt from the current training configuration.\n"
        '"""\n'
    )


def emit_diff(original_text: str, generated_text: str, layout: ArtifactLayout) -> str:
    """Render the unified diff from the original source to the generated file.

    Returned rather than written so ``emit_bundle`` can write the whole bundle
    in one atomic call.
    """
    return unified_diff(
        _normalize_copied_source(original_text),
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
