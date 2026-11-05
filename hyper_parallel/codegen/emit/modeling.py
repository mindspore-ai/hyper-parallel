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
"""Assemble the generated HF-style modeling file.

The generated file is the original modeling source, verbatim, with two
appended layers:

* a header banner recording where it came from,
* the adapter-driven inline patch — replaced modules and the TP / CP / EP
  strategies lowered into the copied source at their semantic sites.

Boundary forwards are then lowered at the boundary classes. The artifact is
both an auditable source export and a directly executable model training entry.
Training still derives its live plan from the supplied YAML through the native
Trainer; preflight validates syntax and the recorded generation contract.
"""

from __future__ import annotations

import ast
import io
import keyword
import os
import tokenize
from typing import Any

from hyper_parallel.codegen.source.composition import compose_nested_model_sources
from hyper_parallel.codegen.source.compat import sanitize_source_compat
from hyper_parallel.codegen.emit.parallel import lower_forward_boundaries
from hyper_parallel.codegen.emit.recorded import materialize_recorded_forwards
from hyper_parallel.codegen.inline.pipeline import render_inline_modeling_with_plan

# ---------------------------------------------------------------------------
# source copy
# ---------------------------------------------------------------------------


def copy_original_modeling(source: Any) -> str:
    """Read the original modeling source text from a ``ResolvedSource``-like.

    ``source`` is ``meta.source`` — the ``ResolvedSource.to_dict()`` form with
    a ``file_path`` key.  Accepts either a dict or a ``ResolvedSource`` (the
    dict form is what the manager records, but the helper tolerates the live
    dataclass so it stays usable outside the meta path).
    """
    if isinstance(source, dict):
        path = source.get("file_path")
    else:
        path = getattr(source, "file_path", None)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"codegen emit: original modeling source is not a readable file "
            f"({path!r}); generation must run where the model's modeling file "
            "exists"
        )
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def rewrite_relative_imports(text: str, module_name: str) -> str:
    """Rewrite ``from .xxx`` / ``from ...xxx`` imports to absolute imports.

    HF modeling files commonly import siblings with a relative spec, e.g.
    ``from .configuration_qwen3 import Qwen3Config`` or ``from ...activations
    import ACT2FN``.  The generated bundle is loaded by ``spec_from_file_location``
    under a synthetic package (``hyper_parallel_generated.<digest>.<stem>``), so a
    relative spec resolves against that synthetic package instead of the real
    transformers package and fails with ``attempted relative import beyond
    top-level package``.

    This anchors the relative spec on ``module_name`` (the original module's
    real dotted path, e.g. ``transformers.models.qwen3.modeling_qwen3``): N
    leading dots climb N-1 package levels above the module's parent package, then
    append the rest.  ``from ...activations`` → ``from transformers.activations``
    and ``from .configuration_qwen3`` →
    ``from transformers.models.qwen3.configuration_qwen3``.

    ``module_name`` is ``"<remote>"`` for trusted remote code, which has no real
    package to anchor on — the rewrite is a no-op there (import handling for
    remote code keeps its relative imports). A non-dotted or unanchorable ``module_name``
    is likewise left untouched.

    Only the module-spec part of each relative import is rewritten; the symbol
    list, comments, and whitespace are preserved.  It uses ``tokenize`` rather
    than a regex so relative-looking text inside string literals or comments is
    never touched.
    """
    unless_anchor = _relative_anchor(module_name)
    if unless_anchor is None:
        return text
    text = _rewrite_dynamic_relative_imports(text, unless_anchor)

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except tokenize.TokenError:
        return text
    if not tokens:
        return text

    edits: list[tuple[int, int, str]] = []
    i = 0
    while i < len(tokens):
        edit, i = _relative_import_edit(tokens, i, unless_anchor)
        if edit is not None:
            edits.append(edit)

    if not edits:
        return text
    return _splice(text, edits)


def _relative_import_edit(
    tokens: list[tokenize.TokenInfo], index: int, anchor: list[str]
) -> tuple[tuple[tuple[int, int], tuple[int, int], str] | None, int]:
    """Parse one token position into an absolute relative-import edit."""
    token = tokens[index]
    if token.type != tokenize.NAME or token.string != "from":
        return None, index + 1
    cursor = _skip_import_trivia(tokens, index + 1)
    if cursor >= len(tokens) or not _is_dot_token(tokens[cursor]):
        return None, index + 1
    dots_token = tokens[cursor]
    dots = 0
    while cursor < len(tokens) and _is_dot_token(tokens[cursor]):
        dots += len(tokens[cursor].string)
        cursor += 1
    path, last_token, cursor = _relative_module_path(tokens, cursor)
    import_index = _skip_import_trivia(tokens, cursor)
    if import_index >= len(tokens) or tokens[import_index].string != "import":
        return None, index + 1
    absolute = _absolute_spec(anchor, dots, path)
    if absolute is None:
        return None, index + 1
    end = last_token.end if last_token is not None else dots_token.end
    return (dots_token.start, end, absolute), cursor + 1


def _skip_import_trivia(tokens: list[tokenize.TokenInfo], index: int) -> int:
    """Skip comments and physical/logical newlines between import tokens."""
    trivia = {tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT}
    while index < len(tokens) and tokens[index].type in trivia:
        index += 1
    return index


def _is_dot_token(token: tokenize.TokenInfo) -> bool:
    """Return whether a token contains only relative-import dots."""
    return token.type == tokenize.OP and bool(token.string) and set(token.string) == {"."}


def _relative_module_path(
    tokens: list[tokenize.TokenInfo], index: int
) -> tuple[list[str], tokenize.TokenInfo | None, int]:
    """Collect the NAME (dot NAME)* portion of a relative import."""
    path: list[str] = []
    last_token = None
    while index < len(tokens):
        token = tokens[index]
        if token.type != tokenize.NAME or token.string == "import":
            break
        path.append(token.string)
        last_token = token
        index += 1
        if index >= len(tokens) or tokens[index].string != ".":
            break
        index += 1
    return path, last_token, index


def _rewrite_dynamic_relative_imports(text: str, anchor: list[str]) -> str:
    """Rewrite imports whose resolved package is not valid Python syntax."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        absolute = _absolute_spec(anchor, node.level, (node.module or "").split("."))
        if absolute is None or _is_import_syntax_safe(absolute):
            continue
        assignments = []
        for alias in node.names:
            if alias.name == "*":
                raise ValueError(
                    f"codegen cannot rewrite a wildcard import from dynamic module {absolute!r}"
                )
            target = alias.asname or alias.name
            assignments.append(
                f"{target} = __import__({absolute!r}, fromlist=[{alias.name!r}]).{alias.name}"
            )
        indent = " " * node.col_offset
        replacement = ("\n" + indent).join(assignments)
        edits.append(
            (
                (node.lineno, node.col_offset),
                (node.end_lineno, node.end_col_offset),
                replacement,
            )
        )
    return _splice(text, edits) if edits else text


def _is_import_syntax_safe(module: str) -> bool:
    """Return whether every dotted component can appear in an import statement."""
    return all(
        part.isidentifier() and not keyword.iskeyword(part)
        for part in module.split(".")
    )


def _relative_anchor(module_name: str) -> list[str] | None:
    """Return the dotted package segments a relative spec anchors on, or None.

    A real installed-module path (``transformers.models.qwen3.modeling_qwen3``)
    anchors.  The ``"<remote>"`` sentinel (trusted remote code) and any
    non-dotted / placeholder-laden name cannot anchor — the caller leaves such
    imports untouched.
    """
    if not isinstance(module_name, str) or not module_name:
        return None
    if "<" in module_name or ">" in module_name:
        return None
    segs = module_name.split(".")
    if len(segs) < 2 or any(not seg for seg in segs):
        return None
    return segs


def _absolute_spec(anchor: list[str], dots: int, path: list[str]) -> str | None:
    """Compute the absolute import spec for one relative import, or None.

    ``anchor`` is the module's package segments (its path minus the final
    module segment).  ``dots`` leading dots climb ``dots - 1`` package levels
    above the module's parent package, then append ``path``.  Returns None when
    the relative import climbs above the package root (invalid source — leave
    it untouched rather than corrupt it).
    """
    parent = anchor[:-1]
    climb = dots - 1
    if climb > len(parent):
        return None
    base_parts = parent if climb <= 0 else parent[: len(parent) - climb]
    parts = base_parts + path
    return ".".join(part for part in parts if part)


def _splice(
    text: str, edits: list[tuple[tuple[int, int], tuple[int, int], str]]
) -> str:
    """Splice absolute specs back into ``text`` by token (row, col) positions.

    Edits are applied last-first so the earlier offsets stay valid.  A single
    contiguous span is replaced in every case: the relative spec starts at the
    dots token and ends at the last module-path token (or the dots token when
    there is no path, i.e. ``from . import``).
    """
    lines = text.splitlines(keepends=True)
    if not lines:
        return text
    line_offsets = []
    acc = 0
    for line in lines:
        line_offsets.append(acc)
        acc += len(line)

    def offset(row: int, col: int) -> int:
        if 1 <= row <= len(line_offsets):
            return line_offsets[row - 1] + col
        return acc

    positioned = sorted(
        ((offset(*start), offset(*end), new) for start, end, new in edits),
        key=lambda item: item[0],
        reverse=True,
    )
    out = text
    for start, end, new in positioned:
        if start < end <= len(out):
            out = out[:start] + new + out[end:]
    return out


# ---------------------------------------------------------------------------
# file assembly
# ---------------------------------------------------------------------------


def emit_modeling_file(meta: Any, *, patch_plan: Any = None) -> str:
    """Assemble the full generated modeling file text.

    Returns the text (not written to disk) so the caller can compute a diff
    first. The header banner carries the generation provenance; the source is
    copied verbatim, rewritten by the adapter-driven inline pipeline, and then
    lowered at the boundary classes.
    """
    text = copy_original_modeling(meta.source)
    # Stale-source sanitize: a trust-remote-code modeling file ships with the
    # checkpoint and reflects the author's transformers, not the installed one.
    # Removing ``is_torch_fx_available`` (a transformers>=5.0 casualty) here
    # keeps the artifact importable without touching the source file. No-op
    # when the source does not import the symbol.
    text = sanitize_source_compat(text)
    module_name = (
        meta.source.get("module_name")
        if isinstance(meta.source, dict)
        else getattr(meta.source, "module_name", "")
    )
    text = rewrite_relative_imports(text, module_name)
    text = compose_nested_model_sources(text, meta)
    text, emission_meta = _apply_inline_modeling(text, meta, patch_plan=patch_plan)
    source_path = (
        meta.source.get("file_path")
        if isinstance(meta.source, dict)
        else getattr(meta.source, "file_path", None)
    )
    text = materialize_recorded_forwards(
        text,
        emission_meta,
        patch_plan=patch_plan,
        source_path=(
            None if getattr(emission_meta, "nested_components", None) else source_path
        ),
    )
    text = _lower_forward_boundaries(text, emission_meta, module_name)
    _verify_recorded_patch_materialization(
        text, emission_meta, patch_plan=patch_plan
    )
    return _banner(meta) + text


def _verify_recorded_patch_materialization(
    text: str,
    meta: Any,
    *,
    patch_plan: Any = None,
) -> None:
    """Require every trainer-installed source-visible patch to be materialized."""
    patches = patch_plan
    local_records = (
        getattr(patches, "local_computes", ()) if patches is not None else ()
    )
    replacement_records = (
        getattr(patches, "module_replacements", ()) if patches is not None else ()
    )
    wrapper_records = (
        getattr(patches, "inner_wrappers", ()) if patches is not None else ()
    )
    if not local_records and not replacement_records and not wrapper_records:
        return
    tree = ast.parse(text)
    classes = _class_index(tree)
    materialized = getattr(meta, "materialized_patches", None)
    if isinstance(materialized, list):
        materialized.clear()
    missing = _verify_local_records(tree, classes, local_records, materialized)
    missing += _verify_replacement_records(
        tree, classes, replacement_records, materialized
    )
    missing += _verify_wrapper_records(
        classes, wrapper_records, replacement_records, materialized
    )
    if missing:
        raise RuntimeError(
            "codegen: trainer-installed source-visible patches were not "
            "materialized in the generated model source: " + ", ".join(missing)
        )


def _class_index(tree: ast.AST) -> dict[str, ast.ClassDef]:
    """Index top-level classes by name."""
    return {
        node.name: node
        for node in getattr(tree, "body", ())
        if isinstance(node, ast.ClassDef)
    }


def _class_methods(host: ast.ClassDef) -> set[str]:
    """Return method names declared directly by a class."""
    return {
        node.name
        for node in host.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _class_state_names(host: ast.ClassDef) -> set[str]:
    """Return simple class attributes declared by assignment."""
    names = set()
    for child in host.body:
        targets = (
            child.targets
            if isinstance(child, ast.Assign)
            else (child.target,)
            if isinstance(child, ast.AnnAssign)
            else ()
        )
        names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _record_materialized(
    materialized: Any, fqn: str, host: str, kind: str, source_form: str
) -> None:
    """Append one materialization record when metadata is mutable."""
    if isinstance(materialized, list):
        materialized.append(
            {"fqn": fqn, "host_class": host, "kind": kind, "source_form": source_form}
        )


def _verify_local_records(
    tree: ast.AST, classes: dict[str, ast.ClassDef], records: Any, materialized: Any
) -> list[str]:
    """Verify local-compute records in one source unit."""
    missing = []
    for record in records:
        class_name = record.module_type.__name__
        if _bound_function_patch_materialized(tree, record):
            _record_materialized(
                materialized,
                record.module_fqn,
                class_name,
                "local_compute",
                "bound_method_toggle_forward",
            )
            continue
        host = classes.get(class_name)
        if host is None:
            continue
        valid = {"_forward_impl", "forward"} <= _class_methods(
            host
        ) and "ep_enable" in _class_state_names(host)
        if valid:
            _record_materialized(
                materialized,
                record.module_fqn,
                class_name,
                "local_compute",
                "same_class_toggle_forward",
            )
        else:
            missing.append(f"{record.module_fqn} ({class_name})")
    return missing


def _verify_replacement_records(
    tree: ast.AST, classes: dict[str, ast.ClassDef], records: Any, materialized: Any
) -> list[str]:
    """Verify replacement constructors in one source unit."""
    bound_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | set(classes)
    missing = []
    for record in records:
        name = record.replacement_type.__name__
        visible = name in bound_names and any(
            isinstance(node, ast.Call) and _call_name(node.func) == name
            for node in ast.walk(tree)
        )
        if visible:
            _record_materialized(
                materialized,
                record.module_fqn,
                name,
                "module_replacement",
                "constructor",
            )
        else:
            missing.append(f"{record.module_fqn} (replacement {name})")
    return missing


def _wrapper_identity(record: Any) -> tuple[Any, str, str]:
    """Return target type, host name, and target FQN for a wrapper record."""
    target_type = getattr(record, "target_type", None)
    host_name = (
        target_type.__name__
        if target_type is not None
        else type(record.request.target).__name__
    )
    target_fqn = (
        record.module_fqn
        if record.target_fqn == "self"
        else f"{record.module_fqn}.{record.target_fqn}"
    )
    return target_type, host_name, target_fqn


def _wrapper_source_form(
    classes: dict[str, ast.ClassDef], host: ast.ClassDef, record: Any, replacements: Any
) -> str | None:
    """Return the concrete source form that materializes a wrapper."""
    if (
        "forward" in _class_methods(host)
        and "cp_enable" in _class_state_names(host)
        and _class_dispatch_reads_state(host, "cp_enable")
    ):
        return "same_class_toggle_forward"
    subclass = next(
        (
            node
            for node in classes.values()
            if any(
                isinstance(base, ast.Name) and base.id == host.name
                for base in node.bases
            )
            and "forward" in _class_methods(node)
        ),
        None,
    )
    if subclass is not None:
        return "subclass_forward"
    target_type = getattr(record, "target_type", None)
    replacement_backed = any(
        item.module_fqn == record.module_fqn and item.replacement_type is target_type
        for item in replacements
    )
    if replacement_backed and {"_forward_impl", "forward"} <= _class_methods(host):
        return "replacement_forward_runtime_binding"
    return None


def _verify_wrapper_records(
    classes: dict[str, ast.ClassDef], records: Any, replacements: Any, materialized: Any
) -> list[str]:
    """Verify inner-wrapper records in one source unit."""
    missing = []
    for record in records:
        _, host_name, target_fqn = _wrapper_identity(record)
        host = classes.get(host_name)
        if host is None:
            continue
        source_form = _wrapper_source_form(classes, host, record, replacements)
        if source_form:
            _record_materialized(
                materialized, target_fqn, host_name, "inner_wrapper", source_form
            )
        else:
            missing.append(f"{target_fqn} (inner wrapper for {host_name})")
    return missing


def _call_name(node: ast.AST) -> str | None:
    """Return the terminal name of a constructor call expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def verify_recorded_patch_bundle(
    files: dict[str, str],
    *,
    patch_plan: Any = None,
) -> None:
    """Verify recorded local-compute patches across all generated source units."""
    records = (
        getattr(patch_plan, "local_computes", ()) if patch_plan is not None else ()
    )
    wrappers = (
        getattr(patch_plan, "inner_wrappers", ()) if patch_plan is not None else ()
    )
    trees = {
        name: ast.parse(text) for name, text in files.items() if name.endswith(".py")
    }
    missing = [
        f"{record.module_fqn} ({record.module_type.__name__})"
        for record in records
        if not any(_tree_has_local_record(tree, record) for tree in trees.values())
    ]
    if missing:
        raise RuntimeError(
            "codegen: recorded local-compute patches were not materialized "
            "in their owning source units: " + ", ".join(missing)
        )
    replacements = getattr(patch_plan, "module_replacements", ())
    missing_wrappers = [
        f"{record.module_fqn} ({_wrapper_identity(record)[1]})"
        for record in wrappers
        if not any(
            _tree_has_wrapper_record(tree, record, replacements)
            for tree in trees.values()
        )
    ]
    if missing_wrappers:
        raise RuntimeError(
            "codegen: recorded inner-wrapper patches were not materialized "
            "in their owning source units: " + ", ".join(missing_wrappers)
        )
    if any("HPCodegen" in text for text in files.values()):
        raise RuntimeError(
            "codegen: generated bundle contains forbidden HPCodegen class names"
        )


def _tree_has_local_record(tree: ast.AST, record: Any) -> bool:
    """Return whether one source tree materializes a local-compute record."""
    if _bound_function_patch_materialized(tree, record):
        return True
    host = _class_index(tree).get(record.module_type.__name__)
    return (
        host is not None
        and {"_forward_impl", "forward"} <= _class_methods(host)
        and "ep_enable" in _class_state_names(host)
    )


def _tree_has_wrapper_record(tree: ast.AST, record: Any, replacements: Any) -> bool:
    """Return whether one source tree materializes an inner-wrapper record."""
    _, class_name, _ = _wrapper_identity(record)
    host = _class_index(tree).get(class_name)
    if host is None or "forward" not in _class_methods(host):
        return False
    if "cp_enable" in _class_state_names(host) and _class_dispatch_reads_state(
        host, "cp_enable"
    ):
        return True
    target_type = getattr(record, "target_type", None)
    replacement_backed = any(
        item.module_fqn == record.module_fqn and item.replacement_type is target_type
        for item in replacements
    )
    return replacement_backed and {"_forward_impl", "forward"} <= _class_methods(host)


def _class_dispatch_reads_state(host: ast.ClassDef, state_name: str) -> bool:
    """Return whether a class dispatches on one state flag in either layer."""
    methods = (
        child
        for child in host.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        and child.name in {"forward", "_forward_impl"}
    )
    return any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == state_name
        for method in methods
        for node in ast.walk(method)
    )


def _bound_function_patch_materialized(tree: ast.AST, record: Any) -> bool:
    """Return whether a ``MethodType`` forward has an explicit EP toggle."""
    function_name = getattr(record.original_forward, "__name__", "")
    if not function_name:
        return False
    impl_name = f"{function_name}_hp_impl"
    functions = {
        node.name: node
        for node in getattr(tree, "body", ())
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    wrapper = functions.get(function_name)
    if wrapper is None or impl_name not in functions:
        return False
    return _function_reads_ep_toggle(wrapper) and _tree_binds_method_impl(
        tree, impl_name
    )


def _function_reads_ep_toggle(wrapper: ast.AST) -> bool:
    """Return whether a function reads the instance EP enable flag."""
    return any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == "ep_enable"
        for node in ast.walk(wrapper)
    )


def _tree_binds_method_impl(tree: ast.AST, impl_name: str) -> bool:
    """Return whether a tree binds an implementation through MethodType."""
    return any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_forward_impl"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and _call_name(node.value.func) == "MethodType"
        and node.value.args
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == impl_name
        for node in ast.walk(tree)
    )


def _apply_inline_modeling(
    text: str,
    meta: Any,
    *,
    patch_plan: Any = None,
) -> tuple[str, Any]:
    """Render the artifact through the adapter-driven inline source pipeline."""

    source = getattr(meta, "source", None)
    architecture = (
        source.get("architecture")
        if isinstance(source, dict)
        else getattr(source, "architecture", None)
    )
    return render_inline_modeling_with_plan(
        text,
        meta,
        model_type=architecture,
        patch_plan=patch_plan,
    )


def _lower_forward_boundaries(text: str, meta: Any, module_name: str) -> str:
    """Sink boundary wrappers into the model's ``forward`` bodies.

    Each boundary class's ``forward`` is rewritten to redistribute through
    the instance-bound compiled plan and run the region's compute
    explicitly. A boundary whose class cannot be resolved or
    whose per-class contract differs across FQNs fails fast — a silently
    dropped redistribution would change the model's numeric behavior.

    Fresh on the unmodified source text, so offsets stay valid for the whole
    call.  The ``local_compute_fn`` / ``inner_wrapper`` members the rewritten
    forward references are bound by the runtime parallelize step, so the
    generated file stays runtime-importable without a mesh.

    ``boundary_classes`` is the observed FQN -> class-name map. When it is
    empty there are no source-level boundaries to lower, so the source remains
    untouched.
    """
    boundary_classes = getattr(meta, "boundary_classes", None) or {}
    if not boundary_classes:
        return text
    return lower_forward_boundaries(
        text,
        meta,
        boundary_classes=boundary_classes,
        module_name=module_name,
    )


def _banner(meta: Any) -> str:
    """Render the provenance banner comment."""
    source = meta.source if isinstance(meta.source, dict) else _source_dict(meta.source)
    pd = meta.parallel_dims or {}
    lines = [
        "# Generated by codegen. DO NOT EDIT MANUALLY.",
        "#",
        f"#   source       : {source.get('module_name', '')}",
        f"#   architecture : {source.get('architecture', '')}",
        f"#   parallel_dims: tp={pd.get('tp_size', 1)} cp={pd.get('cp_size', 1)} "
        f"ep={pd.get('ep_size', 1)} pp={pd.get('pp_size', 1)} "
        f"sp={pd.get('sequence_parallel', False)} "
        f"loss_parallel={pd.get('loss_parallel', False)}",
        f"#   injections   : {len(getattr(meta, 'injections', []) or [])} rule(s)",
        "#   runtime      : mesh / FSDP2 / PP / AC / compile still handled by runtime",
        "#",
        "# [HYPER PLAN] training derives a live plan from its YAML and this model;",
        "# Regenerate this file from the training YAML after strategy changes.",
        "#",
        "",
    ]
    return "\n".join(lines)


def _source_dict(source: Any) -> dict[str, Any]:
    if source is None:
        return {}
    if isinstance(source, dict):
        return source
    return {
        "module_name": getattr(source, "module_name", ""),
        "file_path": getattr(source, "file_path", ""),
        "architecture": getattr(source, "architecture", ""),
    }


__all__ = [
    "copy_original_modeling",
    "emit_modeling_file",
    "rewrite_relative_imports",
]
