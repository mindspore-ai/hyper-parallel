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
"""Apply inline patch IR to copied modeling source."""

from __future__ import annotations

import ast
from collections import defaultdict
from typing import Iterable

from hyper_parallel.codegen.astkit.edits import (
    TextEdit,
    apply_edits,
    insert_after_imports,
    replace_function_body,
)
from hyper_parallel.codegen.astkit.index import build_source_index
from hyper_parallel.codegen.inline.ir import (
    ClassMarkerPatch,
    ClassRemovalPatch,
    ConstructorReplacePatch,
    ForwardExtractPatch,
    ForwardBodyPatch,
    ImportPatch,
    InlinePatchSet,
    ModuleSnippetPatch,
)


def apply_patch_set(source_text: str, patch_set: InlinePatchSet) -> str:
    """Apply imports, constructor edits, class removals, markers, and forward bodies.

    Args:
        source_text: Modeling source the patches are applied to.
        patch_set: Edits emitted by the inline passes.

    Returns:
        The patched source text.
    """

    first_pass: list[TextEdit] = []
    index = build_source_index(source_text)
    import_edit = _import_edit(source_text, index.import_end, index.docstring_end, patch_set.imports)
    if import_edit is not None:
        first_pass.append(import_edit)
    snippet_edit = _snippet_edit(source_text, index.import_end, index.docstring_end, patch_set.module_snippets)
    if snippet_edit is not None:
        first_pass.append(snippet_edit)
    first_pass.extend(_constructor_edits(source_text, patch_set.constructor_replaces))
    text = apply_edits(source_text, first_pass) if first_pass else source_text

    if patch_set.class_removals:
        text = _apply_class_removals(text, patch_set.class_removals)

    second_pass: list[TextEdit] = []
    second_pass.extend(_class_marker_edits(text, patch_set.class_markers))
    second_pass.extend(_forward_extract_edits(text, patch_set.forward_extracts))
    second_pass.extend(_forward_body_edits(text, patch_set.forward_bodies))
    return apply_edits(text, second_pass) if second_pass else text


def _import_edit(
    source_text: str,
    import_end: int | None,
    docstring_end: int | None,
    imports: Iterable[ImportPatch],
) -> TextEdit | None:
    lines = _render_imports(source_text, imports)
    if not lines:
        return None
    return insert_after_imports(
        source_text,
        import_end,
        "\n" + "\n".join(lines) + "\n",
        default_offset=docstring_end or 0,
    )


def _render_imports(source_text: str, imports: Iterable[ImportPatch]) -> list[str]:
    grouped: dict[str, set[str]] = defaultdict(set)
    lines: list[str] = []
    for patch in imports:
        if patch.raw:
            if patch.raw not in source_text:
                lines.append(patch.raw)
            continue
        grouped[patch.module].update(patch.names)
    for module in sorted(grouped):
        names = tuple(sorted(grouped[module]))
        line = f"from {module} import {', '.join(names)}"
        if line not in source_text:
            lines.append(line)
    return lines


def _snippet_edit(
    source_text: str,
    import_end: int | None,
    docstring_end: int | None,
    snippets: Iterable[ModuleSnippetPatch],
) -> TextEdit | None:
    parts = [snippet.text.strip("\n") for snippet in snippets if snippet.text.strip()]
    if not parts:
        return None
    return insert_after_imports(
        source_text,
        import_end,
        "\n\n" + "\n\n".join(parts) + "\n",
        default_offset=docstring_end or 0,
    )


def _constructor_edits(
    source_text: str,
    patches: Iterable[ConstructorReplacePatch],
) -> list[TextEdit]:
    patch_list = list(patches)
    patches_by_old: dict[str, list[ConstructorReplacePatch]] = defaultdict(list)
    for patch in patch_list:
        patches_by_old[patch.old_ctor].append(patch)
    if not patches_by_old:
        return []
    scopes = _scopes_by_old_ctor(patch_list)
    tree = ast.parse(source_text)
    offsets = _line_offsets(source_text)
    edits: list[TextEdit] = []
    hits: dict[str, int] = defaultdict(int)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        attr = _assigned_self_attr(node)
        if attr is None:
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        name = _call_name(call.func)
        if name is None or name not in patches_by_old:
            continue
        scope = scopes[name]
        if scope is not None and attr not in scope:
            continue
        patch = patches_by_old[name][0]
        hits[name] += 1
        if patch.mode == "name":
            start, end = _node_span(call.func, offsets)
            edits.append(TextEdit(start, end, patch.new_ctor))
        elif patch.mode == "wrap_source":
            start, end = _node_span(call, offsets)
            call_source = source_text[start:end]
            extra = "".join(f", {arg}" for arg in patch.keyword_args)
            replacement = f"{patch.new_ctor}(module={call_source}{extra})"
            edits.append(TextEdit(start, end, replacement))
        else:
            raise ValueError(f"unknown constructor replacement mode: {patch.mode!r}")
    _require_scoped_rewrites(patch_list, hits)
    return _drop_nested_edits(edits)


def _scopes_by_old_ctor(
    patches: Iterable[ConstructorReplacePatch],
) -> dict[str, set[str] | None]:
    """Merge each constructor's rewrite scope; ``None`` means unrestricted."""

    scopes: dict[str, set[str] | None] = {}
    for patch in patches:
        if not patch.scope_attrs:
            scopes[patch.old_ctor] = None
            continue
        merged = scopes.get(patch.old_ctor)
        if merged is None and patch.old_ctor in scopes:
            continue
        if merged is None:
            merged = scopes[patch.old_ctor] = set()
        merged.update(patch.scope_attrs)
    return scopes


def _assigned_self_attr(node: ast.Assign) -> str | None:
    """Return ``attr`` for ``self.<attr> = ...`` assignments, else ``None``.

    Module replacements are sunk at the assignment that installs the module, so
    scoping by the assigned attribute is exactly the FQN granularity the native
    plan used to decide the replacement.
    """

    if len(node.targets) != 1:
        return None
    target = node.targets[0]
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return target.attr
    return None


def _require_scoped_rewrites(
    patches: Iterable[ConstructorReplacePatch],
    hits: dict[str, int],
) -> None:
    """Fail when a scoped rule matched FQNs but no source call site.

    Silently dropping the rewrite would emit an artifact that no longer matches
    the plan that decided the replacement, so a missing call site is a
    generation error: the scope handling needs to learn the new shape rather
    than the rewrite being widened back to the whole file.
    """

    missing = sorted(
        {patch.old_ctor for patch in patches if patch.scope_attrs and not hits.get(patch.old_ctor)}
    )
    if missing:
        raise RuntimeError(
            "codegen: no `self.<attr> = <ctor>(...)` call site found for "
            f"{missing}; the plan replaced modules of this type, so the "
            "replacement cannot be sunk without diverging from the plan"
        )


def _drop_nested_edits(edits: list[TextEdit]) -> list[TextEdit]:
    """Keep outer constructor rewrites when constructor calls are nested."""
    kept: list[TextEdit] = []
    covered_end = -1
    for edit in sorted(edits, key=lambda item: (item.start, -item.end)):
        if edit.start < covered_end:
            continue
        kept.append(edit)
        covered_end = edit.end
    return kept


def drop_shadowed_definitions(text: str, imported_names: Iterable[str]) -> str:
    """Remove definitions that shadow a library import of the same name.

    The artifact imports the very helpers the native path calls, but the copied
    modeling source defines its own module-level helpers with those names, and a
    local definition wins over the import.  For a *decorated* helper that means
    a different implementation runs: ``use_kernel_func_from_hub("rotary_pos_emb")``
    resolves its kernel binding per module, and the generated module is not the
    module the binding was registered for.  Both paths must therefore end up with
    the imported implementation, so the shadowing copy is dropped.

    Callers pass names that the *base source* already defines, so the classes and
    snippets the inline passes render (which share names with library imports on
    purpose) keep their definitions.

    Args:
        text: Source to drop the shadowing definitions from.
        imported_names: Names the library imports bind in module scope.

    Returns:
        The source with the shadowing definitions removed.
    """

    names = {name for name in imported_names if name}
    if not names:
        return text
    tree = ast.parse(text)
    offsets = _line_offsets(text)
    edits: list[TextEdit] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name not in names or not node.decorator_list:
            continue
        start, end = _node_span(node, offsets)
        edits.append(TextEdit(start, _skip_trailing_blank_lines(text, end), ""))
    if not edits:
        return text
    return apply_edits(text, edits)


def _name_is_referenced(tree: ast.AST, name: str) -> bool:
    """Return whether ``name`` still appears as a reference in ``tree``."""

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
    return False


def _apply_class_removals(
    text: str,
    patches: Iterable[ClassRemovalPatch],
) -> str:
    """Rewrite surviving bare references to removed classes, then delete them.

    Runs as a **separate pass** after constructor rewrites so the AST reflects
    the already-patched text: every ``old_name(...)`` call has become
    ``new_name(...)`` and only *bare* references (dict values, ``isinstance``
    targets, annotations) remain.  Those are rewritten to ``new_name`` when
    they fall outside a class-deletion range; references inside a class being
    deleted are left alone because the entire block is removed.

    Args:
        text: Source text after the constructor rewrites.
        patches: Class removals requested by the replacement specs.

    Returns:
        The text with unused classes removed.
    """
    patch_list = list(patches)
    if not patch_list:
        return text
    tree = ast.parse(text)
    offsets = _line_offsets(text)
    names_to_remove, new_name_by_old = _removal_targets(tree, patch_list)
    if not names_to_remove:
        return text
    deletion_ranges, classdef_nodes = _class_deletion_spans(tree, offsets, names_to_remove)
    edits = _reference_edits(
        tree, offsets, names_to_remove, new_name_by_old, deletion_ranges
    )
    edits.extend(_deletion_edits(text, offsets, patch_list, classdef_nodes))
    return apply_edits(text, edits) if edits else text


def _removal_targets(
    tree: ast.AST,
    patch_list: list[ClassRemovalPatch],
) -> tuple[set[str], dict[str, str]]:
    """Return the classes still unused after the rewrites, and their renames.

    A scoped rewrite can leave the original class in use -- the recipe keeps
    the HF Q/K norm while replacing the layer norms -- so a class is removed
    only once the patched source no longer references it.  Removing a class
    that is still called would turn a partially replaced model into a
    ``NameError`` instead of the model the plan described.
    """

    names = {patch.old_name for patch in patch_list}
    unused = {name for name in names if not _name_is_referenced(tree, name)}
    return unused, {patch.old_name: patch.new_name for patch in patch_list}


def _class_deletion_spans(
    tree: ast.AST,
    offsets: tuple[int, ...],
    names_to_remove: set[str],
) -> tuple[list[tuple[int, int]], dict[str, ast.ClassDef]]:
    """Return the source spans of the removable classes and their nodes."""

    deletion_ranges: list[tuple[int, int]] = []
    classdef_nodes: dict[str, ast.ClassDef] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in names_to_remove:
            classdef_nodes[node.name] = node
            deletion_ranges.append(_node_span(node, offsets))
    return deletion_ranges, classdef_nodes


def _reference_edits(
    tree: ast.AST,
    offsets: tuple[int, ...],
    names_to_remove: set[str],
    new_name_by_old: dict[str, str],
    deletion_ranges: list[tuple[int, int]],
) -> list[TextEdit]:
    """Return edits rewriting bare references that survive the deletion."""

    edits: list[TextEdit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id not in names_to_remove:
            continue
        start, end = _node_span(node, offsets)
        if any(span <= start < stop for span, stop in deletion_ranges):
            continue
        edits.append(TextEdit(start, end, new_name_by_old[node.id]))
    return edits


def _deletion_edits(
    text: str,
    offsets: tuple[int, ...],
    patch_list: list[ClassRemovalPatch],
    classdef_nodes: dict[str, ast.ClassDef],
) -> list[TextEdit]:
    """Return edits deleting the removable class blocks and trailing blanks."""

    edits: list[TextEdit] = []
    for patch in patch_list:
        cls_node = classdef_nodes.get(patch.old_name)
        if cls_node is None:
            continue
        start, end = _node_span(cls_node, offsets)
        end = _skip_trailing_blank_lines(text, end)
        edits.append(TextEdit(start, end, ""))
    return edits


def _skip_trailing_blank_lines(text: str, end: int) -> int:
    """Advance ``end`` past spaces and blank lines following a deleted block."""

    while end < len(text) and text[end] in " \t":
        end += 1
    while end < len(text) and text[end] == "\n":
        end += 1
        while end < len(text) and text[end] in " \t":
            end += 1
    return end


def _class_marker_edits(
    source_text: str,
    patches: Iterable[ClassMarkerPatch],
) -> list[TextEdit]:
    if not patches:
        return []
    index = build_source_index(source_text)
    offsets = _line_offsets(source_text)
    edits: list[TextEdit] = []
    for patch in patches:
        cls = index.find_class(patch.class_name)
        if cls is None:
            continue
        anchor = offsets[cls.class_line]
        marker = f"    {patch.marker_name} = {patch.marker_value}\n"
        if marker.strip() in source_text[cls.class_offset:cls.end_offset]:
            continue
        edits.append(TextEdit(anchor, anchor, marker))
    return edits


def _forward_body_edits(
    source_text: str,
    patches: Iterable[ForwardBodyPatch],
) -> list[TextEdit]:
    if not patches:
        return []
    index = build_source_index(source_text)
    edits: list[TextEdit] = []
    for patch in patches:
        cls = index.find_class(patch.class_name)
        if cls is None:
            continue
        func = cls.methods.get(patch.method_name)
        if func is None:
            continue
        edits.extend(replace_function_body(source_text, func, patch.body, indent="        "))
    return edits


def _forward_extract_edits(
    source_text: str,
    patches: Iterable[ForwardExtractPatch],
) -> list[TextEdit]:
    if not patches:
        return []
    from hyper_parallel.codegen.emit.parallel import (  # pylint: disable=import-outside-toplevel
        _build_forward_impl_edit,
    )

    index = build_source_index(source_text)
    edits: list[TextEdit] = []
    for patch in patches:
        cls = index.find_class(patch.class_name)
        if cls is None:
            continue
        func = cls.methods.get(patch.method_name)
        if func is None:
            continue
        impl = _build_forward_impl_edit(
            source_text,
            func,
            prefix=patch.marker_prefix,
        )
        if impl is not None:
            edits.append(impl)
        edits.extend(replace_function_body(source_text, func, patch.body, indent="        "))
    return edits


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _node_span(node: ast.AST, offsets: tuple[int, ...]) -> tuple[int, int]:
    """Return the ``(start, end)`` source span of ``node``.

    Decorators are part of the span: deleting only the ``def``/``class`` line
    would leave a dangling decorator that then applies to whatever statement
    follows the removed block.
    """

    if (
        getattr(node, "lineno", None) is None
        or getattr(node, "col_offset", None) is None
        or getattr(node, "end_lineno", None) is None
        or getattr(node, "end_col_offset", None) is None
    ):
        raise ValueError(f"AST node has no source span: {type(node).__name__}")
    line = node.lineno
    column = node.col_offset
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        first = min(decorators, key=lambda item: item.lineno)
        if first.lineno < line:
            # A decorator node starts at its expression, one column *after* the
            # ``@``; the span must start at the ``@`` itself, otherwise deleting
            # the block leaves a stray ``@`` behind.
            line, column = first.lineno, max(0, first.col_offset - 1)
    start = offsets[line - 1] + column
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    return start, end


def _line_offsets(source_text: str) -> tuple[int, ...]:
    offsets = [0]
    total = 0
    for line in source_text.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)
    return tuple(offsets)
