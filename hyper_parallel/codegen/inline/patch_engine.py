# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
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
    """Apply imports, constructor edits, class removals, markers, and forward bodies."""

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
    patches_by_old: dict[str, list[ConstructorReplacePatch]] = defaultdict(list)
    for patch in patches:
        patches_by_old[patch.old_ctor].append(patch)
    if not patches_by_old:
        return []
    tree = ast.parse(source_text)
    offsets = _line_offsets(source_text)
    edits: list[TextEdit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name is None or name not in patches_by_old:
            continue
        patch = patches_by_old[name][0]
        if patch.mode == "name":
            start, end = _node_span(node.func, offsets)
            edits.append(TextEdit(start, end, patch.new_ctor))
        elif patch.mode == "wrap_source":
            start, end = _node_span(node, offsets)
            call_source = source_text[start:end]
            extra = "".join(f", {arg}" for arg in patch.keyword_args)
            replacement = f"{patch.new_ctor}(module={call_source}{extra})"
            edits.append(TextEdit(start, end, replacement))
        else:
            raise ValueError(f"unknown constructor replacement mode: {patch.mode!r}")
    return _drop_nested_edits(edits)


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
    """
    patch_list = list(patches)
    if not patch_list:
        return text
    tree = ast.parse(text)
    offsets = _line_offsets(text)
    edits: list[TextEdit] = []

    names_to_remove = {p.old_name for p in patch_list}
    new_name_by_old = {p.old_name: p.new_name for p in patch_list}

    deletion_ranges: list[tuple[int, int]] = []
    classdef_nodes: dict[str, ast.ClassDef] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in names_to_remove:
            classdef_nodes[node.name] = node
            span = _node_span(node, offsets)
            deletion_ranges.append(span)

    def _in_deletion(pos: int) -> bool:
        return any(s <= pos < e for s, e in deletion_ranges)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Name):
            continue
        if node.id not in names_to_remove:
            continue
        start, end = _node_span(node, offsets)
        if _in_deletion(start):
            continue
        edits.append(TextEdit(start, end, new_name_by_old[node.id]))

    for patch in patch_list:
        cls_node = classdef_nodes.get(patch.old_name)
        if cls_node is None:
            continue
        start, end = _node_span(cls_node, offsets)
        while end < len(text) and text[end] in " \t":
            end += 1
        while end < len(text) and text[end] == "\n":
            end += 1
            while end < len(text) and text[end] in " \t":
                end += 1
        edits.append(TextEdit(start, end, ""))

    return apply_edits(text, edits) if edits else text


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
    from hyper_parallel.codegen.emit.parallel import _build_forward_impl_edit

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
    if (
        getattr(node, "lineno", None) is None
        or getattr(node, "col_offset", None) is None
        or getattr(node, "end_lineno", None) is None
        or getattr(node, "end_col_offset", None) is None
    ):
        raise ValueError(f"AST node has no source span: {type(node).__name__}")
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    return start, end


def _line_offsets(source_text: str) -> tuple[int, ...]:
    offsets = [0]
    total = 0
    for line in source_text.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)
    return tuple(offsets)
