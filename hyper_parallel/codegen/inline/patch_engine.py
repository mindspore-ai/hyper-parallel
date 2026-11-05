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

from hyper_parallel.codegen.astkit.edits import TextEdit, apply_edits
from hyper_parallel.codegen.astkit.index import build_source_index
from hyper_parallel.codegen.inline.declaration_edits import declaration_edits
from hyper_parallel.codegen.inline.forward_edits import forward_extract_edits
from hyper_parallel.codegen.inline.definition_edits import apply_class_removals
from hyper_parallel.codegen.inline.ir import (
    ConstructorReplacePatch,
    ForwardExtractPatch,
    InlinePatchSet,
)


def apply_patch_set(source_text: str, patch_set: InlinePatchSet) -> str:
    """Apply imports, constructor edits, class removals, markers, and forward bodies.

    Args:
        source_text: Modeling source the patches are applied to.
        patch_set: Edits emitted by the inline passes.

    Returns:
        The patched source text.
    """

    text, hits = _apply_source_declarations(source_text, patch_set)
    text = _apply_constructor_followups(text, patch_set, hits)
    return _apply_forward_patches(text, patch_set.forward_extracts)


def _apply_source_declarations(
    source_text: str, patch_set: InlinePatchSet
) -> tuple[str, dict[str, int]]:
    """Apply top-level declarations and the first constructor rewrite wave."""
    index = build_source_index(source_text)
    edits = declaration_edits(
        source_text,
        index.import_end,
        index.docstring_end,
        patch_set.imports,
        patch_set.module_snippets,
    )
    waves = _constructor_waves(list(patch_set.constructor_replaces))
    hits: dict[str, int] = defaultdict(int)
    if waves:
        edits.extend(_constructor_edits(source_text, waves[0], hits))
    text = apply_edits(source_text, edits) if edits else source_text
    for wave in waves[1:]:
        wave_edits = _constructor_edits(text, wave, hits)
        if wave_edits:
            text = apply_edits(text, wave_edits)
    return text, hits


def _apply_constructor_followups(
    source_text: str,
    patch_set: InlinePatchSet,
    hits: dict[str, int],
) -> str:
    """Apply deferred replacements, validate scopes, and remove replaced classes."""
    deferred_edits = _deferred_replacement_edits(
        source_text,
        patch_set.constructor_replaces,
        hits,
    )
    if deferred_edits:
        source_text = apply_edits(source_text, deferred_edits)
    if patch_set.constructor_replaces:
        _require_scoped_rewrites(patch_set.constructor_replaces, hits)
    if patch_set.class_removals:
        source_text = apply_class_removals(source_text, patch_set.class_removals)
    return source_text


def _apply_forward_patches(
    source_text: str, patches: Iterable[ForwardExtractPatch]
) -> str:
    """Apply forward extraction only after the source structure has settled."""
    edits = forward_extract_edits(source_text, patches)
    return apply_edits(source_text, edits) if edits else source_text


def _constructor_waves(
    patch_list: list[ConstructorReplacePatch],
) -> list[list[ConstructorReplacePatch]]:
    """Group constructor rewrites into dependency-ordered waves.

    Two compositions force a later wave, re-parsed from the previous
    wave's output: a patch whose ``old_ctor`` another remaining patch
    introduces (an ``A -> B`` wrap retargeted further by a ``B -> C``
    patch), and a ``name`` retarget sharing its ``old_ctor`` with a
    ``wrap_source`` patch — the inline CP subclass retargets the
    *outer* wrapper call the wrap introduced, and applying the name
    swap first would consume the call site the wrap still needs (a
    scoped wrap with no call site fails loudly, so the order is not
    optional).
    """

    remaining = list(patch_list)
    waves: list[list[ConstructorReplacePatch]] = []
    while remaining:
        introduced = {
            patch.new_ctor for patch in remaining if patch.new_ctor != patch.old_ctor
        }
        wave = [patch for patch in remaining if _is_wave_ready(patch, remaining, introduced)]
        if not wave:
            names = sorted({patch.old_ctor for patch in remaining})
            raise ValueError(
                f"codegen: constructor replacement dependency cycle among {names}"
            )
        chosen = {id(patch) for patch in wave}
        remaining = [patch for patch in remaining if id(patch) not in chosen]
        waves.append(wave)
    return waves


def _is_wave_ready(
    patch: ConstructorReplacePatch,
    remaining: list[ConstructorReplacePatch],
    introduced: set[str],
) -> bool:
    """Return whether a constructor patch has no remaining dependency."""
    if patch.old_ctor in introduced:
        return False
    if patch.mode != "name":
        return True
    return not any(
        other is not patch
        and other.mode == "wrap_source"
        and other.old_ctor == patch.old_ctor
        for other in remaining
    )


def _constructor_edits(
    source_text: str,
    patches: list[ConstructorReplacePatch],
    hits: dict[str, int],
) -> list[TextEdit]:
    """Build the edits that swap constructor calls per patch target."""
    patches_by_old: dict[str, list[ConstructorReplacePatch]] = defaultdict(list)
    for patch in patches:
        patches_by_old[patch.old_ctor].append(patch)
    if not patches_by_old:
        return []
    scopes = _scopes_by_old_ctor(patches)
    tree = ast.parse(source_text)
    offsets = _line_offsets(source_text)
    edits: list[TextEdit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        attr = _assigned_attr(node)
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


def _assigned_attr(node: ast.Assign) -> str | None:
    """Return the stored attribute of an attribute store, else ``None``.

    Module replacements are sunk at the assignment that installs the module, so
    scoping by the assigned attribute is exactly the FQN granularity the native
    plan used to decide the replacement.  Families attach modules at more than
    the ``self.<attr>`` store: a loop-local ``layer.<attr>`` upgrade and a
    ModuleList element ``self.layers[i].<attr>`` install the same matched
    modules, so any attribute store yields its stored attribute -- the rule's
    scope set keeps the rewrite at the plan's matched-FQN granularity.
    """

    if len(node.targets) != 1:
        return None
    target = node.targets[0]
    return target.attr if isinstance(target, ast.Attribute) else None


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


def _deferred_replacement_edits(
    source_text: str,
    patches: Iterable[ConstructorReplacePatch],
    hits: dict[str, int],
) -> list[TextEdit]:
    """Install observed wrappers after an imported owner built its subtree.

    Some source models inherit a decoder whose constructor lives in an
    installed dependency. The copied modeling unit therefore has no direct
    constructor call to rewrite. For a wrap-style replacement, the native
    operation is still expressible exactly: wait until the root model has
    built the observed FQNs, then replace those module attributes in place.
    """
    pending = [
        patch
        for patch in patches
        if patch.scope_fqns and not hits.get(patch.old_ctor)
    ]
    if not pending:
        return []
    tree = ast.parse(source_text)
    offsets = _line_offsets(source_text)
    insertions: dict[int, list[str]] = defaultdict(list)
    for patch in pending:
        if patch.mode != "wrap_source":
            continue
        root_attr = patch.scope_fqns[0].split(".", 1)[0]
        constructors = _root_constructors(tree, root_attr)
        if len(constructors) != 1:
            continue
        function = constructors[0]
        _, end = _node_span(function.body[-1], offsets)
        insertions[end].append(_render_deferred_replacement(patch))
        hits[patch.old_ctor] += 1
    return [
        TextEdit(offset, offset, "".join(parts))
        for offset, parts in insertions.items()
    ]


def _root_constructors(tree: ast.Module, root_attr: str) -> list[ast.FunctionDef]:
    """Return constructors that establish the first segment of an observed FQN."""
    constructors = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if not isinstance(child, ast.FunctionDef) or child.name != "__init__":
                continue
            if any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == root_attr
                for item in ast.walk(child)
                if isinstance(item, (ast.Assign, ast.AnnAssign))
                for target in (
                    item.targets if isinstance(item, ast.Assign) else (item.target,)
                )
            ):
                constructors.append(child)
    return constructors


def _render_deferred_replacement(patch: ConstructorReplacePatch) -> str:
    """Render an exact-FQN replacement loop inside a root constructor."""
    arguments = [
        argument
        for argument in patch.keyword_args
        if not argument.startswith("module_fqn=")
    ]
    extra = "".join(f", {argument}" for argument in arguments)
    fqns = repr(tuple(patch.scope_fqns))
    return (
        "\n        _hp_replacement_cache = {}\n"
        f"        for _hp_fqn in {fqns}:\n"
        "            _hp_parent_fqn, _, _hp_name = _hp_fqn.rpartition('.')\n"
        "            _hp_parent = self.get_submodule(_hp_parent_fqn)\n"
        "            _hp_source = getattr(_hp_parent, _hp_name)\n"
        "            _hp_key = id(_hp_source)\n"
        "            if _hp_key not in _hp_replacement_cache:\n"
        f"                _hp_replacement_cache[_hp_key] = {patch.new_ctor}(\n"
        f"                    module=_hp_source, module_fqn=_hp_fqn{extra})\n"
        "            setattr(_hp_parent, _hp_name, _hp_replacement_cache[_hp_key])"
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
