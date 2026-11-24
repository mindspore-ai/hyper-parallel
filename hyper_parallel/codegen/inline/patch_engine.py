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
import keyword
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


def apply_patch_set(
    source_text: str, patch_set: InlinePatchSet, *, root_class_name: str | None = None,
) -> str:
    """Apply imports, constructor edits, class removals, markers, and forward bodies.

    Args:
        source_text: Modeling source the patches are applied to.
        patch_set: Edits emitted by the inline passes.
        root_class_name: Observed root model class, disambiguating composed source.

    Returns:
        The patched source text.
    """

    text, hits = _apply_source_declarations(source_text, patch_set)
    text = _apply_constructor_followups(text, patch_set, hits, root_class_name)
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
    root_class_name: str | None,
) -> str:
    """Apply deferred replacements, validate scopes, and remove replaced classes."""
    deferred_edits = _deferred_replacement_edits(
        source_text,
        patch_set.constructor_replaces,
        hits,
        root_class_name,
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
        if not patch.defer_to_fqns:
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
        start, end = _node_span(call, offsets)
        replacement = _constructor_replacement(source_text, tree, node, patch, offsets)
        edits.append(TextEdit(start, end, replacement))
    return _drop_nested_edits(edits)


def _constructor_replacement(source: str, tree: ast.Module, node: ast.Assign,
                             patch: ConstructorReplacePatch, offsets: list[int]) -> str:
    """Reuse a source call, wrapping it or binding its existing config argument."""
    call = node.value
    start, end = _node_span(call, offsets)
    original = source[start:end]
    extra = "".join(f", {arg}" for arg in patch.keyword_args)
    if patch.mode == "wrap_source":
        return f"{patch.new_ctor}(module={original}{extra})"
    if patch.mode == "construct":
        return _direct_constructor_replacement(tree, call, patch)
    if patch.mode == "name":
        _, function_end = _node_span(call.func, offsets)
        original = source[function_end:end]
        return patch.new_ctor + original
    if patch.mode != "config":
        raise ValueError(f"unknown constructor replacement mode: {patch.mode!r}")
    return _config_replacement(tree, node, patch, extra)


def _direct_constructor_replacement(tree: ast.Module, call: ast.Call, patch: ConstructorReplacePatch) -> str:
    """Keep source defaults when a concrete member has different direct-construction defaults."""
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(kw.arg is None for kw in call.keywords):
        raise ValueError(f"codegen: {patch.old_ctor} has unpacked arguments that cannot be bound offline")
    positional, defaults = _source_constructor_parameters(tree, patch.old_ctor)
    supplied = set(positional[:len(call.args)]) | {kw.arg for kw in call.keywords}
    arguments = list(call.args)
    keywords = {kw.arg: kw.value for kw in call.keywords}
    keywords.update({name: value for name, value in defaults.items() if name not in supplied})
    overrides = ast.parse(f"_constructor({', '.join(patch.keyword_args)})", mode="eval").body
    for kw in overrides.keywords:
        if kw.arg in positional[:len(arguments)]:
            arguments[positional.index(kw.arg)] = kw.value
        else:
            keywords[kw.arg] = kw.value
    result = ast.Call(func=ast.Name(id=patch.new_ctor, ctx=ast.Load()), args=arguments,
                      keywords=[ast.keyword(arg=name, value=value) for name, value in keywords.items()])
    return ast.unparse(result)


def _source_constructor_parameters(tree: ast.Module, name: str) -> tuple[list[str], dict[str, ast.expr]]:
    """Read parameter names and default expressions from the copied source without executing it."""
    owner = next((item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name), None)
    constructor = next((item for item in owner.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"),
                       None) if owner is not None else None
    if constructor is None:
        raise ValueError(f"codegen: {name} has no source constructor for direct replacement")
    positional = [item.arg for item in (*constructor.args.posonlyargs, *constructor.args.args)][1:]
    defaults = dict(zip(positional[-len(constructor.args.defaults):], constructor.args.defaults))
    defaults.update({arg.arg: value for arg, value in zip(constructor.args.kwonlyargs, constructor.args.kw_defaults)
                     if value is not None})
    return positional, defaults


def _config_replacement(tree: ast.Module, node: ast.Assign, patch: ConstructorReplacePatch, extra: str) -> str:
    """Bind the config from the existing source constructor without evaluating it."""
    call = node.value
    owner = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == patch.old_ctor)
    constructor = next(item for item in owner.body if isinstance(item, ast.FunctionDef) and item.name == "__init__")
    names = [item.arg for item in (*constructor.args.posonlyargs, *constructor.args.args) if item.arg != "self"]
    arguments = dict(zip(names, call.args))
    arguments.update({kw.arg: kw.value for kw in call.keywords if kw.arg is not None})
    if "config" not in arguments:
        raise ValueError(f"codegen: {patch.old_ctor} has no source config argument")
    result = f"{patch.new_ctor}(config={ast.unparse(arguments['config'])}{extra})"
    return result + _preserved_source_attributes(constructor, node, arguments)


def _preserved_source_attributes(constructor: ast.FunctionDef, node: ast.Assign, arguments: dict) -> str:
    """Retain explicit source constructor identity fields such as layer indices."""
    lines = []
    for statement in constructor.body:
        if (isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Name)
                and statement.value.id in arguments and statement.value.id != "config"):
            for target in statement.targets:
                if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                        and target.value.id == "self"):
                    lines.append(f"\n{' ' * node.col_offset}{ast.unparse(node.targets[0])}.{target.attr} = "
                               f"{ast.unparse(arguments[statement.value.id])}")
    return "".join(lines)


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
    root_class_name: str | None,
) -> list[TextEdit]:
    """Install observed wrappers after an imported owner built its subtree.

    Some source models inherit a decoder whose constructor lives in an
    installed dependency. The copied modeling unit therefore has no direct
    constructor call to rewrite. For a wrap-style replacement, the native
    operation is still expressible exactly: wait until the root model has
    built the observed FQNs, then replace those module attributes in place.
    Module-family requests also use this form: different layers can share the
    same source constructor while requiring distinct options and provenance.
    """
    pending = [
        patch
        for patch in patches
        if patch.scope_fqns and (patch.defer_to_fqns or not hits.get(patch.old_ctor))
    ]
    if not pending:
        return []
    tree = ast.parse(source_text)
    offsets = _line_offsets(source_text)
    insertions: dict[int, list[str]] = defaultdict(list)
    for patch in pending:
        if patch.mode != "wrap_source":
            continue
        loop_edit = _prepared_layer_replacement(tree, patch, root_class_name, offsets)
        if loop_edit is not None:
            insertions[loop_edit.start].append(loop_edit.replacement)
            hits[patch.old_ctor] += 1
            continue
        root_attr = patch.scope_fqns[0].split(".", 1)[0]
        constructors = _root_constructors(tree, root_attr, root_class_name)
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


def _prepared_layer_replacement(tree: ast.Module, patch: ConstructorReplacePatch,
                                root_class: str | None, offsets: list[int]) -> TextEdit | None:
    """Reuse an existing layer preparation loop when the child constructor is imported."""
    if patch.shared_fqns:
        return None
    scope = _layer_scope(patch.scope_fqns)
    if scope is None:
        return None
    paths, index, prefix, suffix = scope
    owner = _source_constructor(tree, root_class, prefix[:-1])
    if owner is None:
        return None
    candidates = _prepared_layer_aliases(owner, prefix[-1])
    if len(candidates) != 1:
        return None
    loop, alias = candidates[0]
    if any(isinstance(node, (ast.Break, ast.Continue, ast.Return)) for node in ast.walk(loop)):
        return None
    access = ".".join([alias.targets[0].id, *suffix])
    extra = "".join(f", {arg}" for arg in patch.keyword_args if not arg.startswith("module_fqn="))
    body = f"{access} = {patch.new_ctor}(module={access}{extra})"
    if not patch.scope_complete:
        selected = tuple(int(path[index]) for path in paths)
        body = f"if {loop.target.id} in {selected!r}:\n    {body}"
    indent = " " * alias.col_offset
    body = "\n" + "\n".join(indent + line for line in body.splitlines()) + "\n"
    _, end = _node_span(loop.body[-1], offsets)
    return TextEdit(end, end, body)


def _layer_scope(fqns: tuple[str, ...]) -> tuple | None:
    """Accept one literal indexed container and a common child suffix."""
    paths = [fqn.split(".") for fqn in fqns]
    numeric = [index for index, part in enumerate(paths[0]) if part.isdecimal()]
    if len(numeric) != 1:
        return None
    index = numeric[0]
    prefix, suffix = paths[0][:index], paths[0][index + 1:]
    if not prefix or not suffix or any(path[:index] != prefix or path[index + 1:] != suffix for path in paths):
        return None
    return paths, index, prefix, suffix


def _source_constructor(tree: ast.Module, root_class: str | None, path: list[str]) -> ast.FunctionDef | None:
    """Follow only literal self.child = LocalClass(...) ownership in copied source."""
    constructors = {
        cls.name: next((node for node in cls.body if isinstance(node, ast.FunctionDef)
                        and node.name == "__init__"), None)
        for cls in tree.body if isinstance(cls, ast.ClassDef)
    }
    constructor = constructors.get(root_class)
    for name in path:
        if constructor is None:
            return None
        calls = [node.value for node in ast.walk(constructor)
                 if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                 and _assigned_attr(node) == name and ast.unparse(node.targets[0]) == f"self.{name}"]
        if len(calls) != 1:
            return None
        constructor = constructors.get(_call_name(calls[0].func))
    return constructor


def _prepared_layer_aliases(owner: ast.FunctionDef, container: str) -> list[tuple[ast.For, ast.Assign]]:
    """Recognize native for-loops that bind layer = self.layers[index]."""
    candidates = []
    iteration = _container_iteration(owner, container)
    for loop in (node for node in ast.walk(owner) if isinstance(node, ast.For) and isinstance(node.target, ast.Name)):
        if ast.dump(loop.iter) != iteration:
            continue
        for node in loop.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Attribute)
                    and ast.unparse(node.value.value) == f"self.{container}"
                    and ast.unparse(node.value.slice) == loop.target.id):
                candidates.append((loop, node))
    return candidates


def _container_iteration(owner: ast.FunctionDef, container: str) -> str | None:
    """Prove a preparation loop covers the same range used to create the ModuleList."""
    for node in owner.body:
        if (isinstance(node, ast.Assign) and _assigned_attr(node) == container
                and ast.unparse(node.targets[0]) == f"self.{container}"
                and isinstance(node.value, ast.Call) and node.value.args
                and ast.unparse(node.value.func) in ("nn.ModuleList", "torch.nn.ModuleList")):
            value = node.value.args[0]
            if isinstance(value, ast.ListComp) and len(value.generators) == 1:
                iteration = value.generators[0].iter
                if (isinstance(iteration, ast.Call) and ast.unparse(iteration.func) == "range"
                        and len(iteration.args) == 1 and not iteration.keywords):
                    return ast.dump(iteration)
    return None


def _root_constructors(
    tree: ast.Module, root_attr: str, root_class_name: str | None,
) -> list[ast.FunctionDef]:
    """Return constructors that establish the first segment of an observed FQN."""
    constructors = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if root_class_name not in (None, node.name):
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
    """Render explicit module assignments, preserving observed shared ownership."""
    arguments = [arg for arg in patch.keyword_args if not arg.startswith("module_fqn=")]
    extra = "".join(f", {arg}" for arg in arguments)
    aliases = {fqn: group[0] for group in patch.shared_fqns for fqn in group[1:]}
    primaries = [fqn for fqn in patch.scope_fqns if fqn not in aliases]
    lines = []
    for fqn in primaries:
        value = (f"{patch.new_ctor}(\n"
                 f"            module={_module_access(fqn)}, module_fqn={fqn!r}{extra})")
        lines.extend((f"\n        # {fqn}: {patch.old_ctor} -> {patch.new_ctor}",
                      "        " + _module_assignment(fqn, value)))
    for fqn in patch.scope_fqns:
        if fqn in aliases:
            primary = aliases[fqn]
            if primary not in primaries:
                raise ValueError(f"codegen: shared module {fqn!r} has no replacement at {primary!r}")
            lines.extend((f"\n        # {fqn}: shares the replacement at {primary}",
                          "        " + _module_assignment(fqn, _module_access(primary))))
    return "\n".join(lines) + "\n"


def _module_access(fqn: str) -> str:
    """Access native child names without guessing their container type."""
    expression = "self"
    for name in fqn.split("."):
        expression = (f"{expression}.{name}" if name.isidentifier() and not keyword.iskeyword(name)
                      else f"getattr({expression}, {name!r})")
    return expression


def _module_assignment(fqn: str, value: str) -> str:
    """Keep ordinary attributes readable and support non-identifier child names."""
    parent, _, name = fqn.rpartition(".")
    if name.isidentifier() and not keyword.iskeyword(name):
        return f"{_module_access(fqn)} = {value}"
    return f"setattr({_module_access(parent) if parent else 'self'}, {name!r}, {value})"


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
