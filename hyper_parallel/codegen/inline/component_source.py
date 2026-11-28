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
"""Project native rewrites and replacement components into the existing source pipeline."""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from typing import Any, Callable

from hyper_parallel.codegen.astkit.edits import TextEdit, apply_edits
from hyper_parallel.codegen.inline.attention import (
    _component_dependencies,
    _referenced_global_names,
    _strip_annotations,
    render_attention_class,
)
from hyper_parallel.codegen.inline.expansion import ExpandedSource
from hyper_parallel.codegen.inline.recognition import (
    _assigned_once,
    _closure_source_bindings,
    _companion_attr_bindings,
    _function_source_qualname,
    _nested_forward_body,
    _wrapper_definition,
)
from hyper_parallel.distributed.recipe_spec import WRAPPER_CONTEXT


def render_component_source(
    component: type, *, interface: Callable[..., Any] | None = None,
) -> ExpandedSource:
    """Expose a native forward signature for the existing boundary lowering pass.

    Args:
        component: Concrete implementation already selected by native preparation.
        interface: Optional kernel interface used by the existing attention renderer.
    """
    if interface is not None:
        return render_attention_class(component, interface=interface)
    forward = ast.parse(textwrap.dedent(inspect.getsource(component.forward))).body[0]
    _strip_annotations(forward)
    forward.decorator_list = []
    tuple_output = any(isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
                       for node in ast.walk(forward))
    forward.body = ast.parse(_native_forward_call(forward, tuple_output=tuple_output)).body
    owner = next(base for base in component.__mro__ if "forward" in base.__dict__)
    _, imports = _component_dependencies(owner, ast.unparse(forward), provided={component.__name__},
                                         inline_definitions=False)
    source = (f"class {component.__name__}({component.__module__}.{component.__name__}):\n"
              + textwrap.indent(ast.unparse(forward), "    ") + "\n")
    return ExpandedSource((f"import {component.__module__}", *imports), source)


def _native_forward_call(forward: ast.FunctionDef, *, tuple_output: bool = False) -> str:
    """Delegate all arguments without rebinding the native defining-class closure."""
    args = forward.args
    arguments = [arg.arg for arg in (*args.posonlyargs, *args.args)][1:]
    if args.vararg:
        arguments.append(f"*{args.vararg.arg}")
    arguments.extend(f"{arg.arg}={arg.arg}" for arg in args.kwonlyargs)
    if args.kwarg:
        arguments.append(f"**{args.kwarg.arg}")
    call = f"super().forward({', '.join(arguments)})"
    return f"return (*{call},)" if tuple_output else f"return {call}"


def collapse_unmodified_components(source: str) -> str:
    """Replace untouched native forwarding shells with their original class imports.

    Boundary lowering needs a signature to select its existing templates. Only
    shells that lowering left untouched can collapse; rewritten forwards remain
    visible and continue inheriting native constructors and parallel stages.
    """
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    tree = ast.parse(source)
    edits = []
    collapsed = {}
    for node in tree.body:
        module = _unchanged_component_base(node)
        if module is None:
            continue
        collapsed[node] = (module, node.name)
        edits.append(TextEdit(offsets[node.lineno - 1], offsets[node.end_lineno], ""))
    for node in tree.body:
        if not isinstance(node, ast.Import) or len(node.names) != 1:
            continue
        module = node.names[0].name
        names = sorted(name for owner, name in collapsed.values() if owner == module)
        if not names:
            continue
        still_used = _component_module_used(tree, module, collapsed)
        imports = (f"import {module}\n" if still_used else "") + f"from {module} import {', '.join(names)}\n"
        edits.append(TextEdit(offsets[node.lineno - 1], offsets[node.end_lineno], imports))
    return apply_edits(source, edits) if edits else source


def _component_module_used(tree: ast.Module, module: str, collapsed: dict) -> bool:
    """Keep namespace imports still used by rewritten subclasses or other source."""
    return any(
        isinstance(child, ast.Attribute) and ast.unparse(child).startswith(module + ".")
        for other in tree.body if other not in collapsed
        for child in ast.walk(other)
    )


def _unchanged_component_base(node: ast.AST) -> str | None:
    """Recognize only a same-name native subclass forwarding all arguments unchanged."""
    if not isinstance(node, ast.ClassDef) or len(node.bases) != 1:
        return None
    base = node.bases[0]
    if not isinstance(base, ast.Attribute) or base.attr != node.name:
        return None
    module = ast.unparse(base.value)
    if not module.startswith("hyper_parallel.components.modules."):
        return None
    component = getattr(importlib.import_module(module), node.name)
    expected = ast.parse(render_component_source(component).source).body[0]
    return module if ast.dump(node) == ast.dump(expected) else None


__all__ = ["render_component_source", "observe_wrapper_source"]


def observe_wrapper_source(
    target: Any, forward: Callable[..., Any], companions: dict[str, Any], *, original_forward: Any,
) -> dict[str, Any]:
    """Lower a native rewrite while its closure is alive; retain source strings only.

    ``original_forward`` identifies the method saved by the native patch.
    Models and closures stay scoped; runtime mesh and communication state are
    never queried or synthesized by this source reader.
    """
    if getattr(forward, "__func__", forward) is getattr(original_forward, "__func__", original_forward):
        return {"body": None, "imports": []}
    function = getattr(forward, "__func__", forward)
    node, captures = _wrapper_capture(function)
    source_bindings = _installed_closure_bindings(function, node.name)
    target_names, original_names = _bound_capture_names(captures, target, original_forward)
    bindings, literals = _closure_bindings(function, captures, companions)
    cp_scalars = _native_cp_scalars(function, captures, bindings)
    if cp_scalars:
        offset = 1 if ast.get_docstring(node) is not None else 0
        node.body[offset:offset] = ast.parse("\n".join(
            f"{name} = cp_mesh.{method}()" for name, method in sorted(cp_scalars.items())
        )).body
    original = original_names[0] if original_names else "original_forward"
    target_name = target_names[0] if target_names else "target_module"
    parameters = ", ".join(sorted(set(captures) | {target_name} | ({"cp_mesh"} if cp_scalars else set())))
    attrs = ", ".join(f"{attr!r}: {name}" for name, attr in bindings.items() if attr is not None)
    source = (f"def observed({parameters}):\n    {original} = {target_name}.forward\n"
              + textwrap.indent(ast.unparse(node), "    ")
              + f"\n    return _ForwardRewriteRequest({target_name}, {node.name}, companion_attrs={{{attrs}}})\n")
    tree = ast.parse(source).body[0]
    body = _nested_forward_body(tree, node.name, target_name=target_name, literals=literals,
                                bound_globals=set(source_bindings))
    if body is None:
        raise RuntimeError(f"codegen cannot lower observed forward {forward.__qualname__}")
    globals_used = _referenced_global_names("\n".join((*source_bindings.values(), ast.unparse(node)))) - set(captures)
    owner = function.__globals__["__name__"]
    imports = [(owner, name) for name in sorted(globals_used & function.__globals__.keys())]
    if "_get_cp_submesh" in body:
        imports.append(("hyper_parallel.distributed._builder.applier", "_get_cp_submesh"))
    return {"body": body, "imports": imports, "bindings": tuple(source_bindings.values()),
            "definition": _wrapper_definition(node, body, getattr(forward, "__self__", None) is target)}


def _wrapper_capture(function: Any) -> tuple:
    """Read an already-installed closure without executing its owner or runtime resources."""
    node = ast.parse(textwrap.dedent(inspect.getsource(function.__code__))).body[0]
    node.decorator_list = []
    _strip_annotations(node)
    return node, dict(inspect.getclosurevars(function).nonlocals)


def _installed_closure_bindings(function: Any, name: str) -> dict[str, str]:
    """Resolve the native owner's aliases without inspecting arbitrary global runtime objects."""
    qualname = _function_source_qualname(function)
    owner = function.__globals__.get(qualname.partition(".<locals>.")[0])
    node = ast.parse(textwrap.dedent(inspect.getsource(owner.__code__))).body[0] if inspect.isfunction(owner) else None
    imported = {symbol: value.__name__ for symbol, value in function.__globals__.items()
                if inspect.ismodule(value) or inspect.isfunction(value) or inspect.isbuiltin(value)}
    return _closure_source_bindings(node, name, imported)


def _closure_bindings(function: Any, captures: dict[str, Any], companions: dict[str, Any]) -> tuple[dict, dict]:
    """Project configuration and native-published state without retaining objects."""
    bindings = {name: next((attr for attr, value in companions.items() if value is captured), None)
                for name, captured in captures.items() if not isinstance(captured, (str, int, float, bool, type(None)))}
    declared, parameters = _native_companion_bindings(function)
    bindings.update({name: attr for name, attr in declared.items() if name in captures
                     and attr in companions and companions[attr] is captures[name]})
    literals = {name: value for name, value in captures.items()
                if isinstance(value, (str, int, float, bool, type(None)))
                and name not in {"cp_rank", "cp_size"} and name not in WRAPPER_CONTEXT
                and (not bindings.get(name) or name in parameters)}
    return bindings, literals


def _bound_capture_names(captures: dict[str, Any], target: Any, original: Any) -> tuple[list[str], list[str]]:
    """Identify the native target and the specific saved forward, not arbitrary methods."""
    targets = [name for name, value in captures.items() if value is target]
    forwards = [name for name, value in captures.items()
                if getattr(value, "__self__", None) is target
                and getattr(value, "__func__", None) is getattr(original, "__func__", None)]
    return targets, forwards


def _native_companion_bindings(function: Any) -> tuple[dict[str, str], set[str]]:
    """Preserve the native request's named state, including optional None contexts."""
    qualname = _function_source_qualname(function)
    owner = function.__globals__.get(qualname.partition(".<locals>.")[0])
    if not inspect.isfunction(owner):
        return {}, set()
    tree = ast.parse(textwrap.dedent(inspect.getsource(owner.__code__)))
    return _companion_attr_bindings(tree), set(inspect.signature(owner, follow_wrapped=False).parameters)


def _native_cp_scalars(function: Any, captures: dict[str, Any], bindings: dict[str, str]) -> dict[str, str]:
    """Prove captured CP scalars originate in native context fields without reading a mesh."""
    qualname = _function_source_qualname(function)
    owner = function.__globals__.get(qualname.partition(".<locals>.")[0])
    if not inspect.isfunction(owner):
        return {}
    tree = ast.parse(textwrap.dedent(inspect.getsource(owner.__code__)))
    parameters = set(inspect.signature(owner, follow_wrapped=False).parameters)
    cp_scalars = {}
    for node in tree.body[0].body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if bindings.get(name) or _cp_context_field(node.value, parameters, captures.get(name)) != name:
            continue
        if not _assigned_once(tree, name):
            continue
        cp_scalars[name] = "get_local_rank" if name == "cp_rank" else "size"
    return cp_scalars


def _cp_context_field(value: ast.expr, parameters: set[str], captured: Any) -> str | None:
    """Recognize one integer CP field read from an unchanged native context parameter."""
    if not isinstance(captured, int) or isinstance(captured, bool):
        return None
    arguments = _context_getattr_arguments(value)
    if not arguments:
        return None
    context, field = arguments[:2]
    if not isinstance(context, ast.Name) or context.id not in parameters:
        return None
    if not isinstance(field, ast.Constant) or field.value not in {"cp_rank", "cp_size"}:
        return None
    expected = ast.Constant(value=0 if field.value == "cp_rank" else 1)
    if len(arguments) == 3 and ast.dump(arguments[2]) != ast.dump(expected):
        return None
    return field.value


def _context_getattr_arguments(value: ast.expr) -> list[ast.expr]:
    """Read a native getattr or int(getattr(...)) using syntax accepted by the lint runtime."""
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "int":
        if len(value.args) != 1 or value.keywords:
            return []
        value = value.args[0]
    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "getattr"
            and len(value.args) in (2, 3) and not value.keywords):
        return value.args
    return []
