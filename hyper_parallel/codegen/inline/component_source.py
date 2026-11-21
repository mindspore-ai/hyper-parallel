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
import inspect
import textwrap
from typing import Any, Callable

from hyper_parallel.codegen.inline.attention import (
    render_attention_class, _referenced_global_names, _strip_annotations,
)
from hyper_parallel.codegen.inline.recognition import (
    _companion_attr_bindings, _nested_forward_body,
)
from hyper_parallel.codegen.inline.expansion import ExpandedSource
from hyper_parallel.distributed.recipe_spec import WRAPPER_CONTEXT


def render_component_source(
    component: type, *, interface: Callable[..., Any]
) -> ExpandedSource:
    """Render a component and the source closure of its injected interface."""
    return render_attention_class(component, interface=interface)


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
    target_names, original_names = _bound_capture_names(captures, target, original_forward)
    bindings, literals = _closure_bindings(function, captures, companions)
    original = original_names[0] if original_names else "original_forward"
    target_name = target_names[0] if target_names else "target_module"
    parameters = ", ".join(sorted(set(captures) | {target_name}))
    attrs = ", ".join(f"{attr!r}: {name}" for name, attr in bindings.items() if attr is not None)
    source = (f"def observed({parameters}):\n    {original} = {target_name}.forward\n"
              + textwrap.indent(ast.unparse(node), "    ")
              + f"\n    return _ForwardRewriteRequest({target_name}, {node.name}, companion_attrs={{{attrs}}})\n")
    tree = ast.parse(source).body[0]
    body = _nested_forward_body(tree, node.name, target_name=target_name, literals=literals)
    if body is None:
        raise RuntimeError(f"codegen cannot lower observed forward {forward.__qualname__}")
    globals_used = _referenced_global_names(ast.unparse(node)) - set(captures)
    owner = function.__globals__["__name__"]
    imports = [(owner, name) for name in sorted(globals_used & function.__globals__.keys())]
    if "_get_cp_submesh" in body:
        imports.append(("hyper_parallel.distributed._builder.applier", "_get_cp_submesh"))
    return {"body": body, "imports": imports,
            "definition": _wrapper_definition(node, body, getattr(forward, "__self__", None) is target)}


def _wrapper_capture(function: Any) -> tuple:
    """Read an already-installed closure without executing its owner or runtime resources."""
    node = ast.parse(textwrap.dedent(inspect.getsource(function.__code__))).body[0]
    node.decorator_list = []
    _strip_annotations(node)
    return node, dict(inspect.getclosurevars(function).nonlocals)


def _wrapper_definition(node: ast.FunctionDef, body: str, bound: bool) -> str:
    """Keep the wrapper's actual call contract, including keyword-only forwarding."""
    node.name = "forward"
    if not bound:
        parameters = node.args.posonlyargs if node.args.posonlyargs else node.args.args
        parameters.insert(0, ast.arg(arg="self"))
    if node.args.vararg:
        body = body.replace("__hp_forward_positional__", node.args.vararg.arg)
    if node.args.kwarg:
        body = body.replace("__hp_forward_keywords__", node.args.kwarg.arg)
    node.body = ast.parse(body).body
    return ast.unparse(ast.fix_missing_locations(node))


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
    qualname = getattr(function.__code__, "co_qualname", function.__qualname__)
    owner = function.__globals__.get(qualname.partition(".<locals>.")[0])
    if not inspect.isfunction(owner):
        return {}, set()
    tree = ast.parse(textwrap.dedent(inspect.getsource(owner.__code__)))
    return _companion_attr_bindings(tree), set(inspect.signature(owner, follow_wrapped=False).parameters)
