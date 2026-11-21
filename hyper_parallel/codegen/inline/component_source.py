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

from torch import nn

from hyper_parallel.codegen.inline.attention import (
    render_attention_class, _referenced_global_names, _strip_annotations, _component_dependencies,
)
from hyper_parallel.codegen.inline.recognition import (
    _companion_attr_bindings, _forward_helper_name, _helper_keyword_body, _nested_forward_body,
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
    target: Any, forward: Callable[..., Any], companions: dict[str, Any], *, original_forward: Any, cp_mesh: Any,
) -> dict[str, Any]:
    """Lower a native rewrite while its closure is alive; retain source strings only.

    ``original_forward`` is saved before native factory execution. ``cp_mesh``
    is the exact native argument, so unrelated mesh captures cannot borrow its
    runtime binding. Models, closures and communication objects stay scoped.
    """
    if getattr(forward, "__func__", forward) is getattr(original_forward, "__func__", original_forward):
        return {"body": None, "imports": []}
    function = getattr(forward, "__func__", forward)
    node = ast.parse(textwrap.dedent(inspect.getsource(function.__code__))).body[0]
    node.decorator_list = []
    _strip_annotations(node)
    captures: dict[str, Any] = dict(inspect.getclosurevars(forward).nonlocals)
    if ("cp_mesh" in captures and captures["cp_mesh"] is not cp_mesh
            and not any(value is captures["cp_mesh"] for value in companions.values())):
        raise RuntimeError("codegen cannot bind an observed mesh that differs from the native CP argument")
    delegated = _delegate_executor(node, captures, target)
    if delegated is not None:
        return _executor_source(delegated, target, cp_mesh)
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
    body = _observed_body(tree, node.name, function, target_name, literals)
    if body is None:
        raise RuntimeError(f"codegen cannot lower observed forward {forward.__qualname__}")
    globals_used = _referenced_global_names(ast.unparse(node)) - set(captures)
    owner = function.__globals__["__name__"]
    imports = [(owner, name) for name in sorted(globals_used & function.__globals__.keys())]
    if "_get_cp_submesh" in body:
        imports.append(("hyper_parallel.distributed._builder.applier", "_get_cp_submesh"))
    return {"body": body, "imports": imports}


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


def _delegate_executor(node: ast.FunctionDef, captures: dict, target: Any) -> Any:
    """Recognize the native pure delegation without dropping surrounding branches."""
    body = node.body[1:] if ast.get_docstring(node) is not None else node.body
    if len(body) != 1 or not isinstance(body[0], ast.Return) or not isinstance(body[0].value, ast.Call):
        return None
    call = body[0].value
    if not isinstance(call.func, ast.Name):
        return None
    executor = captures.get(call.func.id)
    if not isinstance(executor, nn.Module) or getattr(executor, "module", None) is not target:
        return None
    args = [ast.unparse(item) for item in call.args]
    keywords = [(item.arg, ast.unparse(item.value)) for item in call.keywords]
    if (not node.args.vararg or args != ["*" + node.args.vararg.arg]
            or not node.args.kwarg or keywords != [(None, node.args.kwarg.arg)]):
        raise RuntimeError("codegen executor delegate adapts arguments; source lowering is not proven")
    return executor


def _executor_methods(executor: nn.Module) -> dict[str, ast.FunctionDef]:
    """Read only reachable native methods, including inherited implementations."""
    methods = {}
    pending = ["forward"]
    while pending:
        name = pending.pop()
        if name in methods:
            continue
        method = getattr(executor, name)
        if inspect.getmodule(method) is not inspect.getmodule(type(executor)):
            raise RuntimeError(f"codegen executor method {name!r} has a different dependency owner")
        node = ast.parse(textwrap.dedent(inspect.getsource(method.__func__.__code__))).body[0]
        node.decorator_list = []
        _strip_annotations(node)
        if name != "forward" and callable(getattr(executor.module, name, None)):
            raise RuntimeError(f"codegen executor helper {name!r} collides with the original model")
        methods[name] = node
        pending.extend(item.func.attr for item in ast.walk(node)
                       if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                       and isinstance(item.func.value, ast.Name) and item.func.value.id == "self"
                       and inspect.ismethod(getattr(executor, item.func.attr, None)))
    return methods


def _executor_bindings(executor: nn.Module, target: Any, cp_mesh: Any) -> dict[str, str]:
    """Bind flat configuration and native CP metadata without serializing executor state."""
    if getattr(executor, "cp_mesh", None) is not cp_mesh:
        raise RuntimeError("codegen executor uses a derived mesh; its source binding is not proven")
    values = {name: repr(value) for name, value in vars(executor).items()
              if isinstance(value, (str, int, float, bool, type(None))) and not name.startswith("_")
              and not name.endswith("rank")}
    mesh = "self._hp_boundary.compiled.mesh['cp']"
    values.update(module="self", training="self.training", cp_mesh=mesh,
                  cp_rank=mesh + ".get_local_rank()", cp_size=mesh + ".size()", cp_group=mesh + ".get_group()")
    if getattr(executor, "module", None) is not target:
        raise RuntimeError("codegen executor does not borrow the native target's parameters")
    return values


def _executor_source(executor: nn.Module, target: Any, cp_mesh: Any) -> dict[str, Any]:
    """Materialize a flat executor into genuine model methods, never a replay object."""
    methods = _executor_methods(executor)
    expressions = _executor_bindings(executor, target, cp_mesh)

    class Bind(ast.NodeTransformer):
        """Map executor-owned reads while leaving original-layer attributes untouched."""

        def visit_Attribute(self, node):  # pylint: disable=invalid-name
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                if not isinstance(node.ctx, ast.Load):
                    raise RuntimeError(f"codegen executor mutates state {node.attr!r}; lowering is not proven")
                if node.attr in expressions:
                    return ast.copy_location(ast.parse(expressions[node.attr], mode="eval").body, node)
                if node.attr not in methods:
                    raise RuntimeError(f"codegen executor state {node.attr!r} needs a proven source binding")
            return self.generic_visit(node)

    for node in methods.values():
        Bind().visit(node)
    definition = "\n\n".join(ast.unparse(methods[name]) for name in sorted(methods) if name != "forward")
    definition += "\n\n" + ast.unparse(methods["forward"])
    code = "class ExecutorSource:\n" + textwrap.indent(definition, "    ")
    helpers, imports = _component_dependencies(type(executor), code, provided={"ExecutorSource"})
    return {"body": "\n".join(ast.unparse(item) for item in methods["forward"].body),
            "definition": definition.strip(), "imports": [], "source_imports": list(imports), "helpers": helpers}


def _observed_body(tree: ast.AST, name: str, function: Any, target_name: str, literals: dict) -> str | None:
    """Reuse the established helper and closure lowering contracts."""
    helper = function.__globals__.get(_forward_helper_name(tree, name))
    body = None
    if inspect.isfunction(helper):
        helper_node = ast.parse(textwrap.dedent(inspect.getsource(helper.__code__))).body[0]
        body = _helper_keyword_body(tree, name, helper_node, literals)
    if body is None:
        body = _nested_forward_body(tree, name, target_name=target_name, literals=literals)
    return body
