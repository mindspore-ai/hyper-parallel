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
"""Observe method implementations produced by the native injection resolver."""

from __future__ import annotations

import ast
from copy import copy
from dataclasses import dataclass
from functools import partial
import inspect
import textwrap
from typing import Any

from hyper_parallel.distributed._builder.rule_resolver import _resolve_local_compute_fn
from hyper_parallel.distributed.mesh import LogicalDeviceMesh
from hyper_parallel.models.registry import resolve_model_adapter


@dataclass(frozen=True)
class _GroupReference:
    axis: str


@dataclass(frozen=True)
class _RankReference:
    axis: str


class _MethodMesh(LogicalDeviceMesh):
    """Expose logical sizes and symbolic group references, never communication."""

    def __getitem__(self, names):
        view = super().__getitem__(names)
        return _MethodMesh(view.mesh_dim_names, view.shape)

    def get_group(self, name=None):
        """Record the resource a native closure will need when executed."""
        if name is None:
            if len(self.mesh_dim_names) != 1:
                raise ValueError("Codegen method capture requires an explicit group axis")
            name = self.mesh_dim_names[0]
        if name not in self.mesh_dim_names:
            raise ValueError(f"Unknown logical method group {name!r}")
        return _GroupReference(name)

    def get_local_rank(self, name=None):
        """Leave the execution rank symbolic so exported source works on every rank."""
        return _RankReference(self.get_group(name).axis)


class _MethodNames(ast.NodeTransformer):
    def __init__(self, expressions):
        self.expressions = expressions

    def visit_Name(self, node):  # pylint: disable=invalid-name
        if isinstance(node.ctx, ast.Load) and node.id in self.expressions:
            return ast.copy_location(ast.parse(self.expressions[node.id], mode="eval").body, node)
        return node


def _expression(value: Any, imports: set[str]) -> str:
    if isinstance(value, _GroupReference):
        return f"self._hp_{value.axis}_group"
    if isinstance(value, _RankReference):
        return f"self._hp_{value.axis}_mesh.get_local_rank()"
    if isinstance(value, _MethodMesh) and len(value.mesh_dim_names) == 1:
        return f"self._hp_{value.mesh_dim_names[0]}_mesh"
    if isinstance(value, tuple):
        items = ", ".join(_expression(item, imports) for item in value)
        return f"({items},)" if value else "()"
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if inspect.isfunction(value) or inspect.isbuiltin(value):
        name = value.__qualname__
        module = value.__module__
        if "<" not in name:
            alias = "_native_" + module.replace(".", "_")
            imports.add(f"import {module} as {alias}")
            return f"{alias}.{name}"
        tree = ast.parse(textwrap.dedent(inspect.getsource(value)))
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
            function = tree.body[0]
            if function.decorator_list:
                raise ValueError(f"Codegen cannot express decorated closure {name!r}")
            closure = inspect.getclosurevars(value)
            alias = "_native_" + module.replace(".", "_")
            imports.add(f"import {module} as {alias}")
            expressions = {key: f"{alias}.{key}" for key in closure.globals}
            for key, item in closure.nonlocals.items():
                expression = _expression(item, imports)
                if "self." in expression:
                    if not function.args.args:
                        raise ValueError(f"Codegen closure {name!r} has no resource owner argument")
                    expression = expression.replace("self.", function.args.args[0].arg + ".")
                expressions[key] = expression
            function = _MethodNames(expressions).visit(function)
            local_name = name.replace(".<locals>.", "_").replace(".", "_")
            function.name = f"_native_{module.replace('.', '_')}_{local_name}"
            function.returns = None
            for argument in ast.walk(function.args):
                if isinstance(argument, ast.arg):
                    argument.annotation = None
            imports.add(ast.unparse(function))
            return function.name
        lambdas = [node for node in ast.walk(tree) if isinstance(node, ast.Lambda)]
        if len(lambdas) == 1 and not inspect.getclosurevars(value).nonlocals:
            return ast.unparse(lambdas[0])
    raise ValueError(
        f"Codegen cannot express native method state {type(value).__name__}: {getattr(value, '__qualname__', '')}"
    )


def _method_source(function: Any, name: str, imports: set[str]) -> str:
    if isinstance(function, partial):
        function = function.func
    if inspect.ismethod(function):
        function = function.__func__
    node, = ast.parse(textwrap.dedent(inspect.getsource(function))).body
    if not isinstance(node, ast.FunctionDef) or node.decorator_list:
        raise ValueError(f"Codegen cannot express native method {name!r}")
    closure = inspect.getclosurevars(function)
    alias = "_native_" + function.__module__.replace(".", "_")
    imports.add(f"import {function.__module__} as {alias}")
    expressions = {key: f"{alias}.{key}" for key in closure.globals}
    expressions.update({key: _expression(value, imports) for key, value in closure.nonlocals.items()})
    first = node.args.args[0].arg
    expressions[first] = "self"
    node = _MethodNames(expressions).visit(node)
    node.name = name
    node.args.args[0].arg = "self"
    node.returns = None
    for argument in ast.walk(node.args):
        if isinstance(argument, ast.arg):
            argument.annotation = None
    return ast.unparse(node)


def _source_spec(spec: Any, provider: Any) -> Any:
    """Pass an optional model-owned source view through the unchanged native resolver."""
    select = getattr(provider, "source_compute_factory", None)
    if select is None:
        return spec
    original = spec.local_compute_fn
    factory = getattr(original, "_target_", original)
    selected = select(factory)
    if selected is factory:
        return spec
    result = copy(spec)
    if hasattr(original, "_target_"):
        target = copy(original)
        target._target_ = selected  # pylint: disable=protected-access
        result.local_compute_fn = target
    else:
        result.local_compute_fn = selected
    return result


def observe_local_methods(plan: Any, model: Any, mesh_context: Any) -> dict[str, dict]:
    """Run the original resolver on the isolated meta model and export its result.

    Logical groups are placeholders for closure resources. Factories that attempt
    real communication fail here; no process group or device is created.
    """
    entries = [(path, spec) for path, spec in getattr(plan, "modules", {}).items()
               if getattr(spec, "local_compute_fn", None) is not None]
    if not entries:
        return {}
    dense = mesh_context.device_mesh
    expert = mesh_context.fsdp_moe_mesh
    mesh = _MethodMesh(dense.mesh_dim_names, dense.shape)
    expert_mesh = _MethodMesh(expert.mesh_dim_names, expert.shape) if expert is not None else None
    modules = dict(model.named_modules(remove_duplicate=False))
    adapter = resolve_model_adapter(model)
    loader = getattr(adapter, "codegen", None)
    provider = loader() if loader is not None else None
    before = {path: dict(module.__dict__) for path, module in modules.items()}
    try:
        resolved = {}
        for path, spec in entries:
            function = _resolve_local_compute_fn(
                modules[path], _source_spec(spec, provider), mesh, plan.mesh_dim_names, expert_mesh,
            )
            factory = spec.local_compute_fn
            factory = getattr(factory, "_target_", factory)
            resolved.setdefault(path, {})[factory.__name__] = function
        return _describe_methods(modules, before, resolved)
    finally:
        for path, module in modules.items():
            module.__dict__.clear()
            module.__dict__.update(before[path])


def _describe_methods(modules: dict, before: dict, resolved: dict) -> dict[str, dict]:
    for path in resolved:
        expected = {key for key, module in modules.items() if type(module) is type(modules[path])}
        selected = {key for key in resolved if type(modules[key]) is type(modules[path])}
        if selected != expected:
            raise ValueError(f"Codegen cannot apply one native local method to a partially selected class at {path!r}")
    result = {}
    for path, module in modules.items():
        methods = dict(resolved.get(path, {}))
        fields = {}
        imports = set()
        for name, value in module.__dict__.items():
            if name in before[path] and value is before[path][name]:
                continue
            if inspect.ismethod(value) and value.__self__ is module:
                methods[name] = value
            elif name not in {"_parameters", "_buffers", "_modules"}:
                fields[name] = _expression(value, imports)
        if not methods and not fields:
            continue
        result[path] = {
            "type": f"{type(module).__module__}:{type(module).__qualname__}",
            "methods": {name: _method_source(function, name, imports) for name, function in methods.items()},
            "fields": fields,
            "imports": sorted(imports),
            "local_compute": path in resolved,
            "dispatch": getattr(before[path].get("forward"), "__name__", None) if path in resolved else None,
        }
    return result
