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
"""Framework structural recognition for inline component selection.

Which generic component replaces a matched source module is decided by the
*structure of the family's own replacement factory* -- the generic component it
constructs, the module that class comes from, the constructor arguments it
passes and the kernel entry it hands over -- never by the model's name and
never by a component-name table.  The factory is not codegen-specific: the YAML
already names it for the native path, so reading it adds no family codegen
declaration.

This is what makes "native HF model + one YAML" sufficient: a family whose
attention is fused-QKV, whose experts are batched, or whose norm is an RMSNorm
is covered the moment its runtime factory exists.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
import inspect
import os
import sys
import textwrap
from typing import Any


@dataclass(frozen=True)
class ReplacementArgument:
    """One constructor keyword a replacement factory passes.

    ``value`` is the argument's source text; ``None`` marks an argument bound to
    the factory's own parameter (a runtime value such as ``module_fqn``), whose
    generated value is the component's constructor default.
    """

    name: str
    value: str | None = None


@dataclass(frozen=True)
class ReplacementTarget:
    """The component a replacement factory constructs, read from its source.

    Every fact the generated replacement needs is here: ``component`` (the class
    the factory builds) and ``module`` (where that class is imported from) give
    the emitted import; ``wraps_source`` says whether the matched module is
    handed over as the component's ``module`` argument (a wrap) or whether the
    component drops in for the source class; ``arguments`` carries the remaining
    constructor keywords; ``interface`` names the family symbol the factory
    hands over -- a component that receives one is rendered rather than merely
    imported, so its class and that interface stay visible in the artifact.
    """

    component: str
    module: str
    wraps_source: bool
    arguments: tuple[ReplacementArgument, ...]
    interface: str | None = None


@dataclass(frozen=True)
class InnerWrapperTarget:
    """Source-visible helper and companion mesh attribute of an inner wrapper."""

    helper_module: str | None = None
    helper_name: str | None = None
    state_attr: str | None = None
    inline_body: str | None = None


def class_name_of(qualified_type: str | None) -> str | None:
    """Return the bare class name of a dotted ``module_type`` path."""
    if not qualified_type:
        return None
    return qualified_type.rsplit(".", 1)[-1] or None


def family_from_target(target: str | None) -> str | None:
    """Infer the adapter family directory from a framework adapter path.

    ``hyper_parallel.models.<family>.adapter...`` -> ``<family>``. Returns
    ``None`` for targets that are not model-adapter paths (for example a
    framework recipe), which callers treat as "no family provider".
    """
    parts = (target or "").split(".")
    if len(parts) >= 3 and parts[:2] == ["hyper_parallel", "models"]:
        return parts[2]
    return None


def adapter_path_parts(target: str | None) -> bool:
    """Whether a target lives under the ``models/<family>/adapter`` convention."""
    parts = (target or "").split(".")
    return len(parts) >= 4 and parts[:2] == ["hyper_parallel", "models"] and parts[3] == "adapter"


def replacement_target(factory: Any) -> ReplacementTarget | None:
    """Read the component one ``@module_replacement`` factory constructs.

    Only a factory that constructs an imported class directly (``Ctor(...)``)
    yields a target: that is the call codegen can sink into the generated
    ``__init__``.  A factory that builds through a class method
    (``SomeExperts.from_module(...)``) has no constructor call to sink, so the
    rule is left untouched -- the same conservative behaviour a component-name
    table gave, without the table.
    """
    try:
        source = textwrap.dedent(inspect.getsource(factory))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return None
    imported = _factory_imported_modules(tree)
    parameters = _parameter_names(factory)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        component = node.func.id
        module = imported.get(component) or _global_class_module(factory, component)
        if module is None:
            continue
        return ReplacementTarget(
            component=component,
            module=module,
            wraps_source=any(
                keyword.arg == "module"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id in parameters
                for keyword in node.keywords
            ),
            arguments=_factory_arguments(node, parameters),
            interface=next(
                (
                    keyword.value.id
                    for keyword in node.keywords
                    if isinstance(keyword.value, ast.Name)
                    and keyword.value.id in getattr(factory, "__globals__", {})
                    and callable(factory.__globals__[keyword.value.id])
                ),
                None,
            ),
        )
    return None


def _factory_imported_modules(tree: ast.AST) -> dict[str, str]:
    """Module path per name the factory source imports (``from X import Y``)."""
    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported.setdefault(alias.asname or alias.name, node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.setdefault(alias.asname or alias.name.split(".")[0], alias.name)
    return imported


def _global_class_module(factory: Any, name: str) -> str | None:
    """Module path of a class the factory's own module imported at module level."""
    value = getattr(factory, "__globals__", {}).get(name)
    module = getattr(value, "__module__", None)
    return module if isinstance(value, type) and module else None


def _parameter_names(factory: Any) -> set[str]:
    """Parameter names of the factory itself (runtime values, not literals)."""
    try:
        return set(inspect.signature(factory).parameters)
    except (TypeError, ValueError):
        return set()


def _factory_arguments(
    call: ast.Call, parameters: set[str]
) -> tuple[ReplacementArgument, ...]:
    """The constructor keywords a factory passes, minus the wrapped module.

    A value that names one of the factory's own parameters is recorded without
    its text (the component's constructor default applies); a value naming a
    callable in the factory's globals -- a kernel entry handed over -- keeps its
    symbol so the artifact stays readable; anything else keeps its own text.
    """
    arguments: list[ReplacementArgument] = []
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        if keyword.arg == "module" and isinstance(keyword.value, ast.Name):
            continue
        value: str | None = (
            None
            if isinstance(keyword.value, ast.Name) and keyword.value.id in parameters
            else ast.unparse(keyword.value)
        )
        arguments.append(ReplacementArgument(keyword.arg, value))
    return tuple(arguments)


def inner_wrapper_target(factory_path: str | None) -> InnerWrapperTarget | None:
    """Recover an inner wrapper's emitted helper contract from its source."""
    if not factory_path:
        return None
    module_name, _, factory_name = factory_path.rpartition(".")
    if not module_name or not factory_name:
        return None
    tree = _module_source_tree(module_name)
    if tree is None:
        return None
    factory_node = _module_function(tree, factory_name)
    if factory_node is None:
        return None
    request_owner = _request_owner(factory_node, tree)
    replacement_name = _replacement_callable(request_owner)
    helper_name = (
        _forward_helper_name(request_owner, replacement_name)
        if request_owner is not None and replacement_name
        else None
    )
    module_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not helper_name or helper_name not in module_functions:
        inline_body = _nested_forward_body(request_owner, replacement_name)
        if inline_body is None:
            return None
        return InnerWrapperTarget(inline_body=inline_body)
    return InnerWrapperTarget(
        helper_module=module_name,
        helper_name=helper_name,
        state_attr=_cp_mesh_companion_attr(tree),
    )


def _module_function(tree: ast.Module, name: str) -> ast.AST | None:
    """Return a named top-level function from a parsed module."""
    return next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ),
        None,
    )


def _replacement_callable(request_owner: ast.AST | None) -> str | None:
    """Resolve the callable installed by an inner-wrapper request."""
    if request_owner is None:
        return None
    return _rewrite_request_callable(request_owner) or _returned_nested_callable(request_owner)


def _request_owner(factory_node: ast.AST, tree: ast.Module) -> ast.AST | None:
    """Return the factory or its module-level delegate that builds the request."""
    if (
        _rewrite_request_callable(factory_node) is not None
        or _returned_nested_callable(factory_node) is not None
    ):
        return factory_node
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for node in ast.walk(factory_node):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Name):
            continue
        delegate = functions.get(node.value.func.id)
        if delegate is not None and _rewrite_request_callable(delegate) is not None:
            return delegate
    return None


def _returned_nested_callable(tree: ast.AST) -> str | None:
    """Return a nested callable returned directly by an inner wrapper."""
    nested = {
        node.name
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name):
            if node.value.id in nested:
                return node.value.id
    return None


@lru_cache(maxsize=None)
def _module_source(module_name: str) -> str | None:
    """Load module source once without importing optional accelerator deps."""
    relative = os.path.join(*module_name.split(".")) + ".py"
    for root in sys.path:
        path = os.path.join(root or os.curdir, relative)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read()
        except (OSError, UnicodeError):
            return None
    return None


def _module_source_tree(module_name: str) -> ast.Module | None:
    """Parse cached module source into an isolated syntax tree."""
    source = _module_source(module_name)
    if source is None:
        return None
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _rewrite_request_callable(tree: ast.AST) -> str | None:
    """Return the replacement callable passed to ``_ForwardRewriteRequest``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == "_ForwardRewriteRequest" and len(node.args) >= 2:
            if isinstance(node.args[1], ast.Name):
                return node.args[1].id
    return None


def _forward_helper_name(tree: ast.AST, replacement_name: str) -> str | None:
    """Return the module-level helper called by the replacement forward."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != replacement_name:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and isinstance(child.value, ast.Call):
                if isinstance(child.value.func, ast.Name):
                    return child.value.func.id
    return None


def _cp_mesh_companion_attr(tree: ast.AST) -> str | None:
    """Return the companion attribute whose value is the wrapper's CP mesh."""
    return (
        _cp_mesh_dict_attr(tree)
        or _cp_mesh_setattr_attr(tree)
        or _cp_mesh_assignment_attr(tree)
    )


def _cp_mesh_dict_attr(tree: ast.AST) -> str | None:
    """Return the CP mesh attribute declared in companion_attrs."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword) or node.arg != "companion_attrs":
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(value, ast.Name)
                and value.id == "cp_mesh"
            ):
                return key.value
    return None


def _cp_mesh_setattr_attr(tree: ast.AST) -> str | None:
    """Return the CP mesh attribute installed through setattr."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) == 3
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and isinstance(node.args[2], ast.Name)
            and node.args[2].id == "cp_mesh"
        ):
            return node.args[1].value
    return None


def _cp_mesh_assignment_attr(tree: ast.AST) -> str | None:
    """Return the CP mesh attribute installed through assignment."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if not isinstance(value, ast.Name) or value.id != "cp_mesh":
            continue
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                return target.attr
    return None


def _nested_forward_body(tree: ast.AST, replacement_name: str | None) -> str | None:
    """Lower a closure-backed rewrite into an imported-host dispatch body.

    A custom wrapper may build its replacement forward as a nested closure
    instead of delegating to a module-level helper.  The closure remains
    source-visible when its captured values are also published through the
    request's ``companion_attrs`` mapping.  In that form, replace captured
    state reads with instance attributes and the saved original forward with
    ``super().forward``; the imported-host materializer supplies the subclass.
    """
    if not replacement_name:
        return None
    forward = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == replacement_name
        ),
        None,
    )
    if forward is None:
        return None
    companion_attrs = _companion_attr_bindings(tree)
    original_forward = _original_forward_binding(tree)
    if original_forward is None:
        return None

    class ClosureBindings(ast.NodeTransformer):
        """Rewrite captured closure bindings to generated-instance state."""

        def visit_Name(self, node: ast.Name) -> ast.AST:  # pylint: disable=invalid-name
            """Rewrite a referenced symbol into its resolved source form."""
            attr = companion_attrs.get(node.id)
            if attr is None:
                return node
            return ast.copy_location(
                ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=attr,
                    ctx=node.ctx,
                ),
                node,
            )

        def visit_Call(self, node: ast.Call) -> ast.AST:  # pylint: disable=invalid-name
            """Rewrite a call after recursively resolving its arguments."""
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == original_forward:
                node.func = ast.Attribute(
                    value=ast.Call(
                        func=ast.Name(id="super", ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    ),
                    attr="forward",
                    ctx=ast.Load(),
                )
            return node

    body = list(forward.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    lowered = [ClosureBindings().visit(ast.fix_missing_locations(node)) for node in body]
    if not lowered:
        return None
    active = "\n".join(ast.unparse(node) for node in lowered)
    return (
        "if self.cp_enable:\n"
        f"{textwrap.indent(active, '    ')}\n\n"
        "return self._forward_impl(__hp_forward_args__)"
    )


def _companion_attr_bindings(tree: ast.AST) -> dict[str, str]:
    """Map closure variable names to their published companion attributes."""
    result: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword) or node.arg != "companion_attrs":
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(value, ast.Name)
            ):
                result[value.id] = key.value
    return result


def _original_forward_binding(tree: ast.AST) -> str | None:
    """Return the local name that saves ``target.forward`` before rewriting."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Attribute)
            and value.attr == "forward"
        ):
            return target.id
    return None


__all__ = [
    "InnerWrapperTarget",
    "ReplacementArgument",
    "ReplacementTarget",
    "adapter_path_parts",
    "class_name_of",
    "family_from_target",
    "inner_wrapper_target",
    "replacement_target",
]
