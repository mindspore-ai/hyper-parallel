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
from collections.abc import Collection
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import inspect
import os
import symtable
import sys
import textwrap
from typing import Any

from hyper_parallel.distributed.recipe_spec import WRAPPER_CONTEXT
from hyper_parallel.codegen.inline.attention import _strip_annotations


def _function_source_qualname(function: Any) -> str:
    """Locate a function's lexical definition even when decorators changed its public name."""
    code = function.__code__
    qualname = getattr(code, "co_qualname", None)
    if qualname is not None:
        return qualname
    lines, _ = inspect.findsource(code)
    qualname = _source_qualname(ast.parse("".join(lines)), code)
    if qualname is None:
        raise RuntimeError(f"codegen: cannot locate source definition {code.co_name!r} "
                           f"at {code.co_filename}:{code.co_firstlineno}")
    return qualname


def _source_qualname(tree: ast.AST, code: Any, prefix: str = "") -> str | None:
    """Match the code's definition line and reconstruct Python's lexical scope path."""
    for node in ast.iter_child_nodes(tree):
        nested_prefix = prefix
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = prefix + node.name
            is_function = not isinstance(node, ast.ClassDef)
            start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
            if is_function and node.name == code.co_name and start == code.co_firstlineno:
                return name
            nested_prefix = name + (".<locals>." if is_function else ".")
        found = _source_qualname(node, code, nested_prefix)
        if found is not None:
            return found
    return None


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
    inline_body: str | None = None
    extra_imports: tuple[tuple[str, str], ...] = ()
    definition: str | None = None
    bindings: tuple[str, ...] = ()


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


def inner_wrapper_target(
    factory_path: str | None, *, construction_kwargs: tuple[str, ...] = (),
) -> InnerWrapperTarget | None:
    """Recover an inner wrapper's emitted helper contract from its source."""
    source = _wrapper_factory_ast(factory_path)
    if source is None:
        return None
    module_name, tree, factory_node = source
    request_owner = _request_owner(factory_node, tree)
    replacement_name = _replacement_callable(request_owner)
    literals = _wrapper_literals(factory_node, construction_kwargs)
    if request_owner is not None and request_owner is not factory_node:
        literals = _delegate_literals(factory_node, request_owner, literals)
    bindings = _closure_source_bindings(request_owner, replacement_name, _factory_imported_modules(tree))
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
    module_symbols = module_functions | _factory_imported_modules(tree).keys()
    if not helper_name or helper_name not in module_functions:
        inline_body = _nested_forward_body(
            request_owner, replacement_name, target_name=_original_forward_target(request_owner),
            literals=literals, bound_globals=set(bindings),
        )
        if inline_body is None:
            return None
        forward = next(node for node in ast.walk(request_owner)
                       if isinstance(node, ast.FunctionDef) and node.name == replacement_name)
        return InnerWrapperTarget(
            inline_body=inline_body,
            extra_imports=_body_imports("\n".join((*bindings.values(), inline_body)), module_name, module_symbols),
            definition=_wrapper_definition(forward, inline_body, False), bindings=tuple(bindings.values()),
        )
    inline_body = _helper_keyword_body(
        request_owner, replacement_name, _module_function(tree, helper_name),
        literals,
    )
    return InnerWrapperTarget(
        helper_module=module_name,
        helper_name=helper_name,
        inline_body=inline_body,
        extra_imports=_body_imports(inline_body, module_name, module_symbols - {helper_name}),
    )


def _wrapper_factory_ast(path: str | None) -> tuple[str, ast.Module, Any] | None:
    """Find the declared factory without importing or invoking it."""
    module_name, _, name = (path or "").rpartition(".")
    if not module_name or not name:
        return None
    tree = _module_source_tree(module_name)
    if tree is None:
        return None
    factory = _module_function(tree, name)
    return (module_name, tree, factory) if factory is not None else None


def _body_imports(body: str | None, module: str, functions: set[str]) -> tuple[tuple[str, str], ...]:
    """Keep native helpers referenced by the lowered closure visible in its new module."""
    names = {node.id for node in ast.walk(ast.parse(body or "")) if isinstance(node, ast.Name)}
    imports = [(module, name) for name in sorted(names & functions)]
    if "_get_cp_submesh" in names:
        imports.append(("hyper_parallel.distributed._builder.applier", "_get_cp_submesh"))
    return tuple(imports)


def _single_helper_delegate(
    tree: ast.AST, replacement_name: str, helper_name: str,
) -> tuple[ast.FunctionDef, ast.Call | None]:
    """Identify a pure return call without discarding surrounding control flow."""
    forward = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == replacement_name)
    body = forward.body[1:] if ast.get_docstring(forward) is not None else forward.body
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return forward, None
    call = body[0].value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id == helper_name and any(item.arg is not None for item in call.keywords)):
        return forward, None
    return forward, call


def _positional_contract(args: ast.arguments, excluded: set[str]) -> tuple[tuple[str, bool, str | None], ...]:
    """Compare names, positional-only constraints and default expressions."""
    parameters = (*args.posonlyargs, *args.args)
    defaults = [None] * (len(parameters) - len(args.defaults)) + list(args.defaults)
    return tuple((item.arg, index < len(args.posonlyargs), ast.dump(value) if value is not None else None)
                 for index, (item, value) in enumerate(zip(parameters, defaults)) if item.arg not in excluded)


def _keyword_names(call: ast.Call) -> set[str]:
    """Collect symbols referenced by explicitly supplied helper keywords."""
    return {node.id for item in call.keywords if item.arg is not None
            for node in ast.walk(item.value) if isinstance(node, ast.Name)}


def _forwards_parameters(args: ast.arguments, call: ast.Call) -> bool:
    """Ensure the token can replace the native delegate's argument forwarding."""
    names = [item.arg for item in (*args.posonlyargs, *args.args)]
    if [ast.unparse(item) for item in call.args[1:]] != names:
        return False
    kwargs = [ast.unparse(item.value) for item in call.keywords if item.arg is None]
    if kwargs != ([args.kwarg.arg] if args.kwarg else []):
        return False
    return not set(names).intersection(_keyword_names(call))


def _helper_signature_matches(forward: ast.FunctionDef, helper: ast.FunctionDef, call: ast.Call) -> bool:
    """Reject signature adaptation rather than changing positional/default behavior."""
    if forward.args.vararg or forward.args.kwonlyargs:
        return False
    parameters = (*helper.args.posonlyargs, *helper.args.args)
    if not parameters:
        return False
    excluded = {item.arg for item in call.keywords if item.arg is not None} | {parameters[0].arg}
    if _positional_contract(forward.args, set()) != _positional_contract(helper.args, excluded):
        return False
    return _forwards_parameters(forward.args, call)


def _original_forward_target(tree: ast.AST | None) -> str | None:
    """Find the owner whose original forward the wrapper saves."""
    if tree is None:
        return None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute)
                and node.value.attr == "forward" and isinstance(node.value.value, ast.Name)):
            return node.value.value.id
    return None


def _wrapper_literals(factory: ast.FunctionDef, construction_kwargs: tuple[str, ...]) -> dict[str, Any]:
    """Read only literal native defaults and recorded overrides, never runtime resources."""
    parameters = (*factory.args.posonlyargs, *factory.args.args)
    values = {}
    for parameter, value in zip(parameters[len(parameters) - len(factory.args.defaults):], factory.args.defaults):
        if parameter.arg in WRAPPER_CONTEXT:
            continue
        try:
            values[parameter.arg] = ast.literal_eval(value)
        except (ValueError, TypeError):
            continue
    for argument in construction_kwargs:
        name, text = argument.split("=", 1)
        if name in WRAPPER_CONTEXT:
            raise ValueError(f"codegen: native runtime argument {name!r} cannot be frozen as configuration")
        values[name] = ast.literal_eval(text)
    return values


def _helper_keyword_body(
    tree: ast.AST, replacement_name: str, helper: ast.FunctionDef, literals: dict[str, Any],
) -> str | None:
    """Keep explicit keywords only when native argument forwarding is preserved."""
    forward, call = _single_helper_delegate(tree, replacement_name, helper.name)
    if call is None:
        return None
    if not _helper_signature_matches(forward, helper, call):
        raise RuntimeError(f"codegen: helper delegate {replacement_name!r} is not an argument passthrough")
    target = _original_forward_target(tree)
    if target is None:
        raise RuntimeError(
            f"codegen: cannot preserve helper arguments for {replacement_name!r} without its target binding"
        )
    if not call.args or not isinstance(call.args[0], ast.Name) or call.args[0].id != target:
        raise RuntimeError(f"codegen: helper delegate {replacement_name!r} has a different target")
    lowered = ast.parse(_nested_forward_body(tree, replacement_name, target_name=target, literals=literals))
    for item in ast.walk(lowered):
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == helper.name:
            names = _keyword_names(item)
            if names - {"self", "_get_cp_submesh"}:
                raise RuntimeError(f"codegen: helper delegate has unresolved keyword names {sorted(names - {'self'})}")
            item.args = [ast.Name(id="self", ctx=ast.Load()), ast.Name(id="__hp_forward_args__", ctx=ast.Load())]
            item.keywords = [keyword for keyword in item.keywords if keyword.arg is not None]
    active = ast.unparse(ast.fix_missing_locations(lowered))
    return (f"if self.cp_enable:\n{textwrap.indent(active, '    ')}\n\n"
            "return self._forward_impl(__hp_forward_args__)")


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
        if delegate is not None and (
            _rewrite_request_callable(delegate) is not None or _returned_nested_callable(delegate) is not None
        ):
            return delegate
    return None


def _delegate_literals(factory: ast.FunctionDef, delegate: ast.FunctionDef, values: dict[str, Any]) -> dict[str, Any]:
    """Bind one same-module delegate using only native defaults, literals and configured arguments."""
    result = {}
    for name, expression in _delegate_expressions(factory, delegate).items():
        if name in WRAPPER_CONTEXT:
            if not isinstance(expression, ast.Name) or expression.id != name:
                raise RuntimeError(f"codegen cannot reinterpret native delegate context {name!r}")
            continue
        if isinstance(expression, ast.Name) and expression.id in values:
            result[name] = values[expression.id]
            continue
        try:
            result[name] = ast.literal_eval(expression)
        except (ValueError, TypeError):
            continue
    return result


def _delegate_expressions(factory: ast.FunctionDef, delegate: ast.FunctionDef) -> dict[str, ast.expr]:
    """Project one direct delegate call onto its native positional and keyword parameters."""
    call = next(node.value for node in ast.walk(factory) if isinstance(node, ast.Return)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == delegate.name)
    parameters = (*delegate.args.posonlyargs, *delegate.args.args)
    expressions = {param.arg: value for param, value in zip(
        parameters[len(parameters) - len(delegate.args.defaults):], delegate.args.defaults,
    )}
    expressions.update({param.arg: value for param, value in zip(delegate.args.kwonlyargs, delegate.args.kw_defaults)
                        if value is not None})
    expressions.update({param.arg: value for param, value in zip(parameters, call.args)})
    if any(keyword.arg is None for keyword in call.keywords):
        raise RuntimeError("codegen cannot bind dynamic inner-wrapper delegate keywords")
    expressions.update({keyword.arg: keyword.value for keyword in call.keywords})
    return expressions


def _closure_source_bindings(
    owner: ast.AST | None, name: str | None, imported: dict[str, str],
) -> dict[str, str]:
    """Keep native module imports and callable aliases outside the generated forward."""
    if owner is None or name is None:
        return {}
    forward = next((node for node in ast.walk(owner) if isinstance(node, ast.FunctionDef) and node.name == name), None)
    if forward is None:
        return {}
    captured = _captured_forward_names(owner, name)
    callables = {node.func.id for node in ast.walk(forward)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assignments = {node.targets[0].id: node for node in owner.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
    bindings: dict[str, str] = {}
    attempted: set[str] = set()

    def resolve(symbol: str) -> bool:
        """Accept only aliases rooted in native imports, with no object construction."""
        if symbol in imported or symbol in bindings:
            return True
        if symbol in attempted:
            return False
        attempted.add(symbol)
        statement = assignments.get(symbol)
        if statement is None:
            return False
        if not _assigned_once(owner, symbol):
            return False
        value = statement.value
        if not _module_import_expression(value, imported):
            if symbol not in callables:
                return False
            while isinstance(value, ast.Attribute):
                value = value.value
            if not isinstance(value, ast.Name) or value.id == symbol or not resolve(value.id):
                return False
        bindings[symbol] = ast.unparse(statement)
        return True

    for symbol in sorted(captured):
        resolve(symbol)
    return bindings


def _assigned_once(owner: ast.AST, name: str) -> bool:
    """Require one lexical assignment before interpreting a captured native binding."""
    return sum(isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store)
               for node in ast.walk(owner)) == 1


def _module_import_expression(value: ast.expr, imported: dict[str, str]) -> bool:
    """Recognize only a literal importlib module load, never an executor constructor."""
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute):
        return False
    function = value.func
    if not isinstance(function.value, ast.Name) or function.attr != "import_module":
        return False
    if imported.get(function.value.id) != "importlib" or value.keywords or len(value.args) != 1:
        return False
    return isinstance(value.args[0], ast.Constant) and isinstance(value.args[0].value, str)


def _wrapper_definition(node: ast.FunctionDef, body: str, bound: bool) -> str:
    """Reuse the native wrapper signature instead of guessing its variadic arguments."""
    node = deepcopy(node)
    node.name = "forward"
    node.decorator_list = []
    _strip_annotations(node)
    if not bound:
        parameters = node.args.posonlyargs if node.args.posonlyargs else node.args.args
        parameters.insert(0, ast.arg(arg="self"))
    if node.args.vararg:
        body = body.replace("__hp_forward_positional__", node.args.vararg.arg)
    if node.args.kwarg:
        body = body.replace("__hp_forward_keywords__", node.args.kwarg.arg)
    node.body = ast.parse(body).body
    return ast.unparse(ast.fix_missing_locations(node))


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


def _captured_forward_names(tree: ast.AST, name: str) -> set[str]:
    """Read lexical captures without importing or executing the wrapper."""
    pending = list(symtable.symtable(ast.unparse(tree), "<codegen wrapper>", "exec").get_children())
    while pending:
        scope = pending.pop()
        if scope.get_name() == name and scope.get_type() == "function":
            return set(scope.get_frees())
        pending.extend(scope.get_children())
    return set()


def _uses_native_cp_argument(tree: ast.FunctionDef) -> bool:
    """Do not reinterpret an unrelated local or reassigned mesh as native CP."""
    return (
        "cp_mesh" in {item.arg for item in (*tree.args.posonlyargs, *tree.args.args, *tree.args.kwonlyargs)}
        and not any(isinstance(item, ast.Name) and item.id == "cp_mesh" and isinstance(item.ctx, ast.Store)
                    for item in ast.walk(tree))
    )


def _nested_forward_body(
    tree: ast.AST, replacement_name: str | None, *, target_name: str | None = None,
    literals: dict[str, Any] | None = None,
    bound_globals: Collection[str] = (),
) -> str | None:
    """Lower a closure-backed rewrite into an imported-host dispatch body.

    Keep the native control flow, including packed and exception branches.
    Captured published state becomes instance attributes; the saved forward
    calls the preserved source method. CP dispatch belongs to the helper path,
    not to every closure (some wrappers run at CP1 or TP-only).
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
    captured = _captured_forward_names(tree, replacement_name)
    runtime_cp = _uses_native_cp_argument(tree)
    variadic = {}
    if forward.args.vararg:
        variadic[forward.args.vararg.arg] = "__hp_forward_positional__"
    if forward.args.kwarg:
        variadic[forward.args.kwarg.arg] = "__hp_forward_keywords__"

    class ClosureBindings(ast.NodeTransformer):
        """Rewrite captured closure bindings to generated-instance state."""

        def visit_Name(self, node: ast.Name) -> ast.AST:  # pylint: disable=invalid-name
            """Rewrite a referenced symbol into its resolved source form."""
            if node.id in variadic and isinstance(node.ctx, ast.Load):
                return ast.copy_location(ast.Name(id=variadic[node.id], ctx=node.ctx), node)
            if node.id == target_name:
                return ast.copy_location(ast.Name(id="self", ctx=node.ctx), node)
            if literals is not None and node.id in captured and node.id in literals:
                return ast.copy_location(ast.parse(repr(literals[node.id]), mode="eval").body, node)
            if runtime_cp and node.id == "cp_mesh" and node.id in captured and node.id not in companion_attrs:
                return ast.copy_location(ast.parse(
                    "_get_cp_submesh(self._hp_boundary.compiled.mesh, "
                    "self._hp_boundary.compiled.mesh_dim_names)", mode="eval",
                ).body, node)
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
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr="_forward_impl",
                    ctx=ast.Load(),
                )
            return node

    body = forward.body[1:] if ast.get_docstring(forward) is not None else forward.body
    lowered = [ClosureBindings().visit(ast.fix_missing_locations(node)) for node in body]
    if not lowered:
        return None
    _require_bound_captures(captured - set(bound_globals), lowered, replacement_name)
    return "\n".join(ast.unparse(node) for node in lowered)


def _require_bound_captures(captured: set[str], statements: list[ast.stmt], name: str) -> None:
    """Reject emitted closure state without a source-visible binding."""
    unresolved = sorted(captured & {
        node.id for statement in statements for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    })
    if unresolved:
        raise RuntimeError(
            f"codegen: cannot inline {name!r}; unresolved captured state {unresolved}"
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
