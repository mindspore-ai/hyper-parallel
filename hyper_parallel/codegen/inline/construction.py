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
"""Express already selected native members at source construction sites."""

from __future__ import annotations

import ast
from importlib import import_module
import inspect
import textwrap
from typing import Any

from hyper_parallel.codegen.inline.ir import ConstructorPatch, ImportPatch, InlinePatchSet
from hyper_parallel.codegen.inline.patch_engine import apply_patch_set


def _symbol(path: str) -> Any:
    module, name = path.split(":", 1)
    value = import_module(module)
    for part in name.split("."):
        value = getattr(value, part)
    return value


def _value_expression(value: Any, imports: list[ImportPatch]) -> str:
    if not isinstance(value, dict):
        return repr(value)
    if len(value) != 1:
        raise ValueError("Codegen construction requires a tagged argument value")
    kind, data = next(iter(value.items()))
    if kind == "mapping":
        return "{" + ", ".join(f"{key!r}: {_value_expression(item, imports)}" for key, item in data.items()) + "}"
    if kind in ("tuple", "list"):
        items = ", ".join(_value_expression(item, imports) for item in data)
        return f"({items},)" if kind == "tuple" and data else "()" if kind == "tuple" else f"[{items}]"
    if kind == "symbol":
        module, name = data.split(":", 1)
        root = name.split(".", 1)[0]
        imports.append(ImportPatch(module, (root,)))
        return name
    if kind == "partial":
        target, args, kwargs = data
        imports.append(ImportPatch("functools", ("partial",)))
        function = _value_expression({"symbol": target}, imports)
        return f"partial({function}, *{_value_expression(args, imports)}, **{_value_expression(kwargs, imports)})"
    raise ValueError(f"Codegen construction cannot express argument kind {kind!r}")


def replacement_constructor(record: dict) -> str:
    """Read the observed concrete class, never a family resolver or replacement factory."""
    selection = record.get("family_request", {}).get("selection", {})
    return selection.get("member") or record["replacement_type"]


def construction_sites(source: str, record: dict) -> list[dict]:
    """Use captured owners, or resolve an unambiguous standalone source fixture."""
    sites = record.get("construction_sites")
    if sites is not None:
        return sites
    attribute = record["primary_fqn"].rsplit(".", 1)[-1]
    hosts = []
    for host in ast.parse(source).body:
        if isinstance(host, ast.ClassDef) and any(
            isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name) and node.value.id == "self" and node.attr == attribute
            for method in host.body if isinstance(method, ast.FunctionDef) and method.name == "__init__"
            for node in ast.walk(method)
        ):
            hosts.append(host.name)
    if len(hosts) != 1:
        raise ValueError(f"Codegen cannot locate source construction for {record['primary_fqn']!r}")
    return [{"owner": hosts[0], "attribute": attribute}]


def materialize_module_constructions(source: str, records: list[dict]) -> str:
    """Extend the existing inline edit pass with ordinary concrete constructor calls."""
    if not records:
        return source
    patches = InlinePatchSet()
    groups = {}
    for record in records:
        for site in construction_sites(source, record):
            key = (site["owner"].split(":")[-1], site["attribute"])
            groups.setdefault(key, []).append((record, site))
    classes = {node.name for node in ast.parse(source).body if isinstance(node, ast.ClassDef)}
    source_aliases = {}
    for (owner, attribute), members in groups.items():
        record, _ = members[0]
        calls = [_constructor_expression(
            item, patches.imports, classes,
            _source_expression(source, item, records, patches.imports, source_aliases),
        ) for item, _ in members]
        selected = {path.rpartition(".")[0] for item, _ in members
                    for path in item.get("module_fqns", [item["primary_fqn"]])}
        expected = {path for _, site in members for path in site.get("instances", selected)}
        if selected != expected or len(set(calls)) != 1:
            expression = _per_instance_expression(source, owner, members, calls, expected)
        else:
            expression = calls[0]
        patches.constructors.append(ConstructorPatch(
            owner, attribute, record["source_type"].split(":")[-1], expression,
        ))
    return apply_patch_set(source, patches)


def layer_selector(source: str, owner: str) -> str:
    """Trace an indexed source comprehension to its existing constructor argument."""
    tree = ast.parse(source)
    host = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner), None)
    initializer = next((node for node in host.body if isinstance(node, ast.FunctionDef)
                        and node.name == "__init__"), None) if host else None
    parameters = [argument.arg for argument in initializer.args.args[1:]] if initializer else []
    selectors = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ListComp) or len(node.generators) != 1:
            continue
        generator = node.generators[0]
        call = node.elt
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == owner
                and isinstance(generator.target, ast.Name) and not generator.ifs
                and isinstance(generator.iter, ast.Call) and isinstance(generator.iter.func, ast.Name)
                and generator.iter.func.id == "range" and len(generator.iter.args) == 1):
            continue
        for parameter, argument in zip(parameters, call.args):
            if isinstance(argument, ast.Name) and argument.id == generator.target.id:
                selectors.add(parameter)
        selectors.update(keyword.arg for keyword in call.keywords if isinstance(keyword.value, ast.Name)
                         and keyword.value.id == generator.target.id and keyword.arg in parameters)
    if len(selectors) != 1:
        raise ValueError(f"Codegen partial construction for {owner} has no unambiguous source layer argument")
    return selectors.pop()


def _per_instance_expression(source: str, owner: str, members: list, calls: list[str], expected: set[str]) -> str:
    prefixes = {path.rpartition(".")[0] for path in expected}
    if len(prefixes) != 1 or not all(path.rsplit(".", 1)[-1].isdigit() for path in expected):
        raise ValueError(f"Codegen partial construction for {owner} does not identify one indexed source container")
    selector = layer_selector(source, owner)
    expression = "__source_module__"
    for (record, site), call in reversed(list(zip(members, calls))):
        indices = tuple(sorted({int(path.rpartition(".")[0].rsplit(".", 1)[-1])
                                for path in record.get("module_fqns", [record["primary_fqn"]])
                                if path.rsplit(".", 1)[-1] == site["attribute"]}))
        expression = f"({call} if {selector} in {indices!r} else {expression})"
    return expression


def _source_expression(source: str, record: dict, records: list[dict], imports: list, aliases: dict) -> str:
    """Keep original conversion type identity when its source class has no artifact edits."""
    path = record['primary_fqn']
    if any(item['primary_fqn'].startswith(path + '.') for item in records):
        return '__source_module__'
    module, name = record['source_type'].split(':', 1)
    if not name.isidentifier():
        return '__source_module__'
    copied = next((node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == name), None)
    if copied is None:
        return '__source_module__'
    try:
        original = ast.parse(textwrap.dedent(inspect.getsource(_symbol(record['source_type'])))).body[0]
    except (OSError, TypeError):
        return '__source_module__'
    if ast.dump(copied) != ast.dump(original):
        return '__source_module__'
    alias = aliases.setdefault(record['source_type'], f'_Source{name}_{len(aliases)}')
    if all(part.isidentifier() for part in module.split('.')):
        statement = f'from {module} import {name} as {alias}'
    else:
        statement = f"{alias} = __import__({module!r}, fromlist=[{name!r}]).{name}"
    imports.append(ImportPatch(module, (), raw=statement))
    return f'{alias}(__source_arguments__)'


def _constructor_expression(record: dict, imports: list[ImportPatch], classes: set[str], source: str) -> str:
    identity = replacement_constructor(record)
    module, name = identity.split(":", 1)
    if not name.isidentifier():
        raise ValueError(f"Codegen cannot express nested replacement class {identity!r}")
    member = _symbol(identity)
    signature = inspect.signature(member)
    if "module" not in signature.parameters:
        raise ValueError(f"Codegen concrete constructor {identity!r} does not accept a source module")
    if name not in classes:
        imports.append(ImportPatch(module, (name,)))
    selection = record.get("family_request", {}).get("selection", {})
    arguments = selection.get("resolved_arguments", record.get("replay", {}).get("constructor_arguments"))
    if arguments is None:
        raise ValueError(f"Codegen lacks concrete construction arguments for {record['primary_fqn']!r}")
    context = selection.get("construction_context", record.get("replay", {}).get("context", {"mapping": {}}))
    # The concrete source-conversion constructor owns parameter reuse and conversion.
    # The generated module neither selects a family nor installs a replacement.
    runtime = {"module": source, "config": "None", "context": _value_expression(context, imports)}
    kwargs = [f"{key}={value}" for key, value in runtime.items() if key in signature.parameters]
    # module_fqn is optional diagnostic context; never freeze one layer's path for all layers.
    if "module_fqn" in signature.parameters and signature.parameters["module_fqn"].default is inspect.Parameter.empty:
        raise ValueError(f"Codegen constructor {identity!r} requires an instance-specific module_fqn")
    if set(arguments) != {"mapping"}:
        raise ValueError(f"Codegen construction arguments for {record['primary_fqn']!r} must be a mapping")
    kwargs.extend(f"{key}={_value_expression(value, imports)}" for key, value in arguments["mapping"].items()
                  if key not in runtime and key != "module_fqn")
    return f"{name}({', '.join(kwargs)})"
