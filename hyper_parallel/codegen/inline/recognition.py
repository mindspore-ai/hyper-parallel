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
import inspect
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
            arguments=_factory_arguments(node, parameters, factory),
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
    call: ast.Call, parameters: set[str], factory: Any
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


def archetype_for_factory(factory_path: str | None) -> Any | None:
    """Reverse-lookup a MoE archetype by its compute-factory ``_target_``.

    A YAML ``local_compute_fn._target_`` that names a known archetype factory
    resolves to that archetype: the strategy kind / target class / boundary
    sub-patterns / EP body keys are then structural, not declared per family.
    """
    if not factory_path:
        return None
    from hyper_parallel.distributed.expert_parallel.archetypes import (  # pylint: disable=C0415
        moe_archetypes,
    )

    for archetype in moe_archetypes().values():
        if archetype.compute_factory == factory_path:
            return archetype
    return None


__all__ = [
    "ReplacementArgument",
    "ReplacementTarget",
    "adapter_path_parts",
    "archetype_for_factory",
    "class_name_of",
    "family_from_target",
    "replacement_target",
]
