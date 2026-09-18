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
"""Expand real orchestration source, stopping at framework and kernel APIs."""

import ast
from dataclasses import dataclass
import inspect
import textwrap
from types import ModuleType
from typing import Any, Callable


_PRIMITIVE_PREFIXES = (
    "hyper_parallel.platform",
    "hyper_parallel.core",
    "hyper_parallel.collectives",
    "hyper_parallel.components.functional",
    "hyper_parallel.components.modules",
    "hyper_parallel.distributed.expert_parallel.experts",
    "hyper_parallel.distributed.context_parallel.collectives",
    "hyper_parallel.codegen.runtime",
)


@dataclass(frozen=True)
class ExpandedSource:
    """Source and imports required by one expanded dependency closure."""

    imports: tuple[str, ...]
    source: str


def expand_function(function: Callable[..., Any]) -> ExpandedSource:
    """Copy an orchestration function and recursively resolve its globals.

    Constants use their source initializer, so generated cache state never
    captures live device tensors from the generation process. Unsupported
    globals fail at generation rather than becoming unresolved artifact names.
    """
    expansion = _Expansion()
    expansion.add_function(function, function.__name__)
    return ExpandedSource(tuple(sorted(expansion.imports)), "\n\n".join(expansion.definitions))


class _Expansion:
    """Resolve one closure with deterministic order and collision checking."""

    def __init__(self) -> None:
        self.imports: set[str] = set()
        self.definitions: list[str] = []
        self.bindings: dict[str, Any] = {}

    def _claim(self, name: str, value: Any) -> bool:
        if name in self.bindings:
            if self.bindings[name] is not value:
                raise ValueError(f"codegen source expansion: conflicting global {name!r}")
            return False
        self.bindings[name] = value
        return True

    def add_function(self, function: Callable[..., Any], name: str) -> None:
        """Resolve dependencies before emitting a function definition."""
        if not self._claim(name, function):
            return
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
        node = tree.body[0]
        if not isinstance(node, ast.FunctionDef) or node.decorator_list:
            raise ValueError(f"codegen source expansion: unsupported decorated function {name!r}")
        global_names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)
                        and isinstance(item.ctx, ast.Load)}
        local_names = set(function.__code__.co_varnames)
        for global_name in sorted(global_names - local_names):
            if global_name in function.__globals__:
                self.add_global(global_name, function.__globals__[global_name], inspect.getmodule(function))
        node.name = name
        self.definitions.append(ast.unparse(node))

    def add_global(self, name: str, value: Any, owner: ModuleType) -> None:
        """Emit an import, dependency function, or literal source initializer."""
        if inspect.isfunction(value) and _is_orchestration(value):
            self.add_function(value, name)
            return
        if not self._claim(name, value):
            return
        if isinstance(value, ModuleType):
            self.imports.add(f"import {value.__name__} as {name}")
            return
        module_name = getattr(value, "__module__", None)
        original_name = getattr(value, "__name__", None)
        if module_name and original_name:
            self.imports.add(f"from {module_name} import {original_name} as {name}")
            return
        initializer = _literal_initializer(owner, name)
        self.definitions.append(f"{name} = {initializer}")


def _is_orchestration(function: Callable[..., Any]) -> bool:
    module = function.__module__
    return module.startswith("hyper_parallel.") and not any(
        module == prefix or module.startswith(prefix + ".") for prefix in _PRIMITIVE_PREFIXES
    )


def _literal_initializer(owner: ModuleType, name: str) -> str:
    for node in ast.parse(inspect.getsource(owner)).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            try:
                ast.literal_eval(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"codegen source expansion: nonliteral global {name!r}") from exc
            return ast.unparse(value)
    raise ValueError(f"codegen source expansion: no literal initializer for {name!r}")
