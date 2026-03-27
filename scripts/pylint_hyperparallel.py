#!/usr/bin/env python3
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
"""Project-specific pylint checks aligned with .claude/rules/code-style.md."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

from astroid import nodes
from pylint.checkers import BaseChecker

APACHE_HEADER_SNIPPETS: Tuple[str, ...] = (
    "Copyright 2026 Huawei Technologies Co., Ltd",
    "Licensed under the Apache License, Version 2.0",
    "limitations under the License.",
)
BACKEND_IMPORTS = {"torch", "mindspore"}
PLATFORM_ALLOWED_PARTS = ("hyper_parallel/platform/", "tests/", "scripts/", ".claude/")
PUBLIC_MAGIC_ALLOWLIST = {
    "__init__", "__call__", "__enter__", "__exit__", "__iter__", "__next__",
    "__len__", "__getitem__", "__setitem__", "__delitem__", "__contains__",
    "__repr__", "__str__", "__bool__", "__eq__", "__ne__", "__lt__", "__le__",
    "__gt__", "__ge__", "__hash__",
}
DOCSTRING_SECTIONS = ("Args:", "Returns:", "Raises:", "Example:", "Note:")


def _normalize_path(path: str) -> str:
    """Normalize a filesystem path for prefix checks."""
    return path.replace("\\", "/")


def _iter_function_arguments(node: nodes.FunctionDef) -> Iterable[nodes.AssignName]:
    """Yield explicit function arguments that should carry type hints."""
    arguments = node.args
    positional = list(arguments.posonlyargs or []) + list(arguments.args or [])
    yield from positional
    yield from arguments.kwonlyargs or []
    if arguments.vararg:
        yield arguments.vararg
    if arguments.kwarg:
        yield arguments.kwarg


def _has_complete_type_hints(node: nodes.FunctionDef) -> bool:
    """Check whether a public function declares all expected annotations."""
    arguments = node.args
    positional = list(arguments.posonlyargs or []) + list(arguments.args or [])
    positional_annotations = list(arguments.posonlyargs_annotations or []) + list(arguments.annotations or [])

    for arg, annotation in zip(positional, positional_annotations):
        if arg.name in {"self", "cls"}:
            continue
        if annotation is None:
            return False

    for arg, annotation in zip(arguments.kwonlyargs or [], arguments.kwonlyargs_annotations or []):
        if arg.name in {"self", "cls"}:
            continue
        if annotation is None:
            return False

    if arguments.vararg and arguments.varargannotation is None:
        return False
    if arguments.kwarg and arguments.kwargannotation is None:
        return False

    return node.returns is not None


def _is_public_function(node: nodes.FunctionDef) -> bool:
    """Check whether a function or method is considered public."""
    name = node.name
    if name.startswith("_") and name not in PUBLIC_MAGIC_ALLOWLIST:
        return False
    if name.startswith("test_"):
        return False
    return True


def _is_platform_agnostic_module(path: str) -> bool:
    """Check whether backend imports should be blocked for this file."""
    normalized = _normalize_path(path)
    if not normalized.endswith(".py"):
        return False
    return not any(part in normalized for part in PLATFORM_ALLOWED_PARTS)


class HyperParallelChecker(BaseChecker):
    """Custom checks for HyperParallel project style rules."""

    name = "hyperparallel-style"
    priority = -1
    msgs = {
        "C9001": (
            "Python files must start with the Apache 2.0 license header",
            "missing-apache-license-header",
            "Used when a Python file is missing the required Apache 2.0 header.",
        ),
        "C9002": (
            "Direct backend import '%s' is forbidden in platform-agnostic code",
            "forbidden-backend-import",
            "Used when torch or mindspore is imported outside approved platform-specific modules.",
        ),
        "C9003": (
            "Avoid assigning platform on instances; use module-level platform = get_platform()",
            "instance-platform-assignment",
            "Used when code stores platform on self.platform instead of module scope.",
        ),
        "C9004": (
            "Avoid mutating sys.path with insert(0, ...)",
            "sys-path-mutation",
            "Used when code mutates sys.path in a discouraged way.",
        ),
        "W9005": (
            "subprocess call should pass arguments as a list and keep shell=False",
            "unsafe-subprocess-call",
            "Used when subprocess APIs are called with shell=True or string commands.",
        ),
        "C9006": (
            "Public function '%s' must declare type hints for all parameters and the return value",
            "missing-public-type-hints",
            "Used when public APIs are missing parameter or return annotations.",
        ),
        "C9007": (
            "Public function '%s' must have a docstring",
            "missing-public-docstring",
            "Used when public APIs are missing docstrings.",
        ),
        "C9008": (
            "Google-style docstring for '%s' should use section markers such as %s when applicable",
            "non-google-docstring",
            "Used when docstrings use unsupported section headings.",
        ),
    }

    def open(self) -> None:
        """Reset per-module state before processing."""
        self._module_path = ""

    def visit_module(self, node: nodes.Module) -> None:
        """Check file-level style requirements.

        Args:
            node: The module AST node being visited.
        """
        path = node.file or ""
        self._module_path = path
        if not path.endswith(".py"):
            return
        self._check_apache_header(node, path)

    def visit_import(self, node: nodes.Import) -> None:
        """Block direct backend imports in platform-agnostic modules.

        Args:
            node: The import AST node being visited.
        """
        if not _is_platform_agnostic_module(self._module_path):
            return
        for name, _alias in node.names:
            root_name = name.split(".", maxsplit=1)[0]
            if root_name in BACKEND_IMPORTS:
                self.add_message("forbidden-backend-import", node=node, args=(name,))

    def visit_importfrom(self, node: nodes.ImportFrom) -> None:
        """Block direct backend imports in platform-agnostic modules.

        Args:
            node: The import-from AST node being visited.
        """
        if not _is_platform_agnostic_module(self._module_path):
            return
        module_name = node.modname or ""
        root_name = module_name.split(".", maxsplit=1)[0]
        if root_name in BACKEND_IMPORTS:
            self.add_message("forbidden-backend-import", node=node, args=(module_name,))

    def visit_assign(self, node: nodes.Assign) -> None:
        """Detect discouraged instance-level platform storage.

        Args:
            node: The assignment AST node being visited.
        """
        for target in node.targets:
            if isinstance(target, nodes.AssignAttr) and target.attrname == "platform":
                if isinstance(target.expr, nodes.Name) and target.expr.name == "self":
                    self.add_message("instance-platform-assignment", node=node)

    def visit_call(self, node: nodes.Call) -> None:
        """Check subprocess and sys.path patterns.

        Args:
            node: The call AST node being visited.
        """
        self._check_sys_path_mutation(node)
        self._check_subprocess_usage(node)

    def visit_functiondef(self, node: nodes.FunctionDef) -> None:
        """Check public function typing and docstring requirements.

        Args:
            node: The function definition AST node being visited.
        """
        self._check_public_function(node)

    visit_asyncfunctiondef = visit_functiondef

    def _check_apache_header(self, node: nodes.Module, path: str) -> None:
        """Verify that Python files start with the required Apache header."""
        file_path = Path(path)
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            return
        header_window = "\n".join(content.splitlines()[:16])
        if all(snippet in header_window for snippet in APACHE_HEADER_SNIPPETS):
            return
        self.add_message("missing-apache-license-header", node=node)

    def _check_sys_path_mutation(self, node: nodes.Call) -> None:
        """Flag discouraged sys.path.insert(0, ...) mutations."""
        func = node.func
        if not isinstance(func, nodes.Attribute) or func.attrname != "insert":
            return
        expr = func.expr
        if not isinstance(expr, nodes.Attribute) or expr.attrname != "path":
            return
        base = expr.expr
        if not isinstance(base, nodes.Name) or base.name != "sys":
            return
        if not node.args:
            return
        first_arg = node.args[0]
        if isinstance(first_arg, nodes.Const) and first_arg.value == 0:
            self.add_message("sys-path-mutation", node=node)

    def _check_subprocess_usage(self, node: nodes.Call) -> None:
        """Enforce safer subprocess invocation defaults."""
        func = node.func
        if not isinstance(func, nodes.Attribute):
            return
        if not isinstance(func.expr, nodes.Name) or func.expr.name != "subprocess":
            return
        if func.attrname not in {"run", "Popen", "call", "check_call", "check_output"}:
            return

        shell_value = self._keyword_bool_value(node, "shell")
        if shell_value is True:
            self.add_message("unsafe-subprocess-call", node=node)
            return

        if node.args and isinstance(node.args[0], nodes.Const) and isinstance(node.args[0].value, str):
            self.add_message("unsafe-subprocess-call", node=node)

    def _check_public_function(self, node: nodes.FunctionDef) -> None:
        """Validate public API docstring and type hint expectations."""
        if not _is_public_function(node):
            return

        if not _has_complete_type_hints(node):
            self.add_message("missing-public-type-hints", node=node, args=(node.name,))

        doc_node = node.doc_node
        if doc_node is None or not getattr(doc_node, "value", "").strip():
            self.add_message("missing-public-docstring", node=node, args=(node.name,))
            return

        docstring = doc_node.value
        if any(marker in docstring for marker in ("Arguments:", "Return:", "Examples:", "Parameters:")):
            markers = ", ".join(DOCSTRING_SECTIONS)
            self.add_message("non-google-docstring", node=node, args=(node.name, markers))

    @staticmethod
    def _keyword_bool_value(node: nodes.Call, keyword_name: str) -> Optional[bool]:
        """Resolve a literal bool keyword argument when available."""
        for keyword in node.keywords:
            if keyword.arg != keyword_name:
                continue
            if isinstance(keyword.value, nodes.Const) and isinstance(keyword.value.value, bool):
                return keyword.value.value
        return None


def register(linter: Any) -> None:
    """Register the checker with pylint.

    Args:
        linter: The pylint linter instance to register the checker with.
    """
    linter.register_checker(HyperParallelChecker(linter))
