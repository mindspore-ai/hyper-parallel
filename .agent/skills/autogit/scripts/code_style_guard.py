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
"""Lightweight code-style guard aligned with .agent/rules/code-style.md."""

import argparse
import ast
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

APACHE_LICENSE_HEADER = """# Copyright 2026 Huawei Technologies Co., Ltd
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
"""

TEXT_SUFFIXES = {
    ".py", ".sh", ".bash", ".md", ".yaml", ".yml", ".json",
    ".toml", ".cmake", ".c", ".cc", ".cpp", ".h", ".hpp",
}


def _is_text_target(path: Path) -> bool:
    """Check whether the path should be style-checked as a text file."""
    return path.name == "CMakeLists.txt" or path.suffix in TEXT_SUFFIXES


def _normalize_text(content: str) -> str:
    """Normalize trailing spaces and final newline."""
    lines = [line.rstrip() for line in content.splitlines()]
    normalized = "\n".join(lines)
    return f"{normalized}\n"


def _has_apache_header(lines: Sequence[str]) -> bool:
    """Check whether the file already starts with the required Apache header."""
    return len(lines) >= 10 and "Licensed under the Apache License, Version 2.0" in "\n".join(lines[:16])


def _ensure_python_header(content: str) -> str:
    """Insert the standard Apache header into Python files when missing."""
    lines = content.splitlines()
    if _has_apache_header(lines):
        return content

    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1

    header_lines = APACHE_LICENSE_HEADER.rstrip("\n").splitlines()
    new_lines = list(lines[:insert_at]) + header_lines + ([""] if lines[insert_at:] else []) + list(lines[insert_at:])
    return "\n".join(new_lines)


def _public_python_issues(path: Path, tree: ast.AST) -> List[str]:
    """Collect high-signal Python style violations that should block follow-up steps."""
    issues: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_") or node.name.startswith("test_"):
            continue

        is_method = False
        parent = getattr(node, "parent", None)
        if isinstance(parent, ast.ClassDef):
            is_method = True

        missing_args = []
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            if is_method and arg.arg in {"self", "cls"}:
                continue
            if arg.annotation is None:
                missing_args.append(arg.arg)
        if node.args.vararg and node.args.vararg.annotation is None:
            missing_args.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg and node.args.kwarg.annotation is None:
            missing_args.append(f"**{node.args.kwarg.arg}")
        if node.returns is None:
            issues.append(f"{path}:{node.lineno}: public function '{node.name}' is missing a return type hint")
        if missing_args:
            issues.append(
                f"{path}:{node.lineno}: public function '{node.name}' is missing type hints for: "
                + ", ".join(missing_args)
            )
        if not ast.get_docstring(node):
            issues.append(f"{path}:{node.lineno}: public function '{node.name}' is missing a docstring")

    return issues


def _attach_parents(tree: ast.AST) -> None:
    """Attach parent links to AST nodes to classify methods."""
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            setattr(child, "parent", parent)


def fix_file(path_str: str) -> Tuple[bool, List[str]]:
    """Apply automatic code-style fixes to one file and report blocking issues.

    Args:
        path_str: Path to the file to process.

    Returns:
        Tuple of (changed, issues) where changed indicates if the file was modified
        and issues is a list of blocking style violations.
    """
    path = Path(path_str)
    if not path.is_file() or not _is_text_target(path):
        return False, []

    original = path.read_text(encoding="utf-8")
    updated = _normalize_text(original)

    if path.suffix == ".py":
        updated = _ensure_python_header(updated)

    changed = updated != original
    if changed:
        path.write_text(updated, encoding="utf-8")

    issues: List[str] = []
    if path.suffix == ".py":
        try:
            tree = ast.parse(updated, filename=str(path))
            _attach_parents(tree)
        except SyntaxError as exc:
            issues.append(f"{path}: syntax error prevents code-style validation: {exc.msg}")
        else:
            issues.extend(_public_python_issues(path, tree))

    return changed, issues


def check_files(files: Iterable[str]) -> Tuple[bool, str]:
    """Fix files in place and return a combined report.

    Args:
        files: Iterable of file paths to check and fix.

    Returns:
        Tuple of (success, report) where success is True if no blocking issues
        and report contains the combined output of fixes and issues.
    """
    fixed_files: List[str] = []
    issues: List[str] = []

    for path in files:
        changed, file_issues = fix_file(path)
        if changed:
            fixed_files.append(path)
        issues.extend(file_issues)

    report_parts: List[str] = []
    if fixed_files:
        report_parts.append("[code_style_auto_fix]\n  " + "\n  ".join(fixed_files))
    if issues:
        report_parts.append("[code_style]\n  " + "\n  ".join(issues))

    if not report_parts:
        return True, ""
    return not issues, "\n".join(report_parts) + "\n"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Apply lightweight code-style fixes.")
    parser.add_argument("--fix", action="store_true", help="Apply fixes in place.")
    parser.add_argument("files", nargs="+", help="Files to validate.")
    return parser.parse_args()


def main() -> int:
    """Run the code-style guard."""
    args = parse_args()
    ok, report = check_files(args.files)
    if report:
        print(report, end="")
    if args.fix:
        return 0 if ok else 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
