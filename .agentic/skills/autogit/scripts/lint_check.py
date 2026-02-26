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
"""Lint check module - runs CI gate checks on staged files.

Covers: pylint, lizard, codespell, markdownlint, cpplint, cmakelint,
clang-format, shellcheck, docstring conventions, dt_design, arg_mark,
and test_coverage.
Tools that are not installed produce warnings but do not block (graceful
degradation).
"""

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# ============================================================================
# Configuration (aligned with CI gate pipeline)
# ============================================================================

PYLINT_MAX_LINE_LENGTH = 120

PYLINT_DISABLE = ",".join([
    "C0103",  # invalid-name
    "C0114",  # missing-module-docstring
    "C0115",  # missing-class-docstring
    "C0116",  # missing-function-docstring
    "C0209",  # consider-using-f-string
    "C0302",  # too-many-lines
    "C0325",  # superfluous-parens
    "E0401",  # import-error (no local torch/mindspore deps)
    "E0611",  # no-name-in-module (cascaded from import-error)
    "R0801",  # duplicate-code (similarities across files)
    "R0901",  # too-many-ancestors
    "R0902",  # too-many-instance-attributes
    "R0903",  # too-few-public-methods
    "R0904",  # too-many-public-methods
    "R0911",  # too-many-return-statements
    "R0912",  # too-many-branches
    "R0913",  # too-many-arguments
    "R0914",  # too-many-locals
    "R0915",  # too-many-statements
    "R0916",  # too-many-boolean-expressions
    "R0917",  # too-many-positional-arguments
    "R1702",  # too-many-nested-blocks
    "W0201",  # attribute-defined-outside-init (common in mixin)
    "W0511",  # fixme (TODO is normal dev marker)
    "W0718",  # broad-exception-caught
    "W0105",  # pointless-string-statement
    "W0212",  # protected-access
    "W0221",  # arguments-differ
    "W0223",  # abstract-method
    "W0231",  # super-init-not-called
    "W0237",  # arguments-renamed
    "W0612",  # unused-variable
    "W0622",  # redefined-builtin
    "W0640",  # cell-var-from-loop
    "W1203",  # logging-fstring-interpolation
])

# Lizard thresholds (aligned with CI gate: cyclomatic_complexity=15)
LIZARD_CCN = 15
LIZARD_NLOC = 100

# Markdownlint disabled rules (aligned with gate pipeline)
MARKDOWNLINT_DISABLE = {
    "MD013": True,   # line-length - tables/URLs often exceed 80
    "MD024": True,   # no-duplicate-heading - multi-command docs repeat
    "MD036": True,   # no-emphasis-as-heading - bold sub-headings common
    "MD060": True,   # table-column-style - gate does not check
}

# Docstring check patterns
_CHINESE_RE = re.compile(r'[\u4e00-\u9fff]')
_FRAMEWORK_RE = re.compile(
    r'\btorch\b|\bpytorch\b|\bPyTorch\b', re.IGNORECASE
)


# ============================================================================
# Diff-aware helpers
# ============================================================================

def get_staged_diff_added_lines() -> Dict[str, Set[int]]:
    """Return the set of added line numbers per file from staged changes.

    Parses ``git diff --cached -U0`` to find which lines were added in
    each file. This allows AST-based checks to only validate functions
    that were actually added or modified, not pre-existing code.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "-U0", "--no-color"],
            capture_output=True, text=True, check=False, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return {}

    if result.returncode != 0:
        return {}

    return _parse_unified_diff_added_lines(result.stdout)


def _parse_unified_diff_added_lines(diff_text: str) -> Dict[str, Set[int]]:
    """Parse unified diff output into per-file added line number sets.

    Args:
        diff_text: Raw output from ``git diff -U0``.
    """
    result: Dict[str, Set[int]] = {}
    current_file: Optional[str] = None
    hunk_re = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@')

    for line in diff_text.split("\n"):
        if line.startswith("+++ b/"):
            current_file = line[6:]
            result.setdefault(current_file, set())
        elif line.startswith("@@ ") and current_file is not None:
            m = hunk_re.match(line)
            if m:
                start = int(m.group(1))
                count = int(m.group(2)) if m.group(2) else 1
                for i in range(start, start + count):
                    result[current_file].add(i)

    return result


def _is_node_in_changed_lines(
    node: ast.AST, changed_lines: Optional[Set[int]]
) -> bool:
    """Check if an AST node overlaps with the set of changed lines.

    Args:
        node: AST node with ``lineno`` and ``end_lineno``.
        changed_lines: Set of added line numbers. If None, treat as
            all lines changed (no filtering).
    """
    if changed_lines is None:
        return True
    start = getattr(node, "lineno", 0)
    end = getattr(node, "end_lineno", start)
    return bool(changed_lines & set(range(start, end + 1)))


# ============================================================================
# Tool detection
# ============================================================================

def check_tool_available(name: str) -> bool:
    """Check whether an external tool is installed.

    Args:
        name: Executable name to look up on PATH.
    """
    return shutil.which(name) is not None


def install_hint() -> str:
    """Return installation hints for missing tools."""
    missing = []
    hints = {
        "pylint": "pip install pylint==3.3.7",
        "lizard": "pip install lizard",
        "codespell": "pip install codespell",
        "markdownlint-cli2": "npm install -g markdownlint-cli2",
        "cpplint": "pip install cpplint",
        "cmakelint": "pip install cmakelint",
        "clang-format": "apt install clang-format  # or conda install clang-tools",
        "shellcheck": "apt install shellcheck  # or brew install shellcheck",
    }
    for tool, cmd in hints.items():
        if not check_tool_available(tool):
            missing.append(f"  {tool}: {cmd}")
    if not missing:
        return ""
    return "Missing tools (install recommended):\n" + "\n".join(missing)


# ============================================================================
# Individual checkers
# ============================================================================

def _run_cmd(cmd: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """Run an external command and capture output.

    Args:
        cmd: Command and arguments as a list.
        timeout: Maximum execution time in seconds.
    """
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )


def run_pylint(files: List[str],
               filter_file: Optional[str] = None) -> Tuple[bool, str]:
    """Run pylint on Python files.

    Args:
        files: List of .py file paths.
        filter_file: Optional project-level filter_pylint.txt path.
    """
    if not files:
        return True, ""
    if not check_tool_available("pylint"):
        return True, "[WARN] pylint not installed, skipped\n"

    cmd = [
        "pylint",
        f"--max-line-length={PYLINT_MAX_LINE_LENGTH}",
        f"--disable={PYLINT_DISABLE}",
        "--score=no",
    ] + files

    result = _run_cmd(cmd)
    output = (result.stdout + result.stderr).strip()

    if filter_file and os.path.isfile(filter_file):
        output = _apply_filter(output, filter_file)

    if not output:
        return True, ""

    has_issues = _has_pylint_issues(output)
    return not has_issues, f"[pylint]\n{output}\n"


def _has_pylint_issues(output: str) -> bool:
    """Check whether pylint output contains actual issues.

    Args:
        output: Combined stdout/stderr from pylint.
    """
    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Issue line format: filename.py:line:col: X0000: message
        if ": " in line and any(
            line.split(": ")[0].endswith(f":{c}")
            for c in "0123456789"
        ):
            return True
    return False


def _parse_filter_rules(filter_file: str) -> List[Tuple[str, str]]:
    """Parse Jenkins filter_pylint.txt into (path_suffix, msg_name) pairs.

    Args:
        filter_file: Path to the filter file.
    """
    rules = []
    with open(filter_file, "r", encoding="utf-8") as fp:
        for raw in fp:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = re.findall(r'"([^"]+)"', raw)
            if len(parts) >= 2:
                # Strip repo name prefix (e.g. "hyper-parallel/..." -> "...")
                path = parts[0]
                slash_idx = path.find("/")
                if slash_idx != -1:
                    path = path[slash_idx + 1:]
                rules.append((path, parts[1]))
    return rules


def _apply_filter(output: str, filter_file: str) -> str:
    """Filter known issues from pylint output.

    Args:
        output: Raw pylint output.
        filter_file: Path to the filter rules file.
    """
    rules = _parse_filter_rules(filter_file)
    if not rules:
        return output
    lines = []
    for line in output.split("\n"):
        matched = any(
            path_pattern in line and msg_name in line
            for path_pattern, msg_name in rules
        )
        if not matched:
            lines.append(line)
    return "\n".join(lines).strip()


def run_lizard(files: List[str]) -> Tuple[bool, str]:
    """Run lizard cyclomatic complexity check.

    Args:
        files: File paths to check.
    """
    if not files:
        return True, ""
    if not check_tool_available("lizard"):
        return True, "[WARN] lizard not installed, skipped\n"

    cmd = [
        "lizard",
        f"-T cyclomatic_complexity={LIZARD_CCN}",
        f"-T nloc={LIZARD_NLOC}",
        "-w",  # warnings only
    ] + files

    result = _run_cmd(cmd)
    output = result.stdout.strip()

    if result.returncode != 0 and output:
        return False, f"[lizard]\n{output}\n"
    return True, ""


def run_codespell(files: List[str]) -> Tuple[bool, str]:
    """Run codespell spelling check.

    Args:
        files: File paths to check.
    """
    if not files:
        return True, ""
    if not check_tool_available("codespell"):
        return True, "[WARN] codespell not installed, skipped\n"

    cmd = ["codespell"] + files
    result = _run_cmd(cmd)
    output = result.stdout.strip()

    if result.returncode != 0 and output:
        return False, f"[codespell]\n{output}\n"
    return True, ""


def run_markdownlint(files: List[str]) -> Tuple[bool, str]:
    """Run markdownlint on Markdown files.

    Args:
        files: List of .md file paths.
    """
    if not files:
        return True, ""
    if not check_tool_available("markdownlint-cli2"):
        return True, "[WARN] markdownlint-cli2 not installed, skipped\n"

    config = {"default": True}
    for rule in MARKDOWNLINT_DISABLE:
        config[rule] = False

    config_dir = tempfile.mkdtemp(prefix="autogit-mdlint-")
    config_path = os.path.join(config_dir, ".markdownlint.jsonc")
    with open(config_path, "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    cmd = ["markdownlint-cli2", "--config", config_path] + files
    result = _run_cmd(cmd)
    output = (result.stdout + result.stderr).strip()

    try:
        os.remove(config_path)
        os.rmdir(config_dir)
    except OSError:
        pass

    if result.returncode != 0 and output:
        return False, f"[markdownlint]\n{output}\n"
    return True, ""


def run_cpplint(files: List[str]) -> Tuple[bool, str]:
    """Run cpplint on C/C++ files.

    Args:
        files: List of C/C++ file paths.
    """
    if not files:
        return True, ""
    if not check_tool_available("cpplint"):
        return True, "[WARN] cpplint not installed, skipped\n"

    cmd = ["cpplint"] + files
    result = _run_cmd(cmd)
    output = result.stderr.strip()

    if result.returncode != 0 and output:
        return False, f"[cpplint]\n{output}\n"
    return True, ""


def run_cmakelint(files: List[str]) -> Tuple[bool, str]:
    """Run cmakelint on CMake files.

    Args:
        files: List of CMakeLists.txt or .cmake file paths.
    """
    if not files:
        return True, ""
    if not check_tool_available("cmakelint"):
        return True, "[WARN] cmakelint not installed, skipped\n"

    cmd = ["cmakelint"] + files
    result = _run_cmd(cmd)
    output = (result.stdout + result.stderr).strip()

    if result.returncode != 0 and output:
        return False, f"[cmakelint]\n{output}\n"
    return True, ""


def run_clang_format(files: List[str]) -> Tuple[bool, str]:
    """Run clang-format dry-run check (does not modify files).

    Args:
        files: List of C/C++ file paths.
    """
    if not files:
        return True, ""
    if not check_tool_available("clang-format"):
        return True, "[WARN] clang-format not installed, skipped\n"

    cmd = ["clang-format", "--dry-run", "--Werror"] + files
    result = _run_cmd(cmd)
    output = result.stderr.strip()

    if result.returncode != 0 and output:
        return False, f"[clang-format]\n{output}\n"
    return True, ""


def run_shellcheck(files: List[str]) -> Tuple[bool, str]:
    """Run shellcheck on shell scripts.

    Args:
        files: List of shell script paths.
    """
    if not files:
        return True, ""
    if not check_tool_available("shellcheck"):
        return True, "[WARN] shellcheck not installed, skipped\n"

    cmd = ["shellcheck"] + files
    result = _run_cmd(cmd)
    output = result.stdout.strip()

    if result.returncode != 0 and output:
        return False, f"[shellcheck]\n{output}\n"
    return True, ""


# ============================================================================
# AST-based checks (diff-aware: only check added/modified code)
# ============================================================================

def _check_single_docstring(node: ast.AST, fpath: str) -> List[str]:
    """Validate one function node's docstring against project rules.

    Args:
        node: AST function node to check.
        fpath: File path for error reporting.
    """
    docstring = ast.get_docstring(node)
    if not docstring:
        return []

    short = os.path.basename(fpath)
    loc = f"{short}:{node.lineno}"
    issues = []

    if _CHINESE_RE.search(docstring):
        issues.append(
            f"  {loc}: {node.name}() - "
            "docstring contains Chinese characters"
        )

    match = _FRAMEWORK_RE.search(docstring)
    if match:
        issues.append(
            f"  {loc}: {node.name}() - "
            f"docstring references '{match.group()}'"
        )

    params = [
        a.arg for a in node.args.args
        if a.arg not in ("self", "cls")
    ]
    if params and "Args:" not in docstring:
        issues.append(
            f"  {loc}: {node.name}() - "
            f"has {len(params)} param(s) but no Args section"
        )

    return issues


def run_docstring_check(
    files: List[str],
    changed_lines: Optional[Dict[str, Set[int]]] = None,
) -> Tuple[bool, str]:
    """Check that public function docstrings follow project conventions.

    Only validates functions whose definition overlaps with changed lines
    in the current diff. If ``changed_lines`` is None, checks all functions
    (fallback for non-git contexts).

    Args:
        files: Python file paths to check.
        changed_lines: Per-file sets of added line numbers from diff.
    """
    py_files = [f for f in files
                if f.endswith(".py")
                and "/test" not in f
                and not os.path.basename(f).startswith("test_")]
    if not py_files:
        return True, ""

    issues = []
    for fpath in py_files:
        file_changed = _resolve_changed_lines(fpath, changed_lines)
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                tree = ast.parse(fp.read(), filename=fpath)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            if not _is_node_in_changed_lines(node, file_changed):
                continue
            issues.extend(_check_single_docstring(node, fpath))

    if issues:
        report = "[docstring]\n" + "\n".join(issues) + "\n"
        return False, report
    return True, ""


def _resolve_changed_lines(
    fpath: str, changed_lines: Optional[Dict[str, Set[int]]]
) -> Optional[Set[int]]:
    """Resolve the changed line set for a given file path.

    Args:
        fpath: File path (may be absolute or relative).
        changed_lines: Per-file changed lines dict from diff parser.
    """
    if changed_lines is None:
        return None
    # Try exact match, then basename-based match
    if fpath in changed_lines:
        return changed_lines[fpath]
    for key, lines in changed_lines.items():
        if fpath.endswith(key) or key.endswith(fpath):
            return lines
    # File not in diff at all - no lines to check
    return frozenset()


def _check_single_test_docstring(
    node: ast.FunctionDef,
    fpath: str,
    required: FrozenSet[str],
) -> Optional[str]:
    """Check one test function for Feature/Description/Expectation fields.

    Args:
        node: AST function node for a test function.
        fpath: File path for error reporting.
        required: Set of required docstring section keywords.
    """
    docstring = ast.get_docstring(node) or ""
    doc_lower = docstring.lower()
    found = {kw for kw in required
             if re.search(rf"^\s*{kw}\s*:", doc_lower, re.MULTILINE)}
    if found == required:
        return None
    lacking = required - found
    return (
        f"  {fpath}:{node.lineno}: {node.name} missing "
        + ", ".join(sorted(lacking))
    )


def _has_arg_mark_decorator(node: ast.FunctionDef) -> bool:
    """Check whether a function node has an ``@arg_mark`` decorator.

    Args:
        node: AST function definition node to inspect.
    """
    for decorator in node.decorator_list:
        name = None
        if isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                name = decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                name = decorator.func.attr
        elif isinstance(decorator, ast.Name):
            name = decorator.id
        if name == "arg_mark":
            return True
    return False


def run_arg_mark_check(
    files: List[str],
    changed_lines: Optional[Dict[str, Set[int]]] = None,
) -> Tuple[bool, str]:
    """Check that test functions have an ``@arg_mark`` decorator.

    Only validates ``test_*`` functions whose definition overlaps with
    changed lines. If ``changed_lines`` is None, checks all test
    functions (fallback for non-git contexts).

    Args:
        files: Python file paths to check.
        changed_lines: Per-file sets of added line numbers from diff.
    """
    test_files = [
        f for f in files
        if os.path.basename(f).startswith("test_") and f.endswith(".py")
    ]
    if not test_files:
        return True, ""

    missing: List[str] = []
    for fpath in test_files:
        file_changed = _resolve_changed_lines(fpath, changed_lines)
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                tree = ast.parse(fp.read(), filename=fpath)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue
            if not _is_node_in_changed_lines(node, file_changed):
                continue
            if not _has_arg_mark_decorator(node):
                missing.append(
                    f"  {fpath}:{node.lineno}: {node.name}() "
                    "missing @arg_mark decorator"
                )

    if missing:
        report = "[arg_mark]\n" + "\n".join(missing) + "\n"
        return False, report
    return True, ""


def run_dt_design(
    files: List[str],
    changed_lines: Optional[Dict[str, Set[int]]] = None,
) -> Tuple[bool, str]:
    """Check that test functions contain Feature/Description/Expectation docs.

    Only validates test functions whose definition overlaps with changed
    lines. If ``changed_lines`` is None, checks all test functions.

    Args:
        files: Python file paths to check.
        changed_lines: Per-file sets of added line numbers from diff.
    """
    test_files = [f for f in files
                  if os.path.basename(f).startswith("test_") and f.endswith(".py")]
    if not test_files:
        return True, ""

    required = frozenset({"feature", "description", "expectation"})
    missing = []

    for fpath in test_files:
        file_changed = _resolve_changed_lines(fpath, changed_lines)
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                tree = ast.parse(fp.read(), filename=fpath)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue
            if not _is_node_in_changed_lines(node, file_changed):
                continue
            issue = _check_single_test_docstring(node, fpath, required)
            if issue:
                missing.append(issue)

    if missing:
        report = "[dt_design]\n" + "\n".join(missing) + "\n"
        return False, report
    return True, ""


def _extract_new_public_functions(
    fpath: str,
    changed_lines: Optional[Dict[str, Set[int]]],
) -> List[str]:
    """Extract public function/method names added or modified in a source file.

    Only returns names whose definition overlaps with *changed_lines*,
    so pre-existing (untouched) functions are never reported.
    Skips private (``_xxx``), dunder (``__xxx__``), and test
    (``test_xxx``) names.

    Args:
        fpath: Path to the Python source file.
        changed_lines: Per-file sets of added line numbers from diff.
    """
    try:
        with open(fpath, "r", encoding="utf-8") as fp:
            tree = ast.parse(fp.read(), filename=fpath)
    except (SyntaxError, OSError):
        return []

    file_changed = _resolve_changed_lines(fpath, changed_lines)
    # If the file is not in the diff at all, nothing to report
    if isinstance(file_changed, frozenset) and not file_changed:
        return []

    result: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = node.name
        if name.startswith("_") or name.startswith("test_"):
            continue
        if not _is_node_in_changed_lines(node, file_changed):
            continue
        result.append(name)
    return result


def _test_file_covers_function(
    test_paths: List[str],
    func_name: str,
) -> bool:
    """Check if any existing test file references the given function name.

    Args:
        test_paths: Candidate test file paths (both ``test_*`` and
            ``_test_*`` conventions).
        func_name: Public function name to search for.
    """
    for tpath in test_paths:
        if not os.path.isfile(tpath):
            continue
        try:
            with open(tpath, "r", encoding="utf-8") as fp:
                if func_name in fp.read():
                    return True
        except OSError:
            continue
    return False


def _derive_test_paths(src_path: str) -> List[str]:
    """Derive conventional test file paths for a source file.

    Args:
        src_path: Source file path, e.g. ``hyper_parallel/core/foo/api.py``.
    """
    parts = src_path.replace("\\", "/").split("/")
    # Expect: hyper_parallel/<...>/<module>/<file>.py
    if len(parts) < 3 or not src_path.endswith(".py"):
        return []
    filename = os.path.basename(src_path)
    # Find the sub-path after "hyper_parallel/"
    try:
        hp_idx = parts.index("hyper_parallel")
    except ValueError:
        return []
    sub_parts = parts[hp_idx + 1:]  # e.g. ["core", "foo", "api.py"]
    # Skip internal layers like "core"
    module_parts = [p for p in sub_parts[:-1] if p not in ("core",)]
    module_path = "/".join(module_parts) if module_parts else ""
    test_name = f"test_{filename}"
    impl_name = f"_test_{filename}"
    candidates = []
    for framework in ("torch", "mindspore"):
        if module_path:
            candidates.append(f"tests/{framework}/{module_path}/{test_name}")
            candidates.append(f"tests/{framework}/{module_path}/{impl_name}")
        else:
            candidates.append(f"tests/{framework}/{test_name}")
            candidates.append(f"tests/{framework}/{impl_name}")
    return candidates


def _check_single_file_coverage(
    fpath: str,
    changed_lines: Optional[Dict[str, Set[int]]],
    warnings: List[str],
    errors: List[str],
) -> None:
    """Check test coverage for a single source file.

    Args:
        fpath: Source file path to check.
        changed_lines: Per-file sets of added line numbers from diff.
        warnings: Accumulator for advisory warnings (mutated in place).
        errors: Accumulator for blocking errors (mutated in place).
    """
    candidates = _derive_test_paths(fpath)
    if not candidates:
        return

    existing_tests = [c for c in candidates if os.path.isfile(c)]

    if not existing_tests:
        expected = " or ".join(candidates)
        warnings.append(
            f"  {fpath}: no test file found (expected {expected})"
        )
        return

    new_funcs = _extract_new_public_functions(fpath, changed_lines)
    for func_name in new_funcs:
        if not _test_file_covers_function(existing_tests, func_name):
            errors.append(
                f"  {fpath}: {func_name}() added/modified but "
                f"not referenced in any test file"
            )


def _build_coverage_report(
    warnings: List[str], errors: List[str]
) -> Tuple[bool, str]:
    """Build the final coverage report from collected warnings and errors.

    Args:
        warnings: Advisory (non-blocking) messages.
        errors: Blocking error messages.

    Returns:
        Tuple of (passed, report_text).
    """
    parts: List[str] = []
    if warnings:
        parts.append(
            "[test_coverage] (advisory, non-blocking)\n"
            + "\n".join(warnings)
        )
    if errors:
        parts.append(
            "[test_coverage] BLOCKING — new public functions lack "
            "test coverage:\n" + "\n".join(errors)
        )

    passed = not errors
    report = "\n".join(parts) + "\n" if parts else ""
    return passed, report


def run_test_coverage_check(
    files: List[str],
    changed_lines: Optional[Dict[str, Set[int]]] = None,
) -> Tuple[bool, str]:
    """Check that new/modified public functions have test coverage.

    Behaviour:
    - **Test file missing**: advisory warning (non-blocking) — avoids
      blocking one framework's commit because the other has no tests.
    - **Test file exists but new public functions are not referenced**:
      blocking error (returns ``False``).

    Only inspects functions whose definition overlaps with the current
    diff (``changed_lines``), so pre-existing untouched code is never
    flagged.

    Args:
        files: Python file paths to check.
        changed_lines: Per-file sets of added line numbers from diff.
    """
    src_files = [
        f for f in files
        if f.endswith(".py")
        and "hyper_parallel/" in f.replace("\\", "/")
        and "/test" not in f
        and not os.path.basename(f).startswith("test_")
        and not f.endswith("__init__.py")
    ]
    if not src_files:
        return True, ""

    warnings: List[str] = []
    errors: List[str] = []

    for fpath in src_files:
        _check_single_file_coverage(fpath, changed_lines, warnings, errors)

    return _build_coverage_report(warnings, errors)


# ============================================================================
# Orchestration
# ============================================================================

def _classify_files(files: List[str]) -> dict:
    """Group files by type for dispatching to checkers.

    Args:
        files: List of all file paths to classify.
    """
    cpp_exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx")
    return {
        "py": [f for f in files if f.endswith(".py")],
        "md": [f for f in files if f.endswith(".md")],
        "cpp": [f for f in files if f.endswith(cpp_exts)],
        "cmake": [f for f in files
                   if os.path.basename(f) == "CMakeLists.txt"
                   or f.endswith(".cmake")],
        "sh": [f for f in files if f.endswith((".sh", ".bash"))],
        "all": files,
    }


def _find_pylint_filter() -> Optional[str]:
    """Find project-level pylint filter file."""
    for candidate in [
        ".jenkins/check/config/filter_pylint.txt",
        "filter_pylint.txt",
        "scripts/filter_pylint.txt",
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


def _build_check_plan(
    by_type: dict,
    filter_file: Optional[str],
    changed_lines: Optional[Dict[str, Set[int]]],
) -> list:
    """Build the check plan: [(checker, files, kwargs), ...].

    Args:
        by_type: Files grouped by type from ``_classify_files``.
        filter_file: Optional pylint filter file path.
        changed_lines: Per-file added line numbers for diff-aware checks.
    """
    return [
        (run_pylint,          by_type["py"],                  {"filter_file": filter_file}),
        (run_lizard,          by_type["py"] + by_type["cpp"], {}),
        (run_docstring_check, by_type["py"],                  {"changed_lines": changed_lines}),
        (run_dt_design,       by_type["py"],                  {"changed_lines": changed_lines}),
        (run_arg_mark_check,  by_type["py"],                  {"changed_lines": changed_lines}),
        (run_test_coverage_check, by_type["py"],              {"changed_lines": changed_lines}),
        (run_markdownlint,    by_type["md"],                  {}),
        (run_cpplint,         by_type["cpp"],                 {}),
        (run_clang_format,    by_type["cpp"],                 {}),
        (run_cmakelint,       by_type["cmake"],               {}),
        (run_shellcheck,      by_type["sh"],                  {}),
        (run_codespell,       by_type["all"],                 {}),
    ]


def run_checks(files: List[str]) -> Tuple[bool, str]:
    """Run all checks on the given file list.

    Automatically computes diff-aware line ranges from staged changes
    so that AST-based checks (docstring, dt_design) only validate
    functions that were actually added or modified.

    Args:
        files: File paths to check (typically staged files).
    """
    if not files:
        return True, "No files to check\n"

    by_type = _classify_files(files)
    filter_file = _find_pylint_filter()
    changed_lines = get_staged_diff_added_lines()
    plan = _build_check_plan(by_type, filter_file, changed_lines or None)

    all_passed = True
    report_parts = []

    for checker, target_files, kwargs in plan:
        passed, msg = checker(target_files, **kwargs)
        if not passed:
            all_passed = False
        if msg:
            report_parts.append(msg)

    report = "\n".join(report_parts)

    if all_passed:
        if report:
            report = f"✅ All checks passed (with warnings)\n\n{report}"
        else:
            report = "✅ All checks passed\n"
    else:
        report = f"❌ Checks failed\n\n{report}"

    hint = install_hint()
    if hint:
        report += f"\n{hint}\n"

    return all_passed, report
