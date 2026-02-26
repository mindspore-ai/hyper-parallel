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
"""Pure diff parsing and structural analysis for AutoGit."""

import re
from typing import List

from models import DiffAnalysis, FileChanges


# ============================================================================
# Line-level parsing
# ============================================================================

def _parse_added_lines(lines: List[str], changes: FileChanges) -> None:
    """Parse added lines to extract functions, classes, imports, and constants.

    Args:
        lines: Added source lines from the diff.
        changes: FileChanges instance to populate.
    """
    for line in lines:
        m = re.match(r'\s*(def|async def)\s+(\w+)\s*\(([^)]*)\)', line)
        if m:
            changes.added_funcs.append(
                {'name': m.group(2), 'params': m.group(3).strip()}
            )
            continue

        m = re.match(r'\s*class\s+(\w+)\s*(?:\(([^)]*)\))?:', line)
        if m:
            changes.added_classes.append(
                {'name': m.group(1), 'base': m.group(2) or ''}
            )
            continue

        m = re.match(r'\s*(from\s+\S+\s+import\s+.+|import\s+.+)', line)
        if m:
            changes.added_imports.append(m.group(1).strip())
            continue

        m = re.match(r'\s*([A-Z][A-Z0-9_]+)\s*=\s*(.+)', line)
        if m:
            changes.added_constants.append(f"{m.group(1)} = {m.group(2)[:50]}")


def _parse_removed_lines(lines: List[str], changes: FileChanges) -> None:
    """Parse removed lines to extract functions, classes, and imports.

    Args:
        lines: Removed source lines from the diff.
        changes: FileChanges instance to populate.
    """
    for line in lines:
        m = re.match(r'\s*(def|async def)\s+(\w+)', line)
        if m:
            changes.removed_funcs.append(m.group(2))
            continue

        m = re.match(r'\s*class\s+(\w+)', line)
        if m:
            changes.removed_classes.append(m.group(1))
            continue

        m = re.match(r'\s*(from\s+\S+\s+import\s+.+|import\s+.+)', line)
        if m:
            changes.removed_imports.append(m.group(1).strip())


def _build_key_changes(changes: FileChanges) -> None:
    """Generate key change descriptions from change information.

    Args:
        changes: FileChanges instance to update with key_changes.
    """
    for cls in changes.added_classes:
        base_info = f"(inherits {cls['base']})" if cls['base'] else ""
        changes.key_changes.append(f"Add class `{cls['name']}` {base_info}")
    for func in changes.added_funcs:
        changes.key_changes.append(
            f"Add method `{func['name']}({func['params'][:30]})`"
        )
    for cls in changes.removed_classes:
        changes.key_changes.append(f"Remove class `{cls}`")
    for func in changes.removed_funcs:
        changes.key_changes.append(f"Remove method `{func}`")


# ============================================================================
# File-level extraction
# ============================================================================

def extract_file_changes(added: List[str], removed: List[str]) -> FileChanges:
    """Extract detailed changes for a single file.

    Args:
        added: Added lines from the diff for this file.
        removed: Removed lines from the diff for this file.

    Returns:
        Populated FileChanges instance.
    """
    changes = FileChanges()
    _parse_added_lines(added, changes)
    _parse_removed_lines(removed, changes)
    _build_key_changes(changes)
    return changes


def _merge_file_changes(result: DiffAnalysis, current_file: str,
                        file_adds: int, file_dels: int,
                        file_added_lines: List[str],
                        file_removed_lines: List[str]) -> None:
    """Merge a single file's changes into the global DiffAnalysis result.

    Args:
        result: DiffAnalysis accumulator to update.
        current_file: Path of the current file.
        file_adds: Number of added lines in this file.
        file_dels: Number of deleted lines in this file.
        file_added_lines: Added source lines.
        file_removed_lines: Removed source lines.
    """
    result.file_stats[current_file] = (file_adds, file_dels)
    changes = extract_file_changes(file_added_lines, file_removed_lines)
    result.file_changes[current_file] = changes

    for func in changes.added_funcs:
        result.added_funcs.append(func['name'])
    result.removed_funcs.extend(changes.removed_funcs)
    for cls in changes.added_classes:
        result.added_classes.append(cls['name'])
    result.removed_classes.extend(changes.removed_classes)
    result.added_imports.extend(changes.added_imports)
    result.removed_imports.extend(changes.removed_imports)
    result.added_constants.extend(changes.added_constants)


def _classify_modified(result: DiffAnalysis) -> None:
    """Identify modified functions and classes (appearing in both added and removed).

    Args:
        result: DiffAnalysis instance to update in place.
    """
    added_set = set(result.added_funcs)
    removed_set = set(result.removed_funcs)
    result.modified_funcs = list(added_set & removed_set)
    result.added_funcs = list(added_set - removed_set)
    result.removed_funcs = list(removed_set - added_set)

    added_cls_set = set(result.added_classes)
    removed_cls_set = set(result.removed_classes)
    result.modified_classes = list(added_cls_set & removed_cls_set)
    result.added_classes = list(added_cls_set - removed_cls_set)
    result.removed_classes = list(removed_cls_set - added_cls_set)


# ============================================================================
# Main analysis entry point
# ============================================================================

def analyze_diff(diff_content: str) -> DiffAnalysis:
    """Deeply analyze diff content and extract detailed code change information.

    Args:
        diff_content: Raw unified diff text.

    Returns:
        Populated DiffAnalysis dataclass.
    """
    result = DiffAnalysis()
    modules: set = set()

    current_file = None
    file_adds = file_dels = 0
    file_added_lines: List[str] = []
    file_removed_lines: List[str] = []

    for line in diff_content.split("\n"):
        if line.startswith("diff --git"):
            if current_file:
                _merge_file_changes(
                    result, current_file, file_adds, file_dels,
                    file_added_lines, file_removed_lines
                )

            parts = line.split(" ")
            current_file = parts[3][2:] if len(parts) >= 4 else None
            if current_file:
                result.files.append(current_file)
                path_parts = current_file.split('/')
                if len(path_parts) > 1:
                    modules.add(path_parts[0])

            file_adds = file_dels = 0
            file_added_lines = []
            file_removed_lines = []

        elif line.startswith("+") and not line.startswith("+++"):
            result.additions += 1
            file_adds += 1
            file_added_lines.append(line[1:])

        elif line.startswith("-") and not line.startswith("---"):
            result.deletions += 1
            file_dels += 1
            file_removed_lines.append(line[1:])

    if current_file:
        _merge_file_changes(
            result, current_file, file_adds, file_dels,
            file_added_lines, file_removed_lines
        )

    _classify_modified(result)
    result.modules = list(modules)

    for filename, changes in result.file_changes.items():
        if changes.key_changes:
            result.change_summary.append({
                'file': filename.split('/')[-1],
                'full_path': filename,
                'changes': changes.key_changes[:5]
            })

    return result
