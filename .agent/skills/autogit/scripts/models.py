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
"""Data models, constants, and shared error types for AutoGit."""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Constants
# ============================================================================

GITCODE_API_BASE = "https://api.gitcode.com/api/v5"
CREDENTIAL_HOST = "api.gitcode.com"

COMMIT_TYPE_LABELS = {
    'feat': '新增功能', 'fix': '问题修复', 'refactor': '代码重构',
    'perf': '性能优化', 'docs': '文档更新', 'test': '测试用例',
    'chore': '工程维护', 'ci': 'CI/CD', 'style': '代码风格',
    'build': '构建系统',
}

CONVENTIONAL_RE = re.compile(
    r'^(' + '|'.join(COMMIT_TYPE_LABELS) + r')'
    r'(?:\(([^)]*)\))?'
    r'!?\s*:\s*(.+)$'
)


# ============================================================================
# Error
# ============================================================================

class AutoGitError(Exception):
    """Base error class for AutoGit operations."""


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass(frozen=True)
class EnvConfig:
    """Validated environment configuration for the fork workflow.

    Args:
        token: GitCode API token (may be None when not required).
        origin_owner: Owner of the origin (fork) remote.
        origin_repo: Repository name of the origin remote.
        upstream_owner: Owner of the upstream (main) remote.
        upstream_repo: Repository name of the upstream remote.
        default_branch: Default branch of the upstream remote.
    """
    token: Optional[str]
    origin_owner: str
    origin_repo: str
    upstream_owner: str
    upstream_repo: str
    default_branch: str


@dataclass
class FileChanges:
    """Parsed changes for a single file from diff.

    Args:
        added_funcs: List of added function dicts with 'name' and 'params'.
        removed_funcs: List of removed function names.
        added_classes: List of added class dicts with 'name' and 'base'.
        removed_classes: List of removed class names.
        added_imports: List of added import statements.
        removed_imports: List of removed import statements.
        added_constants: List of added constant definitions.
        key_changes: List of human-readable key change descriptions.
    """
    added_funcs: List[Dict[str, str]] = field(default_factory=list)
    removed_funcs: List[str] = field(default_factory=list)
    added_classes: List[Dict[str, str]] = field(default_factory=list)
    removed_classes: List[str] = field(default_factory=list)
    added_imports: List[str] = field(default_factory=list)
    removed_imports: List[str] = field(default_factory=list)
    added_constants: List[str] = field(default_factory=list)
    key_changes: List[str] = field(default_factory=list)


@dataclass
class DiffAnalysis:
    """Complete diff analysis result.

    Args:
        files: List of changed file paths.
        file_stats: Per-file (additions, deletions) tuples.
        file_changes: Per-file FileChanges instances.
        additions: Total number of added lines.
        deletions: Total number of deleted lines.
        added_funcs: Globally aggregated added function names.
        removed_funcs: Globally aggregated removed function names.
        modified_funcs: Functions appearing in both added and removed.
        added_classes: Globally aggregated added class names.
        removed_classes: Globally aggregated removed class names.
        modified_classes: Classes appearing in both added and removed.
        added_imports: Globally aggregated added imports.
        removed_imports: Globally aggregated removed imports.
        added_constants: Globally aggregated added constants.
        modules: List of top-level module directories.
        change_summary: Per-file summaries of key changes.
    """
    files: List[str] = field(default_factory=list)
    file_stats: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    file_changes: Dict[str, FileChanges] = field(default_factory=dict)
    additions: int = 0
    deletions: int = 0
    added_funcs: List[str] = field(default_factory=list)
    removed_funcs: List[str] = field(default_factory=list)
    modified_funcs: List[str] = field(default_factory=list)
    added_classes: List[str] = field(default_factory=list)
    removed_classes: List[str] = field(default_factory=list)
    modified_classes: List[str] = field(default_factory=list)
    added_imports: List[str] = field(default_factory=list)
    removed_imports: List[str] = field(default_factory=list)
    added_constants: List[str] = field(default_factory=list)
    modules: List[str] = field(default_factory=list)
    change_summary: List[Dict[str, object]] = field(default_factory=list)
