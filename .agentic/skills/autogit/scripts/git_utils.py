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
"""Git subprocess operations for AutoGit."""

import re
import subprocess
from datetime import datetime
from typing import List, Optional, Tuple


# ============================================================================
# Core git helpers
# ============================================================================

def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Execute a git command as a subprocess.

    Args:
        args: Git sub-command and arguments.
        check: Whether to raise on non-zero exit code.

    Returns:
        Completed process result.
    """
    return subprocess.run(
        ["git"] + list(args),
        capture_output=True,
        text=True,
        check=check
    )


def get_remote_url(remote: str) -> Optional[str]:
    """Get the URL for a named remote.

    Args:
        remote: Remote name (e.g. 'origin', 'upstream').

    Returns:
        Remote URL string, or None if the remote does not exist.
    """
    try:
        return run_git("remote", "get-url", remote).stdout.strip()
    except subprocess.CalledProcessError:
        return None


def parse_gitcode_url(url: str) -> Optional[Tuple[str, str]]:
    """Parse a GitCode URL and return (owner, repo).

    Args:
        url: Git remote URL (SSH or HTTPS).

    Returns:
        Tuple of (owner, repo) or None if the URL is not a GitCode URL.
    """
    patterns = [
        r"git@gitcode\.com:([^/]+)/([^/]+?)(?:\.git)?$",
        r"https?://gitcode\.com/([^/]+)/([^/]+?)(?:\.git)?$",
    ]
    for pattern in patterns:
        m = re.match(pattern, url)
        if m:
            return m.group(1), m.group(2)
    return None


def get_upstream_default_branch() -> Optional[str]:
    """Detect the default branch of the upstream remote.

    Tries symbolic-ref first, then ``remote show``, then common names.

    Returns:
        Default branch name, or None if it cannot be determined.
    """
    result = run_git("symbolic-ref", "refs/remotes/upstream/HEAD", check=False)
    if result.returncode == 0:
        return result.stdout.strip().split("/")[-1]

    result = run_git("remote", "show", "upstream", check=False)
    if result.returncode == 0:
        for line in result.stdout.split("\n"):
            if "HEAD branch:" in line:
                return line.split(":")[-1].strip()

    for branch in ["main", "master", "develop"]:
        if run_git("rev-parse", f"upstream/{branch}", check=False).returncode == 0:
            return branch
    return None


def get_current_branch() -> str:
    """Get the current branch name.

    Returns:
        Current branch name string.
    """
    return run_git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def has_uncommitted_changes() -> bool:
    """Check whether there are uncommitted changes in the working tree.

    Returns:
        True if there are uncommitted changes.
    """
    return bool(run_git("status", "--porcelain").stdout.strip())


def has_staged_changes() -> bool:
    """Check whether there are staged changes in the index.

    Returns:
        True if there are staged changes.
    """
    return bool(run_git("diff", "--cached", "--name-only").stdout.strip())


def stage_all_changes() -> None:
    """Stage all modifications and deletions of tracked files (git add -u)."""
    run_git("add", "-u")


def is_protected_branch(branch: str) -> bool:
    """Check whether a branch is protected and should not be used directly for PRs.

    Args:
        branch: Branch name to check.

    Returns:
        True if the branch is protected.
    """
    protected = ['master', 'main', 'develop', 'release', 'hotfix']
    return branch in protected or branch.startswith('release/') or branch.startswith('hotfix/')


def get_branch_tracking_info(branch: str) -> Optional[str]:
    """Get the remote tracking reference for a branch.

    Args:
        branch: Local branch name.

    Returns:
        Upstream tracking ref, or None if not set.
    """
    result = run_git("rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}", check=False)
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def branch_exists_on_remote(remote: str, branch: str) -> bool:
    """Check whether a branch exists on the specified remote.

    Args:
        remote: Remote name (e.g. 'origin').
        branch: Branch name to look up.

    Returns:
        True if the branch exists on the remote.
    """
    result = run_git("ls-remote", "--heads", remote, branch, check=False)
    return bool(result.stdout.strip())


def parse_pr_ref(ref: str) -> Optional[Tuple[str, str, int]]:
    """Parse a PR reference and return (owner, repo, number).

    Supports full GitCode URLs, owner/repo#N, and bare #N (resolved via upstream).

    Args:
        ref: PR reference string.

    Returns:
        Tuple of (owner, repo, number) or None if unparsable.
    """
    m = re.match(
        r"https?://gitcode\.com/([^/]+)/([^/]+)/(?:pull|pulls|merge_requests)/(\d+)",
        ref
    )
    if m:
        return m.group(1), m.group(2), int(m.group(3))

    m = re.match(r"([^/]+)/([^#]+)#(\d+)", ref)
    if m:
        return m.group(1), m.group(2), int(m.group(3))

    m = re.match(r"#?(\d+)$", ref.strip())
    if m:
        upstream_url = get_remote_url("upstream")
        if upstream_url:
            parsed = parse_gitcode_url(upstream_url)
            if parsed:
                return parsed[0], parsed[1], int(m.group(1))
    return None


# ============================================================================
# Copyright year management
# ============================================================================

def update_copyright_years(filepaths: List[str]) -> List[str]:
    """Check and update copyright years in modified files.

    Rules:
    - ``Copyright YYYY`` -> ``Copyright YYYY-<current_year>`` (across years)
    - ``Copyright YYYY-ZZZZ`` -> ``Copyright YYYY-<current_year>`` (update end year)
    - Only processes copyright declarations within the first 5 lines.

    Args:
        filepaths: List of file paths to check.

    Returns:
        List of file paths that were updated.
    """
    current_year = str(datetime.now().year)
    pat = re.compile(
        r"(Copyright\s+)(\d{4})(?:-(\d{4}))?(\s+)"
    )
    updated: List[str] = []

    for filepath in filepaths:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except (OSError, UnicodeDecodeError):
            continue

        changed = False
        for i in range(min(5, len(lines))):
            m = pat.search(lines[i])
            if not m:
                continue
            start_year = m.group(2)
            end_year = m.group(3)

            if current_year in (end_year, start_year):
                break

            if end_year:
                new_text = f"{m.group(1)}{start_year}-{current_year}{m.group(4)}"
            else:
                new_text = f"{m.group(1)}{start_year}-{current_year}{m.group(4)}"

            lines[i] = lines[i][:m.start()] + new_text + lines[i][m.end():]
            changed = True
            break

        if changed:
            with open(filepath, "w", encoding="utf-8") as f:
                f.writelines(lines)
            updated.append(filepath)

    return updated


# ============================================================================
# Diff helpers for cosmetic change detection
# ============================================================================

def extract_diff_lines(diff_text: str) -> Tuple[List[str], List[str]]:
    """Extract added/removed lines from a unified diff (strip prefix, keep indent).

    Args:
        diff_text: Raw unified diff text.

    Returns:
        Tuple of (added_lines, removed_lines).
    """
    added: List[str] = []
    removed: List[str] = []
    for line in diff_text.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:].rstrip())
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:].rstrip())
    return added, removed


def _is_import_reorder_only(added: List[str], removed: List[str]) -> bool:
    """Detect whether the change is only an import reorder (with symbol sort normalization).

    Supports: ``import x``, ``from x import a, b``,
    ``from x import (a, b)`` and other single-line forms.

    Args:
        added: Non-empty added lines from diff.
        removed: Non-empty removed lines from diff.

    Returns:
        True if the change is an import reorder only.
    """
    if not added or not removed:
        return False

    import_pat = re.compile(
        r"^\s*(from\s+\S+\s+import\s+[\(\s]*.+[\)\s]*|import\s+.+)\s*$"
    )

    if not all(import_pat.match(line) for line in added):
        return False
    if not all(import_pat.match(line) for line in removed):
        return False

    def normalize_import(line: str) -> str:
        """Normalize a single import: strip whitespace/parens, sort symbols.

        Args:
            line: A single import statement string.
        """
        line = line.strip()
        m = re.match(r"(from\s+\S+\s+import\s+)[\(\s]*(.*?)[\)\s]*$", line)
        if m:
            prefix = re.sub(r"\s+", " ", m.group(1).strip())
            symbols = sorted(s.strip() for s in m.group(2).split(","))
            return prefix + ", ".join(symbols)
        return re.sub(r"\s+", " ", line)

    return sorted(normalize_import(l) for l in added) == \
        sorted(normalize_import(l) for l in removed)


def _is_line_reorder_only(added: List[str], removed: List[str]) -> bool:
    """Detect whether the change is a pure line reorder (identical after sorting).

    Args:
        added: Non-empty added lines from diff.
        removed: Non-empty removed lines from diff.

    Returns:
        True if the change is a line reorder only.
    """
    if not added or not removed:
        return False
    return sorted(added) == sorted(removed)


def _check_diff_patterns(filepath: str) -> Tuple[bool, str]:
    """Check import-reorder and line-reorder patterns from staged diff.

    Args:
        filepath: Path to the file being checked.

    Returns:
        Tuple of (is_cosmetic, reason).
    """
    diff_result = run_git("diff", "--cached", "--", filepath, check=False)
    diff_text = diff_result.stdout
    if not diff_text:
        return False, ""

    added, removed = extract_diff_lines(diff_text)
    if not added and not removed:
        return False, ""

    added_nonempty = [l for l in added if l.strip()]
    removed_nonempty = [l for l in removed if l.strip()]

    if filepath.endswith(".py") and added_nonempty and removed_nonempty:
        if _is_import_reorder_only(added_nonempty, removed_nonempty):
            return True, "import reorder"

    if added_nonempty and removed_nonempty:
        if _is_line_reorder_only(added_nonempty, removed_nonempty):
            return True, "line reorder"

    return False, ""


def is_cosmetic_only_change(filepath: str) -> Tuple[bool, str]:
    """Check whether the staged change for a file is cosmetic-only.

    Detection strategies run fast-to-slow with short-circuit returns:
    1. Whitespace only
    2. Whitespace + blank lines
    3. Import reorder (.py only)
    4. Line reorder

    Args:
        filepath: Path to the file being checked.

    Returns:
        Tuple of (is_cosmetic, reason).
    """
    result = run_git("diff", "--cached", "--ignore-all-space", "--quiet",
                     "--", filepath, check=False)
    if result.returncode == 0:
        return True, "whitespace-only change"

    result = run_git("diff", "--cached", "--ignore-all-space",
                     "--ignore-blank-lines", "--quiet",
                     "--", filepath, check=False)
    if result.returncode == 0:
        return True, "whitespace/blank-line change"

    return _check_diff_patterns(filepath)


def get_format_only_files() -> List[Tuple[str, str]]:
    """Identify files in the staging area that only have cosmetic changes.

    Returns:
        List of (filepath, reason) tuples.
    """
    staged_output = run_git("diff", "--cached", "--name-only").stdout.strip()
    if not staged_output:
        return []
    format_only: List[Tuple[str, str]] = []
    for filepath in staged_output.split("\n"):
        if not filepath:
            continue
        is_cosmetic, reason = is_cosmetic_only_change(filepath)
        if is_cosmetic:
            format_only.append((filepath, reason))
    return format_only


# ============================================================================
# Branch scope and cosmetic filtering
# ============================================================================

def detect_base_ref() -> Optional[str]:
    """Get upstream/<default_branch> as the PR range baseline.

    Only returns a valid value on feature branches; returns None on master/main.

    Returns:
        Base ref string (e.g. 'upstream/master'), or None.
    """
    current = get_current_branch()
    if current in ("master", "main", "develop"):
        return None

    default_branch = get_upstream_default_branch()
    if not default_branch:
        return None

    result = run_git("rev-parse", f"upstream/{default_branch}", check=False)
    if result.returncode != 0:
        return None
    return f"upstream/{default_branch}"


def _get_branch_scope_files(base_ref: str) -> set:
    """Get the set of files with substantive changes relative to base_ref.

    Args:
        base_ref: Git ref to diff against (e.g. 'upstream/master').

    Returns:
        Set of file paths.
    """
    result = run_git("diff", "--name-only", f"{base_ref}...HEAD", check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return set()
    return set(result.stdout.strip().split("\n"))


def _get_unrelated_cosmetic_files(
    base_ref: str,
    cosmetic_files: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """Find cosmetic-change files that are not in the branch diff scope.

    If a cosmetic-change file is already in the branch's diff scope
    (had substantive changes before), keep it (do not exclude).
    Only exclude completely unrelated cosmetic-change files.

    Args:
        base_ref: Git ref to diff against.
        cosmetic_files: List of (filepath, reason) tuples.

    Returns:
        List of unrelated (filepath, reason) tuples.
    """
    scope = _get_branch_scope_files(base_ref)
    if not scope:
        return []
    return [(f, reason) for f, reason in cosmetic_files if f not in scope]


def filter_cosmetic_changes(
    base_ref: Optional[str] = None
) -> List[Tuple[str, str]]:
    """Two-layer filter: global cosmetic detection, then out-of-PR-scope detection.

    Args:
        base_ref: Optional git ref for branch scope filtering.

    Returns:
        List of files to exclude: [(filepath, reason), ...].
    """
    cosmetic = get_format_only_files()
    if not cosmetic:
        return []

    if not base_ref:
        return cosmetic

    unrelated_set = {
        f for f, _ in _get_unrelated_cosmetic_files(base_ref, cosmetic)
    }
    result: List[Tuple[str, str]] = []
    for filepath, reason in cosmetic:
        if filepath in unrelated_set:
            result.append((filepath, f"{reason}, and not in branch change scope"))
        else:
            result.append((filepath, reason))
    return result


def get_unpushed_commits(base: str, count: Optional[int] = None) -> List[str]:
    """Get the list of unpushed commit SHAs.

    Args:
        base: Base ref to compare against (e.g. 'upstream/master').
        count: If set, limit to the last N commits.

    Returns:
        List of commit SHA strings.
    """
    cmd = ["log", f"{base}..HEAD", "--pretty=format:%H", "--reverse"]
    result = run_git(*cmd)
    commits = result.stdout.strip().split("\n") if result.stdout.strip() else []
    if count and len(commits) > count:
        return commits[-count:]
    return commits
