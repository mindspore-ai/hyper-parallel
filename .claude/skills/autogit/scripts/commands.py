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
"""Command implementations and argparse CLI for AutoGit."""

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))  # pylint: disable=C9004

from models import AutoGitError, EnvConfig  # pylint: disable=wrong-import-position
from git_utils import (  # pylint: disable=wrong-import-position
    run_git, get_remote_url, parse_gitcode_url,
    get_upstream_default_branch, get_current_branch,
    has_uncommitted_changes, has_staged_changes, stage_all_changes,
    is_protected_branch, branch_exists_on_remote, parse_pr_ref,
    update_copyright_years, detect_base_ref, filter_cosmetic_changes,
    get_unpushed_commits,
)
from api import (  # pylint: disable=wrong-import-position
    get_token, get_pr_info, get_pr_stats,
    get_pr_status_display, create_pr, add_reviewers,
    update_pr_description,
)
from pr_content import generate_pr_content, prepare_pr_analysis  # pylint: disable=wrong-import-position
from lint_check import (  # pylint: disable=wrong-import-position
    run_checks,
    run_pylint_review,
)
from commit_msg_check import validate_commit_message  # pylint: disable=wrong-import-position


# ============================================================================
# Environment check
# ============================================================================

def check_env(require_token: bool = True) -> EnvConfig:
    """Check environment configuration and return validated config.

    Args:
        require_token: Whether a GitCode token is required.

    Returns:
        Validated EnvConfig instance.
    """
    token = get_token()
    if require_token and not token:
        raise AutoGitError(
            "GitCode Token not found.\n\n"
            "Please set the environment variable:\n"
            "  Linux/macOS: export GITCODE_TOKEN=<your-token>\n"
            "  Windows CMD: set GITCODE_TOKEN=<your-token>\n"
            "  Windows PowerShell: $env:GITCODE_TOKEN=\"<your-token>\"\n\n"
            "Get token: https://gitcode.com/setting/token-classic"
        )

    origin_url = get_remote_url("origin")
    if not origin_url:
        raise AutoGitError(
            "Origin remote not found.\n\n"
            "Please fork the main repo first, then clone your fork:\n"
            "  git clone git@gitcode.com:<your-username>/<repo-name>.git"
        )
    origin = parse_gitcode_url(origin_url)
    if not origin:
        raise AutoGitError(f"Cannot parse origin URL: {origin_url}\nOnly GitCode is supported")

    upstream_url = get_remote_url("upstream")
    if not upstream_url:
        raise AutoGitError(
            "Upstream remote not found.\n\n"
            "Fork workflow requires upstream pointing to the main repo:\n"
            "  git remote add upstream git@gitcode.com:<org>/<repo-name>.git\n\n"
            "After configuration, your remotes should be:\n"
            "  origin    -> your fork (writable)\n"
            "  upstream  -> main repo (read-only)"
        )
    upstream = parse_gitcode_url(upstream_url)
    if not upstream:
        raise AutoGitError(f"Cannot parse upstream URL: {upstream_url}\nOnly GitCode is supported")

    run_git("fetch", "upstream", check=False)
    default_branch = get_upstream_default_branch()
    if not default_branch:
        raise AutoGitError("Cannot determine upstream default branch")

    return EnvConfig(
        token=token,
        origin_owner=origin[0],
        origin_repo=origin[1],
        upstream_owner=upstream[0],
        upstream_repo=upstream[1],
        default_branch=default_branch,
    )


# ============================================================================
# Shared helpers
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[4]
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"

def _stage_and_filter_cosmetic(base_ref: Optional[str] = None) -> None:
    """Stage changes, auto-update copyright years, and exclude cosmetic-only changes.

    Args:
        base_ref: Optional git ref for branch scope filtering.
    """
    stage_all_changes()

    staged_output = run_git("diff", "--cached", "--name-only").stdout.strip()
    if staged_output:
        copyright_updated = update_copyright_years(staged_output.split("\n"))
        if copyright_updated:
            for f in copyright_updated:
                run_git("add", f)
            print(f"Updated copyright years in {len(copyright_updated)} files")

    cosmetic = filter_cosmetic_changes(base_ref)
    if cosmetic:
        print(f"Detected {len(cosmetic)} files with cosmetic-only changes, excluded:")
        for filepath, reason in cosmetic:
            print(f"   - {filepath} ({reason})")
            run_git("reset", "HEAD", "--", filepath, check=False)
            run_git("checkout", "--", filepath, check=False)


def _load_pre_commit_dependency_map() -> Dict[str, str]:
    """Parse .pre-commit-config.yaml and return hook id to dependency mapping."""
    if not PRE_COMMIT_CONFIG.is_file():
        return {}

    dependency_map: Dict[str, str] = {}
    current_hook_id: Optional[str] = None
    in_additional_dependencies = False

    for raw_line in PRE_COMMIT_CONFIG.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        hook_match = re.match(r"^\s*-\s+id:\s+(.+?)\s*$", line)
        if hook_match:
            current_hook_id = hook_match.group(1).strip()
            in_additional_dependencies = False
            continue

        if re.match(r"^\s*additional_dependencies:\s*$", line):
            in_additional_dependencies = True
            continue

        if in_additional_dependencies:
            dep_match = re.match(r"^\s*-\s+(.+?)\s*$", line)
            if dep_match and current_hook_id:
                dependency_map[current_hook_id] = dep_match.group(1).strip()
                continue
            if line.strip() and not line.startswith(" " * 10):
                in_additional_dependencies = False

    return dependency_map


def _install_python_dependency(requirement: str) -> None:
    """Install a Python dependency into the current environment."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", requirement],
            capture_output=False,
            check=False,
        )
    except OSError as exc:
        raise AutoGitError(f"Failed to invoke pip for dependency installation: {requirement}") from exc
    if result.returncode != 0:
        raise AutoGitError(f"Failed to install Python dependency: {requirement}")


def _install_node_dependency(requirement: str) -> None:
    """Install a Node dependency globally for CLI availability."""
    try:
        result = subprocess.run(
            ["npm", "install", "-g", requirement],
            capture_output=False,
            check=False,
        )
    except OSError as exc:
        raise AutoGitError(f"Failed to invoke npm for dependency installation: {requirement}") from exc
    if result.returncode != 0:
        raise AutoGitError(f"Failed to install Node dependency: {requirement}")


def _ensure_check_dependencies(files: List[str]) -> None:
    """Install required check dependencies for autogit check when missing."""
    dependency_map = _load_pre_commit_dependency_map()
    needs_pylint = any(path.endswith(".py") for path in files)
    needs_lizard = any(
        path.endswith((".py", ".c", ".cc", ".cpp", ".h", ".hpp"))
        for path in files
    )
    needs_markdownlint = any(path.endswith(".md") for path in files)

    if needs_pylint and not shutil.which("pylint"):
        requirement = dependency_map.get("pylint", "pylint")
        print(f"Installing missing pylint dependency: {requirement}")
        _install_python_dependency(requirement)

    if needs_lizard and not shutil.which("lizard"):
        requirement = dependency_map.get("lizard", "lizard")
        print(f"Installing missing lizard dependency: {requirement}")
        _install_python_dependency(requirement)

    if needs_markdownlint and not shutil.which("markdownlint-cli2"):
        requirement = dependency_map.get("markdownlint", "markdownlint-cli2")
        print(f"Installing missing markdownlint dependency: {requirement}")
        _install_node_dependency(requirement)


def cmd_pylint_review(base_ref: Optional[str] = None) -> str:
    """Run pylint on changed Python files (for review-PR stage). Returns report.

    Args:
        base_ref: Git ref to diff against (default: detect_base_ref(), e.g. upstream/master).

    Returns:
        Pylint report string.
    """
    check_env(require_token=False)
    ref = base_ref or detect_base_ref() or "upstream/master"
    diff_out = run_git("diff", "--name-only", f"{ref}...HEAD").stdout.strip()
    files = [f for f in diff_out.split("\n") if f.strip().endswith(".py")]
    if not files:
        return "No Python files changed (pylint-review skipped).\n"
    print(f"Running pylint on {len(files)} changed .py files (base: {ref})...")
    _, report = run_pylint_review(files)
    print(report)
    return report


def cmd_test() -> None:
    """Run test stage: pytest only."""
    check_env(require_token=False)
    print("Running pytest...")
    result = subprocess.run(
        ["pytest", "tests/ut", "-v"],
        capture_output=False,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        raise AutoGitError("pytest failed")
    print("Test stage passed (pytest).")


# ============================================================================
# Command: commit
# ============================================================================

def cmd_commit(message: Optional[str] = None) -> Dict[str, Any]:
    """Stage, preview commit message, commit, and push.

    Lint runs via the repo's pre-commit git hook (installed by
    ``scripts/pre-commit/install.sh``) — autogit does not duplicate it.
    UT/ST live in ``autogit pr``, not here, because commits should be cheap.

    The only interactive step is the commit-message preview (tty) or the
    ``-m`` requirement (non-tty).

    Args:
        message: Commit message. If omitted, the script generates a default
            from the staged diff and runs the content preview gate.

    Returns:
        Dict with keys: sha, message, branch.
    """
    check_env(require_token=False)

    if not has_uncommitted_changes():
        raise AutoGitError("No changes to commit")

    base_ref = detect_base_ref()
    _stage_and_filter_cosmetic(base_ref)

    if not has_staged_changes():
        raise AutoGitError("No changes to commit after excluding cosmetic changes")

    _warn_if_pre_commit_hook_missing()

    if not message:
        changed_files = run_git("diff", "--cached", "--name-only").stdout.strip().split("\n")
        if len(changed_files) == 1:
            auto_msg = f"Update {changed_files[0].split('/')[-1]}"
        else:
            auto_msg = f"Update {len(changed_files)} files"
        message = _preview_and_confirm(
            "commit message", auto_msg,
            skip_flag_hint='-m "<approved message>"',
        )

    err = validate_commit_message(message)
    if err:
        raise AutoGitError(err)

    run_git("commit", "-m", message)
    sha = run_git("rev-parse", "HEAD").stdout.strip()
    print(f"Created commit: {sha[:8]}")
    print(f"   {message}")

    branch = get_current_branch()
    print(f"Pushing to origin/{branch}...")
    result = run_git("push", "-u", "origin", branch, check=False)
    if result.returncode != 0:
        if "rejected" in result.stderr or "non-fast-forward" in result.stderr:
            raise AutoGitError(
                f"Push failed, remote has updates.\n"
                f"Please run: git pull --rebase origin {branch}"
            )
        raise AutoGitError(f"Push failed: {result.stderr}")

    print(f"Pushed to origin/{branch}")

    return {
        "sha": sha,
        "message": message,
        "branch": branch
    }


# ============================================================================
# Command: check
# ============================================================================

def cmd_check() -> None:
    """Run lint checks independently (without committing)."""
    if has_uncommitted_changes():
        stage_all_changes()
        staged_output = run_git("diff", "--cached", "--name-only").stdout.strip()
        files = staged_output.split("\n") if staged_output else []
        run_git("reset", check=False)
    else:
        raise AutoGitError("No changes to check")

    if not files:
        raise AutoGitError("No files to check")

    _ensure_check_dependencies(files)
    print("Running lint checks...")
    passed, report = run_checks(files)
    print(report)

    if not passed:
        raise AutoGitError("Lint checks failed")


# ============================================================================
# Command: pr
# ============================================================================

def _prepare_pr_branch(current_branch: str, base_ref: str,
                       commits: List[str], squash: bool) -> Tuple[str, bool, int]:
    """Decide and prepare the PR branch.

    Args:
        current_branch: Current git branch.
        base_ref: Base ref to compare against.
        commits: List of commit SHAs.
        squash: Whether to squash commits.

    Returns:
        Tuple of (pr_branch, need_new_branch, final_commit_count).
    """
    if is_protected_branch(current_branch):
        print(f"Currently on protected branch '{current_branch}', creating a new PR branch")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pr_branch = f"pr/{timestamp}"
        final_commits = _cherry_pick_to_new_branch(
            pr_branch, base_ref, current_branch, commits, squash
        )
        return pr_branch, True, final_commits

    pr_branch = current_branch
    print(f"Using current branch '{pr_branch}' to create PR")
    final_commits = _prepare_existing_branch(base_ref, commits, squash)
    return pr_branch, False, final_commits


def _cherry_pick_to_new_branch(pr_branch: str, base_ref: str,
                               current_branch: str,
                               commits: List[str],
                               squash: bool) -> int:
    """Create a new branch and cherry-pick commits.

    Args:
        pr_branch: New branch name.
        base_ref: Base ref to branch from.
        current_branch: Original branch to return to on failure.
        commits: Commit SHAs to cherry-pick.
        squash: Whether to squash afterwards.

    Returns:
        Final commit count.
    """
    print(f"Creating new branch: {pr_branch}")
    backup = f"backup/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_git("branch", backup)
    print(f"Backup: {backup}")

    run_git("checkout", "-b", pr_branch, base_ref)

    print(f"Cherry-picking {len(commits)} commits...")
    for sha in commits:
        result = run_git("cherry-pick", sha, check=False)
        if result.returncode != 0:
            run_git("cherry-pick", "--abort", check=False)
            run_git("checkout", current_branch)
            run_git("branch", "-D", pr_branch, check=False)
            raise AutoGitError(
                f"Cherry-pick {sha[:8]} failed\n"
                f"Please resolve conflicts manually or develop on a feature branch"
            )
    print("Changes applied")

    return _squash_if_needed(squash, commits, base_ref)


def _prepare_existing_branch(base_ref: str, commits: List[str],
                             squash: bool) -> int:
    """Prepare PR on the existing branch.

    Args:
        base_ref: Base ref to compare against.
        commits: List of commit SHAs.
        squash: Whether to squash commits.

    Returns:
        Final commit count.
    """
    upstream_head = run_git("rev-parse", base_ref).stdout.strip()
    merge_base = run_git("merge-base", "HEAD", base_ref).stdout.strip()

    if upstream_head != merge_base:
        base_name = base_ref.split('/')[-1]
        print(f"Tip: current branch is behind {base_ref}")
        print(f"   Consider running: git rebase upstream/{base_name}")

    return _squash_if_needed(squash, commits, base_ref)


def _squash_if_needed(squash: bool, commits: List[str],
                      base_ref: str) -> int:
    """Squash commits if requested.

    Args:
        squash: Whether to squash.
        commits: Commit SHAs.
        base_ref: Base ref for reset.

    Returns:
        Final commit count.
    """
    if not squash or len(commits) <= 1:
        return len(commits)

    print("Squashing commits...")
    backup = f"backup/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_git("branch", backup)
    print(f"Backup: {backup}")

    run_git("reset", "--soft", base_ref)
    msg = run_git("log", "-1", "--pretty=format:%s", commits[0]).stdout.strip()
    msg = msg or "Update code"
    err = validate_commit_message(msg)
    if err:
        raise AutoGitError(err)
    run_git("commit", "-m", msg)
    print("Squashed into a single commit")
    return 1


def _push_pr_branch(pr_branch: str) -> None:
    """Push the PR branch to origin.

    Args:
        pr_branch: Branch name to push.
    """
    print(f"Pushing to origin/{pr_branch}...")

    if branch_exists_on_remote("origin", pr_branch):
        result = run_git("push", "origin", pr_branch, check=False)
        if result.returncode != 0:
            if "non-fast-forward" in result.stderr or "rejected" in result.stderr:
                print("Remote branch has updates, force push required")
                print("   This may be due to a previous squash or rebase")
                run_git("push", "-f", "origin", pr_branch)
                print("Force push complete")
            else:
                raise AutoGitError(f"Push failed: {result.stderr}")
    else:
        run_git("push", "-u", "origin", pr_branch)

    print(f"Pushed to origin/{pr_branch}")


_GATE_LABELS = {
    "ut": "UT (pytest, default scope=changed)",
    "st": "ST (pytest tests/{torch,mindspore}/st, default scope=skip)",
}

_GATE_DEFAULTS = {"ut": "changed", "st": "skip"}
_GATE_CHOICES = ("skip", "changed", "full")


def _warn_if_pre_commit_hook_missing() -> None:
    """Print a one-line reminder if the repo's pre-commit hook is not installed.

    Lint (pylint + markdownlint) is the responsibility of the project's
    pre-commit git hook (installed via ``scripts/pre-commit/install.sh``).
    autogit does not duplicate it — but if the hook is missing the commit
    will go through without lint, so we surface a non-blocking warning.
    """
    hook = REPO_ROOT / ".git" / "hooks" / "pre-commit"
    installer = REPO_ROOT / "scripts" / "pre-commit" / "install.sh"
    if hook.is_file():
        return
    print()
    print("[WARN] pre-commit hook not installed — this commit will skip lint.")
    if installer.is_file():
        print(f"       Install: bash {installer.relative_to(REPO_ROOT)}")
    print()


def _collect_pr_gate_choices(ut: Optional[str],
                             st: Optional[str]) -> Tuple[str, str]:
    """Resolve PR-time UT/ST gates to one of ``skip`` | ``changed`` | ``full``.

    Lint is owned by the project's pre-commit git hook (per-commit), not gated
    here. PR-time gates are tri-state:

    * ``skip``    — do not run.
    * ``changed`` — only run test files that appear in this PR's diff (fast
                    default; covers exactly what the PR introduces).
    * ``full``    — run the entire suite (regression coverage).

    Behaviour:
      * Any value explicitly passed (not ``None``) is honored as-is.
      * On tty + missing values: prompt **one gate at a time** (``一步一步``),
        each prompt accepts ``c`` / ``f`` / ``n`` / Enter (default).
      * On non-tty + missing values: raise once listing every undecided gate
        with the explicit flag the AI must pass, so the AI asks the user in
        chat and then re-invokes with all flags in a single call.

    Args:
        ut: Decision for the UT gate (``None`` means undecided).
        st: Decision for the ST gate (``None`` means undecided).

    Returns:
        Tuple ``(ut, st)`` where each value is one of
        ``'skip'`` | ``'changed'`` | ``'full'``.

    Raises:
        AutoGitError: On non-tty with undecided gates, or invalid tty input.
    """
    undecided = [
        (key, value) for key, value in (("ut", ut), ("st", st))
        if value is None
    ]
    if not undecided:
        return ut, st

    if not sys.stdin.isatty():
        lines = [
            "PR quality gates need an explicit choice (non-interactive context).",
            "",
            "AutoGit is designed to run inside an AI coding agent",
            "(Claude Code / Codex / Copilot / ...). Ask the user one gate at",
            "a time in chat, then re-invoke autogit with all flags in a",
            "single call.",
            "",
            "Each gate accepts: skip | changed | full",
            "  changed  test files appearing in this PR's diff (fast default)",
            "  full     entire suite (regression coverage)",
            "  skip     do not run",
            "",
            "Undecided gates:",
        ]
        for key, _ in undecided:
            lines.append(
                f"  --{key} {{skip,changed,full}}    default: {_GATE_DEFAULTS[key]}"
            )
            lines.append(f"      {_GATE_LABELS[key]}")
        lines += [
            "",
            "Re-invoke example:",
            "  autogit pr --ut changed --st skip",
        ]
        raise AutoGitError("\n".join(lines))

    print()
    print("PR quality gates — answering one at a time")
    print("(Enter for default; c=changed, f=full, n=skip)")
    answers: Dict[str, str] = {}
    for key, _ in undecided:
        default = _GATE_DEFAULTS[key]
        labels = ["c", "f", "n"]
        prompt_letters = "/".join(
            letter.upper() if letter == default[0] else letter for letter in labels
        )
        prompt = f"  {_GATE_LABELS[key]} ? [{prompt_letters}] "
        while True:
            try:
                raw = input(prompt).strip().lower()
            except EOFError as exc:
                raise AutoGitError(
                    f"EOF reading {key} gate. "
                    f"Pass --{key} {{skip,changed,full}} explicitly."
                ) from exc
            if not raw:
                answers[key] = default
                break
            if raw in ("c", "changed"):
                answers[key] = "changed"
                break
            if raw in ("f", "full"):
                answers[key] = "full"
                break
            if raw in ("n", "no", "skip"):
                answers[key] = "skip"
                break
            print(f"    invalid '{raw}', expected c / f / n / Enter")

    if ut is None:
        ut = answers["ut"]
    if st is None:
        st = answers["st"]
    return ut, st


def _preview_and_confirm(label: str, content: str,
                         skip_flag_hint: str,
                         long_text: bool = False) -> str:
    """Show generated content (commit msg / PR title / PR body) and resolve final value.

    Behaviour:
      * tty: render the content in a bordered block, accept ok/edit/cancel.
      * non-tty: raise AutoGitError. AI must generate, get user approval, and
        re-invoke with the explicit flag.

    The script never silently accepts auto-generated content in non-tty contexts
    because the user has not seen it. This is the only way to keep AI honest
    about preview-before-write.

    Args:
        label: Human-readable label (e.g. "commit message", "PR body").
        content: Auto-generated default to preview.
        skip_flag_hint: Flag the caller should pass to bypass the prompt.
        long_text: If True, hint that this is multi-line content.

    Returns:
        The approved content string.

    Raises:
        AutoGitError: Non-tty without explicit flag, or user cancels/edits.
    """
    if not sys.stdin.isatty():
        body_kind = "multi-line body" if long_text else "single-line text"
        raise AutoGitError(
            f"{label} required (non-interactive context).\n"
            f"\n"
            f"AutoGit is designed to run inside an AI coding agent\n"
            f"(Claude Code / Codex / Copilot / ...). Generating a high-quality\n"
            f"{label} requires reading the diff and producing natural-language\n"
            f"content — that is exactly what the LLM in the agent is for.\n"
            f"\n"
            f"Recommended workflow inside an agent:\n"
            f"  1. Read the relevant diff:\n"
            f"       git diff --cached                    (for commit message)\n"
            f"       git diff upstream/master...HEAD      (for PR title/body)\n"
            f"  2. Generate the {body_kind}.\n"
            f"  3. Show it to the user in chat and wait for approval.\n"
            f"  4. Re-invoke autogit with: {skip_flag_hint}\n"
            f"\n"
            f"Outside an agent (script / CI):\n"
            f"  Prepare the {label} yourself and pass {skip_flag_hint}\n"
            f"  directly. The script then runs without this prompt.\n"
            f"\n"
            f"Why this guard exists: an AI may generate plausible-looking\n"
            f"content and silently pass it. The user must see and approve\n"
            f"every commit message and PR description before it lands."
        )

    print()
    print("=" * 68)
    print(f"Preview {label} (auto-generated):")
    print("-" * 68)
    print(content)
    print("=" * 68)
    print("  [Enter] / y  → use as-is and continue")
    print("  e            → abort; edit in chat with user, then re-invoke")
    print("  c            → cancel")
    try:
        choice = input("> ").strip().lower()
    except EOFError as exc:
        raise AutoGitError(
            f"EOF while waiting for {label} confirmation. "
            f"Pass {skip_flag_hint} explicitly."
        ) from exc

    if choice in ("", "y", "yes", "ok"):
        return content
    if choice == "e":
        raise AutoGitError(
            f"User requested edit of {label}. "
            f"Revise it together with the user, then re-invoke with {skip_flag_hint}."
        )
    if choice == "c":
        raise AutoGitError(f"User cancelled at {label} preview.")
    raise AutoGitError(f"Unknown choice '{choice}' (expected y / e / c).")


def _diff_test_files(diff_range: Optional[str], kind: str) -> List[str]:
    """Return changed test files of given ``kind`` (``'ut'`` | ``'st'``).

    Filters:
      * Only files under ``tests/`` whose path contains the ``kind`` segment
        (e.g. ``tests/torch/ut/test_x.py`` for ``'ut'``).
      * Only files named ``test_*.py`` — excludes ``conftest.py``,
        ``__init__.py``, fixtures, helpers, etc.
      * Deleted files are excluded via ``--diff-filter=ACMR`` (Added /
        Copied / Modified / Renamed) so pytest is not handed a path that
        no longer exists.

    Args:
        diff_range: Git ref range like ``upstream/master...HEAD``. If ``None``,
            scan the working tree (staged + unstaged).
        kind: ``'ut'`` or ``'st'`` — selects which test category to filter.

    Returns:
        Sorted list of repo-relative paths suitable for direct ``pytest`` use.
    """
    if diff_range:
        names = run_git(
            "diff", "--name-only", "--diff-filter=ACMR", diff_range
        ).stdout.splitlines()
    else:
        unstaged = run_git(
            "diff", "--name-only", "--diff-filter=ACMR"
        ).stdout.splitlines()
        staged = run_git(
            "diff", "--name-only", "--diff-filter=ACMR", "--cached"
        ).stdout.splitlines()
        names = list(set(unstaged + staged))
    files: List[str] = []
    for raw in names:
        line = raw.strip()
        if not line or not line.startswith("tests/"):
            continue
        path = Path(line)
        if kind not in path.parts:
            continue
        if not (path.name.startswith("test_") and path.name.endswith(".py")):
            continue
        files.append(line)
    return sorted(set(files))


def _run_pr_ut_gate(scope: str, diff_range: Optional[str]) -> None:
    """Run UT pytest at the chosen ``scope`` (``'changed'`` | ``'full'``).

    ``scope='skip'`` is handled by the caller (no-op, do not invoke).

    Args:
        scope: ``'changed'`` runs only test files appearing in the PR's diff;
            ``'full'`` runs the entire ``tests/ut`` suite.
        diff_range: Git ref range used for ``'changed'`` scope. ``None`` means
            scan working tree (used by ``pr --to`` / append flow).
    """
    if scope == "full":
        targets = ["tests/ut"]
        print("Running UT gate (full): pytest tests/ut")
    else:
        targets = _diff_test_files(diff_range, "ut")
        if not targets:
            print("UT gate (changed): no UT test files in this PR's diff — skipping.")
            return
        print(f"Running UT gate (changed, {len(targets)} files):")
        for path in targets:
            print(f"  {path}")
    result = subprocess.run(
        ["pytest", *targets, "-v"],
        capture_output=False,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        raise AutoGitError("UT gate failed.")
    print("UT gate passed.")


def _run_pr_st_gate(scope: str, diff_range: Optional[str]) -> None:
    """Run ST pytest at the chosen ``scope`` (``'changed'`` | ``'full'``).

    ``scope='skip'`` is handled by the caller. Failures (missing dirs,
    missing launcher, single-card env, real test failure) surface as
    ``AutoGitError`` so the user can re-run with ``--st skip`` instead of
    silently skipping.

    Args:
        scope: ``'changed'`` runs only ST test files in the PR's diff;
            ``'full'`` runs ``tests/torch/st`` + ``tests/mindspore/st``.
        diff_range: Git ref range used for ``'changed'`` scope. ``None``
            means scan working tree.
    """
    if scope == "full":
        targets = [d for d in ("tests/torch/st", "tests/mindspore/st")
                   if (REPO_ROOT / d).is_dir()]
        if not targets:
            raise AutoGitError(
                "ST gate (full): no ST test directories found "
                "(looked for tests/torch/st, tests/mindspore/st).\n"
                "Re-run with --st skip if this repo has no ST suite."
            )
        print(f"Running ST gate (full) on: {', '.join(targets)}")
    else:
        targets = _diff_test_files(diff_range, "st")
        if not targets:
            print("ST gate (changed): no ST test files in this PR's diff — skipping.")
            return
        print(f"Running ST gate (changed, {len(targets)} files):")
        for path in targets:
            print(f"  {path}")
    result = subprocess.run(
        ["pytest", *targets, "-v"],
        capture_output=False,
        check=False,
    )
    if result.returncode != 0:
        raise AutoGitError(
            "ST gate failed. Common causes:\n"
            "  - missing torchrun / msrun launcher\n"
            "  - single-card env (ST typically needs multi-card)\n"
            "  - real test failure\n"
            "If the env cannot run ST, re-run with --st skip."
        )
    print("ST gate passed.")


def cmd_pr(base: Optional[str] = None, reviewer: Optional[str] = None,
           squash: bool = False, ut: Optional[str] = None,
           st: Optional[str] = None,
           analyze_only: bool = False,
           title: Optional[str] = None,
           body: Optional[str] = None) -> Dict[str, Any]:
    """Create a PR with safe Git workflow.

    Lint is owned by the project's pre-commit git hook (per-commit), not
    re-run here. PR-time gates focus on UT (default scope=changed) and ST
    (default scope=skip). Each accepts ``'skip'`` | ``'changed'`` | ``'full'``.

    Args:
        base: Target branch (defaults to upstream default branch).
        reviewer: Comma-separated reviewer login names.
        squash: Whether to squash all commits.
        ut: UT gate decision. ``None`` = ask on tty, error on non-tty.
        st: ST gate decision. ``None`` = ask on tty, error on non-tty.
        analyze_only: Only output structured analysis data for LLM-based
            description generation.
        title: Explicit PR title (skip auto-generation).
        body: Explicit PR body (skip auto-generation).

    Returns:
        Dict with keys: url, branch, commits, pr_number.
    """
    env = check_env()

    actual_base = base or env.default_branch
    base_ref = f"upstream/{actual_base}"
    current_branch = get_current_branch()

    if has_uncommitted_changes():
        raise AutoGitError(
            "Uncommitted changes detected, please commit first:\n"
            "  /autogit commit -m \"your message\"\n"
            "Or manually: git add -A && git commit -m \"message\""
        )

    print("Updating remote info...")
    run_git("fetch", "upstream", actual_base, check=False)
    run_git("fetch", "origin", check=False)

    if analyze_only:
        ut_decision, st_decision = "skip", "skip"
    else:
        ut_decision, st_decision = _collect_pr_gate_choices(ut, st)

    diff_range = f"{base_ref}...HEAD"
    if ut_decision != "skip":
        _run_pr_ut_gate(ut_decision, diff_range)

    if st_decision != "skip":
        _run_pr_st_gate(st_decision, diff_range)

    commits = get_unpushed_commits(base_ref)
    if not commits:
        raise AutoGitError(
            f"No new commits relative to upstream/{actual_base}\n"
            f"Please develop and commit on the current branch first"
        )

    print(f"Submitting {len(commits)} commits to {env.upstream_owner}/{env.upstream_repo}")
    for i, sha in enumerate(commits, 1):
        msg = run_git("log", "-1", "--pretty=format:%s", sha).stdout.strip()
        print(f"   {i}. {sha[:8]} {msg[:50]}")

    pr_branch, need_new_branch, final_commits = _prepare_pr_branch(
        current_branch, base_ref, commits, squash
    )

    _push_pr_branch(pr_branch)

    diff = run_git("diff", f"{base_ref}...HEAD").stdout

    if analyze_only:
        analysis_json = prepare_pr_analysis(diff, commits)
        return {
            "analysis": analysis_json,
            "base_ref": base_ref,
            "branch": pr_branch,
            "commits": [sha[:8] for sha in commits],
        }

    auto_title, auto_body = generate_pr_content(diff, commits)
    if not title:
        title = _preview_and_confirm(
            "PR title", auto_title,
            skip_flag_hint='--title "<approved title>"',
        )
    if not body:
        body = _preview_and_confirm(
            "PR body", auto_body,
            skip_flag_hint='--body "<approved multi-line body>"',
            long_text=True,
        )

    print("Creating PR...")
    head = f"{env.origin_owner}:{pr_branch}"
    status, result = create_pr(
        env.upstream_owner, env.upstream_repo, env.token,
        title, body, head, actual_base,
        env.origin_owner, env.origin_repo
    )

    if status not in [200, 201]:
        if "already exists" in str(result).lower() or status == 422:
            raise AutoGitError(
                f"PR creation failed (may already exist): {result}\n"
                f"Please check: https://gitcode.com/{env.upstream_owner}/{env.upstream_repo}/pulls"
            )
        raise AutoGitError(f"PR creation failed: {result}")

    pr_number = result.get("number") or result.get("iid")
    pr_url = (
        result.get("html_url")
        or f"https://gitcode.com/{env.upstream_owner}/{env.upstream_repo}/pull/{pr_number}"
    )

    if reviewer and pr_number:
        reviewer_list = [r.strip() for r in reviewer.split(",")]
        print(f"Adding reviewers: {', '.join(reviewer_list)}")
        add_reviewers(env.upstream_owner, env.upstream_repo, pr_number, env.token, reviewer_list)

    if need_new_branch:
        run_git("checkout", current_branch)
        print(f"Switched back to {current_branch}")

    return {
        "url": pr_url,
        "branch": pr_branch,
        "commits": final_commits,
        "pr_number": pr_number
    }


# ============================================================================
# Command: pr --to (append)
# ============================================================================

def _validate_pr_ownership(pr_number: int, pr_data: Dict,
                           origin_owner: str,
                           default_branch: str) -> Tuple[str, str]:
    """Validate PR ownership and return (source_branch, target_branch).

    Args:
        pr_number: Pull request number.
        pr_data: PR info dict from API.
        origin_owner: Owner of the origin fork.
        default_branch: Default branch name.

    Returns:
        Tuple of (source_branch, target_branch).
    """
    head_info = pr_data.get("head", {})
    base_info = pr_data.get("base", {})
    source_branch = head_info.get("ref")
    target_branch = base_info.get("ref", default_branch)
    head_repo = head_info.get("repo", {})
    head_owner = (head_repo.get("namespace", {}).get("path") or
                  head_repo.get("owner", {}).get("login"))

    if not source_branch:
        raise AutoGitError(f"Cannot get source branch for PR #{pr_number}")

    if head_owner and head_owner != origin_owner:
        raise AutoGitError(
            f"PR #{pr_number} does not belong to your fork ({head_owner} != {origin_owner})"
        )
    return source_branch, target_branch


def _rebase_branch(target_branch: str) -> None:
    """Rebase the current branch onto upstream/target.

    Args:
        target_branch: Target branch name.
    """
    print(f"Rebasing onto upstream/{target_branch}...")
    result = run_git("rebase", f"upstream/{target_branch}", check=False)
    if result.returncode != 0:
        run_git("rebase", "--abort", check=False)
        raise AutoGitError(
            "Rebase failed, please resolve conflicts manually and retry.\n"
            "Or use --no-rebase to skip rebase"
        )
    print("Rebase complete")


def _commit_append(amend: bool, message: Optional[str],
                   pr_commits: int,
                   base_ref: Optional[str] = None) -> int:
    """Execute the append commit operation; return the new total commit count.

    Lint runs via the project's pre-commit git hook (per-commit), not here.

    Args:
        amend: Whether to amend the previous commit.
        message: Optional commit message.
        pr_commits: Current PR commit count.
        base_ref: Optional base ref for cosmetic filtering.

    Returns:
        New total commit count.
    """
    if amend:
        print("Merging into previous commit (amend)...")
        _stage_and_filter_cosmetic(base_ref)
        _warn_if_pre_commit_hook_missing()
        if message:
            err = validate_commit_message(message)
            if err:
                raise AutoGitError(err)
            run_git("commit", "--amend", "-m", message)
        else:
            run_git("commit", "--amend", "--no-edit")
        return pr_commits

    print("Creating new commit...")
    _stage_and_filter_cosmetic(base_ref)
    _warn_if_pre_commit_hook_missing()
    if not message:
        changed = run_git("diff", "--cached", "--name-only").stdout.strip().split("\n")
        if len(changed) > 1:
            auto_msg = f"Update {len(changed)} files"
        else:
            auto_msg = f"Update {changed[0].split('/')[-1]}"
        message = _preview_and_confirm(
            "commit message", auto_msg,
            skip_flag_hint='-m "<approved message>"',
        )
    err = validate_commit_message(message)
    if err:
        raise AutoGitError(err)
    run_git("commit", "-m", message)
    sha = run_git("rev-parse", "HEAD").stdout.strip()
    print(f"New commit: {sha[:8]}")
    return pr_commits + 1


def cmd_pr_append(pr_number: int, amend: bool = False,
                  no_rebase: bool = False, message: Optional[str] = None,
                  ut: Optional[str] = None) -> Dict[str, Any]:
    """Append a commit to an existing PR.

    Lint runs via the project's pre-commit git hook. PR-append asks only the
    UT gate (ST is intentionally skipped for the lighter append flow). UT
    accepts ``'skip'`` | ``'changed'`` | ``'full'``; ``'changed'`` scope
    derives test files from the local working tree (staged + unstaged) since
    the gate runs before any branch operations.

    Args:
        pr_number: Pull request number.
        amend: Whether to amend the last commit.
        no_rebase: Skip rebase if True.
        message: Optional commit message.
        ut: UT gate decision. ``None`` = ask on tty, error on non-tty.

    Returns:
        Dict with keys: url, branch, pr_number, amend, commits.
    """
    env = check_env()

    ut_decision, _ = _collect_pr_gate_choices(ut, st="skip")

    if ut_decision != "skip":
        _run_pr_ut_gate(ut_decision, diff_range=None)

    print(f"Fetching PR #{pr_number} info...")
    status, pr_data = get_pr_info(env.upstream_owner, env.upstream_repo, pr_number, env.token)
    if status != 200:
        raise AutoGitError(f"Cannot get PR #{pr_number} info: {pr_data}")

    source_branch, target_branch = _validate_pr_ownership(
        pr_number, pr_data, env.origin_owner, env.default_branch
    )
    print(f"Source branch: {source_branch} -> {target_branch}")

    current_branch = get_current_branch()
    had_uncommitted = has_uncommitted_changes()

    if had_uncommitted:
        print("Stashing local changes...")
        run_git("stash", "push", "-m", "autogit-temp-stash")

    try:
        print("Updating remote branches...")
        run_git("fetch", "origin", source_branch)
        run_git("fetch", "upstream", target_branch)

        run_git("checkout", source_branch)
        run_git("reset", "--hard", f"origin/{source_branch}")

        pr_commits = pr_data.get("commits") or get_pr_stats(
            env.upstream_owner, env.upstream_repo, pr_number, env.token)["commits"]
        print(f"PR currently has {pr_commits} commits")

        do_rebase = not no_rebase
        if do_rebase:
            _rebase_branch(target_branch)

        if had_uncommitted:
            print("Restoring local changes...")
            result = run_git("stash", "pop", check=False)
            if result.returncode != 0:
                print("Conflicts while restoring changes, please review")

        if not has_uncommitted_changes() and not has_staged_changes():
            raise AutoGitError("No changes to commit")

        pr_base_ref = f"upstream/{target_branch}"
        new_commits = _commit_append(amend, message, pr_commits,
                                     base_ref=pr_base_ref)

        print("Pushing to remote...")
        if do_rebase or amend:
            run_git("push", "-f", "origin", source_branch)
        else:
            run_git("push", "origin", source_branch)

        print(f"Switching back to {current_branch}...")
        run_git("checkout", current_branch)

        pr_url = f"https://gitcode.com/{env.upstream_owner}/{env.upstream_repo}"
        return {
            "url": f"{pr_url}/pull/{pr_number}",
            "branch": source_branch,
            "pr_number": pr_number,
            "amend": amend,
            "commits": new_commits
        }

    except Exception:
        run_git("rebase", "--abort", check=False)
        run_git("checkout", current_branch, check=False)
        if had_uncommitted:
            run_git("stash", "pop", check=False)
        raise


# ============================================================================
# Command: status
# ============================================================================

def cmd_status(pr_ref: str) -> str:
    """View PR status.

    Args:
        pr_ref: PR reference string (number, URL, or owner/repo#N).

    Returns:
        Formatted status string.
    """
    env = check_env()

    parsed = parse_pr_ref(pr_ref)
    if not parsed:
        raise AutoGitError(f"Cannot parse PR reference: {pr_ref}")

    _, _, pr_number = parsed
    return get_pr_status_display(env.upstream_owner, env.upstream_repo, pr_number, env.token)


# ============================================================================
# Command: update
# ============================================================================

def cmd_update(pr_ref: str,
               title: Optional[str] = None,
               body: Optional[str] = None) -> Dict[str, Any]:
    """Regenerate and update a PR description.

    Preview gate: when ``title`` or ``body`` is omitted the auto-generated value
    is gated through :func:`_preview_and_confirm` — tty shows a bordered
    preview with ok/edit/cancel, non-tty raises ``AutoGitError`` demanding the
    explicit flag. Same contract as ``commit`` / ``pr``.

    Args:
        pr_ref: PR reference string.
        title: User-approved PR title; if ``None`` the auto-generated value is
            previewed / required.
        body: User-approved PR body; if ``None`` the auto-generated value is
            previewed / required.

    Returns:
        Dict with keys: pr_number, title, url.
    """
    env = check_env()

    parsed = parse_pr_ref(pr_ref)
    if not parsed:
        raise AutoGitError(f"Cannot parse PR reference: {pr_ref}")

    _, _, pr_number = parsed

    print(f"Fetching PR #{pr_number} info...")

    status, pr_data = get_pr_info(env.upstream_owner, env.upstream_repo, pr_number, env.token)
    if status != 200:
        raise AutoGitError(f"Cannot get PR #{pr_number} info: {pr_data}")

    head_info = pr_data.get("head", {})
    base_info = pr_data.get("base", {})
    source_branch = head_info.get("ref")
    target_branch = base_info.get("ref", env.default_branch)

    if not source_branch:
        raise AutoGitError(f"Cannot get source branch for PR #{pr_number}")

    print(f"Branch: {source_branch} -> {target_branch}")

    print("Analyzing code changes...")
    run_git("fetch", "origin", source_branch, check=False)
    run_git("fetch", "upstream", target_branch, check=False)

    diff_result = run_git("diff", f"upstream/{target_branch}...origin/{source_branch}", check=False)
    if not diff_result.stdout.strip():
        raise AutoGitError("Cannot get PR diff, please ensure branches exist")

    diff = diff_result.stdout

    commits_result = run_git(
        "log", f"upstream/{target_branch}..origin/{source_branch}",
        "--pretty=format:%H", "--reverse", check=False
    )
    branch_commits = (
        commits_result.stdout.strip().split("\n")
        if commits_result.stdout.strip() else []
    )

    print("Generating PR description...")
    auto_title, auto_body = generate_pr_content(diff, branch_commits)

    old_title = pr_data.get("title", "")
    if old_title and not old_title.startswith("Update") and len(old_title) > len(auto_title):
        auto_title = old_title

    if not title:
        title = _preview_and_confirm(
            "PR title", auto_title,
            skip_flag_hint='--title "<approved title>"',
        )
    if not body:
        body = _preview_and_confirm(
            "PR body", auto_body,
            skip_flag_hint='--body "<approved multi-line body>"',
            long_text=True,
        )

    print("Updating PR description...")
    status, result = update_pr_description(
        env.upstream_owner, env.upstream_repo, pr_number, env.token, title, body
    )

    if status not in [200, 201]:
        raise AutoGitError(f"PR update failed: {result}")

    return {
        "pr_number": pr_number,
        "title": title,
        "url": f"https://gitcode.com/{env.upstream_owner}/{env.upstream_repo}/pull/{pr_number}"
    }


# ============================================================================
# Command: squash
# ============================================================================

def cmd_squash(pr_ref: str, message: Optional[str] = None) -> Dict[str, Any]:
    """Squash multiple commits in a PR into one.

    Args:
        pr_ref: PR reference string.
        message: Optional commit message after squash.

    Returns:
        Dict with keys: pr_number, branch, old_commits, new_commits, sha, url.
    """
    env = check_env()

    parsed = parse_pr_ref(pr_ref)
    if not parsed:
        raise AutoGitError(f"Cannot parse PR reference: {pr_ref}")

    _, _, pr_number = parsed

    print(f"Fetching PR #{pr_number} info...")

    status, pr_data = get_pr_info(env.upstream_owner, env.upstream_repo, pr_number, env.token)
    if status != 200:
        raise AutoGitError(f"Cannot get PR #{pr_number} info: {pr_data}")

    head_info = pr_data.get("head", {})
    base_info = pr_data.get("base", {})
    source_branch = head_info.get("ref")
    target_branch = base_info.get("ref", env.default_branch)
    head_repo = head_info.get("repo", {})
    head_owner = (head_repo.get("namespace", {}).get("path") or
                  head_repo.get("owner", {}).get("login"))

    if not source_branch:
        raise AutoGitError(f"Cannot get source branch for PR #{pr_number}")

    if head_owner and head_owner != env.origin_owner:
        raise AutoGitError(
            f"PR #{pr_number} does not belong to your fork ({head_owner} != {env.origin_owner})"
        )

    pr_commits = pr_data.get("commits") or get_pr_stats(
        env.upstream_owner, env.upstream_repo, pr_number, env.token)["commits"]
    if pr_commits <= 1:
        raise AutoGitError(f"PR #{pr_number} has only {pr_commits} commit(s), squash not needed")

    print(f"Branch: {source_branch} -> {target_branch}")
    print(f"Currently has {pr_commits} commits, will squash into 1")

    current_branch = get_current_branch()

    try:
        run_git("fetch", "origin", source_branch)
        run_git("fetch", "upstream", target_branch)

        run_git("checkout", source_branch)
        run_git("reset", "--hard", f"origin/{source_branch}")

        if not message:
            first_commit_msg = run_git(
                "log", "-1", "--pretty=format:%s", f"origin/{source_branch}"
            ).stdout.strip()
            message = first_commit_msg or f"Squash {pr_commits} commits"

        print(f"Rebasing onto upstream/{target_branch}...")
        result = run_git("rebase", f"upstream/{target_branch}", check=False)
        if result.returncode != 0:
            run_git("rebase", "--abort", check=False)
            raise AutoGitError(
                f"Rebase failed, please resolve conflicts manually and retry.\n"
                f"  git checkout {source_branch}\n"
                f"  git rebase upstream/{target_branch}\n"
                f"  # Resolve conflicts, then:\n"
                f"  git push -f origin {source_branch}"
            )

        print("Squashing commits...")
        run_git("reset", "--soft", f"upstream/{target_branch}")

        err = validate_commit_message(message)
        if err:
            run_git("checkout", current_branch, check=False)
            raise AutoGitError(err)
        run_git("commit", "-m", message)
        sha = run_git("rev-parse", "HEAD").stdout.strip()
        print(f"Squash complete: {sha[:8]}")

        print("Pushing to remote...")
        run_git("push", "-f", "origin", source_branch)

        print(f"Switching back to {current_branch}...")
        run_git("checkout", current_branch)

        return {
            "pr_number": pr_number,
            "branch": source_branch,
            "old_commits": pr_commits,
            "new_commits": 1,
            "sha": sha,
            "url": f"https://gitcode.com/{env.upstream_owner}/{env.upstream_repo}/pull/{pr_number}"
        }

    except Exception:
        run_git("checkout", current_branch, check=False)
        raise


# ============================================================================
# Help Text
# ============================================================================

HELP_TEXT = """AutoGit - Automated Git Workflow for GitCode

Commands:
  commit [-m MSG]                    Commit and push (lint via pre-commit hook)
  check                              Run lint checks (installs pylint/markdownlint if needed)
  test                               Run test stage (pytest tests/ut, full)
  pylint-review [--base REF]         Run pylint on changed .py (review-PR stage)
  pr [--base B] [--reviewer R] [--ut S] [--st S]  Create a PR
                                     UT/ST scope: skip | changed | full
  pr --to #N [--amend|--no-rebase|--ut S]  Append to existing PR
  status #N                          View PR status
  update #N                          Regenerate PR description
  squash #N [-m MSG]                 Squash commits in a PR

Examples:
  autogit commit -m "feat: add X"
  autogit pr --reviewer zhangsan
  autogit pr --to #160 --amend
  autogit squash #160

Prerequisites:
  export GITCODE_TOKEN=<your-token>
  git remote add upstream <main-repo-URL>

Detailed help: autogit <command> --help
"""


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """Parse arguments and dispatch to the appropriate command handler."""
    parser = argparse.ArgumentParser(
        description="Automated Git workflow tool for GitCode",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="commands")

    commit_parser = subparsers.add_parser("commit", help="Auto commit and push")
    commit_parser.add_argument("-m", "--message", type=str, help="commit message")

    subparsers.add_parser(
        "check",
        help="Run lint checks and auto-install missing pylint/markdownlint if configured"
    )
    subparsers.add_parser(
        "test",
        help="Run test stage: pytest only"
    )
    pylint_review_parser = subparsers.add_parser(
        "pylint-review",
        help="Run pylint on changed .py files (for review-PR stage)"
    )
    pylint_review_parser.add_argument(
        "--base",
        type=str,
        default=None,
        help="Base ref to diff against (default: upstream default branch)",
    )

    pr_parser = subparsers.add_parser("pr", help="Create or append to a PR")
    pr_parser.add_argument("--to", dest="append_to", type=str, help="Append to existing PR")
    pr_parser.add_argument("--amend", action="store_true", help="Amend the last commit")
    pr_parser.add_argument("--no-rebase", action="store_true", help="Skip rebase")
    pr_parser.add_argument("--squash", action="store_true", help="Squash all commits into one")
    pr_parser.add_argument("--base", type=str, help="Target branch")
    pr_parser.add_argument("--reviewer", type=str, help="Reviewers (comma-separated)")
    pr_parser.add_argument(
        "--ut", dest="ut", choices=list(_GATE_CHOICES), default=None,
        help="UT gate scope: skip / changed (default) / full "
             "(default: ask on tty, error on non-tty)"
    )
    pr_parser.add_argument(
        "--st", dest="st", choices=list(_GATE_CHOICES), default=None,
        help="ST gate scope: skip (default) / changed / full "
             "(needs multi-card; default: ask on tty, error on non-tty)"
    )
    pr_parser.add_argument(
        "--analyze-only", action="store_true",
        help="Output structured JSON analysis for LLM-based description generation"
    )
    pr_parser.add_argument("--title", type=str, help="PR title (skip auto-generation)")
    pr_parser.add_argument("--body", type=str, help="PR body (skip auto-generation)")
    pr_parser.add_argument("-m", "--message", type=str, help="commit message")

    status_parser = subparsers.add_parser("status", help="View PR status")
    status_parser.add_argument("pr_ref", type=str, help="PR number or URL")

    update_parser = subparsers.add_parser("update", help="Regenerate PR description")
    update_parser.add_argument("pr_ref", type=str, help="PR number or URL")
    update_parser.add_argument("--title", type=str, help="PR title (skip auto-generation)")
    update_parser.add_argument("--body", type=str, help="PR body (skip auto-generation)")

    squash_parser = subparsers.add_parser("squash", help="Squash commits in a PR")
    squash_parser.add_argument("pr_ref", type=str, help="PR number or URL")
    squash_parser.add_argument("-m", "--message", type=str, help="Commit message after squash")

    args = parser.parse_args()

    if not args.command:
        print(HELP_TEXT)
        sys.exit(0)

    try:
        _dispatch(args)
    except AutoGitError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {' '.join(e.cmd)}")
        if e.stderr:
            print(f"   {e.stderr.strip()}")
        sys.exit(1)


def _dispatch(args: argparse.Namespace) -> None:
    """Dispatch to the corresponding handler based on the command.

    Args:
        args: Parsed argparse namespace.
    """
    if args.command == "check":
        cmd_check()

    elif args.command == "test":
        cmd_test()

    elif args.command == "pylint-review":
        cmd_pylint_review(base_ref=getattr(args, "base", None))
        print()
        print("=" * 60)
        print("Pylint review report above — include in Code Quality section.")
        print("=" * 60)

    elif args.command == "commit":
        result = cmd_commit(args.message)
        print()
        print("=" * 60)
        print("Commit complete!")
        print(f"   SHA: {result['sha'][:8]}")
        print(f"   Branch: {result['branch']}")
        print("=" * 60)

    elif args.command == "pr":
        _dispatch_pr(args)

    elif args.command == "status":
        print(cmd_status(args.pr_ref))

    elif args.command == "update":
        result = cmd_update(args.pr_ref, title=args.title, body=args.body)
        print()
        print("=" * 60)
        print(f"PR #{result['pr_number']} description updated!")
        print(f"   {result['url']}")
        print(f"   Title: {result['title'][:50]}")
        print("=" * 60)

    elif args.command == "squash":
        result = cmd_squash(args.pr_ref, args.message)
        print()
        print("=" * 60)
        print(f"PR #{result['pr_number']} commits squashed!")
        print(f"   {result['url']}")
        print(f"   {result['old_commits']} commits -> 1 commit")
        print(f"   SHA: {result['sha'][:8]}")
        print("=" * 60)


def _dispatch_pr(args: argparse.Namespace) -> None:
    """Handle the pr subcommand.

    Args:
        args: Parsed argparse namespace.
    """
    if args.append_to:
        parsed = parse_pr_ref(args.append_to)
        if not parsed:
            print(f"Cannot parse: {args.append_to}")
            sys.exit(1)
        _, _, pr_number = parsed
        result = cmd_pr_append(
            pr_number,
            amend=args.amend,
            no_rebase=args.no_rebase,
            message=args.message,
            ut=args.ut,
        )
        print()
        print("=" * 60)
        action = "Amended to" if result.get('amend') else "Appended to"
        print(f"{action} PR #{result['pr_number']}!")
        print(f"   {result['url']}")
        print(f"   Branch: {result['branch']}")
        print("=" * 60)
    else:
        analyze_only = getattr(args, 'analyze_only', False)
        result = cmd_pr(
            base=args.base,
            reviewer=args.reviewer,
            squash=args.squash,
            ut="skip" if analyze_only else args.ut,
            st="skip" if analyze_only else args.st,
            analyze_only=analyze_only,
            title=getattr(args, 'title', None),
            body=getattr(args, 'body', None),
        )
        if analyze_only:
            print(result["analysis"])
            return
        print()
        print("=" * 60)
        print("PR created successfully!")
        print(f"   {result['url']}")
        print(f"   Branch: {result['branch']}")
        print(f"   Commits: {result['commits']}")
        print("=" * 60)
