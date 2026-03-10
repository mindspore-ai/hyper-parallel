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
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
from pr_content import generate_pr_content  # pylint: disable=wrong-import-position
from lint_check import run_checks  # pylint: disable=wrong-import-position


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


def _run_lint_checks_on_staged() -> None:
    """Run lint checks on staged files; unstage and raise on failure."""
    staged_output = run_git("diff", "--cached", "--name-only").stdout.strip()
    staged_files = staged_output.split("\n") if staged_output else []
    if not staged_files:
        return
    print("Running lint checks...")
    passed, report = run_checks(staged_files)
    print(report)
    if not passed:
        run_git("reset", check=False)
        raise AutoGitError("Lint checks failed, staging reverted. Please fix and retry.")


# ============================================================================
# Command: commit
# ============================================================================

def cmd_commit(message: Optional[str] = None,
               no_check: bool = False) -> Dict[str, Any]:
    """Commit and push to origin.

    Args:
        message: Optional commit message.
        no_check: Skip lint checks if True.

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

    if not no_check:
        _run_lint_checks_on_staged()

    if not message:
        changed_files = run_git("diff", "--cached", "--name-only").stdout.strip().split("\n")
        if len(changed_files) == 1:
            message = f"Update {changed_files[0].split('/')[-1]}"
        else:
            message = f"Update {len(changed_files)} files"

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
    run_git("commit", "-m", msg or "Update code")
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


def cmd_pr(base: Optional[str] = None, reviewer: Optional[str] = None,
           squash: bool = False) -> Dict[str, Any]:
    """Create a PR with safe Git workflow.

    Args:
        base: Target branch (defaults to upstream default branch).
        reviewer: Comma-separated reviewer login names.
        squash: Whether to squash all commits.

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
    title, body = generate_pr_content(diff, commits)

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
                   base_ref: Optional[str] = None,
                   no_check: bool = False) -> int:
    """Execute the append commit operation; return the new total commit count.

    Args:
        amend: Whether to amend the previous commit.
        message: Optional commit message.
        pr_commits: Current PR commit count.
        base_ref: Optional base ref for cosmetic filtering.
        no_check: Skip lint checks if True.

    Returns:
        New total commit count.
    """
    if amend:
        print("Merging into previous commit (amend)...")
        _stage_and_filter_cosmetic(base_ref)
        if not no_check:
            _run_lint_checks_on_staged()
        if message:
            run_git("commit", "--amend", "-m", message)
        else:
            run_git("commit", "--amend", "--no-edit")
        return pr_commits

    print("Creating new commit...")
    _stage_and_filter_cosmetic(base_ref)
    if not no_check:
        _run_lint_checks_on_staged()
    if not message:
        changed = run_git("diff", "--cached", "--name-only").stdout.strip().split("\n")
        if len(changed) > 1:
            message = f"Update {len(changed)} files"
        else:
            message = f"Update {changed[0].split('/')[-1]}"
    run_git("commit", "-m", message)
    sha = run_git("rev-parse", "HEAD").stdout.strip()
    print(f"New commit: {sha[:8]}")
    return pr_commits + 1


def cmd_pr_append(pr_number: int, amend: bool = False,
                  no_rebase: bool = False, message: Optional[str] = None,
                  no_check: bool = False) -> Dict[str, Any]:
    """Append a commit to an existing PR.

    Args:
        pr_number: Pull request number.
        amend: Whether to amend the last commit.
        no_rebase: Skip rebase if True.
        message: Optional commit message.
        no_check: Skip lint checks if True.

    Returns:
        Dict with keys: url, branch, pr_number, amend, commits.
    """
    env = check_env()

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
                                     base_ref=pr_base_ref,
                                     no_check=no_check)

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

def cmd_update(pr_ref: str) -> Dict[str, Any]:
    """Regenerate and update a PR description.

    Args:
        pr_ref: PR reference string.

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
    new_title, new_body = generate_pr_content(diff, branch_commits)

    old_title = pr_data.get("title", "")
    if old_title and not old_title.startswith("Update") and len(old_title) > len(new_title):
        new_title = old_title

    print("Updating PR description...")
    status, result = update_pr_description(
        env.upstream_owner, env.upstream_repo, pr_number, env.token, new_title, new_body
    )

    if status not in [200, 201]:
        raise AutoGitError(f"PR update failed: {result}")

    return {
        "pr_number": pr_number,
        "title": new_title,
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
  commit [-m MSG] [--no-check]       Commit and push (runs lint checks by default)
  check                              Run lint checks (no commit)
  pr [--base B] [--reviewer R]       Create a PR
  pr --to #N [--amend|--no-rebase|--no-check]  Append to existing PR
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
    commit_parser.add_argument(
        "--no-check", action="store_true",
        help="Skip lint checks (pylint/lizard/codespell/markdownlint)"
    )

    subparsers.add_parser("check", help="Run lint checks (no commit)")

    pr_parser = subparsers.add_parser("pr", help="Create or append to a PR")
    pr_parser.add_argument("--to", dest="append_to", type=str, help="Append to existing PR")
    pr_parser.add_argument("--amend", action="store_true", help="Amend the last commit")
    pr_parser.add_argument("--no-rebase", action="store_true", help="Skip rebase")
    pr_parser.add_argument(
        "--no-check", action="store_true",
        help="Skip lint checks (pylint/lizard/codespell/markdownlint)"
    )
    pr_parser.add_argument("--squash", action="store_true", help="Squash all commits into one")
    pr_parser.add_argument("--base", type=str, help="Target branch")
    pr_parser.add_argument("--reviewer", type=str, help="Reviewers (comma-separated)")
    pr_parser.add_argument("-m", "--message", type=str, help="commit message")

    status_parser = subparsers.add_parser("status", help="View PR status")
    status_parser.add_argument("pr_ref", type=str, help="PR number or URL")

    update_parser = subparsers.add_parser("update", help="Regenerate PR description")
    update_parser.add_argument("pr_ref", type=str, help="PR number or URL")

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

    elif args.command == "commit":
        result = cmd_commit(args.message, no_check=args.no_check)
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
        result = cmd_update(args.pr_ref)
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
            no_check=args.no_check
        )
        print()
        print("=" * 60)
        action = "Amended to" if result.get('amend') else "Appended to"
        print(f"{action} PR #{result['pr_number']}!")
        print(f"   {result['url']}")
        print(f"   Branch: {result['branch']}")
        print("=" * 60)
    else:
        result = cmd_pr(
            base=args.base,
            reviewer=args.reviewer,
            squash=args.squash
        )
        print()
        print("=" * 60)
        print("PR created successfully!")
        print(f"   {result['url']}")
        print(f"   Branch: {result['branch']}")
        print(f"   Commits: {result['commits']}")
        print("=" * 60)
