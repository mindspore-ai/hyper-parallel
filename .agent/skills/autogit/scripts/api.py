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
"""Token management and GitCode HTTP API operations for AutoGit."""

import json
import os
import subprocess
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from models import CREDENTIAL_HOST, GITCODE_API_BASE


# ============================================================================
# Token Management
# ============================================================================

def get_token() -> Optional[str]:
    """Get the GitCode API token from environment or git credentials.

    Checks ``GITCODE_TOKEN`` environment variable first, then falls back
    to ``git credential fill``.

    Returns:
        Token string, or None if not found.
    """
    token = os.environ.get("GITCODE_TOKEN")
    if token:
        return token

    try:
        proc = subprocess.run(
            ["git", "credential", "fill"],
            input=f"protocol=https\nhost={CREDENTIAL_HOST}\n",
            capture_output=True, text=True, timeout=10,
            check=False
        )
        if proc.returncode == 0:
            for line in proc.stdout.strip().split("\n"):
                if line.startswith("password="):
                    return line.split("=", 1)[1]
    except Exception:  # pylint: disable=broad-except
        pass
    return None


# ============================================================================
# GitCode API
# ============================================================================

def api_request(method: str, endpoint: str, token: str,
                data: Optional[Dict] = None) -> Tuple[int, Any]:
    """Send an HTTP request to the GitCode API.

    Args:
        method: HTTP method (GET, POST, PATCH, etc.).
        endpoint: API endpoint path (appended to GITCODE_API_BASE).
        token: Bearer token for authorization.
        data: Optional JSON-serializable request body.

    Returns:
        Tuple of (HTTP status code, parsed JSON response).
    """
    url = f"{GITCODE_API_BASE}{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:  # pylint: disable=broad-except
            return e.code, {"error": str(e)}
    except Exception as e:  # pylint: disable=broad-except
        return 0, {"error": str(e)}


def get_pr_info(owner: str, repo: str, pr_number: int,
                token: str) -> Tuple[int, Any]:
    """Get PR information from GitCode.

    Args:
        owner: Repository owner.
        repo: Repository name.
        pr_number: Pull request number.
        token: API token.

    Returns:
        Tuple of (status_code, response_data).
    """
    return api_request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}", token)


def get_pr_commits(owner: str, repo: str, pr_number: int,
                   token: str) -> Tuple[int, Any]:
    """Get the list of commits for a PR.

    Args:
        owner: Repository owner.
        repo: Repository name.
        pr_number: Pull request number.
        token: API token.

    Returns:
        Tuple of (status_code, response_data).
    """
    return api_request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}/commits", token)


def get_pr_files(owner: str, repo: str, pr_number: int,
                 token: str) -> Tuple[int, Any]:
    """Get the list of changed files for a PR.

    Args:
        owner: Repository owner.
        repo: Repository name.
        pr_number: Pull request number.
        token: API token.

    Returns:
        Tuple of (status_code, response_data).
    """
    return api_request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}/files", token)


def get_pr_stats(owner: str, repo: str, pr_number: int,
                 token: str) -> Dict[str, int]:
    """Get PR statistics via commits/files sub-endpoints.

    GitCode PR detail does not include commits/additions/deletions/changed_files,
    so this fetches them from sub-endpoints.

    Args:
        owner: Repository owner.
        repo: Repository name.
        pr_number: Pull request number.
        token: API token.

    Returns:
        Dict with keys: commits, changed_files, additions, deletions.
    """
    stats: Dict[str, int] = {"commits": 0, "changed_files": 0, "additions": 0, "deletions": 0}

    status, data = get_pr_commits(owner, repo, pr_number, token)
    if status == 200 and isinstance(data, list):
        stats["commits"] = len(data)

    status, data = get_pr_files(owner, repo, pr_number, token)
    if status == 200 and isinstance(data, list):
        stats["changed_files"] = len(data)
        for f in data:
            stats["additions"] += f.get("additions", 0)
            stats["deletions"] += f.get("deletions", 0)

    return stats


def add_reviewers(owner: str, repo: str, pr_number: int, token: str,
                  reviewers: List[str]) -> Tuple[int, Any]:
    """Add reviewers to a PR.

    Args:
        owner: Repository owner.
        repo: Repository name.
        pr_number: Pull request number.
        token: API token.
        reviewers: List of reviewer login names.

    Returns:
        Tuple of (status_code, response_data).
    """
    data = {"reviewers": reviewers}
    return api_request(
        "POST",
        f"/repos/{owner}/{repo}/pulls/{pr_number}/requested_reviewers",
        token, data
    )


def get_pr_status_display(owner: str, repo: str, pr_number: int,
                          token: str) -> str:
    """Get a formatted display of PR status.

    Args:
        owner: Repository owner.
        repo: Repository name.
        pr_number: Pull request number.
        token: API token.

    Returns:
        Formatted status string for display.
    """
    status_code, pr_data = get_pr_info(owner, repo, pr_number, token)
    if status_code != 200:
        return f"Cannot get PR #{pr_number} info: {pr_data}"

    title = pr_data.get("title", "N/A")
    state = pr_data.get("state", "unknown")
    merged = pr_data.get("merged", False)
    draft = pr_data.get("draft", False)
    created = pr_data.get("created_at", "")[:10]
    updated = pr_data.get("updated_at", "")[:10]

    head_info = pr_data.get("head", {})
    base_info = pr_data.get("base", {})
    head_ref = head_info.get("ref", "?")
    base_ref = base_info.get("ref", "?")

    user = pr_data.get("user", {})
    author = user.get("login", "unknown")

    stats = get_pr_stats(owner, repo, pr_number, token)
    additions = pr_data.get("additions") or stats["additions"]
    deletions = pr_data.get("deletions") or stats["deletions"]
    changed_files = pr_data.get("changed_files") or stats["changed_files"]
    commits = pr_data.get("commits") or stats["commits"]

    if merged:
        status_icon = "Merged"
    elif state == "closed":
        status_icon = "Closed"
    elif draft:
        status_icon = "Draft"
    else:
        status_icon = "Open"

    reviewers_data = pr_data.get("requested_reviewers", [])
    reviewer_names = [r.get("login", "?") for r in reviewers_data]

    output = f"""
PR #{pr_number}: {title[:50]}

Status: {status_icon}
Author: {author}
Branch: {head_ref} -> {base_ref}
Created: {created}  Updated: {updated}

Stats: +{additions} -{deletions} | {changed_files} files | {commits} commits
"""
    if reviewer_names:
        output += f"Reviewers: {', '.join(reviewer_names)}\n"

    output += f"\nhttps://gitcode.com/{owner}/{repo}/pull/{pr_number}\n"
    return output


def create_pr(owner: str, repo: str, token: str, title: str, body: str,
              head: str, base: str, fork_owner: str,
              fork_repo: str) -> Tuple[int, Any]:
    """Create a PR on GitCode.

    Args:
        owner: Upstream repository owner.
        repo: Upstream repository name.
        token: API token.
        title: PR title.
        body: PR description body.
        head: Head branch reference (e.g. 'fork_owner:branch').
        base: Target branch name.
        fork_owner: Fork repository owner.
        fork_repo: Fork repository name.

    Returns:
        Tuple of (status_code, response_data).
    """
    data = {
        "title": title,
        "body": body,
        "head": head,
        "base": base,
        "fork_path": f"{fork_owner}/{fork_repo}"
    }
    status, result = api_request("POST", f"/repos/{owner}/{repo}/pulls", token, data)

    if status in [200, 201] and isinstance(result, dict):
        pr_number = result.get("number") or result.get("iid")
        if pr_number:
            result["html_url"] = f"https://gitcode.com/{owner}/{repo}/pull/{pr_number}"
    return status, result


def update_pr_description(owner: str, repo: str, pr_number: int, token: str,
                          title: str, body: str) -> Tuple[int, Any]:
    """Update a PR title and description.

    Args:
        owner: Repository owner.
        repo: Repository name.
        pr_number: Pull request number.
        token: API token.
        title: New PR title.
        body: New PR description body.

    Returns:
        Tuple of (status_code, response_data).
    """
    data = {"title": title, "body": body}
    return api_request("PATCH", f"/repos/{owner}/{repo}/pulls/{pr_number}", token, data)
