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
"""Commit message validation — reject AI/tool attribution and third-party references.

Blocks commit messages containing:
1. AI/IDE attribution trailers (Made-with: Cursor, Co-authored-by: Claude, etc.)
2. Third-party tool/service names — commit messages should only describe business changes
"""

import re
from typing import Optional

# Third-party tool/service names that should not appear in commit messages.
# Commit messages should only describe business-side changes.
_THIRD_PARTY_NAMES = [
    "cursor", "copilot", "chatgpt", "claude", "gemini", "gpt-4", "gpt-3",
    "openai", "anthropic", "codewhisperer", "tabnine", "codeium", "windsurf",
    "github copilot", "amazon q",
]

_THIRD_PARTY_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(name) for name in _THIRD_PARTY_NAMES) + r")\b",
    re.IGNORECASE,
)

# Attribution trailer patterns — matched case-insensitively against each line.
_FORBIDDEN_PATTERNS = [
    (re.compile(r"made[- ]?with\s*:\s*cursor", re.IGNORECASE),
     "Made-with: Cursor"),
    (re.compile(r"co[- ]?authored[- ]?by\s*:.*cursor", re.IGNORECASE),
     "Co-authored-by: Cursor"),
    (re.compile(r"co[- ]?authored[- ]?by\s*:.*claude", re.IGNORECASE),
     "Co-Authored-By: Claude (Claude Code)"),
    (re.compile(r"generated[- ]?by\s*:.*cursor", re.IGNORECASE),
     "Generated-by: Cursor"),
    (re.compile(r"^made\s+with\s+cursor\s*$", re.IGNORECASE),
     "Made with Cursor"),
]


def validate_commit_message(message: str) -> Optional[str]:
    """Validate commit message; return error string if invalid, else None.

    Rejects messages containing:
    1. AI/IDE attribution trailers (e.g. Made-with: Cursor)
    2. Third-party tool/service names (e.g. Copilot, ChatGPT, Claude)

    Commit messages should only describe business-side changes.

    Args:
        message: Full commit message (subject + body).

    Returns:
        None if valid; otherwise an error message string describing the
        forbidden content.
    """
    if not message or not message.strip():
        return None

    lines = message.strip().split("\n")
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("#"):
            continue
        for pattern, label in _FORBIDDEN_PATTERNS:
            if pattern.search(line_stripped):
                return (
                    f"Commit message contains forbidden attribution: '{label}'.\n"
                    "Remove AI/IDE trailers from commit messages. "
                    "Disable Cursor Settings > Agent > Attribution; "
                    "or Claude Code attribution in extension settings."
                )
        match = _THIRD_PARTY_PATTERN.search(line_stripped)
        if match:
            return (
                f"Commit message contains third-party reference: '{match.group()}'.\n"
                "Commit messages should only describe business-side changes. "
                "Remove references to external tools/services."
            )
    return None


def main() -> int:
    """CLI entry: read commit message from stdin, validate, exit 1 if invalid.

    Used by git commit-msg hook. Git passes the message file path as argv[1].
    """
    import sys  # pylint: disable=import-outside-toplevel

    if len(sys.argv) < 2:
        print("Usage: commit_msg_check.py <path-to-commit-msg-file>", file=sys.stderr)
        return 1

    path = sys.argv[1]
    try:
        with open(path, "r", encoding="utf-8") as fp:
            content = fp.read()
    except OSError as e:
        print(f"Cannot read commit message: {e}", file=sys.stderr)
        return 1

    err = validate_commit_message(content)
    if err:
        print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
