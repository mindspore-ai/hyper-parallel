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
"""Commit message validation — reject AI-assistant attribution trailers.

Commit messages should describe business-side changes only. Auto-generated
attribution trailers are detected by their structure (robot-emoji markers,
tool links, bot co-author no-reply addresses) rather than by hard-coding any
vendor/tool brand name.
"""

import re
from typing import Optional

# Robot-emoji marker used by AI tools in "<emoji> Generated with ..." trailers.
_ROBOT_EMOJI = "\U0001F916"

# Attribution trailers are matched by structure, not by brand name, so this
# source never embeds any vendor/tool name to recognise them. Human co-authors
# using a github.com no-reply address are intentionally allowed.
_FORBIDDEN_PATTERNS = [
    (re.compile(re.escape(_ROBOT_EMOJI)),
     "robot-emoji attribution trailer"),
    (re.compile(r"^\s*made[- ]?with\s*:", re.IGNORECASE),
     "Made-with attribution trailer"),
    (re.compile(r"^\s*generated[- ]?by\s*:", re.IGNORECASE),
     "Generated-by attribution trailer"),
    (re.compile(r"(?:made|generated|created|written)[- ]?(?:with|by)\s+"
                r"\[[^\]]+\]\(https?://", re.IGNORECASE),
     "tool-link attribution trailer"),
    (re.compile(r"co[- ]?authored[- ]?by\s*:\s*.*<[^>]*noreply@"
                r"(?!users\.noreply\.github\.com)[^>]*>", re.IGNORECASE),
     "bot co-author attribution trailer"),
]


def validate_commit_message(message: str) -> Optional[str]:
    """Validate commit message; return error string if invalid, else None.

    Rejects auto-generated AI-assistant attribution trailers (robot-emoji
    markers, tool links, and bot co-author no-reply addresses). Human
    co-authors using a github.com no-reply address are allowed.

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
                    "Remove AI-assistant attribution trailers from commit "
                    "messages; describe business-side changes only. Disable "
                    "attribution in your editor/agent settings."
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
