#!/bin/bash
# PostToolUse hook: auto-fix basic code-style issues after file writes.

set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null)

if [ -z "$FILE_PATH" ] || [ ! -f "$FILE_PATH" ]; then
    exit 0
fi

case "$FILE_PATH" in
    *.py|*.sh|*.bash|*.md|*.yaml|*.yml|*.json|*.toml|*.cmake|CMakeLists.txt|*.c|*.cc|*.cpp|*.h|*.hpp) ;;
    *) exit 0 ;;
esac

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
SCRIPT_PATH="$REPO_ROOT/.claude/skills/autogit/scripts/code_style_guard.py"

if [ ! -f "$SCRIPT_PATH" ]; then
    exit 0
fi

python3 "$SCRIPT_PATH" --fix "$FILE_PATH"
