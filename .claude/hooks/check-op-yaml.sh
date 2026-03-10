#!/bin/bash
# PostToolUse hook: remind to update YAML registration when editing distributed ops

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | grep -o '"file_path":"[^"]*"' | head -1 | cut -d'"' -f4)

if [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Only check Python and YAML files in relevant directories
case "$FILE_PATH" in
    *.py|*.yaml) ;;  # continue
    *) exit 0 ;;     # skip non-code files
esac

case "$FILE_PATH" in
    */core/shard/ops/parallel_*.py)
        echo "REMINDER: You modified a distributed op implementation."
        echo "Check if the YAML registration in core/shard/ops/yaml/ needs updating."
        ;;
    */core/shard/ops/yaml/*.yaml)
        echo "REMINDER: You modified an op YAML registration."
        echo "Check if the Python implementation in core/shard/ops/parallel_*.py is consistent."
        ;;
    */platform/torch/*.py|*/platform/mindspore/*.py)
        echo "REMINDER: You modified platform-specific code."
        echo "Check if the other platform (torch/mindspore) needs a corresponding change."
        ;;
esac
