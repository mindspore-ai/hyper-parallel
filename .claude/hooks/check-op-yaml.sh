#!/bin/bash
# PostToolUse hook: remind to check related files when editing distributed code

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null)

if [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Only check Python and YAML files
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
    */platform/torch/*.py)
        echo "REMINDER: You modified PyTorch platform code."
        echo "Check if platform/mindspore/ needs a corresponding change."
        ;;
    */platform/mindspore/*.py)
        echo "REMINDER: You modified MindSpore platform code."
        echo "Check if platform/torch/ needs a corresponding change."
        ;;
    */fully_shard/*.py)
        echo "REMINDER: You modified FSDP code."
        echo "Check memory lifecycle: storage resize_(0), buffer cleanup, grad nulling."
        ;;
    */pipeline_parallel/*.py)
        echo "REMINDER: You modified pipeline parallelism code."
        echo "Check buffer cleanup: _clear_recv_buffer(), clear_cache() after each micro-batch."
        ;;
    */activation_checkpoint/*.py)
        echo "REMINDER: You modified activation checkpoint/swap code."
        echo "Check swap lifecycle: wait_offload() frees device, wait_load() frees CPU."
        ;;
esac
