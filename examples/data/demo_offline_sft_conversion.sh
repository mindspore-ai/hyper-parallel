#!/usr/bin/env bash
# Convert the bundled Instruction and ShareGPT records to indexed datasets.
#
#   TOKENIZER_NAME_OR_PATH=/path/to/tokenizer bash examples/data/demo_offline_sft_conversion.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-Qwen/Qwen3-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/offline_datasets/sft_demo}"

cd "${PROJECT_ROOT}"

echo "[offline] tokenizer=${TOKENIZER_NAME_OR_PATH}"
echo "[offline] Instruction/Alpaca -> normalized JSONL -> .bin/.idx"
python -m hyper_parallel.data.tools.offline_preparation \
    --dataset-name-or-path "${SCRIPT_DIR}/offline_instruction_demo.jsonl" \
    --output-prefix "${OUTPUT_DIR}/instruction" \
    --json-keys text \
    --text-template $'Instruction: {instruction}\nInput: {input}\nOutput: {output}' \
    --tokenizer-name-or-path "${TOKENIZER_NAME_OR_PATH}" \
    --workers 1 \
    --append-eod true

echo "[offline] ShareGPT -> role normalization -> chat template -> .bin/.idx"
python -m hyper_parallel.data.tools.offline_preparation \
    --dataset-name-or-path "${SCRIPT_DIR}/offline_sharegpt_demo.jsonl" \
    --output-prefix "${OUTPUT_DIR}/sharegpt" \
    --json-keys text \
    --conversation-key conversations \
    --role-key from \
    --content-key value \
    --tokenizer-name-or-path "${TOKENIZER_NAME_OR_PATH}" \
    --workers 1 \
    --append-eod true

echo "[offline] generated files:"
find "${OUTPUT_DIR}" -maxdepth 1 -type f \( -name '*.bin' -o -name '*.idx' -o -name '*_normalized.jsonl' \) -print | sort
