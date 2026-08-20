#!/usr/bin/env bash
# Example: convert a local JSONL file to Megatron .bin/.idx.
#
#   bash hyper_models/components/datasets/tools/demo_local_jsonl.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${PROJECT_ROOT}"

python -m hyper_models.components.datasets.tools.offline_preparation \
    --dataset-name-or-path ./download_datasets/wikitext-103-raw-v1/wikitext-103-raw-v1-train.jsonl \
    --output-prefix ./offline_datasets/hyper_models_preprocess/wikitext-103-train-gpt2 \
    --json-keys text \
    --tokenizer-name-or-path gpt2 \
    --workers 8 \
    --append-eod true
