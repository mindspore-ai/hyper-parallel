#!/usr/bin/env bash
# Example: download a Hugging Face dataset and write fixed-length indexed data.
#
#   bash examples/data/demo_huggingface_offline_pad.sh
#
# Each stored document contains 4096 + 1 tokens. Each partition packs its own
# continuous token stream, and its incomplete final document is dropped.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${PROJECT_ROOT}"

python -m hyper_parallel.data.tools.huggingface_offline \
    --dataset Salesforce/wikitext \
    --dataset-subset wikitext-103-raw-v1 \
    --dataset-split train \
    --json-keys text \
    --output-prefix ./offline_datasets/pad_huggingface/output \
    --tokenizer gpt2 \
    --workers 64 \
    --partitions 16 \
    --keep-sequential-samples \
    --append-eod true \
    --pack-to-seq-len 4096
