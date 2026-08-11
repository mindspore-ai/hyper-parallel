#!/usr/bin/env bash
# Example: convert a directory of local JSONL files into fixed-length indexed data.
#
#   bash hyper_models/components/data/tools/demo_local_jsonl_pad.sh
#
# The input directory must contain JSON/JSONL files with one JSON object per line
# and a string field named "text". Files are consumed in sorted path order.
# Each stored document contains 4096 + 1 tokens. Each partition packs its own
# continuous token stream, and its incomplete final document is dropped.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${PROJECT_ROOT}"

python -m hyper_models.components.data.tools.offline_preparation \
    --dataset-name-or-path ./download_datasets/wikitext-103-raw-v1 \
    --output-prefix ./offline_datasets/pad_local_jsonl/output \
    --json-keys text \
    --tokenizer-name-or-path gpt2 \
    --workers 8 \
    --partitions 2 \
    --keep-sequential-samples \
    --append-eod \
    --pad-to-seq-len 4096
