#!/usr/bin/env bash
# Example: download a Hugging Face dataset and convert to Megatron .bin/.idx.
#
#   bash hyper_models/components/datasets/tools/demo_huggingface_offline.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${PROJECT_ROOT}"

python -m hyper_models.components.datasets.tools.huggingface_offline \
    --dataset Salesforce/wikitext \
    --dataset-subset wikitext-103-raw-v1 \
    --dataset-split train \
    --json-keys text \
    --output-prefix ./offline_datasets/hyper_models_preprocess/wikitext-103-train-gpt2 \
    --tokenizer gpt2 \
    --workers 8 \
    --append-eod
