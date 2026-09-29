#!/usr/bin/env bash
set -euo pipefail

# Validate an external Megatron prefix without rewriting it:
#   bash examples/data/demo_megatron_dataset_adapter.sh /data/megatron/train_text_document
# To copy it to a HyperParallel-owned prefix, pass a second argument.
input_prefix=${1:?usage: $0 INPUT_PREFIX [OUTPUT_PREFIX]}
output_prefix=${2:-}

args=(--input-prefix "${input_prefix}")
if [[ -n "${output_prefix}" ]]; then
  args+=(--output-prefix "${output_prefix}")
fi

python -m hyper_parallel.data.tools.megatron_dataset "${args[@]}"
