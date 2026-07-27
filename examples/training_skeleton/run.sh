#!/bin/bash
# End-to-end multi-card training skeleton example launcher.
#
# Adjust --nproc_per_node to the number of available GPUs. The accelerator
# topology in train.yaml must multiply to --nproc_per_node (default: 4).

set -e

cd "$(dirname "$0")/../.."

NPROC=${NPROC:-4}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

torchrun \
    --nproc_per_node="${NPROC}" \
    --rdzv_id=training_skeleton \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    examples/training_skeleton/main.py \
    examples/training_skeleton/train.yaml \
    "$@"
