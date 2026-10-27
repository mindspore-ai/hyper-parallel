#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /path/to/Qwen3-30B-A3B [trainer overrides...]" >&2
    exit 1
fi

# train_parallel_online.yaml already enables TP=2, CP=2, EP=2 and shards
# FSDP over DP*CP=2. Enabling PP=2 gives an eight-rank mixed topology:
# PP(2) * TP(2) * CP(2) * DP(1), with EP(2) derived inside each stage.
exec bash "${SCRIPT_DIR}/qwen3_moe/run_parallel_online.sh" \
    "$1" \
    --accelerator.pp_size=2 \
    --pipeline.enabled=true \
    "${@:2}"
