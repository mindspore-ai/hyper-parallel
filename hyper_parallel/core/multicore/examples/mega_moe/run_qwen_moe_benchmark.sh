#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
ROOT="${HP_ROOT:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "$PYTHON_BIN" != */* ]]; then
    PYTHON_BIN=$(command -v "$PYTHON_BIN") || die "Python is not executable: $PYTHON_BIN"
fi
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "$PYTHON_BIN")/torchrun}"
CANN_ENV_FILE="${CANN_ENV_FILE:-/usr/local/Ascend/cann/set_env.sh}"  # codespell:ignore cann
NATIVE_ENV_FILE="${NATIVE_ENV_FILE:-$ROOT/build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash}"
SHMEM_LIB_DIR="${SHMEM_LIB_DIR:-$ROOT/build/native/payload/hyper_parallel/core/multicore/shmem/lib/shmem}"
VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NNODES_VALUE="${NNODES:-1}"
NPROC_PER_NODE_VALUE="${NPROC_PER_NODE:-8}"
NODE_RANK_VALUE="${NODE_RANK:-0}"
MASTER_ADDR_VALUE="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_VALUE="${MASTER_PORT:-29500}"
SHMEM_PORT_VALUE="${SHMEM_PORT:-$((MASTER_PORT_VALUE + 1000))}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"

source_env() {
    local env_file=$1
    [[ -f "$env_file" ]] || die "environment file not found: $env_file"
    set +u
    # shellcheck disable=SC1090
    source "$env_file"
    set -u
}

[[ -x "$PYTHON_BIN" ]] || die "Python is not executable: $PYTHON_BIN"
[[ -x "$TORCHRUN_BIN" ]] || die "torchrun is not executable: $TORCHRUN_BIN"
[[ -f "$SCRIPT_DIR/qwen_moe_benchmark.py" ]] || \
    die "benchmark runner is missing under $ROOT"
[[ -d "$SHMEM_LIB_DIR" ]] || die "SHMEM library directory is missing: $SHMEM_LIB_DIR"
[[ "$NNODES_VALUE" =~ ^[0-9]+$ ]] || die "NNODES must be an integer"
[[ "$NPROC_PER_NODE_VALUE" =~ ^[0-9]+$ ]] || die "NPROC_PER_NODE must be an integer"
[[ "$NODE_RANK_VALUE" =~ ^[0-9]+$ ]] || die "NODE_RANK must be an integer"
[[ "$MASTER_PORT_VALUE" =~ ^[0-9]+$ ]] || die "MASTER_PORT must be an integer"
[[ "$SHMEM_PORT_VALUE" =~ ^[0-9]+$ ]] || die "SHMEM_PORT must be an integer"
((NNODES_VALUE * NPROC_PER_NODE_VALUE == 8)) || \
    die "NNODES * NPROC_PER_NODE must equal the benchmark EP size 8"

cd "$ROOT"
source_env "$CANN_ENV_FILE"
source_env "$NATIVE_ENV_FILE"

export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export HYPER_PARALLEL_PLATFORM=torch
SHMEM_HOST_VALUE="${SHMEM_HOST:-$MASTER_ADDR_VALUE}"
export SHMEM_IP_PORT="${SHMEM_IP_PORT:-tcp://$SHMEM_HOST_VALUE:$SHMEM_PORT_VALUE}"
export LD_LIBRARY_PATH="$SHMEM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

IMPORTED_PACKAGE=$(
    "$PYTHON_BIN" -c 'import pathlib, hyper_parallel; print(pathlib.Path(hyper_parallel.__file__).resolve())'
)
case "$IMPORTED_PACKAGE" in
    "$ROOT"/*) ;;
    *) die "hyper_parallel imports from $IMPORTED_PACKAGE instead of $ROOT; run $PYTHON_BIN -m pip install -e $ROOT" ;;
esac

exec timeout --kill-after=60s "$TIMEOUT_SECONDS" \
    "$TORCHRUN_BIN" \
    --nnodes="$NNODES_VALUE" \
    --nproc_per_node="$NPROC_PER_NODE_VALUE" \
    --node_rank="$NODE_RANK_VALUE" \
    --master_addr="$MASTER_ADDR_VALUE" \
    --master_port="$MASTER_PORT_VALUE" \
    hyper_parallel/core/multicore/examples/mega_moe/qwen_moe_benchmark.py \
    "$@"
