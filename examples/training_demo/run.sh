#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set -e

cd "$(dirname "$0")/../.."

NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29501}
export HYPER_PARALLEL_PLATFORM=${HYPER_PARALLEL_PLATFORM:-torch}

python -m examples.training_demo.prepare_model \
    examples/training_demo/train.yaml \
    "$@"

torchrun \
    --nproc_per_node="${NPROC}" \
    --rdzv_id=training_demo \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --module examples.training_demo.train_text \
    examples/training_demo/train.yaml \
    "$@"
