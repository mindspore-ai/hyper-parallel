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

set -euo pipefail

cd "$(dirname "$0")/../.."
STEPS=${STEPS:-1}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
LAYERS=${LAYERS:-6}
OUTPUT_DIR=${OUTPUT_DIR:-output/large_model_compile_8card/seq${SEQ_LENGTH}_layers${LAYERS}}
mkdir -p "${OUTPUT_DIR}"

export HYPER_PARALLEL_PLATFORM=${HYPER_PARALLEL_PLATFORM:-torch}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

run_case() {
    local label=$1
    local compile_enabled=$2
    local port=$3
    local log_file="${OUTPUT_DIR}/${label}.log"

    echo "=== ${label}: compile.enabled=${compile_enabled} ==="
    MASTER_PORT="${port}" \
        RDZV_ID="large_qwen3_compile_${label}_${SEQ_LENGTH}_${LAYERS}" \
        bash examples/large_model_compile/run.sh \
        --training.train_iters="${STEPS}" \
        --model.num_hidden_layers="${LAYERS}" \
        --dataset.data_config.seq_length="${SEQ_LENGTH}" \
        --compile.enabled="${compile_enabled}" \
        --checkpoint.save_ckpt=false \
        --checkpoint.restore_from=null \
        2>&1 | tee "${log_file}"
}

run_case eager false "${MASTER_PORT_EAGER:-29551}"
run_case compile true "${MASTER_PORT_COMPILE:-29552}"

echo
echo "=== Summary (structured performance metrics) ==="
for label in eager compile; do
    log_file="${OUTPUT_DIR}/${label}.log"
    [ -f "${log_file}" ] || continue
    python - "${label}" "${log_file}" "$((SEQ_LENGTH))" <<'PY'
import re
import statistics
import sys

label, path, seq_length = sys.argv[1:]
seq_length = int(seq_length)
text = open(path, encoding="utf-8", errors="ignore").read()
values = [float(v) for v in re.findall(r"performance/step_time=([0-9.]+)", text)]
if not values:
    print(f"{label}: no completed steps")
    raise SystemExit
if len(values) == 1:
    print(
        f"{label}: single_step={values[0]:.5f}s steps=1 "
        f"configured_sequence_length={seq_length} "
        f"single_step_tokens_per_second={seq_length / values[0]:.3f}"
    )
else:
    steady = values[1:]
    print(
        f"{label}: first_step={values[0]:.5f}s "
        f"steady_avg={statistics.mean(steady):.5f}s "
        f"steps={len(values)} "
        f"configured_sequence_length={seq_length} "
        f"steady_tokens_per_second={seq_length / statistics.mean(steady):.3f}"
    )
PY
done
