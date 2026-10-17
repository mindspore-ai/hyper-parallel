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
#
# Check a node before a DeepSeek-V4.1 measurement window opens.
#
# Every check here is a read: no device is allocated, nothing is launched, and
# it is safe to run while somebody else holds the node. Run it the day before
# the window, and again in its first minute.
#
# Why each check exists, from what the last round cost:
#
#   The 1 October round raised ModuleNotFoundError for omni_training_custom_ops
#   on every rank, from the sparse attention forward. V4.1 attention has no
#   torch fallback and the package has to be carried in and built per SoC, so
#   a missing one is not a five minute fix on a cluster with no internet.
#
#   That round lost 20 of its minutes to a node another process was holding,
#   and the 1 October Qwen3.5 round lost 5 of its 20 points to a process
#   holding 40 to 47 GiB on one node.
#
#   Nothing records which indexer path a run took. The launcher sets neither
#   indexer flag, the validation builder defaults fused_indexer true, and the
#   attention passes it as use_provider, which consults the provider and never
#   raises. So a run falls back to the reference indexer SILENTLY, and a
#   profile taken on the reference is not a production run's.
#
# Usage:
#   ./preflight_deepseek_v41_window.sh /path/to/DeepSeek-V4.1-Flash
#
# Exit status is 0 only when every FAIL-level check passed. WARN items do not
# fail the run; they change what the window can be used for, so read them.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
# A device holding more than this is somebody else's run, not driver overhead.
BUSY_MIB=${BUSY_MIB:-1024}
MODEL_PATH=${1:-}

fails=0
warns=0

pass() { printf 'PASS  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; warns=$((warns + 1)); }
fail() { printf 'FAIL  %s\n' "$*"; fails=$((fails + 1)); }
head2() { printf '\n== %s\n' "$*"; }

printf '%s\n' "DeepSeek-V4.1 window pre-flight"
printf '%s\n' "node   $(hostname)"
printf '%s\n' "date   $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf '%s\n' "tree   ${PROJECT_ROOT}"
printf '%s\n' "python $(command -v "${PYTHON_BIN}" || echo MISSING)"

head2 "1. The tree imports, which is the run script's own gate"
if "${PYTHON_BIN}" -c 'import hyper_parallel' >/dev/null 2>&1; then
    pass "hyper_parallel imports from $(
        "${PYTHON_BIN}" -c 'import hyper_parallel, os; print(os.path.dirname(hyper_parallel.__file__))')"
else
    fail "hyper_parallel does not import. Prepare the shell first; the run script refuses too."
fi
if "${PYTHON_BIN}" -c 'from transformers.models.deepseek_v4 import DeepseekV4Config' >/dev/null 2>&1; then
    pass "transformers knows deepseek_v4"
else
    fail "transformers has no deepseek_v4. See docs/guide/trainer/current_hf_model_environment.md"
fi

head2 "2. The NPU operator packages the attention needs"
if "${PYTHON_BIN}" -c 'import omni_training_custom_ops' >/dev/null 2>&1; then
    pass "omni_training_custom_ops imports. The sparse attention forward can run."
else
    fail "omni_training_custom_ops is MISSING. V4.1 attention has no torch fallback on NPU,
      so every rank raises in the forward. This is the one check that must pass before
      a window is worth holding, and it cannot be fixed inside one: the package is built
      per SoC and this cluster has no internet."
fi
if "${PYTHON_BIN}" -c 'import cann_ops_transformer' >/dev/null 2>&1; then
    pass "cann_ops_transformer imports"
    if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import cann_ops_transformer as m
raise SystemExit(0 if hasattr(m, "SparseFlashMla") else 1)
PY
    then
        pass "SparseFlashMla is exported, so the fused attention path is available"
    else
        warn "cann_ops_transformer exports no SparseFlashMla: with the fused kernels on,
      the attention raises by design. Leave the fused flag off, or build it."
    fi
else
    warn "cann_ops_transformer is missing. The Indexer TopK and the KL degrade gracefully,
      so a run on the reference path still measures; it is not what production runs."
fi

head2 "3. Which indexer path a run would take, which nothing records today"
"${PYTHON_BIN}" - <<'PY'
# Report the path rather than assert it: both are measurable, and the point is
# that a profile taken on the reference is not a production run's.
import importlib

found = {}
for name in ("omni_training_custom_ops", "cann_ops_transformer"):
    try:
        found[name] = importlib.import_module(name)
    except Exception as error:                                   # noqa: BLE001
        found[name] = error
fused = [name for name, value in found.items() if not isinstance(value, Exception)]
if len(fused) == 2:
    print("INFO  both operator packages present: a run attempts the FUSED indexer")
elif fused:
    print(f"INFO  only {fused[0]} present: the indexer may fall back per call, silently")
else:
    print("INFO  neither package present: a run takes the REFERENCE indexer throughout")
print("INFO  the two paths are told apart in a kernel table: the reference is chunked")
print("INFO  fp32 matmuls with a relu, repeated per indexer layer; the fused path is one")
print("INFO  indexer kernel a call with none of that")
PY

head2 "4. Is the node actually free, which is what cost the last round 20 minutes"
if command -v npu-smi >/dev/null 2>&1; then
    held=$(npu-smi info 2>/dev/null \
        | awk -v busy="${BUSY_MIB}" '
            # The usage table lists "<used> / <total>" MiB per device.
            match($0, /[0-9]+ *\/ *[0-9]+/) {
                split(substr($0, RSTART, RLENGTH), pair, "/")
                used = pair[1] + 0
                if (used > busy) { count++ ; sum += used }
            }
            END { printf "%d %d", count + 0, sum + 0 }')
    count=${held% *}
    total=${held#* }
    if [[ "${count}" -eq 0 ]]; then
        pass "no device holds more than ${BUSY_MIB} MiB: the node looks idle"
    else
        fail "${count} device(s) hold ${total} MiB in all. Another process is on this node.
      Do not open the window: times measured beside another run do not compare with
      times measured alone, and that is what the 1 October rounds lost points to."
    fi
    devices=$(npu-smi info -l 2>/dev/null | grep -ciE 'NPU ID' || true)
    if [[ "${devices}" -ge "${NPROC_PER_NODE}" ]]; then
        pass "${devices} devices visible, ${NPROC_PER_NODE} needed"
    else
        warn "npu-smi lists ${devices} devices and the run wants ${NPROC_PER_NODE}"
    fi
else
    warn "npu-smi not on PATH, so the node's occupancy was not checked. Check it by hand:
      a busy node is the commonest way one of these windows is wasted."
fi

head2 "5. The model assets the run opens"
if [[ -z "${MODEL_PATH}" ]]; then
    warn "no model directory given, so the assets were not checked.
      Usage: $0 /path/to/DeepSeek-V4.1-Flash"
elif [[ ! -d "${MODEL_PATH}" ]]; then
    fail "model directory does not exist: ${MODEL_PATH}"
else
    for file in config.json tokenizer.json; do
        if [[ -s "${MODEL_PATH}/${file}" ]]; then
            pass "${file} present"
        else
            fail "${MODEL_PATH}/${file} is missing or empty. The tokenizer cannot be substituted."
        fi
    done
fi

head2 "6. Are the last round's profiles still here, which decides whether D9 needs a run"
mapfile -t kernels < <(find "${PROJECT_ROOT}/output" -name kernel_details.csv 2>/dev/null | sort)
if [[ "${#kernels[@]}" -gt 0 ]]; then
    pass "${#kernels[@]} kernel table(s) found, so the per mode kernel split needs NO new run:"
    for path in "${kernels[@]}"; do
        printf '        %s  (%s)\n' "${path#"${PROJECT_ROOT}/"}" "$(du -h "${path}" | cut -f1)"
    done
else
    warn "no kernel_details.csv under output/. The per mode kernel split then needs the two
      profiles taken again, which is 2 more launches inside the window rather than a read."
fi

head2 "7. What the launcher states, which the repeat script has to override"
iters=$(grep -E '^[[:space:]]*train_iters:' "${SCRIPT_DIR}/train_deepseek_v41_online.yaml" \
    | head -1 | tr -d ' ' | cut -d: -f2)
printf 'INFO  train_deepseek_v41_online.yaml states train_iters: %s\n' "${iters:-unknown}"
if [[ "${iters:-1}" -le 5 ]]; then
    warn "at train_iters ${iters:-1} a run has no step after a profiling window, so no honest
      step time can be harvested from it. The repeat script raises it; a run launched
      from the stock script cannot be used for timing."
fi
printf 'INFO  run_deepseek_v41_online.sh tees every run of one mode to the SAME file,\n'
printf 'INFO  output/training_demo/deepseek_v41/run_<mode>.log, so repeated tp1 runs\n'
printf 'INFO  overwrite each other and only the last survives. Use the repeat script.\n'

head2 "8. Room for the run logs"
free_gib=$(df -BG --output=avail "${PROJECT_ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -n "${free_gib}" && "${free_gib}" -ge 10 ]]; then
    pass "${free_gib} GiB free under the tree"
elif [[ -n "${free_gib}" ]]; then
    warn "only ${free_gib} GiB free under the tree"
else
    warn "could not read the free space under ${PROJECT_ROOT}"
fi

printf '\n%s\n' "---"
printf '%s\n' "${fails} FAIL, ${warns} WARN"
if [[ "${fails}" -gt 0 ]]; then
    printf '%s\n' "Do not open the window until the FAIL items are cleared."
    exit 1
fi
printf '%s\n' "Clear to open the window."
