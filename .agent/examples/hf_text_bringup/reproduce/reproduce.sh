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
# Reproduce the recorded HF-native bring-ups, one stage at a time.
#
#   reproduce.sh list                  models, stages and Hub ids
#   reproduce.sh <model> <stage>       run one stage; DRYRUN=1 prints it
#
# Every command goes through the generic kit (run_hf_text.sh,
# run_hf_text_dist.sh, and the repository's run_vlm.sh); this file pins the options
# each recorded run used, so read README.md beside it before running.
#
# Environment (site specifics come from here, never from edits):
#   MODEL_ROOT        directory holding the model directories, each named as
#                     the last component of its Hub id (required to run)
#   OUT               output root (default: <repo>/output/reproduce)
#   NODES             two hostnames, for the two-node cells of qwen25-32b
#   CANN_SETENV       CANN set_env.sh (default /usr/local/Ascend/ascend-toolkit/set_env.sh)
#   PYTHON_BIN        python with torch_npu (default: python)
#   EXTRA_PYTHONPATH  dependency overlay, prepended (e.g. transformers 5.13)
#   S3_EVIDENCE       printed (default: the trainer log, six printed digits,
#                     as recorded) or structured (model_integration.mode=
#                     runtime: full precision, inputs hashed; not yet run on
#                     device with this kit -- see README.md)
#   S3_TRAIN_STEPS    steps per S3 cell (default: 10)
#   S3_EARLY_STEPS    cross-layout comparison window (default: 3, historical recipe)
#   S5_TRAIN_STEPS    historical long-run observation steps (default: 1000)
#   DRYRUN=1          print every command instead of running it
#
# Single-node cells run the admission guard on this node before launching
# and verify release after; the multi-node launcher guards every node itself.
# A busy node rejects the launch; S3 collects remaining checks and reports failure.
# Back off and re-probe later, never queue.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
KIT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
DEMO_DIR=$(cd "${KIT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${DEMO_DIR}/../.." && pwd)
GUARD="${PROJECT_ROOT}/.agent/skills/hf-model-integration/scripts/node_guard.sh"
SKILL_SCRIPTS="${PROJECT_ROOT}/.agent/skills/hf-model-integration/scripts"
BREAKDOWN="${PROJECT_ROOT}/.agent/skills/profiling/scripts/breakdown.py"
OUT=${OUT:-${PROJECT_ROOT}/output/reproduce}
PYTHON_BIN=${PYTHON_BIN:-python}
CANN_SETENV=${CANN_SETENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
S3_EVIDENCE=${S3_EVIDENCE:-printed}
DRYRUN=${DRYRUN:-0}
S3_TRAIN_STEPS=${S3_TRAIN_STEPS:-10}
S3_EARLY_STEPS=${S3_EARLY_STEPS:-3}
S3_FAILURES=0
S5_TRAIN_STEPS=${S5_TRAIN_STEPS:-1000}

MODELS="glm4-9b smollm2-360m dsv2-lite qwen25-32b qwen25-vl-3b qwen35-9b"
if grep -Eq '^[[:space:]]+profiler:[[:space:]]+ProfilingConfig' \
        "${PROJECT_ROOT}/hyper_parallel/trainer/config/trainer.py"; then
    PROFILE=(--profiler.enable=true --profiler.start_step=5 --profiler.stop_step=8)
    TRACE_OUTPUT_OPTION=--profiler.output_path
else
    PROFILE=(--profiling.enabled=true --profiling.start_step=5 --profiling.end_step=8)
    TRACE_OUTPUT_OPTION=--profiling.trace_dir
fi

hub_id() {
    case "$1" in
        glm4-9b) echo "THUDM/GLM-4-9B-0414" ;;
        smollm2-360m) echo "HuggingFaceTB/SmolLM2-360M" ;;
        dsv2-lite) echo "deepseek-ai/DeepSeek-V2-Lite" ;;
        qwen25-32b) echo "Qwen/Qwen2.5-32B" ;;
        qwen25-vl-3b) echo "Qwen/Qwen2.5-VL-3B-Instruct" ;;
        qwen35-9b) echo "Qwen/Qwen3.5-9B" ;;
        *) return 1 ;;
    esac
}

stages_of() {
    case "$1" in
        smollm2-360m) echo "s1 s2a s2b s3 s3-adamw s4 s5" ;;
        qwen25-32b) echo "s1 s2a s2b s2b-full s3 s4 s4-full s5" ;;
        *) echo "s1 s2a s2b s3 s4 s5" ;;
    esac
}

# Options every text run of the model carries. 9B-class dense models need
# bf16 initialization and recompute on 64 GB dies; the MoE and the 32B need
# bf16 initialization.
base_overrides() {
    case "$1" in
        glm4-9b|qwen35-9b) echo "--model_init_dtype=bfloat16 --activation_checkpoint.mode=full" ;;
        dsv2-lite|qwen25-32b) echo "--model_init_dtype=bfloat16" ;;
        *) echo "" ;;
    esac
}

# Layers kept for the S3 matrix and the single-die parity (empty = full depth).
s3_layers() {
    case "$1" in
        glm4-9b) echo 20 ;;
        dsv2-lite) echo 13 ;;
        qwen25-32b) echo 16 ;;
        *) echo "" ;;
    esac
}
parity_layers() {
    case "$1" in
        dsv2-lite|qwen25-32b) echo 13 ;;
        *) echo "" ;;
    esac
}

usage() {
    echo "usage: $0 list | <model> <stage>" >&2
    exit 1
}

list_models() {
    printf '%-14s %-30s %s\n' model "Hub id" stages
    for model in ${MODELS}; do
        printf '%-14s %-30s %s\n' "${model}" "$(hub_id "${model}")" "$(stages_of "${model}")"
    done
}

setup_env() {  # once per process, before the first real command
    [[ "${DRYRUN}" = "1" ]] && return 0
    # shellcheck disable=SC1090
    source "${CANN_SETENV}"
    export HYPER_PARALLEL_PLATFORM=torch
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="${EXTRA_PYTHONPATH:+${EXTRA_PYTHONPATH}:}${PROJECT_ROOT}/.agent/examples:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    # run_vlm.sh calls a bare `torchrun`: make it the matched interpreter's.
    if [[ "${PYTHON_BIN}" = /* ]]; then
        PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
        export PATH
    fi
    cd "${PROJECT_ROOT}"
}

say() {  # DRYRUN prints the command; otherwise run it
    if [[ "${DRYRUN}" = "1" ]]; then
        printf '%q ' "$@"; printf '\n'
    else
        "$@"
    fi
}

wait_rendezvous_port() {  # --standalone reuses 29400; let TIME_WAIT drain
    command -v ss >/dev/null 2>&1 || return 0
    for _ in $(seq 1 30); do
        [[ "$(ss -tan 2>/dev/null | grep -c ':29400 ')" = 0 ]] && return 0
        sleep 2
    done
}

# launch <cell> <command...>: guard this node, run, verify release.
launch() {
    local cell=$1; shift
    local log="${STAGE_DIR}/${cell}.log"
    if [[ "${DRYRUN}" = "1" ]]; then
        printf '[%s] ' "${cell}"; printf '%q ' "$@"; printf '\n'
        return 0
    fi
    bash "${GUARD}" pre || { echo "[${cell}] node busy: back off and re-probe later" >&2; return 3; }
    wait_rendezvous_port
    local rc=0
    "$@" > "${log}" 2>&1 || rc=$?
    if ! bash "${GUARD}" post; then
        echo "[${cell}] residue reported above: clean it before the next cell" >&2
        [[ "${rc}" != 0 ]] || rc=3
    fi
    echo "[${cell}] exit=${rc} log=${log}"
    return ${rc}
}

# launch_dist <cell> <command...>: the multi-node launcher guards every node.
launch_dist() {
    local cell=$1; shift
    if [[ -z "${NODES:-}" ]]; then
        echo "[${cell}] needs NODES=\"host1 host2\"" >&2
        return 1
    fi
    if [[ "${DRYRUN}" = "1" ]]; then
        printf '[%s] NODES=%q ' "${cell}" "${NODES}"; printf '%q ' "$@"; printf '\n'
        return 0
    fi
    local rc=0
    NODES="${NODES}" "$@" > "${STAGE_DIR}/${cell}.log" 2>&1 || rc=$?
    echo "[${cell}] exit=${rc} log=${STAGE_DIR}/${cell}.log"
    return ${rc}
}

text() {  # text <cell> <iters> [overrides...]: single-node kit run
    local cell=$1 iters=$2; shift 2
    # shellcheck disable=SC2046
    launch "${cell}" env NPROC_PER_NODE=8 bash "${KIT_DIR}/run_hf_text.sh" "${MODEL_PATH}" "${iters}" \
        $(base_overrides "${MODEL}") "$@"
}

text_np() {  # text_np <nproc> <cell> <iters> [overrides...]
    local nproc=$1 cell=$2 iters=$3; shift 3
    # shellcheck disable=SC2046
    launch "${cell}" env NPROC_PER_NODE="${nproc}" bash "${KIT_DIR}/run_hf_text.sh" \
        "${MODEL_PATH}" "${iters}" $(base_overrides "${MODEL}") "$@"
}

dist() {  # Recorded two-node recipe uses eight logical devices per node.
    local cell=$1 iters=$2; shift 2
    # shellcheck disable=SC2046
    launch_dist "${cell}" env NPROC_PER_NODE=8 bash "${KIT_DIR}/run_hf_text_dist.sh" "${MODEL_PATH}" "${iters}" \
        $(base_overrides "${MODEL}") "$@"
}

vlm() {  # vlm <cell> <nproc> <data> [overrides...]
    local cell=$1 nproc=$2 data=$3; shift 3
    launch "${cell}" env RUN_NAME="${cell}" NPROC_PER_NODE="${nproc}" \
        bash "${PROJECT_ROOT}/examples/training_demo/run_vlm.sh" "${MODEL_PATH}" "${data}" "$@"
}

evidence() {  # structured S3 evidence lands under the cell's own directory
    [[ "${S3_EVIDENCE}" = structured ]] || return 0
    echo "--model_integration.mode=runtime --checkpoint.checkpoint_dir=${STAGE_DIR}/$1"
}

series() {  # series <cell>: NAME=PATH for the judge
    if [[ "${S3_EVIDENCE}" = structured ]]; then
        echo "--series=$1=${STAGE_DIR}/$1/model_integration"
    else
        echo "--series=$1=${STAGE_DIR}/$1.log"
    fi
}

UNIFORM_CORPUS() { echo "${STAGE_DIR}/uniform.jsonl"; }
make_uniform_corpus() {  # one sequence per document: every sample is identical
    say "${PYTHON_BIN}" -m hf_text_bringup.prepare_text_data \
        --output "$(UNIFORM_CORPUS)" --num-samples 64 --sequence-length 2048 \
        --uniform-tokenizer "${MODEL_PATH}" --sequences-per-document 1
}

# verdict <identical|early> <a> <b> [steps]: one criterion per judge call, so
# a cell that could not run costs only the verdicts that need it.
verdict() {
    local kind=$1 a=$2 b=$3 steps=${4:-${S3_EARLY_STEPS}} flag rc=0
    if [[ "${kind}" = early ]]; then flag="--early-identical=${a},${b},${steps}"
    else flag="--identical=${a},${b}"; fi
    set -- "${PYTHON_BIN}" -m hf_text_bringup.judge_equivalence \
        "$(series "${a}")" "$(series "${b}")" "${flag}" --min-steps="${S3_TRAIN_STEPS}"
    if [[ "${DRYRUN}" = "1" ]]; then
        say "$@"
        return 0
    fi
    "$@" >> "${STAGE_DIR}/judge.txt" 2>&1 || rc=$?
    echo "[verdict ${kind} ${a} ${b}] exit=${rc} (details in ${STAGE_DIR}/judge.txt)"
    return "${rc}"
}

# Catch each result without terminating the matrix; retain launch failures even
# if the partial log happens to satisfy a numerical comparison.
s3_collect() {
    local rc=0
    "$@" || rc=$?
    if [[ "${rc}" != 0 ]]; then
        S3_FAILURES=$((S3_FAILURES + 1))
        echo "[s3 failed] $* exit=${rc}"
    fi
    return 0
}

s3_finish() {
    if [[ "${DRYRUN}" = 1 ]]; then
        echo "[s3] DRYRUN: no acceptance evidence collected"
        return 0
    fi
    echo "[s3] failed or untestable checks=${S3_FAILURES}" | tee "${STAGE_DIR}/summary.txt"
    [[ "${S3_FAILURES}" = 0 ]]
}

untestable() {
    S3_FAILURES=$((S3_FAILURES + 1))
    echo "[$1] UNTESTABLE: $2"
    [[ "${DRYRUN}" = "1" ]] || echo "$1: $2" >> "${STAGE_DIR}/untestable.txt"
}

# ---------------------------------------------------------------- stages
stage_s1() {
    say "${PYTHON_BIN}" "${SKILL_SCRIPTS}/check_released_layout.py" --model-dir "${MODEL_PATH}"
    say "${PYTHON_BIN}" "${SKILL_SCRIPTS}/check_tied_save.py" --model-dir "${MODEL_PATH}"
}

stage_s2a() {
    local layers image=()
    layers=$(parity_layers "${MODEL}")
    [[ "${MODEL}" = qwen25-vl-3b ]] && image=(--image)
    # shellcheck disable=SC2086
    launch parity env ASCEND_RT_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m \
        hf_text_bringup.reproduce.parity_check \
        --model-dir "${MODEL_PATH}" --dtype bfloat16 \
        ${layers:+--num-hidden-layers ${layers}} ${image[@]+"${image[@]}"} \
        --output "${STAGE_DIR}/parity.json"
}

stage_s2b() {
    case "${MODEL}" in
        qwen25-vl-3b) make_vlm_data 768; vlm smoke 8 "$(VLM_DATA 768)" --training.train_iters=30 ;;
        qwen25-32b) dist smoke 30 --model.num_hidden_layers=16 ;;
        *) text smoke 30 ;;
    esac
}

stage_s2b_full() {  # qwen25-32b at full depth: host-side init + recompute
    dist full_depth 10 --training.init_device=cpu --activation_checkpoint.mode=full \
        --training.global_batch_size=16 --fsdp_config.dp_shard_size=16
}

# The VLM packing path consumes samples once (text cycles epochs), so the
# sample count is sized by steps: 768 carried 30 steps, 27000 carry 1000.
VLM_DATA() { echo "${OUT}/${MODEL}/data_$1/synthetic.jsonl"; }
make_vlm_data() {
    say "${PYTHON_BIN}" -m prepare_vlm_synthetic \
        --output-dir "${OUT}/${MODEL}/data_$1" --num-samples "$1"
}

stage_s3() {
    local layers data common
    if [[ "${MODEL}" = qwen25-vl-3b ]]; then
        local samples=$((S3_TRAIN_STEPS * 32 + 768))
        make_vlm_data "${samples}"
        data=$(VLM_DATA "${samples}")
        # shellcheck disable=SC2046
        s3_collect vlm dp8 8 "${data}" "--training.train_iters=${S3_TRAIN_STEPS}" $(evidence dp8)
        # shellcheck disable=SC2046
        s3_collect vlm recompute 8 "${data}" "--training.train_iters=${S3_TRAIN_STEPS}" --activation_checkpoint.mode=full \
            $(evidence recompute)
        # shellcheck disable=SC2046
        s3_collect vlm rerun 8 "${data}" "--training.train_iters=${S3_TRAIN_STEPS}" $(evidence rerun)
        # shellcheck disable=SC2046
        s3_collect vlm dp4 4 "${data}" "--training.train_iters=${S3_TRAIN_STEPS}" --training.global_batch_size=8 \
            $(evidence dp4)
        s3_collect verdict identical dp8 recompute
        s3_collect verdict identical dp8 rerun
        s3_collect verdict early dp8 dp4
        s3_finish
        return $?
    fi
    make_uniform_corpus
    data=$(UNIFORM_CORPUS)
    layers=$(s3_layers "${MODEL}")
    common="--model.load_pretrained=true --training.init_device=cpu --training.seed=42"
    common+=" --dataset.data_path=${data}${layers:+ --model.num_hidden_layers=${layers}}"
    case "${MODEL}" in
        qwen35-9b)
            untestable recompute_off "recompute off exceeds a 64 GB die at step 1 at full depth"
            # shellcheck disable=SC2086,SC2046
            s3_collect text dp8 "${S3_TRAIN_STEPS}" ${common} --training.global_batch_size=8 --fsdp_config.dp_shard_size=8 \
                $(evidence dp8)
            # shellcheck disable=SC2086,SC2046
            s3_collect text rerun "${S3_TRAIN_STEPS}" ${common} --training.global_batch_size=8 --fsdp_config.dp_shard_size=8 \
                $(evidence rerun)
            # shellcheck disable=SC2086,SC2046
            s3_collect text_np 4 dp4 "${S3_TRAIN_STEPS}" ${common} --training.global_batch_size=8 --fsdp_config.dp_shard_size=4 \
                $(evidence dp4)
            s3_collect verdict identical dp8 rerun
            s3_collect verdict early dp8 dp4
            ;;
        qwen25-32b)
            untestable dp4 "four shards hold twice the optimizer state and exceed a 64 GB die"
            # shellcheck disable=SC2086,SC2046
            s3_collect text dp8 "${S3_TRAIN_STEPS}" ${common} --training.global_batch_size=8 --fsdp_config.dp_shard_size=8 \
                $(evidence dp8)
            # shellcheck disable=SC2086,SC2046
            s3_collect text recompute "${S3_TRAIN_STEPS}" ${common} --training.global_batch_size=8 --fsdp_config.dp_shard_size=8 \
                --activation_checkpoint.mode=full $(evidence recompute)
            # shellcheck disable=SC2086,SC2046
            s3_collect text rerun "${S3_TRAIN_STEPS}" ${common} --training.global_batch_size=8 --fsdp_config.dp_shard_size=8 \
                $(evidence rerun)
            # Cross-node: one node dp8 against two nodes dp16 at the same global batch.
            # shellcheck disable=SC2086,SC2046
            s3_collect text node1_dp8 "${S3_TRAIN_STEPS}" ${common} --training.global_batch_size=16 --fsdp_config.dp_shard_size=8 \
                $(evidence node1_dp8)
            # shellcheck disable=SC2086,SC2046
            s3_collect dist nodes2_dp16 "${S3_TRAIN_STEPS}" ${common} --training.global_batch_size=16 --fsdp_config.dp_shard_size=16 \
                $(evidence nodes2_dp16)
            s3_collect verdict identical dp8 recompute
            s3_collect verdict identical dp8 rerun
            s3_collect verdict early node1_dp8 nodes2_dp16
            ;;
        *)
            # shellcheck disable=SC2086,SC2046
            s3_collect text dp8 "${S3_TRAIN_STEPS}" ${common} --training.global_batch_size=8 --fsdp_config.dp_shard_size=8 \
                --activation_checkpoint.mode=off $(evidence dp8)
            # shellcheck disable=SC2086,SC2046
            s3_collect text recompute "${S3_TRAIN_STEPS}" ${common} --training.global_batch_size=8 --fsdp_config.dp_shard_size=8 \
                --activation_checkpoint.mode=full $(evidence recompute)
            # shellcheck disable=SC2086,SC2046
            s3_collect text rerun "${S3_TRAIN_STEPS}" ${common} --training.global_batch_size=8 --fsdp_config.dp_shard_size=8 \
                --activation_checkpoint.mode=off $(evidence rerun)
            # shellcheck disable=SC2086,SC2046
            s3_collect text_np 4 dp4 "${S3_TRAIN_STEPS}" ${common} --training.global_batch_size=8 --fsdp_config.dp_shard_size=4 \
                --activation_checkpoint.mode=off $(evidence dp4)
            s3_collect verdict identical dp8 recompute
            s3_collect verdict identical dp8 rerun
            s3_collect verdict early dp8 dp4
            ;;
    esac
    s3_finish
}

stage_s3_adamw() {  # smollm2-360m: same cross-layout cell under AdamW alone
    local data common recipe="${STAGE_DIR}/train_adamw.yaml"
    # Overrides cannot change an optimizer _target_: the variant is a recipe.
    say "${PYTHON_BIN}" -m hf_text_bringup.reproduce.adamw_recipe \
        --output "${recipe}"
    make_uniform_corpus
    data=$(UNIFORM_CORPUS)
    common="--model.load_pretrained=true --training.init_device=cpu --training.seed=42"
    common+=" --training.global_batch_size=8 --dataset.data_path=${data}"
    # shellcheck disable=SC2086,SC2046
    s3_collect launch dp8 env RECIPE="${recipe}" NPROC_PER_NODE=8 bash "${KIT_DIR}/run_hf_text.sh" "${MODEL_PATH}" "${S3_TRAIN_STEPS}" \
        ${common} --fsdp_config.dp_shard_size=8 $(evidence dp8)
    # shellcheck disable=SC2086,SC2046
    s3_collect launch dp4 env RECIPE="${recipe}" NPROC_PER_NODE=4 bash "${KIT_DIR}/run_hf_text.sh" \
        "${MODEL_PATH}" "${S3_TRAIN_STEPS}" ${common} --fsdp_config.dp_shard_size=4 $(evidence dp4)
    s3_collect verdict early dp8 dp4
    s3_finish
}

breakdown() {  # breakdown <trace dir>
    say "${PYTHON_BIN}" "${BREAKDOWN}" --prof-dir "$1" --out "${STAGE_DIR}/breakdown.html" \
        --json "${STAGE_DIR}/breakdown.json"
}

stage_s4() {
    local trace="${STAGE_DIR}/prof"
    case "${MODEL}" in
        qwen25-vl-3b)
            make_vlm_data 768
            vlm profile 8 "$(VLM_DATA 768)" --training.train_iters=12 "${PROFILE[@]}" \
                "${TRACE_OUTPUT_OPTION}=${trace}" ;;
        qwen25-32b)
            dist profile 12 --model.num_hidden_layers=16 --training.global_batch_size=16 \
                --fsdp_config.dp_shard_size=16 "${PROFILE[@]}" "${TRACE_OUTPUT_OPTION}=${trace}" ;;
        *) text profile 12 "${PROFILE[@]}" "${TRACE_OUTPUT_OPTION}=${trace}" ;;
    esac
    breakdown "${trace}"
}

stage_s4_full() {  # qwen25-32b at full depth
    local trace="${STAGE_DIR}/prof"
    dist profile 12 --training.init_device=cpu --activation_checkpoint.mode=full \
        --training.global_batch_size=16 --fsdp_config.dp_shard_size=16 "${PROFILE[@]}" \
        "${TRACE_OUTPUT_OPTION}=${trace}"
    breakdown "${trace}"
}

s5_long_run() {
    case "${MODEL}" in
        qwen25-vl-3b)
            local samples=$((S5_TRAIN_STEPS * 27))
            make_vlm_data "${samples}" || return $?
            vlm long_run 8 "$(VLM_DATA "${samples}")" "--training.train_iters=${S5_TRAIN_STEPS}" ;;
        qwen25-32b) dist long_run "${S5_TRAIN_STEPS}" --model.num_hidden_layers=16 \
            --training.global_batch_size=16 --fsdp_config.dp_shard_size=16 ;;
        *) text long_run "${S5_TRAIN_STEPS}" ;;
    esac
}

stage_s5() {
    local rc=0
    echo "[s5] Long-run observation only: no baseline comparison or precision acceptance."
    s5_long_run || rc=$?
    if [[ "${DRYRUN}" != 1 ]]; then
        "${PYTHON_BIN}" - "${STAGE_DIR}/observation.json" "${S5_TRAIN_STEPS}" "${rc}" <<'PYOBS'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "kind": "long_run_observation", "requested_train_steps": int(sys.argv[2]),
    "launcher_exit_code": int(sys.argv[3]), "baseline_comparison": False,
    "precision_acceptance": "not_evaluated", "log": "long_run.log",
}, indent=2) + "\n", encoding="utf-8")
PYOBS
    fi
    return "${rc}"
}

# ---------------------------------------------------------------- main
[[ $# -ge 1 ]] || usage
if [[ "$1" = list ]]; then
    list_models
    exit 0
fi
[[ $# -eq 2 ]] || usage
MODEL=$1
STAGE=$2
hub_id "${MODEL}" >/dev/null || { echo "unknown model: ${MODEL}" >&2; list_models >&2; exit 1; }
case " $(stages_of "${MODEL}") " in
    *" ${STAGE} "*) ;;
    *) echo "unknown stage for ${MODEL}: ${STAGE} (have: $(stages_of "${MODEL}"))" >&2; exit 1 ;;
esac
if [[ -z "${MODEL_ROOT:-}" ]]; then
    [[ "${DRYRUN}" = "1" ]] || { echo "MODEL_ROOT is required" >&2; exit 1; }
    MODEL_ROOT=/path/to/models
fi
MODEL_PATH="${MODEL_ROOT}/$(hub_id "${MODEL}" | cut -d/ -f2)"
if [[ "${DRYRUN}" != "1" && ! -s "${MODEL_PATH}/config.json" ]]; then
    echo "model directory not found: ${MODEL_PATH} (download $(hub_id "${MODEL}") there)" >&2
    exit 1
fi
STAGE_DIR="${OUT}/${MODEL}/${STAGE}"
if [[ "${STAGE}" = s3 || "${STAGE}" = s3-adamw ]]; then
    for value in "${S3_TRAIN_STEPS}" "${S3_EARLY_STEPS}"; do
        [[ "${value}" =~ ^[1-9][0-9]{0,8}$ ]] || {
            echo "S3 step counts must be positive integers (at most nine digits)" >&2; exit 2;
        }
    done
    [[ "${S3_EARLY_STEPS}" -le "${S3_TRAIN_STEPS}" ]] || {
        echo "S3_EARLY_STEPS cannot exceed S3_TRAIN_STEPS" >&2; exit 2;
    }
    [[ "${S3_EVIDENCE}" = printed || "${S3_EVIDENCE}" = structured ]] || {
        echo "S3_EVIDENCE must be printed or structured" >&2; exit 2;
    }
    if [[ "${DRYRUN}" = 1 ]]; then
        STAGE_DIR="${STAGE_DIR}/run-DRYRUN"
    else
        mkdir -p "${STAGE_DIR}"
        STAGE_DIR=$(mktemp -d "${STAGE_DIR}/run-XXXXXXXX")
    fi
elif [[ "${STAGE}" = s5 ]]; then
    [[ "${S5_TRAIN_STEPS}" =~ ^[1-9][0-9]{0,8}$ ]] || {
        echo "S5_TRAIN_STEPS must be a positive integer (at most nine digits)" >&2; exit 2;
    }
    if [[ "${DRYRUN}" = 1 ]]; then
        STAGE_DIR="${STAGE_DIR}/run-DRYRUN"
    else
        mkdir -p "${STAGE_DIR}"
        STAGE_DIR=$(mktemp -d "${STAGE_DIR}/run-XXXXXXXX")
    fi
else
    [[ "${DRYRUN}" = "1" ]] || mkdir -p "${STAGE_DIR}"
fi
setup_env
echo "# ${MODEL} ${STAGE}: $(hub_id "${MODEL}") -> ${STAGE_DIR}"
"stage_${STAGE//-/_}"
