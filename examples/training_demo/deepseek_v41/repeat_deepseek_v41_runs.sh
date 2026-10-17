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
# Measure DeepSeek-V4.1 step times REPEATEDLY and WITHOUT the profiler, so the
# strategies can be ranked on something the measurement can resolve.
#
# Why this exists rather than the stock run script:
#
#   The seven points measured on 1 and 5 October were each run once, under the
#   profiler, and ranked on the mean of profiled steps 3 and 4. The whole gap
#   being argued over is 152.6 ms, 1.9% of a step. The same strategy measured
#   twice gave 8268.0 and 8173.4 ms, 1.2% apart. On Qwen3.5 the Ascend profiler
#   costs 0.33 to 0.84 s a step at EP 1 and 0.01 to 0.10 at EP 8, which is a
#   third of the gap and runs the same way as the comparison, and the same
#   strategy repeated UNPROFILED reproduced to 0.013%. So the ordering of those
#   seven points is not sound evidence either way, and this script buys the two
#   things that would make it sound: no instrument, and a spread per strategy.
#
#   run_deepseek_v41_online.sh cannot be used for this. It tees every run of a
#   mode to output/training_demo/deepseek_v41/run_<mode>.log, so repeated tp1
#   runs overwrite each other and only the last survives, and the launcher it
#   reads states train_iters: 1, which leaves no step to harvest.
#
# What it does NOT do: rank anything. It writes raw rows and nothing else, so
# every number is derived off the device from saved output, which is also how
# ND is kept out of the trainer's tree.
#
# Usage:
#   ./repeat_deepseek_v41_runs.sh /path/to/DeepSeek-V4.1-Flash
#
#   POINTS="1:1 1:2"  ./repeat_deepseek_v41_runs.sh <model>   # a subset
#   REPEATS=1         ./repeat_deepseek_v41_runs.sh <model>   # one pass first
#   EDP_SHARD=1       ./repeat_deepseek_v41_runs.sh <model>   # see the note below
#   PROFILE=1         ./repeat_deepseek_v41_runs.sh <model>   # parts, not an order
#   LAUNCHER=<yaml>   ./repeat_deepseek_v41_runs.sh <model>   # time another job
#
#   V41_DISABLE_FUSED_INDEXER=1 POINTS="1:1" RUN_ROOT=<dir> \
#       ./repeat_deepseek_v41_runs.sh <model>
#       The reference leg of the indexer test: the same strategy with the fused
#       Lightning-Indexer off, so the kernel launch count moves and nothing else
#       does. The header says which path was pinned, and the fused path states
#       itself in each run's log, which is the check that the legs really differ.
#
# Interrupted runs: launch it again with the same RUN_ROOT and it skips every
# point already recorded, so a window that closes early can be continued.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
OUTPUT_DIR="${PROJECT_ROOT}/output/training_demo/deepseek_v41"
PYTHON_BIN=${PYTHON_BIN:-python}
# The trainer yaml every run reads. The default is the timing launcher beside this
# script: the released launcher saves checkpoints and runs the grouped expert path,
# neither of which belongs in a round whose only product is a step time. Point it at
# another yaml to time another job, and say which yaml wherever the round is reported.
LAUNCHER=${LAUNCHER:-"${SCRIPT_DIR}/nd_train_deepseek_v41_timing.yaml"}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}

# The seven points measured on N13, as "<ep>:<op>", where op is the FSDP shard
# width the sweep spells dp_shard_size. ND picked 1:1; 2:1 was the fastest of
# everything measured; 16:16 is the human baseline and the one that wins on
# idle alone.
POINTS=${POINTS:-"1:1 1:2 1:4 1:8 1:16 2:1 16:16"}
REPEATS=${REPEATS:-3}
# Steps 1 and 2 are warmup and the data loader's first touch. With no profiler
# there is no window to skip, so everything after SKIP_STEPS is clean.
#
# Six is a CEILING, not a preference, and step 7 dies on every strategy:
# prepare_deepseek_v41_online_data.py writes documents of 24 tokens times 1024
# repetitions plus one, 24577, which is 6 x 4096 + 1, so PlaintextTransform
# cuts each document into six full 4096-token pieces and a ONE-TOKEN tail (the
# appended EOS). Documents 0 to 15 give exactly 96 full pieces, which is six
# steps at a batch of 16, and their sixteen one-token tails stay buffered until
# step 7. There cu_seq_lens carries a boundary of 1, and the V4.1 attention
# refuses it: "packed sample boundaries must align with the CSA2 compression
# ratio 2; misaligned boundaries=[1]". No generator argument avoids it, because
# a document was always 24 x repetitions + 1 tokens, which is odd, and 4096 is
# even, so the tail was odd whatever --sequence-length was passed.
#
# prepare_deepseek_v41_online_data.py now strips the trailing space, which is
# that extra token, so a document is exactly 24576 tokens: six full pieces and
# no tail at all. With data generated by the fixed script this cap can be
# lifted, TRAIN_ITERS=10 SKIP_STEPS=5 as before, and the workload is unchanged
# because every sequence was already one full piece. Delete any jsonl made by
# the old script first, or the old documents are read again.
TRAIN_ITERS=${TRAIN_ITERS:-6}
SKIP_STEPS=${SKIP_STEPS:-2}
GLOBAL_BATCH=${GLOBAL_BATCH:-16}
MICRO_BATCH=${MICRO_BATCH:-1}
SEQ_LEN=${SEQ_LEN:-4096}
# A hung run must not eat the window.
TIMEOUT=${TIMEOUT:-1800}
# How the expert data-parallel group is sharded. "auto" is what the Qwen3.5
# sweep does, max(1, world/ep), which is also what ND's context.expert_shard
# group means. Set it to a fixed number if the round being reproduced used one:
# the three EP1/OP1 repeats are the check, since that point is already known
# twice, 8268.0 and 8173.4 ms profiled, so unprofiled it should read slightly
# under 8173 and not far off it. If it does, the topology matches; if it does
# not, settle that before the other eighteen runs are used for anything.
EDP_SHARD=${EDP_SHARD:-auto}
# The profiler stays OFF for anything being ranked, which is what this script
# exists for. PROFILE=1 turns it on for the one case that needs it: the PARTS,
# meaning the kernel launch count, the Computing column and Free, none of which
# exist without the instrument. Keep the two apart. On this model a single
# profiled run is off the honest step by 3.3% on average and 13.0% at worst, so
# a profiled run measures composition and never an order.
#
# A profiled run also costs much more wall clock than an unprofiled one: the
# default window is steps 3 and 4, and the step after it pays a synchronous CANN
# parse, measured at 166 to 248 s on Qwen3.5. One repeat of one point is enough,
# and the trace lands in <run_dir>/profile.
PROFILE=${PROFILE:-0}

STAMP=$(date -u '+%Y%m%dT%H%M%SZ')
RUN_ROOT=${RUN_ROOT:-"${PROJECT_ROOT}/output/nd_v41_repeats/${STAMP}"}
STEPS_CSV="${RUN_ROOT}/steps.csv"
RUNS_CSV="${RUN_ROOT}/runs.csv"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /path/to/DeepSeek-V4.1-Flash" >&2
    exit 1
fi
MODEL_PATH=$(cd "$1" 2>/dev/null && pwd) || {
    echo "Model directory does not exist: $1" >&2
    exit 1
}
if [[ ! -s "${MODEL_PATH}/config.json" || ! -s "${MODEL_PATH}/tokenizer.json" ]]; then
    echo "Local V4.1 config/tokenizer assets are incomplete: ${MODEL_PATH}" >&2
    exit 1
fi
if ! "${PYTHON_BIN}" -c \
    "import hyper_parallel; from transformers.models.deepseek_v4 import DeepseekV4Config"; then
    echo "The selected Python cannot import this HyperParallel checkout and Transformers" \
        "DeepSeek-V4. Run preflight_deepseek_v41_window.sh first." >&2
    exit 1
fi

ASSETS_PATH="${OUTPUT_DIR}/engram_depth_preserving_d8.json"
# Enough samples that no run wraps its epoch: a wrap re-reads the first packed
# samples, and a misaligned packed boundary is how an EP 16 run failed once
# already. One file, built once, named for what it holds.
SAMPLES=$(( TRAIN_ITERS * GLOBAL_BATCH + GLOBAL_BATCH ))
DATA_PATH="${OUTPUT_DIR}/online_${SEQ_LEN}_${SAMPLES}.jsonl"

mkdir -p "${RUN_ROOT}" "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}" || { echo "cannot enter ${PROJECT_ROOT}" >&2; exit 1; }

if [[ ! -s "${ASSETS_PATH}" ]]; then
    echo "Preparing the Engram assets once"
    "${PYTHON_BIN}" -m examples.training_demo.deepseek_v41.prepare_deepseek_v41_assets \
        --model-dir "${MODEL_PATH}" \
        --output "${ASSETS_PATH}" \
        --bucket-base 4096 \
        --parameter-divisor 8
fi
if [[ ! -s "${DATA_PATH}" ]]; then
    echo "Preparing ${SAMPLES} samples of ${SEQ_LEN} tokens once"
    "${PYTHON_BIN}" -m examples.training_demo.deepseek_v41.prepare_deepseek_v41_online_data \
        --output "${DATA_PATH}" \
        --num-samples "${SAMPLES}" \
        --sequence-length "${SEQ_LEN}"
fi

if [[ ! -s "${STEPS_CSV}" ]]; then
    echo "tag,ep,op,repeat,step,step_time_ms" > "${STEPS_CSV}"
fi
if [[ ! -s "${RUNS_CSV}" ]]; then
    echo "tag,ep,op,repeat,status,steps,mean_ms,min_ms,max_ms,peak_alloc_gb,peak_reserved_gb,seconds,log" \
        > "${RUNS_CSV}"
fi

# Round robin by repeat rather than three runs of a point together: if the node
# drifts over the window, grouping a point's repeats makes the drift look like
# that point's property. Interleaving spreads it over all seven instead.
order=()
for repeat in $(seq 1 "${REPEATS}"); do
    for point in ${POINTS}; do
        order+=("${point}:${repeat}")
    done
done

total=${#order[@]}
started=$(date +%s)
done_count=0
failed=0

duration() {
    local seconds=$1 hours minutes
    hours=$(( seconds / 3600 ))
    minutes=$(( (seconds % 3600) / 60 ))
    if (( hours )); then printf '%dh%02dm' "${hours}" "${minutes}"
    elif (( minutes )); then printf '%dm%02ds' "${minutes}" "$(( seconds % 60 ))"
    else printf '%ds' "${seconds}"; fi
}

echo
echo "===== ${total} launches, ${REPEATS} of each of $(echo "${POINTS}" | wc -w) points"
echo "      tree       ${PROJECT_ROOT}"
echo "      model      ${MODEL_PATH}"
echo "      results    ${RUN_ROOT}"
echo "      iters      ${TRAIN_ITERS}, ranking the steps after ${SKIP_STEPS}"
if [[ "${PROFILE}" == "0" ]]; then
    echo "      profiler   OFF, which is the whole point"
else
    echo "      profiler   ON, so these runs give PARTS and must not be ranked"
fi
if [[ -n "${V41_DISABLE_FUSED_INDEXER:-}" ]]; then
    echo "      indexer    REFERENCE path pinned by V41_DISABLE_FUSED_INDEXER"
fi
echo

for entry in "${order[@]}"; do
    IFS=: read -r ep op repeat <<< "${entry}"
    tag="ep${ep}_op${op}_r${repeat}"
    # Resume skips what SUCCEEDED, not what was attempted: a transient failure
    # should be retried by relaunching, and this model has had one already.
    if awk -F, -v t="${tag}" '$1 == t && $5 == "ok" { ok = 1 } END { exit !ok }' \
            "${RUNS_CSV}" 2>/dev/null; then
        echo "----- ${tag}: already measured, skipped"
        done_count=$(( done_count + 1 ))
        continue
    fi
    # A failed attempt leaves rows behind, and they would be averaged into the
    # retry. Drop this tag's rows from both files before relaunching it.
    if awk -F, -v t="${tag}" '$1 == t { seen = 1 } END { exit !seen }' "${RUNS_CSV}" 2>/dev/null; then
        echo "      dropping an earlier failed attempt's rows for ${tag}"
        for file in "${STEPS_CSV}" "${RUNS_CSV}"; do
            awk -F, -v t="${tag}" '$1 != t' "${file}" > "${file}.keep" && mv "${file}.keep" "${file}"
        done
    fi

    elapsed=$(( $(date +%s) - started ))
    left=""
    if (( done_count )); then
        left=", about $(duration $(( elapsed * (total - done_count) / done_count ))) left"
    fi
    echo "===== run $(( done_count + 1 ))/${total}, $(duration "${elapsed}") elapsed${left}: ${tag}"

    if [[ "${EDP_SHARD}" == "auto" ]]; then
        edp=$(( NPROC_PER_NODE / ep ))
        (( edp < 1 )) && edp=1
    else
        edp=${EDP_SHARD}
    fi

    run_dir="${RUN_ROOT}/${tag}"
    mkdir -p "${run_dir}"
    log="${run_dir}/run.log"
    run_started=$(date +%s)

    # Mirrors run_deepseek_v41_online.sh's tp1 mode, with its own log, its own
    # Built per run because the trace directory is this run's own: a shared one
    # leaves an older profile to be read as if it were this run's, which is the
    # trap that has already cost one measurement.
    if [[ "${PROFILE}" == "0" ]]; then
        profile_args=(--profiling.enabled=false)
    else
        profile_args=(--profiling.enabled=true "--profiling.trace_dir=${run_dir}/profile")
    fi

    # iteration count, and the profiler off unless PROFILE asks for it.
    timeout "${TIMEOUT}" "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --module examples.training_demo.train_text \
        "${LAUNCHER}" \
        --model.config_path="${MODEL_PATH}" \
        --model.engram_assets_path="${ASSETS_PATH}" \
        --dataset.model_assets.tokenizer.pretrained_model_name_or_path="${MODEL_PATH}" \
        --dataset.data_path="${DATA_PATH}" \
        --accelerator.tp_size=1 \
        --accelerator.cp_size=1 \
        --accelerator.pp_size=1 \
        --accelerator.ep_size="${ep}" \
        --fsdp_config.dp_shard_size="${op}" \
        --fsdp_config.edp_shard_size="${edp}" \
        --training.global_batch_size="${GLOBAL_BATCH}" \
        --training.micro_batch_size="${MICRO_BATCH}" \
        --training.train_iters="${TRAIN_ITERS}" \
        "${profile_args[@]}" \
        > "${log}" 2>&1
    code=$?
    run_seconds=$(( $(date +%s) - run_started ))

    # One row a step, and the summary derived from the same rows, so nothing is
    # averaged twice. The trainer's metric line is "step=7 ...
    # performance/step_time=6.2425 ...", in seconds, measured on the slowest rank.
    awk -v tag="${tag}" -v ep="${ep}" -v op="${op}" -v repeat="${repeat}" -v skip="${SKIP_STEPS}" '
        {
            step = ""; value = ""
            for (i = 1; i <= NF; i++) {
                # Split on the "=" rather than counting the prefix: a prefix
                # length off by one reads as zero and looks like a real number.
                if ($i ~ /^step=[0-9]+$/) { step = substr($i, index($i, "=") + 1) }
                else if ($i ~ /^performance\/step_time=/) { value = substr($i, index($i, "=") + 1) }
            }
            if (step != "" && value != "" && step + 0 > skip + 0) {
                printf "%s,%s,%s,%s,%d,%.3f\n", tag, ep, op, repeat, step, value * 1000.0
            }
        }' "${log}" >> "${STEPS_CSV}"

    read -r count mean low high < <(awk -F, -v tag="${tag}" '
        $1 == tag { n++; sum += $6; if (low == "" || $6 < low) low = $6; if ($6 > high) high = $6 }
        END { printf "%d %.3f %.3f %.3f", n + 0, (n ? sum / n : 0), low + 0, high + 0 }' "${STEPS_CSV}")

    read -r alloc reserved < <(awk '
        {
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^memory\/device_max_allocated_gb=/) {
                    value = substr($i, index($i, "=") + 1) + 0; if (value > a) a = value
                } else if ($i ~ /^memory\/device_max_reserved_gb=/) {
                    value = substr($i, index($i, "=") + 1) + 0; if (value > r) r = value
                }
            }
        }
        END { printf "%.3f %.3f", a + 0, r + 0 }' "${log}")

    if [[ "${code}" -eq 0 && "${count}" -gt 0 ]]; then
        status=ok
        printf '      %s steps, mean %.1f ms, %.1f to %.1f, peak %.1f / %.1f GiB, %s\n' \
            "${count}" "${mean}" "${low}" "${high}" "${alloc}" "${reserved}" \
            "$(duration "${run_seconds}")"
    else
        status="failed_${code}"
        failed=$(( failed + 1 ))
        # Keep going: one transient failure already happened on this model, a
        # packed sample boundary not aligned to the CSA2 compression ratio,
        # which then succeeded unchanged. Losing the window to it would be worse.
        echo "      FAILED (exit ${code}, ${count} steps parsed) after $(duration "${run_seconds}")"
        echo "      last lines of ${log#"${PROJECT_ROOT}/"}:"
        tail -3 "${log}" | sed 's/^/        /'
    fi

    printf '%s,%s,%s,%s,%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s\n' \
        "${tag}" "${ep}" "${op}" "${repeat}" "${status}" "${count}" \
        "${mean}" "${low}" "${high}" "${alloc}" "${reserved}" \
        "${run_seconds}" "${log#"${PROJECT_ROOT}/"}" >> "${RUNS_CSV}"

    done_count=$(( done_count + 1 ))
done

echo
echo "===== ${done_count} launches in $(duration $(( $(date +%s) - started ))), ${failed} failed"
echo "      ${RUNS_CSV#"${PROJECT_ROOT}/"}"
echo "      ${STEPS_CSV#"${PROJECT_ROOT}/"}"
echo
echo "Per strategy, every repeat and the spread across them:"
printf '  %-12s %5s  %9s  %9s  %9s  %7s\n' "strategy" "runs" "mean ms" "min" "max" "spread"
awk -F, 'NR > 1 && $5 == "ok" {
        key = "EP" $2 " OP" $3
        n[key]++; sum[key] += $7
        if (!(key in lo) || $7 < lo[key]) lo[key] = $7
        if ($7 > hi[key]) hi[key] = $7
    }
    END {
        for (key in n) {
            mean = sum[key] / n[key]
            printf "  %-12s %5d  %9.1f  %9.1f  %9.1f  %6.2f%%\n", \
                key, n[key], mean, lo[key], hi[key], 100 * (hi[key] - lo[key]) / mean
        }
    }' "${RUNS_CSV}" | sort -k2,2n -k1,1
echo
echo "Send back both CSVs and the run directory. Do not rank them here: the"
echo "comparison belongs beside ND, off the device."
if (( failed )); then
    exit 1
fi
