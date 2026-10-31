#!/usr/bin/env bash
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
set -euo pipefail
usage() {
    cat <<'EOF'
Usage: bash run_alignment.sh VARIANT DATA_PREFIX INDEX_PREFIX OUTPUT_DIR [options]
VARIANT: single | fsdp4 | fsdp4_cp2 | fsdp4_selective | fsdp4_ep2
Options:
  --resume CHECKPOINT   Restore a complete HP native training checkpoint.
  --from-scratch        Explicitly start a non-baseline variant from random weights.
  --dry-run             Validate arguments and print the command without training.
  --field=value         Additional native Trainer overrides, recorded with the run.
Set ASCEND_RT_VISIBLE_DEVICES to one or four distinct free NPU IDs.
INDEX_PREFIX precedes "-document_index.npy", "-sample_index.npy", "-shuffle_index.npy".
EOF
}
fail() { printf '%s\n' "$*" >&2; exit 2; }
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
[[ $# -ge 4 ]] || { usage >&2; exit 2; }
variant=$1
data_prefix=$2
index_prefix=$3
output_dir=$4
shift 4
case "${variant}" in
    single) nproc=1 ;;
    fsdp4|fsdp4_cp2|fsdp4_selective|fsdp4_ep2) nproc=4 ;;
    *) fail "Unknown variant: ${variant}" ;;
esac
resume=""
from_scratch=0
dry_run=0
overrides=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            [[ $# -ge 2 && -n "$2" ]] || fail "--resume requires a checkpoint directory"
            resume=$2
            shift 2 ;;
        --from-scratch) from_scratch=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --*=*)
            key=${1%%=*}
            case "${key}" in
                --checkpoint|--checkpoint.checkpoint_dir|--checkpoint.restore_from|\
                --checkpoint.restore_optimizer|--checkpoint.restore_train_state|--checkpoint.restore_dataloader_state|\
                --dataset|--dataset._target_|--dataset.data_prefix|--dataset.index_paths|--dataset.index_paths.*)
                    fail "Protected field ${key}; use the positional input/output paths and --resume" ;;
            esac
            overrides+=("$1")
            shift ;;
        *) fail "Unknown argument: $1" ;;
    esac
done
[[ -z "${resume}" || "${from_scratch}" == 0 ]] || fail "--resume and --from-scratch conflict"
if [[ "${variant}" != fsdp4 && -z "${resume}" && "${from_scratch}" == 0 ]]; then
    fail "Continuation requires --resume; use --from-scratch for a separate initialization run"
fi
[[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]] || fail "Set ASCEND_RT_VISIBLE_DEVICES explicitly"
[[ "${ASCEND_RT_VISIBLE_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail "Use comma-separated integer NPU IDs"
IFS=',' read -r -a devices <<< "${ASCEND_RT_VISIBLE_DEVICES}"
[[ ${#devices[@]} -eq ${nproc} ]] || fail "Expected ${nproc} visible NPU IDs"
declare -A seen=()
for device in "${devices[@]}"; do
    [[ -z "${seen[${device}]:-}" ]] || fail "Duplicate NPU ID: ${device}"
    seen[${device}]=1
done
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "${script_dir}/../../.." && pwd)
config="${script_dir}/alignment_configs/${variant}.yaml"
[[ -f "${config}" ]] || fail "Missing configuration: ${config}"
data_prefix=$(realpath -m -- "${data_prefix}")
index_prefix=$(realpath -m -- "${index_prefix}")
output_dir=$(realpath -m -- "${output_dir}")
for path in "${data_prefix}.bin" "${data_prefix}.idx" \
    "${index_prefix}-document_index.npy" "${index_prefix}-sample_index.npy" "${index_prefix}-shuffle_index.npy"; do
    [[ -f "${path}" && -s "${path}" && -r "${path}" ]] || fail "Missing or unreadable input: ${path}"
done
[[ ! -e "${output_dir}" ]] || fail "Output directory exists; choose a new run directory"
python_bin="${PYTHON:-python}"
yaml_string() {
    "${python_bin}" -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$1"
}
restore=null
if [[ -n "${resume}" ]]; then
    resume=$(realpath -e -- "${resume}")
    [[ -d "${resume}" && -r "${resume}" ]] || fail "Checkpoint must be a readable HP native directory"
    restore=$(yaml_string "${resume}")
fi
# Expose the example-owned adapter only to the training subprocess.
command=(env "PYTHONPATH=${script_dir}${PYTHONPATH:+:${PYTHONPATH}}" "${python_bin}" -m torch.distributed.run --standalone "--nproc_per_node=${nproc}"
    "${project_root}/examples/training_demo/train_text.py" "${config}"
    "--dataset.data_prefix=$(yaml_string "${data_prefix}")"
    "--dataset.index_paths.document_index=$(yaml_string "${index_prefix}-document_index.npy")"
    "--dataset.index_paths.sample_index=$(yaml_string "${index_prefix}-sample_index.npy")"
    "--dataset.index_paths.shuffle_index=$(yaml_string "${index_prefix}-shuffle_index.npy")"
    "--checkpoint.checkpoint_dir=$(yaml_string "${output_dir}/checkpoints")"
    "--checkpoint.restore_from=${restore}"
    "--checkpoint.restore_optimizer=true"
    "--checkpoint.restore_train_state=true"
    "--checkpoint.restore_dataloader_state=true"
    "${overrides[@]}")
printf 'Command: '; printf '%q ' "${command[@]}"; printf '\n'
if [[ "${dry_run}" == 1 ]]; then exit 0; fi
cd -- "${project_root}"
mkdir -p -- "$(dirname -- "${output_dir}")"
mkdir -- "${output_dir}"
cp -- "${config}" "${output_dir}/input.yaml"
git rev-parse HEAD > "${output_dir}/revision.txt"
git diff --binary HEAD > "${output_dir}/worktree.patch"
"${python_bin}" -m pip freeze > "${output_dir}/environment.txt"
{
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'cd -- %q\n' "${project_root}"
    printf 'export ASCEND_RT_VISIBLE_DEVICES=%q\n' "${ASCEND_RT_VISIBLE_DEVICES}"
    if [[ -n "${HCCL_IF_BASE_PORT:-}" ]]; then printf 'export HCCL_IF_BASE_PORT=%q\n' "${HCCL_IF_BASE_PORT}"; fi
    printf 'exec '; printf '%q ' "${command[@]}"; printf '\n'
} > "${output_dir}/command.bash"
if command -v npu-smi >/dev/null 2>&1; then npu-smi info > "${output_dir}/npu-before.txt"; fi
"${command[@]}" 2>&1 | tee "${output_dir}/train.log"
