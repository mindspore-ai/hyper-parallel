#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd.
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

# Build into the component staging tree; the caller publishes it after ELF validation.
function hp_build_routed_experts() (
    local source_root=$1
    local work_root=$2
    local output_root=$3
    local python_bin=$4
    local native_jobs=$5
    local build_type=$6
    local target
    local target_source
    local library
    local -a configure_args=()

    unset HP_MULTICORE_VENDOR_ROOT CANN_VENDOR_LIBDIR
    rm -rf "${work_root}/kernels" "${work_root}/torch" || return "$?"
    mkdir -p "${work_root}" "${output_root}" || return "$?"
    rm -f "${output_root}/libhyper_parallel_routed_experts_kernels.so" \
        "${output_root}/libhyper_parallel_routed_experts_torch.so" || return "$?"
    for target in kernels torch; do
        if [[ "${target}" == "kernels" ]]; then
            target_source="${source_root}/ops/routed_experts"
            configure_args=()
        else
            target_source="${source_root}/torch/routed_experts"
            configure_args=(
                "-DPython3_EXECUTABLE=${python_bin}"
                "-DROUTED_KERNEL_LIBRARY=${output_root}/libhyper_parallel_routed_experts_kernels.so"
            )
        fi
        cmake -S "${target_source}" -B "${work_root}/${target}" \
            -DCMAKE_BUILD_TYPE="${build_type}" \
            -DCMAKE_INSTALL_PREFIX="${output_root}" \
            "${configure_args[@]}" || return "$?"
        cmake --build "${work_root}/${target}" --parallel "${native_jobs}" || return "$?"
        cmake --install "${work_root}/${target}" || return "$?"
        library="${output_root}/libhyper_parallel_routed_experts_${target}.so"
        if [[ ! -s "${library}" ]]; then
            echo "HP_NATIVE_REASON_CODE=ROUTED_EXPERTS_ARTIFACT_MISSING"
            echo "ERROR: Missing independent routed-expert library: ${library}." >&2
            return 13
        fi
    done
)

function hp_validate_routed_experts_library() {
    local library=$1
    local required_dependency=$2
    local expected_soname=${library##*/}
    local dynamic_section
    local soname
    local dependencies
    dynamic_section=$(readelf -d "${library}") || return "$?"
    soname=$(awk '/\(SONAME\)/ {split($0, parts, /[][]/); print parts[2]}' <<< "${dynamic_section}")
    dependencies=$(awk '/\(NEEDED\)/ {split($0, parts, /[][]/); print parts[2]}' <<< "${dynamic_section}")
    if [[ "${soname}" != "${expected_soname}" ]]; then
        echo "HP_NATIVE_REASON_CODE=ROUTED_EXPERTS_SONAME_INVALID"
        echo "ERROR: Expected SONAME ${expected_soname} in ${library}." >&2
        return 12
    fi
    if ! grep -Fxq "${required_dependency}" <<< "${dependencies}"; then
        echo "HP_NATIVE_REASON_CODE=ROUTED_EXPERTS_DEPENDENCY_MISSING"
        echo "ERROR: ${library} must depend on ${required_dependency}." >&2
        return 12
    fi
    if grep -q '/' <<< "${dependencies}"; then
        echo "HP_NATIVE_REASON_CODE=ROUTED_EXPERTS_DEPENDENCY_PATH_FOUND"
        echo "ERROR: ${library} contains a build-machine path in DT_NEEDED." >&2
        return 12
    fi
    if grep -Eq '^(libcust_opapi|libhyper_parallel_(mega|multicore)|libshmem)' <<< "${dependencies}"; then
        echo "HP_NATIVE_REASON_CODE=ROUTED_EXPERTS_FORBIDDEN_DEPENDENCY"
        echo "ERROR: ${library} depends on a Mega vendor/adapter or generic SHMEM library." >&2
        return 12
    fi
}
