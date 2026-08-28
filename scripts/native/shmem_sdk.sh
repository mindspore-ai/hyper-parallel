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
# Shared SHMEM SDK preparation for symmetric memory and multicore component builds.

function hp_prepare_shmem_sdk() {
    local project_root=$1
    local soc_list=$2
    local native_jobs=$3
    local clean_sdk=${4:-off}
    local native_root="${project_root}/build/native"
    local locked_source_dir="${native_root}/deps/shmem/src"
    local wrapper="${project_root}/scripts/native/cmake/shmem_wrapper"
    local requested_soc
    local -a requested_socs=()

    if [[ -z "${ASCEND_HOME_PATH:-}" || ! -d "${ASCEND_HOME_PATH}" ]]; then
        echo "HP_NATIVE_REASON_CODE=CANN_ENV_NOT_CONFIGURED"
        echo "ERROR: ASCEND_HOME_PATH must identify the selected CANN installation." >&2
        echo "       Source the required CANN set_env.sh before invoking the native build." >&2
        return 3
    fi
    local cann_version_file="${ASCEND_HOME_PATH}/opp/version.info"
    local cann_version
    cann_version=$(awk -F= '$1 == "Version" {print $2}' "${cann_version_file}" 2>/dev/null || true)
    if [[ "${cann_version}" != "9.1.0" ]]; then
        echo "HP_NATIVE_REASON_CODE=UNSUPPORTED_CANN_VERSION"
        echo "ERROR: CANN 9.1.0 is required, found '${cann_version:-unknown}' under ${ASCEND_HOME_PATH}." >&2
        return 3
    fi
    for required_tool in awk bisheng cmake gcc g++ git make python3 readlink sha256sum tar; do
        if ! command -v "${required_tool}" >/dev/null 2>&1; then
            echo "HP_NATIVE_REASON_CODE=BUILD_TOOL_NOT_FOUND"
            echo "ERROR: Required SHMEM SDK build tool not found on PATH: ${required_tool}." >&2
            return 4
        fi
    done

    local cann_root
    local bisheng_path
    local gcc_path
    local gxx_path
    local host_arch
    local toolchain_key
    local work_root
    local source_dir
    local build_dir
    local install_root
    cann_root=$(cd "${ASCEND_HOME_PATH}" && pwd -P)
    bisheng_path=$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
        "$(command -v bisheng)")
    gcc_path=$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
        "$(command -v gcc)")
    gxx_path=$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
        "$(command -v g++)")
    host_arch=$(uname -m)
    toolchain_key=$(
        {
            printf '%s\n' \
                "schema=2" \
                "cann_root=${cann_root}" \
                "bisheng=${bisheng_path}" \
                "host_arch=${host_arch}" \
                "gcc=${gcc_path}" \
                "gxx=${gxx_path}"
            "${bisheng_path}" --version 2>&1 || true
            "${gcc_path}" --version 2>&1 || true
            "${gxx_path}" --version 2>&1 || true
            sha256sum "${cann_version_file}" \
                "${project_root}/scripts/native/config/dependencies.lock.json" \
                "${project_root}/scripts/native/cmake/shmem_wrapper/CMakeLists.txt" \
                "${project_root}/scripts/native/shmem_sdk.sh"
        } | sha256sum | awk '{print substr($1, 1, 16)}'
    )
    if [[ ! "${toolchain_key}" =~ ^[0-9a-f]{16}$ ]]; then
        echo "HP_NATIVE_REASON_CODE=SHMEM_TOOLCHAIN_KEY_INVALID"
        echo "ERROR: Cannot derive a stable SHMEM CANN toolchain identity." >&2
        return 4
    fi
    work_root="${native_root}/work/symmetric_memory/toolchain-${toolchain_key}"
    source_dir="${work_root}/shmem-source"
    build_dir="${work_root}/shmem"
    install_root="${work_root}/shmem-install"

    IFS=',' read -r -a requested_socs <<< "${soc_list}"
    for requested_soc in "${requested_socs[@]}"; do
        requested_soc="${requested_soc//[[:space:]]/}"
        case "${requested_soc}" in
            ascend910b|ascend910_93)
                ;;
            ascend950)
                echo "HP_NATIVE_REASON_CODE=SOC_SOURCE_NOT_SUPPORTED"
                echo "ERROR: SHMEM v1.6.0 does not support ascend950 in this build." >&2
                return 7
                ;;
            *)
                echo "HP_NATIVE_REASON_CODE=UNSUPPORTED_SOC_SELECTION"
                echo "ERROR: Unsupported SoC '${requested_soc}' in --soc-list=${soc_list}." >&2
                return 8
                ;;
        esac
    done

    python3 -m scripts.native.prepare_dependencies --dependency shmem
    if [[ ! -d "${locked_source_dir}/src" || ! -f "${locked_source_dir}/CMakeLists.txt" ]]; then
        echo "HP_NATIVE_REASON_CODE=SHMEM_SOURCE_NOT_PREPARED"
        echo "ERROR: Pinned SHMEM source is not prepared at ${locked_source_dir}." >&2
        return 6
    fi

    if [[ "${clean_sdk}" == "on" ]]; then
        rm -rf "${build_dir}" "${install_root}"
    fi
    rm -rf "${source_dir}"
    mkdir -p "${source_dir}" "${build_dir}"
    git -C "${locked_source_dir}" archive --format=tar HEAD | tar -xf - -C "${source_dir}"
    rm -rf "${install_root}"

    echo "INFO: preparing pinned CANN SHMEM SDK"
    echo "  locked source: ${locked_source_dir}"
    echo "  isolated source: ${source_dir}"
    echo "  CANN toolchain: ${cann_root} (${bisheng_path})"
    echo "  host toolchain: ${host_arch} (${gcc_path}, ${gxx_path})"
    echo "  soc family: Ascend910B (used by ascend910b and ascend910_93)"
    echo "  install root: ${install_root}"

    cmake -S "${wrapper}" \
        -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DHP_SHMEM_SOURCE_DIR="${source_dir}" \
        -DUSE_CXX11_ABI=ON \
        -DUSE_UNIT_TEST=OFF \
        -DUSE_EXAMPLES=OFF \
        -DBUILD_PYTHON=OFF \
        -DCMAKE_DISABLE_FIND_PACKAGE_MPI=TRUE \
        -DSOC_TYPE=Ascend910B
    cmake --build "${build_dir}" --parallel "${native_jobs}"
    cmake --install "${build_dir}" --prefix "${install_root}/shmem"

    HP_SHMEM_INSTALL_ROOT="${install_root}"
    HP_SHMEM_TOOLCHAIN_KEY="${toolchain_key}"
    export HP_SHMEM_INSTALL_ROOT HP_SHMEM_TOOLCHAIN_KEY
}
