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
# Private SHMEM SDK preparation for Multicore builds.

_HP_NATIVE_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
if ! declare -F hp_cann_version_at_least >/dev/null; then
    source "${_HP_NATIVE_SCRIPT_DIR}/../../../../../scripts/native/cann_version.sh"
fi
unset _HP_NATIVE_SCRIPT_DIR
HP_MINIMUM_CANN_VERSION="9.1.0"

function hp_shmem_soc_type() {
    local soc_list=$1
    local selected_type=""
    local requested_soc
    local soc_type
    local -a requested_socs=()
    IFS=',' read -r -a requested_socs <<< "${soc_list}"
    for requested_soc in "${requested_socs[@]}"; do
        requested_soc="${requested_soc//[[:space:]]/}"
        case "${requested_soc}" in
            ascend910b|ascend910_93) soc_type=Ascend910B ;;
            ascend950) soc_type=Ascend950 ;;
            *)
                echo "ERROR: Unsupported SHMEM SoC '${requested_soc}'." >&2
                return 10
                ;;
        esac
        if [[ -n "${selected_type}" && "${selected_type}" != "${soc_type}" ]]; then
            echo "ERROR: Prepare each SHMEM backend separately; cannot use one SDK for ${soc_list}." >&2
            return 10
        fi
        selected_type=${soc_type}
    done
    [[ -n "${selected_type}" ]] || return 10
    printf '%s\n' "${selected_type}"
}

function hp_prepare_shmem_sdk() {
    local project_root=$1
    local soc_list=$2
    local native_jobs=$3
    local clean_sdk=${4:-off}
    local native_root="${project_root}/build/native"
    local python_bin=python
    local locked_source_dir="${native_root}/deps/shmem/src"
    local wrapper="${project_root}/hyper_parallel/core/multicore/shmem/cmake/shmem_wrapper"
    local soc_type
    soc_type=$(hp_shmem_soc_type "${soc_list}") || return $?

    if [[ -z "${ASCEND_HOME_PATH:-}" || ! -d "${ASCEND_HOME_PATH}" ]]; then
        echo "HP_NATIVE_REASON_CODE=CANN_ENV_NOT_CONFIGURED"
        echo "ERROR: ASCEND_HOME_PATH must identify the selected CANN installation." >&2
        echo "       Source the required CANN set_env.sh before invoking the native build." >&2
        return 3
    fi
    local cann_version
    local cann_version_file="${ASCEND_HOME_PATH}/opp/version.info"
    cann_version=$(hp_read_cann_version "${ASCEND_HOME_PATH}")
    if ! hp_cann_version_at_least "${cann_version}" "${HP_MINIMUM_CANN_VERSION}"; then
        echo "HP_NATIVE_REASON_CODE=UNSUPPORTED_CANN_VERSION"
        echo "ERROR: CANN >= ${HP_MINIMUM_CANN_VERSION} is required, found " \
            "'${cann_version:-unknown}' under ${ASCEND_HOME_PATH}." >&2
        return 3
    fi
    if [[ "${soc_type}" == Ascend950 && ! -s "${ASCEND_HOME_PATH}/lib64/libhcomm.so" ]]; then
        echo "HP_NATIVE_REASON_CODE=A5_HCOMM_LIBRARY_NOT_FOUND"
        echo "ERROR: Ascend950 SHMEM requires lib64/libhcomm.so in the selected CANN installation." >&2
        return 3
    fi
    if ! command -v "${python_bin}" >/dev/null 2>&1; then
        echo "HP_NATIVE_REASON_CODE=PYTHON_BUILD_DEPENDENCY_NOT_FOUND"
        echo "ERROR: The selected Python executable is unavailable: ${python_bin}." >&2
        return 4
    fi
    python_bin=$(command -v "${python_bin}")
    for required_tool in awk bisheng cmake gcc g++ git make readlink sha256sum tar; do
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
    local shmem_lock_digest
    local toolchain_key
    local work_root
    local source_dir
    local build_dir
    local install_root
    cann_root=$(cd "${ASCEND_HOME_PATH}" && pwd -P)
    bisheng_path=$("${python_bin}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
        "$(command -v bisheng)")
    gcc_path=$("${python_bin}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
        "$(command -v gcc)")
    gxx_path=$("${python_bin}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
        "$(command -v g++)")
    host_arch=$(uname -m)
    if ! shmem_lock_digest=$("${python_bin}" -c '
import hashlib
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    components = json.load(source)["components"]["multicore"]
inputs = {"shmem": components["shmem"]}
if sys.argv[2] == "Ascend950":
    inputs["nlohmann_json"] = components["nlohmann_json"]
encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(encoded).hexdigest())
' "${project_root}/hyper_parallel/core/multicore/_build/dependencies.lock.json" "${soc_type}"); then
        echo "HP_NATIVE_REASON_CODE=SHMEM_LOCK_INVALID"
        echo "ERROR: Cannot read the locked SHMEM dependency contract." >&2
        return 4
    fi
    toolchain_key=$(
        {
            printf '%s\n' \
                "schema=3" \
                "soc_type=${soc_type}" \
                "cann_root=${cann_root}" \
                "bisheng=${bisheng_path}" \
                "host_arch=${host_arch}" \
                "gcc=${gcc_path}" \
                "gxx=${gxx_path}" \
                "shmem_lock=${shmem_lock_digest}"
            "${bisheng_path}" --version 2>&1 || true
            "${gcc_path}" --version 2>&1 || true
            "${gxx_path}" --version 2>&1 || true
            sha256sum "${cann_version_file}" \
                "${project_root}/hyper_parallel/core/multicore/cmake/hardening.cmake" \
                "${project_root}/hyper_parallel/core/multicore/shmem/cmake/shmem_wrapper/CMakeLists.txt" \
                "${project_root}/hyper_parallel/core/multicore/shmem/_build/shmem_sdk.sh"
            if [[ "${soc_type}" == Ascend950 ]]; then
                sha256sum "${ASCEND_HOME_PATH}/lib64/libhcomm.so"
            fi
        } | sha256sum | awk '{print substr($1, 1, 16)}'
    )
    if [[ ! "${toolchain_key}" =~ ^[0-9a-f]{16}$ ]]; then
        echo "HP_NATIVE_REASON_CODE=SHMEM_TOOLCHAIN_KEY_INVALID"
        echo "ERROR: Cannot derive a stable SHMEM CANN toolchain identity." >&2
        return 4
    fi
    work_root="${native_root}/work/multicore/shmem/toolchain-${toolchain_key}"
    source_dir="${work_root}/shmem-source"
    build_dir="${work_root}/shmem"
    install_root="${work_root}/shmem-install"

    "${python_bin}" "${project_root}/hyper_parallel/core/multicore/_build/prepare_dependencies.py" \
        --dependency shmem || return $?
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
    git -C "${locked_source_dir}" archive --format=tar HEAD | tar -xf - -C "${source_dir}" || return $?
    if ! "${python_bin}" - "${project_root}" "${source_dir}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

project_root, source_dir = (Path(value).resolve() for value in sys.argv[1:])
lock_path = project_root / "hyper_parallel/core/multicore/_build/dependencies.lock.json"
lock = json.loads(lock_path.read_text(encoding="utf-8"))["components"]["multicore"]["shmem"]
environment = os.environ.copy()
environment["GIT_CEILING_DIRECTORIES"] = str(source_dir.parent)
for adapter in lock.get("patches", []):
    adapter_path = project_root / adapter["path"]
    actual_hash = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    if actual_hash != adapter["sha256"]:
        raise ValueError(f"SHMEM adapter hash mismatch: {adapter_path}")
    for options in (("--check",), (), ("--check", "--reverse")):
        subprocess.run(
            ["git", "apply", *options, str(adapter_path)],
            cwd=source_dir,
            env=environment,
            check=True,
        )
PY
    then
        echo "HP_NATIVE_REASON_CODE=SHMEM_ADAPTER_FAILED"
        echo "ERROR: Cannot apply the locked SHMEM compatibility adapter." >&2
        return 6
    fi
    rm -rf "${install_root}"

    if [[ "${soc_type}" == Ascend950 ]]; then
        "${python_bin}" "${project_root}/hyper_parallel/core/multicore/_build/prepare_dependencies.py" \
            --dependency nlohmann_json || return $?
        mkdir -p "${source_dir}/3rdparty/json/single_include/nlohmann"
        cp "${native_root}/deps/nlohmann_json/src/single_include/nlohmann/json.hpp" \
            "${source_dir}/3rdparty/json/single_include/nlohmann/json.hpp"
    fi

    echo "INFO: preparing pinned CANN SHMEM SDK"
    echo "  locked source: ${locked_source_dir}"
    echo "  isolated source: ${source_dir}"
    echo "  CANN toolchain: ${cann_root} (${bisheng_path})"
    echo "  host toolchain: ${host_arch} (${gcc_path}, ${gxx_path})"
    echo "  soc family: ${soc_type}"
    echo "  install root: ${install_root}"

    cmake -S "${wrapper}" \
        -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DHP_SHMEM_SOURCE_DIR="${source_dir}" \
        -DUSE_CXX11_ABI=ON \
        -DUSE_UNIT_TEST=OFF \
        -DUSE_EXAMPLES=OFF \
        -DBUILD_PYTHON=OFF \
        -DACLSHMEM_RDMA_SUPPORT=OFF \
        -DCMAKE_DISABLE_FIND_PACKAGE_MPI=TRUE \
        -DSOC_TYPE="${soc_type}" || return $?
    cmake --build "${build_dir}" --parallel "${native_jobs}" || return $?
    cmake --install "${build_dir}" --prefix "${install_root}/shmem" || return $?

    HP_SHMEM_INSTALL_ROOT="${install_root}"
    HP_SHMEM_TOOLCHAIN_KEY="${toolchain_key}"
    HP_SHMEM_SOC_TYPE="${soc_type}"
    export HP_SHMEM_INSTALL_ROOT HP_SHMEM_TOOLCHAIN_KEY HP_SHMEM_SOC_TYPE
}
