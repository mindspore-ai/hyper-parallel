#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------
# Shared host GCC version policy, aligned with mindspore/CMakeLists.txt:
#   - require GCC >= 7.3.0 (fatal on lower)
#   - warn if GCC >  11.3.0 (non-fatal)
# Source this file from build scripts (do NOT execute), e.g.:
#   source "$(dirname "$0")/check_gcc_version.sh"
#   check_gcc_version
# Or call: check_gcc_version <gcc-binary>   to check a specific compiler.

check_gcc_version() {
    local gcc_bin="${1:-gcc}"
    if ! command -v "${gcc_bin}" >/dev/null 2>&1; then
        echo "ERROR: ${gcc_bin} not found. hyper_parallel requires GCC >= 7.3.0." >&2
        return 1
    fi

    local ver
    ver="$("${gcc_bin}" -dumpfullversion -dumpversion 2>/dev/null | head -1)"
    if [ -z "${ver}" ]; then
        echo "ERROR: cannot determine version of ${gcc_bin}." >&2
        return 1
    fi

    local major minor patch
    IFS='.' read -r major minor patch <<< "${ver}"
    major="${major:-0}"; minor="${minor:-0}"; patch="${patch:-0}"
    local num=$((major * 10000 + minor * 100 + patch))

    if [ "${num}" -lt 70300 ]; then
        echo "ERROR: GCC version ${ver} < 7.3.0. Install GCC >= 7.3.0 (mindspore-compatible)." >&2
        return 1
    fi
    if [ "${num}" -gt 110300 ]; then
        echo "WARNING: GCC version ${ver} > 11.3.0. May cause unknown problems." >&2
    fi
    echo "INFO: GCC ${ver} accepted (target range [7.3.0, 11.3.0])."
    return 0
}
