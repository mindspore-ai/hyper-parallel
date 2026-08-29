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
# Shared CANN toolkit version checks for optional native component builds.

HP_MINIMUM_CANN_VERSION="9.1.0"

function hp_read_cann_version() {
    local ascend_home_path=$1
    awk -F= '$1 == "Version" {print $2}' "${ascend_home_path}/opp/version.info" 2>/dev/null || true
}

function hp_cann_version_at_least() {
    local actual_version=${1:-}
    local minimum_version=${2:-${HP_MINIMUM_CANN_VERSION}}
    local actual_core=${actual_version%%[-+]*}
    local minimum_core=${minimum_version%%[-+]*}
    local -a actual_parts=()
    local -a minimum_parts=()
    local part_index

    IFS=. read -r -a actual_parts <<< "${actual_core}"
    IFS=. read -r -a minimum_parts <<< "${minimum_core}"
    if [[ ${#actual_parts[@]} -ne 3 || ${#minimum_parts[@]} -ne 3 ]]; then
        return 1
    fi
    for part_index in 0 1 2; do
        if [[ ! "${actual_parts[${part_index}]}" =~ ^[0-9]+$ ||
              ! "${minimum_parts[${part_index}]}" =~ ^[0-9]+$ ]]; then
            return 1
        fi
        if ((10#${actual_parts[${part_index}]} > 10#${minimum_parts[${part_index}]})); then
            return 0
        fi
        if ((10#${actual_parts[${part_index}]} < 10#${minimum_parts[${part_index}]})); then
            return 1
        fi
    done
    return 0
}
