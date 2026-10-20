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
# Admission guard for shared accelerator nodes. Run it on the target node as
# the FIRST statement of every launch, and again after the job exits.
#
#   node_guard.sh [pre]        exit 0 = idle, safe to launch
#                              exit 3 = busy (someone else holds the node)
#                              exit 2 = cannot determine -> treat as busy
#   node_guard.sh post         exit 0 = no residue left by this user
#                              exit 3 = residual processes found (listed)
#   node_guard.sh self-test    exit 0 = parser passes its positive controls
#
# Discipline encoded here (each clause cost a real incident):
#   * probe at launch time, every launch - a scan minutes ago proves nothing;
#   * judge by foreign USERS and per-die HBM, never by known command names
#     (scheduler-dispatched jobs match neither torchrun nor pretrain);
#   * the HBM column is fixed-width: a five-digit used value consumes the
#     space before the slash ("36004/ 65536" vs "3158 / 65536") - parse with
#     optional spaces on BOTH sides or a full node reads as idle;
#   * if the node is busy: do not queue, do not touch the occupant - leave,
#     and re-probe after a while or pick another authorized node;
#   * after your job: verify release (processes AND device memory), do not
#     just assume the launcher cleaned up.
#
# Environment knobs (all optional):
#   HBM_TOTAL_MB   per-die HBM total as printed by npu-smi (default 65536)
#   HBM_BUSY_MB    per-die used threshold meaning "occupied" (default 5000)
#   ALLOW_USERS    extra system users to ignore, space-separated

set -u
MODE=${1:-pre}
HBM_TOTAL_MB=${HBM_TOTAL_MB:-65536}
HBM_BUSY_MB=${HBM_BUSY_MB:-5000}
ALLOW_USERS=${ALLOW_USERS:-}
SELF_USER=$(id -un)

hbm_used_list() {  # stdin: npu-smi info text -> one used-MB value per line
    grep -oE "[0-9]+ */ *${HBM_TOTAL_MB}" | grep -oE "^[0-9]+"
}

foreign_users() {  # compute-looking processes owned by neither us nor system
    ps -eo user,args --no-headers 2>/dev/null \
        | grep -iE "python|torch|train|experiment|mindspeed|node_worker" \
        | grep -v "[g]rep" \
        | awk '{print $1}' \
        | grep -vxF "${SELF_USER}" | grep -vxE "root|dbus|datatur.?" \
        | { [ -n "${ALLOW_USERS}" ] && grep -vxF -e ${ALLOW_USERS} || cat; } \
        | sort -u
}

own_residue() {    # our own leftover compute processes
    ps -eo user,pid,args --no-headers 2>/dev/null \
        | awk -v u="${SELF_USER}" '$1 == u' \
        | grep -E "[t]orch\.distributed|[t]orchrun|[t]rain_text|[p]ython -u -m" \
        | grep -v "[g]rep"
}

case "${MODE}" in
pre)
    f=$(foreign_users | tr '\n' ',')
    if [ -n "${f}" ]; then echo "BUSY foreign_users=${f%,}"; exit 3; fi
    smi=$(npu-smi info 2>/dev/null)
    used=$(printf '%s\n' "${smi}" | hbm_used_list)
    if [ -z "${used}" ]; then
        echo "UNDETERMINED: no '<used>/ ${HBM_TOTAL_MB}' lines parsed - fail closed"
        exit 2
    fi
    mx=$(printf '%s\n' "${used}" | sort -rn | head -1)
    busy=$(printf '%s\n' "${used}" | awk -v t="${HBM_BUSY_MB}" '$1 > t' | wc -l)
    if [ "${busy}" -gt 0 ]; then echo "BUSY dies=${busy} maxHBM=${mx}MB"; exit 3; fi
    echo "IDLE maxHBM=${mx}MB"; exit 0 ;;
post)
    left=$(own_residue)
    if [ -n "${left}" ]; then
        echo "RESIDUE:"; printf '%s\n' "${left}" | head -10; exit 3
    fi
    echo "CLEAN"; exit 0 ;;
self-test)
    # Positive controls: the probe must flag a KNOWN-busy fixture, including
    # the five-digit fixed-width form that once read as idle, and must fail
    # closed on unparseable input.
    idle=$(printf '| 0 0 | x | 0 3158 / 65536 |\n| 1 1 | x | 0 2905 / 65536 |\n' | hbm_used_list | sort -rn | head -1)
    busy=$(printf '| 0 0 | x | 0 36004/ 65536 |\n| 1 1 | x | 0 3158 / 65536 |\n' | hbm_used_list | sort -rn | head -1)
    none=$(printf 'no table here\n' | hbm_used_list | head -1)
    [ "${idle}" = "3158" ] || { echo "FAIL idle-parse got=${idle}"; exit 1; }
    [ "${busy}" = "36004" ] || { echo "FAIL busy-parse (fixed-width trap) got=${busy}"; exit 1; }
    [ -z "${none}" ] || { echo "FAIL none-parse got=${none}"; exit 1; }
    echo "SELF-TEST PASS (idle=3158, busy=36004 caught, unparseable empty)"; exit 0 ;;
*)
    echo "usage: node_guard.sh [pre|post|self-test]"; exit 1 ;;
esac
