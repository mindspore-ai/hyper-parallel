#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

set -euo pipefail

IMAGE="${IMAGE:-hyper-parallel:npu}"
NAME="${NAME:-hyper-parallel-npu}"
CARDS="${CARDS:-auto}"
DEFAULT_CARDS="${DEFAULT_CARDS:-0,1,2,3,4,5,6,7}"
CMD=()

usage() {
    cat <<EOF
Usage:
  $(basename "$0") [--image IMAGE] [--name NAME] [--cards LIST] [-- CMD...]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --cards) CARDS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; CMD=("$@"); break ;;
        *) CMD+=("$1"); shift ;;
    esac
done

if [[ ${#CMD[@]} -eq 0 ]]; then
    CMD=(/bin/bash -l)
fi

detect_cards() {
    local device id
    for device in /dev/davinci[0-9]*; do
        [[ -e "${device}" ]] || continue
        id="${device##*/davinci}"
        [[ "${id}" =~ ^[0-9]+$ ]] || continue
        printf '%s\n' "${id}"
    done | sort -n | paste -sd, -
}

if [[ "${CARDS}" == "auto" ]]; then
    CARDS="$(detect_cards)"
    CARDS="${CARDS:-${DEFAULT_CARDS}}"
fi

if docker inspect "${NAME}" >/dev/null 2>&1; then
    docker rm -f "${NAME}" >/dev/null
fi

DEVICES=()
if [[ "${CARDS}" != "none" ]]; then
    IFS=',' read -ra CARD_IDS <<< "${CARDS}"
    for id in "${CARD_IDS[@]}"; do
        id="${id// /}"
        [[ -z "${id}" ]] && continue
        DEVICES+=(--device="/dev/davinci${id}")
    done
fi

echo "============================================="
echo " Starting HyperParallel container"
echo "============================================="
echo " Image     : ${IMAGE}"
echo " Container : ${NAME}"
echo " Cards     : ${CARDS}"
echo " Command   : ${CMD[*]}"
echo "============================================="

docker run \
    --name "${NAME}" \
    --ipc=host \
    --net=host \
    --privileged \
    --ulimit memlock=-1 \
    --ulimit stack=-1 \
    "${DEVICES[@]}" \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -v /root/.ssh:/root/.ssh \
    -it "${IMAGE}" \
    "${CMD[@]}"
