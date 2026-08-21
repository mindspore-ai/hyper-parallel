#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_WORKSPACE="$(cd "${ROOT_DIR}/../.." && pwd)"
IMAGE="${IMAGE:-hyper-parallel:npu}"
NAME="${NAME:-hyper-parallel-npu}"
CARDS="${CARDS:-0,1,2,3,4,5,6,7}"
WORKDIR="${WORKDIR:-/workspace/hyper-parallel}"
CMD=()
TTY_FLAGS=()

usage() {
    cat <<EOF
Usage:
  $(basename "$0") [--image IMAGE] [--name NAME] [--cards LIST] [--workdir PATH] [-- CMD...]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --cards) CARDS="$2"; shift 2 ;;
        --workdir) WORKDIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; CMD=("$@"); break ;;
        *) CMD+=("$1"); shift ;;
    esac
done

if [[ ${#CMD[@]} -eq 0 ]]; then
    CMD=(/bin/bash -l)
fi

if [[ -t 0 && -t 1 ]]; then
    TTY_FLAGS=(-it)
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
echo " Workdir   : ${WORKDIR}"
echo " Command   : ${CMD[*]}"
echo "============================================="

docker run \
    --name "${NAME}" \
    --rm \
    "${TTY_FLAGS[@]}" \
    --ipc=host \
    --net=host \
    --privileged \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "${DEVICES[@]}" \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v "${HOST_WORKSPACE}:${WORKDIR}" \
    -w "${WORKDIR}" \
    "${IMAGE}" \
    "${CMD[@]}"
