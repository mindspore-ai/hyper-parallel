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

# Build and smoke-test the HyperParallel NPU Docker image.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="${DOCKERFILE:-${ROOT_DIR}/docker/Dockerfile.hyper-parallel-npu}"
IMAGE="${IMAGE:-hyper-parallel:npu}"
HP_EXTRA="${HP_EXTRA:-torch29}"
TORCH_EXTRA="${TORCH_EXTRA:-${HP_EXTRA}}"
CANN_QUAY_URL="${CANN_QUAY_URL:-quay.io/ascend/cann}"
CANN_VERSION="${CANN_VERSION:-9.1.0}"
CANN_ARCH="${CANN_ARCH:-a3}"
BASE_OS="${BASE_OS:-ubuntu22.04}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}"
BUILD_MULTICORE_EXTENSION="${BUILD_MULTICORE_EXTENSION:-off}"
BUILD_SHMEM_EXTENSION="${BUILD_SHMEM_EXTENSION:-off}"
BUILD_CUSTOM_OPS_EXTENSION="${BUILD_CUSTOM_OPS_EXTENSION:-off}"
HYPER_PARALLEL_BUILD_STRICT="${HYPER_PARALLEL_BUILD_STRICT:-off}"

usage() {
    cat <<EOF
Usage:
  $(basename "$0") [IMAGE]

Environment:
  DOCKERFILE=${DOCKERFILE}
  IMAGE=${IMAGE}
  HP_EXTRA=${HP_EXTRA}
  TORCH_EXTRA=${TORCH_EXTRA}
  CANN_QUAY_URL=${CANN_QUAY_URL}
  CANN_VERSION=${CANN_VERSION}
  CANN_ARCH=${CANN_ARCH}
  BASE_OS=${BASE_OS}
  PYTHON_VERSION=${PYTHON_VERSION}
  PIP_INDEX_URL=${PIP_INDEX_URL}

Example:
  $(basename "$0") hyper-parallel:npu
  HP_EXTRA=mindspore DOCKERFILE=docker/Dockerfile.mindspore $(basename "$0") hyper-parallel:mindspore
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 0 ]]; then
    IMAGE="$1"
fi

cd "${ROOT_DIR}"

echo "Build image: ${IMAGE}"
echo "Dockerfile : ${DOCKERFILE}"
echo "HP_EXTRA   : ${HP_EXTRA}"

docker build \
    -f "${DOCKERFILE}" \
    -t "${IMAGE}" \
    --build-arg "HP_EXTRA=${HP_EXTRA}" \
    --build-arg "TORCH_EXTRA=${TORCH_EXTRA}" \
    --build-arg "CANN_QUAY_URL=${CANN_QUAY_URL}" \
    --build-arg "CANN_VERSION=${CANN_VERSION}" \
    --build-arg "CANN_ARCH=${CANN_ARCH}" \
    --build-arg "BASE_OS=${BASE_OS}" \
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
    --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}" \
    --build-arg "BUILD_MULTICORE_EXTENSION=${BUILD_MULTICORE_EXTENSION}" \
    --build-arg "BUILD_SHMEM_EXTENSION=${BUILD_SHMEM_EXTENSION}" \
    --build-arg "BUILD_CUSTOM_OPS_EXTENSION=${BUILD_CUSTOM_OPS_EXTENSION}" \
    --build-arg "HYPER_PARALLEL_BUILD_STRICT=${HYPER_PARALLEL_BUILD_STRICT}" \
    .

echo "Verify image: ${IMAGE}"
docker run --rm --entrypoint /bin/bash "${IMAGE}" -lc '
set -e
. /usr/local/Ascend/cann/set_env.sh
python3 -c "import importlib.metadata as md; import hyper_parallel as hp; print(hp.get_platform()); print(md.version(\"hyper_parallel\"))"
'

echo "Done: ${IMAGE}"
