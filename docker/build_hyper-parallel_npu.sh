#!/usr/bin/env bash
# Build and smoke-test the HyperParallel NPU Docker image.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="${DOCKERFILE:-${ROOT_DIR}/docker/Dockerfile.hyper-parallel-npu}"
IMAGE="${IMAGE:-hyper-parallel:npu}"
HP_EXTRA="${HP_EXTRA:-torch29}"

usage() {
    cat <<EOF
Usage:
  $(basename "$0") [IMAGE]

Environment:
  DOCKERFILE=${DOCKERFILE}
  IMAGE=${IMAGE}
  HP_EXTRA=${HP_EXTRA}

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
    .

echo "Verify image: ${IMAGE}"
docker run --rm --entrypoint /bin/bash "${IMAGE}" -lc '
set -e
source /usr/local/Ascend/cann/set_env.sh
python3 -c "import importlib.metadata as md; import hyper_parallel as hp; print(hp.get_platform()); print(md.version(\"hyper_parallel\"))"
'

echo "Done: ${IMAGE}"
