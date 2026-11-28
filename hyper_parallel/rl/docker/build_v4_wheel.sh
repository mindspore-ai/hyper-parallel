#!/usr/bin/env bash
# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help ]]; then
    cat <<'HELP'
Usage: bash build_v4_wheel.sh [OUTPUT_DIR]
Build the patched Ascend 910 V4 wheel in the existing consistency image.
Default output: ../wheels/v4-build, relative to the repository root.
Optional SOURCE_DIR: local flash-attention-npu Git checkout with csrc/catlass
Git objects available; export pinned commits read-only, without network access.
Without SOURCE_DIR, fetch pinned upstream commits using the container network.
The output directory must be empty. No NPU or image modification is required.
HELP
    exit 0
fi
[[ $# -le 1 ]] || { printf 'Expected at most one output directory\n' >&2; exit 1; }
repo_root=$(cd -- "${script_dir}/../../.." && pwd)
output_dir=${1:-$(dirname -- "${repo_root}")/wheels/v4-build}
image=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:consistency-v0.30.0-cann9.2.1-arm64
image_id=$(docker image inspect "${image}" --format '{{.Id}}')
[[ "${image_id}" == sha256:f139fa4776cf71fa1ebf00c47fa1f4c313deec48267c2102a1a42d968634fb08 ]] || {
    printf 'Image differs from the pinned build environment: %s\n' "${image_id}" >&2; exit 1;
}
mkdir -p "${output_dir}"
output_dir=$(realpath "${output_dir}")
[[ -z $(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit) ]] || {
    printf 'Use an empty output directory: %s\n' "${output_dir}" >&2; exit 1;
}
source_args=(--network=host)
if [[ -n ${SOURCE_DIR:-} ]]; then
    source_args=(--network=none -v "$(realpath "${SOURCE_DIR}"):/source:ro")
fi
docker run --rm -i "${source_args[@]}" \
    -e "BUILD_IMAGE_ID=${image_id}" -e "RESULT_UID=$(id -u)" -e "RESULT_GID=$(id -g)" \
    -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY -e http_proxy -e https_proxy -e no_proxy \
    -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e PYTHONDONTWRITEBYTECODE=1 \
    -v "${script_dir}/patches/flash-attn-npu-v4-canonical-rowsum.patch:/v4.patch:ro" \
    -v "${output_dir}:/output" --entrypoint /bin/bash "${image_id}" -s <<'BUILD'
set -eo pipefail
source /usr/local/Ascend/cann/set_env.sh
set -u
trap 'chown -R "${RESULT_UID}:${RESULT_GID}" /output' EXIT
exec > >(tee /output/build.log) 2>&1
export SOURCE_COMMIT=2e5320494cb474365bacaf42c94abf0c7de0affc
export CATLASS_COMMIT=769cd40a8716b28650b6bebb08db4834eea4462f
mkdir /tmp/v4-source
if [[ -d /source ]]; then
    git -c safe.directory='*' -C /source archive "${SOURCE_COMMIT}" | tar -x -C /tmp/v4-source
    git -c safe.directory='*' -C /source/csrc/catlass archive "${CATLASS_COMMIT}" \
        | tar -x -C /tmp/v4-source/csrc/catlass
else
    git init /tmp/v4-fetch
    git -C /tmp/v4-fetch fetch --depth=1 https://github.com/MinghuasLab/flash-attention-npu.git "${SOURCE_COMMIT}"
    git -C /tmp/v4-fetch archive "${SOURCE_COMMIT}" | tar -x -C /tmp/v4-source
    git init /tmp/catlass-fetch
    git -C /tmp/catlass-fetch fetch --depth=1 https://gitcode.com/cann/catlass.git "${CATLASS_COMMIT}"
    git -C /tmp/catlass-fetch archive "${CATLASS_COMMIT}" | tar -x -C /tmp/v4-source/csrc/catlass
fi
# V4/910 uses CATLASS headers; its googletest and 950 DSL submodules are not built.
cd /tmp/v4-source
git apply --check /v4.patch
git apply /v4.patch
export FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_BUILD_VERSION=v4
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_LOCAL_VERSION=hpbi1
export FLASH_ATTN_ENABLE_MSSANITIZER=FALSE
python -m pip wheel --no-deps --no-build-isolation --no-cache-dir --wheel-dir /tmp/v4-dist .
python -m pip install --no-deps --no-compile --target /tmp/v4-install /tmp/v4-dist/*.whl
python -m pip freeze > /output/environment.txt
python - <<'PY'
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import zipfile

import torch
import torch_npu

wheel, = Path('/tmp/v4-dist').glob('*.whl')
with zipfile.ZipFile(wheel) as archive:
    library, = [name for name in archive.namelist()
                if name.startswith('flash_attn_npu_4/') and name.endswith('.so')]
    library_sha256 = hashlib.sha256(archive.read(library)).hexdigest()
# Load the native library without the package's device-detection import.
torch.ops.load_library(str(Path('/tmp/v4-install') / library))
digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
record = {
    'source_commit': os.environ['SOURCE_COMMIT'],
    'catlass_commit': os.environ['CATLASS_COMMIT'],
    'image_id': os.environ['BUILD_IMAGE_ID'],
    'patch_sha256': hashlib.sha256(Path('/v4.patch').read_bytes()).hexdigest(),
    'wheel': wheel.name, 'wheel_sha256': digest,
    'native_library': library, 'native_library_sha256': library_sha256,
    'build_target': 'v4/910', 'local_version': 'hpbi1',
    'torch': torch.__version__, 'torch_npu': importlib.metadata.version('torch-npu'),
    'native_library_load': 'passed',
}
shutil.copy2(wheel, Path('/output') / wheel.name)
Path('/output/build.json').write_text(json.dumps(record, indent=2) + '\n')
Path('/output/SHA256SUMS').write_text(f'{digest}  {wheel.name}\n')
print(json.dumps(record, indent=2))
PY
BUILD
printf 'Wheel and build record: %s\n' "${output_dir}"
