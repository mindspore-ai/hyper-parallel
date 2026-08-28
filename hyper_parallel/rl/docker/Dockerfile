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

FROM hyper-parallel/unified-e2-dev:v0.22.1rc1

ARG FLASH_ATTN_NPU_WHEEL=flash_attn_npu-0.2.0b1-cp312-cp312-linux_aarch64.whl
ARG FLASH_ATTN_NPU_SHA256=9f58e114b77f72079111e2f86fa9750d3be39d1ec9324b309588a540a3e9e12b

COPY ${FLASH_ATTN_NPU_WHEEL} /tmp/${FLASH_ATTN_NPU_WHEEL}

RUN set -eux; \
    echo "${FLASH_ATTN_NPU_SHA256}  /tmp/${FLASH_ATTN_NPU_WHEEL}" | sha256sum -c -; \
    PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --no-cache-dir --no-deps "/tmp/${FLASH_ATTN_NPU_WHEEL}"; \
    rm -f "/tmp/${FLASH_ATTN_NPU_WHEEL}"

RUN python -c "from importlib.metadata import version; \
expected = {'torch': '2.10.0+cpu', 'torch-npu': '2.10.0', 'transformers': '5.5.4', \
'vllm': '0.22.1+empty', 'vllm-ascend': '0.22.1rc1', 'batch_invariant_ops': '1.0.0', \
'flash-attn-npu': '0.2.0b1'}; actual = {name: version(name) for name in expected}; \
assert actual == expected, f'dependency version mismatch: expected={expected}, actual={actual}'; print(actual)"

LABEL org.opencontainers.image.title="Hyper-RL Qwen3 runtime" \
      org.opencontainers.image.description="Pinned Ascend dependencies for Hyper-RL Qwen3 TP consistency" \
      org.opencontainers.image.base.name="hyper-parallel/unified-e2-dev:v0.22.1rc1" \
      hyper-rl.flash-attn-npu.version="0.2.0b1" \
      hyper-rl.batch-invariant-ops.version="1.0.0"
