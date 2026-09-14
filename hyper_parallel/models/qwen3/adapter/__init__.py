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
"""Public adapter surface for the Qwen3 dense model family."""

from hyper_parallel.models.qwen3.adapter.registration import QWEN3_ADAPTER_SPEC
from hyper_parallel.models.qwen3.adapter.replacements import (
    replace_qwen3_flash_attention,
    replace_qwen3_rms_norm,
    replace_qwen3_swiglu_mlp,
)

__all__ = [
    "QWEN3_ADAPTER_SPEC",
    "replace_qwen3_flash_attention",
    "replace_qwen3_rms_norm",
    "replace_qwen3_swiglu_mlp",
]
