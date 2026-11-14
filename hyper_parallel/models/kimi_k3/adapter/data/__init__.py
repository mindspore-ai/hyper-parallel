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
"""Kimi K3 image/text pretraining data adapters."""

from hyper_parallel.models.kimi_k3.adapter.data.collate import KimiK3Collator, build_kimi_k3_collate_fn
from hyper_parallel.models.kimi_k3.adapter.data.runtime import (
    KimiK3PackedRuntime,
    KimiK3RuntimeInputAdapter,
    build_kimi_k3_runtime_input_adapter,
)
from hyper_parallel.models.kimi_k3.adapter.data.transform_fn import (
    KimiK3OmniTransform,
    build_kimi_k3_omni_transform,
    expand_media_tokens,
)

__all__ = [
    "KimiK3Collator",
    "KimiK3PackedRuntime",
    "KimiK3OmniTransform",
    "KimiK3RuntimeInputAdapter",
    "build_kimi_k3_collate_fn",
    "build_kimi_k3_omni_transform",
    "build_kimi_k3_runtime_input_adapter",
    "expand_media_tokens",
]
