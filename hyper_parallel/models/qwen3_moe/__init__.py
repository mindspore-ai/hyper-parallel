# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""qwen3_moe: Qwen3-MoE model family package (HF model example).

The stable entry exposes the adapter-spec query only; replacements,
attention contract and model-specific TP/CP/EP rules live in ``adapter/``
and are filled in by changesets M2/M3 (adjust doc §7.1). Importing this
package must not read YAML, hit the network, build models or initialize
distributed.
"""

__all__ = [
    "get_adapter_spec",
]


def get_adapter_spec():
    """Return this family's ModelAdapterSpec via the shared registry."""
    # Lazy import: keeps the package import free of registry/provider work.
    from hyper_parallel.models.registry import (  # pylint: disable=C0415
        get_model_adapter,
    )
    return get_model_adapter("qwen3_moe")
