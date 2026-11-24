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

The root specification declares public module families and architecture policies.
Native modeling composes the public terminal head and router loss. Package import
performs no model construction, network access or distributed initialization.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hyper_parallel.models.adapter_spec import ModelAdapterSpec


__all__ = [
    "get_adapter_spec",
]


def get_adapter_spec() -> "ModelAdapterSpec":
    """Return this family's ModelAdapterSpec via the shared registry."""
    # Lazy import: keeps the package import free of registry/provider work.
    from hyper_parallel.models.registry import (  # pylint: disable=C0415
        get_model_adapter,
    )
    return get_model_adapter("qwen3_moe")
