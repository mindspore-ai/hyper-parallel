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
"""DeepSeek-V4.1 structure-preserving module replacements."""

from collections.abc import Mapping
from typing import Any

from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedDSAAttention,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41AttentionPlaceholder,
)
from hyper_parallel.models.replacement import module_replacement


@module_replacement
def replace_deepseek_v41_shared_attention(
        *,
        module: nn.Module,
        module_fqn: str,
        context: Mapping[str, Any],
) -> SharedCompressedDSAAttention:
    """Replace a V4.1 parameter holder with its shared-attention forward.

    Args:
        module: Source V4.1 attention parameter holder.
        module_fqn: Fully qualified module name supplied by the executor.
        context: Read-only replacement context.

    Returns:
        A structure-preserving attention module with V4.1 forward semantics.

    Raises:
        TypeError: If the selected source is not a V4.1 placeholder.
    """
    del context
    if not isinstance(module, DeepseekV41AttentionPlaceholder):
        raise TypeError(
            f"{module_fqn}: expected DeepseekV41AttentionPlaceholder, "
            f"got {type(module).__name__}"
        )
    return SharedCompressedDSAAttention(module)


__all__ = ["replace_deepseek_v41_shared_attention"]
