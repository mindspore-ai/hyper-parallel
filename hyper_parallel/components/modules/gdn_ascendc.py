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

"""Structure-preserving GDN kernel replacement."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

# auto_models/modules contains PyTorch-specific module replacements.
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.components.functional.gated_delta_rule import chunk_gated_delta_rule


@module_replacement
def replace_gdn_chunk_rule(
    *,
    module: nn.Module,
    module_fqn: str = "",
    context: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Replace only the GDN chunk primitive, preserving the model's forward and weights.

    Args:
        module: GDN module exposing an instance-level chunk_gated_delta_rule callable.
        module_fqn: Source module's fully qualified name.
        context: Model replacement context.

    Returns:
        A shallow module copy sharing the original parameters and submodules.

    Raises:
        TypeError: The source does not expose the supported GDN primitive.

    Note:
        Select this factory through ``plan_overrides[].replace_module._target_``
        as ``hyper_parallel.components.modules.replace_gdn_chunk_rule``.
        Install a matching fla_npu Python runtime and AscendC OPP, plus
        triton-ascend for the auxiliary kernels. No global monkey patch is used.
        Only stateless, fixed-length chunk training is supported.
        AscendC intermediates use reduced precision and are not bitwise
        equivalent to the Transformers FP32 fallback.
    """
    del context
    if not callable(getattr(module, "chunk_gated_delta_rule", None)):
        raise TypeError(f"{module_fqn}: GDN replacement requires callable chunk_gated_delta_rule")
    replacement = copy.copy(module)
    # CP wraps this callable after replacement; no global monkey patch or forward rewrite.
    replacement.chunk_gated_delta_rule = chunk_gated_delta_rule
    return replacement
