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
"""Qwen3 dense replacements backed by generic high-performance modules."""

from collections.abc import Mapping
from typing import Any

from torch import nn, no_grad  # pylint: disable=forbidden-backend-import

from hyper_parallel.models.qwen3.adapter.attention import (
    run_qwen3_flash_attention,
)
from hyper_parallel.models.replacement import module_replacement


@module_replacement
def replace_qwen3_rms_norm(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> nn.Module:
    """Replace ``Qwen3RMSNorm`` with the generic fused RMSNorm module."""
    # Lazy: importing the implementation requires the optional NPU runtime.
    from hyper_parallel.components.modules import RMSNorm  # pylint: disable=C0415

    return RMSNorm(module=module, module_fqn=module_fqn, context=context)


@module_replacement
def replace_qwen3_flash_attention(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> nn.Module:
    """Replace ``Qwen3Attention`` with fused grouped-query attention."""
    # Lazy: importing the implementation requires the optional NPU runtime.
    from hyper_parallel.components.modules import GQAAttention  # pylint: disable=C0415

    replacement = GQAAttention(
        module=module,
        module_fqn=module_fqn,
        context=context,
        attention_interface=run_qwen3_flash_attention,
    )
    # The one-rank HF path loads live weights before replacement; the meta
    # path is populated later by the shared checkpoint loader.
    if not module.q_proj.weight.is_meta:
        source = {name: parameter.detach() for name, parameter in module.named_parameters()}
        for transform in replacement.make_transforms():
            converted = transform.operations[0].convert(
                source, transform.source_patterns, transform.target_patterns,
            )
            with no_grad():
                for name, value in converted.items():
                    replacement.get_parameter(name).copy_(value)
    return replacement


@module_replacement
def replace_qwen3_swiglu_mlp(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> nn.Module:
    """Replace ``Qwen3MLP`` with the generic fused SwiGLU MLP module."""
    # Lazy: importing the implementation requires the optional NPU runtime.
    from hyper_parallel.components.modules import SwiGLUMLP  # pylint: disable=C0415

    return SwiGLUMLP(module=module, module_fqn=module_fqn, context=context)


__all__ = [
    "replace_qwen3_flash_attention",
    "replace_qwen3_rms_norm",
    "replace_qwen3_swiglu_mlp",
]
