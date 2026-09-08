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
"""replacements: Qwen3-MoE structure replacements onto the generic modules.

The factories keep their historical names and factory parameters (adjust
doc §5.3) but now declare/construct the generic high-performance
``modules.RMSNorm`` / ``modules.GQAAttention`` entries instead of rebinding
model-local forward functions; the NPU kernels live in ``functional`` — no
second family implementation exists here.

The factories stay importable on CPU-only checkouts: the ``modules``
package wraps NPU-only ``functional`` backends, so it is imported lazily
inside each factory body.
"""

from collections.abc import Mapping
from typing import Any

from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.models.qwen3_moe.adapter.attention import (
    run_qwen3_moe_flash_attention,
)


@module_replacement
def replace_qwen3_moe_rms_norm(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> nn.Module:
    """Replace ``Qwen3MoeRMSNorm`` with the generic high-performance RMSNorm.

    ``modules.RMSNorm`` reuses the source ``weight`` parameter and
    ``variance_epsilon`` (identity-preserving) and applies the fused NPU
    kernel through ``functional.rms_norm``.
    """
    # Lazy: the modules package wraps NPU-only functional backends.
    from hyper_parallel.components.modules import RMSNorm  # pylint: disable=C0415

    return RMSNorm(module=module, module_fqn=module_fqn, context=context)


@module_replacement
def replace_qwen3_moe_flash_attention(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> nn.Module:
    """Replace ``Qwen3MoeAttention`` with the generic grouped-query attention.

    ``modules.GQAAttention`` fuses the Q/K/V projections (checkpoint
    conversion declared through ``make_transforms``), keeps the Qwen Q/K-norm
    submodules and RoPE/cache handling, and calls the attention kernel
    through the Qwen mask/cache contract in
    ``adapter.attention.run_qwen3_moe_flash_attention``.
    """
    # Lazy: the modules package wraps NPU-only functional backends.
    from hyper_parallel.components.modules import GQAAttention  # pylint: disable=C0415

    return GQAAttention(
        module=module,
        module_fqn=module_fqn,
        context=context,
        attention_interface=run_qwen3_moe_flash_attention,
    )


def _validate_batched_experts_contract(module: nn.Module, module_fqn: str) -> None:
    """Validate the supported Transformers expert-layout matrix (plan §6.1.1).

    Only the batched layout (``gate_up_proj``/``down_proj`` parameters on the
    ``*.mlp.experts`` target) is supported; the legacy Transformers 4.57
    layout, where ``experts`` is a ``ModuleList`` of per-expert MLPs, is
    explicitly rejected with a clear error instead of being silently
    skipped. This check is pure Python so the rejection also fires on
    CPU-only checkouts, before any NPU-only import is attempted.
    """
    if isinstance(module, nn.ModuleList) or isinstance(
        getattr(module, "experts", None), nn.ModuleList
    ):
        raise TypeError(
            f"{module_fqn}: the legacy Transformers expert layout "
            "(``experts`` as a ModuleList of per-expert MLPs, e.g. "
            "Transformers 4.57) is not supported; use a batched-experts "
            "release whose experts expose gate_up_proj/down_proj parameters"
        )
    missing = [
        name
        for name in ("gate_up_proj", "down_proj")
        if not isinstance(getattr(module, name, None), nn.Parameter)
    ]
    if missing:
        raise TypeError(
            f"{module_fqn}: the GroupedExperts replacement requires batched "
            f"{missing} parameter(s) on the source module; got "
            f"{type(module).__name__}"
        )


@module_replacement
def replace_qwen3_moe_grouped_experts(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> nn.Module:
    """Replace batched Qwen3-MoE routed experts with ``modules.GroupedExperts``.

    Thin declaration for the ``*.mlp.experts`` match boundary (plan Q4 /
    adjust doc §5.3): the legacy whole-block ``replace_qwen3_moe_sparse_moe``
    was deleted in M5; this factory only validates the batched-experts
    contract and delegates construction to the generic module. The contract
    validation runs before the lazy ``modules`` import so the legacy-layout
    rejection also works on CPU-only checkouts.
    """
    _validate_batched_experts_contract(module, module_fqn)
    # Lazy: the modules package wraps NPU-only functional backends.
    from hyper_parallel.components.modules import GroupedExperts  # pylint: disable=C0415

    return GroupedExperts(module=module, module_fqn=module_fqn, context=context)


__all__ = [
    "replace_qwen3_moe_flash_attention",
    "replace_qwen3_moe_grouped_experts",
    "replace_qwen3_moe_rms_norm",
]
