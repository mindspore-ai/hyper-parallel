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
"""Register the Qwen3-MoE adapter capabilities.

Providers remain lazy so registry discovery does not import model
implementations, trainer/data modules, or backend dependencies.
"""

import torch
from torch import nn

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.materialization import register_rebuildable_buffer
from hyper_parallel.models.registry import register_model_adapter


def _register_rotary_materialization(model: nn.Module) -> None:
    """Retain native RoPE buffers across to_empty without reinitializing weights.

    Args:
        model: Qwen3-MoE model before storage materialization.
    """
    for module in model.modules():
        if type(module).__name__ != "Qwen3MoeRotaryEmbedding":
            continue
        with torch.device("cpu"):
            reference = type(module)(module.config, device="cpu")
        for name, value in reference.named_buffers(recurse=False):
            register_rebuildable_buffer(
                module, name, value=value.to(dtype=module.get_buffer(name).dtype), persistent=False,
            )


def _load_replacements():
    """Return the family's replacement-factory module (lazy provider)."""
    from hyper_parallel.models.qwen3_moe.adapter.conversion import (  # pylint: disable=C0415
        module_replacement,
    )
    return module_replacement


def _load_attention():
    """Return the family's attention-contract module (lazy provider)."""
    from hyper_parallel.models.qwen3_moe.adapter.runtime import (  # pylint: disable=C0415
        attention,
    )
    return attention


def _load_expert_parallel():
    """Return the family's EP-factory module (lazy provider)."""
    from hyper_parallel.models.qwen3_moe.adapter.distributed import (  # pylint: disable=C0415
        expert_parallel,
    )
    return expert_parallel


def _load_loss():
    """Return the family's model-integrated output-loss adapter."""
    from hyper_parallel.models.qwen3_moe.adapter.runtime import (  # pylint: disable=C0415
        chunked_loss,
    )
    return chunked_loss


def _load_codegen():
    """Return the shared Qwen3 fused-attention source renderer lazily."""
    from hyper_parallel.models.qwen3.adapter import codegen  # pylint: disable=C0415

    return codegen


QWEN3_MOE_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen3MoeForCausalLM",
    model_type="qwen3_moe",
    replacements=_load_replacements,
    attention=_load_attention,
    expert_parallel=_load_expert_parallel,
    loss=_load_loss,
    codegen=_load_codegen,
    materialization=_register_rotary_materialization,
)

register_model_adapter(QWEN3_MOE_ADAPTER_SPEC)
