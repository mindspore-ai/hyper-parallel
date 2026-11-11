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
"""Keep HP projections and RoPE while delegating attention and cache to vLLM."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from rl.unified_execution.runtimes.vllm.parallel_context import payload_from_vllm_config
from rl.unified_execution.validation import validate_attention_execution_path
from rl.unified_execution.models import get_model_family


_ATTENTION_INTERFACE = "hyper_parallel_vllm"


def paged_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Convert HP/HF BNSD tensors to vLLM's scheduled token layout.

    Request boundaries, absolute cache slots and dummy forwards are owned by
    the current vLLM forward context. No HF cache or mask may be supplied.
    """
    del kwargs, scaling
    if attention_mask is not None or dropout != 0.0:
        raise ValueError("Unified vLLM attention requires no HF mask and zero dropout")
    if any(tensor.ndim != 4 or tensor.shape[0] != 1 for tensor in (query, key, value)):
        raise ValueError("Unified vLLM attention expects one flattened token batch in BNSD layout")
    core = module.vllm_attention
    num_tokens = query.shape[2]
    scheduled = getattr(module, "unified_token_count", num_tokens)
    flattened = [tensor.transpose(1, 2).reshape(num_tokens, tensor.shape[1] * tensor.shape[3])[:scheduled]
                 for tensor in (query, key, value)]
    output = core(*flattened)
    multiple = getattr(module, "unified_sp_multiple", 1)
    if multiple > 1:
        # The same compiled segment must restore both zero and nonzero SP padding.
        output = torch.cat((output, output.new_zeros((multiple - 1, output.shape[-1]))))[:num_tokens]
    output = output.reshape(1, num_tokens, query.shape[1], value.shape[-1])
    return output, None


def install_paged_attention(model: nn.Module, vllm_config: Any, prefix: str = "") -> None:
    """Replace only the core attention interface on each original decoder."""
    # vLLM is optional in training processes and CPU-only tooling.
    from vllm.model_executor.layers.attention import Attention  # pylint: disable=C0415

    payload = payload_from_vllm_config(vllm_config)
    backend_name = payload.get("attention_backend", "native")
    if backend_name not in ("native", "flash_attention_npu_v4"):
        raise ValueError(f"Unknown unified attention backend: {backend_name}")
    backend_options = {}
    family = get_model_family(model.config.model_type)
    attentions = [getattr(layer, family.attention_attribute) for layer in model.layers]
    if backend_name == "flash_attention_npu_v4":
        # Ascend and V4 remain optional for the native mode and training imports.
        from rl.unified_execution.runtimes.vllm.attention_backend import (  # pylint: disable=C0415
            UnifiedAscendAttentionBackend,
        )

        backend_options["attn_backend"] = UnifiedAscendAttentionBackend
        _, attentions = validate_attention_execution_path(model)
    ALL_ATTENTION_FUNCTIONS.register(_ATTENTION_INTERFACE, paged_attention_forward)
    for index, attention in enumerate(attentions):
        config = attention.config
        head_size = getattr(attention, "qk_head_dim", getattr(config, "head_dim", None))
        head_size = head_size or config.hidden_size // config.num_attention_heads
        tp_size = getattr(model, "inference_tp_size", 1)
        if config.num_attention_heads % tp_size or config.num_key_value_heads % tp_size:
            raise ValueError("HP Qwen3 paged attention requires query and KV head counts divisible by TP")
        layer_prefix = f"{prefix}." if prefix else ""
        attention.vllm_attention = Attention(
            num_heads=config.num_attention_heads // tp_size,
            head_size=head_size,
            scale=attention.scaling,
            num_kv_heads=config.num_key_value_heads // tp_size,
            cache_config=vllm_config.cache_config,
            quant_config=None,
            per_layer_sliding_window=getattr(attention, "sliding_window", None),
            prefix=f"{layer_prefix}model.layers.{index}.self_attn.vllm_attention",
            **backend_options,
        )
        if hasattr(attention, "attention_interface"):
            attention.attention_interface = paged_attention_forward
        attention.config._attn_implementation = _ATTENTION_INTERFACE  # pylint: disable=protected-access
    # This is the independent inference config; training retains its own backend.
    model.config._attn_implementation = _ATTENTION_INTERFACE  # pylint: disable=protected-access
