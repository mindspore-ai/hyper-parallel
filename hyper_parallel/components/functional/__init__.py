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
"""Public high-performance function interfaces.

Exports are resolved lazily (PEP 562): most functions wrap NPU-only kernels
whose modules import ``torch_npu`` at top level, so importing this package
must not force every backend onto CPU-only consumers that only need a
single submodule (e.g. ``functional.npu_grouped_swiglu``).
"""

# PEP 562 requires the special ``__getattr__`` name, and the public symbols
# listed in ``__all__`` are materialized dynamically by that hook.
# pylint: disable=invalid-name,undefined-all-variable

import importlib
from typing import Any

_EXPORT_TO_MODULE = {
    "aggregate_hidden": "aggregate_hidden",
    "apply_rotary_pos_emb": "rotary_embedding",
    "apply_rotary_pos_emb_interleave": "rotary_embedding",
    "apply_rotary_pos_emb_single": "rotary_embedding",
    "apply_rotary_pos_emb_single_freqs": "rotary_embedding",
    "attention_rescale": "attention_rescale",
    "aux_loss_auto_scale": "aux_loss",
    "aux_loss_scale_context": "aux_loss",
    "bind_aux_loss_scale": "aux_loss",
    "dsa_indexer": "dsa_indexer",
    "dsa_kl_loss": "dsa_kl_loss",
    "dsa_sparse_attention": "dsa_sparse_attention",
    "dsa_sparse_attention_rescale": "dsa_sparse_attention_rescale",
    "deepseek_apply_rotary": "deepseek_ascend",
    "deepseek_bf16_gemm": "deepseek_ascend",
    "deepseek_cast_back": "deepseek_ascend",
    "deepseek_einsum": "deepseek_ascend",
    "deepseek_engram_gate": "deepseek_ascend",
    "deepseek_engram_hash": "deepseek_ascend",
    "deepseek_ep_buffer": "deepseek_ascend",
    "deepseek_ep_combine": "deepseek_ascend",
    "deepseek_ep_destroy": "deepseek_ascend",
    "deepseek_ep_dispatch": "deepseek_ascend",
    "deepseek_ep_wait": "deepseek_ascend",
    "deepseek_flash_mla_metadata": "deepseek_ascend",
    "deepseek_flash_mla_sparse_decode": "deepseek_ascend",
    "deepseek_flash_mla_sparse_prefill": "deepseek_ascend",
    "deepseek_fp8_fp4_gemm": "deepseek_ascend",
    "deepseek_fp8_gemm": "deepseek_ascend",
    "deepseek_gemm": "deepseek_ascend",
    "deepseek_grouped_bf16_gemm": "deepseek_ascend",
    "deepseek_grouped_fp8_fp4_gemm": "deepseek_ascend",
    "deepseek_grouped_fp8_gemm": "deepseek_ascend",
    "deepseek_hc_prenorm_gemm": "deepseek_ascend",
    "deepseek_mega_moe": "deepseek_ascend",
    "deepseek_mhc_post": "deepseek_ascend",
    "deepseek_mhc_pre_apply": "deepseek_ascend",
    "deepseek_mhc_sinkhorn": "deepseek_ascend",
    "deepseek_mhc_split_mixes": "deepseek_ascend",
    "deepseek_moe_topk_gate": "deepseek_ascend",
    "deepseek_moe_topk_gate_backward": "deepseek_ascend",
    "deepseek_mqa_logits": "deepseek_ascend",
    "deepseek_normalize_routing_weights": "deepseek_ascend",
    "deepseek_paged_mqa_logits_metadata": "deepseek_ascend",
    "deepseek_per_block_cast": "deepseek_ascend",
    "deepseek_per_token_cast": "deepseek_ascend",
    "deepseek_select_stride_requirement": "deepseek_ascend",
    "deepseek_select_topk": "deepseek_ascend",
    "deepseek_swiglu_backward": "deepseek_ascend",
    "deepseek_swiglu_forward": "deepseek_ascend",
    "deepseek_tile_kernel": "deepseek_ascend",
    "deepseek_topk_gate": "deepseek_ascend",
    "deepseek_transform_mega_moe_weights": "deepseek_ascend",
    "deepseek_transform_scaling_factors": "deepseek_ascend",
    "grouped_matmul": "grouped_matmul",
    "chunk_gated_delta_rule": "gated_delta_net",
    "fused_chunk_kda": "kimi_delta_attention",
    "fused_chunk_kda_p2p": "kimi_delta_attention",
    "mhc_post": "mhc_post",
    "mhc_pre": "mhc_pre",
    "moe_token_permute": "moe_token_permute",
    "moe_token_unpermute": "moe_token_unpermute",
    "npu_fusion_attention_forward": "npu_fusion_attention",
    "npu_grouped_swiglu": "npu_grouped_swiglu",
    "rms_norm": "rms_norm",
    "set_aux_loss_scale": "aux_loss",
    "sink_attention": "sink_attention",
    "sinkhorn": "sinkhorn",
    "swiglu": "swiglu",
    "is_deepseek_backend_available": "deepseek_ascend",
}


def __getattr__(name: str) -> Any:
    """Resolve a public function by importing its owning submodule lazily."""
    submodule = _EXPORT_TO_MODULE.get(name)
    if submodule is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    module = importlib.import_module(
        f"hyper_parallel.components.functional.{submodule}"
    )
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "aggregate_hidden",
    "apply_rotary_pos_emb",
    "apply_rotary_pos_emb_interleave",
    "apply_rotary_pos_emb_single",
    "apply_rotary_pos_emb_single_freqs",
    "attention_rescale",
    "aux_loss_auto_scale",
    "aux_loss_scale_context",
    "bind_aux_loss_scale",
    "dsa_indexer",
    "dsa_kl_loss",
    "dsa_sparse_attention",
    "dsa_sparse_attention_rescale",
    "deepseek_apply_rotary",
    "deepseek_bf16_gemm",
    "deepseek_cast_back",
    "deepseek_einsum",
    "deepseek_engram_gate",
    "deepseek_engram_hash",
    "deepseek_ep_buffer",
    "deepseek_ep_combine",
    "deepseek_ep_destroy",
    "deepseek_ep_dispatch",
    "deepseek_ep_wait",
    "deepseek_flash_mla_metadata",
    "deepseek_flash_mla_sparse_decode",
    "deepseek_flash_mla_sparse_prefill",
    "deepseek_fp8_fp4_gemm",
    "deepseek_fp8_gemm",
    "deepseek_gemm",
    "deepseek_grouped_bf16_gemm",
    "deepseek_grouped_fp8_fp4_gemm",
    "deepseek_grouped_fp8_gemm",
    "deepseek_hc_prenorm_gemm",
    "deepseek_mega_moe",
    "deepseek_mhc_post",
    "deepseek_mhc_pre_apply",
    "deepseek_mhc_sinkhorn",
    "deepseek_mhc_split_mixes",
    "deepseek_moe_topk_gate",
    "deepseek_moe_topk_gate_backward",
    "deepseek_mqa_logits",
    "deepseek_normalize_routing_weights",
    "deepseek_paged_mqa_logits_metadata",
    "deepseek_per_block_cast",
    "deepseek_per_token_cast",
    "deepseek_select_stride_requirement",
    "deepseek_select_topk",
    "deepseek_swiglu_backward",
    "deepseek_swiglu_forward",
    "deepseek_tile_kernel",
    "deepseek_topk_gate",
    "deepseek_transform_mega_moe_weights",
    "deepseek_transform_scaling_factors",
    "grouped_matmul",
    "chunk_gated_delta_rule",
    "fused_chunk_kda",
    "fused_chunk_kda_p2p",
    "mhc_post",
    "mhc_pre",
    "moe_token_permute",
    "moe_token_unpermute",
    "npu_fusion_attention_forward",
    "npu_grouped_swiglu",
    "rms_norm",
    "set_aux_loss_scale",
    "sink_attention",
    "sinkhorn",
    "swiglu",
    "is_deepseek_backend_available",
]
