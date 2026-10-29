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
"""Native FA, normalization, RoPE and Ulysses primitives for Head Chunk GQA."""

from typing import Any

import torch

from hyper_parallel.distributed.context_parallel.collectives import ulysses_head_to_seq, ulysses_seq_to_head

_NPU_CAUSAL_MASKS: dict[str, torch.Tensor] = {}
_FA_SAVED_TENSOR_COUNT = 5


def _compressed_causal_mask(reference: torch.Tensor) -> torch.Tensor:
    """Reuse the compressed causal mask required by NPU FA v3."""
    key = str(reference.device)
    mask = _NPU_CAUSAL_MASKS.get(key)
    if mask is None:
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool, device=reference.device), diagonal=1)
        _NPU_CAUSAL_MASKS[key] = mask
    return mask


def _fusion_attention_forward_saved(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Run causal FA and retain only its output and backward statistics."""
    # Optional NPU kernels must not be loaded by the planner's CPU-only import.
    import torch_npu  # pylint: disable=import-outside-toplevel

    output, softmax_max, softmax_sum, _, seed, offset = torch_npu.npu_fusion_attention_v3(
        query, key, value, query.shape[1], "BNSD",
        atten_mask=_compressed_causal_mask(query), scale=scale, sparse_mode=3,
        pre_tockens=2147483647, next_tockens=0,
    )
    return output, (output, softmax_max, softmax_sum, seed, offset)


def _fusion_attention_backward_saved(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    grad_output: torch.Tensor, scale: float, saved: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run native FA backward from one query stage's forward statistics."""
    # Optional NPU kernels are resolved only when execution needs them.
    import torch_npu  # pylint: disable=import-outside-toplevel

    if len(saved) != _FA_SAVED_TENSOR_COUNT:
        raise ValueError(f"Head Chunk FA expects {_FA_SAVED_TENSOR_COUNT} saved tensors, got {len(saved)}")
    attention, softmax_max, softmax_sum, seed, offset = saved
    gradients = torch_npu.npu_fusion_attention_grad_v3(
        query, key, value, grad_output, query.shape[1], "BNSD",
        atten_mask=_compressed_causal_mask(query), softmax_max=softmax_max,
        softmax_sum=softmax_sum, attention_in=attention, scale_value=scale,
        sparse_mode=3, pre_tockens=2147483647, next_tockens=0, seed=seed, offset=offset,
    )
    return gradients[0], gradients[1], gradients[2]


def _rms_norm(tensor: torch.Tensor, weight: torch.Tensor, eps: float, kind: str) -> torch.Tensor:
    """Preserve the validated source normalization's arithmetic and dtype."""
    if kind == "qwen3":
        input_dtype = tensor.dtype
        tensor = tensor.to(torch.float32)
        variance = tensor.pow(2).mean(-1, keepdim=True)
        tensor = tensor * torch.rsqrt(variance + eps)
        return weight * tensor.to(input_dtype)
    if kind != "fused":
        raise ValueError(f"Unsupported Head Chunk RMSNorm contract: {kind}")
    # The HP RMSNorm contract uses this optional NPU operator, without casts.
    import torch_npu  # pylint: disable=import-outside-toplevel

    return torch_npu.npu_rms_norm(tensor, weight, epsilon=eps)[0]


def _apply_rope(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply full-head half RoPE with the existing NPU kernel convention."""
    # Keep the backend optional for importing and inspecting the schedule.
    import torch_npu  # pylint: disable=import-outside-toplevel

    return torch_npu.npu_rotary_mul(tensor.clone(), cos, sin, rotary_mode="half")


def _seq_to_head(tensor: torch.Tensor, cp_mesh: Any | None) -> torch.Tensor:
    """Gather contiguous sequence shards and split destination-major heads."""
    if cp_mesh is None or cp_mesh.size() <= 1:
        return tensor
    return ulysses_seq_to_head(tensor, 2, 1, cp_mesh)


def _head_to_seq(tensor: torch.Tensor, cp_mesh: Any | None) -> torch.Tensor:
    """Apply the inverse Ulysses layout transform."""
    if cp_mesh is None or cp_mesh.size() <= 1:
        return tensor
    return ulysses_head_to_seq(tensor, 2, 1, cp_mesh)
