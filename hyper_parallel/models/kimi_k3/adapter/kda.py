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
"""K3 local KDA and short-convolution adapters for the shared FLA backend."""

from __future__ import annotations

import importlib

import torch  # pylint: disable=forbidden-backend-import
from torch.nn import functional as F  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.linear_attention_packed import PackedSequenceInfo, packed_reference
from hyper_parallel.components.modules.kimi_delta_attention import chunk_kda, torch_recurrent_kda
from hyper_parallel.core.dtensor.dtensor import DTensor


def kda(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, gate: torch.Tensor,
        beta: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor, *,
        lower_bound: float, backend: str = "eager",
        sequence_info: PackedSequenceInfo | None = None) -> torch.Tensor:
    """Apply zero-state causal KDA with differentiable chunk-alignment padding.

    The Triton path requires local BF16 Q/K/V/gate/beta and complete D=128.
    Only the unused right tail of non-packed inputs is padded; packed inputs
    use the shared varlen kernel without inserting tokens between documents.
    No final state escapes this call.
    Existing sample padding is owned by the model and never joins sequences.

    Args:
        query: Queries with shape ``[B, T, H_k, D_k]``, normalized inside this call.
        key: Keys matching the query shape, also normalized inside this call.
        value: Values with shape ``[B, T, H_v, D_v]``.
        gate: Unactivated decay logits with shape ``[B, T, H_v, D_k]``.
        beta: Unactivated update logits with shape ``[B, T, H_v]``.
        a_log: Logarithmic gate scales containing one value per value head.
        dt_bias: Gate biases containing ``H_v * D_k`` values.
        lower_bound: Negative lower bound of the activated log-decay gate.
        backend: ``eager`` recurrent reference or ``triton`` shared FLA kernel.
        sequence_info: Optional packed metadata whose document boundaries reset
            recurrent state; no cross-document state is carried.

    Returns:
        Attention values with the value input's shape and dtype, excluding any
        alignment padding inserted by this adapter.
    """
    if backend not in ("eager", "triton"):
        raise ValueError("kda_backend must be eager or triton")
    arguments = {"a_log": a_log, "dt_bias": dt_bias, "use_qk_l2norm_in_kernel": True,
                 "use_gate_in_kernel": True, "use_beta_sigmoid_in_kernel": True, "lower_bound": lower_bound}
    if backend == "eager":
        if sequence_info is not None:
            return packed_reference(torch_recurrent_kda, (query, key, value, gate, beta),
                                    sequence_info.local_offsets, **arguments)[0]
        return torch_recurrent_kda(query, key, value, gate, beta, **arguments)[0]
    tensors = (query, key, value, gate, beta, a_log, dt_bias)
    if any(isinstance(tensor, DTensor) for tensor in tensors):
        raise TypeError("K3 Triton KDA requires local tensors with complete heads")
    if query.device.type != "npu" or any(tensor.dtype != torch.bfloat16 for tensor in tensors[:5]):
        raise ValueError("K3 Triton KDA requires NPU BF16 query/key/value/gate/beta")
    length = query.shape[1]
    if length == 0:
        raise ValueError("K3 Triton KDA requires a nonempty sequence")
    padding = (-length) % 64 if sequence_info is None else 0
    if padding:
        query, key, value, gate = (F.pad(tensor, (0, 0, 0, 0, 0, padding))
                                   for tensor in (query, key, value, gate))
        beta = F.pad(beta, (0, 0, 0, padding))
    # Preserve the reference's BF16 reciprocal rounding before multiplication.
    # FLA's internal normalization rounds only after its FP32 multiplication.
    query, key = (tensor * torch.rsqrt(tensor.float().square().sum(-1, keepdim=True) + 1e-6).to(tensor.dtype)
                  for tensor in (query, key))
    arguments["use_qk_l2norm_in_kernel"] = False
    if sequence_info is not None:
        arguments["cu_seqlens"] = sequence_info.materialize(query.device, 64).cu_seqlens
    return chunk_kda(query, key, value, gate, beta, backend="triton", **arguments)[0][:, :length]


def short_convolution(projected: torch.Tensor, weight: torch.Tensor, *, backend: str = "eager") -> torch.Tensor:
    """Compute independent causal depthwise convolution plus SiLU.

    Args:
        projected: Local sequence input [B,T,C].
        weight: Existing Conv1d parameter [C,1,W]; parameter ownership is preserved.
        backend: Eager differentiable operations or optional FLA Triton-Ascend, without fallback.

    Returns:
        [B,T,C] with zero initial convolution state and no returned cache.
    """
    if backend not in ("eager", "triton"):
        raise ValueError("causal_conv_backend must be eager or triton")
    if projected.ndim != 3 or weight.ndim != 3 or weight.shape[:2] != (projected.shape[-1], 1):
        raise ValueError("short convolution expects input [B,T,C] and weight [C,1,W]")
    if backend == "eager":
        output = F.conv1d(  # pylint: disable=not-callable
            projected.transpose(1, 2), weight, padding=weight.shape[-1]-1,
                          groups=projected.shape[-1])
        return F.silu(output[:, :, :projected.shape[1]].transpose(1, 2))
    if any(isinstance(tensor, DTensor) for tensor in (projected, weight)):
        raise TypeError("K3 Triton short convolution requires local tensors")
    if projected.device.type != "npu" or weight.device != projected.device or weight.dtype != projected.dtype:
        raise ValueError("K3 Triton short convolution requires same-device/dtype NPU input and weight")
    if projected.shape[1] == 0:
        return projected + weight.reshape(-1)[:0].sum()
    # FLA is optional and must not be imported by the portable model path.
    try:
        module = importlib.import_module("fla.modules.convolution")
    except ImportError as error:
        raise RuntimeError("K3 Triton convolution requires the pinned flash-linear-attention package") from error
    return module.causal_conv1d(projected, weight.squeeze(1), activation="silu", backend="triton",
                               output_final_state=False)[0]
