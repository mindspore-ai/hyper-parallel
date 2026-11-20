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
"""OCP MXFP4 QDQ operator pair for Indexer activations."""

from typing import Any

import torch
from torch import Tensor

from hyper_parallel.components.quantization.config import MX_BLOCK_SIZE_32, MX_SCALE_ALG_OCP
from hyper_parallel.components.quantization.ops.npu_mxfp8 import LowPrecisionCapabilityError

_ANTI_QUANT_DST_TYPES = {torch.float16: 5, torch.float32: 6, torch.bfloat16: 15}


def _native_qdq(x_2d: Tensor) -> Tensor:
    """Use the installed torch_npu ops without coupling to W4A8 GMM."""
    if x_2d.device.type != "npu":
        raise LowPrecisionCapabilityError(f"OCP MXFP4 QDQ requires an NPU tensor, got {x_2d.device}.")
    try:
        # Resolve the NPU runtime only when the CSA2 operator is invoked.
        import torch_npu  # pylint: disable=C0415
    except (ImportError, OSError, RuntimeError) as exc:
        raise LowPrecisionCapabilityError("OCP MXFP4 QDQ requires torch_npu.") from exc
    return _quant_dequant(x_2d, torch_npu)


def _quant_dequant(x_2d: Tensor, torch_npu: Any) -> Tensor:
    """Quantize E2M1/E8M0 block32 with OCP scaling and dequantize."""
    required = ("float4_e2m1fn_x2", "npu_dynamic_mx_quant", "npu_anti_mx_quant")
    missing = [name for name in required if not hasattr(torch_npu, name)]
    if missing:
        raise LowPrecisionCapabilityError(f"The installed torch_npu lacks OCP MXFP4 QDQ operators: {missing}.")
    try:
        dst_type = _ANTI_QUANT_DST_TYPES[x_2d.dtype]
    except KeyError as exc:
        raise ValueError(f"OCP MXFP4 QDQ requires FP16, BF16 or FP32 input, got {x_2d.dtype}.") from exc
    packed, scale = torch_npu.npu_dynamic_mx_quant(
        x_2d, dst_type=torch_npu.float4_e2m1fn_x2,
        block_size=MX_BLOCK_SIZE_32, round_mode="rint", scale_alg=MX_SCALE_ALG_OCP,
    )
    return torch_npu.npu_anti_mx_quant(
        packed, scale, axis=-1, dst_type=dst_type,
        src_type=torch_npu.float4_e2m1fn_x2,
    )


def pack_indexer_k(key: Tensor) -> Tensor:
    """Store each Indexer key row as OCP E2M1 bytes followed by E8M0 scales."""
    if key.device.type != "npu":
        raise LowPrecisionCapabilityError("OCP MXFP4 Indexer cache requires NPU tensors.")
    width = key.shape[-1]
    if width <= 0 or width % MX_BLOCK_SIZE_32:
        raise ValueError("Indexer K width must be positive and divisible by 32.")
    scale_slots = 2 * ((width // MX_BLOCK_SIZE_32 + 1) // 2)
    if key.numel() == 0:
        return torch.empty((*key.shape[:-1], width // 2 + scale_slots), dtype=torch.uint8, device=key.device)
    import torch_npu  # pylint: disable=C0415
    data, scale = torch_npu.npu_dynamic_mx_quant(
        key.reshape(-1, width).contiguous(), dst_type=torch_npu.float4_e2m1fn_x2,
        block_size=MX_BLOCK_SIZE_32, round_mode="rint", scale_alg=MX_SCALE_ALG_OCP,
    )
    return torch.cat((data.reshape(*key.shape[:-1], width // 2),
                      scale.reshape(*key.shape[:-1], -1)), dim=-1)


def unpack_indexer_k(packed: Tensor, logical_width: int, dtype: torch.dtype) -> Tensor:
    """Read packed Indexer K through the same native OCP dequantization operator."""
    if logical_width <= 0 or logical_width % MX_BLOCK_SIZE_32:
        raise ValueError("Indexer K width must be positive and divisible by 32.")
    scale_pairs = (logical_width // MX_BLOCK_SIZE_32 + 1) // 2
    if packed.dtype != torch.uint8 or packed.shape[-1] != logical_width // 2 + 2 * scale_pairs:
        raise ValueError("Indexer K packed row does not match its logical width.")
    if packed.device.type != "npu":
        raise LowPrecisionCapabilityError("OCP MXFP4 Indexer cache requires NPU tensors.")
    if packed.numel() == 0:
        return torch.empty((*packed.shape[:-1], logical_width), dtype=dtype, device=packed.device)
    import torch_npu  # pylint: disable=C0415
    data = packed[..., :logical_width // 2].contiguous()
    scale = packed[..., logical_width // 2:].reshape(*packed.shape[:-1], scale_pairs, 2).contiguous()
    values = torch_npu.npu_anti_mx_quant(
        data.reshape(-1, logical_width // 2), scale.reshape(-1, scale_pairs, 2),
        axis=-1, dst_type=_ANTI_QUANT_DST_TYPES[dtype],
        src_type=torch_npu.float4_e2m1fn_x2,
    )
    return values.reshape(*packed.shape[:-1], logical_width)


__all__ = ["LowPrecisionCapabilityError"]
