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

"""Sequence-first adapter for the VeOmni-style AscendC GDN training path."""

from __future__ import annotations

from importlib import import_module
from typing import Any

# auto_models/ops contains PyTorch-specific high-performance kernels.
import torch  # pylint: disable=forbidden-backend-import


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run GDN forward/backward in sequence-first layout, as in VeOmni's NPU backend.

    Args:
        q: Queries with layout [batch, sequence, heads, key_dim].
        k: Keys in the same layout as queries.
        v: Values with layout [batch, sequence, heads, value_dim].
        g: Log decay with layout [batch, sequence, heads].
        beta: Update gates with layout [batch, sequence, heads].
        scale: Query scaling; defaults to inverse square root of key_dim.
        initial_state: Must be None; recurrent-state gradients are not supported.
        output_final_state: Must be False for the stateless training path.
        use_qk_l2norm_in_kernel: Normalize queries and keys before the recurrence.
        cu_seqlens: Must be None; variable-length kernel dispatch is not enabled.
        chunk_size: Kernel chunk length, currently restricted to 64.
        **kwargs: No additional kernel options are supported.

    Returns:
        Sequence-first output and None (no recurrent-state carry).

    Raises:
        ValueError: Inputs are not on NPU or unsupported options are supplied.
        RuntimeError: An optional AscendC or Triton-Ascend dependency is missing.
    """
    if kwargs:
        raise ValueError(f"Unsupported GDN kernel options: {sorted(kwargs)}")
    if q.device.type != "npu":
        raise ValueError("Fused GDN requires NPU tensors")
    if initial_state is not None or output_final_state:
        raise ValueError("AscendC GDN currently supports stateless chunk training only")
    if cu_seqlens is not None:
        raise ValueError("AscendC GDN variable-length sequences are not supported yet")
    if chunk_size != 64:
        raise ValueError("AscendC GDN currently requires chunk_size=64")
    if q.dtype not in (torch.bfloat16, torch.float16) or q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError("AscendC GDN requires matching BF16/FP16 q, k and v")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("Expected sequence-first q/k/v with matching batch, sequence and heads")
    if g.shape != q.shape[:3] or beta.shape != g.shape:
        raise ValueError("Expected g and beta with shape [batch, sequence, heads]")
    if any(tensor.device != q.device for tensor in (k, v, g, beta)):
        raise ValueError("All GDN inputs must be on the same NPU")
    # Import only on an active NPU: the optional kernel queries device properties.
    try:
        kernel = import_module(
            "hyper_parallel.components.functional._ascendc_gdn"
        ).chunk_gated_delta_rule
        return kernel(
            q, k, v, g, beta, scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    except ModuleNotFoundError as exc:
        missing_module = str(exc.name or "")
        if missing_module.split(".", maxsplit=1)[0] in ("triton", "fla_npu"):
            raise RuntimeError(
                "AscendC GDN requires fla_npu built for the installed NPU/CANN and triton-ascend "
                "for auxiliary kernels; it does not use the forward-only torch_npu GDN API"
            ) from exc
        raise
