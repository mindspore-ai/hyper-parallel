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
"""SwiGLU activation composition with explicit clamping and precision policies."""

from __future__ import annotations

from math import isfinite

import torch

from hyper_parallel.components.ops import swiglu as selected_swiglu, swiglu_clamped


def swiglu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute SiLU(gate) * up from packed projections using configured operators.

    Args:
        x: Packed gate/up activations with an even size along dim.
        dim: Dimension containing the two projection halves.

    Returns:
        Activated states with the split dimension halved.
    """
    return selected_swiglu(x, dim)


def swiglu_activation(gate: torch.Tensor, up: torch.Tensor, *, limit: float = 0.0,
                       activation_dtype: str = "input") -> torch.Tensor:
    """Apply declared separate-projection activation math without changing its return dtype.

    FP32 activation remains FP32 even when clamp is disabled. The projection
    owner decides when to cast back before its down projection.

    Args:
        gate: Gate projection activations.
        up: Matching up-projection activations.
        limit: Positive clamp bound; nonpositive values disable clamping.
        activation_dtype: Compute in the input dtype or float32.

    Returns:
        Activated states in the declared computation dtype.
    """
    if activation_dtype not in ("input", "float32"):
        raise ValueError("activation_dtype must be input or float32")
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not isfinite(limit):
        raise ValueError("SwiGLU limit must be finite")
    if activation_dtype == "float32":
        return swiglu_clamped(gate, up, limit)
    if limit > 0:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return swiglu(torch.cat((gate, up), dim=-1))


def _resolve_activation_dtype(activation_dtype: str | None, limit: float) -> str:
    """Resolve the shared input/FP32 activation policy without changing projection dtype."""
    if activation_dtype is None:
        activation_dtype = "float32" if limit > 0 else "input"
    if activation_dtype not in ("input", "float32"):
        raise ValueError("activation_dtype must be input or float32")
    return activation_dtype


def local_swiglu(x: torch.Tensor, *, limit: float = 0.0, activation_dtype: str | None = None) -> torch.Tensor:
    """Apply packed SwiGLU, retaining the historical FP32 clamp default.

    Args:
        x: Packed gate/up activations ending in twice the intermediate width.
        limit: Positive clamp bound; nonpositive values disable clamping.
        activation_dtype: Explicit compute dtype policy or the clamp-dependent default.

    Returns:
        Activated states in the packed input dtype.
    """
    activation_dtype = _resolve_activation_dtype(activation_dtype, limit)
    if limit <= 0 and activation_dtype == "input":
        return swiglu(x)
    gate, up = x.chunk(2, dim=-1)
    return swiglu_activation(gate, up, limit=limit, activation_dtype=activation_dtype).to(x.dtype)
