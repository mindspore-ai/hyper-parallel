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
"""Independent precision fixtures for basic operators."""

from __future__ import annotations

import torch

from hyper_parallel.components import ops as operators
from tests.common.op_precision import PrecisionCase
from tests.common.op_precision_inputs import Inputs


def _rms(x, weight, eps=1e-5):
    values = x.double() if x.dtype == torch.float64 else x.float()
    denominator = values.norm(dim=-1, keepdim=True).square() / x.shape[-1] + eps
    return (values / denominator.sqrt() * weight.to(values.dtype)).to(x.dtype)


def _swiglu(x, dim=-1):
    gate, up = torch.tensor_split(x, 2, dim=dim)
    values = gate.double() if gate.dtype == torch.float64 else gate.float()
    return (values * values.sigmoid()).to(gate.dtype) * up


def _clamped(x, up, limit):
    gate, up = x.float(), up.float()
    if limit > 0:
        gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
    return gate * gate.sigmoid() * up


def _rope(x, cos, sin):
    width = cos.shape[-1]
    half = width // 2
    first, second = x[..., :half], x[..., half:width]
    rotated = torch.cat((first * cos[..., :half] - second * sin[..., :half],
                         second * cos[..., half:] + first * sin[..., half:]), dim=-1)
    return torch.cat((rotated, x[..., width:]), dim=-1)


def _interleaved(x, key, cos, sin, *, unsqueeze_dim=1):
    """Independently rotate adjacent query and key pairs with a preserved tail."""
    width = cos.shape[-1]
    if cos.ndim == 2:
        cos, sin = cos[None], sin[None]
    cosine, sine = cos[..., :width // 2].unsqueeze(unsqueeze_dim), sin[..., :width // 2].unsqueeze(unsqueeze_dim)
    results = []
    for tensor in (x, key):
        real, imaginary = tensor[..., :width:2], tensor[..., 1:width:2]
        results.append(torch.cat((real * cosine - imaginary * sine, imaginary * cosine + real * sine,
                                  tensor[..., width:]), dim=-1))
    return tuple(results)


def _mhc(x, residual, h_post, h_res, num_stream):
    streams = residual.unflatten(-1, (num_stream, x.shape[-1]))
    outputs = [
        h_post[..., target, None] * x
        + sum(h_res[..., source, target, None] * streams[..., source, :] for source in range(num_stream))
        for target in range(num_stream)
    ]
    return torch.cat(outputs, dim=-1).to(x.dtype)


def _sinkhorn(x, sinkhorn_iters=20, eps=1e-6):
    weights = (x - x.amax(-1, keepdim=True)).exp()
    value = weights / weights.sum(-1, keepdim=True) + eps
    for iteration in range(sinkhorn_iters):
        if iteration:
            value = value / (value.sum(-1, keepdim=True) + eps)
        value = value / (value.sum(-2, keepdim=True) + eps)
    return value


def _interleaved_single(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Reuse the independent rotation oracle for its first branch."""
    return _interleaved(x, x, cos, sin, unsqueeze_dim=unsqueeze_dim)[0]


def pointwise_cases(dtype: torch.dtype, hardware: bool = False) -> list[PrecisionCase]:
    """Cover normalization, activations, both RoPE layouts and mHC mixing."""
    del hardware
    inputs = Inputs(dtype)
    cases = []
    for variant, operation in (("standard", operators.rms_norm), ("fp32_affine", operators.rms_norm_fp32_affine)):
        for name, magnitude in (("regular", 1.0), ("near_zero", 1e-5), ("zero", 0.0)):
            cases.append(PrecisionCase("rms_norm", variant, name, operation, _rms,
                                       (inputs.tensor(2, 5, 128, scale=magnitude),
                                        inputs.tensor(128, dtype=torch.float32 if variant == "fp32_affine" else dtype)),
                                       {"eps": 1e-6}, dtype))
    cases.append(PrecisionCase("swiglu", "standard", "last_dim", operators.swiglu, _swiglu,
                               (inputs.tensor(2, 5, 128, scale=2.0),), precision_dtype=dtype))
    cases.append(PrecisionCase("swiglu", "standard", "nonlast_dim", operators.swiglu, _swiglu,
                               (inputs.tensor(2, 128, 5),), {"dim": 1}, dtype))
    cases.append(PrecisionCase("swiglu", "standard", "saturated_gate", operators.swiglu, _swiglu,
                               (inputs.tensor(2, 5, 128, scale=12.0),), precision_dtype=dtype))
    for limit in (0.0, 1.25):
        cases.append(PrecisionCase("swiglu", "clamped_fp32", f"limit_{limit}", operators.swiglu_clamped, _clamped,
                                   (inputs.tensor(2, 5, 64, scale=3.0), inputs.tensor(2, 5, 64, scale=3.0), limit),
                                   precision_dtype=dtype))
    angles = inputs.tensor(1, 5, 16, grad=False, scale=1.0).repeat(1, 1, 2)
    cos, sin = angles.cos(), angles.sin()
    query, key = inputs.tensor(2, 4, 5, 64), inputs.tensor(2, 2, 5, 64)
    cases.append(PrecisionCase("rope", "half", "partial_tail", operators.rope, _rope,
                               (query, cos[:, None], sin[:, None]), precision_dtype=dtype))
    cases.append(PrecisionCase("rope", "interleaved", "partial_tail", operators.rope_interleaved, _interleaved,
                               (query, key, cos, sin), precision_dtype=dtype))
    for name, frequencies in (("full_rotation", 64), ("identity", 0)):
        angles = inputs.tensor(1, 5, frequencies // 2, grad=False, scale=1.0).repeat(1, 1, 2)
        cases.append(PrecisionCase("rope", "half", name, operators.rope, _rope,
                                   (query, angles.cos()[:, None], angles.sin()[:, None]), precision_dtype=dtype))
        cases.append(PrecisionCase("rope", "interleaved", name, operators.rope_interleaved, _interleaved,
                                   (query, key, angles.cos(), angles.sin()), precision_dtype=dtype))
    cases.append(PrecisionCase("rope", "interleaved", "sequence_first_partial", operators.rope_interleaved,
                               _interleaved, (query.transpose(1, 2), key.transpose(1, 2), cos, sin),
                               {"unsqueeze_dim": 2}, dtype))
    for name, width, axis in (("partial_tail", 32, 1), ("full_rotation", 64, 1),
                              ("identity", 0, 1), ("sequence_first_partial", 32, 2)):
        angles = inputs.tensor(1, 5, width // 2, grad=False).repeat(1, 1, 2)
        states = query if axis == 1 else query.transpose(1, 2)
        cases.append(PrecisionCase("rope", "interleaved_single", name, operators.rope_interleaved_single,
                                   _interleaved_single, (states, angles.cos(), angles.sin()),
                                   {"unsqueeze_dim": axis}, dtype))
    cases.append(PrecisionCase("mhc_post", "standard", "fp32_coefficients", operators.mhc_post, _mhc,
                               (inputs.tensor(2, 5, 128), inputs.tensor(2, 5, 512),
                                inputs.tensor(2, 5, 4, dtype=torch.float32),
                                inputs.tensor(2, 5, 4, 4, dtype=torch.float32), 4), precision_dtype=dtype))
    cases.append(PrecisionCase("sinkhorn", "standard", "twenty_iterations", operators.sinkhorn, _sinkhorn,
                               (inputs.tensor(2, 5, 4, 4, dtype=torch.float32),), precision_dtype=torch.float32))
    return cases
