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
"""Independent precision fixtures for experts operators."""

from __future__ import annotations

import torch

from hyper_parallel.components import ops as operators
from hyper_parallel.components.functional.mome import mome
from tests.common.op_precision import PrecisionCase
from tests.common.op_precision_inputs import Inputs


def _convolution(x, weight, bias, info=None, *, dilation=1, mask=None):
    """Evaluate causal convolution token by token with packed document resets."""
    offsets = (0, x.shape[1]) if info is None else info.local_offsets
    values, kernels = x.float(), weight.float()
    offset = None if bias is None else bias.float()
    results = []
    for start, end in zip(offsets, offsets[1:]):
        for token in range(start, end):
            value = torch.zeros_like(x[:, token], dtype=torch.float32)
            for tap in range(weight.shape[-1]):
                source = token - (weight.shape[-1] - 1 - tap) * dilation
                if source >= start:
                    value = value + values[:, source] * kernels[:, 0, tap]
            value = value.to(x.dtype)
            if offset is not None:
                value = value + offset.to(x.dtype)
            results.append(value)
    result = torch.stack(results, dim=1)
    if mask is None:
        return (result.float() * result.float().sigmoid()).to(x.dtype)
    return x + result * mask[..., None].to(x.dtype)


def _mome(x, mask, weight, bias=None):
    return _convolution(x, weight, bias, mask=mask)


def _grouped_mm(x, weight, offsets):
    counts = torch.diff(torch.cat((offsets.new_zeros(1), offsets)))
    experts = torch.arange(weight.shape[0], device=x.device).repeat_interleave(counts)
    return torch.einsum("tk,tkn->tn", x, weight[experts])


def _grouped_swiglu(x, w1, w2, w3, counts, scores=None):
    experts = torch.arange(w1.shape[0], device=x.device).repeat_interleave(counts)
    gate = torch.einsum("td,tfd->tf", x, w1[experts])
    up = torch.einsum("td,tfd->tf", x, w3[experts])
    hidden = (gate.float() * gate.float().sigmoid()).to(gate.dtype) * up
    if scores is not None:
        hidden = hidden * scores[:, None]
    return torch.einsum("tf,tdf->td", hidden, w2[experts])


def _router(x, weight, bias, vision_bias, image_mask, top_k, scoring_func, scale):
    """Compute expert selections and weights using an independent routing formula."""
    logits = (x.reshape(-1, x.shape[-1]).float()[:, None, :] * weight.float()[None]).sum(-1)
    if scoring_func == "sqrtsoftplus":
        scores = torch.logaddexp(logits, torch.zeros_like(logits)).sqrt()
    elif scoring_func == "softmax":
        scores = logits.softmax(-1)
    else:
        scores = logits.sigmoid()
    correction = bias if image_mask is None else torch.where(image_mask.reshape(-1, 1), vision_bias, bias)
    indices = (scores + correction).argsort(dim=-1, descending=True)[..., :top_k].sort(dim=-1).values
    selected = scores.gather(-1, indices)
    if top_k > 1:
        selected = selected / (selected.sum(-1, keepdim=True) + 1e-20)
    return logits, selected * scale, indices


def _canonical_router(*args):
    logits, weights, indices = operators.moe_router(*args)
    order = indices.argsort(dim=-1)
    return logits, weights.gather(-1, order), indices.gather(-1, order)


def _permute(x, indices):
    slots = sorted(range(indices.numel()), key=lambda slot: int(indices.flatten()[slot]))
    order = torch.tensor(slots, device=x.device)
    inverse = torch.tensor([slots.index(slot) for slot in range(len(slots))], device=x.device)
    states = x[order // indices.shape[1]]
    return states, order, inverse


def _restore(x, positions, weights):
    selected = x[positions.reshape_as(weights)]
    return (selected * weights[..., None]).sum(1).to(x.dtype)


def expert_cases(dtype: torch.dtype, hardware: bool = False) -> list[PrecisionCase]:
    """Cover expert grouping, modality routing, permutation and trainable route weights.

    Args:
        dtype: Training dtype for expert activations and parameters.
        hardware: Use shapes compatible with fused accelerator kernels.

    Returns:
        Independent forward and gradient precision cases for expert operator families.
    """
    inputs = Inputs(dtype)
    cases = []
    for counts in ([2, 0, 3], [0, 0, 0]):
        dim = 128 if hardware else 8
        tokens, hidden = sum(counts), dim * 2
        name = "unused_expert" if tokens else "empty"
        counts_tensor = torch.tensor(counts)
        cases.append(PrecisionCase("grouped_matmul", "standard", name, operators.grouped_matmul, _grouped_mm,
                                   (inputs.tensor(tokens, dim), inputs.tensor(3, dim, hidden), counts_tensor.cumsum(0)),
                                   precision_dtype=dtype, expert_parameters=(1,)))
        for weighted in (False, True):
            cases.append(PrecisionCase(
                "grouped_swiglu", "standard", f"{name}_weighted_{weighted}", operators.grouped_swiglu, _grouped_swiglu,
                (inputs.tensor(tokens, dim), inputs.tensor(3, hidden, dim), inputs.tensor(3, dim, hidden),
                 inputs.tensor(3, hidden, dim), counts_tensor, inputs.tensor(tokens) if weighted else None),
                precision_dtype=dtype, expert_parameters=(1, 2, 3),
            ))
    dim, experts, topk = (256, 128, 8) if hardware else (8, 6, 2)
    for scoring in (("sqrtsoftplus",) if hardware else ("sqrtsoftplus", "softmax", "sigmoid")):
        cases.append(PrecisionCase(
            "moe_router", "deepseek_v41", scoring, _canonical_router, _router,
            (inputs.tensor(1, 8, dim, scale=0.05), inputs.tensor(experts, dim, scale=0.05),
             torch.linspace(-8, 8, experts), torch.linspace(8, -8, experts),
             torch.tensor([[False, True, False, True, True, False, False, True]]), topk, scoring, 1.7),
            precision_dtype=dtype,
        ))
    indices = torch.tensor([[2, 0], [1, 2], [0, 1]], dtype=torch.int64)
    cases.append(PrecisionCase("moe_token_permute", "standard", "repeated_experts", operators.moe_token_permute,
                               _permute, (inputs.tensor(3, 128), indices), precision_dtype=dtype))
    positions = torch.tensor([4, 0, 2, 5, 1, 3], dtype=torch.int64)
    cases.append(PrecisionCase("moe_token_unpermute", "standard", "fp32_route_weights",
                               operators.moe_token_unpermute,
                               _restore, (inputs.tensor(6, 128), positions, inputs.tensor(3, 2, dtype=torch.float32)),
                               precision_dtype=dtype))
    cases.append(PrecisionCase(
        "moe_token_unpermute", "standard", "caller_rounded_route_weights", operators.moe_token_unpermute,
        _restore, (inputs.tensor(6, 128), positions, inputs.tensor(3, 2, dtype=torch.float32).to(dtype)),
        precision_dtype=dtype,
    ))
    cancelling_states = torch.tensor([256, 64, -256, -64, 128, -128], dtype=dtype)
    cancelling_states = cancelling_states[:, None].expand(-1, 128).clone().requires_grad_()
    precise_weights = torch.tensor([[0.5001, 0.4999]]).expand(3, -1).clone().requires_grad_()
    cases.append(PrecisionCase(
        "moe_token_unpermute", "standard", "fp32_weight_cancellation", operators.moe_token_unpermute,
        _restore, (cancelling_states, torch.tensor([0, 2, 1, 3, 4, 5]), precise_weights), precision_dtype=dtype,
    ))
    return cases


def mome_cases(dtype: torch.dtype, hardware: bool = False) -> list[PrecisionCase]:
    """Check functional MOME composition through the aggregate_hidden operator family."""
    inputs = Inputs(dtype)
    hidden = 128 if hardware else 8
    mask = torch.tensor([[True, False, True, True, False], [False, True, True, False, True]])
    return [PrecisionCase(
        "aggregate_hidden", "standard", f"mome_bias_{bias is not None}", mome, _mome,
        (inputs.tensor(2, 5, hidden), mask, inputs.tensor(hidden, 1, 3), bias), precision_dtype=dtype,
    ) for bias in (None, inputs.tensor(hidden))]


def _aggregate_hidden(input_tensor, weight, mask):
    """Evaluate every output from explicit historical tokens and filter taps."""
    if not input_tensor.numel():
        return input_tensor + weight[:0].sum().to(input_tensor.dtype)
    dtype = torch.float32 if input_tensor.dtype in (torch.float16, torch.bfloat16) else input_tensor.dtype
    values, taps = input_tensor.to(dtype), weight.to(dtype)
    outputs = []
    for position in range(input_tensor.shape[0]):
        total = torch.zeros_like(values[position])
        for tap in range(weight.shape[0]):
            source = position - weight.shape[0] + 1 + tap
            if source >= 0:
                total = total + values[source] * taps[tap]
        outputs.append(total.to(input_tensor.dtype) * mask[:, position, None])
    return torch.stack(outputs)


def aggregate_hidden_cases(dtype: torch.dtype, hardware: bool = False) -> list[PrecisionCase]:
    """Build independent causal aggregation cases, including masked and empty states."""
    inputs = Inputs(dtype)
    hidden = 128 if hardware else 4
    shapes = [(8, 2, 3, "mixed_mask"), (8, 2, 3, "all_masked")]
    if not hardware:
        shapes += [(1, 2, 3, "short_sequence"), (4, 1, 1, "pointwise"),
                   (0, 2, 3, "empty_sequence"), (4, 0, 3, "empty_batch")]
    cases = []
    for sequence, batch, window, name in shapes:
        mask = (torch.arange(batch * sequence).reshape(batch, sequence) % 3 != 1)
        if name == "all_masked":
            mask = torch.zeros_like(mask)
        cases.append(PrecisionCase(
            "aggregate_hidden", "standard", name, operators.aggregate_hidden, _aggregate_hidden,
            (inputs.tensor(sequence, batch, hidden), inputs.tensor(window, hidden), mask), precision_dtype=dtype,
        ))
    return cases
