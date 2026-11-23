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
"""Deterministic operator fixtures shared by CPU UT and accelerator parity ST.

Oracles use scalar recurrences, token-wise formulas and explicit document
loops. They never call the native implementation being tested.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from hyper_parallel.components import ops as operators
from hyper_parallel.components.modules.attention.linear_attention_utils.sequence_layout import PackedSequenceInfo
from tests.common.op_precision import PrecisionCase


class Inputs:
    """Construct already-quantized reproducible CPU leaves without global RNG state."""

    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.generator = torch.Generator().manual_seed(20261120)

    def tensor(
        self, *shape: int, dtype: torch.dtype | None = None, grad: bool = True, scale: float = 0.2,
    ) -> torch.Tensor:
        """Return a leaf with moderate values to avoid accidental unstable TopK ties."""
        values = torch.randn(shape, generator=self.generator) * scale
        return values.to(self.dtype if dtype is None else dtype).requires_grad_(grad)


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
            if offset is not None:
                value = value + offset
            results.append(value.to(x.dtype))
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
    return (selected * weights.to(x.dtype)[..., None]).sum(1)


def _attention_values(query, key, value, mask=None, scale=None):
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
    value = value.repeat_interleave(query.shape[1] // value.shape[1], dim=1)
    scores = torch.einsum("bhqd,bhkd->bhqk", query.to(dtype), key.to(dtype))
    scores = scores * (query.shape[-1] ** -0.5 if scale is None else scale)
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf) if mask.dtype == torch.bool else scores + mask
    probabilities = scores.softmax(-1).nan_to_num()
    return torch.einsum("bhqk,bhkv->bhqv", probabilities, value.to(dtype)).to(query.dtype).transpose(1, 2)


def _dense_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """Reconstruct packed, causal and sliding-window attention masks independently."""
    if dropout:
        raise ValueError("Strict attention precision fixtures require dropout=0")
    lengths = kwargs.get("actual_seq_len")
    if lengths is not None:
        outputs, start = [], 0
        for end in lengths:
            outputs.append(_dense_attention(
                module, query[:, :, start:end], key[:, :, start:end], value[:, :, start:end], None,
                scaling=scaling,
            )[0])
            start = end
        return torch.cat(outputs, dim=1), None
    q_pos = torch.arange(query.shape[2], device=query.device) + key.shape[2] - query.shape[2]
    k_pos = torch.arange(key.shape[2], device=query.device)
    mask = k_pos[None] <= q_pos[:, None] if getattr(module, "is_causal", True) else None
    window = kwargs.get("sliding_window")
    if window is not None:
        mask = mask & (k_pos[None] > q_pos[:, None] - window)
    if attention_mask is not None:
        if attention_mask.ndim == 2:
            mask = mask & attention_mask[:, None, None, :]
        elif attention_mask.dtype == torch.bool:
            mask = attention_mask & mask
        else:
            mask = attention_mask.masked_fill(~mask, -torch.inf)
    return _attention_values(query, key, value, mask, scaling), None


def _mla_stats(query, key, value, query_ends, key_ends, *, scale, need_stats):
    """Reconstruct packed MLA attention and detached softmax statistics."""
    outputs, maxima, totals = [], [], []
    qs = ks = 0
    for qe, ke in zip(query_ends.tolist(), key_ends.tolist()):
        q, k, v = (tensor[:, start:end].transpose(1, 2)
                   for tensor, start, end in ((query, qs, qe), (key, ks, ke), (value, ks, ke)))
        allowed = torch.arange(ke - ks, device=q.device)[None] <= (
            torch.arange(qe - qs, device=q.device)[:, None] + ke - ks - (qe - qs)
        )
        outputs.append(_attention_values(q, k, v, allowed, scale))
        scores = (q.float() @ k.float().transpose(-1, -2) * scale).masked_fill(~allowed, -torch.inf)
        maximum = scores.amax(-1, keepdim=True)
        maxima.append(maximum.detach())
        totals.append((scores - maximum).exp().sum(-1, keepdim=True).detach())
        qs, ks = qe, ke
    output = torch.cat(outputs, dim=1)
    return (output, torch.cat(maxima, dim=2)[0], torch.cat(totals, dim=2)[0]) if need_stats else (output, None, None)


def _delta_rule(x, key, value, gate, beta, *, chunk_size=64, normalize_qk=True, info=None,
                a_log=None, dt_bias=None, lower_bound=None, scale=None):
    """Evaluate zero-state GDN or KDA with an independent token recurrence."""
    del chunk_size
    output_dtype = x.dtype
    is_kda = a_log is not None
    if normalize_qk:
        if is_kda:
            x = x * (x.float().square().sum(-1, keepdim=True) + 1e-6).rsqrt().to(x.dtype)
            key = key * (key.float().square().sum(-1, keepdim=True) + 1e-6).rsqrt().to(key.dtype)
        else:
            x = x * (x.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
            key = key * (key.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    x, key, value, beta = (tensor.float() for tensor in (x, key, value, beta))
    if is_kda:
        shifted = gate.float() + dt_bias.reshape(gate.shape[-2:]).float()
        activation = shifted * a_log.float().exp()[None, None, :, None]
        gate = (-a_log.float().exp()[None, None, :, None] * torch.logaddexp(shifted, torch.zeros_like(shifted))
                if lower_bound is None else lower_bound * activation.sigmoid())
        beta = beta.sigmoid()
    else:
        gate = gate.float().unsqueeze(-1)
    offsets = (0, x.shape[1]) if info is None else info.local_offsets
    output = []
    for start, end in zip(offsets, offsets[1:]):
        state = x.new_zeros(x.shape[0], value.shape[2], key.shape[-1], value.shape[-1])
        for token in range(start, end):
            state = state * gate[:, token].exp().unsqueeze(-1)
            predicted = (key[:, token, :, :, None] * state).sum(-2)
            update = (value[:, token] - predicted) * beta[:, token, :, None]
            state = state + key[:, token, :, :, None] * update[..., None, :]
            output.append((x[:, token, :, :, None] * state).sum(-2) * (
                x.shape[-1] ** -0.5 if scale is None else scale
            ))
    return torch.stack(output, dim=1).to(output_dtype)


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
    cases.append(PrecisionCase("mhc_post", "standard", "fp32_coefficients", operators.mhc_post, _mhc,
                               (inputs.tensor(2, 5, 128), inputs.tensor(2, 5, 512),
                                inputs.tensor(2, 5, 4, dtype=torch.float32),
                                inputs.tensor(2, 5, 4, 4, dtype=torch.float32), 4), precision_dtype=dtype))
    cases.append(PrecisionCase("sinkhorn", "standard", "twenty_iterations", operators.sinkhorn, _sinkhorn,
                               (inputs.tensor(2, 5, 4, 4, dtype=torch.float32),), precision_dtype=torch.float32))
    return cases


def expert_cases(dtype: torch.dtype, hardware: bool = False) -> list[PrecisionCase]:
    """Cover expert grouping, modality routing, permutation and trainable route weights."""
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
    cases.append(PrecisionCase("moe_token_reorder", "standard", "repeated_experts", operators.moe_token_reorder,
                               _permute, (inputs.tensor(3, 128), indices), precision_dtype=dtype))
    positions = torch.tensor([4, 0, 2, 5, 1, 3], dtype=torch.int64)
    cases.append(PrecisionCase("moe_token_reorder", "weighted_restore", "fp32_route_weights",
                               operators.moe_token_restore,
                               _restore, (inputs.tensor(6, 128), positions, inputs.tensor(3, 2, dtype=torch.float32)),
                               precision_dtype=dtype))
    mask = torch.tensor([[True, False, True, True, False], [False, True, True, False, True]])
    for bias in (None, inputs.tensor(128)):
        cases.append(PrecisionCase("mome", "standard", f"bias_{bias is not None}", operators.mome, _mome,
                                   (inputs.tensor(2, 5, 128), mask, inputs.tensor(128, 1, 3), bias),
                                   precision_dtype=dtype))
    return cases


def attention_cases(dtype: torch.dtype, hardware: bool = False) -> list[PrecisionCase]:
    """Cover dense, grouped-query, unequal-value and document-packed attention."""
    inputs = Inputs(dtype)
    width = 128 if hardware else 8
    query, key, value = (inputs.tensor(1, 4, 5, width) for _ in range(3))
    cases = [PrecisionCase("attention", "vision", "bidirectional", operators.vision_attention, _attention_values,
                           (query, key, value), precision_dtype=dtype)]
    module = SimpleNamespace(is_causal=True)
    for variant, operation in (("gqa", operators.grouped_query_attention), ("mla", operators.multi_latent_attention)):
        key = inputs.tensor(1, 2 if variant == "gqa" else 4, 5, width)
        value = inputs.tensor(1, key.shape[1], 5, width if variant == "gqa" else width // 2)
        for name, kwargs in (("causal", {}), ("packed_documents", {"actual_seq_len": [2, 5]}),
                             ("sliding_window", {"sliding_window": 2})):
            cases.append(PrecisionCase("attention", variant, name, operation, _dense_attention,
                                       (module, query, key, value, None), kwargs, dtype))
    ends = torch.tensor([2, 5], dtype=torch.int32)
    for need_stats in (False, True):
        cases.append(PrecisionCase("attention", "mla_with_stats", f"stats_{need_stats}",
                                   operators.multi_latent_attention_with_stats, _mla_stats,
                                   (query.transpose(1, 2), inputs.tensor(1, 5, 4, width),
                                    inputs.tensor(1, 5, 4, width // 2), ends, ends),
                                   {"scale": width ** -0.5, "need_stats": need_stats}, dtype))
    return cases


def linear_attention_cases(dtype: torch.dtype, hardware: bool = False) -> list[PrecisionCase]:
    """Cover convolution reset and zero-state GDN/KDA, including chunk tails."""
    inputs = Inputs(dtype)
    cases = []
    length, width = (65, 128) if hardware else (9, 4)
    for packed in (False, True):
        info = PackedSequenceInfo((0, 3, length), 0, length) if packed else None
        for bias in (None, inputs.tensor(32)):
            cases.append(PrecisionCase("causal_conv1d", "standard", f"packed_{packed}_bias_{bias is not None}",
                                       operators.causal_conv1d, _convolution,
                                       (inputs.tensor(1, length, 32), inputs.tensor(32, 1, 3), bias, info),
                                       precision_dtype=dtype))
        for family, operation in (("gdn", operators.gdn), ("kda", operators.kda)):
            kwargs = {"info": info}
            if family == "gdn":
                gate = -inputs.tensor(1, length, 2, dtype=torch.float32).abs() * 0.1
                beta = inputs.tensor(1, length, 2).sigmoid()
            else:
                gate, beta = inputs.tensor(1, length, 2, width), inputs.tensor(1, length, 2)
                kwargs.update(a_log=inputs.tensor(2, dtype=torch.float32),
                              dt_bias=inputs.tensor(2 * width, dtype=torch.float32), lower_bound=-3.0)
            cases.append(PrecisionCase(family, "zero_state", f"packed_{packed}", operation, _delta_rule,
                                       (inputs.tensor(1, length, 2, width), inputs.tensor(1, length, 2, width),
                                        inputs.tensor(1, length, 2, width), gate, beta), kwargs, dtype))
    return cases
