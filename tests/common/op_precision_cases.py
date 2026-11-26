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

from typing import Any

from types import SimpleNamespace
import math

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


def _mhc_split(mixes, scale, base, num_stream, *, post_multiplier=2.0, epsilon=1e-6):
    weights = torch.cat((scale[0].expand(num_stream), scale[1].expand(num_stream),
                         scale[2].expand(num_stream * num_stream)))
    values = mixes * weights + base
    return (values[..., :num_stream].sigmoid() + epsilon,
            values[..., num_stream:2 * num_stream].sigmoid() * post_multiplier,
            values[..., 2 * num_stream:].reshape(*mixes.shape[:-1], num_stream, num_stream))


def _mhc_apply(hidden, pre):
    return sum(hidden[..., i, :].float() * pre[..., i, None]
               for i in range(hidden.shape[-2])).to(hidden.dtype)


def _mhc_pre(x, phi, alpha, beta, num_stream, sinkhorn_iters=20, norm_eps=1e-6, hc_eps=1e-6, gamma=None):
    """Independent complete Pre reference with explicit RMS and projection expressions."""
    normalized = x.float() / (x.float().square().sum(-1, keepdim=True) / x.shape[-1] + norm_eps).sqrt()
    if gamma is not None:
        normalized = normalized * gamma.float()
    mixes = normalized @ phi.float().T
    pre, post, residual = _mhc_split(mixes, alpha, beta, num_stream, epsilon=hc_eps)
    hidden = x.reshape(*x.shape[:-1], num_stream, -1)
    return _mhc_apply(hidden, pre), post, _sinkhorn(residual, sinkhorn_iters, hc_eps), x


def mhc_pre_cases(dtype: torch.dtype, hardware: bool = False) -> list[PrecisionCase]:
    """Cover complete Pre, independent coefficient outputs and stream aggregation."""
    inputs = Inputs(dtype)
    hidden = 128 if hardware else 16
    cases = []
    for use_gamma in (False, True):
        cases.append(PrecisionCase(
            "mhc_pre", "standard", f"gamma_{use_gamma}", operators.mhc_pre, _mhc_pre,
            (inputs.tensor(2, 5, 4 * hidden), inputs.tensor(24, 4 * hidden, dtype=torch.float32),
             inputs.tensor(3, dtype=torch.float32), inputs.tensor(24, dtype=torch.float32), 4),
            {"sinkhorn_iters": 20 if hardware else 3,
             "gamma": inputs.tensor(4 * hidden, dtype=torch.float32) if use_gamma else None},
            dtype,
        ))
    cases.append(PrecisionCase(
        "mhc_pre", "split_mixes", "separate_coefficients", operators.mhc_split_mixes, _mhc_split,
        (inputs.tensor(2, 5, 24, dtype=torch.float32), inputs.tensor(3, dtype=torch.float32),
         inputs.tensor(24, dtype=torch.float32), 4), {"post_multiplier": 3.0, "epsilon": 1e-4}, torch.float32,
    ))
    cases.append(PrecisionCase(
        "mhc_pre", "apply_mix", "supplied_pre_mix", operators.mhc_pre_apply, _mhc_apply,
        (inputs.tensor(2, 5, 4, hidden), inputs.tensor(2, 5, 4, dtype=torch.float32)), precision_dtype=dtype,
    ))
    for width in (64, 2560):
        cases.append(PrecisionCase(
            "mhc_pre", "apply_mix", f"accumulation_width_{width}", operators.mhc_pre_apply, _mhc_apply,
            (inputs.tensor(1, 7, 4, width), inputs.tensor(1, 7, 4, dtype=torch.float32)), precision_dtype=dtype,
        ))
    return cases


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


def _cached_rope_reference(query, cache, key=None, positions=None, *, interleaved=False,
                           conjugate=False, sequence_offset=0):
    width = cache.shape[-1]

    def rotate(x: Any) -> Any:
        """Rotate the active prefix while retaining the surrounding tensor layout."""
        shape = x.shape
        flat = x.reshape(-1, shape[-2], shape[-1])
        rows = []
        for token, values in enumerate(flat):
            position = token % shape[-3] if positions is None else int(positions.flatten()[token])
            cos, sin = cache[position + sequence_offset].chunk(2)
            first = values[..., :width:2] if interleaved else values[..., :width // 2]
            second = values[..., 1:width:2] if interleaved else values[..., width // 2:width]
            sign = -1 if conjugate else 1
            left = first.float() * cos - second.float() * sin * sign
            right = second.float() * cos + first.float() * sin * sign
            rotated = torch.stack((left, right), -1).flatten(-2) if interleaved else torch.cat((left, right), -1)
            rows.append(torch.cat((rotated.to(x.dtype), values[..., width:]), -1))
        return torch.stack(rows).reshape(shape)

    return rotate(query), None if key is None else rotate(key)


def cached_rope_cases(dtype: torch.dtype) -> list[PrecisionCase]:
    """Exercise the shared cache contract without resizing existing RoPE fixtures."""
    inputs = Inputs(dtype)
    cases = []
    for interleaved in (False, True):
        for batched in (False, True):
            for indexed in (False, True):
                width = 128 if batched else 64
                shape = (2, 3, 4, width + 32) if batched else (3, 4, width + 32)
                query = inputs.tensor(*shape)
                key = inputs.tensor(*shape[:-2], 2, shape[-1]) if batched else None
                positions = torch.tensor([[2, 0, 2], [1, 2, 1]]) if batched else torch.tensor([2, 0, 2])
                cases.append(PrecisionCase(
                    "rope", "cached", f"cached_pair_{interleaved}_batch_{batched}_indices_{indexed}",
                    operators.rope_cached, _cached_rope_reference,
                    (query, inputs.tensor(7, width, dtype=torch.float32), key, positions if indexed else None),
                    {"interleaved": interleaved, "conjugate": indexed, "sequence_offset": 1}, dtype,
                ))
    return cases


def _routed_swiglu_reference(x, fmt, **options):
    gate, up = x.float().chunk(2, -1)
    limit = options.get("clamp_value")
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
    result = gate * gate.sigmoid() * up
    if options.get("topk_weights") is not None:
        result = result * options["topk_weights"][:, None] * options.get("routed_scaling_factor", 1.0)
    prefixes = options.get("psum_num_tokens_per_expert")
    if prefixes is not None:
        mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        previous = 0
        for end in prefixes.tolist():
            start = math.ceil(previous / options.get("token_alignment", 128)) * options.get("token_alignment", 128)
            mask[start:end] = True
            previous = end
        result = result * mask[:, None]
    return result.to(torch.bfloat16 if fmt == "bf16" else torch.float32)


def _swiglu_backward_reference(x, gradient, fmt, **options):
    with torch.enable_grad():
        values = x.detach().float().requires_grad_()
        weights = options.get("topk_weights")
        params = (values,) if weights is None else (values, weights.detach().clone().requires_grad_())
        settings = dict(options)
        settings.pop("do_recompute", None)
        if weights is not None:
            settings["topk_weights"] = params[1]
        output = _routed_swiglu_reference(values, "fp32", **settings)
        grads = torch.autograd.grad(output, params, gradient.float())
        result = [grads[0].detach().to(torch.bfloat16 if fmt == "bf16" else torch.float32)]
        if weights is not None:
            result.append(grads[1].detach())
        if options.get("do_recompute"):
            result.append(output.detach().to(torch.bfloat16 if fmt == "bf16" else torch.float32))
        return tuple(result)


def _quantized_swiglu(x, fmt, channels, **options):
    data, scale = operators.swiglu_forward_and_per_token_cast(x, fmt, channels, **options)
    return data.view(torch.uint8), scale


def _quantized_swiglu_reference(x, fmt, channels, **options):
    output = _routed_swiglu_reference(x, "fp32", **options)
    encoded, scale = _quant_reference(output, fmt, (1, channels))
    prefixes = options.get("psum_num_tokens_per_expert")
    if prefixes is not None:
        valid = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        previous = 0
        for end in prefixes.tolist():
            alignment = options.get("token_alignment", 128)
            valid[math.ceil(previous / alignment) * alignment:end] = True
            previous = end
        scale[~valid] = 1
    return encoded, scale


def _quantized_swiglu_backward(x, gradient, fmt, channels, **options):
    result = operators.swiglu_backward_and_per_token_cast(x, gradient, fmt, channels, **options)
    return result[0], (result[1][0].view(torch.uint8), result[1][1]), *result[2:]


def _quantized_swiglu_backward_reference(x, gradient, fmt, channels, **options):
    values = _dequant_reference(x, "fp32", (1, channels))
    result = _swiglu_backward_reference(values, gradient, "fp32", **options)
    encoded = _quant_reference(result[0], fmt, (1, channels))
    suffix = list(result[1:])
    if options.get("do_recompute"):
        suffix[-1] = suffix[-1].to(torch.bfloat16)
    return result[0].to(torch.bfloat16), encoded, *suffix


def _counted_swiglu(x, fmt, **options):
    counts = torch.arange(4, dtype=torch.int64, device=x.device)
    for _ in range(2):
        result = operators.swiglu_forward(x, fmt, clamped_count=counts, **options)
    return result, counts


def _counted_swiglu_reference(x, fmt, **options):
    gate, up = x.detach().chunk(2, -1)
    limit = options["clamp_value"]
    counts = torch.stack(((gate > limit).sum(), (up > limit).sum(), (up < -limit).sum(),
                          gate.new_tensor(gate.numel(), dtype=torch.int64)))
    return _routed_swiglu_reference(x, fmt, **options), torch.arange(4, device=x.device) + 2 * counts


def swiglu_extended_cases(dtype: torch.dtype) -> list[PrecisionCase]:
    """Keep weighted, padded and quantized ABI fixtures separate from legacy inputs."""
    inputs = Inputs(torch.float32 if dtype == torch.float8_e4m3fn else dtype)
    x = inputs.tensor(128, 256, scale=1.0)
    weights = inputs.tensor(128, dtype=torch.float32, scale=0.5)
    options = {"topk_weights": weights, "routed_scaling_factor": 1.3, "clamp_value": 1.25}
    cases = []
    for name, settings in (("weighted", options),
                           ("expert_padding", {**options, "psum_num_tokens_per_expert":
                                               torch.tensor([3, 128], dtype=torch.int32)})):
        source = x
        if name == "expert_padding":
            source = x.detach().clone()
            source[3:] = 0
            source.requires_grad_()
            routed_weights = weights.detach().clone()
            routed_weights[3:] = 0
            settings = {**settings, "topk_weights": routed_weights.requires_grad_()}
        cases.append(PrecisionCase("swiglu", "routed", f"routed_{name}", operators.swiglu_forward,
                                   _routed_swiglu_reference, (source, "fp32"), settings, inputs.dtype))
    cases.append(PrecisionCase("swiglu", "routed", "accumulating_clamp_counts", _counted_swiglu,
                               _counted_swiglu_reference, (x[:2], "fp32"), {"clamp_value": 1.25}, inputs.dtype))
    gradient = inputs.tensor(128, 128, grad=False)
    cases.append(PrecisionCase("swiglu", "backward", "explicit_weighted_recompute", operators.swiglu_backward,
                               _swiglu_backward_reference, (x.detach(), gradient, "bf16" if dtype == torch.bfloat16
                                                          else "fp32"),
                               {**options, "do_recompute": True}, inputs.dtype))
    # Separate activation VJP error from FP8 half-way rounding; random activations are covered above.
    up = torch.tensor([1.0, 0.5, 0.25, 0.125]).repeat(32).expand(128, -1)
    quant_forward = torch.cat((torch.ones_like(up), up), -1).to(inputs.dtype)
    cases.append(PrecisionCase("swiglu", "quantized", "quantized_weighted", _quantized_swiglu,
                               _quantized_swiglu_reference, (quant_forward, "e4m3", 32), options, inputs.dtype))
    padded = quant_forward.detach().clone()
    padded[3:] = 0
    padded_weights = weights.detach().clone()
    padded_weights[3:] = 0
    cases.append(PrecisionCase("swiglu", "quantized", "quantized_expert_padding", _quantized_swiglu,
                               _quantized_swiglu_reference, (padded, "e4m3", 32),
                               {"topk_weights": padded_weights, "psum_num_tokens_per_expert":
                                torch.tensor([3, 128], dtype=torch.int32)}, inputs.dtype))
    quant_input = torch.cat((torch.zeros(128, 128), torch.ones(128, 128)), -1)
    codes, scales = _quant_reference(quant_input, "e4m3", (1, 32))
    quant_options = {"topk_weights": torch.full((128,), 0.5), "routed_scaling_factor": 1.0, "do_recompute": True}
    cases.append(PrecisionCase("swiglu", "quantized_backward", "fp8_weighted_recompute", _quantized_swiglu_backward,
                               _quantized_swiglu_backward_reference,
                               ((codes.view(torch.float8_e4m3fn), scales), gradient.to(torch.bfloat16), "e4m3", 32),
                               quant_options, torch.bfloat16))
    return cases


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
    return cases + cached_rope_cases(dtype) + swiglu_extended_cases(dtype)


def _engram_hash(tokens, multipliers, vocab_sizes, offsets, image_token_mask=None):
    result = torch.empty(multipliers.shape[0], tokens.shape[0], offsets.shape[-1], dtype=torch.int32)
    histories, factors = tokens.cpu().tolist(), multipliers.cpu().tolist()
    for layer, layer_factors in enumerate(factors):
        for row, history in enumerate(histories):
            value = 0
            columns = []
            for index, token in enumerate(history):
                product = token * layer_factors[index] % 2**64
                product = product if product < 2**63 else product - 2**64
                value ^= product
                if index:
                    columns.extend(value % size for size in vocab_sizes[layer, index - 1].cpu().tolist())
            result[layer, row] = torch.tensor(columns, dtype=torch.int32) + offsets[layer].cpu()
    if image_token_mask is not None:
        result[:, image_token_mask.cpu()] = -1
    return result.to(tokens.device)


def _engram_gate(hidden, kv, wh, we, clamp_value, eps, image_token_mask=None):
    shape, width = hidden.shape, hidden.shape[-1]
    x = hidden.float().reshape(-1, shape[-2], width)
    packed = kv.float().reshape(-1, shape[-2] + 1, width)
    outputs = []
    for head in range(shape[-2]):
        left = x[:, head] / (x[:, head].square().mean(-1, keepdim=True) + eps).sqrt()
        right = packed[:, head] / (packed[:, head].square().mean(-1, keepdim=True) + eps).sqrt()
        dot = ((left * wh[head]) * (right * we[head])).sum(-1) / math.sqrt(width)
        root = torch.where(dot >= 0, dot.abs().clamp_min(clamp_value).sqrt(),
                           -dot.abs().clamp_min(clamp_value).sqrt())
        outputs.append(x[:, head] + root.sigmoid()[:, None] * packed[:, -1])
    result = torch.stack(outputs, dim=1).reshape(shape).to(hidden.dtype)
    if image_token_mask is not None:
        result = torch.where(image_token_mask.reshape(shape[:-2])[..., None, None], hidden, result)
    return result


def engram_cases(dtype: torch.dtype, hardware: bool = False) -> list[PrecisionCase]:
    """Exercise hash overflow/masks and every differentiable gate input, including zero-dot."""
    inputs = Inputs(torch.bfloat16 if dtype == torch.int32 else dtype)
    width = 128 if hardware else 16
    tokens = torch.tensor([[1, 3, 5], [-7, 2, 0], [4, 0, 0]], dtype=torch.int32)
    factors = torch.tensor([[2**62 + 3, 11, 7], [2**40 + 1, 3, 5]], dtype=torch.int64)
    sizes = torch.tensor([[[37, 41], [43, 47]], [[31, 29], [23, 19]]], dtype=torch.int32)
    offsets = torch.cat((torch.zeros(2, 1, dtype=torch.int32), sizes.flatten(1).cumsum(1)[:, :-1]), 1).int()
    cases = [PrecisionCase("engram", "hash", "hash_masked" if masked else "hash_overflow", operators.engram_hash,
                           _engram_hash, (tokens, factors, sizes, offsets,
                                          torch.tensor([False, True, False]) if masked else None))
             for masked in (False, True)]
    for name, magnitude, mask in (("regular", 0.2, None), ("zero_dot", 0.0, None),
                                  ("masked", 0.2, torch.tensor([False, True, False]))):
        cases.append(PrecisionCase(
            "engram", "gate", name, operators.engram_gate, _engram_gate,
            (inputs.tensor(3, 4, width, scale=magnitude), inputs.tensor(3, 5 * width),
             inputs.tensor(4, width), inputs.tensor(4, width), 1e-4, 1e-6, mask), precision_dtype=inputs.dtype,
        ))
    return cases


def _canonical_gate(*args, **kwargs):
    indices, weights = operators.moe_topk_gate(*args, **kwargs)
    order = indices.argsort(-1)
    return indices.gather(-1, order), weights.gather(-1, order)


def _gate_reference(logits, top_k, shared=False, shared_count=0, scale=1.0, rank=0, **options):
    scores = torch.logaddexp(logits, torch.zeros_like(logits)).sqrt()
    correction = options.get("bias")
    if correction is None:
        correction = torch.zeros_like(scores)
    if options.get("image_token_mask") is not None:
        correction = torch.where(options["image_token_mask"][:, None], options["image_bias"], correction)
    indices = (scores + correction).argsort(-1, descending=True)[..., :top_k].sort(-1).values
    selected = scores.gather(-1, indices)
    weights = selected / (selected.sum(-1, keepdim=True) + 1e-20) * scale
    if shared:
        extra = torch.arange(shared_count, device=logits.device).expand(logits.shape[0], -1) + logits.shape[-1]
        indices = torch.cat((indices, extra), -1)
        weights = torch.cat((weights, weights.new_ones(logits.shape[0], shared_count)), -1)
    if options.get("to_physical_map") is not None:
        seed = rank + torch.arange(logits.shape[0], device=logits.device)[:, None] * 23333
        replica = seed % options["logical_count"][indices]
        indices = options["to_physical_map"][indices, replica].long()
    if options.get("mask") is not None:
        valid = options["mask"][:, None]
        indices, weights = indices.masked_fill(~valid, -1), weights.masked_fill(~valid, 0)
    order = indices.argsort(-1)
    return indices.gather(-1, order), weights.gather(-1, order)


def _normalize_reference(weights):
    denominator = sum(weights[:, column] for column in range(weights.shape[1])) + 1e-20
    return denominator, torch.stack([weights[:, column] / denominator for column in range(weights.shape[1])], -1)


def _gate_backward_reference(scores, indices, weights, gradient, scale, **options):
    del weights
    with torch.enable_grad():
        recovered = scores.detach().requires_grad_()
        selected = recovered.gather(-1, indices.clamp_min(0)).masked_fill(indices < 0, 0)
        normalized = selected / (selected.sum(-1, keepdim=True) + 1e-20) * scale
        loss = (normalized * gradient).sum()
        auxiliary = options.get("score_sum_gradient", options.get("grad_scores_sum"))
        if auxiliary is not None:
            dense = recovered / recovered.sum(-1, keepdim=True)
            dense = dense.masked_fill(~options["mask"][:, None], 0)
            groups = auxiliary.shape[0]
            loss = loss + (dense.reshape(groups, -1, dense.shape[-1]).sum(1) * auxiliary).sum()
        score_gradient = torch.autograd.grad(loss, recovered)[0]
        logits = scores.square().expm1().log()
        derivative = torch.where(scores > 0, logits.sigmoid() / (2 * scores), 0)
        return score_gradient * derivative


def router_stage_cases() -> list[PrecisionCase]:
    """Router stage cases."""
    inputs = Inputs(torch.float32)
    logits, bias = inputs.tensor(4, 128), torch.linspace(-2, 2, 128)
    cases = []
    for top_k in (1, 8):
        for masked in (False, True):
            options = {"bias": bias, "image_bias": -bias, "image_token_mask": torch.tensor([False, True] * 2)}
            if masked:
                options["mask"] = torch.tensor([True, False, True, False])
            # Pin the normalized Top-1 structural-zero case; random Top-1 roundoff is checked separately.
            source = torch.zeros_like(logits, requires_grad=True) if top_k == 1 else logits
            cases.append(PrecisionCase("moe_router", "topk_gate", f"fused_top{top_k}_mask_{masked}",
                                       _canonical_gate, _gate_reference, (source, top_k, False, 0, 1.7, 0), options))
    for zero in (False, True):
        weights = inputs.tensor(4, 8, scale=0.0 if zero else 0.2).abs()
        cases.append(PrecisionCase("moe_router", "normalize_weights", f"normalize_zero_{zero}",
                                   operators.normalize_routing_weights, _normalize_reference, (weights,)))
    physical = torch.arange(260, dtype=torch.int32).reshape(130, 2)
    cases.append(PrecisionCase("moe_router", "topk_gate", "mapped_shared_experts", _canonical_gate,
                               _gate_reference, (logits, 8, True, 2, 1.7, 3),
                               {"to_physical_map": physical,
                                "logical_count": torch.full((130,), 2, dtype=torch.int32)}))
    scores = inputs.tensor(4, 128, grad=False).square().add(0.5)
    indices = scores.topk(8, sorted=True).indices
    selected = scores.gather(-1, indices)
    weights = selected * 1.7 / selected.sum(-1, keepdim=True)
    for aux in (False, True):
        options = {"mask": torch.tensor([True, False, True, False]),
                   "score_sum_gradient": inputs.tensor(2, 128, grad=False)} if aux else {}
        cases.append(PrecisionCase("moe_router", "topk_gate_backward", f"backward_aux_{aux}",
                                   operators.moe_topk_gate_backward, _gate_backward_reference,
                                   (scores, indices, weights, inputs.tensor(4, 8, grad=False), 1.7), options))
    return cases


def router_cases(dtype: torch.dtype, hardware: bool = False) -> list[PrecisionCase]:
    """Retain original router fixtures and add independent fused-stage contracts."""
    return [case for case in expert_cases(dtype, hardware) if case.family == "moe_router"] + router_stage_cases()


def _quant_reference(x, fmt, block_size, *, round_sf=False):
    """Build scaling blocks independently on the host, with scalar FP4 nearest-even rounding."""
    source = x.detach().float().cpu()
    rows, columns = source.shape
    scale = torch.empty(math.ceil(rows / block_size[0]), math.ceil(columns / block_size[1]))
    scaled = torch.empty_like(source)
    maximum = 448 if fmt == "e4m3" else 6
    for row in range(scale.shape[0]):
        for column in range(scale.shape[1]):
            region = (slice(row * block_size[0], (row + 1) * block_size[0]),
                      slice(column * block_size[1], (column + 1) * block_size[1]))
            amax = max(float(source[region].abs().max()), 1e-4 if fmt == "e4m3" else 6 * 2**-126)
            factor = amax / maximum
            if round_sf:
                factor = 2**math.ceil(math.log2(factor))
            scale[row, column] = factor
            scaled[region] = source[region] * (1 / factor if round_sf else maximum / amax)
    if fmt == "e4m3":
        data = scaled.to(torch.float8_e4m3fn)
    else:
        levels = (0, 0.5, 1, 1.5, 2, 3, 4, 6)
        codes = [min(range(8), key=lambda i, v=value: (abs(abs(v) - levels[i]), i % 2))
                 | (8 if math.copysign(1, value) < 0 else 0) for value in scaled.flatten().tolist()]
        codes = torch.tensor(codes, dtype=torch.uint8).reshape(rows, columns)
        data = (codes[:, ::2] | (codes[:, 1::2] << 4)).view(torch.int8)
    return data.view(torch.uint8).to(x.device), scale.to(x.device)


def _token_quant_reference(x, fmt, channels, **options):
    return _quant_reference(x, fmt, (1, channels), **options)


def _token_quant(x, *args, **options):
    data, scale = operators.per_token_cast(x, *args, **options)
    return data.view(torch.uint8), scale


def _block_quant(x, *args, **options):
    data, scale = operators.per_block_cast(x, *args, **options)
    return data.view(torch.uint8), scale


def _dequant_reference(pair, fmt, block_size):
    data, scale = (tensor.detach().cpu() for tensor in pair)
    if data.dtype == torch.int8:
        raw = data.view(torch.uint8)
        codes = torch.stack((raw & 15, raw >> 4), -1).flatten(1)
        levels = torch.tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6))
        values = levels[(codes & 7).long()] * torch.where(codes < 8, 1.0, -1.0)
    else:
        values = data.float()
    result = torch.empty_like(values)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            result[row, column] = values[row, column] * scale[row // block_size[0], column // block_size[1]]
    return result.to(device=pair[0].device, dtype=torch.bfloat16 if fmt == "bf16" else torch.float32)


def quantization_cases(dtype: torch.dtype, hardware: bool = False) -> list[PrecisionCase]:
    """Check packed data bytes, scale metadata and dequantization without STE gradients."""
    del hardware
    input_dtype = dtype if dtype in (torch.float32, torch.float16, torch.bfloat16) else torch.float32
    inputs, cases = Inputs(input_dtype), []
    for fmt in ("e4m3", "e2m1"):
        for rounded in (False, True):
            cases.append(PrecisionCase("quantization", "per_token", f"token_{fmt}_rounded_{rounded}",
                                       _token_quant, _token_quant_reference,
                                       (inputs.tensor(4, 256, grad=False), fmt, 128), {"round_sf": rounded}))
        cases.append(PrecisionCase("quantization", "per_block", f"block_{fmt}", _block_quant, _quant_reference,
                                   (inputs.tensor(128, 256, grad=False), fmt, (128, 128))))
    formats = ("e2m1",) if dtype == torch.int8 else ("e4m3",) if dtype == torch.float8_e4m3fn else ("e4m3", "e2m1")
    for fmt in formats:
        raw, scale = _quant_reference(inputs.tensor(2, 128, grad=False), fmt, (1, 128))
        data = raw.view(torch.float8_e4m3fn if fmt == "e4m3" else torch.int8)
        for output in ("fp32", "bf16"):
            cases.append(PrecisionCase("quantization", "cast_back", f"decode_{fmt}_{output}",
                                       operators.cast_back, _dequant_reference, ((data, scale), output, (1, 128)),
                                       precision_dtype=torch.float32))
    return cases


def tile_kernels_cases(family: str, dtype: torch.dtype) -> list[PrecisionCase]:
    """Keep TileKernels-specific dimensions separate from existing backend fixtures."""
    inputs = Inputs(dtype)
    builders = {"engram": engram_cases, "quantization": quantization_cases, "mhc_pre": mhc_pre_cases}
    if family in builders:
        return builders[family](dtype, hardware=True)
    if family == "mhc_post":
        return [PrecisionCase(
            family, "standard", f"tile_kernels_hidden_{hidden}", operators.mhc_post, _mhc,
            (inputs.tensor(2, 5, hidden), inputs.tensor(2, 5, 4 * hidden),
             inputs.tensor(2, 5, 4, dtype=torch.float32), inputs.tensor(2, 5, 4, 4, dtype=torch.float32), 4),
            precision_dtype=dtype,
        ) for hidden in (128, 1280)]
    if family == "sinkhorn":
        return [PrecisionCase(
            family, "standard", f"tile_kernels_iterations_{iterations}", operators.sinkhorn, _sinkhorn,
            (inputs.tensor(2, 5, 4, 4),), {"sinkhorn_iters": iterations, "eps": eps}, dtype,
        ) for iterations, eps in ((1, 1e-6), (3, 1e-4), (20, 1e-6))]
    if family == "moe_router":
        return tile_kernels_router_cases(dtype)
    if family == "swiglu":
        extended = swiglu_extended_cases(dtype)
        for case_index, case in enumerate(extended):
            extended[case_index] = PrecisionCase(case.family, case.variant, "tile_kernels_" + case.name,
                                                 case.operation, case.reference, case.args, case.kwargs,
                                                 case.precision_dtype)
        return [
            PrecisionCase(family, "standard", "tile_kernels_last_dim", operators.swiglu, _swiglu,
                          (inputs.tensor(2, 5, 256, scale=2.0),), precision_dtype=dtype),
            PrecisionCase(family, "standard", "tile_kernels_nonlast_dim", operators.swiglu, _swiglu,
                          (inputs.tensor(2, 256, 5),), {"dim": 1}, dtype),
            PrecisionCase(family, "standard", "tile_kernels_saturated_gate", operators.swiglu, _swiglu,
                          (inputs.tensor(2, 5, 256, scale=12.0),), precision_dtype=dtype),
        ] + [PrecisionCase(family, "clamped_fp32", f"tile_kernels_clamped_{limit}",
                          operators.swiglu_clamped, _clamped,
                          (inputs.tensor(2, 128, scale=3.0), inputs.tensor(2, 128, scale=3.0), limit),
                          precision_dtype=dtype) for limit in (0.0, 1.25)] + extended
    if family != "rope":
        raise ValueError(f"No TileKernels precision fixtures for {family!r}")
    query, key = inputs.tensor(2, 4, 5, 128), inputs.tensor(2, 2, 5, 128)
    cases = []
    for name, width in (("partial_tail", 64), ("full_rotation", 128), ("identity", 0)):
        angles = inputs.tensor(1, 5, width // 2, grad=False, scale=1.0).repeat(1, 1, 2)
        cases.append(PrecisionCase(family, "interleaved", f"tile_kernels_{name}", operators.rope_interleaved,
                                   _interleaved, (query, key, angles.cos(), angles.sin()), precision_dtype=dtype))
    cos = inputs.tensor(1, 5, 64, dtype=torch.float32)
    sin = inputs.tensor(1, 5, 64, dtype=torch.float32)
    for name, args, kwargs in (
        ("frequency_gradients", (query, key, cos, sin), {}),
        ("sequence_first_partial", (query.transpose(1, 2), key.transpose(1, 2), cos, sin), {"unsqueeze_dim": 2}),
        ("unbatched_tables", (query, key, cos.squeeze(0), sin.squeeze(0)), {}),
        ("broadcast_sequence", (query, key, cos[:, :1], sin[:, :1]), {}),
    ):
        cases.append(PrecisionCase(family, "interleaved", f"tile_kernels_{name}", operators.rope_interleaved,
                                   _interleaved, args, kwargs, dtype))
    for width in (64, 128):
        cases.append(PrecisionCase(
            family, "interleaved", f"tile_kernels_batched_frequency_gradients_{width}", operators.rope_interleaved,
            _interleaved, (query, key, inputs.tensor(2, 5, width), inputs.tensor(2, 5, width)),
            precision_dtype=dtype,
        ))
    cached = cached_rope_cases(dtype)
    return cases + [PrecisionCase(case.family, case.variant, "tile_kernels_" + case.name, case.operation,
                                  case.reference, case.args, case.kwargs, case.precision_dtype) for case in cached]


def tile_kernels_router_cases(dtype: torch.dtype) -> list[PrecisionCase]:
    """Cover learned projection gradients, modality corrections and Top-1 normalization."""
    inputs = Inputs(dtype)
    cases = []
    for top_k, modality in ((1, "text"), (8, "text"), (8, "mixed"), (8, "image")):
        mask = None if modality == "text" else torch.tensor([[False, True, False, True, True]])
        if modality == "image":
            mask = torch.ones(1, 5, dtype=torch.bool)
        cases.append(PrecisionCase(
            "moe_router", "deepseek_v41", f"tile_kernels_top{top_k}_{modality}", _canonical_router, _router,
            (inputs.tensor(1, 5, 128, scale=0.05), inputs.tensor(128, 128, scale=0.05),
             torch.linspace(-8, 8, 128), torch.linspace(8, -8, 128) if mask is not None else None,
             mask, top_k, "sqrtsoftplus", 1.7), precision_dtype=dtype,
        ))
    return cases + router_stage_cases()


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
