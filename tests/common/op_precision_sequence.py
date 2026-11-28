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
"""Independent precision fixtures for sequence operators."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from hyper_parallel.components import ops as operators
from hyper_parallel.components.modules.attention.linear_attention_utils.sequence_layout import PackedSequenceInfo
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
            if offset is not None:
                value = value + offset
            results.append(value.to(x.dtype))
    result = torch.stack(results, dim=1)
    if mask is None:
        return (result.float() * result.float().sigmoid()).to(x.dtype)
    return x + result * mask[..., None].to(x.dtype)


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
