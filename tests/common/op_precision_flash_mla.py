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
"""Independent scalar-head sparse inference oracles and known-format KV cache fixtures."""

from __future__ import annotations

import math

import torch

from hyper_parallel.components import ops as operators
from tests.common.op_precision import PrecisionCase


def _cache_reference(cache):
    if cache.shape[-1] == 512:
        return cache.float()
    raw = cache.view(torch.uint8).reshape(-1, cache.shape[-1])
    rows = []
    for row in raw:
        if row.numel() == 528:
            values = row[:512].contiguous().view(torch.float8_e4m3fn).float()
            factors = [2.0**(int(exponent) - 127) for exponent in row[512:]]
            columns = torch.cat([values[i*32:(i+1)*32] * factor for i, factor in enumerate(factors)])
        else:
            levels = (0, 0.5, 1, 1.5, 2, 3, 4, 6)
            values = []
            for byte in row[:256].tolist():
                for code in (byte & 15, byte >> 4):
                    values.append(levels[code & 7] * (-1 if code & 8 else 1))
            scales = row[256:].contiguous().view(torch.float8_e4m3fn).float()
            columns = torch.stack([scales[i // 16] * value for i, value in enumerate(values)])
        rows.append(columns)
    return torch.stack(rows).reshape(*cache.shape[:-1], 512).to(cache.device)


def _attention_reference(query, banks, sparse_ids, scale, sink, lengths):
    tokens, heads, width = query.shape
    output = torch.zeros_like(query)
    maximum = query.new_full((tokens, heads), -float("inf"), dtype=torch.float32)
    lse = query.new_full((tokens, heads), float("inf"), dtype=torch.float32)
    with torch.no_grad():
        for token in range(tokens):
            for head in range(heads):
                values = []
                for bank, ids, limit in zip(banks, sparse_ids, lengths):
                    kv_head = head // (heads // bank.shape[-2])
                    row = ids[token, kv_head]
                    count = row.numel() if limit is None else int(limit[token])
                    for index in row[:count].tolist():
                        if 0 <= index < bank.shape[0]:
                            values.append(bank[index, kv_head].float())
                if not values:
                    continue
                vectors = torch.stack(values)
                scores = torch.stack([(query[token, head].float() * value).sum() * scale for value in values])
                weights = scores.softmax(0)
                maximum[token, head] = scores.max()
                lse[token, head] = scores.logsumexp(0)
                if sink is not None:
                    weights = weights * (lse[token, head] - sink[head]).sigmoid()
                output[token, head] = (vectors * weights[:, None]).sum(0).to(query.dtype)
    return output, maximum, lse


def _prefill_reference(query, keys, indices, scale, *, attention_sink=None, topk_length=None, **options):
    del options
    return _attention_reference(query, [_cache_reference(keys)], [indices], scale, attention_sink, [topk_length])


def _decode_reference(query, cache, indices, scheduler_metadata=None, *, softmax_scale=None,
                      attention_sink=None, topk_length=None, extra_key_cache=None,
                      extra_indices=None, extra_topk_length=None, **options):
    del options, scheduler_metadata
    batch, sequence, heads, width = query.shape
    banks = [_cache_reference(cache).reshape(-1, 1, width)]
    ids = [indices.reshape(batch * sequence, 1, -1)]
    lengths = [None if topk_length is None else topk_length[:, None].expand(-1, sequence).flatten()]
    if extra_key_cache is not None:
        banks.append(_cache_reference(extra_key_cache).reshape(-1, 1, width))
        ids.append(extra_indices.reshape(batch * sequence, 1, -1))
        lengths.append(None if extra_topk_length is None else extra_topk_length[:, None].expand(-1, sequence).flatten())
    output, _, lse = _attention_reference(query.flatten(0, 1), banks, ids,
                                         width**-0.5 if softmax_scale is None else softmax_scale,
                                         attention_sink, lengths)
    return output.reshape(query.shape), lse.reshape(batch, sequence, heads).transpose(1, 2)


def _stats(result):
    # Stats legitimately contain infinities for empty rows; compare flags as well as all finite entries.
    values = []
    for tensor in result:
        flags = tensor.isinf()
        values.extend((tensor.masked_fill(flags, 0), tensor.isposinf(), tensor.isneginf()))
    return tuple(values)


def _prefill(*args, **options):
    return _stats(operators.flash_mla_sparse_prefill(*args, **options))


def _prefill_oracle(*args, **options):
    return _stats(_prefill_reference(*args, **options))


def _decode(*args, **options):
    return _stats(operators.flash_mla_sparse_decode(*args, **options))


def _decode_oracle(*args, **options):
    return _stats(_decode_reference(*args, **options))


def flash_mla_cases(dtype: torch.dtype, hardware: bool = False) -> list[PrecisionCase]:
    """Flash mla cases."""
    generator = torch.Generator().manual_seed(1711051)
    heads = 64 if hardware else 2
    query = (torch.randn(3, heads, 512, generator=generator) * 0.2).to(dtype)
    keys = (torch.randn(8, 1, 512, generator=generator) * 0.2).to(dtype)
    indices = torch.tensor([[[0, 2, 4, -1]], [[1, 3, -1, -1]], [[-1, -1, -1, -1]]], dtype=torch.int32)
    cases = [PrecisionCase("deepseek_sparse_attention", "sparse_prefill", f"sparse_sink_{sink}",
                           _prefill, _prefill_oracle, (query, keys, indices, 1 / math.sqrt(512)),
                           {"attention_sink": torch.zeros(heads) if sink else None,
                            "topk_length": torch.tensor([2, 1, 0], dtype=torch.int32)}, dtype)
             for sink in (False, True)]
    data = (torch.arange(8*512).reshape(2, 4, 1, 512) % 9 - 4).to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.full((2, 4, 1, 16), 127, dtype=torch.uint8)
    cache = torch.cat((data, scales), -1)
    extra = torch.cat((torch.full((2, 4, 1, 256), 0x91, dtype=torch.uint8),
                       torch.full((2, 4, 1, 32), 0.25).to(torch.float8_e4m3fn).view(torch.uint8)), -1)
    q = query[:2].reshape(1, 2, heads, 512)
    ids = indices[:2, 0].reshape(1, 2, 4)
    for extra_cache in (False, True):
        options = {"attention_sink": torch.zeros(heads), "topk_length": torch.tensor([2], dtype=torch.int32)}
        if extra_cache:
            options.update(extra_key_cache=extra, extra_indices=ids,
                           extra_topk_length=torch.tensor([1], dtype=torch.int32))
        cases.append(PrecisionCase("deepseek_sparse_attention", "sparse_decode", f"fp8_extra_fp4_{extra_cache}",
                                   _decode, _decode_oracle, (q, cache, ids), options, dtype))
    return cases
