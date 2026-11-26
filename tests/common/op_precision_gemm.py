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
"""Independent contraction and quantized scoring fixtures for DeepGEMM acceptance."""

from __future__ import annotations

from typing import Any

import torch

from hyper_parallel.components.ops import (
    gemm,
    head_einsum,
    m_grouped_gemm,
    k_grouped_gemm,
    mhc_prenorm_projection,
    mqa_logits,
    paged_mqa_logits,
    paged_mqa_metadata,
    ops,
)
from .op_precision import PrecisionCase


def quantized_fixture(shape: Any, dtype: Any, generator: Any) -> Any:
    """Use exactly representable values and power-of-two K32 scales, avoiding a second quantizer."""
    if dtype == torch.int8:
        codes = torch.randint(0, 16, (*shape[:-1], shape[-1] // 2, 2), generator=generator, dtype=torch.uint8)
        data = (codes[..., 0] | (codes[..., 1] << 4)).view(torch.int8)
    else:
        data = torch.randint(-3, 4, shape, generator=generator).to(torch.float8_e4m3fn)
    exponents = torch.randint(125, 128, (*shape[:-1], shape[-1] // 32), generator=generator, dtype=torch.uint8)
    return data, exponents.contiguous().view(torch.int16)


def _decode_reference(value, transpose=False):
    if not isinstance(value, tuple):
        return value.float().transpose(-1, -2) if transpose else value.float()
    data, scales = value
    if transpose:
        data, scales = data.transpose(-1, -2), scales.transpose(-1, -2)
    if data.dtype == torch.int8:
        axis = -2 if data.stride(-2) == 1 else -1
        raw = data.movedim(axis, -1).contiguous().view(torch.uint8)
        codes = torch.stack((raw.remainder(16), raw.div(16, rounding_mode="floor")), -1).flatten(-2)
        decoded = torch.zeros_like(codes, dtype=torch.float32)
        for code, magnitude in enumerate((0, 0.5, 1, 1.5, 2, 3, 4, 6)):
            decoded += (codes == code) * magnitude - (codes == code + 8) * magnitude
        decoded = decoded.movedim(-1, axis)
    else:
        decoded = data.float()
    exponents = scales.reshape(-1).view(torch.uint8).reshape(*scales.shape[:-1], -1)
    scale = torch.exp2(exponents.float() - 127).repeat_interleave(32, -1)
    return decoded * scale[..., : decoded.shape[-1]]


def gemm_reference(
    a: Any,
    b: Any,
    *,
    layout: Any="nt",
    c: Any=None,
    output_dtype: Any=torch.bfloat16,
    alpha: Any=1.0,
    recipe: Any=(1, 1, 32),
) -> Any:
    """An Einstein contraction independent of the GEMM implementation's matmul path."""
    del recipe
    a, b = _decode_reference(a, layout[0] == "t"), _decode_reference(b, layout[1] == "n")
    result = torch.einsum("mk,nk->mn", a, b) * alpha
    return (result if c is None else result + c.float()).to(output_dtype)


def einsum_reference(
    a: Any,
    b: Any,
    expression: Any,
    *,
    c: Any=None,
    output_dtype: Any=torch.bfloat16,
    recipe: Any=(1, 1, 32),
) -> Any:
    """Einsum reference."""
    del recipe
    if isinstance(a, tuple):
        permutations = {
            "bhr,hdr->bhd": ((1, 0, 2), (0, 1, 2)),
            "bhd,hdr->bhr": ((1, 0, 2), (0, 2, 1)),
            "bhd,bhr->hdr": ((1, 2, 0), (1, 2, 0)),
        }
        decoded = []
        for value, order in zip((a, b), permutations[expression]):
            inverse = tuple(order.index(i) for i in range(3))
            decoded.append(_decode_reference(tuple(t.permute(order) for t in value)).permute(inverse))
        a, b = decoded[0], decoded[1]
    result = torch.einsum(expression, a.float(), b.float())
    return (result if c is None else result + c.float()).to(output_dtype)


def grouped_reference(
    a: Any,
    b: Any,
    ends: Any,
    *,
    mode: Any,
    alignment: Any=256,
    layout: Any="nt",
    c: Any=None,
    output_dtype: Any=torch.bfloat16,
    recipe: Any=(1, 1, 32),
) -> Any:
    """Grouped reference."""
    del recipe
    if mode == "m":
        left, right = _decode_reference(a), _decode_reference(b, layout == "nn")
        result = torch.zeros(left.shape[0], right.shape[1], device=left.device)
    else:
        left, right = _decode_reference(a, layout == "tn"), _decode_reference(b, layout == "tn")
        result = torch.zeros(ends.numel(), left.shape[0], right.shape[0], device=left.device)
    start = 0
    for expert, end in enumerate(ends.tolist()):
        if mode == "m":
            result[start:end] = torch.einsum("mk,nk->mn", left[start:end], right[expert])
        else:
            result[expert] = torch.einsum("mk,nk->mn", left[:, start:end], right[:, start:end])
        start = (end + alignment - 1) // alignment * alignment
    return (result if c is None else result + c.float()).to(output_dtype)


def prenorm_reference(x: Any, weight: Any, *, num_splits: Any=None) -> Any:
    """Prenorm reference."""
    blocks = torch.split(x.float(), 256, -1)
    count, remainder = divmod(len(blocks), num_splits or 1)
    projections, squared = [], []
    start = 0
    for split in range(num_splits or 1):
        part = torch.cat(blocks[start : start + count + (split < remainder)], -1)
        wpart = weight[:, start * 256 : (start + count + (split < remainder)) * 256].float()
        projections.append(torch.einsum("mk,nk->mn", part, wpart))
        squared.append(torch.einsum("mk,mk->m", part, part))
        start += count + (split < remainder)
    return (torch.stack(projections), torch.stack(squared)) if num_splits is not None else (projections[0], squared[0])


def gemm_cases(dtype: Any, hardware: Any=False) -> Any:
    """Cover all physical dense layouts, all head expressions and BF16/FP8/FP4 operands."""
    generator = torch.Generator().manual_seed(578)
    m, n, k = (64, 64, 256) if hardware else (3, 4, 64)
    quantized = dtype in (torch.float8_e4m3fn, torch.int8)
    a = (
        quantized_fixture((m, k), dtype, generator)
        if quantized
        else torch.randn(m, k, generator=generator).to(dtype).requires_grad_()
    )
    b = (
        quantized_fixture((n, k), dtype, generator)
        if quantized
        else torch.randn(n, k, generator=generator).to(dtype).requires_grad_()
    )
    precision = torch.bfloat16 if quantized else dtype
    cases = []
    for layout in ("nt", "nn", "tn", "tt"):
        left = tuple(t.T for t in a) if quantized and layout[0] == "t" else (a.T if layout[0] == "t" else a)
        right = tuple(t.T for t in b) if quantized and layout[1] == "n" else (b.T if layout[1] == "n" else b)
        cases.append(
            PrecisionCase(
                "gemm",
                "standard",
                f"gemm_{layout}",
                gemm,
                gemm_reference,
                (left, right),
                {"layout": layout, "output_dtype": precision},
                precision,
            )
        )
    for accumulate in (False, True):
        c = torch.randn(m if accumulate else 1, n, generator=generator).to(precision).requires_grad_(not quantized)
        cases.append(
            PrecisionCase(
                "gemm",
                "standard",
                "gemm_accumulate" if accumulate else "gemm_bias",
                gemm,
                gemm_reference,
                (a, b),
                {"c": c, "alpha": 0.5 if accumulate else 1.0, "output_dtype": precision},
                precision,
            )
        )
    # FP8 has head einsum kernels; FP4 head einsum is not exported upstream.
    head_dtype = torch.float8_e4m3fn if quantized else dtype
    batch, heads, dim, rank = (64, 4, 64, 64) if hardware else (64, 2, 64, 64)
    for expression in ("bhr,hdr->bhd", "bhd,hdr->bhr", "bhd,bhr->hdr"):
        sizes = {"b": batch, "h": heads, "d": dim, "r": rank}
        inputs, _ = expression.split("->")
        tensors = []
        for axes in inputs.split(","):
            shape = tuple(sizes[axis] for axis in axes)
            if quantized:
                contraction = next(axis for axis in axes if axis not in expression.split("->")[1])
                order = (
                    (1, 2, 0)
                    if contraction == "b"
                    else (0, 2, 1)
                    if contraction == "d" and axes == "hdr"
                    else (1, 0, 2)
                    if axes.startswith("b")
                    else (0, 1, 2)
                )
                pair = quantized_fixture(tuple(shape[i] for i in order), head_dtype, generator)
                inverse = tuple(order.index(i) for i in range(3))
                tensors.append(tuple(t.permute(inverse) for t in pair))
            else:
                tensors.append(torch.randn(shape, generator=generator).to(dtype).requires_grad_())
        cases.append(
            PrecisionCase(
                "gemm",
                "head_einsum",
                expression,
                head_einsum,
                einsum_reference,
                (*tensors, expression),
                {"output_dtype": precision},
                precision,
            )
        )
    return cases


def grouped_gemm_cases(dtype: Any, hardware: Any=False) -> Any:
    """Include an empty expert and padding whose input values must not leak into outputs/gradients."""
    generator = torch.Generator().manual_seed(582)
    alignment = 256 if hardware else 32
    width = 256 if hardware else 64
    ends = torch.tensor([2, alignment, alignment + 3], dtype=torch.int32)
    total = alignment * 2
    quantized = dtype in (torch.float8_e4m3fn, torch.int8)

    def values(shape: Any) -> Any:
        """Values."""
        return (
            quantized_fixture(shape, dtype, generator)
            if quantized
            else torch.randn(shape, generator=generator).to(dtype).requires_grad_()
        )

    precision = torch.bfloat16 if quantized else dtype
    cases = []
    for layout in ("nt", "nn"):
        a, b = values((total, width)), values((3, width, width))
        if layout == "nn":
            b = tuple(t.transpose(-1, -2) for t in b) if quantized else b.transpose(-1, -2)
        cases.append(
            PrecisionCase(
                "grouped_matmul",
                "m_padded",
                f"m_{layout}",
                m_grouped_gemm,
                lambda *args, **kwargs: grouped_reference(*args, mode="m", **kwargs),
                (a, b, ends),
                {"layout": layout, "alignment": alignment, "output_dtype": precision},
                precision,
            )
        )
    for layout in ("tn", "nt"):
        # K-group kernels store [K,M/N] in MN-major order; FP4 packs M/N,
        # while the scaling factors still encode 32-wide groups along K.
        if quantized:
            a_data, b_data = values((total, width))[0], values((total, width))[0]
            a_scale, b_scale = values((width, total))[1].T, values((width, total))[1].T
            a, b = (a_data, a_scale), (b_data, b_scale)
        else:
            a, b = values((total, width)), values((total, width))
        with torch.no_grad():
            for end in ends.tolist():
                stop = (end + alignment - 1) // alignment * alignment
                (a[0] if quantized else a)[end:stop] = 0
                (b[0] if quantized else b)[end:stop] = 0
        if layout == "nt":
            a = tuple(t.T for t in a) if quantized else a.T
            b = tuple(t.T for t in b) if quantized else b.T
        cases.append(
            PrecisionCase(
                "grouped_matmul",
                "k_padded",
                f"k_{layout}",
                k_grouped_gemm,
                lambda *args, **kwargs: grouped_reference(*args, mode="k", **kwargs),
                (a, b, ends),
                {"layout": layout, "alignment": alignment, "output_dtype": precision},
                precision,
            )
        )
    return cases


def prenorm_cases(dtype: Any, hardware: Any=False) -> Any:
    """Exercise remainder blocks and all input gradients with/without an explicit split dimension."""
    generator = torch.Generator().manual_seed(583)
    x = torch.randn(64 if hardware else 3, 768, generator=generator).to(dtype).requires_grad_()
    weight = torch.randn(8, 768, generator=generator).requires_grad_()
    return [
        PrecisionCase(
            "mhc_pre",
            "prenorm_projection",
            f"prenorm_{splits}",
            mhc_prenorm_projection,
            prenorm_reference,
            (x, weight),
            {"num_splits": splits},
            dtype,
        )
        for splits in (None, 2)
    ]


def _finite_scores(scores):
    return scores.masked_fill(torch.isneginf(scores), 0), torch.isneginf(scores)


def _mqa_reference(q, kv, weights, starts, ends, max_length):
    query = _decode_reference((q[0].flatten(0, 1), q[1].reshape(-1, q[1].shape[-1])))
    query = query.reshape(q[0].shape[0], q[0].shape[1], -1)
    keys = _decode_reference(kv)
    rows = []
    for token, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        per_head = torch.einsum("hd,kd->hk", query[token], keys[start:end]).relu()
        values = torch.einsum("hk,h->k", per_head, weights[token].float())
        rows.append(torch.nn.functional.pad(values, (0, max_length - (end - start)), value=-float("inf")))
    return torch.stack(rows).to(torch.bfloat16)


def _mqa_operation(*args):
    return _finite_scores(mqa_logits(*args))


def _mqa_oracle(*args):
    return _finite_scores(_mqa_reference(*args))


def _paged_operation(q, cache, weights, lengths, table, max_length, requests, ascend_layout=False):
    metadata = (
        paged_mqa_metadata(lengths, q[0].shape[-2], requests)
        if ops.get_backends().get("dsa_indexer") == "deep_gemm"
        else None
    )
    return _finite_scores(
        paged_mqa_logits(
            q, cache, weights, lengths, table, metadata, max_length, requests, use_ascend_kv_layout=ascend_layout
        )
    )


def _paged_oracle(q, cache, weights, lengths, table, max_length, requests, ascend_layout=False):
    del requests
    pages, page_size = cache.shape[:2]
    data_width = 32 if q[0].dtype == torch.int8 else 64
    raw = cache.reshape(pages, -1)
    data = raw[:, : page_size * data_width].contiguous().view(q[0].dtype).reshape(pages, page_size, data_width)
    scales = raw[:, page_size * data_width :].contiguous().view(torch.int16).reshape(pages, page_size, 1)
    if ascend_layout:
        data = data.reshape(pages, -1, page_size, 32).transpose(1, 2).reshape_as(data)
        scales = scales.reshape(pages, page_size // 16, -1, 16).transpose(-1, -2).reshape_as(scales)
    # This fixture uses exact unit scales; the independent dense oracle reads only selected pages.
    order = table[0].long()
    keys = (data[order].flatten(0, 1), scales[order].flatten(0, 1))
    query = (q[0].squeeze(1), q[1].squeeze(1))
    starts = torch.zeros_like(lengths.reshape(-1))
    return _finite_scores(_mqa_reference(query, keys, weights, starts, lengths.reshape(-1), max_length))


def mqa_cases(dtype: Any, hardware: Any=False) -> Any:
    """Compare prequantized FP8/FP4 scores, compressed prefixes and native/Ascend paged formats."""
    del hardware
    generator = torch.Generator().manual_seed(587)
    dtype = dtype if dtype in (torch.float8_e4m3fn, torch.int8) else torch.float8_e4m3fn
    q = quantized_fixture((2, 4, 64), dtype, generator)
    kv = quantized_fixture((128, 64), dtype, generator)
    weights = torch.randn(2, 4, generator=generator).to(torch.bfloat16)
    starts, ends = torch.tensor([0, 3], dtype=torch.int32), torch.tensor([5, 10], dtype=torch.int32)
    cases = [
        PrecisionCase(
            "dsa_indexer",
            "quantized_mqa",
            "mqa_compressed",
            _mqa_operation,
            _mqa_oracle,
            (q, kv, weights, starts, ends, 8),
            precision_dtype=torch.bfloat16,
        )
    ]
    pages, page_size = 2, 64
    data = kv[0].reshape(pages, page_size, -1)
    scales = kv[1].reshape(pages, page_size, -1)
    lengths = torch.tensor([[65], [67]], dtype=torch.int32)
    table = torch.tensor([[1, 0], [1, 0]], dtype=torch.int32)
    requests = torch.zeros(2, dtype=torch.int32)
    query = tuple(t.unsqueeze(1) for t in q)
    for ascend_layout in (False, True):
        physical_data, physical_scale = data, scales
        if ascend_layout:
            physical_data = data.reshape(pages, page_size, -1, 32).transpose(1, 2).contiguous().reshape_as(data)
            physical_scale = scales.reshape(pages, page_size // 16, 16, -1).transpose(-1, -2).contiguous()
        fused = torch.cat(
            (
                physical_data.contiguous().view(torch.uint8).reshape(pages, -1),
                physical_scale.contiguous().view(torch.uint8).reshape(pages, -1),
            ),
            -1,
        )
        fused = fused.reshape(pages, page_size, 1, -1)
        cases.append(
            PrecisionCase(
                "dsa_indexer",
                "paged_quantized_mqa",
                f"mqa_pages_{ascend_layout}",
                _paged_operation,
                _paged_oracle,
                (query, fused, weights, lengths, table, 70, requests, ascend_layout),
                precision_dtype=torch.bfloat16,
            )
        )
    return cases
