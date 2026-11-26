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
"""Causal TopK selection for the DeepSeek V4.1 DSA Indexer."""

from __future__ import annotations

from functools import partial
from collections.abc import Callable
from typing import Any, Protocol

import torch

from hyper_parallel.components.functional.dsa_indexer import dsa_indexer as fused_packed_indexer
from hyper_parallel.components.functional.lightning_indexer import (
    lightning_indexer, sort_key_indices, validate_indexer_tensors,
)

from .._dependency import load_cann, require_package, require_symbol
from .._dsa_layout import DsaQueryLayout, ReduceSum, packed_segments
from .gemm import decode_operand, kernel_symbol, require_quantized_inference
from .quantization import QuantTensor
from ..registry import OpRegistry
from ..spec import OpSpec


class SelectTopKFn(Protocol):
    """Discrete row-wise Top-K with explicit bounds, offsets and padding values."""

    def __call__(self, x: torch.Tensor, topk: int, **options: Any) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return optional values and indices without an autograd graph."""


DSA_TOPK = OpSpec[SelectTopKFn](config_key="dsa_indexer", variant="topk")


@torch.no_grad()
def _native_topk(x, topk, *, sorted=False, begin=None, end=None, indices_type=torch.int32,
                 sorted_index=False, hint=None, output_indices=None, output_indices_offset=None,
                 index_out_of_bounds_fill_value=2147483647, value_out_of_bounds_fill_value=-float("inf"),
                 return_value=True, abort_when_nan_found=True):
    if begin is not None or hint is not None:
        raise NotImplementedError("DeepSelect reserves begin/hint; neither is currently supported")
    if sorted and sorted_index:
        raise ValueError("value sorting and index sorting are mutually exclusive")
    values = x.new_full((x.shape[0], topk), value_out_of_bounds_fill_value)
    indices = torch.full((x.shape[0], topk), index_out_of_bounds_fill_value, dtype=indices_type, device=x.device)
    for row in range(x.shape[0]):
        length = x.shape[1] if end is None else int(end[row])
        source = x[row, :length]
        if source.isnan().any():
            if abort_when_nan_found:
                raise ValueError("NaN in Top-K input")
            raise NotImplementedError("NaN fallback is backend-specific and has no portable value result")
        count = min(topk, length)
        selected_values, selected = source.topk(count, sorted=sorted)
        if sorted_index:
            order = selected.argsort()
            selected, selected_values = selected[order], selected_values[order]
        values[row, :count] = selected_values
        offset = 0 if output_indices_offset is None else output_indices_offset[row]
        indices[row, :count] = selected.to(indices_type) + offset
    if output_indices is not None:
        output_indices.copy_(indices)
        indices = output_indices
    return values if return_value else None, indices


def _load_select() -> SelectTopKFn:
    kernels = require_package("deep_select")
    require_symbol(kernels, "topk")
    require_symbol(kernels, "get_stride_requirement")

    @torch.no_grad()
    def run(x: Any, topk: Any, **options: Any) -> Any:
        """Align row storage without changing the logical vocabulary width."""
        alignment, _ = kernels.get_stride_requirement()
        columns = (x.shape[-1] * x.element_size() + alignment - 1) // alignment * alignment // x.element_size()
        if x.stride(-1) != 1 or x.stride(0) * x.element_size() % alignment:
            padded = torch.nn.functional.pad(x.contiguous(), (0, columns - x.shape[-1]), value=-float("inf"))
            x = padded[:, :x.shape[-1]]
        translations = {"output_indices": "output_idx", "output_indices_offset": "output_idx_offset",
                        "index_out_of_bounds_fill_value": "idx_oob_fill_value",
                        "value_out_of_bounds_fill_value": "value_oob_fill_value"}
        settings = {translations.get(name, name): value for name, value in options.items()}
        settings.setdefault("indices_type", torch.int32)
        return kernels.topk(x, topk, **settings)

    return run


def deep_select_stride_requirement() -> tuple[int, int]:
    """Return the optional backend's input/output row-stride requirements in bytes."""
    kernels = require_package("deep_select")
    require_symbol(kernels, "get_stride_requirement")
    return kernels.get_stride_requirement()


class DsaIndexerFn(Protocol):
    """Produce sorted causal TopK key IDs with -1 marking unused slots."""

    def __call__(
        self, x: torch.Tensor, key: torch.Tensor, merge_weight: torch.Tensor, *,
        layout: DsaQueryLayout, sparse_count: int, query_chunk_size: int = 256,
        reduce_sum: ReduceSum | None = None,
    ) -> torch.Tensor:
        """Return int32 key IDs for each query."""


DSA_INDEXER = OpSpec[DsaIndexerFn](config_key="dsa_indexer", variant="compressed_causal_topk")


class PackedDsaIndexerFn(Protocol):
    """Select document-local key IDs and retain the flattened indexer states."""

    def __call__(
        self, x: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
        query_ends: torch.Tensor, key_ends: torch.Tensor, sparse_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return right-aligned causal TopK with -1 padding, followed by TND states and TN weights."""


DSA_PACKED_INDEXER = OpSpec[PackedDsaIndexerFn](config_key="dsa_indexer", variant="packed_causal_topk")


def _native_packed_indexer(
    x: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
    query_ends: torch.Tensor, key_ends: torch.Tensor, sparse_count: int, *, _selection=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select causal document-local keys and return flattened indexer inputs."""
    query, key, weights = (tensor.flatten(0, 1) for tensor in (x, key, weights))
    indices = []
    with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
        for qs, qe, ks, ke in packed_segments(query_ends, key_ends):
            keys = key[ks:ke, 0].float()
            count = min(sparse_count, keys.shape[0])
            key_pos = torch.arange(keys.shape[0], device=x.device)
            for start in range(qs, qe, 128):
                end = min(start + 128, qe)
                q_pos = torch.arange(start, end, device=x.device) - qe + ke - ks
                scores = (query[start:end].float() @ keys.T).relu()
                scores = (scores * weights[start:end].float().unsqueeze(-1)).sum(dim=1)
                scores = scores.masked_fill(key_pos[None, :] > q_pos[:, None], -torch.inf)
                values, ids = (scores.topk(count, dim=-1) if _selection is None
                               else _selection(scores, count))
                ids = ids.masked_fill(~values.isfinite(), -1).to(torch.int32)
                indices.append(torch.nn.functional.pad(ids, (0, sparse_count - count), value=-1))
    return torch.cat(indices).unsqueeze(1), query, key, weights


def _load_packed_indexer() -> PackedDsaIndexerFn:
    kernels = require_package("cann_ops_transformer.ops")
    require_symbol(kernels, "lightning_indexer")
    require_symbol(kernels, "lightning_indexer_metadata")
    return fused_packed_indexer


def _validate_indexer_inputs(
    query: torch.Tensor, key: torch.Tensor, merge_weight: torch.Tensor, layout: DsaQueryLayout, chunk_size: int,
) -> None:
    """Validate compressed indexer tensor layouts and query chunk size."""
    compress_ratio = layout.compress_ratio
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {chunk_size}")
    if query.ndim != 4:
        raise ValueError(f"Indexer query must have shape [B, Q, H, D], got {tuple(query.shape)}")
    batch_size, _, _, head_dim = query.shape
    if key.ndim != 3 or key.shape[::2] != (batch_size, head_dim):
        raise ValueError(
            "compressed index key must have shape [batch, compressed_sequence, head_dim]"
        )
    if merge_weight.shape != query.shape[:3]:
        raise ValueError(
            f"merge_weight must have shape {tuple(query.shape[:3])}, got {tuple(merge_weight.shape)}"
        )
    if layout.minimum_key_indices is not None and layout.minimum_key_indices.shape != query.shape[:2]:
        raise ValueError("minimum_key_indices must have shape [batch, query]")


@torch.no_grad()
def native_dsa_indexer(
    x: torch.Tensor, key: torch.Tensor, merge_weight: torch.Tensor, *,
    layout: DsaQueryLayout, sparse_count: int, query_chunk_size: int = 256,
    reduce_sum: ReduceSum | None = None, _selection: Callable | None = None,
) -> torch.Tensor:
    """Reference ratio-aware causal TopK over a compressed key bank."""
    query = x
    _validate_indexer_inputs(query, key, merge_weight, layout, query_chunk_size)
    batch_size, sequence_length = query.shape[:2]
    compressed_length = key.shape[1]
    top_k = min(sparse_count, compressed_length)
    if top_k <= 0 or sequence_length == 0:
        return torch.empty(
            batch_size,
            sequence_length,
            max(top_k, 0),
            dtype=torch.int32,
            device=query.device,
        )

    visible_keys, minimum_key_indices = layout.compressed_bounds(query)
    key_fp32 = key.float().transpose(1, 2).unsqueeze(1)
    key_positions = torch.arange(compressed_length, device=query.device).view(1, 1, -1)
    selected_chunks = []
    for start in range(0, sequence_length, query_chunk_size):
        end = min(start + query_chunk_size, sequence_length)
        # SAC may retain the matmul result; do not mutate its cached tensor.
        scores = torch.matmul(query[:, start:end].float(), key_fp32).relu()
        scores = (scores * merge_weight[:, start:end].float().unsqueeze(-1)).sum(dim=2)
        if reduce_sum is not None:
            scores = reduce_sum(scores)
        # Bounds are [1, S]; slice the query axis so the mask covers this chunk.
        visible = visible_keys[:, start:end].unsqueeze(-1)
        scores.masked_fill_(key_positions >= visible, float("-inf"))
        if minimum_key_indices is not None:
            minimum = minimum_key_indices[:, start:end].unsqueeze(-1)
            scores.masked_fill_(key_positions < minimum, float("-inf"))
        if _selection is None:
            values, indices = scores.topk(top_k, dim=-1, sorted=False)
        else:
            values, indices = _selection(scores.reshape(-1, compressed_length), top_k)
            values, indices = (value.reshape(*scores.shape[:-1], top_k) for value in (values, indices))
        indices = indices.masked_fill(~torch.isfinite(values), compressed_length)
        indices = sort_key_indices(indices, compressed_length)
        indices = indices.masked_fill(indices == compressed_length, -1)
        selected_chunks.append(indices.to(torch.int32))
    return torch.cat(selected_chunks, dim=1)


def indexer_native_only(x: torch.Tensor, layout: DsaQueryLayout, reduce_sum: ReduceSum | None) -> str | None:
    """Explain why the LI V2 candidate cannot represent this call."""
    if (x.dtype != torch.bfloat16 or x.shape[-1] != 128 or layout.segments is None or reduce_sum is not None):
        return "LI V2 requires BF16 D=128, document segments and no TP score reduction"
    return None


@torch.no_grad()
def _cann_indexer(
    x: torch.Tensor, key: torch.Tensor, merge_weight: torch.Tensor, *,
    layout: DsaQueryLayout, sparse_count: int, query_chunk_size: int = 256,
    reduce_sum: ReduceSum | None = None,
) -> torch.Tensor:
    del reduce_sum
    _validate_indexer_inputs(x, key, merge_weight, layout, query_chunk_size)
    validate_indexer_tensors(x, key, merge_weight)
    return lightning_indexer(x, key, merge_weight, layout.compress_ratio, min(sparse_count, key.shape[1]),
                             layout.segments)


def _load_select_indexer(packed=False):
    select = _load_select()
    implementation = _native_packed_indexer if packed else native_dsa_indexer

    def run(*args: Any, **kwargs: Any) -> Any:
        """Apply the operator contract using the selected implementation."""
        return implementation(*args, **kwargs, _selection=lambda scores, count: select(scores, count, sorted=True))

    return run


class MqaLogitsFn(Protocol):
    """Score selected KV row intervals; each output row starts at its interval's first key."""

    def __call__(
        self,
        q: QuantTensor,
        kv: QuantTensor,
        weights: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        max_length: int,
    ) -> torch.Tensor:
        """Return BF16 [tokens,max_length], with -inf outside each valid prefix."""


class PagedMqaLogitsFn(Protocol):
    """Score quantized paged KV with one physical page table row per query."""

    def __call__(
        self,
        q: QuantTensor,
        cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor | None,
        max_length: int,
        request_ids: torch.Tensor,
        *,
        use_ascend_kv_layout: bool = False,
    ) -> torch.Tensor:
        """Caller refreshes schedule_meta whenever lengths, requests or core count change."""


DSA_MQA = OpSpec[MqaLogitsFn](config_key="dsa_indexer", variant="quantized_mqa")
DSA_PAGED_MQA = OpSpec[PagedMqaLogitsFn](config_key="dsa_indexer", variant="paged_quantized_mqa")


def paged_mqa_metadata(context_lens: torch.Tensor, num_heads: int, request_ids: torch.Tensor) -> torch.Tensor:
    """Construct native scheduling metadata without caching mutable lengths globally."""
    kernel = kernel_symbol(require_package("deep_gemm"), "get_paged_mqa_logits_metadata")
    return kernel(context_lens, num_heads, request_ids)


def _query(q):
    data, scale = q
    tokens, heads = data.shape[0], data.shape[-2]
    values = decode_operand((data.reshape(-1, data.shape[-1]), scale.reshape(tokens * heads, -1)))
    return values.reshape(tokens, heads, -1)


def _mask_prefix(output, lengths):
    return output.masked_fill(
        torch.arange(output.shape[-1], device=output.device) >= lengths.reshape(-1, 1), -float("inf")
    )


def native_mqa(
    q: QuantTensor, kv: QuantTensor, weights: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, max_length: int
) -> torch.Tensor:
    """Compute independent head scores, ReLU and weighted head reduction."""
    require_quantized_inference(q, kv, weights)
    query, keys = _query(q), decode_operand(kv)
    result = torch.full((query.shape[0], max_length), -float("inf"), dtype=torch.bfloat16, device=query.device)
    for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        logits = (query[row] @ keys[start:end].T).relu() * weights[row, :, None].float()
        result[row, : end - start] = logits.sum(0).to(result.dtype)
    return result


def _paged_keys(cache, query_dtype, ascend_layout):
    pages, page_size, _, storage = cache.shape
    raw = cache.reshape(pages, -1)
    # FP8 has one UE8M0 byte per 32 values; FP4 has two values per byte.
    width = storage * 32 // (17 if query_dtype == torch.int8 else 33)
    data_width = width // 2 if query_dtype == torch.int8 else width
    data = raw[:, : page_size * data_width].contiguous().view(query_dtype).reshape(pages, page_size, data_width)
    scales = raw[:, page_size * data_width :].contiguous().view(torch.int16)
    if ascend_layout:
        data = data.reshape(pages, -1, page_size, 32).permute(0, 2, 1, 3).contiguous().reshape_as(data)
        scales = scales.reshape(pages, page_size // 16, -1, 16).permute(0, 1, 3, 2).contiguous()
    return decode_operand((data, scales.reshape(pages, page_size, -1)))


def native_paged_mqa(
    q: QuantTensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor | None,
    max_length: int,
    request_ids: torch.Tensor,
    *,
    use_ascend_kv_layout: bool = False,
) -> torch.Tensor:
    """Decode native page storage and gather logical pages in block_table order."""
    del schedule_meta, request_ids
    require_quantized_inference(q, weights)
    query = _query(q)
    keys = _paged_keys(cache, q[0].dtype, use_ascend_kv_layout)
    result = torch.full((query.shape[0], max_length), -float("inf"), dtype=torch.bfloat16, device=query.device)
    for row, length in enumerate(context_lens.reshape(-1).tolist()):
        pages = (length + cache.shape[1] - 1) // cache.shape[1]
        vectors = keys[block_table[row, :pages].long()].flatten(0, 1)[:length]
        result[row, :length] = ((query[row] @ vectors.T).relu() * weights[row, :, None].float()).sum(0).to(result.dtype)
    return result


def load_mqa() -> MqaLogitsFn:
    """Load unpaged quantized MQA scoring and define its unwritten tail values."""
    kernel = kernel_symbol(require_package("deep_gemm"), "fp8_fp4_mqa_logits")

    def run(q: Any, kv: Any, weights: Any, starts: Any, ends: Any, max_length: Any) -> Any:
        """Apply the operator contract using the selected implementation."""
        require_quantized_inference(q, kv, weights)
        return _mask_prefix(kernel(q, kv, weights, starts, ends, max_length), ends - starts)

    return run


def load_paged_mqa() -> PagedMqaLogitsFn:
    """Load paged scoring; metadata lifetime remains explicit in the public call."""
    kernel = kernel_symbol(require_package("deep_gemm"), "fp8_fp4_paged_mqa_logits")

    def run(
        q: Any,
        cache: Any,
        weights: Any,
        context_lens: Any,
        block_table: Any,
        schedule_meta: Any,
        max_length: Any,
        request_ids: Any,
        *,
        use_ascend_kv_layout: Any=False,
    ) -> Any:
        """Apply the operator contract using the selected implementation."""
        require_quantized_inference(q, weights)
        if schedule_meta is None:
            raise ValueError("DeepGEMM paged MQA requires caller-owned schedule metadata")
        output = kernel(
            q,
            cache,
            weights,
            context_lens,
            block_table,
            schedule_meta,
            max_length,
            request_ids,
            use_ascend_kv_layout=use_ascend_kv_layout,
        )
        return _mask_prefix(output, context_lens)

    return run


def register_dsa_indexer(registry: OpRegistry) -> None:
    """Register the compressed causal TopK contract and its implementations."""
    registry.register_family("dsa_indexer", specs=(DSA_INDEXER, DSA_PACKED_INDEXER, DSA_TOPK, DSA_MQA, DSA_PAGED_MQA))
    for spec, native, fused in ((DSA_MQA, native_mqa, load_mqa), (DSA_PAGED_MQA, native_paged_mqa, load_paged_mqa)):
        registry.register_impl(spec, name="native", devices=("cpu", "cuda", "npu"), load=lambda fn=native: fn)
        registry.register_impl(spec, name="deep_gemm", devices=("npu",), load=fused)
    registry.register_impl(
        DSA_INDEXER, name="native", devices=("cpu", "cuda", "npu"), load=lambda: native_dsa_indexer,
    )
    registry.register_impl(
        DSA_INDEXER, name="cann", devices=("npu",), load=partial(load_cann, _cann_indexer, "lightning_indexer"),
    )
    registry.register_impl(
        DSA_PACKED_INDEXER, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native_packed_indexer,
    )
    registry.register_impl(
        DSA_PACKED_INDEXER, name="cann", devices=("npu",),
        load=_load_packed_indexer,
    )
    registry.register_impl(DSA_TOPK, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native_topk)
    registry.register_impl(DSA_TOPK, name="deep_select", devices=("npu",), load=_load_select)
    registry.register_impl(DSA_INDEXER, name="deep_select", devices=("npu",), load=_load_select_indexer)
    registry.register_impl(DSA_PACKED_INDEXER, name="deep_select", devices=("npu",),
                           load=lambda: _load_select_indexer(packed=True))
