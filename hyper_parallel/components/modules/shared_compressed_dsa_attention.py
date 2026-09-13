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
"""High-performance shared compressed DeepSeek Sparse Attention.

This module implements the DeepSeek-V4.1 CSA2 training contract: Full layers
publish compressed K=V and index K, Reindex layers produce fresh Top-K indices,
and Reuse layers consume the latest selection. The decoder's first Full layer
can additionally publish the blockwise candidate pool used by later Reindex
layers. Ascend executes selected-token attention through the Omni sparse-
FlashAttention operators. CPU and CUDA use a dense numerical reference that
keeps validation independent of optional NPU packages.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale


class SequenceGatherHandle(Protocol):
    """Handle returned by an asynchronous sequence all-gather."""

    def wait(self) -> torch.Tensor:
        """Wait for communication and return the differentiable result."""


@dataclass
class _DeferredSequenceGather:
    """Synchronous fallback with the same interface as an async gather."""

    tensor: torch.Tensor
    sequence_dim: int
    gather_sequence: Callable[[torch.Tensor, int], torch.Tensor]

    def wait(self) -> torch.Tensor:
        """Run the fallback gather at the first consumer boundary."""
        return self.gather_sequence(self.tensor, self.sequence_dim)


@dataclass
class SharedCompressedAttentionState:
    """Cross-layer tensors published by one compressed-attention source."""

    compressed_kv: torch.Tensor | None = None
    index_key: torch.Tensor | None = None
    topk_indices: torch.Tensor | None = None
    candidate_blocks: torch.Tensor | None = None


@dataclass
class SharedCompressedIndexerOutput:
    """Tensors produced by one Full or Reindex Indexer invocation."""

    topk_indices: torch.Tensor
    index_key: torch.Tensor
    candidate_blocks: torch.Tensor | None
    index_query: torch.Tensor
    merge_weight: torch.Tensor


@dataclass(frozen=True)
class SharedCompressedAttentionCPContext:
    """KV-all-gather context for shared compressed DSA.

    Query-side tensors remain local sequence shards. Raw KV, compressed KV,
    and index keys are gathered in CP-rank order. ``launch_sequence`` is
    optional so numerical tests and non-overlapped backends can supply only a
    differentiable synchronous gather.
    """

    size: int
    rank: int
    gather_sequence: Callable[[torch.Tensor, int], torch.Tensor]
    launch_sequence: Callable[[torch.Tensor, int], SequenceGatherHandle] | None = None

    def launch(self, tensor: torch.Tensor, sequence_dim: int) -> SequenceGatherHandle:
        """Launch an async gather or create a deferred synchronous gather."""
        if self.launch_sequence is not None:
            return self.launch_sequence(tensor, sequence_dim)
        return _DeferredSequenceGather(tensor, sequence_dim, self.gather_sequence)


@dataclass(frozen=True)
class SharedCompressedAttentionTPContext:
    """Tensor-parallel score reduction context for a head-sharded Indexer."""

    size: int
    rank: int
    reduce_sum: Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class SharedCompressedPackedSequence:
    """Compact sample boundaries for one contiguous CP query shard."""

    cu_seq_lens: torch.Tensor
    local_query_start: int
    local_query_length: int
    global_sequence_length: int

    def local_segment_starts(self, device: torch.device) -> torch.Tensor:
        """Return each local query's global packed-sample start position."""
        boundaries = self.cu_seq_lens.to(device=device, dtype=torch.long)
        if boundaries.ndim != 1 or boundaries.numel() < 2 or int(boundaries[0]) != 0:
            raise ValueError("packed cu_seq_lens must be one-dimensional and start with zero")
        if int(boundaries[-1]) != self.global_sequence_length:
            raise ValueError(
                "packed cu_seq_lens must cover global_sequence_length: "
                f"{int(boundaries[-1])} versus {self.global_sequence_length}"
            )
        if torch.any(boundaries[1:] <= boundaries[:-1]):
            raise ValueError("packed cu_seq_lens must be strictly increasing")
        query_positions = torch.arange(
            self.local_query_start,
            self.local_query_start + self.local_query_length,
            device=device,
        )
        segment_ids = torch.bucketize(query_positions, boundaries[1:], right=True)
        return boundaries[:-1].index_select(0, segment_ids).unsqueeze(0)

    def validate_compression_alignment(self, compress_ratio: int) -> None:
        """Reject packed samples whose compressor groups would cross boundaries."""
        if compress_ratio <= 1:
            return
        boundaries = self.cu_seq_lens.to(dtype=torch.long)
        misaligned = boundaries[boundaries.remainder(compress_ratio) != 0]
        if misaligned.numel():
            raise ValueError(
                "packed sample boundaries must align with the CSA2 compression ratio "
                f"{compress_ratio}; misaligned boundaries={misaligned.tolist()}"
            )


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent real/imaginary pairs for V4 interleaved RoPE."""
    first = tensor[..., 0::2]
    second = tensor[..., 1::2]
    return torch.stack((-second, first), dim=-1).flatten(-2)


def _apply_v41_rope(
        tensor: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Apply V4 interleaved RoPE to the trailing rotary channels."""
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = cos.shape[-1]
    pass_through, rotary = tensor[..., :-rope_dim], tensor[..., -rope_dim:]
    rotary = ((rotary.float() * cos) + (_rotate_half(rotary).float() * sin)).to(tensor.dtype)
    return torch.cat((pass_through, rotary), dim=-1)


def build_sliding_window_indices(
        batch_size: int,
        sequence_length: int,
        window_size: int,
        device: torch.device,
        *,
        query_offset: int = 0,
        key_length: int | None = None,
) -> torch.Tensor:
    """Build fixed-width causal sliding-window indices."""
    if key_length is None:
        key_length = sequence_length
    width = min(key_length, window_size)
    query = torch.arange(
        query_offset,
        query_offset + sequence_length,
        device=device,
    ).unsqueeze(-1)
    offsets = torch.arange(width - 1, -1, -1, device=device)
    indices = query - offsets
    indices = indices.masked_fill(indices < 0, -1)
    return indices.unsqueeze(0).expand(batch_size, -1, -1)


def compressed_causal_topk(
        query: torch.Tensor,
        key: torch.Tensor,
        merge_weight: torch.Tensor,
        *,
        compress_ratio: int,
        sparse_count: int,
        query_offset: int = 0,
        query_chunk_size: int = 256,
        reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None = None,
        minimum_key_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select compressed positions with V4.1's ratio-aware causal rule.

    The Lightning Indexer operator used by ordinary DSA assumes query and key
    positions share one token coordinate. CSA2 keys instead represent closed
    groups of ``compress_ratio`` source tokens. This implementation retains
    the source rule ``key < floor((query + 1) / ratio)`` while using batched
    matrix multiplication and bounded query chunks on accelerator cores.
    """
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    batch_size, sequence_length, num_heads, head_dim = query.shape
    if key.ndim != 3 or key.shape[0] != batch_size or key.shape[2] != head_dim:
        raise ValueError(
            "compressed index key must have shape [batch, compressed_sequence, head_dim]"
        )
    if merge_weight.shape != query.shape[:3]:
        raise ValueError(
            f"merge_weight must have shape {tuple(query.shape[:3])}, got {tuple(merge_weight.shape)}"
        )
    compressed_length = key.shape[1]
    if minimum_key_indices is not None and minimum_key_indices.shape != query.shape[:2]:
        raise ValueError("minimum_key_indices must have shape [batch, query]")
    top_k = min(sparse_count, compressed_length)
    if top_k <= 0:
        return torch.empty(
            batch_size,
            sequence_length,
            0,
            dtype=torch.int32,
            device=query.device,
        )

    key_fp32 = key.float().transpose(1, 2).unsqueeze(1)
    key_positions = torch.arange(compressed_length, device=query.device).view(1, 1, -1)
    selected_chunks = []
    for start in range(0, sequence_length, query_chunk_size):
        end = min(start + query_chunk_size, sequence_length)
        scores = torch.matmul(query[:, start:end].float(), key_fp32).relu_()
        scores = (scores * merge_weight[:, start:end].float().unsqueeze(-1)).sum(dim=2)
        if reduce_sum is not None:
            scores = reduce_sum(scores)
        visible = (
            torch.arange(query_offset + start, query_offset + end, device=query.device) + 1
        ) // compress_ratio
        scores.masked_fill_(key_positions >= visible.view(1, -1, 1), float("-inf"))
        if minimum_key_indices is not None:
            minimum = minimum_key_indices[:, start:end].unsqueeze(-1)
            scores.masked_fill_(key_positions < minimum, float("-inf"))
        top = scores.topk(top_k, dim=-1, sorted=False)
        indices = top.indices.masked_fill(~torch.isfinite(top.values), compressed_length)
        indices = indices.sort(dim=-1).values
        indices = indices.masked_fill(indices == compressed_length, -1)
        selected_chunks.append(indices.to(torch.int32))
    return torch.cat(selected_chunks, dim=1)


def select_candidate_blocks(
        logits: torch.Tensor,
        compress_lens: torch.Tensor | int,
        topk_blocks: int,
        block_size: int,
) -> torch.Tensor:
    """Select the official block-max hierarchical candidate pool.

    The newest reachable partial block is pinned into the candidate set, then
    the remaining blocks compete by their maximum position score. ``logits``
    must already contain ``-inf`` at causally unreachable positions.
    """
    if topk_blocks <= 0:
        raise ValueError(f"topk_blocks must be positive, got {topk_blocks}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    width = logits.shape[-1]
    if width == 0:
        return torch.zeros_like(logits, dtype=torch.bool)
    padding = -width % block_size
    scores = torch.nn.functional.pad(logits, (0, padding), value=float("-inf"))
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.shape[-1]
    if isinstance(compress_lens, torch.Tensor):
        last = (compress_lens.to(device=logits.device) - 1) // block_size
    else:
        last = (compress_lens - 1) // block_size
    block_positions = torch.arange(num_blocks, device=logits.device)
    scores = scores.masked_fill(block_positions == last, float("inf"))
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep.scatter_(-1, top.indices, top.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def select_candidate_block_indices(
        logits: torch.Tensor,
        compress_lens: torch.Tensor | int,
        topk_blocks: int,
        block_size: int,
) -> torch.Tensor:
    """Return compact block ids for the official hierarchical candidates.

    Storing block ids uses ``block_size`` times less cross-layer state than
    expanded position ids and avoids the full ``[batch, query, key]`` boolean
    mask used by the released inference reference. Invalid slots are ``-1``.
    """
    if topk_blocks <= 0:
        raise ValueError(f"topk_blocks must be positive, got {topk_blocks}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    width = logits.shape[-1]
    if width == 0:
        return torch.empty(*logits.shape[:-1], 0, dtype=torch.int32, device=logits.device)
    padding = -width % block_size
    scores = torch.nn.functional.pad(logits, (0, padding), value=float("-inf"))
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.shape[-1]
    if isinstance(compress_lens, torch.Tensor):
        last = (compress_lens.to(device=logits.device) - 1) // block_size
    else:
        last = (compress_lens - 1) // block_size
    block_positions = torch.arange(num_blocks, device=logits.device)
    scores = scores.masked_fill(block_positions == last, float("inf"))
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1, sorted=False)
    return top.indices.masked_fill(top.values == float("-inf"), -1).to(torch.int32)


def compressed_causal_candidates(
        query: torch.Tensor,
        key: torch.Tensor,
        merge_weight: torch.Tensor,
        *,
        compress_ratio: int,
        topk_blocks: int,
        block_size: int,
        query_offset: int = 0,
        query_chunk_size: int = 256,
        reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None = None,
        minimum_key_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build compact blockwise candidate ids in bounded query chunks."""
    _, sequence_length, _, _ = query.shape
    compressed_length = key.shape[1]
    if minimum_key_indices is not None and minimum_key_indices.shape != query.shape[:2]:
        raise ValueError("minimum_key_indices must have shape [batch, query]")
    key_fp32 = key.float().transpose(1, 2).unsqueeze(1)
    key_positions = torch.arange(compressed_length, device=query.device).view(1, 1, -1)
    candidate_chunks = []
    for start in range(0, sequence_length, query_chunk_size):
        end = min(start + query_chunk_size, sequence_length)
        scores = torch.matmul(query[:, start:end].float(), key_fp32).relu_()
        scores = (scores * merge_weight[:, start:end].float().unsqueeze(-1)).sum(dim=2)
        if reduce_sum is not None:
            scores = reduce_sum(scores)
        visible = (
            torch.arange(query_offset + start, query_offset + end, device=query.device) + 1
        ) // compress_ratio
        scores.masked_fill_(key_positions >= visible.view(1, -1, 1), float("-inf"))
        if minimum_key_indices is not None:
            minimum = minimum_key_indices[:, start:end].unsqueeze(-1)
            scores.masked_fill_(key_positions < minimum, float("-inf"))
        candidate_chunks.append(
            select_candidate_block_indices(
                scores,
                visible.view(1, -1, 1),
                topk_blocks,
                block_size,
            )
        )
    return torch.cat(candidate_chunks, dim=1)


def compressed_causal_topk_and_candidates(
        query: torch.Tensor,
        key: torch.Tensor,
        merge_weight: torch.Tensor,
        *,
        compress_ratio: int,
        sparse_count: int,
        topk_blocks: int,
        block_size: int,
        query_offset: int = 0,
        query_chunk_size: int = 256,
        reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None = None,
        minimum_key_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select Full-layer Top-K and candidate blocks from one score pass."""
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    batch_size, sequence_length, _, head_dim = query.shape
    if key.ndim != 3 or key.shape[0] != batch_size or key.shape[2] != head_dim:
        raise ValueError(
            "compressed index key must have shape [batch, compressed_sequence, head_dim]"
        )
    if merge_weight.shape != query.shape[:3]:
        raise ValueError(
            f"merge_weight must have shape {tuple(query.shape[:3])}, got {tuple(merge_weight.shape)}"
        )
    compressed_length = key.shape[1]
    if minimum_key_indices is not None and minimum_key_indices.shape != query.shape[:2]:
        raise ValueError("minimum_key_indices must have shape [batch, query]")
    top_k = min(sparse_count, compressed_length)
    if top_k <= 0:
        empty_topk = torch.empty(
            batch_size,
            sequence_length,
            0,
            dtype=torch.int32,
            device=query.device,
        )
        return empty_topk, empty_topk

    key_fp32 = key.float().transpose(1, 2).unsqueeze(1)
    key_positions = torch.arange(compressed_length, device=query.device).view(1, 1, -1)
    selected_chunks = []
    candidate_chunks = []
    for start in range(0, sequence_length, query_chunk_size):
        end = min(start + query_chunk_size, sequence_length)
        scores = torch.matmul(query[:, start:end].float(), key_fp32).relu_()
        scores = (scores * merge_weight[:, start:end].float().unsqueeze(-1)).sum(dim=2)
        if reduce_sum is not None:
            scores = reduce_sum(scores)
        visible = (
            torch.arange(query_offset + start, query_offset + end, device=query.device) + 1
        ) // compress_ratio
        scores.masked_fill_(key_positions >= visible.view(1, -1, 1), float("-inf"))
        if minimum_key_indices is not None:
            minimum = minimum_key_indices[:, start:end].unsqueeze(-1)
            scores.masked_fill_(key_positions < minimum, float("-inf"))
        top = scores.topk(top_k, dim=-1, sorted=False)
        indices = top.indices.masked_fill(~torch.isfinite(top.values), compressed_length)
        indices = indices.sort(dim=-1).values
        selected_chunks.append(indices.masked_fill(indices == compressed_length, -1).to(torch.int32))
        candidate_chunks.append(
            select_candidate_block_indices(
                scores,
                visible.view(1, -1, 1),
                topk_blocks,
                block_size,
            )
        )
    return torch.cat(selected_chunks, dim=1), torch.cat(candidate_chunks, dim=1)


def compressed_candidate_topk(
        query: torch.Tensor,
        key: torch.Tensor,
        merge_weight: torch.Tensor,
        candidate_blocks: torch.Tensor,
        *,
        compress_ratio: int,
        sparse_count: int,
        block_size: int,
        query_offset: int = 0,
        query_chunk_size: int = 256,
        reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None = None,
        minimum_key_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score only hierarchical candidate blocks and return global key ids."""
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    batch_size, sequence_length, _, head_dim = query.shape
    if key.ndim != 3 or key.shape[0] != batch_size or key.shape[2] != head_dim:
        raise ValueError(
            "compressed index key must have shape [batch, compressed_sequence, head_dim]"
        )
    if merge_weight.shape != query.shape[:3]:
        raise ValueError(
            f"merge_weight must have shape {tuple(query.shape[:3])}, got {tuple(merge_weight.shape)}"
        )
    if candidate_blocks.ndim != 3 or candidate_blocks.shape[:2] != query.shape[:2]:
        raise ValueError(
            "candidate_blocks must have shape [batch, query, candidate_blocks], "
            f"got {tuple(candidate_blocks.shape)}"
        )
    if minimum_key_indices is not None and minimum_key_indices.shape != query.shape[:2]:
        raise ValueError("minimum_key_indices must have shape [batch, query]")

    compressed_length = key.shape[1]
    candidate_width = candidate_blocks.shape[-1] * block_size
    top_k = min(sparse_count, candidate_width)
    if top_k <= 0:
        return torch.empty(
            batch_size,
            sequence_length,
            0,
            dtype=torch.int32,
            device=query.device,
        )

    block_offsets = torch.arange(block_size, device=query.device)
    batch_indices = torch.arange(batch_size, device=query.device).view(-1, 1, 1)
    selected_chunks = []
    for start in range(0, sequence_length, query_chunk_size):
        end = min(start + query_chunk_size, sequence_length)
        blocks = candidate_blocks[:, start:end].long()
        block_positions = blocks.unsqueeze(-1) * block_size + block_offsets
        positions = block_positions.flatten(-2)
        visible = (
            torch.arange(query_offset + start, query_offset + end, device=query.device) + 1
        ) // compress_ratio
        valid = (blocks.unsqueeze(-1) >= 0).expand_as(block_positions).flatten(-2)
        valid = valid & (positions < compressed_length) & (positions < visible.view(1, -1, 1))
        if minimum_key_indices is not None:
            valid = valid & (positions >= minimum_key_indices[:, start:end].unsqueeze(-1))
        safe_positions = positions.clamp(min=0, max=max(compressed_length - 1, 0))
        selected_key = key[batch_indices, safe_positions].float()
        dots = torch.einsum(
            "bchd,bckd->bchk",
            query[:, start:end].float(),
            selected_key,
        ).relu_()
        scores = (dots * merge_weight[:, start:end].float().unsqueeze(-1)).sum(dim=2)
        if reduce_sum is not None:
            scores = reduce_sum(scores)
        scores.masked_fill_(~valid, float("-inf"))
        top = scores.topk(top_k, dim=-1, sorted=False)
        selected = positions.gather(-1, top.indices)
        selected.masked_fill_(~torch.isfinite(top.values), compressed_length)
        selected = selected.sort(dim=-1).values
        selected.masked_fill_(selected == compressed_length, -1)
        selected_chunks.append(selected.to(torch.int32))
    return torch.cat(selected_chunks, dim=1)


class _SharedCompressedIndexerKLLoss(torch.autograd.Function):
    """PanGu-style precomputed-gradient KL loss for compressed index keys."""

    @staticmethod
    def forward(
            ctx: Any,
            index_query: torch.Tensor,
            index_key: torch.Tensor,
            merge_weight: torch.Tensor,
            attention_query: torch.Tensor,
            compressed_key: torch.Tensor,
            topk_indices: torch.Tensor,
            sinks: torch.Tensor,
            attention_scale: float,
            loss_coeff: float,
            query_chunk_size: int,
            reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> torch.Tensor:
        """Compute sparse KL and save only the three Indexer gradients."""
        batch_size, sequence_length, _, _ = index_query.shape
        denominator = batch_size * sequence_length
        total_loss = index_query.new_zeros((), dtype=torch.float32)
        grad_index_query = torch.zeros_like(index_query, dtype=torch.float32)
        grad_index_key = torch.zeros_like(index_key, dtype=torch.float32)
        grad_merge_weight = torch.zeros_like(merge_weight, dtype=torch.float32)

        with torch.no_grad():
            for start in range(0, sequence_length, query_chunk_size):
                end = min(start + query_chunk_size, sequence_length)
                selected = topk_indices[:, start:end]
                valid = selected >= 0
                valid_rows = valid.any(dim=-1)
                if not torch.any(valid_rows):
                    continue
                safe_indices = selected.clamp_min(0).long()
                batch_indices = torch.arange(batch_size, device=index_query.device).view(-1, 1, 1)

                query_chunk = index_query[:, start:end].float()
                weight_chunk = merge_weight[:, start:end].float()
                selected_index_key = index_key[batch_indices, safe_indices].float()
                index_dots = torch.einsum(
                    "bcid,bckd->bcik", query_chunk, selected_index_key
                )
                index_relu = index_dots.relu()
                index_scores = (index_relu * weight_chunk.unsqueeze(-1)).sum(dim=2)
                if reduce_sum is not None:
                    index_scores = reduce_sum(index_scores)
                index_scores.masked_fill_(~valid, -1.0e9)

                selected_attention_key = compressed_key[batch_indices, safe_indices].float()
                attention_scores = torch.einsum(
                    "bhcd,bckd->bhck",
                    attention_query[:, :, start:end].float(),
                    selected_attention_key,
                ) * attention_scale
                attention_scores.masked_fill_(~valid.unsqueeze(1), -1.0e9)
                sink_logits = sinks.float().view(1, -1, 1, 1).expand(
                    batch_size, -1, end - start, -1
                )
                target = torch.cat((attention_scores, sink_logits), dim=-1).softmax(dim=-1)
                target = target[..., :-1].masked_fill(~valid.unsqueeze(1), 0.0).sum(dim=1)
                if reduce_sum is not None:
                    target = reduce_sum(target)
                target = target / target.sum(dim=-1, keepdim=True).clamp_min(
                    torch.finfo(torch.float32).tiny
                )

                log_prediction = index_scores.log_softmax(dim=-1)
                target_log = target.clamp_min(torch.finfo(torch.float32).tiny).log()
                row_loss = (target * (target_log - log_prediction)).sum(dim=-1)
                total_loss.add_(row_loss[valid_rows].sum() * (loss_coeff / denominator))

                grad_scores = (log_prediction.exp() - target) * (loss_coeff / denominator)
                grad_scores.masked_fill_(~valid, 0.0)
                grad_dots = (
                    grad_scores.unsqueeze(2)
                    * weight_chunk.unsqueeze(-1)
                    * (index_dots > 0).to(index_dots.dtype)
                )
                grad_index_query[:, start:end] = torch.einsum(
                    "bcik,bckd->bcid", grad_dots, selected_index_key
                )
                grad_merge_weight[:, start:end] = (
                    grad_scores.unsqueeze(2) * index_relu
                ).sum(dim=-1)
                selected_key_grad = torch.einsum(
                    "bcik,bcid->bckd", grad_dots, query_chunk
                )
                selected_key_grad.masked_fill_(~valid.unsqueeze(-1), 0.0)
                scatter_indices = safe_indices.unsqueeze(-1).expand_as(selected_key_grad)
                grad_index_key.scatter_add_(
                    1,
                    scatter_indices.reshape(batch_size, -1, index_key.shape[-1]),
                    selected_key_grad.reshape(batch_size, -1, index_key.shape[-1]),
                )

        ctx.save_for_backward(grad_index_query, grad_index_key, grad_merge_weight)
        return total_loss

    @staticmethod
    def backward(ctx: Any, grad_loss: torch.Tensor) -> tuple:
        """Return the precomputed Indexer gradients and detach the teacher."""
        grad_index_query, grad_index_key, grad_merge_weight = ctx.saved_tensors
        return (
            grad_index_query * grad_loss,
            grad_index_key * grad_loss,
            grad_merge_weight * grad_loss,
            *((None,) * 8),
        )


def shared_compressed_indexer_kl_loss(
        index_query: torch.Tensor,
        index_key: torch.Tensor,
        merge_weight: torch.Tensor,
        attention_query: torch.Tensor,
        compressed_key: torch.Tensor,
        topk_indices: torch.Tensor,
        sinks: torch.Tensor,
        *,
        attention_scale: float,
        loss_coeff: float,
        query_chunk_size: int = 256,
        tp_context: SharedCompressedAttentionTPContext | None = None,
) -> torch.Tensor:
    """Compute the sparse-stage DSA KL objective on V4.1 compressed keys.

    The target first softmaxes selected main-attention scores per attention
    head, then averages and L1-normalizes across heads. Indexer inputs are
    detached by the caller, so this objective updates only Indexer parameters.
    """
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    if not loss_coeff:
        return index_query.sum(dtype=torch.float32) * 0.0
    if topk_indices.shape[:2] != index_query.shape[:2]:
        raise ValueError("topk_indices and index_query must share batch/query dimensions")
    return _SharedCompressedIndexerKLLoss.apply(
        index_query,
        index_key,
        merge_weight,
        attention_query.detach(),
        compressed_key.detach(),
        topk_indices,
        sinks.detach(),
        attention_scale,
        loss_coeff,
        query_chunk_size,
        None if tp_context is None else tp_context.reduce_sum,
    )


class SharedCompressedDSAIndexer(nn.Module):
    """Accelerator-oriented V4.1 compressed Lightning Indexer replacement."""

    def __init__(self, module: nn.Module) -> None:
        """Transfer the source projections without changing checkpoint names."""
        super().__init__()
        required = ("q_b_proj", "weights_proj")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(f"shared compressed DSA indexer is missing: {missing}")
        for name, child in module._modules.items():  # pylint: disable=protected-access
            self.add_module(name, child)
        for name, parameter in module._parameters.items():  # pylint: disable=protected-access
            self.register_parameter(name, parameter)
        self.compress_ratio = module.compress_ratio
        self.global_num_heads = module.num_heads
        self.head_dim = module.head_dim
        self.index_topk = module.index_topk
        self.owns_key = module.owns_key
        self.is_candidate_source = module.is_candidate_source
        self.uses_candidates = module.uses_candidates
        self.candidate_topk_blocks = module.candidate_topk_blocks
        self.candidate_block_size = module.candidate_block_size
        self.loss_coeff = module.loss_coeff
        self.query_chunk_size = int(getattr(module, "query_chunk_size", 256))
        self.train(module.training)

    def forward(
            self,
            hidden_states: torch.Tensor,
            query_residual: torch.Tensor,
            latent: torch.Tensor | None,
            compress_position_embeddings: tuple[torch.Tensor, torch.Tensor],
            *,
            index_key: torch.Tensor | None = None,
            candidate_blocks: torch.Tensor | None = None,
            cp_context: SharedCompressedAttentionCPContext | None = None,
            tp_context: SharedCompressedAttentionTPContext | None = None,
            query_offset: int = 0,
            minimum_key_indices: torch.Tensor | None = None,
    ) -> SharedCompressedIndexerOutput:
        """Project local queries and score them against global compressed keys."""
        batch_size, sequence_length, _ = hidden_states.shape
        cos, sin = compress_position_embeddings

        key_handle = None
        if self.owns_key:
            if latent is None:
                raise RuntimeError("a Full Indexer requires the compressor latent")
            key = self.k_norm(self.wk(latent.detach()))
            compressed_length = key.shape[1]
            key_cos = cos[:, :compressed_length * self.compress_ratio:self.compress_ratio]
            key_sin = sin[:, :compressed_length * self.compress_ratio:self.compress_ratio]
            key = _apply_v41_rope(key.unsqueeze(1), key_cos, key_sin).squeeze(1)
            key_handle = cp_context.launch(key, 1) if cp_context is not None else None
        else:
            if index_key is None:
                raise RuntimeError("a Reindex layer requires index K from the preceding Full layer")
            key = index_key

        query = self.q_b_proj(query_residual.detach())
        if query.shape[-1] % self.head_dim:
            raise ValueError(
                f"Indexer query width {query.shape[-1]} is not divisible by head_dim {self.head_dim}"
            )
        local_num_heads = query.shape[-1] // self.head_dim
        query = query.view(batch_size, sequence_length, local_num_heads, self.head_dim)
        query = _apply_v41_rope(query.transpose(1, 2), cos, sin).transpose(1, 2)
        merge_weight = self.weights_proj(hidden_states.detach())
        if merge_weight.shape[-1] != local_num_heads:
            raise ValueError(
                "Indexer q_b_proj and weights_proj must use the same TP head shard: "
                f"{local_num_heads} query heads versus {merge_weight.shape[-1]} weights"
            )
        merge_weight = merge_weight * (self.head_dim**-0.5 * self.global_num_heads**-0.5)
        reduce_sum = None if tp_context is None else tp_context.reduce_sum
        if key_handle is not None:
            key = key_handle.wait()
        if self.is_candidate_source:
            topk_indices, candidate_blocks = compressed_causal_topk_and_candidates(
                query,
                key,
                merge_weight,
                compress_ratio=self.compress_ratio,
                sparse_count=self.index_topk,
                topk_blocks=self.candidate_topk_blocks,
                block_size=self.candidate_block_size,
                query_offset=query_offset,
                query_chunk_size=self.query_chunk_size,
                reduce_sum=reduce_sum,
                minimum_key_indices=minimum_key_indices,
            )
        elif self.uses_candidates:
            if candidate_blocks is None:
                raise RuntimeError("a hierarchical Reindex layer requires candidate block ids")
            topk_indices = compressed_candidate_topk(
                query,
                key,
                merge_weight,
                candidate_blocks,
                compress_ratio=self.compress_ratio,
                sparse_count=self.index_topk,
                block_size=self.candidate_block_size,
                query_offset=query_offset,
                query_chunk_size=self.query_chunk_size,
                reduce_sum=reduce_sum,
                minimum_key_indices=minimum_key_indices,
            )
        else:
            candidate_blocks = None
            topk_indices = compressed_causal_topk(
                query,
                key,
                merge_weight,
                compress_ratio=self.compress_ratio,
                sparse_count=self.index_topk,
                query_offset=query_offset,
                query_chunk_size=self.query_chunk_size,
                reduce_sum=reduce_sum,
                minimum_key_indices=minimum_key_indices,
            )
        return SharedCompressedIndexerOutput(
            topk_indices=topk_indices,
            index_key=key,
            candidate_blocks=candidate_blocks,
            index_query=query,
            merge_weight=merge_weight,
        )


def _reference_sparse_attention(
        query: torch.Tensor,
        key_value: torch.Tensor,
        sparse_indices: torch.Tensor,
        sinks: torch.Tensor,
        scale: float,
) -> torch.Tensor:
    """Compute indexed attention through a dense mask for numeric validation."""
    batch_size, num_heads, sequence_length, _ = query.shape
    key_length = key_value.shape[2]
    valid = sparse_indices >= 0
    safe_indices = sparse_indices.clamp_min(0).long()
    allowed_counts = torch.zeros(
        batch_size,
        sequence_length,
        key_length,
        dtype=torch.int32,
        device=query.device,
    )
    allowed_counts.scatter_add_(2, safe_indices, valid.to(torch.int32))
    allowed = allowed_counts > 0
    logits = torch.matmul(query, key_value.transpose(-1, -2)) * scale
    logits = logits.masked_fill(~allowed.unsqueeze(1), float("-inf"))
    sink_logits = sinks.view(1, num_heads, 1, 1).expand(batch_size, -1, sequence_length, -1)
    probabilities = torch.cat((logits, sink_logits), dim=-1).softmax(dim=-1, dtype=torch.float32)
    return torch.matmul(probabilities[..., :-1].to(key_value.dtype), key_value).transpose(1, 2)


class _NpuSparseAttentionWithScalarSink(torch.autograd.Function):
    """Autograd bridge adding V4.1's scalar sink to sparse attention."""

    @staticmethod
    def forward(
            ctx: Any,
            query: torch.Tensor,
            key_value: torch.Tensor,
            sparse_indices: torch.Tensor,
            sinks: torch.Tensor,
            rope_head_dim: int,
            scale: float,
    ) -> torch.Tensor:
        """Run sparse attention and merge the analytically zero-valued sink."""
        import omni_training_custom_ops  # noqa: F401  # pylint: disable=C0415,unused-import

        batch_size, num_heads, sequence_length, head_dim = query.shape
        key_length = key_value.shape[2]
        query_rope = query.new_zeros(batch_size, num_heads, sequence_length, rope_head_dim)
        key_rope = key_value.new_zeros(batch_size, 1, key_length + 1, rope_head_dim)
        dummy_key_value = key_value.new_zeros(batch_size, 1, 1, head_dim)
        key_value_with_dummy = torch.cat((key_value, dummy_key_value), dim=2)
        valid_indices = sparse_indices >= 0
        safe_indices = sparse_indices.masked_fill(~valid_indices, key_length)
        query_lengths = torch.arange(
            sequence_length,
            (batch_size + 1) * sequence_length,
            sequence_length,
            dtype=torch.int32,
            device=query.device,
        )
        key_lengths = torch.arange(
            key_length + 1,
            (batch_size + 1) * (key_length + 1),
            key_length + 1,
            dtype=torch.int32,
            device=query.device,
        )
        sparse_indices_tnd = safe_indices.reshape(-1, 1, safe_indices.shape[-1]).to(torch.int32)
        output, softmax_max, softmax_sum = torch.ops.custom.npu_sparse_flash_attention_enhance(
            query.transpose(1, 2).reshape(-1, num_heads, head_dim),
            key_value_with_dummy.transpose(1, 2).reshape(-1, 1, head_dim),
            key_value_with_dummy.transpose(1, 2).reshape(-1, 1, head_dim),
            sparse_indices_tnd,
            scale,
            block_table=None,
            actual_seq_lengths_query=query_lengths,
            actual_seq_lengths_kv=key_lengths,
            query_rope=query_rope.transpose(1, 2).reshape(-1, num_heads, rope_head_dim),
            key_rope=key_rope.transpose(1, 2).reshape(-1, 1, rope_head_dim),
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="TND",
            sparse_mode=0,
            attention_mode=2,
            return_softmax_lse=True,
        )
        output = output.view(batch_size, sequence_length, num_heads, head_dim)
        sparse_max = softmax_max.squeeze(0).view(batch_size, sequence_length, num_heads)
        sparse_sum = softmax_sum.squeeze(0).view(batch_size, sequence_length, num_heads)
        sink_logits = sinks.float().view(1, 1, num_heads)
        combined_max = torch.maximum(sparse_max, sink_logits)
        dummy_count = (~valid_indices).sum(dim=-1, dtype=torch.float32).unsqueeze(-1)
        dummy_mass = dummy_count * torch.exp(-sparse_max)
        actual_sparse_sum = (sparse_sum - dummy_mass).clamp_min(torch.finfo(torch.float32).tiny)
        sparse_mass = actual_sparse_sum * torch.exp(sparse_max - combined_max)
        sink_mass = torch.exp(sink_logits - combined_max)
        desired_sum = sparse_mass + sink_mass
        kernel_mass = sparse_sum * torch.exp(sparse_max - combined_max)
        output_scale = kernel_mass / desired_sum
        sink_probability = sink_mass / desired_sum
        rescaled_output = output * output_scale.unsqueeze(-1).to(output.dtype)
        ctx.save_for_backward(
            query,
            key_value,
            sparse_indices_tnd,
            softmax_max,
            softmax_sum,
            rescaled_output,
            output_scale,
            sink_probability,
        )
        ctx.params = (rope_head_dim, scale, query_lengths, key_lengths)
        return rescaled_output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Apply sparse-attention backward and the analytic sink gradient."""
        (
            query,
            key_value,
            sparse_indices_tnd,
            softmax_max,
            softmax_sum,
            rescaled_output,
            output_scale,
            sink_probability,
        ) = ctx.saved_tensors
        rope_head_dim, scale, query_lengths, key_lengths = ctx.params
        batch_size, num_heads, sequence_length, head_dim = query.shape
        key_length = key_value.shape[2]
        query_tnd = query.transpose(1, 2).reshape(-1, num_heads, head_dim)
        query_rope = query.new_zeros(batch_size, num_heads, sequence_length, rope_head_dim)
        dummy_key_value = key_value.new_zeros(batch_size, 1, 1, head_dim)
        key_value_with_dummy = torch.cat((key_value, dummy_key_value), dim=2)
        key_tnd = key_value_with_dummy.transpose(1, 2).reshape(-1, 1, head_dim)
        key_rope = key_value.new_zeros(batch_size, 1, key_length + 1, rope_head_dim)
        scaled_grad = grad_output * output_scale.unsqueeze(-1).to(grad_output.dtype)
        grad_query, grad_key, grad_value, _, _ = torch.ops.custom.npu_sparse_flash_attention_grad_enhance(
            query_tnd,
            key_tnd,
            key_tnd,
            sparse_indices_tnd,
            scaled_grad.reshape(-1, num_heads, head_dim).to(key_value.dtype),
            rescaled_output.reshape(-1, num_heads, head_dim),
            softmax_max,
            softmax_sum,
            scale,
            sparse_block_size=1,
            actual_seq_qlen=query_lengths,
            actual_seq_kvlen=key_lengths,
            query_rope=query_rope.transpose(1, 2).reshape(-1, num_heads, rope_head_dim),
            key_rope=key_rope.transpose(1, 2).reshape(-1, 1, rope_head_dim),
            layout="TND",
            sparse_mode=0,
            attention_mode=2,
            deterministic=torch.are_deterministic_algorithms_enabled(),
        )
        grad_query = grad_query.view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)
        grad_key = (grad_key + grad_value).view(batch_size, key_length + 1, 1, head_dim)
        grad_key = grad_key[:, :key_length].transpose(1, 2)
        sink_grad = -(sink_probability * (grad_output.float() * rescaled_output.float()).sum(-1)).sum((0, 1))
        return grad_query, grad_key, None, sink_grad, None, None


def npu_sparse_attention_with_scalar_sink(
        query: torch.Tensor,
        key_value: torch.Tensor,
        sparse_indices: torch.Tensor,
        sinks: torch.Tensor,
        rope_head_dim: int,
        scale: float,
) -> torch.Tensor:
    """Run V4.1 sparse attention through enhanced Ascend operators."""
    return _NpuSparseAttentionWithScalarSink.apply(
        query,
        key_value,
        sparse_indices,
        sinks,
        rope_head_dim,
        scale,
    )


class SharedCompressedDSAAttention(nn.Module):
    """High-performance V4.1 shared compressed attention replacement."""

    def __init__(self, module: nn.Module) -> None:
        """Transfer source state and install the compressed indexer module."""
        super().__init__()
        for name, child in module._modules.items():  # pylint: disable=protected-access
            self.add_module(name, child)
        for name, parameter in module._parameters.items():  # pylint: disable=protected-access
            self.register_parameter(name, parameter)
        if hasattr(self, "indexer"):
            self.indexer = SharedCompressedDSAIndexer(self.indexer)
        self.config = module.config
        self.layer_idx = module.layer_idx
        self.num_heads = module.num_heads
        self.num_key_value_groups = module.num_key_value_groups
        self.num_groups = module.config.o_groups
        self.compress_ratio = module.compress_ratio
        self.is_kv_source = module.is_kv_source
        self.is_index_source = module.is_index_source
        self.head_dim = module.head_dim
        self.rope_head_dim = module.config.qk_rope_head_dim
        self.sliding_window = module.sliding_window
        self.scaling = module.scaling
        self.train(module.training)

    @staticmethod
    def _validate_cp_inputs(
            cp_context: SharedCompressedAttentionCPContext,
            position_ids: torch.Tensor,
            sequence_length: int,
            compress_ratio: int,
            is_kv_source: bool,
    ) -> tuple[int, int]:
        """Validate contiguous CP shards and return global query geometry."""
        if cp_context.size <= 1 or not 0 <= cp_context.rank < cp_context.size:
            raise ValueError(
                "shared compressed attention CP requires size > 1 and "
                f"0 <= rank < size, got rank={cp_context.rank}, size={cp_context.size}"
            )
        if is_kv_source and sequence_length % compress_ratio:
            raise ValueError(
                "compressed-KV CP requires every local sequence shard to divide "
                f"compress_ratio={compress_ratio}, got {sequence_length}"
            )
        query_offset = cp_context.rank * sequence_length
        if not torch.all(position_ids[..., 0] == query_offset):
            raise ValueError(
                "shared compressed attention CP requires contiguous global position_ids; "
                f"rank {cp_context.rank} expected first position {query_offset}"
            )
        return query_offset, sequence_length * cp_context.size

    def _project_raw_kv(
            self,
            hidden_states: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            cp_context: SharedCompressedAttentionCPContext | None,
    ) -> tuple[torch.Tensor | None, SequenceGatherHandle | None]:
        """Project raw K=V and launch CP communication before query GEMMs."""
        key_value = self.kv_norm(self.kv_proj(hidden_states)).unsqueeze(1)
        key_value = _apply_v41_rope(key_value, cos, sin)
        if cp_context is None:
            return key_value, None
        return None, cp_context.launch(key_value, 2)

    def _project_query(
            self,
            hidden_states: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and normalize the local query heads."""
        batch_size, sequence_length, _ = hidden_states.shape
        query_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        query = self.q_b_proj(query_residual).view(batch_size, sequence_length, -1, self.head_dim)
        query = self.q_b_norm(query.transpose(1, 2))
        return query_residual, _apply_v41_rope(query, cos, sin)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
            position_ids: torch.Tensor,
            attention_mask: torch.Tensor | None,
            past_key_values: Any | None = None,
            **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Publish or consume compressed KV and execute indexed K=V attention."""
        if past_key_values is not None:
            raise NotImplementedError("shared compressed DSA training does not support KV cache")
        shared_state = kwargs.pop("shared_attention_state", None)
        if not isinstance(shared_state, SharedCompressedAttentionState):
            raise ValueError("shared_attention_state is required for shared compressed DSA")
        if attention_mask is not None:
            raise NotImplementedError(
                "shared compressed DSA consumes compact packed_seq_params instead of a dense mask"
            )

        batch_size, sequence_length, _ = hidden_states.shape
        packed_sequence = kwargs.pop("packed_seq_params", None)
        if packed_sequence is not None and not isinstance(
                packed_sequence, SharedCompressedPackedSequence
        ):
            raise TypeError(
                "packed_seq_params must be SharedCompressedPackedSequence, "
                f"got {type(packed_sequence).__name__}"
            )
        cp_context = kwargs.pop("shared_attention_cp_context", None)
        if cp_context is not None and not isinstance(cp_context, SharedCompressedAttentionCPContext):
            raise TypeError(
                "shared_attention_cp_context must be SharedCompressedAttentionCPContext, "
                f"got {type(cp_context).__name__}"
            )
        tp_context = kwargs.pop("shared_attention_tp_context", None)
        if tp_context is not None and not isinstance(tp_context, SharedCompressedAttentionTPContext):
            raise TypeError(
                "shared_attention_tp_context must be SharedCompressedAttentionTPContext, "
                f"got {type(tp_context).__name__}"
            )
        query_offset = 0
        global_sequence_length = sequence_length
        if cp_context is not None:
            query_offset, global_sequence_length = self._validate_cp_inputs(
                cp_context,
                position_ids,
                sequence_length,
                self.compress_ratio,
                self.is_kv_source,
            )
        segment_starts = None
        minimum_key_indices = None
        if packed_sequence is not None:
            if batch_size != 1:
                raise ValueError("V4.1 compact packed attention currently requires micro_batch_size=1")
            if (
                    packed_sequence.local_query_start != query_offset
                    or packed_sequence.local_query_length != sequence_length
                    or packed_sequence.global_sequence_length != global_sequence_length
            ):
                packed_geometry = (
                    packed_sequence.local_query_start,
                    packed_sequence.local_query_length,
                    packed_sequence.global_sequence_length,
                )
                raise ValueError(
                    "packed sequence geometry does not match the local CP shard: "
                    f"packed={packed_geometry}, "
                    f"attention={(query_offset, sequence_length, global_sequence_length)}"
                )
            packed_sequence.validate_compression_alignment(self.compress_ratio)
            segment_starts = packed_sequence.local_segment_starts(hidden_states.device)
            if self.compress_ratio:
                minimum_key_indices = segment_starts // self.compress_ratio

        rope_type = "compress" if self.compress_ratio else "main"
        cos, sin = position_embeddings[rope_type]
        key_value, raw_kv_handle = self._project_raw_kv(hidden_states, cos, sin, cp_context)
        query_residual, query = self._project_query(hidden_states, cos, sin)

        compressed_handle = None
        latent = None
        if self.is_kv_source:
            latent, compressed = self.compressor(hidden_states, position_embeddings["compress"])
            shared_state.index_key = None
            shared_state.candidate_blocks = None
            if cp_context is None:
                shared_state.compressed_kv = compressed
            else:
                compressed_handle = cp_context.launch(compressed, 1)

        indexer_output = None
        if self.is_index_source:
            indexer_output = self.indexer(
                hidden_states,
                query_residual,
                latent,
                position_embeddings["compress"],
                index_key=shared_state.index_key,
                candidate_blocks=shared_state.candidate_blocks,
                cp_context=cp_context,
                tp_context=tp_context,
                query_offset=query_offset,
                minimum_key_indices=minimum_key_indices,
            )
            shared_state.index_key = indexer_output.index_key
            shared_state.topk_indices = indexer_output.topk_indices
            if indexer_output.candidate_blocks is not None:
                shared_state.candidate_blocks = indexer_output.candidate_blocks

        if raw_kv_handle is not None:
            key_value = raw_kv_handle.wait()
        if compressed_handle is not None:
            shared_state.compressed_kv = compressed_handle.wait()
        if key_value is None:
            raise RuntimeError("raw KV all-gather did not produce a tensor")

        window = build_sliding_window_indices(
            batch_size,
            sequence_length,
            self.sliding_window,
            hidden_states.device,
            query_offset=query_offset,
            key_length=global_sequence_length,
        )
        if segment_starts is not None:
            window.masked_fill_(window < segment_starts.unsqueeze(-1), -1)
        if self.compress_ratio:
            if shared_state.compressed_kv is None or shared_state.topk_indices is None:
                raise RuntimeError(f"layer {self.layer_idx} consumed shared attention before its source ran")
            combined_key_value = torch.cat((key_value, shared_state.compressed_kv.unsqueeze(1)), dim=2)
            compressed_indices = shared_state.topk_indices.long() + global_sequence_length
            compressed_indices.masked_fill_(shared_state.topk_indices < 0, -1)
            sparse_indices = torch.cat((window, compressed_indices), dim=-1)
        else:
            combined_key_value = key_value
            sparse_indices = window

        if (
                indexer_output is not None
                and self.training
                and self.indexer.loss_coeff
                and shared_state.compressed_kv is not None
        ):
            indexer_loss = shared_compressed_indexer_kl_loss(
                indexer_output.index_query,
                indexer_output.index_key,
                indexer_output.merge_weight,
                query,
                shared_state.compressed_kv,
                indexer_output.topk_indices,
                self.sinks,
                attention_scale=self.scaling,
                loss_coeff=self.indexer.loss_coeff,
                query_chunk_size=self.indexer.query_chunk_size,
                tp_context=tp_context,
            )
            query = aux_loss_auto_scale(query, indexer_loss)

        if hidden_states.device.type == "npu":
            attention_output = npu_sparse_attention_with_scalar_sink(
                query,
                combined_key_value,
                sparse_indices,
                self.sinks,
                self.rope_head_dim,
                self.scaling,
            )
        else:
            attention_output = _reference_sparse_attention(
                query,
                combined_key_value,
                sparse_indices,
                self.sinks,
                self.scaling,
            )
        attention_output = _apply_v41_rope(attention_output.transpose(1, 2), cos, -sin).transpose(1, 2)
        grouped = attention_output.reshape(batch_size, sequence_length, self.num_groups, -1)
        hidden_per_group = grouped.shape[-1]
        output_per_group = self.o_a_proj.weight.shape[0] // self.num_groups
        projection = self.o_a_proj.weight.view(
            self.num_groups, output_per_group, hidden_per_group
        ).transpose(1, 2)
        flattened = grouped.reshape(-1, self.num_groups, hidden_per_group).transpose(0, 1)
        projected = torch.bmm(flattened, projection).transpose(0, 1)
        projected = projected.reshape(batch_size, sequence_length, -1)
        return self.o_b_proj(projected), None


__all__ = [
    "SharedCompressedAttentionCPContext",
    "SharedCompressedAttentionState",
    "SharedCompressedPackedSequence",
    "SharedCompressedAttentionTPContext",
    "SharedCompressedDSAAttention",
    "SharedCompressedDSAIndexer",
    "build_sliding_window_indices",
    "compressed_candidate_topk",
    "compressed_causal_topk_and_candidates",
    "compressed_causal_topk",
    "select_candidate_block_indices",
    "select_candidate_blocks",
    "shared_compressed_indexer_kl_loss",
    "npu_sparse_attention_with_scalar_sink",
]
