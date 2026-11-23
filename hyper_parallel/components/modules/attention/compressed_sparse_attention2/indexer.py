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
"""CSA2 Indexer projections, causal selection and candidate-pool reference math."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.functional.lightning_indexer import (
    lightning_indexer, sort_key_indices, validate_indexer_tensors,
)
from hyper_parallel.components.modules.common.rms_norm import NativeRMSNormModule, RMSNorm
from hyper_parallel.core.activation_memory.api import checkpoint_exclude_wrapper


@dataclass
class CSA2IndexerOutput:
    """Tensors produced by one Full or Reindex Indexer invocation."""

    topk_indices: torch.Tensor
    index_key: torch.Tensor
    candidate_blocks: torch.Tensor | None
    index_query: torch.Tensor
    merge_weight: torch.Tensor


def _run_lightning_indexer(
        query: torch.Tensor, key: torch.Tensor, merge_weight: torch.Tensor,
        top_k: int, compress_ratio: int, query_offset: int,
        reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None,
        minimum_key_indices: torch.Tensor | None, query_segments: tuple | None,
) -> torch.Tensor:
    """Run LI V2 with explicit adapter errors and native capability validation."""
    if reduce_sum is not None:
        raise NotImplementedError("Fused LightningIndexer does not support TP score reduction before TopK")
    if minimum_key_indices is not None and query_segments is None:
        raise ValueError("Fused LightningIndexer requires query_segments when minimum_key_indices is set")
    validate_indexer_tensors(query, key, merge_weight)
    if query.device.type != "npu":
        raise RuntimeError(f"Fused LightningIndexer requires NPU inputs; got device={query.device}")
    if query_segments is None:
        length = query.shape[1]
        raw_end = query_offset + length
        query_segments = ((0, length, 0, raw_end // compress_ratio, raw_end % compress_ratio),)
    return lightning_indexer(query, key, merge_weight, compress_ratio, top_k, query_segments)

def _causal_key_bounds(
        query: torch.Tensor, ratio: int, query_offset: int,
        minimum_key_indices: torch.Tensor | None, segments: tuple | None,
        causal_upper: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Derive visibility from original query positions within each independently compressed document.

    The returned upper bound always has shape ``[1, queries]`` (shared across
    the batch), so chunked consumers slice it along the sequence dimension.
    ``causal_upper`` is the packed per-query exclusive bound shared by eager and
    graph execution; Python ``segments`` remain available for eager callers and
    the fused native kernels.
    """
    batch_size, length = query.shape[:2]
    if causal_upper is not None:
        return causal_upper, minimum_key_indices
    if segments is None:
        positions = torch.arange(query_offset, query_offset + length, device=query.device)
        return ((positions + 1) // ratio).unsqueeze(0), minimum_key_indices
    upper = torch.empty(length, device=query.device, dtype=torch.long)
    lower = torch.empty_like(upper)
    for begin, end, key_start, key_end, residual in segments:
        first = ratio * key_end + residual - (end - begin)
        positions = torch.arange(first, first + end - begin, device=query.device)
        upper[begin:end] = (positions + 1) // ratio
        lower[begin:end] = key_start
    return upper.unsqueeze(0), lower.unsqueeze(0).expand(batch_size, -1)

@torch.no_grad()
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
        query_segments: tuple[tuple[int, int, int, int, int], ...] | None = None,
        causal_upper: torch.Tensor | None = None,
        use_fused: bool = False,
) -> torch.Tensor:
    """Select compressed positions with V4.1's ratio-aware causal rule.

    CSA2 keys represent closed groups of ``compress_ratio`` source tokens.
    Both LI V2 and the batched reference preserve the source rule
    ``key < floor((query + 1) / ratio)``. The optional native path requires
    document-local K prefixes; the reference retains TP score reductions.

    Args:
        query_segments: Local Q intersections and global compressed K prefixes.
        causal_upper: Packed per-query exclusive causal bound shared by eager
            and graph execution; supersedes ``query_segments`` for reference
            selection when provided.
        use_fused: Select LI V2 and raise if a nonempty call is unsupported.
            False explicitly selects the numerical reference.
    """
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    if query.ndim != 4:
        raise ValueError(f"Indexer query must have shape [B, Q, H, D], got {tuple(query.shape)}")
    batch_size, sequence_length, _, head_dim = query.shape
    if key.ndim != 3 or key.shape[::2] != (batch_size, head_dim):
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
    if top_k <= 0 or sequence_length == 0:
        return torch.empty(
            batch_size,
            sequence_length,
            max(top_k, 0),
            dtype=torch.int32,
            device=query.device,
        )

    if use_fused:
        return _run_lightning_indexer(
            query, key, merge_weight, top_k, compress_ratio, query_offset,
            reduce_sum, minimum_key_indices, query_segments,
        )

    visible_keys, minimum_key_indices = _causal_key_bounds(
        query, compress_ratio, query_offset, minimum_key_indices, query_segments, causal_upper,
    )
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
        top = scores.topk(top_k, dim=-1, sorted=False)
        indices = top.indices.masked_fill(~torch.isfinite(top.values), compressed_length)
        indices = sort_key_indices(indices, compressed_length)
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
        scores = torch.matmul(query[:, start:end].float(), key_fp32).relu()
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
        causal_upper: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select Full-layer Top-K and candidate blocks from one score pass.

    ``causal_upper`` supplies the packed per-query causal bound for eager and
    graph execution; it must be compression-aligned packed input when given.
    """
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    batch_size, sequence_length, _, head_dim = query.shape
    if key.ndim != 3 or key.shape[::2] != (batch_size, head_dim):
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
        scores = torch.matmul(query[:, start:end].float(), key_fp32).relu()
        scores = (scores * merge_weight[:, start:end].float().unsqueeze(-1)).sum(dim=2)
        if reduce_sum is not None:
            scores = reduce_sum(scores)
        visible = (
            causal_upper[:, start:end] if causal_upper is not None else
            (torch.arange(query_offset + start, query_offset + end, device=query.device) + 1) // compress_ratio
        )
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
        causal_upper: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score only hierarchical candidate blocks and return global key ids.

    ``causal_upper`` supplies the packed per-query causal bound for eager and
    graph execution; it must be compression-aligned packed input when given.
    """
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    batch_size, sequence_length, _, head_dim = query.shape
    if key.ndim != 3 or key.shape[::2] != (batch_size, head_dim):
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
            causal_upper[:, start:end] if causal_upper is not None else
            (torch.arange(query_offset + start, query_offset + end, device=query.device) + 1) // compress_ratio
        )
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
        ).relu()
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

class _IndexerParameters(nn.Module):
    """Construct the released Full or Reindex Lightning Indexer state."""

    def __init__(self, config: Any, compress_ratio: int, layer_idx: int) -> None:
        """Create layer-local queries and Full-only shared-key projections."""

        super().__init__()
        self.compress_ratio = compress_ratio
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        # Query rows processed per indexer scoring pass. Larger chunks cut the
        # Python-loop launch count but enlarge the transient fp32 score
        # buffer; 256 is the released default, ~1024 measured fastest at 32K.
        self.query_chunk_size = int(getattr(config, "v41_indexer_query_chunk_size", 256) or 256)
        self.owns_key = layer_idx in config.v41_kv_source_layer_ids
        candidate_source = int(getattr(config, "v41_candidate_source_layer_id", -1))
        self.is_candidate_source = layer_idx == candidate_source
        self.uses_candidates = 0 <= candidate_source < layer_idx
        self.candidate_topk_blocks = int(getattr(config, "v41_candidate_topk_blocks", 0))
        self.candidate_block_size = int(getattr(config, "v41_candidate_block_size", 1))
        self.loss_coeff = float(getattr(config, "v41_indexer_loss_coeff", 0.0))
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        if self.owns_key:
            self.wk = nn.Linear(config.head_dim, self.head_dim, bias=False)
            # Keep this source subtree eligible for enclosing CSA2 replacement.
            norm = NativeRMSNormModule(
                config=SimpleNamespace(hidden_size=self.head_dim, eps=config.rms_norm_eps),
                cast_order="before_weight",
            )
            self.k_norm = RMSNorm(module=norm)


class CSA2Indexer(nn.Module):
    """Local Indexer math with overridable key preparation and selection stages."""

    _selection_region: Callable[..., Any] | None = None

    def __init__(self, module: nn.Module | None = None, *, config: Any = None,
                 compress_ratio: int = 1, layer_idx: int = 0, apply_rope: Callable | None = None) -> None:
        """Create or adopt projections without changing checkpoint names."""
        if (module is None) == (config is None):
            raise ValueError("CSA2 Indexer requires exactly one of config or module")
        if module is None:
            module = _IndexerParameters(config, compress_ratio, layer_idx)
        super().__init__()
        required = ("q_b_proj", "weights_proj")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(f"shared compressed DSA indexer is missing: {missing}")
        for name, child in module._modules.items():  # pylint: disable=protected-access
            self.add_module(name, child)
        for name, parameter in module._parameters.items():  # pylint: disable=protected-access
            self.register_parameter(name, parameter)
        for name, buffer in module._buffers.items():  # pylint: disable=protected-access
            self.register_buffer(name, buffer, persistent=name not in module._non_persistent_buffers_set)
        self.apply_rope = apply_rope if apply_rope is not None else getattr(module, "apply_rope", None)
        if self.apply_rope is None:
            raise ValueError("CSA2 Indexer requires the enclosing attention's rotary function")
        self.use_fused = getattr(module, "use_fused", False)
        self.compress_ratio = module.compress_ratio
        self.global_num_heads = getattr(module, "global_num_heads", getattr(module, "num_heads", None))
        self.head_dim = module.head_dim
        self.index_topk = module.index_topk
        self.owns_key = module.owns_key
        self.is_candidate_source = module.is_candidate_source
        self.uses_candidates = module.uses_candidates
        self.candidate_topk_blocks = module.candidate_topk_blocks
        self.candidate_block_size = module.candidate_block_size
        self.loss_coeff = module.loss_coeff
        self.query_chunk_size = int(getattr(module, "query_chunk_size", 256))
        # Children are adopted by identity; preserve their individual training modes.
        self.training = module.training

    def _project_index_query(
            self, query_residual: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
            *, sequence_major: bool = False,
    ) -> torch.Tensor:
        """Project queries with their original rotary positions, including moved tasks."""
        query = self.q_b_proj(query_residual.detach())
        if query.shape[-1] % self.head_dim:
            raise ValueError(
                f"Indexer query width {query.shape[-1]} is not divisible by head_dim {self.head_dim}"
            )
        query = query.view(*query_residual.shape[:2], -1, self.head_dim)
        if sequence_major:
            return self.apply_rope(query, cos, sin, unsqueeze_dim=2)
        return self.apply_rope(query.transpose(1, 2), cos, sin).transpose(1, 2)

    @torch.no_grad()
    def _project_worker_index_query(
            self, query_residual: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
            owner_query_length: int,
    ) -> torch.Tensor:
        """Preserve the owner's projection row shape for discrete worker selection.

        BF16 GEMM rounding can depend on its row count. Match the original
        owner's GEMM shape so task redistribution does not perturb TopK inputs.
        Padding is confined to the no-grad projection and never reaches LI.

        Args:
            query_residual: Low-rank query rows assigned to this worker.
            cos: Original-position cosine rows for the worker queries.
            sin: Original-position sine rows for the worker queries.
            owner_query_length: Query count in the original owner projection.

        Returns:
            Worker Index Q in sequence-major layout, excluding padded rows.
        """
        if owner_query_length <= 0:
            raise ValueError("Indexer owner query length must be positive")
        batch, length = query_residual.shape[:2]
        if length == owner_query_length:
            return self._project_index_query(query_residual, cos, sin, sequence_major=True)
        if not length:
            query = self.q_b_proj(query_residual.detach())
            return query.view(batch, 0, query.shape[-1] // self.head_dim, self.head_dim)

        query = None
        for start in range(0, length, owner_query_length):
            end = min(start + owner_query_length, length)
            chunk = query_residual[:, start:end].detach()
            if end - start < owner_query_length:
                chunk = torch.nn.functional.pad(chunk, (0, 0, 0, owner_query_length - (end - start)))
            projected = self.q_b_proj(chunk)
            if query is None:
                query = projected.new_empty((batch, length, projected.shape[-1]))
            query[:, start:end].copy_(projected[:, :end - start])
            del projected, chunk
        query = query.view(batch, length, query.shape[-1] // self.head_dim, self.head_dim)
        return self.apply_rope(query, cos, sin, unsqueeze_dim=2)

    def _project_index_key(
            self,
            latent: torch.Tensor | None,
            cos: torch.Tensor,
            sin: torch.Tensor,
            index_key: torch.Tensor | None,
            key_position_embeddings: tuple | None = None,
    ) -> torch.Tensor:
        """Project or adopt the index key without communicating."""
        if self.owns_key:
            if latent is None:
                raise RuntimeError("a Full Indexer requires the compressor latent")
            key = self.k_norm(self.wk(latent.detach()))
            compressed_length = key.shape[1]
            if key_position_embeddings is None:
                key_cos = cos[:, :compressed_length * self.compress_ratio:self.compress_ratio]
                key_sin = sin[:, :compressed_length * self.compress_ratio:self.compress_ratio]
            else:
                key_cos, key_sin = key_position_embeddings
            key = self.apply_rope(key.unsqueeze(1), key_cos, key_sin).squeeze(1)
            return key
        if index_key is None:
            raise RuntimeError("a Reindex layer requires index K from the preceding Full layer")
        return index_key

    def prepare_key(self, latent: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor,
                    index_key: torch.Tensor | None, block_plan: Any, key_position_embeddings: tuple | None) -> Any:
        """Prepare a local ready bank; parallel members may return a pending transfer."""
        del block_plan
        return self._project_index_key(latent, cos, sin, index_key, key_position_embeddings)

    def exchange_key(self, prepared: Any) -> torch.Tensor:
        """Keep the local bank unchanged."""
        return prepared

    def forward(
            self, hidden_states: torch.Tensor, query_residual: torch.Tensor, latent: torch.Tensor | None,
            compress_position_embeddings: tuple[torch.Tensor, torch.Tensor], *,
            index_key: torch.Tensor | None = None, candidate_blocks: torch.Tensor | None = None,
            reduce_sum: Callable | None = None, query_offset: int = 0,
            minimum_key_indices: torch.Tensor | None = None, query_segments: tuple | None = None,
            document_lengths: tuple[int, ...] = (), prepare_attention: Callable | None = None,
            block_plan: Any = None, key_position_embeddings: tuple | None = None,
            causal_upper: torch.Tensor | None = None,
    ) -> CSA2IndexerOutput:
        """Project owner queries and select keys; exclusion fixes TopK during replay."""
        del document_lengths
        cos, sin = compress_position_embeddings
        prepared_key = self.prepare_key(latent, cos, sin, index_key, block_plan, key_position_embeddings)
        if prepare_attention is not None:
            prepare_attention()
        query = self._project_index_query(query_residual, cos, sin)
        weights = self.weights_proj(hidden_states.detach())
        if weights.shape[-1] != query.shape[2]:
            raise ValueError("Indexer query and merge weights must have matching head counts")
        weights = weights * (self.head_dim**-0.5 * self.global_num_heads**-0.5)
        key = self.exchange_key(prepared_key)
        selected, candidates = self._run_selection(
            query, key, weights, candidate_blocks, query_offset, reduce_sum,
            minimum_key_indices, query_segments, causal_upper,
        )
        return CSA2IndexerOutput(selected, key, candidates, query, weights)

    def enable_selection_exclude(self) -> None:
        """Reuse the discrete selection during activation recomputation."""
        object.__setattr__(self, "_selection_region", checkpoint_exclude_wrapper(self._select_tokens))

    def _run_selection(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Select directly, or reuse the result retained by the checkpoint region."""
        selection = self._select_tokens if self._selection_region is None else self._selection_region
        return selection(*args, **kwargs)

    def _select_tokens(self, *args: Any, **kwargs: Any) -> tuple:
        """Run only selection inside the checkpoint exclusion region."""
        return self._select_indices(*args, **kwargs)

    def _select_indices(
            self, query: torch.Tensor, key: torch.Tensor, merge_weight: torch.Tensor,
            candidate_blocks: torch.Tensor | None, query_offset: int,
            reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None,
            minimum_key_indices: torch.Tensor | None, query_segments: tuple | None,
            causal_upper: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Select the requested implementation without changing candidate policies."""
        if self.use_fused and (self.is_candidate_source or self.uses_candidates):
            raise NotImplementedError("V4.1 fused Indexer does not support candidate-pool selection")
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
                causal_upper=causal_upper,
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
                causal_upper=causal_upper,
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
                query_segments=query_segments,
                causal_upper=causal_upper,
                use_fused=self.use_fused,
            )
        return topk_indices, candidate_blocks
