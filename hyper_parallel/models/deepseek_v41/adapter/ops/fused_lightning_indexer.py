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
"""Fused Lightning-Indexer selection for the DeepSeek-V4.1 CSA chains.

The operator is Ascend specific and its constraints are V4.1 specific, so the
adapter lives with the model family; the shared module only holds the hook.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    register_fused_selection_provider,
)


_FUSED_LOGGER = logging.getLogger(__name__)
_FUSED_INDEXER_STATE = {"checked": False, "available": False, "engaged": False}
_FUSED_INDEXER_HEADS = (16, 24, 32, 48, 64)


def _note_fused_engaged(chain: str) -> None:
    """Record once, in the run's own log, that the operator actually ran.

    A run that silently takes the torch path looks exactly like a run that
    takes the fused path: no warning is emitted either way. State the positive
    fact instead of inferring it from the absence of a fallback warning.
    """
    if not _FUSED_INDEXER_STATE.get("engaged"):
        _FUSED_INDEXER_STATE["engaged"] = True
        _FUSED_LOGGER.info("fused lightning-indexer path engaged (%s)", chain)


def _fused_indexer_available() -> bool:
    """Lazily detect the fused Lightning-Indexer op (compressed-causal mode).

    Whether the fused path is wanted comes from configuration; this probe
    only answers whether the operator exists in the running environment.
    ``V41_DISABLE_FUSED_INDEXER`` stays as a debugging override so a run can
    be pinned to the reference path without editing the recipe.
    """
    import os  # pylint: disable=C0415

    state = _FUSED_INDEXER_STATE
    if not state["checked"]:
        state["checked"] = True
        state["available"] = False
        if not os.environ.get("V41_DISABLE_FUSED_INDEXER"):
            try:  # pragma: no cover - platform probe
                import omni_training_custom_ops  # noqa: F401  # pylint: disable=C0415,W0611
                state["available"] = hasattr(torch.ops.custom, "npu_lightning_indexer_enhance")
            except Exception:  # noqa: BLE001  # pylint: disable=W0703
                state["available"] = False
    return state["available"]


def _fused_compressed_causal_topk(
        query: torch.Tensor,
        key: torch.Tensor,
        merge_weight: torch.Tensor,
        *,
        compress_ratio: int,
        top_k: int,
        query_offset: int,
) -> torch.Tensor:
    """Run the causal indexer via the fused op; matches the reference layout.

    The op returns score-ordered indices padded with -1; the reference
    contract is ascending indices with the -1 padding trailing. When top_k
    is below the op's minimum sparse_count, the score outputs select the
    exact top-k without assuming any op-side ordering.
    """
    batch, seq, _, _ = query.shape
    comp_len = key.shape[1]
    top_k = min(top_k, comp_len)  # torch path returns min(sparse_count, comp_len) columns
    sparse_count = min(8192, max(1024, ((top_k + 1023) // 1024) * 1024))
    mode = 4 | (int(compress_ratio) << 8)
    indices, values, _unused_candidates = torch.ops.custom.npu_lightning_indexer_enhance(
        query, key.unsqueeze(2), merge_weight.to(query.dtype),
        layout_query="BSND", layout_key="BSND",
        sparse_count=sparse_count, sparse_mode=mode,
        pre_tokens=int(query_offset), return_value=True,
    )
    indices = indices.reshape(batch, seq, sparse_count)
    if top_k < sparse_count:
        scores = values.reshape(batch, seq, sparse_count).float()
        scores = scores.masked_fill(indices < 0, float("-inf"))
        keep = scores.topk(top_k, dim=-1).indices
        indices = indices.gather(-1, keep)
    indices = indices.masked_fill(indices < 0, comp_len)
    indices = indices.sort(dim=-1).values
    return indices.masked_fill(indices == comp_len, -1).to(torch.int32)


_FUSED_FULLSCORE_MAX_COMP = 8192
_FUSED_ROW_CHUNK = 1024


def _fused_topsc_pairs(
        query: torch.Tensor,
        key: torch.Tensor,
        merge_weight: torch.Tensor,
        *,
        compress_ratio: int,
        query_offset: int,
):
    """Return the op's score-DESCENDING (indices, values) full cover.

    With sparse_count >= comp_len every visible position appears once, in
    descending score order, padded with -1; downstream consumers exploit the
    ordering (slice for top-k, ordered prefix selection for reindex).
    """
    batch, seq, _, _ = query.shape
    comp_len = key.shape[1]
    sc = min(_FUSED_FULLSCORE_MAX_COMP, max(1024, ((comp_len + 1023) // 1024) * 1024))
    mode = 4 | (int(compress_ratio) << 8)
    indices, values, _ = torch.ops.custom.npu_lightning_indexer_enhance(
        query.contiguous(), key.unsqueeze(2), merge_weight.to(query.dtype).contiguous(),
        layout_query="BSND", layout_key="BSND",
        sparse_count=sc, sparse_mode=mode,
        pre_tokens=int(query_offset), return_value=True,
    )
    rows = batch * seq
    return indices.reshape(rows, sc).to(torch.int64), values.reshape(rows, sc).float()


def _fused_fullscore_usable(query, key, reduce_sum, compress_ratio) -> bool:
    """Gate shared by the source and reindex fused branches."""
    return (_fused_indexer_available() and query.device.type == "npu"
            and reduce_sum is None
            and query.dtype in (torch.bfloat16, torch.float16)
            and query.shape[2] in _FUSED_INDEXER_HEADS and query.shape[3] == 128
            and key.shape[1] <= _FUSED_FULLSCORE_MAX_COMP and compress_ratio >= 1)


def _fused_causal_usable(query, reduce_sum, minimum_key_indices,
                         sparse_count, compress_ratio) -> bool:
    """Gate for the fused causal top-k branch."""
    return (_fused_indexer_available() and query.device.type == "npu"
            and reduce_sum is None and minimum_key_indices is None
            and query.dtype in (torch.bfloat16, torch.float16)
            and query.shape[2] in _FUSED_INDEXER_HEADS and query.shape[3] == 128
            and 0 < sparse_count <= 8192 and compress_ratio >= 1)


def _ascending_with_invalid_tail(indices: torch.Tensor, invalid_value: int) -> torch.Tensor:
    """Sort ascending keeping invalid slots (== invalid_value) trailing as -1.

    ArgSort has no AI Core implementation for int32/int64 and would fall back
    to AI CPU, which dominates the step at long sequences; compressed indices
    are far below 2**24, so sort as float32 (exact) and cast back.
    """
    indices = indices.float().sort(dim=-1).values.to(torch.int64)
    return indices.masked_fill(indices == invalid_value, -1).to(torch.int32)


def _fused_source_segment(
        query, key, merge_weight, *,
        compress_ratio, query_offset, sparse_count, topk_blocks, block_size):
    """Source-layer outputs for one query-row segment (descending-order op)."""
    batch, seq = query.shape[0], query.shape[1]
    comp_len = key.shape[1]
    pair_idx, pair_val = _fused_topsc_pairs(
        query, key, merge_weight, compress_ratio=compress_ratio,
        query_offset=query_offset)
    rows = batch * seq
    top_k = min(sparse_count, comp_len)
    # descending order: the first top_k valid slots ARE the top-k
    head = pair_idx[:, :top_k]
    topk_out = _ascending_with_invalid_tail(
        head.masked_fill(head < 0, comp_len), comp_len)
    num_blocks = (comp_len + block_size - 1) // block_size
    blk_of = pair_idx.clamp(min=0) // block_size
    blocks = torch.full((rows, num_blocks + 1), float("-inf"), device=query.device)
    blocks.scatter_reduce_(1, blk_of.masked_fill(pair_idx < 0, num_blocks),
                           pair_val, reduce="amax", include_self=True)
    blocks = blocks[:, :num_blocks]
    vis = (torch.arange(query_offset, query_offset + seq,
                        device=query.device) + 1) // compress_ratio
    vis = vis.repeat(batch)
    last = (vis - 1).clamp(min=0) // block_size
    row_pos = torch.arange(num_blocks, device=query.device)
    force = (row_pos.unsqueeze(0) == last.unsqueeze(1)) & (vis > 0).unsqueeze(1)
    blocks = blocks.masked_fill(force, float("inf"))
    blk_k = min(topk_blocks, num_blocks)
    cb_v, cb_i = blocks.topk(blk_k, dim=-1, sorted=False)
    cand = cb_i.masked_fill(cb_v == float("-inf"), -1).to(torch.int32)
    return topk_out.view(batch, seq, top_k), cand.view(batch, seq, blk_k)


def _fused_reindex_segment(
        query, key, merge_weight, candidate_blocks, minimum_key_indices, *,
        compress_ratio, query_offset, sparse_count, block_size):
    """Candidate-restricted top-k for one query-row segment."""
    batch, seq = query.shape[0], query.shape[1]
    comp_len = key.shape[1]
    pair_idx, _ = _fused_topsc_pairs(
        query, key, merge_weight, compress_ratio=compress_ratio,
        query_offset=query_offset)
    rows = batch * seq
    num_blocks = (comp_len + block_size - 1) // block_size
    cand = candidate_blocks.reshape(rows, -1).to(torch.int64)
    in_cand = torch.zeros(rows, num_blocks + 1, dtype=torch.bool, device=query.device)
    in_cand.scatter_(1, cand.masked_fill(cand < 0, num_blocks),
                     torch.ones_like(cand, dtype=torch.bool))
    in_cand = in_cand[:, :num_blocks]
    entry_ok = pair_idx >= 0
    entry_ok &= in_cand.gather(1, pair_idx.clamp(min=0) // block_size)
    if minimum_key_indices is not None:
        minimum = minimum_key_indices.reshape(rows, 1).to(pair_idx.dtype)
        entry_ok &= pair_idx >= minimum
    top_k = min(sparse_count, comp_len)
    # descending order: the first top_k qualifying entries are the
    # candidate-restricted top-k (ordered prefix selection)
    keep = entry_ok & (entry_ok.cumsum(dim=-1, dtype=torch.int32) <= top_k)
    sel = pair_idx.masked_fill(~keep, comp_len)
    sel = sel.sort(dim=-1).values[:, :top_k]
    return _ascending_with_invalid_tail(sel, comp_len).view(batch, seq, top_k)


class FusedLightningIndexerProvider:
    """Serve the three CSA selection chains with the fused operator.

    Every method returns ``None`` when this provider cannot serve the call, so
    the reference implementation in the shared module runs instead. OOM is
    raised rather than swallowed: degrading silently would disguise a fallback
    run as a fused one. Any other operator failure warns once and latches the
    provider off for the rest of the process.
    """

    @staticmethod
    def causal_topk(
            query: torch.Tensor,
            key: torch.Tensor,
            merge_weight: torch.Tensor,
            *,
            compress_ratio: int,
            sparse_count: int,
            query_offset: int,
            reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None,
            minimum_key_indices: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return causal top-k positions, or None to use the torch path."""
        if not _fused_causal_usable(query, reduce_sum, minimum_key_indices,
                                    sparse_count, compress_ratio):
            return None
        try:
            selected = _fused_compressed_causal_topk(
                query, key, merge_weight, compress_ratio=compress_ratio,
                top_k=sparse_count, query_offset=query_offset)
        except torch.OutOfMemoryError:
            raise
        except Exception:  # noqa: BLE001  # pylint: disable=W0703
            _disable_after_error()
            return None
        _note_fused_engaged("compressed_causal_topk")
        return selected

    @staticmethod
    def topk_and_candidates(
            query: torch.Tensor,
            key: torch.Tensor,
            merge_weight: torch.Tensor,
            *,
            compress_ratio: int,
            sparse_count: int,
            topk_blocks: int,
            block_size: int,
            query_offset: int,
            reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None,
            minimum_key_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return (positions, candidate blocks), or None to use the torch path."""
        if minimum_key_indices is not None:
            return None
        if not _fused_fullscore_usable(query, key, reduce_sum, compress_ratio):
            return None
        try:
            with torch.no_grad():  # indexer selection is index-valued
                seq = query.shape[1]
                outs = []
                for s0 in range(0, seq, _FUSED_ROW_CHUNK):
                    s1 = min(seq, s0 + _FUSED_ROW_CHUNK)
                    outs.append(_fused_source_segment(
                        query[:, s0:s1], key, merge_weight[:, s0:s1],
                        compress_ratio=compress_ratio,
                        query_offset=query_offset + s0,
                        sparse_count=sparse_count, topk_blocks=topk_blocks,
                        block_size=block_size))
                selected = (torch.cat([seg[0] for seg in outs], dim=1),
                            torch.cat([seg[1] for seg in outs], dim=1))
        except torch.OutOfMemoryError:
            raise
        except Exception:  # noqa: BLE001  # pylint: disable=W0703
            _disable_after_error()
            return None
        _note_fused_engaged("compressed_causal_topk_and_candidates")
        return selected

    @staticmethod
    def candidate_topk(
            query: torch.Tensor,
            key: torch.Tensor,
            merge_weight: torch.Tensor,
            candidate_blocks: torch.Tensor,
            *,
            compress_ratio: int,
            sparse_count: int,
            block_size: int,
            query_offset: int,
            reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None,
            minimum_key_indices: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return candidate-restricted positions, or None to use the torch path."""
        if not _fused_fullscore_usable(query, key, reduce_sum, compress_ratio):
            return None
        try:
            with torch.no_grad():  # indexer selection is index-valued
                seq = query.shape[1]
                outs = []
                for s0 in range(0, seq, _FUSED_ROW_CHUNK):
                    s1 = min(seq, s0 + _FUSED_ROW_CHUNK)
                    mki = None
                    if minimum_key_indices is not None:
                        mki = minimum_key_indices[:, s0:s1]
                    outs.append(_fused_reindex_segment(
                        query[:, s0:s1], key, merge_weight[:, s0:s1],
                        candidate_blocks[:, s0:s1], mki,
                        compress_ratio=compress_ratio,
                        query_offset=query_offset + s0,
                        sparse_count=sparse_count, block_size=block_size))
                selected = torch.cat(outs, dim=1)
        except torch.OutOfMemoryError:
            raise
        except Exception:  # noqa: BLE001  # pylint: disable=W0703
            _disable_after_error()
            return None
        _note_fused_engaged("compressed_candidate_topk")
        return selected


def _disable_after_error() -> None:
    """Warn once and keep the torch path for the rest of the process."""
    _FUSED_LOGGER.warning(
        "fused lightning-indexer path disabled after error", exc_info=True)
    _FUSED_INDEXER_STATE["available"] = False


def register_fused_indexer() -> None:
    """Install this provider into the shared CSA selection chains."""
    register_fused_selection_provider(FusedLightningIndexerProvider())
