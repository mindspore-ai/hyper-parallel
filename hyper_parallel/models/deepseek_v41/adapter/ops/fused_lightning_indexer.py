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
    # The operator accepts any sparse_count and its output is a prefix of any
    # wider request (probed, prefixes identical), so ask for exactly top_k:
    # the operator's cost is comp-scan dominated either way, and the former
    # round-up-to-1024 forced a value-ranked truncation pass afterwards.
    mode = 4 | (int(compress_ratio) << 8)
    indices, _, _ = torch.ops.custom.npu_lightning_indexer_enhance(
        query, key.unsqueeze(2), merge_weight.to(query.dtype),
        layout_query="BSND", layout_key="BSND",
        sparse_count=top_k, sparse_mode=mode,
        pre_tokens=int(query_offset), return_value=True,
    )
    indices = indices.reshape(batch, seq, top_k)
    indices = indices.masked_fill(indices < 0, comp_len)
    # int sort has no AI Core kernel (falls back to AI CPU) and even the
    # float32 full sort loses to the topk kernel by two orders of magnitude
    # at this shape; indices are far below 2**24, so float32 is exact.
    width = indices.shape[-1]
    indices = indices.float().topk(width, dim=-1, largest=False, sorted=True).values.long()
    return indices.masked_fill(indices == comp_len, -1).to(torch.int32)


_GATHER_ROW_CHUNK = 4096


def _gathered_usable(query, reduce_sum, compress_ratio) -> bool:
    """Gate for the gathered candidate-reselection path.

    The path calls no operator, so it needs no operator availability, no
    head-count constraint, and no compressed-length cap; it only wants the
    accelerator and a local (non-reduced) score.
    """
    return (query.device.type == "npu" and reduce_sum is None
            and query.dtype in (torch.bfloat16, torch.float16)
            and compress_ratio >= 1)


def _fused_source_usable(query, reduce_sum, compress_ratio) -> bool:
    """Gate for the two-call source path (no compressed-length cap).

    Both calls ask the operator for a fixed sparse_count (position top-k,
    then block top-k), so usability does not depend on how long the
    compressed key is - unlike the full-coverage path this stays usable at
    long sequences.
    """
    return (_fused_indexer_available() and query.device.type == "npu"
            and reduce_sum is None
            and query.dtype in (torch.bfloat16, torch.float16)
            and query.shape[2] in _FUSED_INDEXER_HEADS and query.shape[3] == 128
            and compress_ratio >= 1)


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
    width = indices.shape[-1]
    ordered = indices.float().topk(width, dim=-1, largest=False, sorted=True).values
    # stay in the float domain (exact below 2**24) until the final cast
    return ordered.masked_fill(ordered == invalid_value, -1).to(torch.int32)


def _fused_source_two_call(
        query, key, merge_weight, *,
        compress_ratio, query_offset, sparse_count, topk_blocks, block_size):
    """Source-layer outputs from two fixed-sparse-count operator calls.

    Call A (sparse_block_size=1) returns score-descending positions whose
    first ``top_k`` entries are the exact top-k; call B
    (sparse_block_size=block_size) returns block-max top ``topk_blocks``
    block indices natively. Neither call needs to cover every visible
    position, so the compressed length is unbounded - this is what unlocks
    the source chain beyond the full-coverage cap. Only the reference's
    forced last block needs a torch-side fixup: when absent it replaces the
    weakest selected block.
    """
    batch, seq = query.shape[0], query.shape[1]
    comp_len = key.shape[1]
    mode = 4 | (int(compress_ratio) << 8)
    key_b = key.unsqueeze(2)
    weight = merge_weight.to(query.dtype).contiguous()
    rows = batch * seq
    top_k = min(sparse_count, comp_len)
    idx_a, _, _ = torch.ops.custom.npu_lightning_indexer_enhance(
        query.contiguous(), key_b, weight,
        layout_query="BSND", layout_key="BSND",
        sparse_count=top_k, sparse_mode=mode, sparse_block_size=1,
        pre_tokens=int(query_offset), return_value=True,
    )
    head = idx_a.reshape(rows, top_k)  # operator emits int32; keep it
    topk_out = _ascending_with_invalid_tail(
        head.masked_fill(head < 0, comp_len), comp_len)

    num_blocks = (comp_len + block_size - 1) // block_size
    blk_k = min(topk_blocks, num_blocks)
    sc_b = blk_k * block_size
    idx_b, val_b, _ = torch.ops.custom.npu_lightning_indexer_enhance(
        query.contiguous(), key_b, weight,
        layout_query="BSND", layout_key="BSND",
        sparse_count=sc_b, sparse_mode=mode, sparse_block_size=block_size,
        pre_tokens=int(query_offset), return_value=True,
    )
    cand = idx_b.reshape(rows, -1)[:, :blk_k]  # operator emits int32; keep it
    cand_val = val_b.reshape(rows, -1)[:, :blk_k].float()
    # The block mode can emit out-of-range slots on boundary rows
    # (sparse_block_size > 1 with query_offset > 0); scrub them so the
    # forced-last fixup and any downstream consumers stay safe.
    bad = (cand < 0) | (cand >= num_blocks)
    cand = cand.masked_fill(bad, -1)
    cand_val = cand_val.masked_fill(bad, float("-inf"))
    # Forced last visible block, matching select_candidate_block_indices.
    # Branch-free: a data-dependent host check would synchronize the stream,
    # so the weakest-slot replacement is computed unconditionally and gated
    # per row by torch.where.
    vis = (torch.arange(query_offset, query_offset + seq,
                        device=query.device, dtype=torch.int32) + 1) // compress_ratio
    vis = vis.repeat(batch)
    last = (vis - 1).clamp(min=0) // block_size
    has_last = (cand == last.unsqueeze(1)).any(dim=1)
    need = (vis > 0) & ~has_last
    weak = cand_val.masked_fill(cand < 0, float("inf")).argmin(dim=1)
    col = torch.arange(blk_k, device=query.device)
    replace = need.unsqueeze(1) & (col.unsqueeze(0) == weak.unsqueeze(1))
    cand = torch.where(replace, last.unsqueeze(1), cand)
    cand = cand.masked_fill(cand < 0, -1)
    cand = cand.masked_fill((vis == 0).unsqueeze(1), -1).to(torch.int32)
    return topk_out.view(batch, seq, top_k), cand.view(batch, seq, blk_k)


def _gathered_candidate_topk(
        query, key, merge_weight, candidate_blocks, minimum_key_indices, *,
        compress_ratio, query_offset, sparse_count, block_size):
    """Candidate-restricted top-k over gathered candidate keys.

    The reference path's per-position advanced-index gather runs ~90x below
    memory bandwidth on this backend and dominates the chain at long
    sequences; ``index_select`` restores it to bandwidth. Scoring stays in
    the reference's fp32 formula (bf16 dot products flipped ~20% of rows
    past the tie tolerance), every ordering runs on the topk kernel, and the
    cost scales with the candidate width - not the compressed length - so
    there is no length cap at all.
    """
    batch, seq = query.shape[0], query.shape[1]
    comp_len = key.shape[1]
    device = query.device
    # int64 vector arithmetic runs 2-4x slower than int32 on this backend;
    # positions stay int32 throughout and widen to int64 only at the index
    # arguments torch requires (index_select / gather).
    block_offsets = torch.arange(block_size, device=device, dtype=torch.int32)
    outs = []
    for s0 in range(0, seq, _GATHER_ROW_CHUNK):
        s1 = min(seq, s0 + _GATHER_ROW_CHUNK)
        rows = s1 - s0
        blocks = candidate_blocks[:, s0:s1]
        block_positions = blocks.unsqueeze(-1) * block_size + block_offsets
        positions = block_positions.flatten(-2)
        visible = (torch.arange(query_offset + s0, query_offset + s1,
                                device=device, dtype=torch.int32) + 1) // compress_ratio
        valid = (blocks.unsqueeze(-1) >= 0).expand_as(block_positions).flatten(-2)
        valid = valid & (positions < comp_len) & (positions < visible.view(1, -1, 1))
        if minimum_key_indices is not None:
            valid = valid & (positions >= minimum_key_indices[:, s0:s1].unsqueeze(-1))
        safe = positions.clamp(min=0, max=max(comp_len - 1, 0)).long()
        width = safe.shape[-1]
        gathered = torch.stack([
            torch.index_select(key[b], 0, safe[b].reshape(-1)).view(rows, width, -1)
            for b in range(batch)
        ]).float()
        dots = torch.einsum("bchd,bcwd->bchw", query[:, s0:s1].float(), gathered).relu()
        # head reduction as a 1x16 @ 16xW matmul: one cube pass instead of a
        # broadcast-multiply intermediate plus a vector sum
        scores = torch.matmul(merge_weight[:, s0:s1].float().unsqueeze(2), dots).squeeze(2)
        scores = scores.masked_fill(~valid, float("-inf"))
        top_k = min(sparse_count, width)
        top = scores.topk(top_k, dim=-1, sorted=False)
        sel = positions.gather(-1, top.indices)
        sel = sel.float().masked_fill(top.values == float("-inf"), comp_len)
        # ascending order with the -1 padding trailing, via the topk kernel
        # (int sort has no AI Core implementation on this backend); stay in
        # the float domain until the final int32 cast
        sel = sel.topk(top_k, dim=-1, largest=False, sorted=True).values
        outs.append(sel.masked_fill(sel == comp_len, -1).to(torch.int32))
    return torch.cat(outs, dim=1)


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
        if not _fused_source_usable(query, reduce_sum, compress_ratio):
            return None
        try:
            with torch.no_grad():  # indexer selection is index-valued
                selected = _fused_source_two_call(
                    query, key, merge_weight, compress_ratio=compress_ratio,
                    query_offset=query_offset, sparse_count=sparse_count,
                    topk_blocks=topk_blocks, block_size=block_size)
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
        if not _gathered_usable(query, reduce_sum, compress_ratio):
            return None
        try:
            with torch.no_grad():  # indexer selection is index-valued
                selected = _gathered_candidate_topk(
                    query, key, merge_weight, candidate_blocks,
                    minimum_key_indices, compress_ratio=compress_ratio,
                    query_offset=query_offset, sparse_count=sparse_count,
                    block_size=block_size)
        except torch.OutOfMemoryError:
            raise
        except Exception:  # noqa: BLE001  # pylint: disable=W0703
            _disable_after_error()
            return None
        return selected


def _disable_after_error() -> None:
    """Warn once and keep the torch path for the rest of the process."""
    _FUSED_LOGGER.warning(
        "fused lightning-indexer path disabled after error", exc_info=True)
    _FUSED_INDEXER_STATE["available"] = False


def register_fused_indexer() -> None:
    """Install this provider into the shared CSA selection chains."""
    register_fused_selection_provider(FusedLightningIndexerProvider())
