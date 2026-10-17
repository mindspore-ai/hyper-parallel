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
"""SparseLightningIndexerKLLossGrad adapter with CSA teacher and scalar KL."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.lightning_indexer import validate_indexer_tensors
from hyper_parallel.components.functional.sparse_flash_mla import CSATNDPlan, csa_tnd_plan


_MAX_FP32_BANK_BYTES = 128 * 1024 * 1024


@lru_cache(maxsize=1)
def _load_kl_ops() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Import the fused KL gradient and its metadata entry together."""
    # Resolve concrete optional functions before allocating teacher tensors.
    try:
        from cann_ops_transformer import (  # pylint: disable=import-outside-toplevel
            sparse_lightning_indexer_kl_loss_grad,
            sparse_lightning_indexer_kl_loss_grad_metadata,
        )
    except ImportError as err:
        raise RuntimeError(
            "use_fused_ops=True requires cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad "
            "and sparse_lightning_indexer_kl_loss_grad_metadata. "
            "See hyper_parallel/components/functional/README.md for ops-transformer installation, "
            "environment setup and tested versions."
        ) from err
    return sparse_lightning_indexer_kl_loss_grad, sparse_lightning_indexer_kl_loss_grad_metadata


def validate_fused_kl_inputs(
        query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor,
) -> None:
    """Check the adapter's ordinary TopK512 contract before preparing the teacher.

    Native operators validate hardware limits, including head dimension,
    head count and sequence length, for the installed operator version.

    Args:
        query: Indexer Q [batch, queries, heads, dimension].
        key: Indexer K [batch, keys, dimension].
        weights: Per-query Indexer head weights.
        indices: Ordinary causal selections, before optional width padding.

    Raises:
        ValueError: Tensor relationships or logical selection width are invalid.
        RuntimeError: Explicit fusion was requested on a non-NPU device.
    """
    validate_indexer_tensors(query, key, weights)
    if indices.ndim != 3 or indices.shape[:2] != query.shape[:2]:
        raise ValueError("Fused KL indices must have shape [B, Q, TopK] matching Indexer Q")
    if key.shape[1] == 0 or indices.shape[-1] != min(512, key.shape[1]):
        raise ValueError(f"Fused KL requires ordinary TopK512 selections from a nonempty bank; "
                         f"got K={key.shape[1]}, TopK={indices.shape[-1]}")
    if indices.device != query.device:
        raise ValueError(f"Fused KL indices must share Indexer Q's device, got {indices.device} and {query.device}")
    if query.device.type != "npu":
        raise RuntimeError(f"Fused KL requires NPU inputs; got device={query.device}")


@lru_cache(maxsize=128)
def _kl_metadata(
        device: torch.device, batch: int, length: int, heads: int, dim: int,
        key_length: int, topk: int, ratio: int, residual: int,
) -> tuple:
    """Cache shape-only causal metadata without retaining activations."""
    residue = torch.full((batch,), residual, device=device, dtype=torch.int32) if ratio > 1 else None
    options = {"layout_q": "BSND", "layout_k": "BSND", "mask_mode": 3,
               "cmp_ratio": ratio, "cmp_residual_k": residue}
    _, metadata_op = _load_kl_ops()
    metadata = metadata_op(
        heads, 1, dim, batch_size=batch, max_seqlen_q=length, max_seqlen_k=key_length,
        topk=topk, **options,
    )
    return options, metadata


def gather_selected_keys(bank: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather batch-local nonnegative indices without multidimensional indexing.

    Args:
        bank: Key bank with shape [batch, keys, dimension].
        indices: Safe, nonnegative indices with shape [batch, queries, selected].
    """
    offsets = torch.arange(bank.shape[0], device=bank.device).view(-1, 1, 1) * bank.shape[1]
    flat_indices = (indices + offsets).reshape(-1)
    selected = bank.reshape(-1, bank.shape[-1]).index_select(0, flat_indices)
    return selected.reshape(*indices.shape, bank.shape[-1])


def _teacher_bank(bank: torch.Tensor, selected_count: int) -> torch.Tensor:
    """Cast a bank only if it saves a larger gathered low-precision temporary."""
    cast_bytes = bank.numel() * 4
    gathered_bytes = selected_count * bank.shape[-1] * bank.element_size()
    if (bank.dtype != torch.float32 and cast_bytes <= _MAX_FP32_BANK_BYTES
            and cast_bytes < gathered_bytes):
        return bank.float()
    return bank


def gather_selected_keys_fp32(bank: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Choose the smaller temporary without converting an unbounded global bank.

    Args:
        bank: Key bank with shape [batch, keys, dimension].
        indices: Nonnegative key IDs with shape [batch, queries, selected].
    """
    return gather_selected_keys(_teacher_bank(bank, indices.numel()), indices).float()


def compressed_attention_teacher(
        query: torch.Tensor, key: torch.Tensor, indices: torch.Tensor, sinks: torch.Tensor,
        scale: float, reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute the detached teacher for one Q chunk, using all heads and selected keys.

    Args:
        query: Main attention query [batch, heads, queries, dimension].
        key: Compressed main key bank [batch, keys, dimension].
        indices: Batch-local main-key indices; negative slots are invalid.
        sinks: Scalar attention sink per head.
        scale: Main attention logit scale.
        reduce_sum: Optional TP SUM of head contributions, before normalization.
    """
    with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
        width = indices.shape[-1]
        if width == 0:
            return query.new_zeros(indices.shape, dtype=torch.float32)
        valid = indices >= 0
        selected = gather_selected_keys_fp32(key, indices.clamp_min(0).long())
        queries = query.float()
        scores = torch.einsum("bhqd,bqkd->bhqk", queries, selected) * scale
        del selected, queries
        scores.masked_fill_(~valid.unsqueeze(1), -1.0e9)
        sink_logits = sinks.float().view(1, -1, 1, 1).expand(*scores.shape[:-1], 1)
        probabilities = torch.cat((scores, sink_logits), dim=-1).softmax(dim=-1)
        target = probabilities[..., :-1].masked_fill(~valid.unsqueeze(1), 0).sum(dim=1)
        if reduce_sum is not None:
            target = reduce_sum(target)
        return target / target.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)


def causal_attention_teacher(
        query: torch.Tensor, key: torch.Tensor, indices: torch.Tensor, sinks: torch.Tensor,
        scale: float, ratio: int, residual: int, chunk_size: int,
) -> torch.Tensor:
    """Preserve the selected-main-plus-sink target without expanding prefix keys.

    Args:
        query: Detached main Q [batch, heads, local queries, dimension].
        key: Document-local main K prefix [batch, completed groups, dimension].
        indices: Sorted selections, all visible keys for an ordinary partial row.
        sinks: Detached scalar logits, one per main attention head.
        scale: Main attention dot-product scale.
        ratio: Compression ratio.
        residual: Incomplete raw-token group at the end of the K prefix.
        chunk_size: Maximum queries per teacher chunk, shared with the eager KL path.
    """
    with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
        batch, heads, length, _ = query.shape
        width = indices.shape[-1]
        offset = key.shape[1] * ratio + residual - length
        partial = min(length, max(0, ratio * width - 1 - offset))
        target = query.new_zeros(indices.shape, dtype=torch.float32)
        max_keys = min(key.shape[1], (offset + partial) // ratio) if partial else 0
        if max_keys:
            prefix = key[:, :max_keys].float()
            counts = (torch.arange(partial, device=query.device) + offset + 1) // ratio
            valid = torch.arange(max_keys, device=query.device)[None, :] < counts[:, None]
            for start in range(0, partial, chunk_size):
                end = min(start + chunk_size, partial)
                scores = torch.einsum("bhqd,bkd->bhqk", query[:, :, start:end].float(), prefix) * scale
                mask = valid[None, None, start:end]
                scores.masked_fill_(~mask, -1.0e9)
                sink = sinks.float().view(1, -1, 1, 1).expand(batch, heads, end-start, 1)
                probability = torch.cat((scores, sink), -1).softmax(-1)[..., :-1]
                target[:, start:end, :max_keys] = probability.masked_fill(~mask, 0).sum(1)
                del scores, sink, probability
            del prefix
            prefix_target = target[:, :partial]
            prefix_target.div_(prefix_target.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny))
        # Reuse a bounded FP32 bank across Q chunks within this teacher call.
        # Larger banks remain in their input dtype and are converted only
        # after selected-key gathering; never retain activation caches across
        # layers or training steps.
        bank = _teacher_bank(key, batch * (length-partial) * width)
        for start in range(partial, length, chunk_size):
            end = min(start + chunk_size, length)
            target[:, start:end] = compressed_attention_teacher(
                query[:, :, start:end], bank, indices[:, start:end], sinks, scale,
            )
        return target


def _stable_prediction_log(
        prediction: torch.Tensor, query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
        indices: torch.Tensor, chunk_size: int,
) -> torch.Tensor:
    """Recover valid underflowed probabilities without treating padding as underflow."""
    valid = indices >= 0
    bad = ((prediction <= 0) & valid).any(dim=-1)
    log_prediction = prediction.masked_fill(~valid, 1).log()
    # The native interface does not return student LSE. One boundary check
    # avoids silently clamping the scalar objective when probabilities vanish.
    if not bool(bad.any()):
        return log_prediction
    for batch in range(query.shape[0]):
        rows = bad[batch].nonzero().flatten()
        for start in range(0, rows.numel(), chunk_size):
            chosen = rows[start:start+chunk_size]
            selected = indices[batch:batch+1].index_select(1, chosen).long()
            keys = gather_selected_keys_fp32(key[batch:batch+1], selected.clamp_min(0))
            queries = query[batch:batch+1].index_select(1, chosen).float()
            merge = weights[batch:batch+1].index_select(1, chosen).float()
            # SAC may cache the contraction output for backward replay.
            logits = torch.einsum("bqhd,bqkd->bqhk", queries, keys).relu()
            logits = (logits * merge.unsqueeze(-1)).sum(dim=2).masked_fill(selected < 0, -1.0e9)
            log_prediction[batch, chosen] = logits.log_softmax(-1).masked_fill(selected < 0, 0)[0]
    return log_prediction


class _SparseLightningIndexerKLLoss(torch.autograd.Function):
    """Adapt SparseLightningIndexerKLLossGrad to Hyper's scalar-loss autograd contract."""

    @staticmethod
    def forward(
            ctx: Any, query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
            attention_query: torch.Tensor, main_key: torch.Tensor, indices: torch.Tensor, sinks: torch.Tensor,
            scale: float, coefficient: float, chunk_size: int, ratio: int, residual: int,
    ) -> torch.Tensor:
        """Save native gradients while retaining Hyper's detached teacher and loss.

        Args:
            ctx: Autograd context holding native gradients and the loss scale.
            query: Student queries [batch, queries, heads, dimension].
            key: Student key bank [batch, keys, dimension].
            weights: Per-query student head weights.
            attention_query: Detached main-attention queries for the teacher.
            main_key: Detached compressed main-attention key bank.
            indices: Selected key IDs, with invalid slots set to -1.
            sinks: Detached per-head main-attention sink logits.
            scale: Main-attention dot-product scale.
            coefficient: Indexer auxiliary loss coefficient.
            chunk_size: Maximum query count for sparse teacher temporaries.
            ratio: Key compression ratio.
            residual: Raw prefix length modulo the compression ratio.

        Returns:
            Scalar KL normalized by this document segment's batch and query count.
        """
        batch, length, heads, dim = query.shape
        kl_grad, _ = _load_kl_ops()
        with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
            target = causal_attention_teacher(
                attention_query, main_key, indices, sinks, scale, ratio, residual, chunk_size,
            )
            options, metadata = _kl_metadata(
                query.device, batch, length, heads, dim, key.shape[1], indices.shape[-1], ratio, residual,
            )
            dq, dk, dw, prediction = kl_grad(
                query.contiguous(), key.unsqueeze(2).contiguous(), weights.float().contiguous(),
                indices.unsqueeze(2).int().contiguous(), target.unsqueeze(2), metadata=metadata, **options,
            )
            log_prediction = _stable_prediction_log(prediction[:, :, 0], query, key, weights, indices, chunk_size)
            loss = (target * (target.clamp_min(torch.finfo(torch.float32).tiny).log() - log_prediction)).sum()
            ctx.save_for_backward(dq, dk.squeeze(2), dw)
            ctx.factor = coefficient / (batch * length)
        return loss * ctx.factor

    @staticmethod
    def backward(ctx: Any, seed: torch.Tensor) -> tuple:
        """Apply the caller's seed in FP32, without differentiating the teacher.

        Args:
            ctx: Autograd context containing native gradients and the loss scale.
            seed: Upstream gradient of the scalar KL output.

        Returns:
            Student Q/K/weight gradients and None for teacher and sequence metadata.
        """
        gradients = tuple(value.float() * (seed * ctx.factor) for value in ctx.saved_tensors)
        return (*gradients, *((None,) * 9))


def _bsnd_sparse_lightning_indexer_kl_loss(
        query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
        attention_query: torch.Tensor, main_key: torch.Tensor, indices: torch.Tensor, sinks: torch.Tensor,
        scale: float, coefficient: float, chunk_size: int, ratio: int,
        segment: tuple[int, int, int, int, int],
) -> torch.Tensor:
    """Fuse one document segment per batch item with the original KL denominator.

    Submit one BSND call for the whole physical batch. The external
    teacher and scalar-loss recovery use ``chunk_size`` for temporaries.

    Args:
        query: Student queries [batch, queries, heads, 128].
        key: Global student key bank [batch, keys, 128].
        weights: Student per-head weights including the model's scaling.
        attention_query: Detached teacher Q [batch, heads, queries, 512].
        main_key: Detached global compressed teacher K bank.
        indices: Global sorted TopK IDs, invalid slots at the tail.
        sinks: Detached main attention sink logits.
        scale: Main attention score scale.
        coefficient: KL coefficient applied once to loss and gradient.
        chunk_size: Maximum sparse teacher queries in a Python chunk.
        ratio: Compression ratio.
        segment: Local Q start/end, document K start/prefix end and residual,
            shared by all physical batch items.
    """
    # Zero-length views attach zero gradients even when the segment has no key.
    total = query[..., :0].float().sum() + key[..., :0].float().sum() + weights[..., :0].float().sum()
    start, end, key_start, key_end, residual = segment
    if key_end <= key_start or start == end:
        return total
    selected = indices[:, start:end]
    selected = torch.where(selected >= 0, selected - key_start, -1)
    if selected.shape[-1] < 512:
        selected = torch.nn.functional.pad(selected, (0, 512-selected.shape[-1]), value=-1)
    loss = _SparseLightningIndexerKLLoss.apply(
        query[:, start:end], key[:, key_start:key_end], weights[:, start:end],
        attention_query[:, :, start:end], main_key[:, key_start:key_end], selected, sinks,
        scale, coefficient, chunk_size, ratio, residual,
    )
    return total + loss * ((end-start) / query.shape[1])


@lru_cache(maxsize=128)
@torch.inference_mode(False)
def _tnd_kl_metadata(plan, device, heads, dim, ratio):
    """Cache KL metadata using the same Q/K partition as native selection."""
    options = plan.indexer_options(device, ratio)
    _, metadata_op = _load_kl_ops()
    metadata = metadata_op(
        heads, 1, dim, topk=512, batch_size=len(plan.segments),
        max_seqlen_q=max(plan.q_lengths), max_seqlen_k=max(plan.k_lengths), **options,
    )
    return options, metadata


class _SparseLightningIndexerKLLossTND(torch.autograd.Function):
    """Fuse the student gradients for several documents, retaining Hyper's teacher."""

    @staticmethod
    def forward(
            ctx: Any, query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
            indices: torch.Tensor, target: torch.Tensor, ratio: int, plan: CSATNDPlan, chunk_size: int,
    ) -> torch.Tensor:
        """Calculate native student gradients and the original scalar KL objective."""
        options, metadata = _tnd_kl_metadata(plan, query.device, query.shape[1], query.shape[2], ratio)
        kernel, _ = _load_kl_ops()
        with torch.autocast(device_type=query.device.type, enabled=False):
            dq, dk, dw, prediction = kernel(
                query, key.unsqueeze(1), weights.float(), indices, target.unsqueeze(1), metadata=metadata, **options,
            )
            logs = []
            q_start = k_start = 0
            for q_length, k_length in zip(plan.q_lengths, plan.k_lengths):
                q_slice = slice(q_start, q_start+q_length)
                logs.append(_stable_prediction_log(
                    prediction[q_slice, 0].unsqueeze(0), query[q_slice].unsqueeze(0),
                    key[k_start:k_start+k_length].unsqueeze(0), weights[q_slice].unsqueeze(0),
                    indices[q_slice, 0].unsqueeze(0), chunk_size,
                )[0])
                q_start += q_length
                k_start += k_length
            log_prediction = torch.cat(logs)
            loss = (target * (target.clamp_min(torch.finfo(torch.float32).tiny).log() - log_prediction)).sum()
        ctx.save_for_backward(dq, dk.squeeze(1), dw)
        return loss

    @staticmethod
    def backward(ctx: Any, seed: torch.Tensor) -> tuple:
        """Apply the upstream scalar seed to all three precomputed student gradients."""
        return (*(value.float() * seed for value in ctx.saved_tensors), None, None, None, None, None)


def sparse_lightning_indexer_kl_loss(
        query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, attention_query: torch.Tensor,
        main_key: torch.Tensor, indices: torch.Tensor, sinks: torch.Tensor, scale: float,
        coefficient: float, chunk_size: int, ratio: int, segments: tuple,
) -> torch.Tensor:
    """Batch packed student KL without changing the detached teacher or normalization.

    Args:
        query: Student queries [batch, queries, heads, 128].
        key: Global student key bank [batch, keys, 128].
        weights: Per-query student head weights including model scaling.
        attention_query: Detached teacher Q [batch, heads, queries, 512].
        main_key: Detached global compressed teacher K bank.
        indices: Sorted global TopK IDs with invalid tail slots -1.
        sinks: Detached main-attention sink logits.
        scale: Main-attention score scale.
        coefficient: KL coefficient, applied once to loss and gradients.
        chunk_size: Maximum queries per external teacher/recovery chunk.
        ratio: Compression ratio.
        segments: Local Q start/end, document K start/prefix end and residual.

    Returns:
        Scalar KL normalized by all local queries, including empty-prefix rows.
        The external teacher retains its per-document bounded computation.
    """
    if len(segments) == 1:
        return _bsnd_sparse_lightning_indexer_kl_loss(
            query, key, weights, attention_query, main_key, indices, sinks,
            scale, coefficient, chunk_size, ratio, segments[0],
        )
    if query.shape[0] != 1:
        raise ValueError("Packed Lightning Indexer KL requires batch size one")
    total = query[..., :0].float().sum() + key[..., :0].float().sum() + weights[..., :0].float().sum()
    plan = csa_tnd_plan(segments, ratio)
    if plan is None:
        return total
    selected = plan.local_indices(indices, padded_width=512)
    targets = []
    cursor = 0
    with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
        for start, end, key_start, key_end, residual in plan.segments:
            targets.append(causal_attention_teacher(
                attention_query[:, :, start:end], main_key[:, key_start:key_end],
                selected[cursor:cursor+end-start, 0].unsqueeze(0), sinks, scale, ratio, residual, chunk_size,
            )[0])
            cursor += end-start
    loss = _SparseLightningIndexerKLLossTND.apply(
        plan.query(query), plan.key(key), plan.query(weights), selected,
        torch.cat(targets).contiguous(), ratio, plan, chunk_size,
    )
    return total + loss * (coefficient / query.shape[1])
