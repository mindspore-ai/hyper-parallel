# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""context_parallel.attention: CP attention-layout adaptations.

Model-side CP adaptations on top of the collectives in
context_parallel/collectives.py: MoME halo exchange, MLA/DSA all-to-all
layouts, head/tail load-balanced attention and the D-04 offset-aware
causal mask.

Split out of components/distributed/cp_utils.py in stage 4e.
"""

import contextvars
import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional
import torch
from torch import Tensor
from torch.distributed.nn.functional import all_gather as differentiable_all_gather
from hyper_parallel.distributed.context_parallel.collectives import (
    _ULYSSES_WRAPPED_FLAG,
    _gather_sequence,
    _global_seq_len,
    _head_to_sequence,
    _sequence_to_head,
    _slice_sink,
    flex_cp_allgather,
    platform,
)


@dataclass
class _DSATensorContext:
    query: Tensor
    key: Tensor
    q_pe: Tensor
    k_pe: Tensor
    actual_q_len: Any
    actual_kv_len: Any


_dsa_tensor_context = contextvars.ContextVar(
    "hyper_dsa_tensor_context", default=None)


def _dsa_gather_causal_kv(tensor, actual_seq_len, context):
    """Gather and compact the causal KV context for this rank's queries."""
    gathered = _gather_sequence(tensor, context)
    batch_size, global_length = gathered.shape[:2]
    local_length = tensor.size(1)
    expected_global_length = local_length * context.size
    if global_length != expected_global_length:
        raise RuntimeError(
            f"gathered DSA sequence length {global_length} does not match "
            f"local length {local_length} * CP size {context.size}")

    if actual_seq_len is None:
        boundaries = [
            (batch_index + 1) * global_length
            for batch_index in range(batch_size)
        ]
    elif isinstance(actual_seq_len, Tensor):
        boundaries = [int(value) for value in actual_seq_len.tolist()]
    else:
        boundaries = [int(value) for value in actual_seq_len]
    total_length = batch_size * global_length
    if not boundaries or boundaries[-1] != total_length:
        raise ValueError(
            "DSA CP requires global cumulative sequence lengths ending at "
            f"{total_length}, got {boundaries}")
    if any(right <= left for left, right in zip([0] + boundaries, boundaries)):
        raise ValueError(
            "DSA cumulative sequence lengths must be increasing, got "
            f"{boundaries}")

    sequences = []
    sequence_start = 0
    for sequence_end in boundaries:
        sequences.append((sequence_start, sequence_end))
        sequence_start = sequence_end

    query_lengths = []
    key_lengths = []
    key_ranges = []
    query_total = 0
    key_total = 0
    for batch_index in range(batch_size):
        query_start = batch_index * global_length + context.rank * local_length
        query_end = query_start + local_length
        for sequence_start, sequence_end in sequences:
            local_start = max(sequence_start, query_start)
            local_end = min(sequence_end, query_end)
            if local_start >= local_end:
                continue
            query_total += local_end - local_start
            key_total += local_end - sequence_start
            query_lengths.append(query_total)
            key_lengths.append(key_total)
            key_ranges.append((sequence_start, local_end))

    expected_query_total = batch_size * local_length
    if query_total != expected_query_total:
        raise ValueError(
            "global cumulative sequence lengths do not cover this CP rank's "
            f"{expected_query_total} local query tokens")
    flat_gathered = gathered.flatten(0, 1)
    compact = torch.cat(
        [flat_gathered[start:end] for start, end in key_ranges], dim=0)
    compact = compact.unsqueeze(0).contiguous()
    if isinstance(actual_seq_len, Tensor):
        query_lengths = torch.tensor(
            query_lengths, dtype=torch.int32, device=tensor.device)
        key_lengths = torch.tensor(
            key_lengths, dtype=torch.int32, device=tensor.device)
    return compact, query_lengths, key_lengths


def _mome_cp_halo_exchange(attention_module, context):
    """Configure cross-rank halo exchange for MoME convolution."""
    original = getattr(attention_module, "_apply_mome")
    if getattr(original, _ULYSSES_WRAPPED_FLAG, False):
        return

    @functools.wraps(original)
    def apply_mome_with_halo(
        hidden_states: Tensor,
        mome_mask: Tensor,
        conv: Any,
        use_fused: bool,
    ) -> Tensor:
        """Run MoME convolution with the previous rank's halo rows prepended."""
        halo = conv.kernel_size[0] - 1
        if halo == 0:
            return original(hidden_states, mome_mask, conv, use_fused)
        if hidden_states.size(1) < halo:
            raise ValueError(
                f"local sequence {hidden_states.size(1)} is shorter than "
                f"MOME halo {halo}")
        tails = differentiable_all_gather(
            hidden_states[:, -halo:].contiguous(), group=context.group)
        mask_tail = mome_mask[:, -halo:].to(
            hidden_states.dtype).contiguous()
        masks = differentiable_all_gather(mask_tail, group=context.group)
        if context.rank == 0:
            left_states = tails[-1] * 0
            left_mask = torch.zeros_like(mask_tail, dtype=torch.bool)
        else:
            left_states = tails[context.rank - 1]
            left_mask = masks[context.rank - 1].bool()
        output = original(
            torch.cat((left_states, hidden_states), dim=1),
            torch.cat((left_mask, mome_mask.bool()), dim=1), conv, use_fused)
        return output[:, halo:].contiguous()

    setattr(apply_mome_with_halo, _ULYSSES_WRAPPED_FLAG, True)
    setattr(attention_module, "_apply_mome", apply_mome_with_halo)


def _mla_cp_alltoall(attention_functions, context):
    """Configure CP all-to-all around the MLA backend."""
    original = attention_functions["npu_fa_rescale"]
    if getattr(original, _ULYSSES_WRAPPED_FLAG, False):
        return

    @functools.wraps(original)
    def mla_with_sequence_head_exchange(
            module: Any, query: Tensor, key: Tensor, value: Tensor,
            attention_mask: Any, **kwargs: Any) -> Tensor:
        """Run the MLA backend with Q/K/V exchanged to head-sharded layout."""
        if module.attention_type != "mla":
            return original(
                module, query, key, value, attention_mask, **kwargs)
        if not module.apply_FA_rescale or module.use_fused_sink_fa:
            raise ValueError(
                "MLA CP supports only non-fused npu_fa_rescale")
        local_shape = tuple(query.shape)
        query = _sequence_to_head(query, context)
        key = _sequence_to_head(key, context)
        value = _sequence_to_head(value, context)
        length = query.size(1)
        call_kwargs = kwargs.copy()
        call_kwargs.update(
            seq_length=length,
            n_head=query.size(2),
            actual_q_len=_global_seq_len(
                kwargs.get("actual_q_len"), length, query.device),
            actual_kv_len=_global_seq_len(
                kwargs.get("actual_kv_len"), length, query.device),
            param_sink_key=_slice_sink(
                kwargs.get("param_sink_key"), context),
            param_sink_value=_slice_sink(
                kwargs.get("param_sink_value"), context),
        )
        output = _head_to_sequence(
            original(
                module, query, key, value, attention_mask, **call_kwargs),
            context)
        if output.shape[:3] != torch.Size(local_shape[:3]):
            raise RuntimeError(
                f"MLA CP output {tuple(output.shape)} does not restore "
                f"{local_shape}")
        return output

    setattr(mla_with_sequence_head_exchange, _ULYSSES_WRAPPED_FLAG, True)
    attention_functions["npu_fa_rescale"] = mla_with_sequence_head_exchange


def _mla_cp_allgather(attention_functions, context):
    """Configure local-query KV AllGather around the MLA backend."""
    original = attention_functions["npu_fa_rescale"]
    if getattr(original, _ULYSSES_WRAPPED_FLAG, False):
        return

    @functools.wraps(original)
    def mla_with_gathered_kv(
            module: Any, query: Tensor, key: Tensor, value: Tensor,
            attention_mask: Any, **kwargs: Any) -> Tensor:
        """Run MLA with local full-head queries and compact causal K/V."""
        if module.attention_type != "mla":
            return original(
                module, query, key, value, attention_mask, **kwargs)
        if not module.apply_FA_rescale or module.use_fused_sink_fa:
            raise ValueError(
                "MLA KV AllGather CP supports only non-fused npu_fa_rescale")
        local_shape = tuple(query.shape)
        sequence_lengths = (
            kwargs.get("actual_q_len")
            if kwargs.get("actual_q_len") is not None
            else kwargs.get("actual_kv_len"))
        key, local_q_len, local_kv_len = _dsa_gather_causal_kv(
            key, sequence_lengths, context)
        value, _, _ = _dsa_gather_causal_kv(
            value, sequence_lengths, context)
        call_kwargs = kwargs.copy()
        call_kwargs.update(
            seq_length=query.size(1),
            n_head=query.size(2),
            actual_q_len=local_q_len,
            actual_kv_len=local_kv_len,
        )
        output = original(
            module, query, key, value, attention_mask, **call_kwargs)
        if output.shape[:3] != torch.Size(local_shape[:3]):
            raise RuntimeError(
                f"MLA KV AllGather CP output {tuple(output.shape)} does not "
                f"preserve {local_shape}")
        return output

    setattr(mla_with_gathered_kv, _ULYSSES_WRAPPED_FLAG, True)
    attention_functions["npu_fa_rescale"] = mla_with_gathered_kv


def _dsa_cp_allgather(attention_module, attention_functions, context):
    """Configure local-query CP for the DSA indexer, attention, and KL loss."""
    original_indexer = attention_module.dsa_lightning_indexer_forward
    original_sparse = attention_functions["dsa_sparse_attention"]
    original_kl = attention_module.SparseLightningIndexerKLLossTrainFunction

    if not getattr(original_indexer, _ULYSSES_WRAPPED_FLAG, False):
        @functools.wraps(original_indexer)
        def index_with_local_query(
                module: Any, index_query: Tensor, index_key: Tensor,
                merge_weight: Tensor,
                actual_q_len: Any, actual_kv_len: Any) -> Any:
            """Select TopK for local full-head queries and causal global keys."""
            sequence_lengths = (
                actual_q_len if actual_q_len is not None else actual_kv_len)
            index_key, local_q_len, local_kv_len = _dsa_gather_causal_kv(
                index_key, sequence_lengths, context)
            return original_indexer(
                module, index_query, index_key, merge_weight,
                local_q_len, local_kv_len)

        setattr(index_with_local_query, _ULYSSES_WRAPPED_FLAG, True)
        attention_module.dsa_lightning_indexer_forward = (
            index_with_local_query)

    if not getattr(original_sparse, _ULYSSES_WRAPPED_FLAG, False):
        @functools.wraps(original_sparse)
        def sparse_attention_with_gathered_kv(
                module: Any, query: Tensor, key: Tensor, value: Tensor,
                attention_mask: Any, **kwargs: Any) -> tuple[Tensor, Any, Any]:
            """Run DSA sparse attention with local queries and causal global K/V."""
            del attention_mask
            local_shape = tuple(query.shape)
            try:
                q_pe = kwargs["q_pe"]
                k_pe = kwargs["k_pe"]
            except KeyError as exc:
                raise ValueError(
                    "DSA CP requires q_pe and k_pe keyword arguments") from exc
            sequence_lengths = (
                kwargs.get("actual_q_len")
                if kwargs.get("actual_q_len") is not None
                else kwargs.get("actual_kv_len"))
            key, local_q_len, local_kv_len = _dsa_gather_causal_kv(
                key, sequence_lengths, context)
            value, _, _ = _dsa_gather_causal_kv(
                value, sequence_lengths, context)
            k_pe, _, _ = _dsa_gather_causal_kv(
                k_pe, sequence_lengths, context)
            call_kwargs = kwargs.copy()
            call_kwargs.update(
                q_pe=q_pe, k_pe=k_pe, seq_length=query.size(1),
                n_head=query.size(2),
                actual_q_len=local_q_len,
                actual_kv_len=local_kv_len,
            )
            output, softmax_max, softmax_sum = original_sparse(
                module, query, key, value, None, **call_kwargs)
            if module.training and not module.freeze_dsa:
                _dsa_tensor_context.set(_DSATensorContext(
                    query=query, key=key, q_pe=q_pe, k_pe=k_pe,
                    actual_q_len=local_q_len,
                    actual_kv_len=local_kv_len))
            if output.shape[:3] != torch.Size(local_shape[:3]):
                raise RuntimeError(
                    f"DSA CP output {tuple(output.shape)} does not preserve "
                    f"{local_shape}")
            return output, softmax_max, softmax_sum

        setattr(sparse_attention_with_gathered_kv, _ULYSSES_WRAPPED_FLAG, True)
        attention_functions["dsa_sparse_attention"] = (
            sparse_attention_with_gathered_kv)

    if not getattr(original_kl, _ULYSSES_WRAPPED_FLAG, False):
        class CPDSAKLLoss:
            """Proxy the DSA KL loss with CP-transformed attention inputs."""

            @staticmethod
            def apply(index_query: Any, index_key: Any, merge_weight: Any,
                      query: Any, key: Any,
                      topk_indices: Any, softmax_max: Any, softmax_sum: Any,
                      query_rope: Any,
                      key_rope: Any, actual_seq_qlen: Any, actual_seq_klen: Any,
                      scale: Any,
                      loss_coeff: Any) -> Any:
                """Apply the original KL loss with saved global sequence tensors."""
                saved = _dsa_tensor_context.get()
                if saved is None:
                    return original_kl.apply(
                        index_query, index_key, merge_weight, query, key,
                        topk_indices, softmax_max, softmax_sum, query_rope,
                        key_rope, actual_seq_qlen, actual_seq_klen, scale,
                        loss_coeff)
                _dsa_tensor_context.set(None)
                query_tnd, key_tnd, q_pe_tnd, k_pe_tnd = [
                    tensor.flatten(0, 1) for tensor in
                    (saved.query, saved.key, saved.q_pe, saved.k_pe)]
                return original_kl.apply(
                    index_query, index_key, merge_weight, query_tnd, key_tnd,
                    topk_indices, softmax_max, softmax_sum, q_pe_tnd,
                    k_pe_tnd,
                    saved.actual_q_len, saved.actual_kv_len,
                    scale, loss_coeff)

        setattr(CPDSAKLLoss, _ULYSSES_WRAPPED_FLAG, True)
        attention_module.SparseLightningIndexerKLLossTrainFunction = (
            CPDSAKLLoss)


_dsa_cp_alltoall = _dsa_cp_allgather


def head_tail_load_balance_attention(
        attention_fn: Callable[[Tensor, Tensor, Tensor, dict[str, Any]], Any],
        query: Tensor, key: Tensor, value: Tensor,
        attention_kwargs: dict[str, Any], cp_mesh: Any, *,
        peer_attention_kwargs: Optional[dict[str, Any]] = None) -> Any:
    """Run local-tensor Colossal Head-Tail communication.

    The caller prepares split-specific mask or position metadata. When
    ``peer_attention_kwargs`` is omitted, both attention calls reuse
    ``attention_kwargs`` for backward compatibility.
    """
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError("Head-Tail load balance requires an active CP mesh")
    local_q_len = query.shape[2]
    if local_q_len % 2:
        raise ValueError(
            "Head-Tail load balance requires an even local Q sequence "
            f"length, got {local_q_len}; pad the global sequence to a "
            f"multiple of 2 * cp_size ({2 * cp_mesh.size()})"
        )

    rank_list = list(cp_mesh.rank_list)
    local_rank = rank_list.index(platform.get_rank())
    peer_index = cp_mesh.size() - 1 - local_rank
    peer_rank = rank_list[peer_index]
    half = local_q_len // 2
    query_keep = query.narrow(2, 0, half)
    query_tail = query.narrow(2, half, half)
    query_peer = platform.p2p_exchange(query_tail, peer_rank)
    global_key, global_value = flex_cp_allgather(key, value, 2, cp_mesh)

    def run_half(
            query_half: Tensor,
            call_kwargs: dict[str, Any],
    ) -> Tensor:
        """Run attention for one Head-Tail query half."""
        output = attention_fn(
            query_half, global_key, global_value, call_kwargs
        )
        if not isinstance(output, Tensor):
            raise TypeError(
                "Head-Tail load balance requires the attention callable to "
                f"return a Tensor, got {type(output).__name__}"
            )
        return output

    keep_output = run_half(query_keep, attention_kwargs)
    peer_output = run_half(
        query_peer,
        attention_kwargs if peer_attention_kwargs is None else peer_attention_kwargs,
    )
    tail_output = platform.p2p_exchange(peer_output, peer_rank)
    return platform.cat([keep_output, tail_output], dim=2)


def _cp_offset_causal_mask(q_len: int, kv_len: int, lo: int,
                           device, dtype=torch.bool):
    """D-04: offset-aware causal mask (this rank's Q chunk has global offset lo).

    Attendable positions: j <= lo + i (i is the local Q row index).
    Replaces is_causal=True -- torch SDPA's is_causal is top-left aligned when
    q_len != kv_len (equivalent to assuming Q starts at global position 0), so
    under CP the chunks of rank>0 would be incorrectly masked (G4).
    """
    i = torch.arange(q_len, device=device).view(-1, 1)
    j = torch.arange(kv_len, device=device).view(1, -1)
    return (j <= (lo + i)).to(dtype)
