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


_dsa_tensor_context = contextvars.ContextVar(
    "hyper_dsa_tensor_context", default=None)


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


def _dsa_cp_alltoall(attention_module, attention_functions, context):
    """Configure CP all-to-all for the DSA indexer, attention, and KL loss."""
    original_indexer = attention_module.dsa_lightning_indexer_forward
    original_sparse = attention_functions["dsa_sparse_attention"]
    original_kl = attention_module.SparseLightningIndexerKLLossTrainFunction

    if not getattr(original_indexer, _ULYSSES_WRAPPED_FLAG, False):
        @functools.wraps(original_indexer)
        def index_with_gathered_sequence(
                module: Any, index_query: Tensor, index_key: Tensor,
                merge_weight: Tensor,
                actual_q_len: Any, actual_kv_len: Any) -> Any:
            """Run the DSA indexer on the CP-exchanged sequence layout."""
            index_query = _sequence_to_head(index_query, context)
            merge_weight = _sequence_to_head(
                merge_weight.unsqueeze(-1), context).squeeze(-1)
            index_key = _gather_sequence(index_key, context)
            length = index_query.size(1)
            return original_indexer(
                module, index_query, index_key, merge_weight,
                _global_seq_len(
                    actual_q_len, length, index_query.device),
                _global_seq_len(
                    actual_kv_len, length, index_query.device))

        setattr(index_with_gathered_sequence, _ULYSSES_WRAPPED_FLAG, True)
        attention_module.dsa_lightning_indexer_forward = (
            index_with_gathered_sequence)

    if not getattr(original_sparse, _ULYSSES_WRAPPED_FLAG, False):
        @functools.wraps(original_sparse)
        def sparse_attention_with_gathered_kv(
                module: Any, query: Tensor, key: Tensor, value: Tensor,
                attention_mask: Any, **kwargs: Any) -> tuple[Tensor, Any, Any]:
            """Run DSA sparse attention with head-sharded Q and gathered K/V."""
            del attention_mask
            local_shape = tuple(query.shape)
            query = _sequence_to_head(query, context)
            q_pe = _sequence_to_head(kwargs["q_pe"], context)
            key = _gather_sequence(key, context)
            value = _gather_sequence(value, context)
            k_pe = _gather_sequence(kwargs["k_pe"], context)
            length = query.size(1)
            call_kwargs = kwargs.copy()
            call_kwargs.update(
                q_pe=q_pe, k_pe=k_pe, seq_length=length,
                n_head=query.size(2),
                actual_q_len=_global_seq_len(
                    kwargs.get("actual_q_len"), length, query.device),
                actual_kv_len=_global_seq_len(
                    kwargs.get("actual_kv_len"), length, query.device),
            )
            old_heads = module.num_heads
            module.num_heads = query.size(2)
            try:
                output, softmax_max, softmax_sum = original_sparse(
                    module, query, key, value, None, **call_kwargs)
            finally:
                module.num_heads = old_heads
            if module.training and not module.freeze_dsa:
                _dsa_tensor_context.set(_DSATensorContext(
                    query=query, key=key, q_pe=q_pe, k_pe=k_pe))
            output = _head_to_sequence(output, context)
            if output.shape[:3] != torch.Size(local_shape[:3]):
                raise RuntimeError(
                    f"DSA CP output {tuple(output.shape)} does not restore "
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
                length = saved.query.size(1)
                return original_kl.apply(
                    index_query, index_key, merge_weight, query_tnd, key_tnd,
                    topk_indices, softmax_max, softmax_sum, q_pe_tnd,
                    k_pe_tnd,
                    _global_seq_len(
                        actual_seq_qlen, length, saved.query.device),
                    _global_seq_len(
                        actual_seq_klen, length, saved.query.device),
                    scale, loss_coeff)

        setattr(CPDSAKLLoss, _ULYSSES_WRAPPED_FLAG, True)
        attention_module.SparseLightningIndexerKLLossTrainFunction = (
            CPDSAKLLoss)


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
