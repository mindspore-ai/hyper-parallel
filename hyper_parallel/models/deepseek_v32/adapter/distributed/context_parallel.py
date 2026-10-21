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
"""DeepSeek-V3.2 DSA Context Parallel wrappers.

The DeepSeek DSA replacement keeps the query shard local, while indexer keys
and compressed MLA keys are gathered from the CP peers.  Every rank only
consumes the causal prefix ending at its local query shard.  The wrapper owns
that communication and length conversion; the high-performance DSA operators
remain in :mod:`hyper_parallel.components.functional`.
"""
# This model-specific adapter orchestrates PyTorch/NPU tensors directly.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.functional.dsa_indexer import dsa_indexer
from hyper_parallel.components.functional.dsa_sparse_attention import (
    dsa_sparse_attention,
)
from hyper_parallel.components.functional.npu_fusion_attention import (
    resolve_packed_sequence_lengths,
)
from hyper_parallel.components.modules.dsa_attention import (
    _apply_attention_rope,
    _restore_attention_projection,
)
from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
)
from hyper_parallel.distributed.context_parallel.collectives import (
    async_cp_allgather_launch,
)
from hyper_parallel.distributed.recipe_spec import inner_wrapper


def _intersection_length(
    start: int,
    end: int,
    window_start: int,
    window_end: int,
) -> int:
    """Return the number of tokens shared by two half-open intervals."""
    return max(min(end, window_end) - max(start, window_start), 0)


def _cp_local_sequence_lengths(
    boundaries: Sequence[int],
    *,
    batch_size: int,
    local_seq_len: int,
    cp_size: int,
    cp_rank: int,
    kind: str,
) -> torch.Tensor:
    """Convert global packed boundaries to this rank's Q/KV cumulative ends.

    ``boundaries`` describe the unsharded ``[B, S_global]`` token stream.  A
    query shard covers ``[offset, offset + local_seq_len)`` in each row, while
    the gathered KV tensor covers the causal prefix ``[0, prefix_len)``.  The
    conversion deliberately retains zero-length entries: they preserve the
    original packed-sequence numbering for ranks that own no query token from
    a particular pack.
    """
    global_seq_len = local_seq_len * cp_size
    prefix_len = (cp_rank + 1) * local_seq_len
    offset = cp_rank * local_seq_len
    global_tokens = batch_size * global_seq_len
    previous = 0
    cumulative: list[int] = []
    total = 0

    for raw_end in boundaries:
        end = int(raw_end)
        if end <= previous or end > global_tokens:
            raise ValueError(
                f"global {kind} sequence boundaries must be strictly increasing "
                f"within [1, {global_tokens}], got {end} after {previous}"
            )
        start = previous
        previous = end
        first_row = start // global_seq_len
        last_row = (end - 1) // global_seq_len
        length = 0
        for row in range(first_row, last_row + 1):
            row_base = row * global_seq_len
            if kind == "query":
                window_start = row_base + offset
                window_end = window_start + local_seq_len
            else:
                window_start = row_base
                window_end = row_base + prefix_len
            length += _intersection_length(start, end, window_start, window_end)
        total += length
        cumulative.append(total)

    expected_total = (
        batch_size * local_seq_len
        if kind == "query"
        else batch_size * prefix_len
    )
    if previous != global_tokens or total != expected_total:
        raise ValueError(
            f"global {kind} sequence boundaries cover {previous} tokens, "
            f"expected {global_tokens}; CP-local projection contains {total} "
            f"tokens, expected {expected_total}"
        )
    return torch.tensor(cumulative, dtype=torch.int32)


def _resolve_cp_sequence_lengths(
    actual_seq_len: torch.Tensor | Sequence[int] | None,
    kwargs: dict[str, Any],
    *,
    batch_size: int,
    local_seq_len: int,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve aliases and produce CP-local cumulative Q and KV lengths."""
    global_seq_len = local_seq_len * cp_size
    global_tokens = batch_size * global_seq_len
    packed_kwargs = dict(kwargs)
    packed_kwargs["actual_seq_len"] = actual_seq_len
    query_lengths, key_lengths = resolve_packed_sequence_lengths(
        packed_kwargs,
        global_tokens,
        global_tokens,
    )
    if query_lengths is None:
        query_lengths = [
            (row + 1) * global_seq_len
            for row in range(batch_size)
        ]
        key_lengths = list(query_lengths)
    elif key_lengths is None:
        # ``resolve_packed_sequence_lengths`` normally rejects this case; keep
        # the guard here so the CP conversion has a single explicit contract.
        raise ValueError("packed DSA attention requires query and KV lengths")

    if len(query_lengths) != len(key_lengths):
        raise ValueError(
            "DeepSeek-V3.2 DSA requires the same number of packed query and "
            "KV sequences for CP"
        )

    query_cp = _cp_local_sequence_lengths(
        query_lengths,
        batch_size=batch_size,
        local_seq_len=local_seq_len,
        cp_size=cp_size,
        cp_rank=cp_rank,
        kind="query",
    ).to(device=device)
    key_cp = _cp_local_sequence_lengths(
        key_lengths,
        batch_size=batch_size,
        local_seq_len=local_seq_len,
        cp_size=cp_size,
        cp_rank=cp_rank,
        kind="KV",
    ).to(device=device)
    return query_cp, key_cp


def _validate_cp_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    local_seq_len: int,
    global_seq_len: int,
) -> None:
    """Validate the shape of an optional Transformers causal mask.

    DSA generates the causal mask from cumulative lengths, so the supplied
    mask is intentionally not forwarded to the custom operators.  Both the
    pre-sharded local form and the Trainer's global form are accepted.
    """
    if attention_mask is None:
        return
    if attention_mask.ndim != 4 or attention_mask.shape[:2] != (batch_size, 1):
        raise ValueError(
            "DeepseekV32DSAAttention CP expects attention_mask with shape "
            "[batch_size, 1, query_length, key_length]"
        )
    query_length, key_length = attention_mask.shape[-2:]
    if query_length not in (local_seq_len, global_seq_len):
        raise ValueError(
            "DeepseekV32DSAAttention CP attention_mask query dimension must "
            f"be {local_seq_len} or {global_seq_len}, got {query_length}"
        )
    if key_length not in (local_seq_len, global_seq_len):
        raise ValueError(
            "DeepseekV32DSAAttention CP attention_mask key dimension must "
            f"be {local_seq_len} or {global_seq_len}, got {key_length}"
        )


def _validate_cp_target(target_module: nn.Module, cp_mesh: Any) -> None:
    """Validate the replacement surface before installing a CP forward."""
    wrapper_name = "deepseek_v32_dsa_attention_cp_wrapper"
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError(f"{wrapper_name} requires an active CP mesh")
    if getattr(target_module, "_hp_deepseek_v32_dsa_cp_config", None) is not None:
        raise RuntimeError(
            "DeepSeek-V3.2 DSA Context Parallel has already been applied to this module"
        )

    required = (
        "_validate_forward_inputs",
        "_project_attention_states",
        "_project_index_states",
        "_apply_auxiliary_loss",
        "o_proj",
    )
    missing = [name for name in required if not hasattr(target_module, name)]
    if missing:
        raise TypeError(
            f"{wrapper_name} requires a DeepseekV32DSAAttention replacement; "
            f"missing attributes: {missing}"
        )


def _run_cp_dsa_kernels(
    module: nn.Module,
    index_states: tuple[torch.Tensor, ...],
    attention_states: tuple[torch.Tensor, ...],
    sequence_lengths: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Run the DSA indexer, sparse attention, and auxiliary-loss path."""
    index_query, index_key, merge_weight = index_states
    absorbed_query, kv_nope, q_rot, k_rot = attention_states
    actual_q_len, actual_kv_len = sequence_lengths
    topk_indices, index_query_tnd, index_key_tnd, merge_weight_tnd = dsa_indexer(
        index_query,
        index_key,
        merge_weight,
        actual_q_len,
        actual_kv_len,
        module.index_topk,
    )
    attn_output, softmax_max, softmax_sum = dsa_sparse_attention(
        absorbed_query,
        kv_nope,
        q_rot,
        k_rot,
        topk_indices,
        module.scaling,
        actual_q_len,
        actual_kv_len,
    )
    return module._apply_auxiliary_loss(  # pylint: disable=protected-access
        attn_output,
        (index_query_tnd, index_key_tnd, merge_weight_tnd),
        attention_states,
        topk_indices,
        (softmax_max, softmax_sum),
        actual_q_len,
        actual_kv_len,
    )


def _deepseek_v32_dsa_cp_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any | None = None,
    position_ids: torch.Tensor | None = None,
    actual_seq_len: torch.Tensor | Sequence[int] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run one DeepSeek DSA layer with all-gathered causal KV prefixes."""
    if hidden_states.ndim != 3:
        raise ValueError(
            "DeepseekV32DSAAttention CP expects hidden_states with shape [B, S, H]"
        )
    batch_size, local_seq_len = hidden_states.shape[:2]
    cp_mesh = getattr(module, "_hp_deepseek_v32_dsa_cp_mesh", None)
    if cp_mesh is None:
        raise RuntimeError("DeepSeek-V3.2 DSA CP forward is missing its CP mesh")
    cp_size = cp_mesh.size()
    cp_rank = cp_mesh.get_local_rank()
    global_seq_len = local_seq_len * cp_size
    _validate_cp_attention_mask(
        attention_mask,
        batch_size=batch_size,
        local_seq_len=local_seq_len,
        global_seq_len=global_seq_len,
    )
    module._validate_forward_inputs(  # pylint: disable=protected-access
        hidden_states,
        None,
        past_key_values,
        position_ids,
        dict(kwargs),
    )

    q_resid, absorbed_query, kv_nope, q_rot, k_rot, kv_weight = (
        module._project_attention_states(hidden_states)  # pylint: disable=protected-access
    )
    q_rot, k_rot = _apply_attention_rope(
        q_rot,
        k_rot,
        position_embeddings,
        interleaved=module.rotary_interleaved,
    )
    index_query, index_key, merge_weight = module._project_index_states(  # pylint: disable=protected-access
        hidden_states,
        q_resid,
        position_embeddings,
    )

    # Launch all three independent gathers before waiting so packed-length
    # conversion overlaps with communication on the NPU stream.
    index_key_gather = async_cp_allgather_launch(index_key, 1, cp_mesh)
    kv_nope_gather = async_cp_allgather_launch(kv_nope, 1, cp_mesh)
    k_rot_gather = async_cp_allgather_launch(k_rot, 1, cp_mesh)
    actual_q_len, actual_kv_len = _resolve_cp_sequence_lengths(
        actual_seq_len,
        kwargs,
        batch_size=batch_size,
        local_seq_len=local_seq_len,
        cp_size=cp_size,
        cp_rank=cp_rank,
        device=hidden_states.device,
    )

    prefix_len = (cp_rank + 1) * local_seq_len
    gathered_index_key = index_key_gather.wait()
    gathered_kv_nope = kv_nope_gather.wait()
    gathered_k_rot = k_rot_gather.wait()
    for name, tensor in (
        ("index_key", gathered_index_key),
        ("kv_nope", gathered_kv_nope),
        ("k_rot", gathered_k_rot),
    ):
        if tensor.shape[1] != global_seq_len:
            raise ValueError(
                f"gathered {name} sequence dimension is {tensor.shape[1]}, "
                f"expected {global_seq_len}"
            )
    index_key = gathered_index_key[:, :prefix_len].contiguous()
    kv_nope = gathered_kv_nope[:, :prefix_len].contiguous()
    k_rot = gathered_k_rot[:, :prefix_len].contiguous()
    attn_output = _run_cp_dsa_kernels(
        module,
        (index_query, index_key, merge_weight),
        (absorbed_query, kv_nope, q_rot, k_rot),
        (actual_q_len, actual_kv_len),
    )
    value_up_weight = kv_weight[:, module.qk_nope_head_dim :].transpose(1, 2)
    attn_output = _restore_attention_projection(
        attn_output,
        value_up_weight,
        num_heads=module.num_heads,
        batch_size=batch_size,
        seq_length=local_seq_len,
        kv_lora_rank=module.kv_lora_rank,
        value_head_dim=module.v_head_dim,
    )
    return module.o_proj(attn_output), None


@inner_wrapper
def deepseek_v32_dsa_attention_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> _ForwardRewriteRequest:
    """Install manual all-gather CP for ``DeepseekV32DSAAttention``."""
    del mesh, tp_mesh, ep_mesh
    _validate_cp_target(target_module, cp_mesh)
    original_forward = target_module.forward

    @wraps(original_forward)
    def cp_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        position_ids: torch.Tensor | None = None,
        actual_seq_len: torch.Tensor | Sequence[int] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Execute the CP DSA path using the target module's projections."""
        return _deepseek_v32_dsa_cp_forward(
            target_module,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            position_ids,
            actual_seq_len,
            **kwargs,
        )

    return _ForwardRewriteRequest(
        target_module,
        cp_forward,
        companion_attrs={
            "_hp_deepseek_v32_dsa_cp_mesh": cp_mesh,
            "_hp_deepseek_v32_dsa_cp_config": {
                "cp_size": cp_mesh.size(),
            },
        },
    )


__all__ = [
    "deepseek_v32_dsa_attention_cp_wrapper",
]
