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

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.functional import dsa_indexer, dsa_sparse_attention
from hyper_parallel.components.functional.npu_fusion_attention import (
    resolve_packed_sequence_lengths,
)
from hyper_parallel.components.modules.dsa_attention import (
    _apply_attention_rope,
    _deepseek_v32_dense_attention,
    _restore_attention_projection,
)
from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
)
from hyper_parallel.distributed.context_parallel.collectives import (
    async_cp_allgather_launch,
)
from hyper_parallel.distributed.recipe_spec import inner_wrapper
from hyper_parallel.models.deepseek_v32.adapter.distributed.kv_compact import (
    CompactPlan,
    get_compact_plan,
    interval_select,
)


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
    Dense DSA requires every packed sequence to contribute query tokens on each
    CP rank, so its cumulative query lengths remain strictly increasing.
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


def _global_document_boundaries(
    actual_seq_len: torch.Tensor | Sequence[int] | None,
    kwargs: dict[str, Any],
    *,
    batch_size: int,
    global_seq_len: int,
) -> list[int] | None:
    """Resolve the flat global cumulative document ends, or ``None``.

    Returns the query-side boundaries of the unsharded ``[B, S_global]``
    stream: the coordinate basis both CP-local length projection and KV
    compaction derive their windows from.
    """
    global_tokens = batch_size * global_seq_len
    packed_kwargs = dict(kwargs)
    packed_kwargs["actual_seq_len"] = actual_seq_len
    query_lengths, key_lengths = resolve_packed_sequence_lengths(
        packed_kwargs,
        global_tokens,
        global_tokens,
    )
    if query_lengths is None:
        return None
    if len(query_lengths) != len(key_lengths):
        raise ValueError(
            "DeepSeek-V3.2 DSA requires the same number of packed query and "
            "KV sequences for CP"
        )
    return list(query_lengths)


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
    boundaries = _global_document_boundaries(
        actual_seq_len,
        kwargs,
        batch_size=batch_size,
        global_seq_len=global_seq_len,
    )
    if boundaries is None:
        query_lengths = [
            (row + 1) * global_seq_len
            for row in range(batch_size)
        ]
        key_lengths = list(query_lengths)
    else:
        query_lengths = boundaries
        key_lengths = boundaries

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
    *,
    use_dense: bool,
) -> torch.Tensor:
    """Run the selected sparse or dense DSA CANN path."""
    index_query, index_key, merge_weight = index_states
    absorbed_query, kv_nope, q_rot, k_rot = attention_states
    actual_q_len, actual_kv_len = sequence_lengths
    if use_dense:
        return _deepseek_v32_dense_attention(module, index_states, attention_states, sequence_lengths)

    topk_indices, index_query_tnd, index_key_tnd, merge_weight_tnd = dsa_indexer(  # pylint: disable=not-callable
        index_query,
        index_key,
        merge_weight,
        actual_q_len,
        actual_kv_len,
        module.index_topk,
        backend="cann",
    )
    attn_output, softmax_max, softmax_sum = dsa_sparse_attention(  # pylint: disable=not-callable
        absorbed_query,
        kv_nope,
        q_rot,
        k_rot,
        topk_indices,
        module.scaling,
        actual_q_len,
        actual_kv_len,
        backend="cann",
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


def _cp_compact_plan(
    actual_seq_len: torch.Tensor | Sequence[int] | None,
    kwargs: dict,
    *,
    batch_size: int,
    local_seq_len: int,
    cp_size: int,
    cp_rank: int,
) -> CompactPlan | None:
    """Build this rank's document-aware KV compaction plan.

    Args:
        actual_seq_len: Packed-sequence cumulative lengths of the local
            forward call.
        kwargs: Remaining forward keyword arguments carrying the TND
            metadata keys.
        batch_size: Number of packed sequences in the batch.
        local_seq_len: Per-rank query-block length.
        cp_size: Context-parallel world size.
        cp_rank: This rank's position in the CP group.

    Returns:
        The plan whose windows trim each query block's causal KV prefix
        down to its earliest active document, or ``None`` when the caller
        runs the plain prefix path.

    Raises:
        ValueError: If the packed-sequence document boundaries are missing;
            without documents there is nothing to compact.
    """
    global_seq_len = local_seq_len * cp_size
    boundaries = _global_document_boundaries(
        actual_seq_len,
        kwargs,
        batch_size=batch_size,
        global_seq_len=global_seq_len,
    )
    if boundaries is None:
        raise ValueError(
            "dsa_tnd_compact_kv requires the packed-sequence document "
            "boundaries (actual_seq_len); without documents there is "
            "nothing to compact"
        )
    return get_compact_plan(
        boundaries,
        batch_size=batch_size,
        global_seq_len=global_seq_len,
        local_q_start=cp_rank * local_seq_len,
        local_q_end=(cp_rank + 1) * local_seq_len,
    )


def _select_gathered_prefixes(
    gathered: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    compact_plan: CompactPlan | None,
    prefix_len: int,
    actual_q_len: torch.Tensor,
    actual_kv_len: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize one rank's gathered prefix window and its TND metadata.

    Without a plan the plain causal prefix ``[:, :prefix_len]`` is kept and
    the resolved lengths pass through unchanged. With a plan the window is
    document-aware: the dead head of the causal prefix is dropped and the
    metadata is rebased to the compact window on both sides — the TND
    operators pair the q and KV sub-sequences of one document, so the
    dead-head documents drop from the query lengths together with the KV
    ends (indexer, SFA and KL loss keep one coordinate system,
    document-local Top-K included).

    Args:
        gathered: The all-gathered ``index_key``, ``kv_nope`` and ``k_rot``.
        compact_plan: The compaction plan built for this rank's query
            block, or ``None`` for the plain prefix path.
        prefix_len: Length of the plain causal prefix on this rank.
        actual_q_len: Plain-path query-side cumulative lengths.
        actual_kv_len: Plain-path KV-side cumulative lengths.
        device: Device for the rebased int32 cumulative length tensors.

    Returns:
        The windowed ``(index_key, kv_nope, k_rot)`` plus the TND metadata
        ``(actual_q_len, actual_kv_len)`` matching the window.
    """
    gathered_index_key, gathered_kv_nope, gathered_k_rot = gathered
    if compact_plan is not None:
        index_key = interval_select(gathered_index_key, compact_plan)
        kv_nope = interval_select(gathered_kv_nope, compact_plan)
        k_rot = interval_select(gathered_k_rot, compact_plan)
        actual_q_len = compact_plan.q_ends.to(device=device, dtype=torch.int32)
        actual_kv_len = compact_plan.kv_ends.to(device=device, dtype=torch.int32)
    else:
        index_key = gathered_index_key[:, :prefix_len].contiguous()
        kv_nope = gathered_kv_nope[:, :prefix_len].contiguous()
        k_rot = gathered_k_rot[:, :prefix_len].contiguous()
    return index_key, kv_nope, k_rot, actual_q_len, actual_kv_len


def _project_cp_attention_output(
    module: nn.Module,
    attn_output: torch.Tensor,
    kv_weight: torch.Tensor,
    *,
    batch_size: int,
    local_seq_len: int,
    use_dense: bool,
) -> tuple[torch.Tensor, None]:
    """Project the kernel output back to the model width.

    Args:
        module: The DSA attention module carrying the projection weights.
        attn_output: Kernel output before the output projection.
        kv_weight: The fused KV projection weight for the value path.
        batch_size: Number of packed sequences in the batch.
        local_seq_len: Per-rank query-block length.
        use_dense: Whether the dense warm-up path ran.

    Returns:
        The output-projected hidden states and a ``None`` cache entry.
    """
    if use_dense:
        return module.o_proj(attn_output.flatten(2)), None
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


def _deepseek_v32_dsa_cp_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any | None = None,
    position_ids: torch.Tensor | None = None,
    actual_seq_len: torch.Tensor | Sequence[int] | None = None,
    use_dense: bool | None = None,
    use_compact: bool | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run one DeepSeek DSA layer with all-gathered causal KV prefixes."""
    del attention_mask
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
    use_dense = bool(module.dsa_dense_warm_up) if use_dense is None else use_dense
    use_compact = (
        bool(getattr(module, "dsa_tnd_compact_kv", False)) if use_compact is None else use_compact
    )
    module._validate_forward_inputs(  # pylint: disable=protected-access
        hidden_states,
        None,
        past_key_values,
        position_ids,
        dict(kwargs),
    )

    q_resid, absorbed_query, kv_nope, q_rot, k_rot, kv_weight = (
        module._project_attention_states(hidden_states, use_dense=use_dense)  # pylint: disable=protected-access
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

    compact_plan = (
        _cp_compact_plan(
            actual_seq_len,
            kwargs,
            batch_size=batch_size,
            local_seq_len=local_seq_len,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )
        if use_compact
        else None
    )

    prefix_len = (cp_rank + 1) * local_seq_len
    gathered_index_key = index_key_gather.wait()
    gathered_kv_nope = kv_nope_gather.wait()
    gathered_k_rot = k_rot_gather.wait()
    index_key, kv_nope, k_rot, actual_q_len, actual_kv_len = _select_gathered_prefixes(
        (gathered_index_key, gathered_kv_nope, gathered_k_rot),
        compact_plan,
        prefix_len,
        actual_q_len,
        actual_kv_len,
        device=hidden_states.device,
    )
    attn_output = _run_cp_dsa_kernels(
        module,
        (index_query, index_key, merge_weight),
        (absorbed_query, kv_nope, q_rot, k_rot),
        (actual_q_len, actual_kv_len),
        use_dense=use_dense,
    )
    return _project_cp_attention_output(
        module,
        attn_output,
        kv_weight,
        batch_size=batch_size,
        local_seq_len=local_seq_len,
        use_dense=use_dense,
    )


@inner_wrapper
def deepseek_v32_dsa_attention_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    dense_warmup: bool | None = None,
    kv_compact: bool | None = None,
) -> _ForwardRewriteRequest:
    """Install CP and follow the source DSA dense-warm-up state by default.

    ``dense_warmup`` explicitly overrides the stage flag when supplied.
    Otherwise the forward reads ``target_module.dsa_dense_warm_up`` so training
    code can switch between dense warm-up and sparse DSA without reinstalling
    the wrapper. Set it to ``true`` or ``false`` in the recipe only to force a
    path for the whole run; leave it ``null`` to follow the model's stage flag.

    ``kv_compact`` follows the same pattern for document-aware KV compaction
    (:mod:`.kv_compact`): ``null`` follows the model configuration's
    ``dsa_tnd_compact_kv`` flag, ``true``/``false`` force it for the whole
    run. Compaction requires global packed-sequence boundaries and trims each
    query block's causal KV prefix down to its earliest active document;
    AllGather bytes are unchanged.

    Args:
        target_module: The DSA attention module to wrap.
        mesh: Outer device mesh; unused, kept for the wrapper signature.
        tp_mesh: Tensor-parallel mesh; unused, kept for the wrapper signature.
        cp_mesh: Context-parallel mesh driving the causal-prefix gather.
        ep_mesh: Expert-parallel mesh; unused, kept for the wrapper signature.
        dense_warmup: Force the dense warm-up path for the whole run
            (``true``/``false``), or ``None`` to follow the module's stage
            flag per forward.
        kv_compact: Force document-aware KV compaction for the whole run
            (``true``/``false``), or ``None`` to follow the module's
            ``dsa_tnd_compact_kv`` configuration flag.

    Returns:
        The forward-rewrite request installing the CP forward.
    """
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
        """Execute the CP DSA path using the target module's projections.

        Args:
            hidden_states: Layer input ``[B, S_local, H]``.
            position_embeddings: Rotary embeddings pair, or ``None``.
            attention_mask: Kept for signature compatibility; the CP path
                does not use masks.
            past_key_values: Must be ``None``; the DSA path has no KV cache.
            position_ids: Optional ``[1 or B, S]`` position ids.
            actual_seq_len: Packed-sequence cumulative lengths of the local
                block, or ``None`` to resolve from the TND kwargs.
            **kwargs: Remaining forward arguments carrying the TND metadata
                (``actual_seq_kvlen`` and friends).

        Returns:
            The projected attention output and ``None`` cache entry.
        """
        return _deepseek_v32_dsa_cp_forward(
            target_module,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            position_ids,
            actual_seq_len,
            use_dense=dense_warmup,
            use_compact=kv_compact,
            **kwargs,
        )

    return _ForwardRewriteRequest(
        target_module,
        cp_forward,
        companion_attrs={
            "_hp_deepseek_v32_dsa_cp_mesh": cp_mesh,
            "_hp_deepseek_v32_dsa_cp_config": {
                "cp_size": cp_mesh.size(),
                "dense_warmup_override": dense_warmup,
                "kv_compact_override": kv_compact,
            },
            "_hp_deepseek_v32_dsa_cp_dense_warmup": dense_warmup,
        },
    )


__all__ = [
    "deepseek_v32_dsa_attention_cp_wrapper",
]
