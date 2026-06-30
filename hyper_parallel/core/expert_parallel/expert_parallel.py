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
"""Expert Parallelism distributed strategies.

Provides token permutation helpers and four parallel styles that compose with
:class:`~hyper_parallel.core.expert_parallel.moe.GroupedExperts`:

- :class:`BaseExpertParallel` — abstract base for EP strategies with
  all-to-all token dispatch/combine.
- :class:`ExpertParallel` — standard EP: each rank owns a shard of experts;
  tokens are routed via differentiable all-to-all.
- :class:`TensorParallel` — TP-only weight sharding for experts with no token
  dispatch; for use when EP degree = 1.
- :class:`ExpertTensorParallel` — combined EP + TP on a 2-D mesh ``[ep, tp]``;
  weights are doubly sharded, dispatch uses the EP sub-mesh.
"""
__all__ = [
    "AllToAllTokenDispatcher",
    "DeredundencyTokenDispatcher",
    "BaseExpertParallel",
    "ExpertParallel",
    "TensorParallel",
    "ExpertTensorParallel",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import (
    distribute_module,
    distribute_tensor,
    _distribute_module_iter_params,
    _distribute_module_new_parameter,
    _distribute_module_param_source,
    _distribute_module_set_param,
)
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.tensor_parallel.style import ParallelStyle
from hyper_parallel.platform import AsyncHandle, get_platform

platform = get_platform()
Module = platform.Module


# ---------------------------------------------------------------------------
# Token permutation helpers
# ---------------------------------------------------------------------------

def _generate_permute_indices(
    tokens_per_expert_group,
    experts_per_rank: int,
    num_ranks: int,
):
    """Generate permutation indices for rank-major → expert-major reordering.

    After all-to-all, received tokens are laid out in rank-major order::

        [rank0·expert0 tokens | rank0·expert1 tokens | ... |
         rank1·expert0 tokens | rank1·expert1 tokens | ...]

    Expert computation requires expert-major order::

        [all tokens for local expert 0 | all tokens for local expert 1 | ...]

    Args:
        tokens_per_expert_group: 1-D integer tensor of shape
            ``[num_ranks * experts_per_rank]``.  Entry ``[r * E + e]`` is the
            number of tokens received from rank ``r`` for local expert ``e``.
        experts_per_rank: Number of experts owned by each rank.
        num_ranks: EP degree (total number of ranks in the EP group).

    Returns:
        Tuple of:

        - ``permuted_indices``: 1-D long tensor of length
          ``total_received_tokens``.  ``permuted_indices[i]`` is the source
          position in the rank-major buffer for destination position ``i`` in
          the expert-major buffer.
        - ``num_tokens_per_expert``: 1-D integer tensor of length
          ``experts_per_rank`` with the token count per local expert.
    """
    counts = tokens_per_expert_group  # [num_ranks * experts_per_rank]

    # num_tokens_per_expert[e] = Σ_r counts[r * E + e]
    counts_2d = counts.view(num_ranks, experts_per_rank)  # [R, E]
    num_tokens_per_expert = counts_2d.sum(dim=0)          # [E]

    # ``total`` must be a host int because ``arange`` needs a scalar size.
    # That single D2H drain is unavoidable.  Everything else stays on
    # device — no per-block ``.item()`` in a loop.
    total = int(num_tokens_per_expert.sum())
    if total == 0:
        return counts.new_zeros(0, dtype=counts.dtype), num_tokens_per_expert

    # ---- Vectorized expert-major permutation, no host stalls -----------
    # Source offsets in the rank-major receive buffer for each (r, e) block.
    src_offsets_rm = counts.cumsum(0) - counts            # [R*E], starts of each block
    # Reorder src offsets to expert-major iteration order: block (e, r).
    src_offsets_em = (
        src_offsets_rm.view(num_ranks, experts_per_rank).T.contiguous().view(-1)
    )                                                     # [E*R]
    # Counts in expert-major iteration order.
    counts_em = counts_2d.T.contiguous().view(-1)         # [E*R]

    # ``repeat_interleave`` expands each block's src start to one entry per
    # token in that block — gives the source position of each output token.
    block_src_starts = src_offsets_em.repeat_interleave(counts_em)   # [total]

    # Destination block starts in expert-major order, then expanded.  The
    # ``arange(total) - dst_block_starts_per_token`` produces 0..n-1 within
    # each block, i.e. the intra-block offset.
    dst_block_starts = counts_em.cumsum(0) - counts_em               # [E*R]
    dst_block_starts_per_token = dst_block_starts.repeat_interleave(counts_em)
    intra = platform.arange(0, total, device=counts.device) - dst_block_starts_per_token

    permuted_indices = (block_src_starts + intra).long()
    return permuted_indices, num_tokens_per_expert


def _permute(x, tokens_per_expert_group, ep_degree: int, num_local_experts: int):
    """Apply rank-major → expert-major permutation to routed tokens.

    Args:
        x: Received token tensor of shape
            ``[sum(tokens_per_expert_group), *feature_dims]``.
        tokens_per_expert_group: 1-D integer tensor of shape
            ``[ep_degree * num_local_experts]`` (output of the first
            all-to-all that exchanges token counts).
        ep_degree: EP group size (number of ranks).
        num_local_experts: Number of experts owned by this rank.

    Returns:
        Tuple of:

        - ``original_shape``: shape of *x* before permutation.
        - ``permuted_x``: tokens reordered to expert-major layout.
        - ``permuted_indices``: permutation indices (needed for
          :func:`_unpermute`).
        - ``num_tokens_per_expert``: token count per local expert.
    """
    original_shape = x.shape
    permuted_indices, num_tokens_per_expert = _generate_permute_indices(
        tokens_per_expert_group, num_local_experts, ep_degree
    )
    # ``x[permuted_indices]`` works for empty indices too (returns a
    # shape-0 tensor with a real grad_fn).  Avoid the early-return with
    # ``new_zeros`` which would produce a leaf tensor without grad_fn and
    # silently break autograd for ranks that happen to receive zero tokens.
    permuted_x = x[permuted_indices]
    return original_shape, permuted_x, permuted_indices, num_tokens_per_expert


def _unpermute(out, original_shape, permuted_indices):
    """Reverse the permutation applied by :func:`_permute`.

    Args:
        out: Expert-major output tensor of shape
            ``[sum(num_tokens_per_expert), *feature_dims]``.
        original_shape: Shape before permutation (from :func:`_permute`).
        permuted_indices: Permutation indices from :func:`_permute`.

    Returns:
        Token tensor restored to the rank-major layout received after
        all-to-all, with shape ``original_shape``.
    """
    # ``result[permuted_indices] = out`` is a differentiable scatter that
    # also handles the empty-index case (no-op assignment, but autograd
    # still connects ``result`` back to ``out``).  Do NOT short-circuit
    # with a bare ``new_zeros`` — that returns a leaf tensor without
    # grad_fn and the downstream combine a2a loses its backward path,
    # which manifests as "element 0 of tensors does not require grad".
    result = out.new_zeros(*original_shape)
    result[permuted_indices] = out
    return result


# ---------------------------------------------------------------------------
# DispatchContext — state shared between token dispatch and combine
# ---------------------------------------------------------------------------

@dataclass
class DispatchContext:
    """Intermediate state between token dispatch and combine.

    Stored in ``module._ep_dispatch_ctx`` for a single forward pass.
    This solves the instance sharing problem when the same ExpertParallel
    style object is applied to multiple layers:

    Example problem (before this fix):
        ep_style = ExpertParallel()
        ep_style.apply(layer1.experts, mesh)  # registers hooks
        ep_style.apply(layer2.experts, mesh)  # reuses same ep_style

        # During forward:
        # layer1.dispatch writes to ep_style._state_stack
        # layer2.dispatch pushes to same stack ← INTERLEAVING
        # layer1.combine pops wrong state (LIFO violation)

    Solution: Store context per-module, not per-style-instance.

    Built by :meth:`AllToAllTokenDispatcher.dispatch` and consumed by
    :meth:`AllToAllTokenDispatcher.combine`.  The caller
    (e.g. :class:`ExpertParallel`) stores this on the module between the
    paired dispatch/combine calls.
    """

    input_splits: List[int]
    output_splits: List[int]
    input_shape: Tuple[int, ...]
    permuted_indices: Any


@dataclass
class DeredundencyDispatchContext(DispatchContext):
    """State shared by deredundency dispatch and combine.

    The inherited split and permutation fields describe the inner-EP
    all-to-all.  The extra fields describe the OEP shared token view and the
    whiteboard scatter used before the final reduce-scatter combine.
    """

    dispatch_indices: Optional[object] = None
    router_coeff: Optional[object] = None
    gathered_shape: Optional[tuple] = None
    oep_size: int = 1



@dataclass(frozen=True)
class _DeredundencyMeshInfo:
    """Resolved mesh metadata for two-stage deredundency token exchange."""

    oep_group: object
    iep_group: object
    oep_size: int
    iep_size: int
    outer_rank: int
    inner_rank: int


def _get_deredundency_mesh_info(device_mesh: DeviceMesh) -> _DeredundencyMeshInfo:
    """Resolve ``oep`` / ``iep`` groups from a 1-D or 2-D EP mesh."""
    ndim = getattr(device_mesh, "ndim", 1)
    if not isinstance(ndim, int):
        ndim = 1
    if ndim == 1:
        return _DeredundencyMeshInfo(
            oep_group=None,
            iep_group=device_mesh.get_group(),
            oep_size=1,
            iep_size=device_mesh.size(),
            outer_rank=0,
            inner_rank=device_mesh.get_local_rank(),
        )
    if ndim != 2:
        raise ValueError(
            "DeredundencyTokenDispatcher expects a 1-D EP mesh or a 2-D "
            f"[oep, iep] EP mesh, but got ndim={ndim}."
        )

    mesh_dim_names = getattr(device_mesh, "mesh_dim_names", None) or ()
    oep_dim = mesh_dim_names.index("oep") if "oep" in mesh_dim_names else 0
    iep_dim = mesh_dim_names.index("iep") if "iep" in mesh_dim_names else 1
    if oep_dim == iep_dim:
        raise ValueError("DeredundencyTokenDispatcher requires distinct oep and iep mesh dimensions.")

    return _DeredundencyMeshInfo(
        oep_group=device_mesh.get_group(oep_dim),
        iep_group=device_mesh.get_group(iep_dim),
        oep_size=device_mesh.size(oep_dim),
        iep_size=device_mesh.size(iep_dim),
        outer_rank=device_mesh.get_local_rank(oep_dim),
        inner_rank=device_mesh.get_local_rank(iep_dim),
    )


def _generate_deredundency_dispatch_indices(
    tokens_per_expert_by_source,
    expert_start: int,
    iep_size: int,
    num_local_experts: int,
):
    """Generate gather-view indices ordered by IEP destination rank.

    ``tokens_per_expert_by_source`` is shaped ``[oep_size, num_experts]`` and
    describes each source rank's expert-major routed buffer after the OEP
    all-gather.  The returned indices select the current outer expert range
    and order it as ``[iep_dst, local_expert, oep_source]`` so each IEP
    destination chunk keeps local-expert blocks contiguous for the later
    rank-major → expert-major permutation.
    """
    oep_size = tokens_per_expert_by_source.shape[0]
    experts_per_outer = iep_size * num_local_experts
    expert_end = expert_start + experts_per_outer

    source_totals = tokens_per_expert_by_source.sum(dim=1)
    source_offsets = source_totals.cumsum(0) - source_totals
    expert_offsets = (
        tokens_per_expert_by_source.cumsum(dim=1)
        - tokens_per_expert_by_source
        + source_offsets.view(oep_size, 1)
    )

    selected_counts = tokens_per_expert_by_source[:, expert_start:expert_end].view(
        oep_size, iep_size, num_local_experts,
    )
    selected_offsets = expert_offsets[:, expert_start:expert_end].view(
        oep_size, iep_size, num_local_experts,
    )
    counts_by_destination = selected_counts.permute(1, 2, 0).contiguous()
    offsets_by_destination = selected_offsets.permute(1, 2, 0).contiguous()

    block_counts = counts_by_destination.view(-1)
    token_counts_by_destination_expert = selected_counts.sum(dim=0).contiguous().view(-1)
    total = int(block_counts.sum())
    if total == 0:
        return block_counts.new_zeros(0, dtype=block_counts.dtype).long(), token_counts_by_destination_expert

    block_starts = offsets_by_destination.view(-1).repeat_interleave(block_counts)
    block_offsets = block_counts.cumsum(0) - block_counts
    block_offsets_per_token = block_offsets.repeat_interleave(block_counts)
    intra = platform.arange(0, total, device=tokens_per_expert_by_source.device) - block_offsets_per_token

    return (block_starts + intra).long(), token_counts_by_destination_expert


def _scale_by_router_coeff(tokens, router_coeff):
    """Scale routed expert outputs by optional router coefficients."""
    if router_coeff is None:
        return tokens
    if router_coeff.shape[0] != tokens.shape[0]:
        raise ValueError(
            "router_coeff length must match routed token count, got "
            f"{router_coeff.shape[0]} and {tokens.shape[0]}."
        )
    coeff = router_coeff
    if len(coeff.shape) == 1 and len(tokens.shape) > 1:
        coeff = coeff.reshape((-1,) + (1,) * (len(tokens.shape) - 1))
    return tokens * coeff


def _scatter_add_first_dim(src, indices, output_shape):
    """Scatter-add rows of ``src`` into a zero tensor along dim 0."""
    result = src.new_zeros(*output_shape)
    if len(src.shape) == 1:
        scatter_indices = indices
    else:
        scatter_indices = indices.reshape((-1,) + (1,) * (len(src.shape) - 1)).expand(
            -1, *src.shape[1:],
        )
    if hasattr(result, "scatter_add"):
        return result.scatter_add(0, scatter_indices, src)
    if hasattr(result, "index_add"):
        return result.index_add(0, indices, src)
    raise RuntimeError(
        "DeredundencyTokenDispatcher.combine requires tensor scatter_add or "
        "index_add support for exdispatch_idx accumulation."
    )


class _DeredundencyCombineHandle(AsyncHandle):
    """Async handle that finishes deredundency combine post-processing."""

    def __init__(
        self,
        async_tensor: object,
        mesh_info: _DeredundencyMeshInfo,
        ctx: DeredundencyDispatchContext,
    ) -> None:
        super().__init__(async_tensor)
        self._mesh_info = mesh_info
        self._ctx = ctx
        self._combined: Optional[object] = None

    def wait(self) -> object:
        """Wait for IEP a2a, then finish OEP scatter/reduce combine once."""
        if self._combined is None:
            outer_output = super().wait()
            weighted_output = _scale_by_router_coeff(outer_output, self._ctx.router_coeff)
            combine_whiteboard = _scatter_add_first_dim(
                weighted_output,
                self._ctx.dispatch_indices,
                self._ctx.gathered_shape,
            )
            if self._ctx.oep_size == 1:
                self._combined = combine_whiteboard
            else:
                self._combined = platform.differentiable_reduce_scatter(
                    combine_whiteboard,
                    self._ctx.oep_size,
                    0,
                    "sum",
                    self._mesh_info.oep_group,
                )
        return self._combined


# ---------------------------------------------------------------------------
# BaseExpertParallel — abstract base for all-to-all EP strategies
# ---------------------------------------------------------------------------

class BaseExpertParallel(ParallelStyle, ABC):
    """Abstract base class for Expert Parallel strategies with token dispatch.

    Subclasses implement :meth:`_partition_fn`, :meth:`_token_dispatch`, and
    :meth:`_token_combine`; this class wires them into :func:`distribute_module`.
    """

    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:
        """Apply EP sharding and dispatch/combine hooks to *module*.

        Args:
            module: A :class:`~hyper_parallel.core.expert_parallel.moe.GroupedExperts`
                instance to shard.
            device_mesh: Device mesh for this EP strategy.

        Returns:
            The module with distributed parameters and dispatch/combine hooks.
        """
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            self._token_dispatch,
            self._token_combine,
        )

    @abstractmethod
    def _partition_fn(
        self, name: str, module: Module, device_mesh: DeviceMesh
    ) -> None:
        """Shard module parameters according to this strategy.

        Args:
            name: Submodule name.
            module: The module whose parameters are being sharded.
            device_mesh: Device mesh for this EP strategy.
        """

    @abstractmethod
    def _token_dispatch(self, module: Module, inputs, device_mesh: DeviceMesh):
        """Pre-hook: route input tokens to their assigned ranks.

        Args:
            module: The ``GroupedExperts`` module.
            inputs: Forward inputs tuple.
            device_mesh: Device mesh for this EP strategy.

        Returns:
            Transformed inputs for local expert computation.
        """

    @abstractmethod
    def _token_combine(self, module: Module, routed_output, device_mesh: DeviceMesh):
        """Post-hook: gather expert outputs back to the originating ranks.

        Args:
            module: The ``GroupedExperts`` module.
            routed_output: Expert output tensor in expert-major order.
            device_mesh: Device mesh for this EP strategy.

        Returns:
            Token tensor in the original token-major layout.
        """


# ---------------------------------------------------------------------------
# AllToAllTokenDispatcher — token dispatch/combine via all-to-all
# ---------------------------------------------------------------------------

class AllToAllTokenDispatcher:
    """Token dispatch and combine via all-to-all for expert parallelism.

    Provides :meth:`dispatch` and :meth:`combine` as static methods that
    receive and return a :class:`DispatchContext` object.  This decouples
    the all-to-all token routing logic from the parallel style class so
    that it can be reused or tested independently.

    Callers (e.g. :class:`ExpertParallel`) are responsible for storing the
    context between the paired dispatch/combine calls.
    """

    @staticmethod
    def dispatch(module: Module, inputs: tuple, device_mesh: DeviceMesh) -> tuple:
        """Dispatch tokens to their assigned ranks via all-to-all.

        Called as an ``input_fn`` hook by :func:`distribute_module`.  Receives
        the module's forward inputs and returns transformed inputs.

        Args:
            module: The ``GroupedExperts`` module.
            inputs: Tuple ``(routed_input, num_tokens_per_expert)`` where
                ``routed_input`` has shape ``[total_tokens, dim]`` and
                ``num_tokens_per_expert`` has shape ``[num_experts]``.
            device_mesh: EP device mesh (1-D).

        Returns:
            Tuple ``(permuted_local_input, local_token_counts, ctx)`` —
            the first two elements are the transformed inputs for local
            expert computation; *ctx* is a :class:`DispatchContext`
            carrying the updated state to be stored by the caller.
        """
        del module  # module unused, kept for API consistency
        routed_input, num_tokens_per_expert = inputs[0], inputs[1]
        ep_group = device_mesh.get_group()
        ep_size = device_mesh.size()
        num_local_experts = num_tokens_per_expert.shape[0] // ep_size

        # --- Step 1: exchange token counts (no gradient needed) ---
        # Each rank needs to know how many tokens it will receive from every
        # other rank (for each local expert).  Uses ``async_op=True`` + an
        # explicit ``handle.wait()`` rather than ``async_op=False`` because
        # the implicit cross-stream sync is NCCL-only; on HCCL the compute
        # stream may read ``counts_out`` before the collective write is
        # visible, producing garbage values that blow up the downstream
        # ``torch.empty(sum(output_splits), ...)`` allocation.
        counts_out, handle = platform.all_to_all_single(
            num_tokens_per_expert,
            output_shape=[num_tokens_per_expert.shape[0]],
            group=ep_group,
            async_op=True,
        )
        if handle is not None:
            handle.wait()
        # counts_out shape: [ep_size * num_local_experts]
        # counts_out[r * num_local_experts + e] = tokens from rank r for expert e

        # --- Step 2: compute input / output splits ---
        # input_splits[r] = tokens this rank sends to rank r
        # output_splits[r] = tokens this rank receives from rank r
        # Reshape to [ep_size, num_local_experts] and sum per rank on device;
        # a single ``tolist()`` drains the rank-sum vector to host, replacing
        # ``2 * ep_size`` scalar ``int()`` D2H syncs with 2.
        input_splits = num_tokens_per_expert.view(ep_size, num_local_experts).sum(dim=1).tolist()
        output_splits = counts_out.view(ep_size, num_local_experts).sum(dim=1).tolist()

        # --- Step 3: exchange actual tokens (differentiable) ---
        dispatched = platform.differentiable_all_to_all_single(
            routed_input, input_splits, output_splits, group=ep_group,
        )

        # --- Step 4: rank-major → expert-major permutation ---
        input_shape, permuted, permuted_indices, local_counts = _permute(
            dispatched, counts_out, ep_size, num_local_experts
        )

        # Build dispatch context for combine step.
        # Caller (e.g., ExpertParallel._token_dispatch) is responsible for storing
        # this context and passing it to combine(). This decouples dispatch/combine
        # from module state and solves the instance sharing problem.
        ctx = DispatchContext(
            input_splits=input_splits,
            output_splits=output_splits,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
        )

        return permuted, local_counts, ctx

    @staticmethod
    def combine(module: Module, routed_output: object, device_mesh: DeviceMesh, ctx: DispatchContext) -> object:
        """Gather expert outputs back to the originating ranks via all-to-all.

        Called as an ``output_fn`` hook by :func:`distribute_module`.
        Receives dispatch context from the caller (previously returned by dispatch).

        Args:
            module: The ``GroupedExperts`` module (unused, for API consistency).
            routed_output: Expert output tensor in expert-major order,
                shape ``[sum(local_counts), dim]``.
            device_mesh: EP device mesh (1-D).
            ctx: :class:`DispatchContext` previously returned by
                :meth:`dispatch`.

        Returns:
            Token tensor in the original token-major layout,
            shape ``[sum(input_splits), dim]``.
        """
        del module  # module not used, kept for API consistency
        ep_group = device_mesh.get_group()

        # expert-major → rank-major
        unpermuted = _unpermute(routed_output, ctx.input_shape, ctx.permuted_indices)

        # reverse all-to-all (output/input splits are swapped)
        combined = platform.differentiable_all_to_all_single(
            unpermuted,
            ctx.output_splits,   # was output, now becomes input
            ctx.input_splits,    # was input, now becomes output
            group=ep_group,
        )
        return combined

    @staticmethod
    def combine_start(routed_output, device_mesh, ctx):
        """Launch async combine all-to-all without waiting for completion.

        Splits the combine into two phases so that the caller can overlap
        the a2a communication with independent computation (e.g. a shared
        expert forward pass).  The caller must later call
        :meth:`combine_wait` or ``handle.wait()`` to obtain the final
        result.

        Step 1 (synchronous, local): expert-major → rank-major unpermute.
        Step 2 (asynchronous, cross-rank): reverse all-to-all.

        Args:
            routed_output: Expert output tensor in expert-major order,
                shape ``[sum(local_counts), dim]``.
            device_mesh: EP device mesh (1-D).
            ctx: :class:`DispatchContext` previously returned by
                :meth:`dispatch`.

        Returns:
            :class:`AsyncHandle` carrying the state needed by
            :meth:`combine_wait`.
        """
        ep_group = device_mesh.get_group()

        # expert-major → rank-major (local, no communication)
        unpermuted = _unpermute(routed_output, ctx.input_shape, ctx.permuted_indices)

        # async reverse all-to-all (output/input splits are swapped)
        combined_async = platform.differentiable_all_to_all_single_async(
            unpermuted,
            ctx.output_splits,
            ctx.input_splits,
            group=ep_group,
        )

        return AsyncHandle(combined_async)

    @staticmethod
    def combine_wait(handle):
        """Wait for the async combine all-to-all to complete.

        Args:
            handle: :class:`AsyncHandle` returned by :meth:`combine_start`.

        Returns:
            Combined tensor in the original token-major layout.
        """
        return handle.wait()


# ---------------------------------------------------------------------------
# DeredundencyTokenDispatcher — token dispatch via OEP all-gather + IEP all-to-all
# ---------------------------------------------------------------------------

class DeredundencyTokenDispatcher:
    """Token dispatch/combine via OEP all-gather plus IEP all-to-all.

    This dispatcher keeps the same public contract as
    :class:`AllToAllTokenDispatcher`, but decomposes the global EP all-to-all
    into the deredundency flow described in
    ``docs/moe_alltoall_deredundency_token_permutation.md``:

    1. Form a shared token/count view across the OEP group.
    2. Select only the current outer expert range.
    3. Send selected tokens to concrete local-expert ranks inside the IEP
       group.
    4. Sort received tokens into local expert-major order.

    For a 2-D mesh, dimension ``"oep"`` / ``0`` is the outer group and
    ``"iep"`` / ``1`` is the inner group.  A 1-D mesh is treated as
    ``oep_size == 1`` and degenerates to the standard all-to-all data flow.
    """

    @staticmethod
    def _oep_gather_for_dispatch(
        num_tokens_per_expert,
        routed_input,
        router_coeff,
        mesh_info: _DeredundencyMeshInfo,
    ) -> tuple:
        """All-gather token counts and routed input across the OEP group.

        Args:
            num_tokens_per_expert: Token count per expert ``[num_experts]``.
            routed_input: Routed token tensor ``[total_tokens, dim]``.
            router_coeff: Optional router coefficients ``[total_tokens]``.
            mesh_info: Resolved OEP/IEP mesh descriptor.

        Returns:
            Tuple ``(gathered_counts, gathered_routed, gathered_router_coeff)``
            where ``gathered_counts`` has shape ``[oep_size, num_experts]``,
            ``gathered_routed`` has shape ``[oep_size * total_tokens, dim]``,
            and ``gathered_router_coeff`` is the gathered coefficients or None.

        Raises:
            ValueError: If routed token counts differ across OEP ranks.
        """
        if mesh_info.oep_size == 1:
            gathered_counts = num_tokens_per_expert.view(1, num_tokens_per_expert.shape[0])
            return gathered_counts, routed_input, router_coeff

        gathered_counts, handle = platform.all_gather_single(
            num_tokens_per_expert,
            output_shape=[mesh_info.oep_size * num_tokens_per_expert.shape[0]],
            group=mesh_info.oep_group,
            async_op=True,
        )
        if handle is not None:
            handle.wait()
        gathered_counts = gathered_counts.view(mesh_info.oep_size, num_tokens_per_expert.shape[0])
        source_token_totals = gathered_counts.sum(dim=1).tolist()
        if any(total != routed_input.shape[0] for total in source_token_totals):
            raise ValueError(
                "DeredundencyTokenDispatcher requires equal routed token "
                "counts within each OEP group because the shared token view "
                f"uses all-gather, got totals {source_token_totals}."
            )
        gathered_routed = platform.differentiable_all_gather_concat(
            routed_input, mesh_info.oep_group, mesh_info.oep_size, 0,
        )
        if router_coeff is None:
            gathered_router_coeff = None
        else:
            gathered_router_coeff = platform.differentiable_all_gather_concat(
                router_coeff, mesh_info.oep_group, mesh_info.oep_size, 0,
            )
        return gathered_counts, gathered_routed, gathered_router_coeff

    @staticmethod
    def dispatch(module: Module, inputs: tuple, device_mesh: DeviceMesh) -> tuple:
        """Dispatch tokens using OEP all-gather and IEP all-to-all.

        Args:
            module: The ``GroupedExperts`` module (unused here).
            inputs: Tuple ``(routed_input, num_tokens_per_expert)`` where
                ``routed_input`` has shape ``[total_tokens, dim]`` and
                ``num_tokens_per_expert`` has shape ``[num_experts]``.
            device_mesh: 1-D EP mesh or 2-D ``[oep, iep]`` EP mesh.

        Returns:
            Tuple ``(permuted_local_input, local_token_counts, ctx)`` with the
            same meaning as :meth:`AllToAllTokenDispatcher.dispatch`.

        Raises:
            ValueError: If the expert count is not divisible by the full EP
                size represented by the deredundency mesh.
        """
        del module
        routed_input, num_tokens_per_expert = inputs[0], inputs[1]
        router_coeff = inputs[2] if len(inputs) > 2 else None
        if router_coeff is not None and router_coeff.shape[0] != routed_input.shape[0]:
            raise ValueError(
                "router_coeff length must match routed_input token count, got "
                f"{router_coeff.shape[0]} and {routed_input.shape[0]}."
            )
        mesh_info = _get_deredundency_mesh_info(device_mesh)
        ep_size = mesh_info.oep_size * mesh_info.iep_size
        if num_tokens_per_expert.shape[0] % ep_size != 0:
            raise ValueError(
                "num_tokens_per_expert length must be divisible by the full "
                f"EP size {ep_size}, got {num_tokens_per_expert.shape[0]}."
            )
        num_local_experts = num_tokens_per_expert.shape[0] // ep_size
        experts_per_outer = mesh_info.iep_size * num_local_experts
        expert_start = mesh_info.outer_rank * experts_per_outer

        gathered_counts, gathered_routed, gathered_router_coeff = (
            DeredundencyTokenDispatcher._oep_gather_for_dispatch(
                num_tokens_per_expert, routed_input, router_coeff, mesh_info,
            )
        )

        dispatch_indices, node_counts_per_expert = _generate_deredundency_dispatch_indices(
            gathered_counts,
            expert_start,
            mesh_info.iep_size,
            num_local_experts,
        )
        iep_input_splits = node_counts_per_expert.view(mesh_info.iep_size, num_local_experts).sum(dim=1).tolist()

        iep_counts_out, handle = platform.all_to_all_single(
            node_counts_per_expert,
            output_shape=[node_counts_per_expert.shape[0]],
            group=mesh_info.iep_group,
            async_op=True,
        )
        if handle is not None:
            handle.wait()
        iep_output_splits = iep_counts_out.view(mesh_info.iep_size, num_local_experts).sum(dim=1).tolist()

        outer_routed_input = gathered_routed[dispatch_indices]
        outer_router_coeff = (
            None if gathered_router_coeff is None else gathered_router_coeff[dispatch_indices]
        )
        dispatched = platform.differentiable_all_to_all_single(
            outer_routed_input,
            iep_input_splits,
            iep_output_splits,
            group=mesh_info.iep_group,
        )

        input_shape, permuted, permuted_indices, local_counts = _permute(
            dispatched, iep_counts_out, mesh_info.iep_size, num_local_experts,
        )
        ctx = DeredundencyDispatchContext(
            input_splits=iep_input_splits,
            output_splits=iep_output_splits,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            dispatch_indices=dispatch_indices,
            router_coeff=outer_router_coeff,
            gathered_shape=gathered_routed.shape,
            oep_size=mesh_info.oep_size,
        )
        return permuted, local_counts, ctx

    @staticmethod
    def combine(module: Module, routed_output: object, device_mesh: DeviceMesh,
                ctx: DeredundencyDispatchContext) -> object:
        """Gather expert outputs back to the originating ranks.

        Args:
            module: The ``GroupedExperts`` module (unused).
            routed_output: Expert output tensor in expert-major order.
            device_mesh: 1-D EP mesh or 2-D ``[oep, iep]`` EP mesh.
            ctx: Context returned by :meth:`dispatch`.

        Returns:
            Token tensor in the original source-rank routed order.
        """
        del module
        mesh_info = _get_deredundency_mesh_info(device_mesh)
        DeredundencyTokenDispatcher._validate_combine_mesh(mesh_info, ctx)

        unpermuted = _unpermute(routed_output, ctx.input_shape, ctx.permuted_indices)
        outer_output = platform.differentiable_all_to_all_single(
            unpermuted,
            ctx.output_splits,
            ctx.input_splits,
            group=mesh_info.iep_group,
        )

        weighted_output = _scale_by_router_coeff(outer_output, ctx.router_coeff)
        combine_whiteboard = _scatter_add_first_dim(
            weighted_output, ctx.dispatch_indices, ctx.gathered_shape,
        )
        if ctx.oep_size == 1:
            return combine_whiteboard

        return platform.differentiable_reduce_scatter(
            combine_whiteboard,
            ctx.oep_size,
            0,
            "sum",
            mesh_info.oep_group,
        )

    @staticmethod
    def _validate_combine_mesh(
        mesh_info: _DeredundencyMeshInfo,
        ctx: DeredundencyDispatchContext,
    ) -> None:
        """Validate that dispatch context and combine mesh are compatible."""
        if mesh_info.oep_size != ctx.oep_size:
            raise ValueError(
                "DeredundencyTokenDispatcher.combine received a context for "
                f"oep_size={ctx.oep_size}, but the mesh resolves to oep_size={mesh_info.oep_size}."
            )

    @staticmethod
    def combine_start(
        routed_output: object,
        device_mesh: DeviceMesh,
        ctx: DeredundencyDispatchContext,
    ) -> AsyncHandle:
        """Launch async IEP combine all-to-all and defer deredundency post-processing.

        The local expert-major → rank-major unpermute is performed
        synchronously.  The reverse IEP all-to-all is launched asynchronously,
        and :meth:`combine_wait` finishes router weighting, whiteboard
        scatter-add, and optional OEP reduce-scatter after the async output is
        materialised.

        Args:
            routed_output: Expert output tensor in expert-major order.
            device_mesh: 1-D EP mesh or 2-D ``[oep, iep]`` EP mesh.
            ctx: Context returned by :meth:`dispatch`.

        Returns:
            :class:`AsyncHandle` carrying the pending IEP a2a and deredundency
            combine state.
        """
        mesh_info = _get_deredundency_mesh_info(device_mesh)
        DeredundencyTokenDispatcher._validate_combine_mesh(mesh_info, ctx)

        unpermuted = _unpermute(routed_output, ctx.input_shape, ctx.permuted_indices)
        outer_output_async = platform.differentiable_all_to_all_single_async(
            unpermuted,
            ctx.output_splits,
            ctx.input_splits,
            group=mesh_info.iep_group,
        )
        return _DeredundencyCombineHandle(outer_output_async, mesh_info, ctx)

    @staticmethod
    def combine_wait(handle: AsyncHandle) -> object:
        """Wait for async deredundency combine and return the final tensor.

        Args:
            handle: :class:`AsyncHandle` returned by :meth:`combine_start`.

        Returns:
            Token tensor in the original source-rank routed order.
        """
        return handle.wait()


_TOKEN_DISPATCHERS = {
    "all_to_all": AllToAllTokenDispatcher,
    "deredundency": DeredundencyTokenDispatcher,
}


def _resolve_token_dispatcher(token_dispatcher: str):
    """Resolve a token dispatcher name to its implementation class."""
    try:
        return _TOKEN_DISPATCHERS[token_dispatcher]
    except KeyError as exc:
        supported = "', '".join(sorted(_TOKEN_DISPATCHERS))
        raise ValueError(
            f"token_dispatcher must be one of '{supported}', got {token_dispatcher!r}."
        ) from exc


def _get_flattened_ep_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """Return a 1-D EP mesh, flattening a 2-D deredundency mesh if needed."""
    if getattr(device_mesh, "ndim", 1) == 1:
        return device_mesh
    mesh_dim_names = getattr(device_mesh, "mesh_dim_names", None) or ()
    if "ep" in mesh_dim_names or "ep" in device_mesh.get_flatten_mapping():
        return device_mesh["ep"]
    if set(mesh_dim_names) == {"oep", "iep"}:
        return device_mesh.flatten("ep")
    raise ValueError(
        "Deredundency ExpertParallel expects a 1-D EP mesh or a 2-D "
        "[oep, iep] mesh when partitioning expert weights."
    )


# ---------------------------------------------------------------------------
# ExpertParallel — standard all-to-all EP
# ---------------------------------------------------------------------------

class ExpertParallel(BaseExpertParallel):
    """Expert Parallel: shard experts across ranks via all-to-all token routing.

    Applies :meth:`apply` to a :class:`GroupedExperts` module:

    1. **Partition** — distributes expert weights on dim 0 (``Shard(0)``) so
       each rank holds ``num_experts // ep_degree`` local experts.
    2. **Token dispatch** (forward pre-hook) — two-step all-to-all:
       a. Exchange token counts (non-differentiable).
       b. Exchange actual tokens (differentiable, gradient flows back).
       Followed by rank-major → expert-major permutation.
    3. **Token combine** (forward post-hook) — expert-major → rank-major
       unpermute, then reverse all-to-all (differentiable).

    All collectives use ``platform.differentiable_all_to_all_single`` /
    ``platform.all_to_all_single`` — no direct ``torch.distributed`` calls.

    The token dispatcher is selectable. ``"all_to_all"`` uses
    :class:`AllToAllTokenDispatcher`; ``"deredundency"`` uses
    :class:`DeredundencyTokenDispatcher`.

    Args:
        token_dispatcher: Token dispatch strategy. Supported values are
            ``"all_to_all"`` and ``"deredundency"``.
        async_combine: When ``True``, the combine all-to-all is launched
            asynchronously so that the caller (e.g. :class:`MoE`) can
            overlap it with shared-expert computation.  When ``False``
            (default), combine is fully synchronous — no overlap, identical
            to the baseline.

    Example::
        >>> ep_style = ExpertParallel()
        >>> sharded_experts = ep_style.apply(experts_module, ep_device_mesh)
        >>> # With async combine for shared-expert overlap:
        >>> ep_style = ExpertParallel(async_combine=True)
        >>> sharded_experts = ep_style.apply(experts_module, ep_device_mesh)
    """

    def __init__(self, token_dispatcher: Union[str, bool] = "all_to_all", async_combine: bool = False) -> None:
        """Initialize ExpertParallel.

        Args:
            token_dispatcher: Token dispatch strategy. Supported values are
                ``"all_to_all"`` and ``"deredundency"``.
            async_combine: If ``True``, use asynchronous combine all-to-all
                to overlap communication with shared-expert computation.
        """
        if isinstance(token_dispatcher, bool):
            async_combine = token_dispatcher
            token_dispatcher = "all_to_all"
        self._dispatch_ctx: Optional[DispatchContext] = None
        self.async_combine = async_combine
        self._token_dispatcher_name = token_dispatcher
        self._token_dispatcher = _resolve_token_dispatcher(token_dispatcher)

    def _token_dispatch(self, module: Module, inputs, device_mesh: DeviceMesh):
        """Dispatch tokens to their assigned ranks via all-to-all.

        Delegates to the configured token dispatcher and stores the
        returned :class:`DispatchContext` on the instance for the matching
        :meth:`_token_combine` call.

        Args:
            module: The ``GroupedExperts`` module.
            inputs: Tuple ``(routed_input, num_tokens_per_expert)`` or
                ``(routed_input, num_tokens_per_expert, scores)``.
            device_mesh: EP device mesh (1-D).

        Returns:
            Tuple ``(permuted_local_input, local_token_counts)``.

        Raises:
            ValueError: If ``score_before_experts=False`` (scores passed as
                a positional argument) and EP degree > 1.  After dispatch,
                the token order changes but scores remain in the pre-dispatch
                order, causing a silent correctness bug.
        """
        ep_size = device_mesh.size()
        # When EP reorders tokens across ranks, scores (if provided) would
        # no longer align with the dispatched token order.  The caller must
        # use score_before_experts=True so that scores are multiplied in
        # before dispatch.
        if ep_size > 1 and len(inputs) > 2 and inputs[2] is not None:
            raise ValueError(
                "ExpertParallel does not support score_before_experts=False "
                "when ep_size > 1.  After all-to-all dispatch the token order "
                "changes but scores remain in the pre-dispatch order, causing "
                "incorrect routing weights.  Set score_before_experts=True in "
                "MoE so that scores are multiplied before dispatch."
            )

        permuted, local_counts, ctx = (
            self._token_dispatcher.dispatch(module, inputs, device_mesh)
        )
        # Store context in module attribute for _token_combine to read.
        # Using module attribute ensures each module has its own context,
        # solving the instance sharing problem when the same ExpertParallel
        # style object is applied to multiple GroupedExperts modules.
        # pylint: disable=W0212
        module._ep_dispatch_ctx = ctx
        return permuted, local_counts

    def _token_combine(self, module: Module, routed_output, device_mesh: DeviceMesh):
        """Gather expert outputs back to the originating ranks via all-to-all.

        When ``async_combine=True``, launches the combine all-to-all
        asynchronously and returns an :class:`AsyncCollectiveTensor`.  The
        actual device-side wait is deferred until the downstream consumer
        (e.g. MoE unpermutation) first reads the tensor, enabling overlap
        with shared-expert computation.

        When ``async_combine=False`` (default), uses the synchronous
        :meth:`AllToAllTokenDispatcher.combine` — identical to the baseline.

        Args:
            module: The ``GroupedExperts`` module.
            routed_output: Expert output tensor in expert-major order.
            device_mesh: EP device mesh (1-D).

        Returns:
            Token tensor in the original token-major layout.  When
            ``async_combine=True``, this may be an async collective tensor
            whose values are not yet materialised.

        Raises:
            RuntimeError: If dispatch context is not found (dispatch was not called).
        """
        # Read dispatch context from module attribute set by _token_dispatch.
        # pylint: disable=W0212
        ctx = getattr(module, "_ep_dispatch_ctx", None)
        if ctx is None:
            raise RuntimeError(
                "_token_combine called but no dispatch context found in module. "
                "This indicates _token_dispatch was not called before _token_combine, "
                "or the context was already consumed by a previous combine call."
            )

        # Note: Do NOT delete the context here. In PyTorch, the tensors in ctx
        # are captured by autograd graph and don't need the attribute. But in
        # MindSpore PyNative mode, deleting the attribute may break backward.
        # The context will be overwritten on the next forward call.

        if self.async_combine:
            handle = self._token_dispatcher.combine_start(
                routed_output, device_mesh, ctx
            )
            # Store on module for external inspection / advanced use cases.
            # pylint: disable=W0212
            module._ep_combine_handle = handle
            # Return the async tensor.  The first non-view access by the
            # downstream consumer (e.g. MoE unpermutation) will trigger the
            # implicit wait, overlapping with shared_expert computation.
            return handle.wait()

        return self._token_dispatcher.combine(
            module, routed_output, device_mesh, ctx,
        )

    def _partition_mesh(self, device_mesh: DeviceMesh) -> DeviceMesh:
        """Return the mesh used to shard expert weights."""
        if self._token_dispatcher_name == "deredundency":
            return _get_flattened_ep_mesh(device_mesh)
        return device_mesh

    def _partition_fn(
        self, name: str, module: Module, device_mesh: DeviceMesh
    ) -> None:
        """Shard all expert parameters along dim 0 (expert dimension).

        Args:
            name: Submodule name (unused).
            module: The module whose parameters are being sharded.
            device_mesh: EP device mesh.
        """
        del name
        partition_mesh = self._partition_mesh(device_mesh)
        for key, param in _distribute_module_iter_params(module):
            if param is None:
                continue
            src = _distribute_module_param_source(param)
            requires_grad = bool(getattr(param, "requires_grad", True))
            dt = distribute_tensor(src, partition_mesh, [Shard(0)])
            new_param = _distribute_module_new_parameter(key, dt, requires_grad)
            _distribute_module_set_param(module, key, new_param)


# ---------------------------------------------------------------------------
# TensorParallel — TP-only weight sharding for experts (no token dispatch)
# ---------------------------------------------------------------------------
class TensorParallel(ParallelStyle):
    """Tensor Parallel for expert weights (no token dispatch).

    Shards the ``GroupedExperts`` weight tensors in the column/row-wise
    pattern used by standard TP:

    - ``w1`` / ``w3``: ``Shard(1)`` — column-wise (hidden_dim dimension).
    - ``w2``: ``Shard(2)`` — row-wise (output dim dimension).

    Use this when EP degree is 1 and you want TP across experts without
    any all-to-all token dispatch.  Typically combined with the standard
    :class:`~hyper_parallel.core.tensor_parallel.style.ColwiseParallel` /
    :class:`~hyper_parallel.core.tensor_parallel.style.RowwiseParallel`
    pattern for attention layers.

    Example::
        >>> tp_style = TensorParallel()
        >>> sharded_experts = tp_style.apply(experts_module, tp_device_mesh)
    """

    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:
        """Apply TP weight sharding to *module*.

        Args:
            module: A :class:`GroupedExperts` instance.
            device_mesh: 1-D TP device mesh (``mesh_dim_names=("tp",)``).

        Returns:
            The module with TP-sharded expert parameters.
        """
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
        )

    @staticmethod
    def _partition_fn(name: str, module: Module, device_mesh: DeviceMesh) -> None:
        """Shard expert weights column-wise (w1/w3) or row-wise (w2).

        ``GroupedExperts`` weight layout is ``[num_experts, out_dim, in_dim]``
        so:

        - ``w1``/``w3``: shard ``Shard(1)`` → split ``hidden_dim``
          (column-wise analogue).
        - ``w2``: shard ``Shard(2)`` → split ``in_dim = hidden_dim``
          (row-wise analogue).

        Args:
            name: Submodule name (unused).
            module: The module whose parameters are being sharded.
            device_mesh: TP device mesh.
        """
        del name
        for key, param in _distribute_module_iter_params(module):
            if param is None:
                continue
            src = _distribute_module_param_source(param)
            requires_grad = bool(getattr(param, "requires_grad", True))
            # w1, w3: column-wise → Shard(1); w2: row-wise → Shard(2).
            shard_dim = 2 if key == "w2" else 1
            dt = distribute_tensor(src, device_mesh, [Shard(shard_dim)])
            new_param = _distribute_module_new_parameter(key, dt, requires_grad)
            _distribute_module_set_param(module, key, new_param)


# ---------------------------------------------------------------------------
# ExpertTensorParallel — combined EP + TP on a 2-D [ep, tp] mesh
# ---------------------------------------------------------------------------

class ExpertTensorParallel(ExpertParallel):
    """Combined Expert + Tensor Parallel on a 2-D ``[ep, tp]`` device mesh.

    Extends :class:`ExpertParallel` to operate on a 2-D mesh with named
    dimensions ``"ep"`` and ``"tp"``:

    - **Partition**: each expert weight ``[num_experts, out, in]`` is doubly
      sharded — ``Shard(0)`` along the EP dim (expert ownership) and
      ``Shard(1)``/``Shard(2)`` along the TP dim (column-wise / row-wise).
    - **Dispatch / Combine**: use only the 1-D ``device_mesh["ep"]`` sub-mesh
      so that token routing uses EP-group collectives, not the full 2-D mesh.

    Args:
        token_dispatcher: Token dispatch strategy. Supported values are
            ``"all_to_all"`` and ``"deredundency"``.
        async_combine: Forwarded to :class:`ExpertParallel`.  When ``True``,
            the combine all-to-all is launched asynchronously for
            shared-expert overlap.

    Example::
        >>> etp_style = ExpertTensorParallel()
        >>> sharded = etp_style.apply(experts_module, ep_tp_2d_mesh)
    """

    def __init__(self, token_dispatcher: Union[str, bool] = "all_to_all", async_combine: bool = False) -> None:
        """Initialize ExpertTensorParallel.

        Args:
            async_combine: If ``True``, use asynchronous combine all-to-all.
        """
        super().__init__(token_dispatcher=token_dispatcher, async_combine=async_combine)

    def _dispatch_mesh(self, device_mesh: DeviceMesh) -> DeviceMesh:
        """Return the mesh used for token dispatch in ETP."""
        if self._token_dispatcher_name == "deredundency":
            raise NotImplementedError(
                "ExpertTensorParallel does not yet support "
                "token_dispatcher='deredundency'. Use ExpertParallel with a "
                "[oep, iep] mesh, or add [oep, iep, tp] mesh handling first."
            )
        return device_mesh["ep"]

    def _token_dispatch(self, module: Module, inputs, device_mesh: DeviceMesh):
        """Dispatch tokens using only the EP sub-mesh.

        Args:
            module: The ``GroupedExperts`` module.
            inputs: Forward inputs tuple.
            device_mesh: 2-D device mesh with dims ``("ep", "tp")``.

        Returns:
            Transformed inputs for local expert computation.\

        Raises:
            ValueError: If ``score_before_experts=False`` and EP degree > 1.
        """
        ep_mesh = device_mesh["ep"]
        # Same score_before_experts check as ExpertParallel, but using
        # the EP sub-mesh size.
        ep_size = ep_mesh.size()
        if ep_size > 1 and len(inputs) > 2 and inputs[2] is not None:
            raise ValueError(
                "ExpertTensorParallel does not support score_before_experts=False "
                "when ep_size > 1.  After all-to-all dispatch the token order "
                "changes but scores remain in the pre-dispatch order, causing "
                "incorrect routing weights.  Set score_before_experts=True in "
                "MoE so that scores are multiplied before dispatch."
            )

        dispatch_mesh = self._dispatch_mesh(device_mesh)
        permuted, local_counts, ctx = (
            self._token_dispatcher.dispatch(module, inputs, dispatch_mesh)
        )
        # pylint: disable=W0212
        # Store context in module attribute for _token_combine to read.
        module._ep_dispatch_ctx = ctx
        return permuted, local_counts

    def _token_combine(self, module: Module, routed_output, device_mesh: DeviceMesh):
        """Combine tokens using only the EP sub-mesh.

        When ``async_combine=True``, launches the combine all-to-all
        asynchronously via :meth:`self._token_dispatcher.combine_start`.

        Args:
            module: The ``GroupedExperts`` module.
            routed_output: Expert output tensor in expert-major order.
            device_mesh: 2-D device mesh with dims ``("ep", "tp")``.

        Returns:
            Token tensor in the original token-major layout.

        Raises:
            RuntimeError: If dispatch context is not found.
        """
        # pylint: disable=W0212
        # Read dispatch context from module attribute set by _token_dispatch.
        ctx = getattr(module, "_ep_dispatch_ctx", None)
        if ctx is None:
            raise RuntimeError(
                "_token_combine called but no dispatch context found in module. "
                "This indicates _token_dispatch was not called before _token_combine, "
                "or the context was already consumed by a previous combine call."
            )

        # Note: Do NOT delete the context here. In PyTorch, the tensors in ctx
        # are captured by autograd graph and don't need the attribute. But in
        # MindSpore PyNative mode, deleting the attribute may break backward.
        # The context will be overwritten on the next forward call.

        dispatch_mesh = self._dispatch_mesh(device_mesh)

        if self.async_combine:
            handle = self._token_dispatcher.combine_start(
                routed_output, dispatch_mesh, ctx
            )
            # pylint: disable=W0212
            module._ep_combine_handle = handle
            return handle.wait()

        return self._token_dispatcher.combine(
            module, routed_output, dispatch_mesh, ctx,
        )

    def _partition_fn(
        self, name: str, module: Module, device_mesh: DeviceMesh
    ) -> None:
        """Shard expert weights along both EP (dim 0) and TP (dim 1 or 2).

        Weight layout ``[num_experts, out_dim, in_dim]``:

        - ``w1``/``w3``: ``[Shard(0), Shard(1)]`` — EP shards experts,
          TP splits hidden_dim (column-wise).
        - ``w2``: ``[Shard(0), Shard(2)]`` — EP shards experts, TP splits
          the input dimension (row-wise).

        Args:
            name: Submodule name (unused).
            module: The module whose parameters are being sharded.
            device_mesh: 2-D device mesh with dims ``("ep", "tp")``.
        """
        del name
        for key, param in _distribute_module_iter_params(module):
            if param is None:
                continue
            src = _distribute_module_param_source(param)
            requires_grad = bool(getattr(param, "requires_grad", True))
            # EP shards expert ownership (dim 0); TP shards weight dim.
            tp_dim = 2 if key == "w2" else 1
            dt = distribute_tensor(src, device_mesh, [Shard(0), Shard(tp_dim)])
            new_param = _distribute_module_new_parameter(key, dt, requires_grad)
            _distribute_module_set_param(module, key, new_param)
