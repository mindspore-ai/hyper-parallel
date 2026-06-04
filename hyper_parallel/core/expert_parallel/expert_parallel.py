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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Tuple

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
from hyper_parallel.platform import get_platform

platform = get_platform()
Module = platform.Module

__all__ = [
    "AllToAllTokenDispatcher",
    "BaseExpertParallel",
    "ExpertParallel",
    "TensorParallel",
    "ExpertTensorParallel",
]


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


# ---------------------------------------------------------------------------
# AllToAllTokenDispatcher — token dispatch/combine via all-to-all
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

    Args:
        None

    Example::
        >>> ep_style = ExpertParallel()
        >>> sharded_experts = ep_style.apply(experts_module, ep_device_mesh)
    """

    def __init__(self) -> None:
        """Initialize ExpertParallel."""

    def _token_dispatch(self, module: Module, inputs, device_mesh: DeviceMesh):
        """Dispatch tokens to their assigned ranks via all-to-all.

        Delegates to :meth:`AllToAllTokenDispatcher.dispatch`, stores the
        returned :class:`DispatchContext` on the module attribute for the
        matching :meth:`_token_combine` call.

        Args:
            module: The ``GroupedExperts`` module.
            inputs: Tuple ``(routed_input, num_tokens_per_expert)``.
            device_mesh: EP device mesh (1-D).

        Returns:
            Tuple ``(permuted_local_input, local_token_counts)``.
        """
        permuted, local_counts, ctx = (
            AllToAllTokenDispatcher.dispatch(module, inputs, device_mesh)
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

        Reads :class:`DispatchContext` from module attribute set by
        :meth:`_token_dispatch` and delegates to
        :meth:`AllToAllTokenDispatcher.combine`.

        Args:
            module: The ``GroupedExperts`` module.
            routed_output: Expert output tensor in expert-major order.
            device_mesh: EP device mesh (1-D).

        Returns:
            Token tensor in the original token-major layout.
            
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

        return AllToAllTokenDispatcher.combine(
            module, routed_output, device_mesh, ctx,
        )

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
        for key, param in _distribute_module_iter_params(module):
            if param is None:
                continue
            src = _distribute_module_param_source(param)
            requires_grad = bool(getattr(param, "requires_grad", True))
            dt = distribute_tensor(src, device_mesh, [Shard(0)])
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

    def _partition_fn(
        self, name: str, module: Module, device_mesh: DeviceMesh
    ) -> None:
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
        None

    Example::
        >>> etp_style = ExpertTensorParallel()
        >>> sharded = etp_style.apply(experts_module, ep_tp_2d_mesh)
    """

    def _token_dispatch(self, module: Module, inputs, device_mesh: DeviceMesh):
        """Dispatch tokens using only the EP sub-mesh.

        Args:
            module: The ``GroupedExperts`` module.
            inputs: Forward inputs tuple.
            device_mesh: 2-D device mesh with dims ``("ep", "tp")``.

        Returns:
            Transformed inputs for local expert computation.
        """
        permuted, local_counts, ctx = (
            AllToAllTokenDispatcher.dispatch(module, inputs, device_mesh["ep"])
        )
        # Store context in module attribute for _token_combine to read.
        # pylint: disable=W0212
        module._ep_dispatch_ctx = ctx
        return permuted, local_counts

    def _token_combine(self, module: Module, routed_output, device_mesh: DeviceMesh):
        """Combine tokens using only the EP sub-mesh.

        Args:
            module: The ``GroupedExperts`` module.
            routed_output: Expert output tensor in expert-major order.
            device_mesh: 2-D device mesh with dims ``("ep", "tp")``.

        Returns:
            Token tensor in the original token-major layout.
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

        return AllToAllTokenDispatcher.combine(
            module, routed_output, device_mesh["ep"], ctx,
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
