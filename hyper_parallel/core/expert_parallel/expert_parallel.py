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
from typing import Optional

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

    total = int(num_tokens_per_expert.sum())
    if total == 0:
        return counts.new_zeros(0, dtype=counts.dtype), num_tokens_per_expert

    # Cumulative start position of each (rank, expert) block in the
    # rank-major received buffer.
    offsets = counts.new_zeros(num_ranks * experts_per_rank + 1)
    offsets[1:] = counts.cumsum(0)

    # Build permuted_indices by iterating expert-major order.
    device = counts.device
    permuted_indices = platform.arange(0, total, device=device)
    dst = 0
    for e in range(experts_per_rank):
        for r in range(num_ranks):
            n = int(counts_2d[r, e])
            if n == 0:
                continue
            src_start = int(offsets[r * experts_per_rank + e])
            permuted_indices[dst:dst + n] = platform.arange(
                src_start, src_start + n, device=device
            )
            dst += n

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
    if permuted_indices.numel() == 0:
        return original_shape, x.new_zeros(0, *x.shape[1:]), permuted_indices, num_tokens_per_expert
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
    if permuted_indices.numel() == 0:
        return out.new_zeros(*original_shape)
    result = out.new_zeros(*original_shape)
    result[permuted_indices] = out
    return result


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
        # State saved between _token_dispatch and _token_combine within one
        # forward pass.  Safe for standard (non-pipeline) training.
        self._input_splits: list = []
        self._output_splits: list = []
        self._input_shape: Optional[tuple] = None
        self._permuted_indices = None

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

    def _token_dispatch(self, module: Module, inputs, device_mesh: DeviceMesh):
        """Dispatch tokens to their assigned ranks via all-to-all.

        Called as an ``input_fn`` hook by ``distribute_module``.  Receives the
        module's forward inputs and returns transformed inputs.

        Args:
            module: The ``GroupedExperts`` module (unused here).
            inputs: Tuple ``(routed_input, num_tokens_per_expert)`` where
                ``routed_input`` has shape ``[total_tokens, dim]`` and
                ``num_tokens_per_expert`` has shape ``[num_experts]``.
            device_mesh: EP device mesh (1-D).

        Returns:
            Tuple ``(permuted_local_input, local_token_counts)`` ready for
            local expert computation.
        """
        del module
        routed_input, num_tokens_per_expert = inputs[0], inputs[1]
        ep_group = device_mesh.get_group()
        ep_size = device_mesh.size()
        num_local_experts = num_tokens_per_expert.shape[0] // ep_size

        # --- Step 1: exchange token counts (no gradient needed) ---
        # Each rank needs to know how many tokens it will receive from every
        # other rank (for each local expert).
        counts_out, _ = platform.all_to_all_single(
            num_tokens_per_expert,
            output_shape=[num_tokens_per_expert.shape[0]],
            group=ep_group,
        )
        # counts_out shape: [ep_size * num_local_experts]
        # counts_out[r * num_local_experts + e] = tokens from rank r for expert e

        # --- Step 2: compute input / output splits ---
        # input_splits[r] = tokens this rank sends to rank r
        input_splits = [
            int(num_tokens_per_expert[r * num_local_experts:(r + 1) * num_local_experts].sum())
            for r in range(ep_size)
        ]
        # output_splits[r] = tokens this rank receives from rank r
        output_splits = [
            int(counts_out[r * num_local_experts:(r + 1) * num_local_experts].sum())
            for r in range(ep_size)
        ]
        self._input_splits = input_splits
        self._output_splits = output_splits

        # --- Step 3: exchange actual tokens (differentiable) ---
        dispatched = platform.differentiable_all_to_all_single(
            routed_input, input_splits, output_splits, group=ep_group,
        )

        # --- Step 4: rank-major → expert-major permutation ---
        self._input_shape, permuted, self._permuted_indices, local_counts = _permute(
            dispatched, counts_out, ep_size, num_local_experts
        )
        return permuted, local_counts

    def _token_combine(self, module: Module, routed_output, device_mesh: DeviceMesh):
        """Gather expert outputs back to the originating ranks via all-to-all.

        Called as an ``output_fn`` hook by ``distribute_module``.

        Args:
            module: The ``GroupedExperts`` module (unused).
            routed_output: Expert output tensor in expert-major order,
                shape ``[sum(local_counts), dim]``.
            device_mesh: EP device mesh (1-D).

        Returns:
            Token tensor in the original token-major layout,
            shape ``[sum(input_splits), dim]``.
        """
        del module
        ep_group = device_mesh.get_group()

        # expert-major → rank-major
        unpermuted = _unpermute(routed_output, self._input_shape, self._permuted_indices)

        # reverse all-to-all (output/input splits are swapped)
        combined = platform.differentiable_all_to_all_single(
            unpermuted,
            self._output_splits,   # was output, now becomes input
            self._input_splits,    # was input, now becomes output
            group=ep_group,
        )
        return combined


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

    def _token_dispatch(self, module: Module, inputs, device_mesh: DeviceMesh):
        """Dispatch tokens using only the EP sub-mesh.

        Args:
            module: The ``GroupedExperts`` module.
            inputs: Forward inputs tuple.
            device_mesh: 2-D device mesh with dims ``("ep", "tp")``.

        Returns:
            Transformed inputs for local expert computation.
        """
        return super()._token_dispatch(module, inputs, device_mesh["ep"])

    def _token_combine(self, module: Module, routed_output, device_mesh: DeviceMesh):
        """Combine tokens using only the EP sub-mesh.

        Args:
            module: The ``GroupedExperts`` module.
            routed_output: Expert output tensor in expert-major order.
            device_mesh: 2-D device mesh with dims ``("ep", "tp")``.

        Returns:
            Token tensor in the original token-major layout.
        """
        return super()._token_combine(module, routed_output, device_mesh["ep"])
