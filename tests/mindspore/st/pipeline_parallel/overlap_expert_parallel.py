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
"""Self-contained EP wrapper for MindSpore PP+EP+overlap testing.

The file vendors a minimal subset of an Expert-Parallel strategy plus a
toy ``GroupedMLP`` so the dual-thread PP+EP+overlap PoC has no
``mindformers`` dependency.  The classes intentionally mirror just the
surface the PoC needs — they are **not** a general-purpose EP
implementation.

Classes:

- :class:`MiniGroupedMLP` — gated FFN MoE experts, sharded by EP rank
  at construction time (each rank only allocates weights for its
  ``num_local_experts``).  Forward expects pre-permuted tokens in
  expert-major order and a cumsum-style ``num_tokens_per_expert``.
- :class:`MiniBaseExpertParallel` — operator bindings + ``_apply``
  that wraps the experts module so calls go through
  ``_token_dispatch → original construct → _token_combine``.
- :class:`OverlapExpertParallel` — adds A/B/C/D sync hooks around the
  dispatch / combine all-to-all and replaces ``ops.AlltoAll`` /
  ``ops.AlltoAllV`` with ``platform.all_to_all_single`` /
  ``differentiable_all_to_all_single_async`` so every EP HCCL on this
  group lands on the same launch stream — required for cross-rank
  ordering when ``CommComputeOverlap`` drives the paired FWD/BWD
  threads.  Passing ``overlap=None`` disables the hooks, making this
  class the **sync baseline** used by the accuracy test.
"""
# pylint: disable=W0212
import numpy as np

import mindspore as ms
from mindspore import mint, nn, ops
from mindspore.common import dtype as mstype

from hyper_parallel.platform import get_platform
from hyper_parallel.core.pipeline_parallel.comm_compute_overlap import CommComputeOverlap


platform = get_platform()


# =========================================================================
# MiniGroupedMLP — per-local-expert gated FFN
# =========================================================================

class MiniGroupedMLP(nn.Cell):
    """Toy ``GroupedMLP`` substitute: one gated FFN per local expert.

    Weights are allocated **only** for ``num_local_experts`` (=
    ``num_experts / ep_size``); the EP-rank already determines which
    experts live here, so no further weight sharding is needed at
    ``_apply`` time.

    Args:
        hidden_size: Per-token embedding dimension.
        ffn_hidden_size: Inner FFN dimension.
        num_local_experts: How many experts this rank holds.
        rng: ``numpy.random.RandomState`` consumed in a fixed
            ``w_fc1 → w_fc2`` order so two builds with the same seed
            land on bit-identical weights.  Using numpy (not
            ``mint.normal``) is the reproducibility anchor: MS PyNative
            does not deterministically reset every per-op kernel RNG on
            ``ms.set_seed``, so the accuracy comparison would otherwise
            see different initial state on the two paths.
        compute_dtype: Parameter dtype (also the activation dtype).

    Forward signature mirrors mindformers' ``GroupedMLP`` so the EP
    strategies in this file can call it uniformly:
        ``construct(tokens, probs, topk_indices, num_tokens_per_expert)``
    where ``tokens`` are pre-permuted in expert-major order and
    ``num_tokens_per_expert`` is the cumsum of per-local-expert token
    counts produced by :meth:`MiniBaseExpertParallel._token_dispatch`.
    """

    def __init__(self, hidden_size: int, ffn_hidden_size: int,
                 num_local_experts: int, rng: np.random.RandomState,
                 compute_dtype=mstype.float32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_local_experts = num_local_experts
        w_fc1_np = rng.normal(
            0.0, 0.02, size=(num_local_experts, hidden_size, 2 * ffn_hidden_size),
        ).astype(np.float32)
        w_fc2_np = rng.normal(
            0.0, 0.02, size=(num_local_experts, ffn_hidden_size, hidden_size),
        ).astype(np.float32)
        self.w_fc1 = ms.Parameter(
            ms.Tensor(w_fc1_np).to(compute_dtype), name="w_fc1",
        )
        self.w_fc2 = ms.Parameter(
            ms.Tensor(w_fc2_np).to(compute_dtype), name="w_fc2",
        )

    def construct(self, tokens, probs, topk_indices, num_tokens_per_expert):
        """Per-local-expert gated FFN.

        ``probs`` / ``topk_indices`` are unused here (the routing-weight
        multiplication happens in :meth:`_token_combine`); kept in the
        signature for parity with mindformers' API.
        """
        del probs, topk_indices
        outs = []
        start = 0
        # ``num_tokens_per_expert`` is a cumsum (int64 device tensor);
        # converting to host ints once per call is the same host sync
        # mindformers' GroupedMLP does, so cost is comparable.
        for e in range(self.num_local_experts):
            end = int(num_tokens_per_expert[e].asnumpy())
            seg = tokens[start:end]
            fc1_out = mint.matmul(seg, self.w_fc1[e])
            gate = fc1_out[..., : self.ffn_hidden_size]
            up = fc1_out[..., self.ffn_hidden_size:]
            activated = mint.nn.functional.gelu(gate) * up
            outs.append(mint.matmul(activated, self.w_fc2[e]))
            start = end
        return mint.cat(outs, dim=0)


# =========================================================================
# MiniBaseExpertParallel — operator bindings + _apply wrap
# =========================================================================

class MiniBaseExpertParallel:
    """Minimal Expert-Parallel strategy base class.

    Provides:
      * Operator references (``self.reshape``, ``self.cast``, ...) so
        subclasses can transcribe mindformers-style dispatch / combine
        code without depending on ``mindformers``.
      * State holders (``self.ctx``, ``self.input_layout``,
        ``self.ep_group``, ``self.ep_size``) shared between dispatch
        and combine.
      * ``_apply`` to wrap an experts module so user calls
        (``experts(...)``) go through
        ``_token_dispatch → original construct → _token_combine``.

    Subclasses override :meth:`_token_dispatch` / :meth:`_token_combine`.
    """

    def __init__(self) -> None:
        # State populated by ``_apply`` and by the dispatch/combine pair.
        self.ctx = None
        self.input_layout = None
        self.ep_group = None
        self.ep_size = None

    # ------------------------------------------------------------------
    # Operator bindings — bound once on the class so subclasses can use
    # ``self.reshape(...)`` without per-instance allocation overhead.
    # ------------------------------------------------------------------

    @staticmethod
    def reshape(x, shape):
        return mint.reshape(x, shape)

    @staticmethod
    def cast(x, dtype):
        return x.astype(dtype)

    @staticmethod
    def concat(tensors, dim=0):
        return mint.cat(tensors, dim=dim)

    @staticmethod
    def transpose(x, dim0, dim1):
        """Two-argument transpose (mindformers convention): swap dims.

        MS' ``ops.transpose`` expects a full permutation, so build one.
        """
        rank = len(x.shape)
        perm = list(range(rank))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return ops.transpose(x, tuple(perm))

    @staticmethod
    def sort(x, dim=-1):
        return ops.sort(x, axis=dim)

    @staticmethod
    def fmod(x, y):
        return ops.fmod(x, y)

    @staticmethod
    def index_select(x, dim, index):
        """Gather rows of ``x`` along ``dim`` using a 1-D ``index`` tensor.

        Routes through ``mint.index_select`` (pyboost → ``aclnnIndexSelect``)
        rather than ``ops.gather`` because ``gather_op.yaml`` has no
        ``dispatch: enable`` block and falls through to the generic aclop
        kernel, which carries a noticeably higher host dispatch cost.  Same
        ``(input, dim, 1-D index)`` contract as ``ops.gather``, so every
        caller (``_permute``, ``_token_dispatch``, ``_token_combine``)
        benefits from this single switch without touching the call sites.
        """
        return mint.index_select(x, dim, index)

    @staticmethod
    def sum(x, dim=None, keepdim=False):
        if dim is None:
            return x.sum()
        return mint.sum(x, dim=dim, keepdim=keepdim)

    @staticmethod
    def cumsum(x, dim=0):
        return ops.cumsum(x, axis=dim)

    @staticmethod
    def mul(a, b):
        return mint.mul(a, b)

    @staticmethod
    def strided_slice(x, begin, end, stride):
        return ops.strided_slice(x, begin, end, stride)

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def _apply(self, module, device_mesh):
        """Wrap ``module.construct`` so user calls go through dispatch/combine.

        Records the EP HCCL group from the supplied device mesh —
        ``device_mesh.get_group()`` returns the MS group name string
        that the comm collectives expect.  Weight sharding is handled
        at construction time by :class:`MiniGroupedMLP` (it only
        allocates ``num_local_experts`` worth of weights), so this
        method only patches the call path.
        """
        self.ep_size = device_mesh.mesh_shape[0]
        self.ep_group = device_mesh.get_group()

        original_construct = module.construct
        ep_self = self

        def wrapped(*args):
            dispatched = ep_self._token_dispatch(device_mesh, module, args)
            out = original_construct(*dispatched)
            return ep_self._token_combine(device_mesh, module, args, out)

        module.construct = wrapped
        return module

    # Subclass hooks ----------------------------------------------------

    def _token_dispatch(self, device_mesh, cell, args):
        raise NotImplementedError

    def _token_combine(self, device_mesh, cell, args, routed_output):
        raise NotImplementedError


# =========================================================================
# OverlapExpertParallel — EP with A/B/C/D sync hooks + single-stream HCCL
# =========================================================================

class OverlapExpertParallel(MiniBaseExpertParallel):
    """Expert-Parallel strategy with optional comm/compute overlap hooks.

    Args:
        overlap:        Shared :class:`CommComputeOverlap` whose
                        coordinator drives the FWD/BWD-thread rendezvous.
                        Pass ``None`` for the **sync baseline path**:
                        the A/B/C/D ``differentiable_sync_hook`` calls
                        become no-ops, all three EP HCCLs still route
                        through the same ``comm_func.all_to_all_single``
                        path so per-EP-group launch stream is identical
                        between baseline and overlap runs (important for
                        a meaningful accuracy comparison).
        is_last_layer:  When ``True``, the closing ``D`` hook is tagged
                        ``"D_LAST"`` so the rendezvous is skipped on
                        both forward (no Attention follows the last
                        layer) and backward (combine.bwd has already
                        dispatched freely before any rendezvous).

    All EP-group HCCLs (counts, main, routing-map) live inside the A→B
    window and route through ``comm_func.all_to_all_single``.  The
    ``ops.AlltoAll`` / ``ops.AlltoAllV`` Primitives dispatch on a
    separate stream from ``comm_func.all_to_all_single``, so mixing
    them lets the FWD / BWD threads push HCCL on two different streams
    against the same EP group; cross-rank the two streams' order
    interleaves non-deterministically and the next collective on the
    group deadlocks once ``MS_DEV_LAUNCH_BLOCKING`` is unset.
    Funnelling everything through one path keeps dispatch FIFO within
    the stream.
    """

    def __init__(self, overlap: CommComputeOverlap = None,
                 is_last_layer: bool = False,
                 moe_permute_fusion: bool = False) -> None:
        super().__init__()
        self._overlap = overlap
        self._d_hook = "D_LAST" if is_last_layer else "D"
        # When True, the dispatch/combine permute chain is replaced by
        # ``ops.moe_token_permute`` / ``ops.moe_token_unpermute`` — single
        # fused kernels that fold the sort + fmod + index_select pipeline
        # so host-side op dispatch cost is amortised over one call instead
        # of ~5 small kernels.  Mirrors mindformers' ExpertParallel.moe_permute_fusion.
        self._moe_permute_fusion = moe_permute_fusion

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_sync_hook(self, x, hook_name: str):
        """Fire ``differentiable_sync_hook`` if an overlap coordinator
        was supplied; otherwise no-op (sync baseline path)."""
        if self._overlap is None:
            return x
        return platform.differentiable_sync_hook(x, hook_name, self._overlap.coordinator)

    def _permute(self, tokens, topk_indices, moe_permute_fusion: bool):
        """Route tokens into expert-major order.

        Vendored from mindformers'
        ``ExpertParallel._permute`` so we share the same fast / slow
        switch.  Fusion mode collapses the manual ``sort + transpose +
        fmod + index_select`` chain into a single ``ops.moe_token_permute``
        kernel call, removing several host-bound op dispatches from the
        per-layer dispatch path; non-fusion mode preserves the original
        behavior for cross-validation.

        Returns:
            ``(routed_input, sorted_topk_indices, topk_indices_flat,
              unsort_token_indices_experts)``.  ``topk_indices_flat`` is
            the flattened-by-topk version, threaded through to the
            experts' ``construct`` for mindformers ``GroupedMLP`` API
            parity (the experts ignore it; counts now come from the
            caller-supplied ``num_tokens_per_expert``).
        """
        tokens_shape = tokens.shape
        topk_indices_shape = topk_indices.shape
        if moe_permute_fusion:
            topk_indices_kn = self.transpose(topk_indices, 1, 0)
            topk_indices_kn = self.reshape(topk_indices_kn, (-1,))
            sorted_topk_indices, _ = self.sort(
                self.cast(topk_indices_kn, mstype.float32), dim=-1,
            )
            tokens = ops.reshape(tokens, (-1, tokens_shape[-1]))
            topk_indices = ops.reshape(topk_indices, (-1, topk_indices.shape[-1]))
            routed_input, unsort_token_indices_experts = ops.moe_token_permute(
                tokens, topk_indices.astype(mstype.int32),
            )
            routed_input = self.reshape(
                routed_input, (tokens_shape[0], -1, tokens_shape[-1]),
            )
            unsort_token_indices_experts = self.reshape(
                unsort_token_indices_experts,
                (topk_indices_shape[0], topk_indices_shape[1]),
            )
            return routed_input, sorted_topk_indices, topk_indices_kn, unsort_token_indices_experts

        # Non-fusion (manual) path.
        topk_indices = self.transpose(topk_indices, 1, 0)
        topk_indices = self.reshape(topk_indices, (-1,))
        sorted_topk_indices, token_indices_experts_sorted = self.sort(
            self.cast(topk_indices, mstype.float32), dim=-1,
        )
        _, unsort_token_indices_experts = self.sort(
            self.cast(token_indices_experts_sorted, mstype.float32), dim=-1,
        )
        unsort_token_indices_experts = self.reshape(
            unsort_token_indices_experts,
            (topk_indices_shape[1], topk_indices_shape[0]),
        )
        unsort_token_indices_experts = self.transpose(unsort_token_indices_experts, 1, 0)
        inter_map = self.fmod(token_indices_experts_sorted, topk_indices_shape[0])
        index = self.reshape(inter_map, (-1,))
        routed_input = self.index_select(tokens, 0, index)
        routed_input = self.reshape(
            routed_input, (tokens_shape[0], -1, tokens_shape[-1]),
        )
        return routed_input, sorted_topk_indices, topk_indices, unsort_token_indices_experts

    # ------------------------------------------------------------------
    # _token_dispatch override
    # ------------------------------------------------------------------

    def _token_dispatch(self, device_mesh, cell, args):
        """Dispatch path with A→B hooks bracketing the main token a2a."""
        # pylint: disable=R0914,C0415
        from hyper_parallel.core.dtensor.dtensor import DTensor
        tokens, probs, topk_indices, num_tokens_per_expert = args
        if isinstance(tokens, DTensor):
            self.input_layout = tokens.layout
            tokens = tokens.to_local()

        tokens_shape = tokens.shape
        tokens = self.reshape(tokens, (-1, tokens_shape[-1]))

        ep_degree = device_mesh.mesh_shape[0]
        num_experts = num_tokens_per_expert.shape[-1]
        moe_router_topk = topk_indices.shape[-1]

        # ---- Pre-A2A: padding + sort ----
        # ``pad_topk_indices`` is a LOCAL variable.  Vanilla mindformers
        # stashes it on the instance, but that shared mutable state
        # races under dual-thread overlap and trips MS PyNative's lazy
        # shape inference (the tensor's shape can change between print
        # and downstream op).
        pad_tokens = mint.zeros((num_experts, tokens.shape[-1]), dtype=tokens.dtype)
        pad_probs = mint.zeros((num_experts, moe_router_topk), dtype=probs.dtype)
        pad_topk_indices = mint.arange(
            num_experts * moe_router_topk, dtype=topk_indices.dtype,
        ) % num_experts
        pad_topk_indices = pad_topk_indices.reshape((num_experts, moe_router_topk))
        pad_size = num_experts

        tokens = self.concat((pad_tokens, tokens), dim=0)
        probs = self.concat((pad_probs, probs), dim=0)
        topk_indices = self.concat((pad_topk_indices, topk_indices), dim=0)

        # Permute tokens into expert-major order via fused / manual path.
        # ``sorted_topk_indices`` (the expert-id sequence after the source-side
        # sort) was the payload of the now-removed routing-map a2a; we ignore
        # it here because the receiver reconstructs the same reorder index
        # directly from the counts matrix produced by the counts a2a (see
        # below).
        (routed_input, _, topk_indices,
         unsort_token_indices_experts) = self._permute(
            tokens, topk_indices, self._moe_permute_fusion,
        )

        # Reuse the caller-supplied ``num_tokens_per_expert`` (the router
        # already counted the un-padded tokens) and add the padding
        # contribution as a constant: ``pad_topk_indices = arange(num_experts
        # * topk) % num_experts`` hits each expert id exactly ``moe_router_topk``
        # times, so the post-padding count is ``router_count + moe_router_topk``.
        # Mirrors the torch ``ExpertParallel`` design in
        # ``examples/torch/pp_overlap/pp_overlap_moe_example.py`` where
        # dispatch consumes pre-computed counts and never recomputes via
        # ``one_hot``/``bincount`` — collapses the previous
        # ``cast → one_hot → sum → cast`` chain (4 host dispatches + an
        # ``[N, num_experts]`` intermediate) into a single ``add`` with no
        # device→host sync.
        num_tokens_per_expert = (
            self.cast(num_tokens_per_expert, mstype.float32) + moe_router_topk
        )

        original_shape = list(routed_input.shape)

        # ---- A hook opens the comm window BEFORE any EP HCCL on this layer ----
        flat_in = self.reshape(routed_input, (-1,))
        flat_in = self._maybe_sync_hook(flat_in, "A")

        # counts a2a — uniform splits, sync.  Its output (per-source,
        # per-local-expert count matrix ``M_local``) is the only piece of
        # routing info the receiver needs to reorder the incoming tokens
        # into expert-major order: the source-side ``_permute`` already
        # guarantees that within each src block tokens arrive sorted by
        # destination expert id, so ``M_local[s, e]`` alone fixes the
        # permutation.  That lets us drop the separate routing-map a2a.
        counts_size = int(num_tokens_per_expert.shape[0])
        num_tokens_per_expert_group, _ = platform.all_to_all_single(
            num_tokens_per_expert,
            output_shape=[counts_size],
            group=self.ep_group,
            async_op=False,
        )

        num_tokens_per_expert_reshaped = self.reshape(num_tokens_per_expert, (ep_degree, -1))
        input_splits = self.cast(
            self.sum(num_tokens_per_expert_reshaped, dim=-1, keepdim=False), mstype.int64,
        )
        num_tokens_per_expert_group_reshaped = self.reshape(
            num_tokens_per_expert_group, (ep_degree, -1),
        )
        num_tokens_per_expert = self.cumsum(
            self.sum(num_tokens_per_expert_group_reshaped, dim=-2, keepdim=False), 0,
        )
        num_tokens_per_expert = self.cast(num_tokens_per_expert, mstype.int64)

        # HCCL ``all_to_all_single`` needs split sizes as host Python lists
        # (CANN API constraint).  Drain the full ``(ep_size, num_local_experts)``
        # count matrix to host once — both ``output_split_list`` and the
        # dispatch reorder index are derived from it without further syncs.
        # ``input_split_list`` still needs its own sync since the source-side
        # breakdown lives in a separate device tensor.  Net cost: 2 host
        # syncs per dispatch (same as before), but the routing-map a2a —
        # one variable-splits HCCL roundtrip of ``≈num_tokens`` elements —
        # is gone.
        group_counts = num_tokens_per_expert_group_reshaped.asnumpy().astype(np.int64)
        input_split_list = input_splits.asnumpy().tolist()
        output_split_list = group_counts.sum(axis=1).tolist()

        # Reconstruct the dispatch reorder index (source-block order →
        # expert-major) from ``group_counts`` (= ``M_local``).  After the
        # main token a2a the receiver buffer is laid out as
        # ``[from_src_0, from_src_1, ..., from_src_{P-1}]``; within each
        # source block tokens are already in destination-local-expert order.
        # Per local expert ``e`` we therefore gather ``group_counts[s, e]``
        # consecutive tokens starting at offset
        # ``src_block_starts[s] + within_src_offsets[s, e]`` for each
        # source ``s``.  ``combine_index`` is the inverse permutation,
        # cached for ``_token_combine`` to undo this reorder before the
        # combine a2a.
        src_block_starts = np.concatenate(
            ([0], group_counts.sum(axis=1).cumsum()[:-1]),
        ).astype(np.int64)
        within_src_offsets = np.concatenate(
            (np.zeros((ep_degree, 1), dtype=np.int64),
             group_counts[:, :-1].cumsum(axis=1)),
            axis=1,
        )
        total_recv_tokens = int(group_counts.sum())
        dispatch_index_np = np.empty(total_recv_tokens, dtype=np.int32)
        write_pos = 0
        for e in range(group_counts.shape[1]):
            for s in range(ep_degree):
                start = int(src_block_starts[s] + within_src_offsets[s, e])
                count = int(group_counts[s, e])
                dispatch_index_np[write_pos:write_pos + count] = np.arange(
                    start, start + count, dtype=np.int32,
                )
                write_pos += count
        combine_index_np = np.empty(total_recv_tokens, dtype=np.int32)
        combine_index_np[dispatch_index_np] = np.arange(
            total_recv_tokens, dtype=np.int32,
        )
        dispatch_index = ms.Tensor(dispatch_index_np)
        combine_index = ms.Tensor(combine_index_np)

        # main token a2a — async / differentiable (returns AsyncCollectiveTensor).
        # Element-unit splits are computed host-side from the cached
        # token-unit lists; no extra device→host sync needed.
        hidden = cell.hidden_size
        flat_out = platform.differentiable_all_to_all_single_async(
            flat_in,
            [s * hidden for s in input_split_list],
            [s * hidden for s in output_split_list],
            self.ep_group,
        )
        flat_out = self._maybe_sync_hook(flat_out, "B")
        # First real op on flat_out triggers AsyncCollectiveTensor's
        # ``__ms_dispatch__`` → ``CommHandle.wait()`` → unwrap to plain Tensor.
        global_input_tokens = self.reshape(flat_out, (1, -1, cell.hidden_size))

        # ---- Post-A2A reorder into expert-major via precomputed index ----
        global_input_tokens_shape = global_input_tokens.shape
        global_input_tokens = self.reshape(global_input_tokens, (-1, global_input_tokens_shape[-1]))
        global_input_tokens = self.index_select(global_input_tokens, 0, dispatch_index)
        global_input_tokens = self.reshape(global_input_tokens, (-1, global_input_tokens_shape[-1]))

        self.ctx = (
            probs, combine_index, unsort_token_indices_experts,
            input_split_list, output_split_list, original_shape, pad_size,
        )
        return global_input_tokens, probs, topk_indices, num_tokens_per_expert

    # ------------------------------------------------------------------
    # _token_combine override
    # ------------------------------------------------------------------

    def _token_combine(self, device_mesh, cell, args, routed_output):  # pylint: disable=W0613
        """Combine path with C→D hooks bracketing the combine a2a."""
        from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=C0415
        (
            probs, combine_index, unsort_token_indices_experts,
            input_split_list, output_split_list, original_shape, pad_size,
        ) = self.ctx

        routed_output = self.reshape(routed_output, (1, -1, cell.hidden_size))
        routed_output_shape = routed_output.shape
        routed_output = self.reshape(routed_output, (-1, routed_output_shape[-1]))
        routed_output = self.index_select(routed_output, 0, combine_index)
        routed_output = self.reshape(
            routed_output, (routed_output_shape[0], -1, routed_output_shape[-1]),
        )

        # ---- Combine a2a bracketed by C/D hooks ----
        # Combine uses output→input splits (reverse direction).  Both lists
        # are already host-side (cached in dispatch), so element-unit splits
        # are computed without any device→host sync.
        flat_in = self.reshape(routed_output, (-1,))
        flat_in = self._maybe_sync_hook(flat_in, "C")
        hidden = cell.hidden_size
        flat_out = platform.differentiable_all_to_all_single_async(
            flat_in,
            [s * hidden for s in output_split_list],
            [s * hidden for s in input_split_list],
            self.ep_group,
        )
        flat_out = self._maybe_sync_hook(flat_out, self._d_hook)
        permutated_local_input_tokens = self.reshape(flat_out, original_shape)

        # ---- Post-combine routing ----
        permutated_local_input_tokens = self.reshape(
            permutated_local_input_tokens,
            (-1, permutated_local_input_tokens.shape[-1]),
        )
        unsort_token_indices_experts_shape = unsort_token_indices_experts.shape
        if self._moe_permute_fusion:
            unsort_token_indices_experts = self.reshape(
                unsort_token_indices_experts, (-1,),
            )
            routed_output = ops.moe_token_unpermute(
                permutated_local_input_tokens,
                unsort_token_indices_experts.astype(mstype.int32),
            )
        else:
            index = self.reshape(unsort_token_indices_experts, (-1,))
            routed_output = self.index_select(permutated_local_input_tokens, 0, index)
        routed_output = self.reshape(
            routed_output,
            (unsort_token_indices_experts_shape[0],
             unsort_token_indices_experts_shape[1], -1),
        )
        probs = self.reshape(probs, (probs.shape[0], probs.shape[1], 1))
        routed_output = self.mul(routed_output, self.cast(probs, routed_output.dtype))
        routed_output = self.sum(routed_output, dim=1, keepdim=False)
        routed_output = self.strided_slice(
            routed_output, (pad_size, 0),
            (routed_output.shape[0], routed_output.shape[-1]), (1, 1),
        )
        if self.input_layout is not None:
            return DTensor.from_local(
                routed_output, self.input_layout.mesh, self.input_layout.alias_placements,
            )
        return routed_output
