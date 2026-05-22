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
        # MS ``ops.gather(input, indices, axis)`` is the equivalent.
        return ops.gather(x, index, axis=dim)

    @staticmethod
    def sum(x, dim=None, keepdim=False):
        if dim is None:
            return x.sum()
        return mint.sum(x, dim=dim, keepdim=keepdim)

    @staticmethod
    def one_hot(indices, depth):
        return ops.one_hot(
            indices, depth,
            ms.Tensor(1.0, mstype.float32),
            ms.Tensor(0.0, mstype.float32),
        )

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
                 is_last_layer: bool = False) -> None:
        super().__init__()
        self._overlap = overlap
        self._d_hook = "D_LAST" if is_last_layer else "D"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_sync_hook(self, x, hook_name: str):
        """Fire ``differentiable_sync_hook`` if an overlap coordinator
        was supplied; otherwise no-op (sync baseline path)."""
        if self._overlap is None:
            return x
        return platform.differentiable_sync_hook(x, hook_name, self._overlap.coordinator)

    @staticmethod
    def _splits_to_elem_list(splits_tensor, block_size: int):
        """Convert per-rank token-count splits (Tensor) to a Python
        list of element counts (``count * block_size`` per rank).

        Same host sync semantics as ``ops.AlltoAllV``'s internal splits
        materialisation, so no extra cost.
        """
        elem_splits = splits_tensor * block_size
        return elem_splits.asnumpy().tolist()

    def _async_a2a(self, flat_input, input_splits_t, output_splits_t, block_size: int):
        """Run our async a2a on a flat tensor with token-unit splits.

        Returns:
            An :class:`AsyncCollectiveTensor` that defers
            ``CommHandle.wait()`` to the first consumer op.
        """
        input_splits = self._splits_to_elem_list(input_splits_t, block_size)
        output_splits = self._splits_to_elem_list(output_splits_t, block_size)
        return platform.differentiable_all_to_all_single_async(
            flat_input, input_splits, output_splits, self.ep_group,
        )

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
        tokens_shape = tokens.shape

        topk_indices_shape = topk_indices.shape
        topk_indices = self.transpose(topk_indices, 1, 0)
        topk_indices = self.reshape(topk_indices, (-1,))
        sorted_topk_indices, token_indices_experts_sorted = self.sort(
            self.cast(topk_indices, mstype.float32), dim=-1,
        )
        _, unsort_token_indices_experts = self.sort(
            self.cast(token_indices_experts_sorted, mstype.float32), dim=-1,
        )
        unsort_token_indices_experts = self.reshape(
            unsort_token_indices_experts, (topk_indices_shape[1], topk_indices_shape[0]),
        )
        unsort_token_indices_experts = self.transpose(unsort_token_indices_experts, 1, 0)

        inter_map = self.fmod(token_indices_experts_sorted, topk_indices_shape[0])
        index = self.reshape(inter_map, (-1,))
        routed_input = self.index_select(tokens, 0, index)
        routed_input = self.reshape(routed_input, (tokens_shape[0], -1, tokens_shape[-1]))

        num_tokens_per_expert = self.sum(
            self.one_hot(self.cast(topk_indices, mstype.int32), num_experts), dim=0,
        )
        num_tokens_per_expert = self.cast(num_tokens_per_expert, mstype.float32)

        original_shape = list(routed_input.shape)

        # ---- A hook opens the comm window BEFORE any EP HCCL on this layer ----
        flat_in = self.reshape(routed_input, (-1,))
        flat_in = self._maybe_sync_hook(flat_in, "A")

        # counts a2a — uniform splits, sync (need output for splits compute)
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
        output_splits = self.cast(
            self.sum(num_tokens_per_expert_group_reshaped, dim=-1, keepdim=False),
            mstype.int64,
        )
        num_tokens_per_expert = self.cumsum(
            self.sum(num_tokens_per_expert_group_reshaped, dim=-2, keepdim=False), 0,
        )
        num_tokens_per_expert = self.cast(num_tokens_per_expert, mstype.int64)

        # routing-map a2a — variable splits via the same comm_func path.
        from mindspore.ops.function import comm_func as _comm_func
        input_split_list = self._splits_to_elem_list(input_splits, 1)
        output_split_list = self._splits_to_elem_list(output_splits, 1)
        routing_map = self.reshape(self.cast(sorted_topk_indices, mstype.float32), (-1,))
        routing_map_out = mint.empty(
            (int(sum(output_split_list)),), dtype=routing_map.dtype,
        )
        routing_map_result = _comm_func.all_to_all_single(
            routing_map_out, routing_map,
            output_split_sizes=output_split_list,
            input_split_sizes=input_split_list,
            group=self.ep_group,
            async_op=False,
        )
        routing_map = (
            routing_map_result if isinstance(routing_map_result, ms.Tensor)
            else routing_map_result[0]
        )
        routing_map = self.reshape(routing_map, (1, -1))

        # main token a2a — async / differentiable (returns AsyncCollectiveTensor)
        flat_out = self._async_a2a(flat_in, input_splits, output_splits, cell.hidden_size)
        flat_out = self._maybe_sync_hook(flat_out, "B")
        # First real op on flat_out triggers AsyncCollectiveTensor's
        # ``__ms_dispatch__`` → ``CommHandle.wait()`` → unwrap to plain Tensor.
        global_input_tokens = self.reshape(flat_out, (1, -1, cell.hidden_size))

        # ---- Post-A2A sort + final permute layout ----
        _, sorted_map = self.sort(routing_map)
        _, unsorted_map = self.sort(self.cast(sorted_map, mstype.float32))
        index = self.reshape(sorted_map, (sorted_map.shape[0] * sorted_map.shape[1],))
        global_input_tokens_shape = global_input_tokens.shape
        global_input_tokens = self.reshape(global_input_tokens, (-1, global_input_tokens_shape[-1]))
        global_input_tokens = self.index_select(global_input_tokens, 0, index)
        global_input_tokens = self.reshape(global_input_tokens, (-1, global_input_tokens_shape[-1]))

        self.ctx = (
            probs, unsorted_map, unsort_token_indices_experts,
            input_splits, output_splits, original_shape, pad_size,
        )
        return global_input_tokens, probs, topk_indices, num_tokens_per_expert

    # ------------------------------------------------------------------
    # _token_combine override
    # ------------------------------------------------------------------

    def _token_combine(self, device_mesh, cell, args, routed_output):  # pylint: disable=W0613
        """Combine path with C→D hooks bracketing the combine a2a."""
        from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=C0415
        (
            probs, unsorted_map, unsort_token_indices_experts,
            input_splits, output_splits, original_shape, pad_size,
        ) = self.ctx

        routed_output = self.reshape(routed_output, (1, -1, cell.hidden_size))
        index = self.reshape(unsorted_map, (-1,))
        routed_output_shape = routed_output.shape
        routed_output = self.reshape(routed_output, (-1, routed_output_shape[-1]))
        routed_output = self.index_select(routed_output, 0, index)
        routed_output = self.reshape(
            routed_output, (routed_output_shape[0], -1, routed_output_shape[-1]),
        )

        # ---- Combine a2a bracketed by C/D hooks ----
        flat_in = self.reshape(routed_output, (-1,))
        flat_in = self._maybe_sync_hook(flat_in, "C")
        # Combine uses output→input splits (reverse direction).
        flat_out = self._async_a2a(flat_in, output_splits, input_splits, cell.hidden_size)
        flat_out = self._maybe_sync_hook(flat_out, self._d_hook)
        permutated_local_input_tokens = self.reshape(flat_out, original_shape)

        # ---- Post-combine routing ----
        index = self.reshape(unsort_token_indices_experts, (-1,))
        permutated_local_input_tokens = self.reshape(
            permutated_local_input_tokens,
            (-1, permutated_local_input_tokens.shape[-1]),
        )
        routed_output = self.index_select(permutated_local_input_tokens, 0, index)
        unsort_token_indices_experts_shape = unsort_token_indices_experts.shape
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
