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
"""
PP Schedules - Graph-mode pipeline schedule drivers.

Single module per the house convention (``core/pipeline_parallel/scheduler.py``
keeps its eager-mode schedules the same way): shared P2P / microbatch
machinery in :class:`PipelineScheduleBase` plus one class per algorithm —
:class:`ScheduleGPipe`, :class:`Schedule1F1B`, and the VPP
:class:`ScheduleInterleaved1F1B` today — dispatched by name from
:data:`SCHEDULE_REGISTRY` / ``PassConfig.pp_schedule``.

Both schedules run the per-stage FX subgraphs produced by ``PpPass`` across
microbatches and share the same P2P / loss / gradient contract:

- Model inputs are routed by dataflow: each user input is fed directly to
  every stage that consumes it (resolved by ``PpPass``; a shared input is
  replicated, not shipped over P2P) — every rank receives the full flattened
  user-input list, slices its OWNED tensor inputs per microbatch, and ignores
  the rest. All tensor user inputs must carry the leading batch dim.
- Boundary values are exchanged as ORDERED LISTS: tensors go as-is, int
  scalars (dynamic-shape ``sym_size`` nodes crossing the cut) are packed as
  0-d int64 tensors and unwrapped with ``item()`` on arrival — mirroring how
  ``torch.distributed.pipelining``'s ``PipelineStage`` ships full argument
  lists. The P2P exchange uses eager ``dist.isend``/``irecv`` on the PP
  process group; sends are async so the sweeps overlap receive/compute across
  stages, receives wait before first use, and each microbatch's boundary sends
  are retained under that microbatch's key — forward and backward boundaries
  share the key — until their ``Work`` handles are waited (1F1B: one turn
  after the backward posts, so a forward send rides along; GPipe: at the end
  of the step).
- Microbatch gradients are un-normalized sums while sweeping, then divided by
  ``num_microbatches`` at the end, so the step gradient matches the non-PP
  semantics of a full-batch mean loss.

The schedules differ in their action order — and in the activation lifetime
that order allows:

``ScheduleGPipe``
    1. Forward sweep: run every microbatch's forward (shipping boundary
       activations downstream), holding all forward outputs.
    2. Backward sweep (reverse order): seed ``ones_like(loss)`` on the last
       stage, run each backward against the activations saved by the forward
       sweep, and pass boundary gradients one stage back.

``Schedule1F1B``
    Warmup runs ``min(num_microbatches, pp_degree - stage_idx)`` forwards (a
    stage nearer the front keeps one more microbatch in flight), then the
    steady state alternates one backward with one forward per turn, and a
    cooldown drains the remaining backwards. Interleaving backward earlier
    shrinks the pipeline bubble relative to GPipe. Requires
    ``num_microbatches >= pp_degree``.

Forward activations and backward gradients share one PP process group, whose
per-pair FIFO must stay complementary — at each position one end sends while
the other receives — or the group deadlocks. ``Schedule1F1B`` staggers its
boundary sends to hold that invariant despite the ``pp_degree``-asymmetric
warmup; the placement mirrors ``core/pipeline_parallel/scheduler.py`` and is
spelled out in ``Schedule1F1B.forward``.
"""

__all__ = [
    "PipelineScheduleBase",
    "ScheduleGPipe",
    "Schedule1F1B",
    "ScheduleInterleaved1F1B",
    "SCHEDULE_REGISTRY",
    "get_schedule_class",
]

import bisect
import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import torch
import torch.distributed as dist
from torch import nn


class PipelineScheduleBase(nn.Module):
    """Shared P2P / microbatch machinery for the graph-mode PP schedules.

    Installed by ``PpPass`` as a submodule of the compiled GraphModule and
    invoked through a ``call_module`` node, so the trainer's
    ``graph_module(*flat_inputs)`` dispatches to the concrete schedule
    unchanged. Subclasses implement :meth:`forward` (the action order) from
    the per-microbatch building blocks defined here.

    Args:
        fwd_gm: Stage forward subgraph. Signature per stage:
            stage 0    -> ``(*state, *owned_inputs)`` returning
                          ``(*act_out, *saved)``;
            other      -> ``(*state, *act_in, *owned_inputs)`` returning
                          ``(*act_out, *saved)`` for non-last stages and
                          ``(loss, *saved)`` for the last one, where
                          ``owned_inputs`` are the user inputs routed to
                          this stage (may be empty).
        bwd_gm: Stage backward subgraph. Signature:
            ``(*state, *grad_in, *fwd_outs)`` where ``grad_in`` are the
            received boundary gradients (``ones_like(loss)`` on the last
            stage) and ``fwd_outs`` are the forward subgraph's outputs
            verbatim. Returns ``(*param_grads,)`` plus the boundary
            gradients for every stage except stage 0.
        stage_idx: This rank's stage index (0-based).
        pp_degree: Number of pipeline stages.
        num_state: Number of leading state tensors (stage params/buffers).
        num_trainable: Number of trainable parameters on this stage; the
            backward subgraph returns exactly this many parameter gradients
            first.
        num_send: Number of forward boundary values to ship downstream
            (the forward outputs' prefix); 0 on the last stage.
        grad_send_count: Number of backward boundary values to ship
            upstream (the backward outputs' suffix); 0 on stage 0.
        microbatch_size: Samples per microbatch (leading dim).
        pp_group: PP process group used for all boundary P2P (activations and
            gradients) and for stage->global rank mapping.
        recv_spec: Per-value descriptors for the incoming forward
            boundary: ``("tensor", shape, dtype, device)`` or
            ``("scalar",)``. Empty on stage 0.
        grad_recv_spec: Same, for the incoming backward boundary
            gradients. Empty on the last stage.
        user_input_stages: Per flattened user input (stub arg order), its
            primary consuming stage.
        user_input_stage_sets: Optional per-input owner sets; when omitted,
            each input is owned by its single ``user_input_stages`` entry.
            This stage slices and feeds the inputs mapped to itself.
    """

    def __init__(
        self,
        fwd_gm: nn.Module,
        bwd_gm: nn.Module,
        stage_idx: int,
        pp_degree: int,
        num_state: int,
        num_trainable: int,
        num_send: int,
        grad_send_count: int,
        microbatch_size: int,
        pp_group: Any = None,
        recv_spec: Sequence[Tuple[Any, ...]] = (),
        grad_recv_spec: Sequence[Tuple[Any, ...]] = (),
        user_input_stages: Sequence[int] = (),
        user_input_stage_sets: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        """Store the stage subgraphs, P2P group, and boundary specs."""
        super().__init__()
        self.fwd_gm = fwd_gm
        self.bwd_gm = bwd_gm
        self.stage_idx = stage_idx
        self.pp_degree = pp_degree
        self.num_state = num_state
        self.num_trainable = num_trainable
        self.num_send = num_send
        self.grad_send_count = grad_send_count
        self.microbatch_size = microbatch_size
        self.pp_group = pp_group
        self.recv_spec = list(recv_spec)
        self.grad_recv_spec = list(grad_recv_spec)
        self.user_input_stages = list(user_input_stages)
        self.user_input_stage_sets = (
            [list(stages) for stages in user_input_stage_sets]
            if user_input_stage_sets is not None
            else [[stage] for stage in self.user_input_stages]
        )
        self.is_first = stage_idx == 0
        self.is_last = stage_idx == pp_degree - 1
        # In-flight isend Work handles keyed by microbatch index — forward and
        # backward boundaries of one microbatch share the key; the sent
        # tensors stay referenced so their buffers outlive the send. Schedules
        # release a microbatch's buffers once its sends provably completed
        # (1F1B: next turn's gradient receipt) or at the end of the step.
        self._pending_sends: Dict[int, List[Tuple[Any, torch.Tensor]]] = {}

    def forward(self, *flat_inputs: Any) -> Tuple[torch.Tensor, ...]:
        """Run one pipeline step.

        Args:
            *flat_inputs: ``(*state, *user_inputs)`` for this stage.

        Returns:
            ``(loss, *param_grads)``.

        Raises:
            NotImplementedError: Always; concrete schedules override this.
        """
        raise NotImplementedError

    def map_stage_graphs(self, fn: Callable[[nn.Module], nn.Module]) -> None:
        """Apply ``fn`` to each owned stage subgraph and store the result.

        Terminal backend compilation (``InductorPass``) runs after ``PpPass``
        replaced the joint graph with this schedule stub: the stage subgraphs
        — not the stub — carry the FX compute, so the pass compiles each of
        them through this hook and writes the returned module back (in-place
        compilers may return the same object).
        """
        self.fwd_gm = fn(self.fwd_gm)
        self.bwd_gm = fn(self.bwd_gm)

    # ------------------------------------------------------------------
    # Input / microbatch resolution
    # ------------------------------------------------------------------

    def _split_inputs(
        self, flat_inputs: Sequence[Any]
    ) -> Tuple[Sequence[Any], List[Any]]:
        """Split ``(*state, *user_inputs)`` and validate the input arity."""
        state = flat_inputs[: self.num_state]
        user_inputs = list(flat_inputs[self.num_state :])
        if len(user_inputs) != len(self.user_input_stages):
            raise RuntimeError(
                f"PP schedule expects {len(self.user_input_stages)} user "
                f"inputs (per its routing) but received {len(user_inputs)}"
            )
        return state, user_inputs

    def _resolve_microbatches(
        self, user_inputs: Sequence[Any]
    ) -> Tuple[int, List[Any]]:
        """Derive the microbatch count and this stage's owned user inputs.

        Every rank receives every user input, so the batch size is derived
        identically on all stages from the first tensor input.
        """
        batch_size = next(
            (v.shape[0] for v in user_inputs if isinstance(v, torch.Tensor)), None
        )
        if batch_size is None:
            raise ValueError(
                "PP requires at least one tensor model input to derive the "
                "microbatch split"
            )
        if batch_size % self.microbatch_size != 0:
            raise ValueError(
                f"PP microbatch mismatch: batch size {batch_size} is not "
                f"divisible by pp_microbatch_size {self.microbatch_size}"
            )
        num_microbatches = batch_size // self.microbatch_size
        owned_inputs = [
            v
            for v, stages in zip(user_inputs, self.user_input_stage_sets)
            if self.stage_idx in stages
        ]
        return num_microbatches, owned_inputs

    def _slice_inputs(self, owned_inputs: Sequence[Any], mb_index: int) -> List[Any]:
        """Slice this stage's owned inputs for microbatch ``mb_index``.

        Tensors are sliced along the leading batch dim; non-tensors pass
        through unchanged.
        """
        lo = mb_index * self.microbatch_size
        hi = lo + self.microbatch_size
        return [v[lo:hi] if isinstance(v, torch.Tensor) else v for v in owned_inputs]

    # ------------------------------------------------------------------
    # Per-microbatch building blocks
    # ------------------------------------------------------------------
    #
    # Forward / backward are split into a receive step and a compute step so
    # a schedule can interleave boundary sends with compute independently;
    # ``_forward_microbatch`` / ``_backward_microbatch`` compose them for
    # callers that don't need that control.

    def _recv_activation(self) -> List[Any]:
        """Receive this stage's incoming boundary activations (none on stage 0)."""
        if self.is_first:
            return []
        return self._recv_values(self.recv_spec, src=self.stage_idx - 1)

    def _apply_forward(
        self,
        state: Sequence[Any],
        act_in: Sequence[Any],
        owned_inputs: Sequence[Any],
        mb_index: int,
    ) -> Tuple[Any, ...]:
        """Run one microbatch's forward subgraph against ``act_in``.

        ``owned_inputs`` are this stage's user inputs (see
        ``user_input_stages``): tensors are sliced per microbatch along the
        leading batch dim, non-tensors pass through. They are appended after
        the received activations in the fwd subgraph's arg order.
        """
        owned_mb = self._slice_inputs(owned_inputs, mb_index)
        if self.is_first:
            out = self.fwd_gm(*state, *owned_mb)
        else:
            out = self.fwd_gm(*state, *act_in, *owned_mb)
        return tuple(out)

    def _forward_microbatch(
        self, state: Sequence[Any], owned_inputs: Sequence[Any], mb_index: int
    ) -> Tuple[Any, ...]:
        """Receive activations then run one microbatch's forward subgraph."""
        act_in = self._recv_activation()
        return self._apply_forward(state, act_in, owned_inputs, mb_index)

    def _send_forward(self, out: Sequence[Any], mb_index: int) -> None:
        """Ship microbatch ``mb_index``'s boundary activations downstream."""
        if self.num_send:
            self._send_values(
                out[: self.num_send], dst=self.stage_idx + 1, mb_index=mb_index
            )

    def _recv_gradient(self, fwd_outs: Sequence[Any]) -> List[Any]:
        """Receive the next stage's output grads (a seed on the last stage).

        The traced backward graph seeds its own loss gradient internally, so
        the ``ones_like`` seed on the last stage only fixes the shape.
        """
        if self.is_last:
            return [torch.ones_like(fwd_outs[0])]
        return self._recv_values(self.grad_recv_spec, src=self.stage_idx + 1)

    def _apply_backward(
        self,
        state: Sequence[Any],
        grad_in: Sequence[Any],
        fwd_outs: Sequence[Any],
        mb_index: int,
    ) -> Sequence[torch.Tensor]:
        """Run microbatch ``mb_index``'s backward subgraph and ship grads.

        Gradients flow UPSTREAM, so the previous stage is the destination,
        mirroring the forward direction. The grads returned are un-normalized
        sums over the microbatches; the shipped boundary grads are keyed by
        ``mb_index`` so the schedule can release their buffers later.
        """
        out = self.bwd_gm(*state, *grad_in, *fwd_outs)
        param_grads = out[: self.num_trainable]
        if self.grad_send_count:
            self._send_values(
                out[self.num_trainable :], dst=self.stage_idx - 1, mb_index=mb_index
            )
        return param_grads

    def _backward_microbatch(
        self, state: Sequence[Any], fwd_outs: Sequence[Any], mb_index: int
    ) -> Sequence[torch.Tensor]:
        """Receive boundary grads then run one microbatch's backward subgraph."""
        grad_in = self._recv_gradient(fwd_outs)
        return self._apply_backward(state, grad_in, fwd_outs, mb_index)

    @staticmethod
    def _accumulate_grads(
        accumulated: Sequence[torch.Tensor], param_grads: Sequence[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Sum one microbatch's param grads into the running accumulation."""
        if not accumulated:
            return list(param_grads)
        return [acc + grad for acc, grad in zip(accumulated, param_grads)]

    def _finalize_loss(
        self, fwd_outs_per_mb: Sequence[Sequence[Any]], user_inputs: Sequence[Any]
    ) -> torch.Tensor:
        """Return the step loss (last stage) or a zero placeholder."""
        if self.is_last:
            # On the last stage each forward output tuple starts with the
            # per-microbatch loss.
            losses = [outs[0] for outs in fwd_outs_per_mb]
            return torch.stack(losses).mean()
        return torch.zeros((), device=self._anchor_device(user_inputs))

    @staticmethod
    def _average_grads(
        grads: Sequence[torch.Tensor], num_microbatches: int
    ) -> List[torch.Tensor]:
        """Turn un-normalized sums over microbatches into the mean."""
        return [grad / num_microbatches for grad in grads]

    # ------------------------------------------------------------------
    # P2P helpers
    # ------------------------------------------------------------------

    def _flush_pending_sends(self) -> None:
        """Wait every in-flight isend and drop all buffer references."""
        for sends in self._pending_sends.values():
            for work, _ in sends:
                work.wait()
        self._pending_sends = {}

    def _wait_sends(self, mb_index: int) -> None:
        """Wait microbatch ``mb_index``'s in-flight sends, freeing its buffers.

        Callers must only invoke this once the sends are provably complete
        (e.g. the peer consumed the values), so the wait never blocks.
        """
        for work, _ in self._pending_sends.pop(mb_index, ()):
            work.wait()

    @staticmethod
    def _anchor_device(user_inputs: Sequence[Any]) -> torch.device:
        """Device for the zero-loss placeholder: first tensor user input."""
        for value in user_inputs:
            if isinstance(value, torch.Tensor):
                return value.device
        return torch.device("cpu")

    def _send_values(self, values: Sequence[Any], dst: int, mb_index: int) -> None:
        """Async-send boundary values to the neighbouring stage ``dst``.

        Tensors go as-is (made contiguous, with THAT buffer retained); int
        scalars (dynamic-shape ``sym_size`` nodes) are packed as 0-d int64
        tensors on the first tensor value's device — CPU tensors cannot ride
        an NCCL/NPU-backend group — and unwrapped with ``item()`` on the
        receiving side. The ``Work`` handles are queued under ``mb_index``
        (forward and backward boundaries of a microbatch share the key) and
        the sent tensors kept referenced so their buffers outlive the send;
        the schedule waits them (per turn or at step end) before the buffers
        are released.
        """
        dst_rank = self._global_rank(dst)
        anchor_device = next(
            (v.device for v in values if isinstance(v, torch.Tensor)), None
        )
        for value in values:
            if isinstance(value, torch.Tensor):
                tensor = value.contiguous()
            else:
                tensor = torch.tensor(value, dtype=torch.int64, device=anchor_device)
            work = dist.isend(tensor, dst=dst_rank, group=self.pp_group)
            self._pending_sends.setdefault(mb_index, []).append((work, tensor))

    def _recv_values(self, spec: Sequence[Tuple[Any, ...]], src: int) -> List[Any]:
        """Receive boundary values from the neighbouring stage ``src``, in order.

        The buffer for each entry comes from ``spec`` (built by ``PpPass``
        from the boundary values' fake-tensor ``val`` metas, including the
        device each value was traced on); scalar entries arrive as 0-d
        int64 tensors and are unwrapped to Python ints so the graph sees
        the same types it was traced with. Scalar buffers are anchored to
        the first tensor entry's device (``PpPass._value_spec`` guarantees
        a non-empty list carries at least one tensor).
        """
        src_rank = self._global_rank(src)
        anchor_device = next((entry[3] for entry in spec if entry[0] == "tensor"), None)
        values: List[Any] = []
        for entry in spec:
            if entry[0] == "scalar":
                buffer = torch.empty((), dtype=torch.int64, device=anchor_device)
            else:
                _, shape, dtype, device = entry
                buffer = torch.empty(shape, dtype=dtype, device=device)
            work = dist.irecv(buffer, src=src_rank, group=self.pp_group)
            work.wait()
            values.append(int(buffer.item()) if entry[0] == "scalar" else buffer)
        return values

    def _global_rank(self, stage_idx: int) -> int:
        """Map a stage index to its global rank within the PP group."""
        if self.pp_group is None:
            return stage_idx
        return dist.get_global_rank(self.pp_group, stage_idx)


class ScheduleGPipe(PipelineScheduleBase):
    """GPipe driver over two per-stage FX subgraphs (forward / backward).

    Runs ALL forwards before ANY backward, so every microbatch's forward
    outputs (boundary activations + saved values) are held until the backward
    sweep completes — activation memory scales with ``num_microbatches``.
    Larger ``microbatch_size`` trades pipeline bubble for fewer P2P
    round-trips and fewer retained activations.
    """

    def forward(self, *flat_inputs: Any) -> Tuple[torch.Tensor, ...]:
        """Run one full GPipe step (fwd sweep then bwd sweep).

        Args:
            flat_inputs: ``(*state, *user_inputs)`` — the same surface
                ``run_traced_graph`` feeds the compiled graph, with
                ``user_inputs`` the flattened model inputs in trace order.

        Returns:
            ``(loss, *param_grads)``. On non-last stages the loss is a zero
            scalar placeholder (the real loss lives on the last stage);
            parameter gradients are the microbatch-averaged accumulation.
        """
        state, user_inputs = self._split_inputs(flat_inputs)
        num_microbatches, owned_inputs = self._resolve_microbatches(user_inputs)

        fwd_outs_per_mb = self._forward_sweep(state, owned_inputs, num_microbatches)
        grads = self._backward_sweep(state, fwd_outs_per_mb, num_microbatches)

        self._flush_pending_sends()

        loss = self._finalize_loss(fwd_outs_per_mb, user_inputs)
        return (loss, *grads)

    def _forward_sweep(
        self,
        state: Sequence[Any],
        owned_inputs: Sequence[Any],
        num_microbatches: int,
    ) -> List[Tuple[Any, ...]]:
        """Run forward microbatch 0..N-1, shipping boundary values downstream.

        Returns per-microbatch forward output tuples (replayed verbatim
        into the backward subgraph by ``_backward_sweep``).
        """
        fwd_outs_per_mb: List[Tuple[Any, ...]] = []
        for mb_index in range(num_microbatches):
            out = self._forward_microbatch(state, owned_inputs, mb_index)
            fwd_outs_per_mb.append(out)
            self._send_forward(out, mb_index)
        return fwd_outs_per_mb

    def _backward_sweep(
        self,
        state: Sequence[Any],
        fwd_outs_per_mb: Sequence[Tuple[Any, ...]],
        num_microbatches: int,
    ) -> List[torch.Tensor]:
        """Run backward microbatches N-1..0, accumulating full-batch gradients.

        The traced backward graph seeds its own loss gradient internally
        (an ``ones_like`` over the saved loss), so the grads that come back
        are un-normalized sums over the microbatches. Dividing the
        accumulation by ``num_microbatches`` turns that into the mean, which
        is exactly the full-batch mean-loss gradient the non-PP path
        produces.
        """
        grads: List[torch.Tensor] = []
        for mb_index in reversed(range(num_microbatches)):
            grads = self._accumulate_grads(
                grads,
                self._backward_microbatch(state, fwd_outs_per_mb[mb_index], mb_index),
            )
        return self._average_grads(grads, num_microbatches)


class Schedule1F1B(PipelineScheduleBase):
    """1F1B driver over two per-stage FX subgraphs (forward / backward).

    Warmup runs ``min(num_microbatches, pp_degree - stage_idx)`` forwards
    (one on the last stage), the steady state alternates one backward then
    one forward per turn, and the cooldown drains the remaining backwards.
    Running each backward as soon as its forward output is available shrinks
    the pipeline bubble relative to GPipe's fill-drain order.

    Requires ``num_microbatches >= pp_degree`` (checked at runtime), the same
    constraint ``torch.distributed.pipelining.Schedule1F1B`` enforces.

    Note:
        Unlike GPipe, each microbatch's saved activations are freed as soon
        as its backward consumes them. Its boundary sends share one key for
        both directions and are released one turn after the backward posts —
        the next turn's gradient receipt proves the previous microbatch's
        transfers completed, so the wait cannot deadlock. Peak memory
        therefore tracks the warmup depth (``pp_degree - stage_idx``) instead
        of ``num_microbatches``; only the final microbatch's sends are still
        waited at the end of the step. The last stage has no downstream
        gradient to mark that progress, so its per-turn wait may briefly block
        on the upstream peer.
    """

    def forward(self, *flat_inputs: Any) -> Tuple[torch.Tensor, ...]:
        """Run one full 1F1B step (warmup -> steady 1B1F -> cooldown).

        Args:
            flat_inputs: ``(*state, *user_inputs)`` — the same surface
                ``run_traced_graph`` feeds the compiled graph, with
                ``user_inputs`` the flattened model inputs in trace order.

        Returns:
            ``(loss, *param_grads)``. On non-last stages the loss is a zero
            scalar placeholder (the real loss lives on the last stage);
            parameter gradients are the microbatch-averaged accumulation.

        Raises:
            ValueError: When ``num_microbatches < pp_degree`` (the schedule
                cannot fill the pipeline).
        """
        state, user_inputs = self._split_inputs(flat_inputs)
        num_microbatches, owned_inputs = self._resolve_microbatches(user_inputs)
        if num_microbatches < self.pp_degree:
            raise ValueError(
                f"1F1B requires at least pp_degree={self.pp_degree} "
                f"microbatches, got {num_microbatches} "
                f"(batch {num_microbatches * self.microbatch_size} with "
                f"pp_microbatch_size={self.microbatch_size}); lower "
                f"pp_microbatch_size or use the gpipe schedule"
            )

        fwd_outs_per_mb: List[Any] = [None] * num_microbatches
        grads: List[torch.Tensor] = []
        fwd_index = 0
        bwd_index = 0
        # Forward output whose boundary activation has not shipped yet, with
        # its microbatch index; the send is deferred one slot to keep the
        # pair's FIFO complementary.
        pending_index = -1
        pending_out: Any = None
        even = self.stage_idx % 2 == 0

        # Warmup: fill the pipeline with ``pp_degree - stage_idx`` in-flight
        # microbatches (one on the last stage) before the first backward.
        # Even stages ship each activation right after its forward, except
        # the last; odd stages ship the previous one just before the next
        # forward. The parity stagger keeps neighbours complementary.
        warmup = min(num_microbatches, self.pp_degree - self.stage_idx)
        for k in range(warmup):
            act_in = self._recv_activation()
            if not even and k > 0:
                self._send_forward(pending_out, pending_index)
                pending_out = None
            out = self._apply_forward(state, act_in, owned_inputs, fwd_index)
            fwd_outs_per_mb[fwd_index] = out
            fwd_index += 1
            if even:
                if k != warmup - 1:
                    self._send_forward(out, fwd_index - 1)
                else:
                    pending_index, pending_out = fwd_index - 1, out
            else:
                pending_index, pending_out = fwd_index - 1, out

        # Steady turn: recv_grad (+ release the previous turn's boundary
        # buffers), send_activation, backward, send_grad, recv_activation,
        # forward.
        for _ in range(num_microbatches - warmup):
            grad_in = self._recv_gradient(fwd_outs_per_mb[bwd_index])
            if bwd_index > 0:
                # This gradient's arrival proves the peer consumed microbatch
                # ``bwd_index - 1``'s boundary values on both directions, so
                # its in-flight sends completed and their buffers can go.
                self._wait_sends(bwd_index - 1)
            if pending_out is not None:
                self._send_forward(pending_out, pending_index)
                pending_out = None
            grads = self._accumulate_grads(
                grads,
                self._apply_backward(
                    state, grad_in, fwd_outs_per_mb[bwd_index], bwd_index
                ),
            )
            # Backward was the last consumer of these outputs; keep only the
            # last stage's loss scalar for _finalize_loss.
            fwd_outs_per_mb[bwd_index] = (
                fwd_outs_per_mb[bwd_index][:1] if self.is_last else None
            )
            bwd_index += 1
            act_in = self._recv_activation()
            out = self._apply_forward(state, act_in, owned_inputs, fwd_index)
            fwd_outs_per_mb[fwd_index] = out
            fwd_index += 1
            pending_index, pending_out = fwd_index - 1, out
        # Drop the frame-local reference to the final forward's outputs (the
        # warmup's when the steady state never runs); it would otherwise pin
        # them through the cooldown.
        out = None

        # Cooldown: drain the remaining backwards; the final forward's
        # activation ships on the first turn.
        for _ in range(warmup):
            grad_in = self._recv_gradient(fwd_outs_per_mb[bwd_index])
            if bwd_index > 0:
                # Same release invariant as the steady state.
                self._wait_sends(bwd_index - 1)
            if pending_out is not None:
                self._send_forward(pending_out, pending_index)
                pending_out = None
            grads = self._accumulate_grads(
                grads,
                self._apply_backward(
                    state, grad_in, fwd_outs_per_mb[bwd_index], bwd_index
                ),
            )
            # Backward was the last consumer of these outputs; keep only the
            # last stage's loss scalar for _finalize_loss.
            fwd_outs_per_mb[bwd_index] = (
                fwd_outs_per_mb[bwd_index][:1] if self.is_last else None
            )
            bwd_index += 1

        self._flush_pending_sends()

        loss = self._finalize_loss(fwd_outs_per_mb, user_inputs)
        return (loss, *self._average_grads(grads, num_microbatches))


@dataclass(frozen=True)
class _Action:
    """One graph-mode pipeline action."""

    kind: str
    stage: int
    micro: int


@dataclass
class _InterleavedStep:
    """Mutable buffers shared by the VPP step's per-action handlers."""

    state: Sequence[Any]
    user_inputs: Sequence[Any]
    act_cache: Dict[Tuple[int, int], List[Any]]
    grad_cache: Dict[Tuple[int, int], List[Any]]
    fwd_outs: Dict[Tuple[int, int], Tuple[Any, ...]]
    bwd_outs: Dict[Tuple[int, int], Tuple[Any, ...]]
    grads: List[List[torch.Tensor]]
    losses: List[Optional[torch.Tensor]]


_FWD = "fwd"
_BWD = "bwd"
_FWD_SEND = "fwd_send"
_FWD_RECV = "fwd_recv"
_BWD_SEND = "bwd_send"
_BWD_RECV = "bwd_recv"


def _round_layout(
    num_microbatches: int, pp_degree: int, local_stages: int
) -> Tuple[List[int], List[int]]:
    """Per-round microbatch counts and their local-stage prefix sum.

    Mirrors ``core.pipeline_parallel.scheduler.ScheduleInterleaved1F1B`` so
    the graph-mode VPP order matches the eager schedule's warmup/cooldown
    rhythm, including the ``num_microbatches < pp_degree`` case.
    """
    rounds = max(1, num_microbatches // pp_degree)
    if num_microbatches < pp_degree:
        base = num_microbatches - pp_degree
        remainder = 0
    else:
        extra = num_microbatches % pp_degree
        base = extra // rounds
        remainder = extra % rounds
    per_round = [pp_degree + base + (1 if i < remainder else 0) for i in range(rounds)]
    accu = [x * local_stages for x in itertools.accumulate(per_round)]
    accu.insert(0, 0)
    return per_round, accu


def _warmup_ops(
    stage_index: int,
    num_microbatches: int,
    pp_degree: int,
    local_stages: int,
    per_round: Sequence[int],
) -> int:
    """Number of forwards this physical rank runs before its first backward."""
    last_stage_warmup = (local_stages - 1) * per_round[0]
    depth = pp_degree - 1 - stage_index
    return min(last_stage_warmup + 2 * depth, num_microbatches * local_stages)


def _forward_stage_index(
    op_index: int,
    stage_index: int,
    per_round: Sequence[int],
    accu: Sequence[int],
    pp_degree: int,
) -> int:
    """Virtual stage a forward op at ``op_index`` belongs to."""
    round_index = bisect.bisect_right(accu, op_index) - 1
    local_index = (op_index - accu[round_index]) // per_round[round_index]
    return local_index * pp_degree + stage_index


def _backward_stage_index(
    op_index: int,
    stage_index: int,
    per_round: Sequence[int],
    accu: Sequence[int],
    pp_degree: int,
    local_stages: int,
) -> int:
    """Virtual stage a backward op at ``op_index`` belongs to."""
    round_index = bisect.bisect_right(accu, op_index) - 1
    local_index = (op_index - accu[round_index]) // per_round[round_index]
    local_index = local_stages - 1 - local_index
    return local_index * pp_degree + stage_index


def _emit_warmup_actions(
    stage_index: int,
    warmup_ops: int,
    num_microbatches: int,
    pp_degree: int,
    per_round: Sequence[int],
    accu: Sequence[int],
    fwd_micro: Dict[int, int],
) -> List[Optional[_Action]]:
    """Emit pure-forward warmup actions with the short-pipeline padding."""
    actions: List[Optional[_Action]] = []
    short = num_microbatches < pp_degree
    last_micro = num_microbatches - 1
    last_stage = pp_degree - 1
    bubble = pp_degree - num_microbatches
    for op_index in range(warmup_ops):
        fwd_stage = _forward_stage_index(
            op_index, stage_index, per_round, accu, pp_degree
        )
        micro = fwd_micro[fwd_stage]
        actions.append(_Action(_FWD, fwd_stage, micro))
        pad = (
            short
            and micro == last_micro
            and (op_index != warmup_ops - 1 or stage_index == last_stage)
        )
        if pad:
            actions.extend([None] * bubble)
        fwd_micro[fwd_stage] += 1
    return actions


def _emit_1f1b_actions(
    stage_index: int,
    warmup_ops: int,
    fwd_bwd_ops: int,
    num_microbatches: int,
    pp_degree: int,
    local_stages: int,
    per_round: Sequence[int],
    accu: Sequence[int],
    fwd_micro: Dict[int, int],
    bwd_micro: Dict[int, int],
) -> List[Optional[_Action]]:
    """Emit the interleaved forward/backward steady state."""
    actions: List[Optional[_Action]] = []
    short = num_microbatches < pp_degree
    last_micro = num_microbatches - 1
    last_stage = pp_degree - 1
    bubble = 2 * (pp_degree - num_microbatches)
    for op_index in range(warmup_ops, warmup_ops + fwd_bwd_ops):
        fwd_stage = _forward_stage_index(
            op_index, stage_index, per_round, accu, pp_degree
        )
        actions.append(_Action(_FWD, fwd_stage, fwd_micro[fwd_stage]))
        fwd_micro[fwd_stage] += 1
        bwd_stage = _backward_stage_index(
            op_index - warmup_ops,
            stage_index,
            per_round,
            accu,
            pp_degree,
            local_stages,
        )
        actions.append(_Action(_BWD, bwd_stage, bwd_micro[bwd_stage]))
        bwd_micro[bwd_stage] += 1
        if (
            short
            and bwd_micro[bwd_stage] - 1 == last_micro
            and stage_index == last_stage
        ):
            actions.extend([None] * bubble)
    return actions


def _emit_cooldown_actions(
    stage_index: int,
    warmup_ops: int,
    fwd_bwd_ops: int,
    total_ops: int,
    num_microbatches: int,
    pp_degree: int,
    local_stages: int,
    per_round: Sequence[int],
    accu: Sequence[int],
    bwd_micro: Dict[int, int],
) -> List[Optional[_Action]]:
    """Emit pure-backward cooldown actions with the matching padding."""
    actions: List[Optional[_Action]] = []
    short = num_microbatches < pp_degree
    last_micro = num_microbatches - 1
    bubble = 2 * (pp_degree - num_microbatches)
    for op_index in range(warmup_ops + fwd_bwd_ops, total_ops):
        actions.append(None)
        bwd_stage = _backward_stage_index(
            op_index - warmup_ops,
            stage_index,
            per_round,
            accu,
            pp_degree,
            local_stages,
        )
        micro = bwd_micro[bwd_stage]
        actions.append(_Action(_BWD, bwd_stage, micro))
        bwd_micro[bwd_stage] += 1
        if short and micro == last_micro:
            actions.extend([None] * bubble)
    return actions


def _interleaved_stage_order(
    stage_index: int,
    num_microbatches: int,
    pp_degree: int,
    num_stages: int,
) -> List[Optional[_Action]]:
    """Build one physical rank's compute order for VPP (interleaved 1F1B)."""
    local_stages = num_stages // pp_degree
    per_round, accu = _round_layout(num_microbatches, pp_degree, local_stages)
    warmup_ops = _warmup_ops(
        stage_index, num_microbatches, pp_degree, local_stages, per_round
    )
    fwd_bwd_ops = local_stages * num_microbatches - warmup_ops
    total_ops = 2 * warmup_ops + fwd_bwd_ops
    fwd_micro: Dict[int, int] = defaultdict(int)
    bwd_micro: Dict[int, int] = defaultdict(int)

    actions: List[Optional[_Action]] = [None] * stage_index
    actions.extend(
        _emit_warmup_actions(
            stage_index,
            warmup_ops,
            num_microbatches,
            pp_degree,
            per_round,
            accu,
            fwd_micro,
        )
    )
    depth = pp_degree - 1 - stage_index
    actions.extend([None] * max(0, 2 * depth - num_microbatches))
    actions.extend([None] * (pp_degree - 1 - stage_index))
    actions.extend(
        _emit_1f1b_actions(
            stage_index,
            warmup_ops,
            fwd_bwd_ops,
            num_microbatches,
            pp_degree,
            local_stages,
            per_round,
            accu,
            fwd_micro,
            bwd_micro,
        )
    )
    actions.extend(
        _emit_cooldown_actions(
            stage_index,
            warmup_ops,
            fwd_bwd_ops,
            total_ops,
            num_microbatches,
            pp_degree,
            local_stages,
            per_round,
            accu,
            bwd_micro,
        )
    )
    return actions


def _insert_interleaved_comms(
    schedule: Dict[int, List[Optional[_Action]]],
    num_stages: int,
    pp_degree: int,
) -> Dict[int, List[Optional[_Action]]]:
    """Insert SEND/RECV actions with the eager schedule's column scan.

    Neighbouring ranks alternate send-first / recv-first so the per-peer P2P
    FIFO stays complementary on the shared PP group; a naive per-action
    insertion would make both ends send-first and deadlock.
    """
    ordered = {rank: [] for rank in range(pp_degree)}

    def _peer_of(stage: int, delta: int) -> Optional[int]:
        target = stage + delta
        if target < 0 or target >= num_stages:
            return None
        peer = target % pp_degree
        return peer if peer != stage % pp_degree else None

    def _insert(
        action: Optional[_Action], rank: int, out: Dict[int, List[Optional[_Action]]]
    ) -> None:
        if action is None:
            return
        if action.kind == _FWD:
            peer = _peer_of(action.stage, 1)
            if peer is not None:
                out[rank].append(_Action(_FWD_SEND, action.stage, action.micro))
                out[peer].append(_Action(_FWD_RECV, action.stage + 1, action.micro))
        elif action.kind == _BWD:
            peer = _peer_of(action.stage, -1)
            if peer is not None:
                out[rank].append(_Action(_BWD_SEND, action.stage, action.micro))
                out[peer].append(_Action(_BWD_RECV, action.stage - 1, action.micro))

    max_length = max(len(actions) for actions in schedule.values())
    for time_step in range(max_length):
        current: Dict[int, Optional[_Action]] = {}
        for rank in range(pp_degree):
            if time_step < len(schedule[rank]):
                action = schedule[rank][time_step]
                current[rank] = action
                ordered[rank].append(action)
            else:
                current[rank] = None
        for rank in range(0, pp_degree, 2):
            _insert(current.get(rank), rank, ordered)
        for rank in range(1, pp_degree, 2):
            _insert(current.get(rank), rank, ordered)
    return ordered


def _interleaved_action_order(
    num_stages: int, pp_degree: int, num_microbatches: int
) -> Dict[int, List[Optional[_Action]]]:
    """Return the full per-physical-rank VPP action schedule."""
    if num_stages % pp_degree != 0:
        raise ValueError(
            f"num_stages ({num_stages}) must be divisible by pp_degree ({pp_degree})"
        )
    schedule = {
        rank: _interleaved_stage_order(rank, num_microbatches, pp_degree, num_stages)
        for rank in range(pp_degree)
    }
    return _insert_interleaved_comms(schedule, num_stages, pp_degree)


class ScheduleInterleaved1F1B(PipelineScheduleBase):
    """Graph-mode VPP (Interleaved 1F1B) schedule.

    Each physical rank owns ``num_stages / pp_degree`` virtual stages. The
    model is split into ``num_stages`` chunks and rank ``r`` owns virtual
    stages ``r, r + pp_degree, ...``; the action order is the classic
    interleaved-1F1B warmup / steady / cooldown sequence, and P2P actions are
    inserted with the same complementary column scan as the eager schedule
    (``core/pipeline_parallel/scheduler.py``).

    Args:
        fwd_gms: Forward subgraphs, ordered by ``stage_indices``.
        bwd_gms: Backward subgraphs, same order.
        stage_indices: Virtual stages this physical rank owns, ascending.
        num_stages: Total virtual stage count (``pp_degree * chunks``).
        pp_degree: Number of physical pipeline ranks.
        state_positions: Per-stage positions into the flat state argument.
        num_states: Per-stage state-tensor count.
        num_trainables: Per-stage trainable-parameter count.
        num_sends: Per-stage forward boundary value count.
        grad_send_counts: Per-stage backward boundary value count.
        microbatch_size: Samples per microbatch.
        pp_group: PP process group.
        recv_specs: Per-stage forward receive descriptors.
        grad_recv_specs: Per-stage gradient receive descriptors.
        user_input_stages: Per user input, its consuming virtual stage.
        grad_owner: For each emitted gradient in flat state order, the index
            in ``stage_indices`` that owns it.
    """

    def __init__(
        self,
        fwd_gms: Sequence[nn.Module],
        bwd_gms: Sequence[nn.Module],
        stage_indices: Sequence[int],
        num_stages: int,
        pp_degree: int,
        state_positions: Sequence[Sequence[int]],
        num_states: Sequence[int],
        num_trainables: Sequence[int],
        num_sends: Sequence[int],
        grad_send_counts: Sequence[int],
        microbatch_size: int,
        pp_group: Any = None,
        recv_specs: Sequence[Sequence[Tuple[Any, ...]]] = (),
        grad_recv_specs: Sequence[Sequence[Tuple[Any, ...]]] = (),
        user_input_stages: Sequence[int] = (),
        grad_owner: Sequence[int] = (),
        user_input_stage_sets: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        """Store the per-chunk subgraphs and boundary contracts."""
        super().__init__(
            fwd_gms[0],
            bwd_gms[0],
            stage_idx=stage_indices[0],
            pp_degree=pp_degree,
            num_state=sum(num_states),
            num_trainable=num_trainables[0],
            num_send=num_sends[0],
            grad_send_count=grad_send_counts[0],
            microbatch_size=microbatch_size,
            pp_group=pp_group,
            recv_spec=recv_specs[0] if recv_specs else (),
            grad_recv_spec=grad_recv_specs[0] if grad_recv_specs else (),
            user_input_stages=user_input_stages,
            user_input_stage_sets=user_input_stage_sets,
        )
        self.fwd_gm = None
        self.bwd_gm = None
        self.fwd_gms = nn.ModuleList(fwd_gms)
        self.bwd_gms = nn.ModuleList(bwd_gms)
        self.stage_indices = list(stage_indices)
        self.num_stages = num_stages
        self.pp_degree = pp_degree
        self.rank = self.stage_indices[0] % pp_degree if self.stage_indices else 0
        self.state_positions = [list(positions) for positions in state_positions]
        self.num_states = list(num_states)
        self.num_trainables = list(num_trainables)
        self.num_sends = list(num_sends)
        self.grad_send_counts = list(grad_send_counts)
        self.microbatch_size = microbatch_size
        self.pp_group = pp_group
        self.recv_specs = [list(spec) for spec in recv_specs]
        self.grad_recv_specs = [list(spec) for spec in grad_recv_specs]
        self.user_input_stages = list(user_input_stages)
        self.user_input_stage_sets = (
            [list(stages) for stages in user_input_stage_sets]
            if user_input_stage_sets is not None
            else [[stage] for stage in self.user_input_stages]
        )
        self.grad_owner = list(grad_owner)
        self.num_state = sum(self.num_states)
        self._owned_pos = {v: i for i, v in enumerate(self.stage_indices)}
        self._pending_sends: Dict[Any, List[Tuple[Any, torch.Tensor]]] = {}

    def map_stage_graphs(self, fn: Callable[[nn.Module], nn.Module]) -> None:
        """Apply ``fn`` to every virtual stage's fwd and bwd subgraph."""
        self.fwd_gms = nn.ModuleList([fn(gm) for gm in self.fwd_gms])
        self.bwd_gms = nn.ModuleList([fn(gm) for gm in self.bwd_gms])

    def _resolve_num_microbatches(self, user_inputs: Sequence[Any]) -> int:
        """Derive the microbatch count from the first tensor model input."""
        batch_size = next(
            (v.shape[0] for v in user_inputs if isinstance(v, torch.Tensor)), None
        )
        if batch_size is None:
            raise ValueError(
                "PP requires at least one tensor model input to derive the "
                "microbatch split"
            )
        if batch_size % self.microbatch_size != 0:
            raise ValueError(
                f"PP microbatch mismatch: batch size {batch_size} is not "
                f"divisible by pp_microbatch_size {self.microbatch_size}"
            )
        return batch_size // self.microbatch_size

    def _apply_stage_forward(
        self,
        index: int,
        stage_idx: int,
        state: Sequence[Any],
        act_in: Sequence[Any],
        owned_inputs: Sequence[Any],
        mb_index: int,
    ) -> Tuple[Any, ...]:
        """Run one virtual stage's forward subgraph for one microbatch."""
        owned_mb = self._slice_inputs(owned_inputs, mb_index)
        if stage_idx == 0:
            out = self.fwd_gms[index](*state, *owned_mb)
        else:
            out = self.fwd_gms[index](*state, *act_in, *owned_mb)
        return tuple(out)

    def forward(self, *flat_inputs: Any) -> Tuple[torch.Tensor, ...]:
        """Run one full interleaved-1F1B step across this rank's chunks."""
        state, user_inputs = self._split_inputs(flat_inputs)
        num_microbatches = self._resolve_num_microbatches(user_inputs)
        order = _interleaved_action_order(
            self.num_stages, self.pp_degree, num_microbatches
        )[self.rank]
        step = _InterleavedStep(
            state=state,
            user_inputs=user_inputs,
            act_cache={},
            grad_cache={},
            fwd_outs={},
            bwd_outs={},
            grads=[[] for _ in self.stage_indices],
            losses=[None] * num_microbatches,
        )

        for action in order:
            if action is not None:
                self._run_action(action, step)

        self._flush_pending_sends()
        loss = self._finalize_interleaved_loss(step.losses, user_inputs)
        return (loss, *self._ordered_grads(step.grads, num_microbatches))

    def _run_action(self, action: _Action, step: _InterleavedStep) -> None:
        """Apply one forward or backward action to the step buffers."""
        stage_idx = action.stage
        mb_index = action.micro
        key = (stage_idx, mb_index)
        index = self._owned_pos[stage_idx]
        if action.kind == _FWD_RECV:
            step.act_cache[key] = self._recv_values(
                self.recv_specs[index], src=(stage_idx - 1) % self.pp_degree
            )
        elif action.kind == _FWD:
            state_v = [step.state[pos] for pos in self.state_positions[index]]
            owned_inputs = [
                value
                for value, owners in zip(step.user_inputs, self.user_input_stage_sets)
                if stage_idx in owners
            ]
            step.fwd_outs[key] = self._apply_stage_forward(
                index,
                stage_idx,
                state_v,
                step.act_cache.pop(key, []),
                owned_inputs,
                mb_index,
            )
        elif action.kind == _FWD_SEND:
            self._send_values(
                step.fwd_outs[key][: self.num_sends[index]],
                dst=(stage_idx + 1) % self.pp_degree,
                mb_index=(_FWD_SEND, stage_idx, mb_index),
            )
        elif action.kind == _BWD_RECV:
            step.grad_cache[key] = self._recv_values(
                self.grad_recv_specs[index],
                src=(stage_idx + 1) % self.pp_degree,
            )
        elif action.kind == _BWD:
            state_v = [step.state[pos] for pos in self.state_positions[index]]
            if stage_idx == self.num_stages - 1:
                grad_in = [torch.ones_like(step.fwd_outs[key][0])]
            else:
                grad_in = step.grad_cache.pop(key)
            out = self.bwd_gms[index](*state_v, *grad_in, *step.fwd_outs[key])
            trainable = self.num_trainables[index]
            step.grads[index] = self._accumulate_grads(
                step.grads[index], out[:trainable]
            )
            if self.grad_send_counts[index]:
                step.bwd_outs[key] = tuple(out[trainable:])
            if stage_idx == self.num_stages - 1:
                step.losses[mb_index] = step.fwd_outs[key][0]
            step.fwd_outs.pop(key, None)
        elif action.kind == _BWD_SEND:
            self._send_values(
                step.bwd_outs.pop(key),
                dst=(stage_idx - 1) % self.pp_degree,
                mb_index=(_BWD_SEND, stage_idx, mb_index),
            )

    def _finalize_interleaved_loss(
        self,
        losses: Sequence[Optional[torch.Tensor]],
        user_inputs: Sequence[Any],
    ) -> torch.Tensor:
        """Return the mean last-stage loss or this rank's zero placeholder."""
        if self.num_stages - 1 in self._owned_pos:
            return torch.stack(losses).mean()
        return torch.zeros((), device=self._anchor_device(user_inputs))

    def _ordered_grads(
        self,
        grads: Sequence[Sequence[torch.Tensor]],
        num_microbatches: int,
    ) -> List[torch.Tensor]:
        """Average per-stage grads and flatten them into state-FQN order."""
        averaged = [
            self._average_grads(stage_grads, num_microbatches) for stage_grads in grads
        ]
        pointers = [0] * len(averaged)
        ordered: List[torch.Tensor] = []
        for owner in self.grad_owner:
            ordered.append(averaged[owner][pointers[owner]])
            pointers[owner] += 1
        return ordered


SCHEDULE_REGISTRY: Dict[str, Type[PipelineScheduleBase]] = {
    "gpipe": ScheduleGPipe,
    "1f1b": Schedule1F1B,
    "interleaved_1f1b": ScheduleInterleaved1F1B,
    "vpp": ScheduleInterleaved1F1B,
}


def get_schedule_class(name: str) -> Type[PipelineScheduleBase]:
    """Resolve a ``PassConfig.pp_schedule`` name to its schedule class.

    Args:
        name: Schedule name, case-insensitive (``"gpipe"`` / ``"1f1b"``).

    Returns:
        The schedule class.

    Raises:
        ValueError: When ``name`` is not a registered schedule.
    """
    key = str(name).lower()
    schedule_cls = SCHEDULE_REGISTRY.get(key)
    if schedule_cls is None:
        raise ValueError(
            f"Unknown pp_schedule {name!r}; expected one of {sorted(SCHEDULE_REGISTRY)}"
        )
    return schedule_cls
