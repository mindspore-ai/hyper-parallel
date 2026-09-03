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
machinery plus one class per algorithm — ``ScheduleGPipe`` today;
``Schedule1F1B`` etc. land here as follow-ups, dispatched by name from
``PassConfig``.

``ScheduleGPipe`` runs the per-stage FX subgraphs produced by ``PpPass``
across microbatches:

1. Forward sweep: stage 0 slices the step batch into microbatches and runs
   its forward subgraph per microbatch, sending each boundary value list to
   the next stage with async ``dist.isend``; intermediate stages receive,
   compute, and forward; the last stage additionally computes the
   per-microbatch loss.
2. Backward sweep (reverse order): the last stage feeds ``ones_like(loss)``
   into its backward subgraph, accumulates parameter gradients and sends
   the boundary gradients to the previous stage; earlier stages receive the
   gradients, run their backward subgraph against the activations saved
   from the forward sweep, and pass their boundary gradients one stage
   back.

Boundary values are exchanged as ORDERED LISTS: tensors go as-is, int
scalars (dynamic-shape ``sym_size`` nodes crossing the cut) are packed as
0-d int64 tensors and unwrapped with ``item()`` on arrival — mirroring how
``torch.distributed.pipelining``'s ``PipelineStage`` ships full argument
lists. The P2P exchange uses eager ``dist.isend``/``irecv`` on the PP
process group; sends are async (``Work`` handles waited at the end of the
step) so the forward sweep overlaps receive/compute across stages,
receives wait before first use.

Microbatch gradients are un-normalized sums while sweeping, then divided by
``num_microbatches`` at the end, so the step gradient matches the non-PP
semantics of a full-batch mean loss.
"""

from typing import Any, List, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import nn


class ScheduleGPipe(nn.Module):
    """GPipe driver over two per-stage FX subgraphs (forward / backward).

    Installed by ``PpPass`` as a submodule of the compiled GraphModule and
    invoked through a ``call_module`` node, so the trainer's
    ``graph_module(*flat_inputs)`` dispatches here unchanged.

    Args:
        fwd_gm: Stage forward subgraph. Signature per stage:
            stage 0    -> ``(*state, input_mb)`` returning
                          ``(*act_out, *saved)``;
            middle     -> ``(*state, *act_in)`` returning
                          ``(*act_out, *saved)``;
            last       -> ``(*state, *act_in, label_mb)`` returning
                          ``(loss, *saved)``.
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
        pp_group: PP process group object used for isend/irecv.
        recv_spec: Per-value descriptors for the incoming forward
            boundary: ``("tensor", shape, dtype, device)`` or
            ``("scalar",)``. Empty on stage 0.
        grad_recv_spec: Same, for the incoming backward boundary
            gradients. Empty on the last stage.

    Note:
        GPipe runs ALL forwards before ANY backward, so every microbatch's
        forward outputs (boundary activations + saved values) are held
        until the backward sweep completes — activation memory scales with
        ``num_microbatches``. Larger ``microbatch_size`` trades pipeline
        bubble for fewer P2P round-trips and fewer retained activations.
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
        self.is_first = stage_idx == 0
        self.is_last = stage_idx == pp_degree - 1
        # Work handles of in-flight isend ops; tensors are kept referenced
        # so the underlying buffers stay alive until the send completes.
        self._pending_sends: List[Tuple[Any, torch.Tensor]] = []

    def forward(self, *flat_inputs: Any) -> Tuple[torch.Tensor, ...]:
        """Run one full GPipe step (fwd sweep then bwd sweep).

        Args:
            flat_inputs: ``(*state, input_batch, label_batch)`` — the same
                surface ``run_traced_graph`` feeds the compiled graph.

        Returns:
            ``(loss, *param_grads)``. On non-last stages the loss is a zero
            scalar placeholder (the real loss lives on the last stage);
            parameter gradients are the microbatch-averaged accumulation.
        """
        state = flat_inputs[: self.num_state]
        input_batch = flat_inputs[-2]
        label_batch = flat_inputs[-1]

        batch_size = input_batch.shape[0]
        if batch_size % self.microbatch_size != 0:
            raise ValueError(
                f"PP microbatch mismatch: batch size {batch_size} is not "
                f"divisible by pp_microbatch_size {self.microbatch_size}"
            )
        num_microbatches = batch_size // self.microbatch_size

        fwd_outs_per_mb = self._forward_sweep(
            state, input_batch, label_batch, num_microbatches
        )
        grads = self._backward_sweep(state, fwd_outs_per_mb, num_microbatches)

        for work, _ in self._pending_sends:
            work.wait()
        self._pending_sends = []

        if self.is_last:
            # On the last stage each forward output tuple starts with the
            # per-microbatch loss.
            losses = [outs[0] for outs in fwd_outs_per_mb]
            loss = torch.stack(losses).mean()
        else:
            loss = torch.zeros((), device=input_batch.device)
        return (loss, *grads)

    def _forward_sweep(
        self,
        state: Sequence[torch.Tensor],
        input_batch: torch.Tensor,
        label_batch: torch.Tensor,
        num_microbatches: int,
    ) -> List[Tuple[Any, ...]]:
        """Run forward microbatch 0..N-1, shipping boundary values downstream.

        Returns per-microbatch forward output tuples (replayed verbatim
        into the backward subgraph by ``_backward_sweep``).
        """
        mb = self.microbatch_size
        fwd_outs_per_mb: List[Tuple[Any, ...]] = []

        for i in range(num_microbatches):
            lo, hi = i * mb, (i + 1) * mb
            if self.is_first:
                out = self.fwd_gm(*state, input_batch[lo:hi])
            else:
                # Forward activations arrive from the PREVIOUS stage.
                act_in = self._recv_values(self.recv_spec, src=self.stage_idx - 1)
                if self.is_last:
                    out = self.fwd_gm(*state, *act_in, label_batch[lo:hi])
                else:
                    out = self.fwd_gm(*state, *act_in)

            fwd_outs_per_mb.append(tuple(out))
            if self.num_send:
                self._send_values(out[: self.num_send], dst=self.stage_idx + 1)
        return fwd_outs_per_mb

    def _backward_sweep(
        self,
        state: Sequence[torch.Tensor],
        fwd_outs_per_mb: Sequence[Tuple[Any, ...]],
        num_microbatches: int,
    ) -> List[torch.Tensor]:
        """Run backward microbatches N-1..0, accumulating full-batch gradients.

        The traced backward graph seeds its own loss gradient internally
        (an ``ones_like`` over the saved loss), so the ``grad_in`` we pass on
        the last stage is a placeholder — the grads that come back are
        un-normalized sums over the microbatches. Dividing the accumulation
        by ``num_microbatches`` turns that into the mean, which is exactly
        the full-batch mean-loss gradient the non-PP path produces.
        """
        grads: List[torch.Tensor] = []

        for i in reversed(range(num_microbatches)):
            fwd_outs = fwd_outs_per_mb[i]
            if self.is_last:
                # Seed shape only; the traced graph ignores the value.
                grad_in: List[Any] = [torch.ones_like(fwd_outs[0])]
            else:
                # Boundary gradients arrive from the NEXT stage (they flow
                # upstream during the backward sweep).
                grad_in = self._recv_values(self.grad_recv_spec, src=self.stage_idx + 1)
            out = self.bwd_gm(*state, *grad_in, *fwd_outs)

            param_grads = out[: self.num_trainable]
            if not grads:
                grads = list(param_grads)
            else:
                grads = [acc + g for acc, g in zip(grads, param_grads)]

            if self.grad_send_count:
                # Gradients flow UPSTREAM: the previous stage is the
                # destination, mirroring the forward direction.
                self._send_values(out[self.num_trainable :], dst=self.stage_idx - 1)
        # Un-normalized sums over the microbatches -> mean.
        return [g / num_microbatches for g in grads]

    def _send_values(self, values: Sequence[Any], dst: int) -> None:
        """Async-send boundary values to the neighbouring stage ``dst``.

        Tensors go as-is (made contiguous, with THAT buffer retained); int
        scalars (dynamic-shape ``sym_size`` nodes) are packed as 0-d int64
        tensors on the first tensor value's device — CPU tensors cannot ride
        an NCCL/NPU-backend group — and unwrapped with ``item()`` on the
        receiving side. The ``Work`` handles are queued (waited at the end of
        the step) and the sent tensors kept referenced so their buffers
        outlive the send.
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
            self._pending_sends.append((work, tensor))

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


__all__ = ["ScheduleGPipe"]
