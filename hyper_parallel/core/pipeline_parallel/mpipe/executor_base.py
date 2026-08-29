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
"""Platform-agnostic base for MPipe Transpose execution.

The parameter broadcast, P2P transport, and step orchestration are identical
across backends (they go through the ``platform`` abstraction), so they live
here.  Each backend subclass implements only the autograd-specific hooks:
running the preprocess forward (detached vs graph-connected), marking a tensor
as a grad-requiring leaf, and the stage-0 backward.
"""
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from hyper_parallel.platform import get_platform
from hyper_parallel.core.fully_shard.hsdp_utils import GroupInfo

if TYPE_CHECKING:
    from hyper_parallel.core.pipeline_parallel.scheduler import MetaStep, PipelineContext
    from hyper_parallel.core.pipeline_parallel.mpipe.schedule import ScheduleMPipeTranspose

platform = get_platform()

# Stage 0's transposed-feature recvs are batched (all posted non-blocking, then
# waited once) by default, so the NT-1 transfers pipeline and overlap stage 0's
# own preprocess forward instead of being received-and-waited serially.  Set
# ``HYPER_PARALLEL_MPIPE_RECV_BATCH=0`` to fall back to the per-micro blocking path.
_RECV_BATCH = os.environ.get("HYPER_PARALLEL_MPIPE_RECV_BATCH", "1").lower() not in ("", "0", "false", "no")


class MPipeTransposeExecutorBase(ABC):
    """Backend-agnostic runtime for the ``MPIPE_*`` steps of MPipe Transpose.

    Args:
        schedule (ScheduleMPipeTranspose): The schedule instance, providing the
            per-rank preprocess module, the body stages (for the PP group and
            global-rank mapping), and the transposed micro-batch count.
    """

    def __init__(self, schedule: "ScheduleMPipeTranspose") -> None:
        """Bind the executor to its schedule and cache the per-rank transpose state."""
        self._schedule = schedule
        self._preprocess = schedule.preprocess_module
        first_stage = schedule.stages[0]
        self._device = first_stage.device
        # Named wrapper over the body pp_group (same ranks), so the mpipe wire
        # can later move to a dedicated communicator without touching callers.
        self._mpipe_group_info = GroupInfo("mpipe_pp", first_stage.pp_group, schedule.real_stage_num)
        self._mpipe_group = self._mpipe_group_info.group
        self._this_rank = first_stage.stage_index % schedule.real_stage_num
        self._output_arity_for_comm = None
        self._input_arity_for_comm = None
        # Trainable: mark shipped tensors grad-requiring and run the stage-0
        # backward. Frozen or param-free ships as-is (input_ids must not be marked).
        self._has_trainable = schedule.has_trainable_preprocess
        # micro_index -> retained raw input args (stage-0 backward / shipping).
        self._inputs_for_explicit_forward = {}
        # micro_index -> detached preprocess output buffer (stage 0 body input).
        self._outputs_for_stage0 = {}
        # Transposed micro-batches whose output stage 0 has already received (the
        # batched recv posts all NT-1 at once, so later FWD_RECV steps no-op).
        self._fwd_received = set()
        # Owner-does-backward: the owner retains the tower graph, stage 0 ships
        # dL/dfeatures back, and param-grads are SUM all-reduced across pp.
        self._owner_backward = bool(getattr(schedule, "owner_backward", False))
        # micro_index -> the feature tensors stage 0 placed; the body backward
        # deposits dL/dfeatures on them for the stage-0 or owner backward.
        self._keep_grad = {}
        # Owner-backward: see reset() — lets GRAD_REDUCE reduce only this
        # run's contribution.
        self._grad_snapshot = None
        # Owner-backward: micro_index -> the owner's graph-connected preprocess
        # output tuple (same objects fwd_send ships detached).
        self._outputs_for_bwd = {}

    def reset(self) -> None:
        """Clear the per-step compute caches at the start of each schedule run."""
        self._output_arity_for_comm = None
        self._input_arity_for_comm = None
        assert len(self._inputs_for_explicit_forward) == 0
        assert len(self._outputs_for_stage0) == 0
        assert len(self._outputs_for_bwd) == 0
        assert len(self._keep_grad) == 0
        self._fwd_received.clear()
        # Snapshot the already-reduced grads from prior runs so GRAD_REDUCE
        # reduces only this run's added contribution.
        if self._owner_backward:
            self._grad_snapshot = self._snapshot_tower_grads()

    def _send_meta(self, tensor, dst) -> None:
        """Send a tensor's ``(shape, dtype)`` to ``dst``.

        Exchanged every step (not cached): MPipe Transpose is used mostly with
        dynamic shapes (variable sequence / image-token lengths), where a cached
        shape would be wrong; the meta is tiny so the per-step cost is negligible
        (``T = 0`` step time ~= 1F1B).
        """
        platform.send_object_list([tuple(tensor.shape), tensor.dtype], dst, self._mpipe_group)

    def _recv_meta(self, src):
        """Receive a tensor's ``(shape, dtype)`` from ``src`` (exchanged every step)."""
        meta: list = [None, None]
        platform.recv_object_list(meta, src, self._mpipe_group)
        return meta[0], meta[1]

    def _output_arity(self):
        """Number of output tensors to communicate (cached after first use)."""
        if self._output_arity_for_comm is not None:
            return self._output_arity_for_comm
        # Fallback: derive the arity from the first retained micro.
        if self._outputs_for_stage0.get(0):
            self._output_arity_for_comm = len(self._outputs_for_stage0[0])
        elif self._outputs_for_bwd.get(0):
            self._output_arity_for_comm = len(self._outputs_for_bwd[0])
        else:
            raise ValueError("Cannot determine the arity, number of elements to be communicated")
        return self._output_arity_for_comm

    def _input_arity(self):
        """Number of input tensors to communicate (cached after first use)."""
        if self._input_arity_for_comm is not None:
            return self._input_arity_for_comm
        # Fallback: derive the arity from the first retained micro.
        if self._inputs_for_explicit_forward.get(0):
            self._input_arity_for_comm = len(self._inputs_for_explicit_forward[0])
        else:
            raise ValueError("Cannot determine the arity, number of elements to be communicated")
        return self._input_arity_for_comm

    def _global_rank(self, group_rank: int) -> int:
        """Global rank of group pipeline rank ``group_rank``."""
        return platform.get_global_rank(self._mpipe_group, group_rank)

    @staticmethod
    def _as_tuple(args):
        """Normalize a micro-batch arg slot to a positional-args tuple."""
        if isinstance(args, (list, tuple)):
            return tuple(args)
        return (args,)

    @staticmethod
    def _kwargs_for(ctx, micro):
        """Keyword args for ``micro``'s preprocess forward (e.g. position_ids)."""
        kwarg_mbs = getattr(ctx, "kwarg_mbs", None)
        if not kwarg_mbs:
            return {}
        return kwarg_mbs[micro] or {}

    def broadcast_params(self, step: "MetaStep", ctx: "PipelineContext") -> None:  # pylint: disable=unused-argument
        """Broadcast the preprocess parameters from stage 0 to all ranks.

        Args:
            step (MetaStep): The ``MPIPE_PARAM_BROADCAST`` schedule step (unused).
            ctx (PipelineContext): The pipeline run context (unused).
        """
        src = self._global_rank(0)
        for tensor in self._broadcast_tensors():
            platform.broadcast(tensor, src, self._mpipe_group)

    def transpose_forward(self, step: "MetaStep", ctx: "PipelineContext") -> None:
        """Run the preprocess forward for ``step.micro_index``.

        The output is currently detached and the input retained for the stage-0
        recompute backward -- every micro takes this uniform path. (The old
        graph-connected inline forward for non-transposed / overflow micros was
        dropped in the round-robin change; the in-flight owner-does-backward work
        for a non-stage-0 backward may reintroduce a connected forward on the
        owner.)

        Args:
            step (MetaStep): The schedule step; ``step.micro_index`` selects the micro-batch.
            ctx (PipelineContext): The pipeline run context (``arg_mbs`` / ``kwarg_mbs``).
        """
        micro = step.micro_index
        args = self._as_tuple(ctx.arg_mbs[micro])
        kwargs = self._kwargs_for(ctx, micro)
        self._input_arity_for_comm = len(args)
        if self._owner_backward:
            # Stage 0's own micros backprop the connected tower during the body
            # backward, so only the other ranks retain a graph for the owner backward.
            out = self._as_tuple(self._connected_forward(args, kwargs))
            if self._this_rank == 0:
                self._output_arity_for_comm = len(out)
                self._transfer_out_for_fwd(ctx, micro, out)
            else:
                self._outputs_for_stage0[micro] = out  # shipped detached by fwd_send
                self._outputs_for_bwd[micro] = out  # retained graph for the owner backward
        else:
            if self._has_trainable:
                self._inputs_for_explicit_forward[micro] = args
            # The output may be one tensor or a tuple (VL: image_embeds + DeepStack
            # levels); normalize so transport and placement are uniform.
            out = self._as_tuple(self._detached_forward(args, kwargs))
            if self._has_trainable:
                for tensor in out:
                    self._mark_requires_grad(tensor)
            if self._this_rank == 0:
                self._output_arity_for_comm = len(out)
                self._save_grad_for_bwd(micro, out)
                self._transfer_out_for_fwd(ctx, micro, out)
            else:
                self._outputs_for_stage0[micro] = out
        # Stages built under meta-init only learn their real device here.
        if out and getattr(self._device, "type", None) in (None, "meta"):
            self._device = out[0].device

    def _save_grad_for_bwd(self, micro, out) -> None:
        """Record a transposed micro-batch's placed feature tensors for the backward.

        The stage-0 backward reads ``dL/dfeatures`` off these objects after the
        body backward deposits them. Routing-agnostic: it holds the same tensors
        whether they were handed to the forward as ``arg_mbs`` (a text body input)
        or as a kwarg (a VL ``mpipe_visual`` payload).
        """
        if self._has_trainable:
            self._keep_grad[micro] = tuple(out)

    def _transfer_out_for_fwd(self, ctx, micro, out) -> None:
        """Deliver a transposed micro-batch's preprocess output to stage 0's forward.

        Default: the output is stage 0's body input (placed into ``arg_mbs``). A
        model may override this via the schedule's ``output_consumer`` -- e.g. VL
        routes the visual payload into an injection kwarg the stage forward reads,
        since the visual output is injected mid-forward, not as the body input. The
        stage forward consumes these slots on the later ``FWD`` step, via
        ``forward_one_chunk(micro, arg_mbs[micro], kwarg_mbs[micro])``.

        Args:
            ctx: The pipeline run context.
            micro: The transposed micro-batch index.
            out: The preprocess output as a tuple of tensors.
        """
        consumer = self._schedule.output_consumer
        if consumer is not None:
            consumer(ctx, micro, out)
        else:
            ctx.arg_mbs[micro] = list(out)

    def fwd_send(self, step: "MetaStep", ctx: "PipelineContext") -> None:
        """Send the preprocess output of ``step.micro_index`` to stage 0.

        Args:
            step (MetaStep): The schedule step; ``step.micro_index`` selects the micro-batch.
            ctx (PipelineContext): The pipeline run context; the send handles are appended to it.
        """
        micro = step.micro_index
        dst = self._global_rank(0)
        # Under owner-backward the output is graph-connected, so ship a detached
        # wire copy rather than the graph tensor (keeps the owner's graph local).
        for tensor in self._outputs_for_stage0[micro]:
            wire = self._detach_for_wire(tensor) if self._owner_backward else self._contiguous(tensor)
            self._send_meta(wire, dst)
            ctx.schedule._send_handles.append([platform.isend(wire, dst, self._mpipe_group)])  # pylint: disable=protected-access
        # What is in flight is the wire copy, not this cache entry, so the
        # source can go now; the handles are drained by ``run_microbatches``.
        del self._outputs_for_stage0[micro]

    def fwd_recv(self, step: "MetaStep", ctx: "PipelineContext") -> None:
        """Receive transposed micro-batches' preprocess outputs and hand them to stage 0.

        Batched path (default): the first ``MPIPE_FWD_RECV`` step posts the recvs
        for every transposed micro-batch non-blocking, then waits once, so the
        ``NT - 1`` transfers pipeline and overlap stage 0's own preprocess forward
        (whose kernels are still in flight on the compute stream) instead of being
        received-and-waited serially.  Per-peer P2P order and tags are unchanged
        (only the waits are deferred), so collective ordering is unperturbed; the
        later ``MPIPE_FWD_RECV`` steps then find their micro already received.  Set
        ``HYPER_PARALLEL_MPIPE_RECV_BATCH=0`` for the per-micro blocking path.

        Args:
            step (MetaStep): The schedule step; ``step.micro_index`` selects the micro-batch.
            ctx (PipelineContext): The pipeline run context; received buffers
                are handed to the forward via ``_transfer_out_for_fwd`` and
                recorded via ``_save_grad_for_bwd``.
        """
        if not _RECV_BATCH:
            self._fwd_recv_one(step.micro_index, ctx)
            return
        if step.micro_index in self._fwd_received:
            return
        self._fwd_recv_all(ctx)

    def _recv_device(self):
        """Real (non-``meta``) device for receive buffers.

        ``self._device`` is captured from ``first_stage.device`` at construction.
        Under deferred / meta init (FSDP2 builds the stage on ``meta`` and only
        materializes it later) that capture is ``meta``; allocating a recv buffer
        there yields a Meta tensor that ``c10d::recv_`` rejects on the wire
        (``NotImplementedError: ... Meta tensor``). Resolve lazily from a tensor
        stage 0 has already materialized this step -- its own micro-0 preprocess
        output (``_outputs_for_stage0[0]``) or retained input (``_inputs_for_explicit_forward[0]``)
        or grad  (``_keep_grad[0]``) -- cache it, and fall back to the captured device.
        """
        if getattr(self._device, "type", None) != "meta":
            return self._device
        for materialized in (self._outputs_for_stage0.get(0),
                             self._inputs_for_explicit_forward.get(0),
                             self._keep_grad.get(0)):
            if materialized:
                self._device = materialized[0].device
                return self._device
        return self._device

    def _recv_payload(self, micro):
        """Post the recv of ``micro``'s preprocess output; return ``(buffers, handles)``.

        Exchanges each tensor's meta, allocates its buffer and issues a
        non-blocking ``irecv`` (the caller decides when to wait).  Stage 0 always
        owns transposed micro 0, so its own output reveals the per-micro arity
        (1 for a text body input, >1 for a VL tuple payload).
        """
        src = self._global_rank(self._schedule.owner_of(micro))
        arity = self._output_arity()
        buffers = []
        handles = []
        for _ in range(arity):
            shape, dtype = self._recv_meta(src)
            buffer = platform.empty(shape, dtype=dtype, device=self._recv_device())
            if self._has_trainable:
                self._mark_requires_grad(buffer)
            handles.append(platform.irecv(buffer, src, self._mpipe_group))
            buffers.append(buffer)
        return buffers, handles

    def _fwd_recv_one(self, micro: int, ctx: "PipelineContext") -> None:
        """Receive a single transposed micro-batch's output, waiting inline (fallback)."""
        buffers, handles = self._recv_payload(micro)
        for handle in handles:
            handle.wait()
        out = tuple(buffers)
        self._save_grad_for_bwd(micro, out)
        self._transfer_out_for_fwd(ctx, micro, out)
        self._fwd_received.add(micro)

    def _fwd_recv_all(self, ctx: "PipelineContext") -> None:
        """Post every outstanding transposed recv, then wait once so the transfers
        overlap each other and stage 0's own preprocess forward."""
        owned = self._schedule.owned_micros(self._this_rank)
        pending = [(micro, *self._recv_payload(micro))
                   for micro in range(self._schedule.micro_batch_num)
                   if micro not in owned and micro not in self._fwd_received]
        for micro, buffers, handles in pending:
            for handle in handles:
                handle.wait()
            out = tuple(buffers)
            self._save_grad_for_bwd(micro, out)
            self._transfer_out_for_fwd(ctx, micro, out)
            self._fwd_received.add(micro)

    def fwd_bwd_graph_input_send(self, step: "MetaStep", ctx: "PipelineContext") -> None:
        """Send the preprocess input of ``step.micro_index`` to stage 0 (for the stage-0 backward).

        Args:
            step (MetaStep): The schedule step; ``step.micro_index`` selects the micro-batch.
            ctx (PipelineContext): The pipeline run context; send handles are appended to it.
        """
        micro = step.micro_index
        dst = self._global_rank(0)
        for tensor in self._inputs_for_explicit_forward[micro]:
            contiguous = self._contiguous(tensor)
            self._send_meta(contiguous, dst)
            ctx.schedule._send_handles.append([platform.isend(contiguous, dst, self._mpipe_group)])  # pylint: disable=protected-access
        # What is in flight is the wire copy, not this cache entry, so the
        # source can go now; the handles are drained by ``run_microbatches``.
        del self._inputs_for_explicit_forward[micro]

    def fwd_bwd_graph_input_recv(self, step: "MetaStep", ctx: "PipelineContext") -> None:  # pylint: disable=unused-argument
        """Receive a transposed micro-batch's preprocess input for the stage-0 backward.

        Args:
            step (MetaStep): The schedule step; ``step.micro_index`` selects the micro-batch.
            ctx (PipelineContext): The pipeline run context (unused).
        """
        micro = step.micro_index
        src = self._global_rank(self._schedule.owner_of(micro))
        # Stage 0 always owns transposed micro 0, so its retained input reveals
        # how many input tensors each transposed micro-batch ships.
        arity = self._input_arity()
        tensors = []
        for _ in range(arity):
            shape, dtype = self._recv_meta(src)
            buffer = platform.empty(shape, dtype=dtype, device=self._recv_device())
            platform.irecv(buffer, src, self._mpipe_group).wait()
            tensors.append(buffer)
        self._inputs_for_explicit_forward[micro] = tuple(tensors)

    def grad_send(self, step: "MetaStep", ctx: "PipelineContext") -> None:
        """Owner-backward: ship ``dL/dfeatures`` for transposed ``micro`` back to its owner.

        The body backward has deposited the gradient on the feature tensors stage
        0 placed for ``micro`` (``self._keep_grad[micro]``); each is shipped (absent
        per-tensor grads become zeros so the owner's backward is well-formed). The
        destination is ``self._schedule.owner_of(micro)`` -- under round-robin
        overflow (``"full"``) that is not ``micro`` itself once ``micro >= NT``.
        Stage 0's own micro(s) already backpropagated via the connected tower
        during the body backward, so those are a no-op (never actually scheduled;
        the guard is defensive).

        Args:
            step (MetaStep): The schedule step; ``step.micro_index`` selects the micro-batch.
            ctx (PipelineContext): The pipeline run context; send handles are appended to it.
        """
        micro = step.micro_index
        dst = self._global_rank(self._schedule.owner_of(micro))
        if dst == self._global_rank(0):
            return
        for tensor in self._keep_grad[micro]:
            grad = getattr(tensor, "grad", None)
            wire = self._detach_for_wire(grad) if grad is not None else self._zeros_like(tensor)
            self._send_meta(wire, dst)
            ctx.schedule._send_handles.append([platform.isend(wire, dst, self._mpipe_group)])  # pylint: disable=protected-access
        # What is in flight is the wire copy, not this cache entry, so the
        # source can go now; the handles are drained by ``run_microbatches``.
        del self._keep_grad[micro]

    def grad_recv_with_backward(self, step: "MetaStep", ctx: "PipelineContext") -> None:  # pylint: disable=unused-argument
        """Owner-backward: receive ``dL/dfeatures`` and backprop the retained tower graph.

        The graph is ``self._outputs_for_bwd[micro]`` (this owner's connected
        preprocess output); the received grads are its cotangents.

        Args:
            step (MetaStep): The schedule step; ``step.micro_index`` selects the owned micro-batch.
            ctx (PipelineContext): The pipeline run context (unused).
        """
        micro = step.micro_index
        src = self._global_rank(0)
        retained = self._outputs_for_bwd[micro]
        device = retained[0].device
        grads = []
        handles = []
        # Post every recv, then wait once: the payloads overlap in flight
        # (same per-peer posting order as the sends, so matching is unchanged).
        for _ in range(len(retained)):
            shape, dtype = self._recv_meta(src)
            buffer = platform.empty(shape, dtype=dtype, device=device)
            handles.append(platform.irecv(buffer, src, self._mpipe_group))
            grads.append(buffer)
        for handle in handles:
            handle.wait()
        self._owner_transpose_backward(retained, grads)
        del self._outputs_for_bwd[micro]

    def reduce_tower_grads(self, step: "MetaStep", ctx: "PipelineContext") -> None:  # pylint: disable=unused-argument
        """Owner-backward: SUM all-reduce the tower param-grads across the pp replicas.

        A collective -- emitted on every pp rank; ranks that produced no grad for a
        param contribute zeros so the all-reduce stays symmetric. Only this run's
        contribution is reduced (``grad - snapshot``), so prior gradient-
        accumulation passes (already reduced) are not re-reduced.

        Args:
            step (MetaStep): The ``MPIPE_GRAD_REDUCE`` step (unused).
            ctx (PipelineContext): The pipeline run context (unused).
        """
        self._reduce_grads(self._mpipe_group_info, self._grad_snapshot)

    def transpose_backward(self, step: "MetaStep", ctx: "PipelineContext") -> None:
        """Recompute the preprocess forward on stage 0 and backprop dL/dfeatures.

        The body backward deposits the feature gradient on the tensors stage 0
        placed for ``micro`` (``self._keep_grad[micro]`` -- the same objects whether
        they were routed into ``arg_mbs`` (a text body input) or into a kwarg (the
        VL ``mpipe_visual`` payload)). Recompute the preprocess forward from the
        retained input and backprop those grads into it. Reading ``_keep_grad`` (not
        ``arg_mbs[micro][0]``) is what makes the stage-0 backward correct for a
        trainable VL tower, whose features live in the kwarg.
        Assume that ``self._keep_grad[micro]`` was previously fill before calling the BWD

        Args:
            step (MetaStep): The schedule step; ``step.micro_index`` selects the micro-batch.
            ctx (PipelineContext): The pipeline run context.
        """
        micro = step.micro_index
        grads = [getattr(tensor, "grad", None) for tensor in self._keep_grad[micro]]
        if all(grad is None for grad in grads):
            self._inputs_for_explicit_forward.pop(micro, None)
            del self._keep_grad[micro]
            return
        self._explicit_forward_before_backward(
            self._inputs_for_explicit_forward[micro], self._kwargs_for(ctx, micro), grads)
        del self._inputs_for_explicit_forward[micro]
        del self._keep_grad[micro]

    @staticmethod
    def _contiguous(tensor):
        """Return a contiguous tensor suitable for P2P (overridden where needed)."""
        return tensor

    @abstractmethod
    def _broadcast_tensors(self):
        """Yield the preprocess tensors (params/buffers) to broadcast from stage 0."""

    @abstractmethod
    def _detached_forward(self, args, kwargs):
        """Run the preprocess forward and return a detached output value.

        The base marks it grad-requiring via :meth:`_mark_requires_grad` only for
        a trainable preprocess; for a frozen / param-free one (frozen visual
        tower, T=0 identity) the value is shipped as-is.
        """

    @abstractmethod
    def _connected_forward(self, args, kwargs):
        """Owner-backward: run the preprocess forward graph-connected (no detach),
        so the owner rank can later backprop into it from a received gradient."""

    @abstractmethod
    def _mark_requires_grad(self, tensor) -> None:
        """Mark ``tensor`` as a grad-requiring leaf so the body backward deposits a grad on it."""

    @abstractmethod
    def _explicit_forward_before_backward(self, inputs, kwargs, grads) -> None:
        """Recompute the preprocess forward, then backprop the per-output ``grads``
        (one ``dL/dfeature`` per preprocess output, ``None`` where absent) into it,
        accumulating preprocess grads."""

    @abstractmethod
    def _detach_for_wire(self, tensor):
        """Return a detached, contiguous wire copy of ``tensor`` (keeps the autograd graph local)."""

    @abstractmethod
    def _zeros_like(self, tensor):
        """Return a contiguous zero tensor matching ``tensor`` (for an absent feature gradient)."""

    @abstractmethod
    def _owner_transpose_backward(self, retained_out, grads) -> None:
        """Owner-backward: backprop ``grads`` through the retained preprocess output graph."""

    @abstractmethod
    def _snapshot_tower_grads(self):
        """Owner-backward: clone the trainable preprocess param-grads (``None`` where
        absent), in parameter order, so :meth:`_reduce_grads` can reduce only the
        current run's contribution under gradient accumulation."""

    @abstractmethod
    def _reduce_grads(self, group_info, snapshot) -> None:
        """Owner-backward: SUM-reduce only the *current run's* trainable preprocess
        param-grad contribution (``grad - snapshot``) over ``group_info`` and add it
        back to ``snapshot``, so accumulated grads are not re-reduced each run."""
