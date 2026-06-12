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
as a grad-requiring leaf, and the recompute backward.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from hyper_parallel.platform import get_platform

if TYPE_CHECKING:
    from hyper_parallel.core.pipeline_parallel.scheduler import MetaStep, PipelineContext
    from hyper_parallel.core.pipeline_parallel.mpipe.schedule import ScheduleMPipeTranspose

platform = get_platform()


class MPipeTransposeExecutorBase(ABC):
    """Backend-agnostic runtime for the ``MPIPE_*`` steps of MPipe Transpose.

    Args:
        schedule: The :class:`ScheduleMPipeTranspose` instance, providing the
            per-rank preprocess module, the body stages (for the PP group and
            global-rank mapping), and the transposed micro-batch count.

    Attributes:
        nontransposed_connected (bool): Whether non-transposed micro-batches
            can rely on the autograd graph flowing the body backward into the
            preprocess automatically (torch).  When ``False`` (mindspore), the
            schedule also emits ``MPIPE_TRANSPOSE_BWD`` for them and the
            preprocess forward is always detached.
    """

    nontransposed_connected = False

    def __init__(self, schedule: "ScheduleMPipeTranspose") -> None:
        """Bind the executor to its schedule and cache the per-rank transpose state."""
        self._schedule = schedule
        self._preprocess = schedule.preprocess_module
        first_stage = schedule.stages[0]
        self._device = first_stage.device
        self._pp_group = first_stage.pp_group
        self._this_rank = first_stage.stage_index % schedule.real_stage_num
        self._num_transpose = schedule.num_transpose_micro_batches
        # Trainable preprocess → mark shipped tensors grad-requiring (the body
        # backward / recompute use them) and recompute the backward. Frozen or
        # param-free (T=0 identity, a frozen visual tower) → ship the output
        # as-is (e.g. integer input_ids must not be marked grad-requiring).
        self._has_trainable = schedule.has_trainable_preprocess
        # micro_index -> retained raw input args (recompute backward / shipping).
        self._inputs = {}
        # micro_index -> detached preprocess output buffer (stage 0 body input).
        self._outputs = {}

    def reset(self) -> None:
        """Clear the per-step compute caches at the start of each schedule run."""
        self._inputs.clear()
        self._outputs.clear()

    @staticmethod
    def _send_meta(tensor, dst) -> None:
        """Send a tensor's ``(shape, dtype)`` to ``dst``.

        Exchanged every step (not cached): MPipe Transpose is used mostly with
        dynamic shapes (variable sequence / image-token lengths), where a cached
        shape would be wrong; the meta is tiny so the per-step cost is negligible
        (``T = 0`` step time ~= 1F1B).
        """
        platform.send_object_list([tuple(tensor.shape), tensor.dtype], dst)

    @staticmethod
    def _recv_meta(src):
        """Receive a tensor's ``(shape, dtype)`` from ``src`` (exchanged every step)."""
        meta: list = [None, None]
        platform.recv_object_list(meta, src)
        return meta[0], meta[1]

    def _global_rank(self, physical_rank: int) -> int:
        """Global rank of physical pipeline rank ``physical_rank``."""
        return platform.get_global_rank(self._pp_group, physical_rank)

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
            step: The ``MPIPE_PARAM_BROADCAST`` schedule step (unused).
            ctx: The pipeline run context (unused).
        """
        src = self._global_rank(0)
        for tensor in self._broadcast_tensors():
            platform.broadcast(tensor, src, self._pp_group)

    def transpose_forward(self, step: "MetaStep", ctx: "PipelineContext") -> None:
        """Run the preprocess forward for ``step.micro_index``.

        Non-transposed micro-batches stay graph-connected only when the backend
        supports automatic backward into the preprocess; otherwise (and for all
        transposed micro-batches) the output is detached and the input retained
        for the recompute backward.

        Args:
            step: The schedule step; ``step.micro_index`` selects the micro-batch.
            ctx: The pipeline run context (``arg_mbs`` / ``kwarg_mbs``).
        """
        micro = step.micro_index
        args = self._as_tuple(ctx.arg_mbs[micro])
        kwargs = self._kwargs_for(ctx, micro)
        if micro >= self._num_transpose and self.nontransposed_connected:
            ctx.arg_mbs[micro] = [self._connected_forward(args, kwargs)]
            return
        self._inputs[micro] = args
        out = self._detached_forward(args, kwargs)
        if self._has_trainable:
            self._mark_requires_grad(out)
        self._outputs[micro] = out
        if self._this_rank == 0:
            ctx.arg_mbs[micro] = [out]

    def fwd_send(self, step: "MetaStep", ctx: "PipelineContext") -> None:
        """Send the preprocess output of ``step.micro_index`` to stage 0.

        Args:
            step: The schedule step; ``step.micro_index`` selects the micro-batch.
            ctx: The pipeline run context; the send handle is appended to it.
        """
        micro = step.micro_index
        out = self._contiguous(self._outputs[micro])
        dst = self._global_rank(0)
        self._send_meta(out, dst)
        # Deferred sends are drained at the end of ``run_microbatches`` via the
        # schedule's ``_send_handles`` (each entry is a handle group).
        ctx.schedule._send_handles.append([platform.isend(out, dst)])  # pylint: disable=protected-access

    def fwd_recv(self, step: "MetaStep", ctx: "PipelineContext") -> None:
        """Receive a transposed micro-batch's preprocess output into stage 0's input slot.

        Args:
            step: The schedule step; ``step.micro_index`` selects the micro-batch.
            ctx: The pipeline run context; the received buffer is placed in its ``arg_mbs``.
        """
        micro = step.micro_index
        src = self._global_rank(micro)
        shape, dtype = self._recv_meta(src)
        buffer = platform.empty(shape, dtype=dtype, device=self._device)
        platform.irecv(buffer, src).wait()
        if self._has_trainable:
            self._mark_requires_grad(buffer)
        ctx.arg_mbs[micro] = [buffer]

    def graph_send(self, step: "MetaStep", ctx: "PipelineContext") -> None:
        """Send the preprocess input of ``step.micro_index`` to stage 0 (for recompute).

        Args:
            step: The schedule step; ``step.micro_index`` selects the micro-batch.
            ctx: The pipeline run context; send handles are appended to it.
        """
        micro = step.micro_index
        dst = self._global_rank(0)
        for tensor in self._inputs[micro]:
            contiguous = self._contiguous(tensor)
            self._send_meta(contiguous, dst)
            ctx.schedule._send_handles.append([platform.isend(contiguous, dst)])  # pylint: disable=protected-access

    def graph_recv(self, step: "MetaStep", ctx: "PipelineContext") -> None:  # pylint: disable=unused-argument
        """Receive a transposed micro-batch's preprocess input for the recompute backward.

        Args:
            step: The schedule step; ``step.micro_index`` selects the micro-batch.
            ctx: The pipeline run context (unused).
        """
        micro = step.micro_index
        src = self._global_rank(micro)
        # Stage 0 always owns transposed micro 0, so its retained input reveals
        # how many input tensors each transposed micro-batch ships.
        arity = len(self._inputs[0])
        tensors = []
        for _ in range(arity):
            shape, dtype = self._recv_meta(src)
            buffer = platform.empty(shape, dtype=dtype, device=self._device)
            platform.irecv(buffer, src).wait()
            tensors.append(buffer)
        self._inputs[micro] = tuple(tensors)

    def transpose_backward(self, step: "MetaStep", ctx: "PipelineContext") -> None:
        """Recompute the preprocess forward on stage 0 and backprop the body's input grad.

        Args:
            step: The schedule step; ``step.micro_index`` selects the micro-batch.
            ctx: The pipeline run context; the body's input grad is read from its ``arg_mbs``.
        """
        micro = step.micro_index
        grad = ctx.arg_mbs[micro][0].grad
        if grad is None:
            return
        self._recompute_backward(self._inputs[micro], self._kwargs_for(ctx, micro), grad)

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
        """Run the preprocess forward graph-connected to the body (backends that support it)."""

    @abstractmethod
    def _mark_requires_grad(self, tensor) -> None:
        """Mark ``tensor`` as a grad-requiring leaf so the body backward deposits a grad on it."""

    @abstractmethod
    def _recompute_backward(self, inputs, kwargs, grad) -> None:
        """Recompute ``preprocess(*inputs, **kwargs)`` and backprop ``grad``, accumulating preprocess grads."""
