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
"""The MPipe Transpose schedule (an Interleaved-1F1B variant).

Layers the ``MPIPE_*`` transpose steps around the inherited Interleaved-1F1B
body order and registers their handlers via the generic custom-function
registry, so the core ``scheduler`` module carries no MPipe-specific code.
"""
from typing import Optional, TYPE_CHECKING

from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType
from hyper_parallel.core.pipeline_parallel.scheduler import (
    MetaStep,
    MetaStepType,
    ScheduleInterleaved1F1B,
)
from hyper_parallel.core.pipeline_parallel.mpipe.step_types import MpipeStepType

if TYPE_CHECKING:
    from hyper_parallel.core.pipeline_parallel.utils import BatchDimSpec
    from hyper_parallel.dmodule.module import Module

platform = get_platform()


class ScheduleMPipeTranspose(ScheduleInterleaved1F1B):
    """The MPipe Transpose schedule.

    A variant of Interleaved 1F1B that shrinks the warmup pipeline bubble by
    **transposing** the forward of a model's first ``T`` layers (the
    *preprocess* block, which logically belongs to stage 0's first chunk).

    Instead of stage 0 computing the preprocess forward serially for every
    micro-batch, the preprocess parameters are broadcast to all ``PP`` ranks
    and, for the first ``NT = min(PP, micro_batch_num)`` micro-batches, each
    rank ``i`` computes the preprocess forward of micro-batch ``i`` in parallel
    during what would otherwise be its warmup idle time.  Each rank ``i > 0``
    then ships to stage 0:

      * the preprocess **output** (``MPIPE_FWD_SEND`` / ``MPIPE_FWD_RECV``) so
        stage 0 can run its body forward, and
      * the preprocess **input** (``MPIPE_GRAPH_SEND`` / ``MPIPE_GRAPH_RECV``)
        so the preprocess backward can be recomputed centrally on stage 0
        (gradients accumulate on stage 0 only).

    The remaining ``micro_batch_num - NT`` micro-batches run the preprocess
    forward inline on stage 0 (graph-connected to the body, so their backward
    is automatic), exactly as ordinary Interleaved 1F1B.

    The body model (stage 0 = the layers after the preprocess block, all other
    stages unchanged) is scheduled by the inherited Interleaved 1F1B logic; the
    preprocess steps are layered around it: a transpose-phase prefix per rank,
    plus inline preprocess forward / recompute backward steps on stage 0.

    Args:
        stages (list[PipelineStage], PipelineStage): The body pipeline stages.
            Stage 0 must wrap only the layers **after** the preprocess block.
        micro_batch_num (int): The number of micro-batches.
        preprocess_module (Optional[Module]): The preprocess block (first ``T``
            layers of stage 0).  Following Option A, it must exist on **every** rank: on rank 0
            it holds the trained parameters; on other ranks it is a structural
            copy whose parameters are overwritten each step by the broadcast.
        num_transpose_layers (int): ``T`` — the number of preprocess layers,
            must be smaller than the layer count of stage 0's first chunk.
            ``0`` is allowed and means *only the data loading is transposed*:
            each rank loads its micro-batch and ships the raw input to stage 0,
            with no parameter broadcast, no preprocess compute, and no
            recompute backward.
        args_batch_dim (list, optional): See ``PipelineScheduleRuntime``.
        kwargs_batch_dim (dict, optional): See ``PipelineScheduleRuntime``.
        output_concat_dim (int, optional): See ``PipelineScheduleRuntime``.
        overlap_p2p (bool, optional): See ``ScheduleInterleaved1F1B``.
            Default ``False``.
        swap (bool, optional): Whether to inject activation-swap steps.
            Default ``False``.

    Note:
        This class builds the schedule ordering and registers the ``MPIPE_*``
        execution handlers; the handlers themselves live in the platform
        executors (see :class:`MPipeTransposeExecutorBase`).
    """

    def __init__(self,
                 stages: list,
                 micro_batch_num: int,
                 preprocess_module: "Optional[Module]",
                 num_transpose_layers: int,
                 args_batch_dim: "Optional[BatchDimSpec]" = None,
                 kwargs_batch_dim: "Optional[BatchDimSpec]" = None,
                 output_concat_dim: Optional[int] = None,
                 overlap_p2p: bool = False,
                 swap: bool = False) -> None:
        """Build an interleaved-1F1B schedule that transposes the preprocess block.

        Args:
            stages (list): The local pipeline stages (as for :class:`ScheduleInterleaved1F1B`).
            micro_batch_num (int): Number of micro-batches per optimizer step.
            preprocess_module (Optional[Module]): The block transposed to every
                rank — the first ``num_transpose_layers`` layers, a visual tower,
                or a param-free identity for the dataload-only (``T = 0``) mode.
            num_transpose_layers (int): ``T`` — the (informational) transposed-layer count.
            args_batch_dim (Optional[BatchDimSpec]): Positional-arg batch-dim spec (forwarded to the base).
            kwargs_batch_dim (Optional[BatchDimSpec]): Keyword-arg batch-dim spec (forwarded to the base).
            output_concat_dim (Optional[int]): Output concatenation dim (forwarded to the base).
            overlap_p2p (bool): Whether to overlap P2P (forwarded to the base).
            swap (bool): Whether to enable activation swapping (forwarded to the base).
        """
        if not isinstance(num_transpose_layers, int) or num_transpose_layers < 0:
            raise ValueError(
                f"Argument 'num_transpose_layers' must be a non-negative int, "
                f"but got {num_transpose_layers!r}."
            )
        # ``preprocess_module`` is the resolved block to transpose (first T text
        # layers, the visual tower, or a param-free identity for dataload-only).
        self._preprocess_module = preprocess_module
        self._num_transpose_layers = num_transpose_layers
        # Whether the preprocess has *trainable* params decides the path:
        #   trainable  -> broadcast (the trainable params only) + centralized
        #                 recompute backward;
        #   frozen / param-free (T=0, a frozen visual tower) -> ship the output
        #                 only (no broadcast, no recompute).
        self._has_trainable_preprocess = self._module_has_trainable_params(preprocess_module)
        # MindSpore's grad_fn is scoped to the body submodule's weights, so a
        # *trainable* preprocess also needs an explicit recompute backward for the
        # non-transposed micro-batches; torch's autograd handles them via the
        # connected graph.
        self._explicit_nontransposed_backward = platform.platform_type == PlatformType.MINDSPORE
        super().__init__(stages,
                         micro_batch_num,
                         args_batch_dim=args_batch_dim,
                         kwargs_batch_dim=kwargs_batch_dim,
                         output_concat_dim=output_concat_dim,
                         overlap_p2p=overlap_p2p,
                         overlap_b_f=False,
                         swap=swap)
        self._executor = None
        self._setup_mpipe_execution()

    @staticmethod
    def _module_has_trainable_params(module) -> bool:
        """Whether ``module`` has any trainable (grad-requiring) parameter."""
        if module is None:
            return False
        if platform.platform_type == PlatformType.PYTORCH:
            return any(p.requires_grad for p in module.parameters())
        if platform.platform_type == PlatformType.MINDSPORE:
            return any(p.requires_grad for p in module.get_parameters())
        raise NotImplementedError(
            f"MPipe Transpose is not implemented for platform {platform.platform_type}."
        )

    @property
    def preprocess_module(self) -> "Optional[Module]":
        """The preprocess block transposed to every rank (present on every rank)."""
        return self._preprocess_module

    @property
    def has_trainable_preprocess(self) -> bool:
        """Whether the preprocess has trainable params (drives broadcast/recompute)."""
        return self._has_trainable_preprocess

    @property
    def num_transpose_layers(self) -> int:
        """``T`` — informational transposed-layer count (``0`` = dataload only)."""
        return self._num_transpose_layers

    @property
    def num_transpose_micro_batches(self) -> int:
        """``NT = min(PP, micro_batch_num)`` — the count of transposed micro-batches."""
        return min(self.real_stage_num, self.micro_batch_num)

    def _setup_mpipe_execution(self) -> None:
        """Build the platform execution backend and register the MPIPE_* handlers."""
        # Lazy import: the backend executor pulls in torch/mindspore and is
        # resolved only when a schedule is actually constructed.
        if platform.platform_type == PlatformType.PYTORCH:
            from hyper_parallel.platform.torch.pipeline_parallel.mpipe_transpose import (  # pylint: disable=C0415
                MPipeTransposeExecutor,
            )
        elif platform.platform_type == PlatformType.MINDSPORE:
            from hyper_parallel.platform.mindspore.pipeline_parallel.mpipe_transpose import (  # pylint: disable=C0415
                MPipeTransposeExecutor,
            )
        else:
            raise NotImplementedError(
                f"MPipe Transpose execution is not implemented for platform {platform.platform_type}."
            )
        self._executor = MPipeTransposeExecutor(self)
        handlers = {
            MpipeStepType.MPIPE_PARAM_BROADCAST: self._executor.broadcast_params,
            MpipeStepType.MPIPE_TRANSPOSE_FWD: self._executor.transpose_forward,
            MpipeStepType.MPIPE_FWD_SEND: self._executor.fwd_send,
            MpipeStepType.MPIPE_FWD_RECV: self._executor.fwd_recv,
            MpipeStepType.MPIPE_GRAPH_SEND: self._executor.graph_send,
            MpipeStepType.MPIPE_GRAPH_RECV: self._executor.graph_recv,
            MpipeStepType.MPIPE_TRANSPOSE_BWD: self._executor.transpose_backward,
        }
        for step_type, handler in handlers.items():
            self.register_custom_function(step_type, handler)

    def run_microbatches(self, arg_mbs: list, kwarg_mbs: list, losses: list) -> None:
        """Reset the executor's per-step caches, then run the schedule.

        Args:
            arg_mbs (list): Per-micro-batch positional args.
            kwarg_mbs (list): Per-micro-batch keyword args.
            losses (list): Mutable list collecting per-step losses.
        """
        if self._executor is not None:
            self._executor.reset()
        super().run_microbatches(arg_mbs, kwarg_mbs, losses)

    def construct_exec_order(self) -> None:
        """Build the body Interleaved 1F1B order, then layer the preprocess
        transpose phase and the centralized preprocess backward on top.

        The parameter broadcast, the recompute-input transport, and the
        recompute backward are emitted only for a **trainable** preprocess; a
        frozen or param-free one (``T == 0``, a frozen visual tower) only
        transposes the forward and ships its output.
        """
        super().construct_exec_order()
        body_order = self.exec_order
        num_transpose = self.num_transpose_micro_batches
        has_trainable = self._has_trainable_preprocess
        body_order[0] = self._insert_rank0_preprocess_steps(
            body_order[0], num_transpose, has_trainable,
            backward_all=self._explicit_nontransposed_backward)
        self.exec_order = {
            rank: self._build_transpose_prefix(rank, num_transpose, has_trainable) + body_order[rank]
            for rank in range(self.real_stage_num)
        }

    @staticmethod
    def _build_transpose_prefix(rank, num_transpose, has_trainable):
        """Build the transpose-phase prefix prepended to ``rank``'s body order.

        A rank that owns a transposed micro-batch (``rank < num_transpose``)
        computes its preprocess forward and ranks ``> 0`` ship the output to
        stage 0.  For a **trainable** preprocess every rank also broadcasts its
        (trainable) parameters and ranks ``> 0`` additionally ship the input for
        the centralized recompute backward; for a frozen / param-free preprocess
        only the transpose forward and its output send/recv remain.
        """
        prefix = []
        if has_trainable:
            prefix.append(MetaStep(None, MpipeStepType.MPIPE_PARAM_BROADCAST, 0))
        if rank < num_transpose:
            prefix.append(MetaStep(rank, MpipeStepType.MPIPE_TRANSPOSE_FWD, 0))
            if rank != 0:
                prefix.append(MetaStep(rank, MpipeStepType.MPIPE_FWD_SEND, 0))
                if has_trainable:
                    prefix.append(MetaStep(rank, MpipeStepType.MPIPE_GRAPH_SEND, 0))
        if rank == 0:
            for micro_index in range(1, num_transpose):
                prefix.append(MetaStep(micro_index, MpipeStepType.MPIPE_FWD_RECV, 0))
            if has_trainable:
                for micro_index in range(1, num_transpose):
                    prefix.append(MetaStep(micro_index, MpipeStepType.MPIPE_GRAPH_RECV, 0))
        return prefix

    @staticmethod
    def _insert_rank0_preprocess_steps(order, num_transpose, has_trainable, backward_all=False):
        """Patch stage 0's (rank 0) body order with preprocess fwd/bwd steps.

        For a **trainable** preprocess: before each ``FWD(stage 0, micro >=
        num_transpose)`` an inline ``MPIPE_TRANSPOSE_FWD`` runs the preprocess
        forward for that non-transposed micro-batch, and an
        ``MPIPE_TRANSPOSE_BWD`` is inserted after each ``BWD(stage 0, micro)``
        needing a centralized recompute backward: the transposed micro-batches
        (``micro < num_transpose``) always, and — when ``backward_all`` is set
        (MindSpore, whose body backward does not flow into the preprocess) — the
        non-transposed ones too. On torch (``backward_all`` False) non-transposed
        micro-batches backprop into the preprocess via the connected graph.

        For a frozen / param-free preprocess (no trainable params) there is no
        recompute backward, so the body order is returned unchanged (transposed
        outputs are placed by ``MPIPE_FWD_RECV``; non-transposed micro-batches
        are handled by stage 0 directly). VL's frozen-visual injection is wired
        per-model rather than through this text-style body-input path.
        """
        if not has_trainable:
            return order
        patched = []
        for step in order:
            is_stage0_fwd = (
                step is not None
                and step.type == MetaStepType.FWD
                and step.stage_index == 0
            )
            if is_stage0_fwd and step.micro_index >= num_transpose:
                patched.append(MetaStep(step.micro_index, MpipeStepType.MPIPE_TRANSPOSE_FWD, 0))
            patched.append(step)
            is_stage0_bwd = (
                step is not None
                and step.type == MetaStepType.BWD
                and step.stage_index == 0
            )
            if is_stage0_bwd and (step.micro_index < num_transpose or backward_all):
                patched.append(MetaStep(step.micro_index, MpipeStepType.MPIPE_TRANSPOSE_BWD, 0))
        return patched
