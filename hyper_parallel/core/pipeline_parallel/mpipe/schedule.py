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
import logging
from typing import Callable, Optional, TYPE_CHECKING

from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType
from hyper_parallel.core.pipeline_parallel.scheduler import (
    MetaStep,
    MetaStepType,
    ScheduleInterleaved1F1B,
)
from hyper_parallel.core.pipeline_parallel.mpipe.sampler import mpipe_owned_micros
from hyper_parallel.core.pipeline_parallel.mpipe.step_types import MpipeStepType

if TYPE_CHECKING:
    from hyper_parallel.core.pipeline_parallel.utils import BatchDimSpec
    from hyper_parallel.dmodule.module import Module

platform = get_platform()
logger = logging.getLogger(__name__)


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
        so the preprocess backward can run centrally on stage 0
        (gradients accumulate on stage 0 only).

    The remaining ``micro_batch_num - NT`` micro-batches run the preprocess
    forward inline on stage 0 (graph-connected to the body, so their backward
    is automatic), exactly as ordinary Interleaved 1F1B.

    The body model (stage 0 = the layers after the preprocess block, all other
    stages unchanged) is scheduled by the inherited Interleaved 1F1B logic; the
    preprocess steps are layered around it: a transpose-phase prefix per rank,
    plus inline preprocess forward / stage-0 backward steps.

    Args:
        stages (list[PipelineStage], PipelineStage): The body pipeline stages,
            given **after** stage 0's first chunk has been partitioned into the
            preprocess block and the body.  Stage 0 here must wrap only the
            layers that remain *after* the preprocess block; the transposed
            preprocess layers are passed separately as ``preprocess_module`` and
            must not appear in ``stages``.
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
            stage-0 backward.
        args_batch_dim (list, optional): See ``PipelineScheduleRuntime``.
        kwargs_batch_dim (dict, optional): See ``PipelineScheduleRuntime``.
        output_concat_dim (int, optional): See ``PipelineScheduleRuntime``.
        overlap_p2p (bool, optional): See ``ScheduleInterleaved1F1B``.
            Default ``False``.
        swap (bool, optional): Reserved for API compatibility. MPipe activation
            swap is not supported and ``True`` raises ``ValueError``.

    Note:
        This class builds the schedule ordering and registers the ``MPIPE_*``
        execution handlers; the handlers themselves live in the platform
        executors (see :class:`MPipeTransposeExecutorBase`).
    """

    # Each rank runs one transposed preprocess forward, so every rank needs its
    # own micro-batch input — drivers (the trainer) check this marker to feed all
    # ranks rather than only the first stage.
    requires_all_rank_input = True

    def __init__(self,
                 stages: list,
                 micro_batch_num: int,
                 preprocess_module: "Optional[Module]",
                 num_transpose_layers: int,
                 num_visual_layers: Optional[int] = None,
                 args_batch_dim: "Optional[BatchDimSpec]" = None,
                 kwargs_batch_dim: "Optional[BatchDimSpec]" = None,
                 output_concat_dim: Optional[int] = None,
                 overlap_p2p: bool = False,
                 swap: bool = False,
                 output_consumer: Optional[Callable] = None,
                 owner_backward: bool = False,
                 less_memory: bool = False,
                 overflow_mode: str = "full") -> None:
        """Build an interleaved-1F1B schedule that transposes the preprocess block.

        Args:
            stages (list): The local pipeline stages (as for :class:`ScheduleInterleaved1F1B`).
            micro_batch_num (int): Number of micro-batches per optimizer step.
            preprocess_module (Optional[Module]): The block transposed to every
                rank — the first ``num_transpose_layers`` layers, a visual tower,
                or a param-free identity for the dataload-only (``T = 0``) mode.
            num_transpose_layers (int): ``T`` — the (informational) transposed-layer count.
            num_visual_layers (int): total number of visual layers.
            args_batch_dim (Optional[BatchDimSpec]): Positional-arg batch-dim spec (forwarded to the base).
            kwargs_batch_dim (Optional[BatchDimSpec]): Keyword-arg batch-dim spec (forwarded to the base).
            output_concat_dim (Optional[int]): Output concatenation dim (forwarded to the base).
            overlap_p2p (bool): Whether to overlap P2P (forwarded to the base).
            swap (bool): Reserved for API compatibility. Must be ``False``.
            output_consumer: Optional ``(ctx, micro, out_tuple) -> None`` callback
                routing a transposed micro-batch's preprocess output (``None`` =
                default, the output becomes stage 0's body input).
            less_memory (bool): Shallow-warmup interleaved body (forwarded to
                the base) — see :class:`ScheduleInterleaved1F1B`. The transpose
                prefix is unaffected.
            owner_backward (bool): Opt-in owner-does-backward for a TRAINABLE
                preprocess on torch. When effective, the owner rank retains the
                forward graph, stage 0 ships the feature gradient back, and the
                owner runs the tower backward in its cooldown (instead of the
                centralized stage-0 backward); tower grads are SUM all-reduced
                across the pp replicas. Ignored (stage-0 backward kept) for a frozen / param-free
                preprocess and on MindSpore.
            overflow_mode (str): How to distribute the ``M > NT`` overflow micros.
                ``"full"`` (default) — round-robin: each owner takes ``M/NT``
                micros, ViT phase balanced. ``"min"`` — rank 0 absorbs the
                overflow (saves inter-rank P2P at the cost of a longer rank-0
                ViT phase).
        """
        if swap:
            raise ValueError(
                "MPipe Transpose with pipeline activation swap is not yet supported."
            )
        if not isinstance(num_transpose_layers, int) or num_transpose_layers < 0:
            raise ValueError(
                f"Argument 'num_transpose_layers' must be a non-negative int, "
                f"but got {num_transpose_layers!r}."
            )
        if overflow_mode not in ("full", "min"):
            raise ValueError(
                f"overflow_mode must be 'full' or 'min', got {overflow_mode!r}"
            )
        self._overflow_mode = overflow_mode
        # ``preprocess_module`` is the resolved block to transpose (first T text
        # layers, the visual tower, or a param-free identity for dataload-only).
        self._preprocess_module = preprocess_module
        self._num_transpose_layers = num_transpose_layers
        self._num_visual_layers = num_visual_layers
        # ``None`` = the preprocess output is stage 0's body input (arg_mbs);
        # VL passes a consumer that routes it into an injection kwarg instead.
        self._output_consumer = output_consumer
        # Trainable preprocess -> broadcast + stage-0 backward; frozen or
        # param-free -> ship the output only (no broadcast, no recompute).
        self._has_trainable_preprocess = self._module_has_trainable_params(preprocess_module)
        # Owner-does-backward needs a trainable preprocess AND torch (MindSpore's
        # grad_fn is body-scoped); otherwise fall back to the stage-0 backward.
        self._owner_backward = (
            bool(owner_backward)
            and self._has_trainable_preprocess
            and platform.platform_type == PlatformType.PYTORCH
        )
        if owner_backward and self._has_trainable_preprocess and not self._owner_backward:
            logger.warning(
                "[mpipe] pp_mpipe_owner_backward requested but unsupported on "
                "platform %s; falling back to the stage-0 backward.",
                platform.platform_type,
            )
        super().__init__(stages,
                         micro_batch_num,
                         args_batch_dim=args_batch_dim,
                         kwargs_batch_dim=kwargs_batch_dim,
                         output_concat_dim=output_concat_dim,
                         overlap_p2p=overlap_p2p,
                         overlap_b_f=False,
                         swap=swap,
                         less_memory=less_memory)
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
    def output_consumer(self) -> Optional[Callable]:
        """Optional ``(ctx, micro, out_tuple)`` placement callback (None = body input)."""
        return self._output_consumer

    @property
    def has_trainable_preprocess(self) -> bool:
        """Whether the preprocess has trainable params (drives broadcast / stage-0 backward)."""
        return self._has_trainable_preprocess

    @property
    def owner_backward(self) -> bool:
        """Whether owner-does-backward is effective (trainable preprocess + torch)."""
        return self._owner_backward

    @property
    def num_transpose_layers(self) -> int:
        """The informational transposed-layer count ``T`` (``0`` = dataload only)."""
        return self._num_transpose_layers

    @property
    def num_transpose_micro_batches(self) -> int:
        """``NT = min(PP, micro_batch_num)`` — count of owner ranks per cycle."""
        return min(self.real_stage_num, self.micro_batch_num)

    def owned_micros(self, pp_rank: int) -> frozenset:
        """Owner set for ``pp_rank`` under :attr:`overflow_mode` (see ``mpipe_owned_micros``)."""
        return mpipe_owned_micros(self.real_stage_num, self.micro_batch_num,
                                  pp_rank, mode=self._overflow_mode)

    def owner_of(self, micro: int) -> int:
        """Rank that owns micro ``micro`` under :attr:`overflow_mode`."""
        nt = self.num_transpose_micro_batches
        if self._overflow_mode == "min":
            return 0 if micro >= nt else micro
        return micro % nt

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
        self._apply_backward_retain_flag()
        handlers = {
            MpipeStepType.MPIPE_PARAM_BROADCAST: self._executor.broadcast_params,
            MpipeStepType.MPIPE_TRANSPOSE_FWD: self._executor.transpose_forward,
            MpipeStepType.MPIPE_FWD_SEND: self._executor.fwd_send,
            MpipeStepType.MPIPE_FWD_RECV: self._executor.fwd_recv,
            MpipeStepType.MPIPE_GRAPH_SEND: self._executor.fwd_bwd_graph_input_send,
            MpipeStepType.MPIPE_GRAPH_RECV: self._executor.fwd_bwd_graph_input_recv,
            MpipeStepType.MPIPE_TRANSPOSE_BWD: self._executor.transpose_backward,
            MpipeStepType.MPIPE_GRAD_SEND: self._executor.grad_send,
            MpipeStepType.MPIPE_GRAD_RECV_WITH_BACKWARD: self._executor.grad_recv_with_backward,
            MpipeStepType.MPIPE_GRAD_REDUCE: self._executor.reduce_tower_grads,
        }
        for step_type, handler in handlers.items():
            self.register_custom_function(step_type, handler)

    def _apply_backward_retain_flag(self) -> None:
        """Flag the stages to retain their body-backward graph for a TRAINABLE tower.

        A trainable transposed tower + FSDP shares one all-gather node across the
        micro graphs that back-propagate into it (the connected micro-0 body graph
        plus the deferred owner/stage-0 tower backwards). The first backward to
        traverse that shared node frees it, so a later one raises "backward through
        the graph a second time" (surfaces at ``vpp >= 2``). Retaining the body
        graph keeps the shared node alive. A frozen tower (no shared trainable
        node) and plain PP (no flag) are unaffected.
        """
        if self._has_trainable_preprocess:
            for stage in self.stages:
                stage.retain_backward_graph = True

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

        The parameter broadcast, the stage-0-backward input transport, and the
        stage-0 backward are emitted only for a **trainable** preprocess; a
        frozen (visual tower) or param-free (``T == 0``) one only transposes
        the forward and ships its output.

        When owner-does-backward is effective (trainable + torch + opt-in) the
        stage-0-backward input transport and stage-0 backward are replaced by a
        feature-gradient ship-back (``MPIPE_GRAD_SEND`` on stage 0,
        ``MPIPE_GRAD_RECV_WITH_BACKWARD`` cooldown suffix on the owners) and a final
        ``MPIPE_GRAD_REDUCE`` collective on every rank.
        """
        super().construct_exec_order()
        body_order = self.exec_order
        nt = self.num_transpose_micro_batches
        micro_batch_num = self.micro_batch_num
        has_trainable = self._has_trainable_preprocess
        owner_backward = self._owner_backward
        mode = self._overflow_mode
        # Body DATA_LOAD steps are rescheduled into the mpipe prefix (and, under
        # "min", inline onto rank 0's body for the overflow micros).
        for i, order in body_order.items():
            body_order[i] = [step for step in order
                             if step is not None and step.type != MetaStepType.DATA_LOAD]
        body_order[0] = self._insert_rank0_preprocess_steps(
            body_order[0], nt, has_trainable, mode, owner_backward=owner_backward,
            num_transpose_layers=self._num_transpose_layers)
        exec_order = {}
        for rank in range(self.real_stage_num):
            order = (
                self._build_transpose_prefix(
                    rank, nt, micro_batch_num, has_trainable, mode,
                    owner_backward=owner_backward,
                    num_transpose_layers=self._num_transpose_layers)
                + body_order[rank]
            )
            if owner_backward:
                # Cooldown suffix: stage 0 ships dL/dfeatures, owners run the
                # tower backward, then all ranks SUM-all-reduce the tower grads.
                order = (
                    order
                    + self._build_owner_backward_suffix(rank, nt, micro_batch_num, mode)
                    + [MetaStep(None, MpipeStepType.MPIPE_GRAD_REDUCE, 0)]
                )
            exec_order[rank] = order
        self.exec_order = exec_order
        # frozen + all visual layers transposed, so stage 0 needs no pixels
        if (not self._has_trainable_preprocess and
                self._num_visual_layers is not None and
                self._num_transpose_layers >= self._num_visual_layers):
            self._DATA_KEYS = ("input_ids",) + tuple(
                k for k in self.kwargs_batch_dim if k != "pixel_values"
            )
        else:
            self._DATA_KEYS = ("input_ids",) + tuple(self.kwargs_batch_dim)

        # Length-M so overflow micros and VPP chunks index in bounds; the
        # prefix key -1 ships owner to rank 0, body stages chain via stages[0].
        self._data_dst = {-1: [0 for _ in range(self.micro_batch_num)]}
        self._data_src = {-1: [self.owner_of(m) for m in range(self.micro_batch_num)]}
        for i in range(self._stage_num):
            self._data_dst[i] = [self.stages[0].dst_stage for _ in range(self.micro_batch_num)]
            self._data_src[i] = [self.stages[0].src_stage for _ in range(self.micro_batch_num)]

    @staticmethod
    def _build_transpose_prefix(rank, num_transpose_micro_batches, micro_batch_num,
                                has_trainable, overflow_mode,
                                owner_backward=False, num_transpose_layers=None):
        """Build the transpose-phase prefix prepended to ``rank``'s body order.

        Under ``overflow_mode="full"`` (round-robin) rank ``rank`` (``< NT``)
        owns ``{m : m % NT == rank}`` — emits DATA_LOAD → MPIPE_TRANSPOSE_FWD
        per owned micro, and (when not rank 0) ships each to rank 0. Rank 0
        receives every non-owned micro. For a **trainable** preprocess every
        rank also broadcasts its (trainable) parameters and ranks ``> 0``
        additionally ship the input for the centralized stage-0 backward; for
        a frozen (visual tower) preprocess only the transpose forward and its
        output send/recv remain.

        Under ``"min"`` each owner (including rank 0) emits exactly one
        transposed micro (``{rank}``). Rank 0 receives only ``1..NT-1``; the
        overflow micros ``NT..M-1`` are loaded inline on rank 0 by
        :meth:`_insert_rank0_preprocess_steps`.

        Dataload-only mode (``num_transpose_layers == 0``): every emission
        of MPIPE_PARAM_BROADCAST, MPIPE_TRANSPOSE_FWD, MPIPE_FWD_SEND/RECV,
        MPIPE_GRAPH_SEND/RECV is suppressed. Only DATA_LOAD + DATA_SEND/RECV
        remain; stage 0 runs the visual tower itself during body forward.
        Ownership (round-robin vs single-owner) still follows
        ``overflow_mode`` so the dataload I/O sharding stays balanced.

        Under owner-does-backward the input ship-back (``MPIPE_GRAPH_SEND`` /
        ``MPIPE_GRAPH_RECV``) is dropped -- the owner keeps its forward graph and
        receives only the feature gradient later (``MPIPE_GRAD_RECV_WITH_BACKWARD``).
        """
        nt = num_transpose_micro_batches
        if rank >= nt:
            owned = []
        elif overflow_mode == "min":
            owned = [rank]
        else:
            owned = sorted(m for m in range(micro_batch_num) if m % nt == rank)
        is_dataload_only = num_transpose_layers == 0
        prefix = []
        if has_trainable and not is_dataload_only:
            prefix.append(MetaStep(rank, MpipeStepType.MPIPE_PARAM_BROADCAST, -1))
        # DATA_LOAD + MPIPE_TRANSPOSE_FWD are local (no comm) — emit per-micro.
        for m in owned:
            prefix.append(MetaStep(m, MetaStepType.DATA_LOAD, -1))
            if not is_dataload_only:
                prefix.append(MetaStep(m, MpipeStepType.MPIPE_TRANSPOSE_FWD, -1))
        # Type-major send order must match rank 0's receive block below, or
        # the meta recv drifts and decodes garbage shapes.
        if rank != 0:
            for m in owned:
                prefix.append(MetaStep(m, MetaStepType.DATA_SEND, -1))
            if not is_dataload_only:
                for m in owned:
                    prefix.append(MetaStep(m, MpipeStepType.MPIPE_FWD_SEND, -1))
                if has_trainable and not owner_backward:
                    for m in owned:
                        prefix.append(MetaStep(m, MpipeStepType.MPIPE_GRAPH_SEND, -1))
        if rank == 0:
            # Non-owned = micros rank 0 didn't transpose locally ("min":
            # overflow micros load inline, so they are not received).
            if overflow_mode == "min":
                non_owned = list(range(1, nt))
            else:
                non_owned = [m for m in range(micro_batch_num) if m not in set(owned)]
            for m in non_owned:
                prefix.append(MetaStep(m, MetaStepType.DATA_RECV, -1))
            if not is_dataload_only:
                for m in non_owned:
                    prefix.append(MetaStep(m, MpipeStepType.MPIPE_FWD_RECV, -1))
                if has_trainable and not owner_backward:
                    for m in non_owned:
                        prefix.append(MetaStep(m, MpipeStepType.MPIPE_GRAPH_RECV, -1))
        return prefix

    @staticmethod
    def _build_owner_backward_suffix(rank, num_transpose_micro_batches, micro_batch_num, overflow_mode):
        """Cooldown suffix (owner-backward), appended AFTER all body steps so the
        feature-grad transport never interleaves with the body's P2P.

        Stage 0 ships ``dL/dfeatures`` for every micro it does not own itself
        (``MPIPE_GRAD_SEND``, one per micro whose owner != 0). Each owner rank
        receives a gradient for **every** micro it owns (``MPIPE_GRAD_RECV_WITH_BACKWARD``,
        one step per owned micro) and backprops its retained tower graph for each.
        Ranks ``>= NT`` own no transposed micro-batch.

        Under ``"min"`` an owner rank owns exactly one micro (``{rank}``), so this
        degenerates to the original one-gradient-per-rank exchange. Under ``"full"``
        (round-robin) with ``M > NT`` an owner rank can own several micros
        (``{m : m % NT == rank}``); each gets its own SEND/RECV pair. Both sides
        iterate ``range(micro_batch_num)`` in ascending order, so for any fixed
        (rank 0, owner) pair the send order matches the receive order -- required
        since HCCL serializes P2P per rank-pair on a single channel.
        """
        nt = num_transpose_micro_batches

        def owner_of(micro: int) -> int:
            """Rank that owns ``micro`` under the active overflow mode."""
            if overflow_mode == "min":
                return 0 if micro >= nt else micro
            return micro % nt

        if rank == 0:
            return [MetaStep(micro, MpipeStepType.MPIPE_GRAD_SEND, 0)
                    for micro in range(micro_batch_num) if owner_of(micro) != 0]
        if rank < nt:
            return [MetaStep(micro, MpipeStepType.MPIPE_GRAD_RECV_WITH_BACKWARD, 0)
                    for micro in range(micro_batch_num) if owner_of(micro) == rank]
        return []

    @staticmethod
    def _insert_rank0_preprocess_steps(order, num_transpose_micro_batches,
                                       has_trainable, overflow_mode, owner_backward=False,
                                       num_transpose_layers=None):
        """Patch stage 0's body order for the overflow / recompute paths.

        Under ``"full"``: every micro is transposed (somewhere). No inline
        preprocess on rank 0; under trainable, MPIPE_TRANSPOSE_BWD fires after
        every stage-0 BWD (the preprocess output is detached either way).

        Under ``"min"``: rank 0 absorbs the overflow micros ``NT..M-1``. Before
        each stage-0 FWD on an overflow micro we insert DATA_LOAD and (under
        trainable or frozen) MPIPE_TRANSPOSE_FWD — running the preprocess
        locally with a detached graph. For frozen this is just "rank 0
        runs the ViT here so the body forward has the encoded features /
        visual-injection kwarg"; for trainable the body BWD also needs an
        MPIPE_TRANSPOSE_BWD stage-0 backward for the preprocess gradient. (The
        previous design used a graph-connected forward to avoid the
        stage-0 backward on overflow micros; that optimization isn't restored
        here — `"min"` saves comm for slightly more rank-0 compute on the
        trainable path.)

        Under owner-does-backward (``owner_backward``) the centralized stage-0
        backward is replaced: the body backward deposits ``dL/dfeatures`` on the
        placed feature tensors, and a stage-0 cooldown SUFFIX
        (:meth:`_build_owner_backward_suffix`) ships it back to the owners -- NOT
        inline here, since the blocking grad meta-exchange would deadlock against
        the body's P2P. So no per-BWD step is inserted for any micro; every
        owned/inline forward on stage 0 already runs graph-connected
        (``transpose_forward``'s owner-backward branch), so the body backward
        flows into the tower automatically.

        Dataload-only (``num_transpose_layers == 0``) suppresses the inline
        MPIPE_TRANSPOSE_FWD, matching the contract documented on
        :meth:`_build_transpose_prefix`: with an identity preprocess there is
        nothing to run ahead of the body, and stage 0 consumes the raw loaded
        data directly.
        """
        if not has_trainable and overflow_mode == "full":
            return order
        nt = num_transpose_micro_batches
        is_dataload_only = num_transpose_layers == 0
        patched = []
        for step in order:
            is_stage0_fwd = (
                step is not None
                and step.type == MetaStepType.FWD
                and step.stage_index == 0
            )
            if (overflow_mode == "min" and is_stage0_fwd
                    and step.micro_index >= nt):
                patched.append(MetaStep(step.micro_index, MetaStepType.DATA_LOAD, -1))
                if not is_dataload_only:
                    # Populates the visual-injection kwarg via output_consumer;
                    # without it rank 0's body forward crashes.
                    patched.append(MetaStep(step.micro_index, MpipeStepType.MPIPE_TRANSPOSE_FWD, -1))
            patched.append(step)
            is_stage0_bwd = (
                step is not None
                and step.type == MetaStepType.BWD
                and step.stage_index == 0
            )
            if is_stage0_bwd and has_trainable and not owner_backward:
                patched.append(MetaStep(step.micro_index, MpipeStepType.MPIPE_TRANSPOSE_BWD, -1))
        return patched
