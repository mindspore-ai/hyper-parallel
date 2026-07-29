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
"""pipeline schedule"""
from abc import ABC, abstractmethod
from enum import Enum, auto
from collections import defaultdict
import itertools
import bisect
import logging
import re
import hyper_parallel
from hyper_parallel.platform import get_platform
from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.core.pipeline_parallel.utils import BatchDimSpec
platform = get_platform()
logger = logging.getLogger(__name__)


class MetaStepType(Enum):
    """Specify the enumeration type for MetaStep."""
    FWD = auto()
    BWD = auto()
    BWD_INPUT = auto()
    BWD_WEIGHT = auto()
    FWD_RECV = auto()
    FWD_SEND = auto()
    BWD_RECV = auto()
    BWD_SEND = auto()
    # Composite P2P: a contiguous run of FWD_SEND/FWD_RECV/BWD_SEND/BWD_RECV
    # coalesced by ``coalesce_p2p`` into one step whose ``sub_steps`` the runtime
    # groups by peer and issues as ``batch_isend_irecv`` (same-peer send+recv ->
    # duplex).  Produced under ``p2p_transport="batch"`` or ``"multi_stream"``.
    BATCH_SEND_RECV = auto()
    OVERLAP_F_B = auto()
    OVERLAP_B_F = auto()
    FSDP_UNSHARD = auto()
    FSDP_RESHARD = auto()
    FSDP_REDUCE_GRAD = auto()
    FSDP_WAIT_REDUCE_GRAD = auto()
    SWAP_SET_GROUP = auto()
    SWAP_LAUNCH_OFFLOAD = auto()
    SWAP_WAIT_OFFLOAD = auto()
    SWAP_LAUNCH_LOAD = auto()
    SWAP_WAIT_LOAD = auto()


class MetaStep:
    """
    Meta step of PipelineSchedule.
    An execution list composed of MetaStep can be constructed
    and fed into the PipelineSchedule for execution.

    Args:
        micro_index (int | None): The index of micro-batch.  ``None`` for
            composite types (``OVERLAP_F_B`` / ``OVERLAP_B_F``) whose real
            micro index lives in each ``sub_steps`` entry.
        type (MetaStepType): Specify the type of current step.
        stage_index (int | None): Stage index of current step.  ``None``
            for composite types; use ``sub_steps`` to get each direction's
            stage.
        sub_steps (tuple[MetaStep, MetaStep] | None): For composite types
            only: ``(fwd, bwd)`` for ``OVERLAP_F_B``, ``(bwd, fwd)`` for
            ``OVERLAP_B_F``.
        boundary_p2p (tuple[MetaStep, ...] | None): For ``OVERLAP_B_F`` under
            the ``"boundary"`` P2P transport only: P2P steps to issue at the
            fwd/bwd boundary inside the overlap (the forward's ``FWD_SEND``
            plus the next slot's recvs), hoisted out of the following gap by
            ``attach_fwd_boundary_p2p``.  Issued via
            :meth:`PipelineScheduleRuntime.exec_boundary_p2p`.
    """
    def __init__(self, micro_index, meta_type, stage_index, sub_steps=None,
                 boundary_p2p=None):
        self._type = meta_type
        self._micro_index = micro_index
        self._stage_index = stage_index
        self._sub_steps = sub_steps
        self._boundary_p2p = boundary_p2p

    @property
    def micro_index(self):
        """Return the micro-batch index of this step."""
        return self._micro_index

    @property
    def stage_index(self):
        """Return the stage index of this step."""
        return self._stage_index

    @property
    def type(self):
        """Return the MetaStepType of this step."""
        return self._type

    @property
    def sub_steps(self):
        """Sub-steps for composite types: ``(fwd, bwd)`` for OVERLAP_F_B,
        ``(bwd, fwd)`` for OVERLAP_B_F, or ``None``."""
        return self._sub_steps

    @property
    def boundary_p2p(self):
        """P2P steps to issue at the overlap's fwd/bwd boundary, or ``None``."""
        return self._boundary_p2p

    def __eq__(self, value):
        if not isinstance(value, MetaStep):
            return NotImplemented
        return (self.type == value.type
                and self.micro_index == value.micro_index
                and self.stage_index == value.stage_index
                and self.sub_steps == value.sub_steps)

    def __ne__(self, value):
        if not isinstance(value, MetaStep):
            return NotImplemented
        return not self.__eq__(value)

    def __hash__(self):
        return hash((self.type, self.micro_index, self.stage_index))

    def __str__(self):
        if self.sub_steps:
            sub = ", ".join(str(s) for s in self.sub_steps)
            return (f"MetaStep(type={self.type}, micro_index={self.micro_index}, "
                    f"stage_index={self.stage_index}, sub_steps=[{sub}])")
        return f"MetaStep(type={self.type}, micro_index={self.micro_index}, stage_index={self.stage_index})"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_str(step_str):
        """Parse a MetaStep from its string representation."""
        pass


def generate_stage_to_rank_mapping(real_stage_num, stage_num, style='loop'):
    """Generate stage to rank mapping for loop or V schedules."""
    if style == 'loop':
        return {stage_index: stage_index % real_stage_num for stage_index in range(stage_num)}
    if style == 'v':
        if stage_num % real_stage_num != 0:
            raise ValueError(
                f"stage_num {stage_num} must be evenly divisible by real_stage_num {real_stage_num} for V schedules."
            )
        mapping = {}
        rank_index = 0
        for stage_index in range(stage_num):
            mapping[stage_index] = rank_index
            if (stage_index + 1) % real_stage_num == 0:
                continue
            if (stage_index // real_stage_num) % 2 == 0:
                rank_index += 1
            else:
                rank_index -= 1
        return mapping
    raise ValueError(f"Unsupported stage rank mapping style: {style}")


def generate_rank_to_stage_mapping(real_stage_num, stage_num, style='loop'):
    """Invert the stage to rank mapping."""
    stage_to_rank = generate_stage_to_rank_mapping(real_stage_num, stage_num, style)
    rank_to_stages = defaultdict(list)
    for stage_index, rank in stage_to_rank.items():
        rank_to_stages[rank].append(stage_index)

    for stages in rank_to_stages.values():
        stages.sort()
    return dict(rank_to_stages)


def iter_leaf_meta_steps(step):
    """Yield leaf MetaSteps, recursively expanding OVERLAP containers.

    Both ``OVERLAP_F_B`` and ``OVERLAP_B_F`` carry their real FWD/BWD work in
    ``sub_steps``; the FSDP unshard/reshard injection pass relies on this to
    see the FWD/BWD buried inside an overlap. Missing ``OVERLAP_B_F`` here let
    an overlapped FWD run against a resharded stage → "expected HSDPModule
    parameters in unsharded state". Mirror the other expansion sites
    (run/_expand/add_fsdp_*) which already handle both composite types.
    """
    if step is None:
        return
    if step.type in (MetaStepType.OVERLAP_F_B, MetaStepType.OVERLAP_B_F) and step.sub_steps:
        for sub_step in step.sub_steps:

            yield from iter_leaf_meta_steps(sub_step)
        return
    yield step


class PipelineContext:
    """Per-run state handed to a custom execution function (see
    :meth:`PipelineScheduleRuntime.register_custom_function`).

    A plain data carrier for one :meth:`PipelineScheduleRuntime.run_microbatches`
    call.  The P2P helpers (``wait_fwd_recv`` / ``wait_bwd_recv`` / ``send_fwd``
    / ``send_bwd``) and the ``enable_dxdw_split`` flag live on the schedule, so a
    callback reaches them through :attr:`schedule`, e.g.
    ``ctx.schedule.send_bwd(stage, micro_index)``.

    Attributes:
        schedule: The owning :class:`PipelineScheduleRuntime`.
        arg_mbs: Per-micro-batch positional args.
        kwarg_mbs: Per-micro-batch keyword args.
        losses: Mutable list collecting per-step losses.
    """

    def __init__(self, schedule: "PipelineScheduleRuntime", arg_mbs: list,
                 kwarg_mbs: list, losses: list) -> None:
        """Bundle the active schedule with one run's micro-batch inputs and losses."""
        self.schedule = schedule
        self.arg_mbs = arg_mbs
        self.kwarg_mbs = kwarg_mbs
        self.losses = losses


def _exec_fsdp_unshard(stage):
    """Unshard every HSDPModule in the stage's submodule tree."""
    for _, module in platform.get_cells_and_names(stage.submodule):
        if isinstance(module, HSDPModule):
            module.unshard()


def _exec_fsdp_reshard(stage):
    """Reshard every HSDPModule in the stage's submodule tree."""
    for _, module in platform.get_cells_and_names(stage.submodule):
        if isinstance(module, HSDPModule):
            module.reshard()


def _exec_fsdp_reduce_grad(stage):
    """Launch HSDP reduction after the stage's final backward."""
    stage.launch_reduce_grad()


def _exec_fsdp_wait_reduce_grad(stage):
    """Drain the final pending HSDP reduction tail."""
    stage.wait_reduce_grad()


# FSDP control MetaStep -> handler(stage). Membership also marks which
# MetaStepTypes are FSDP control steps, so the runtime loop dispatches with a
# single table lookup instead of re-switching on the step type.
_FSDP_STEP_HANDLERS = {
    MetaStepType.FSDP_UNSHARD: _exec_fsdp_unshard,
    MetaStepType.FSDP_RESHARD: _exec_fsdp_reshard,
    MetaStepType.FSDP_REDUCE_GRAD: _exec_fsdp_reduce_grad,
    MetaStepType.FSDP_WAIT_REDUCE_GRAD: _exec_fsdp_wait_reduce_grad,
}


class PipelineScheduleRuntime(ABC):
    """
    Base class for pipeline schedule.
    Implements the `split_microbatches` and `run_microbatches` method.
    Derived classes should implement `run_microbatches` method and `run` method.

    Supports registering **custom execution functions** for any
    :class:`MetaStepType` via :meth:`register_custom_function`.  When
    ``run_microbatches`` encounters a step whose type has a registered
    handler, it creates a :class:`PipelineContext` and delegates execution
    to the handler instead of using the built-in logic.

    Args:
        stages (list[PipelineStage], PipelineStage):  PipelineStage used to run_microbatches.
        micro_batch_num (int): The number of micro-batch.
        args_batch_dim (int | BatchDimSpec | list | tuple, optional): Per
            positional-arg batch dim, indexed by arg position.  Entries may be
            plain ``int`` (or ``None`` to keep the default); a single-input
            model may pass a bare ``int``/``BatchDimSpec`` instead of a
            one-element list (wrapped automatically). Default ``None``.
        kwargs_batch_dim (dict, optional): Per keyword-arg batch dim, mapping
            arg name to a plain ``int`` or ``BatchDimSpec``. Default ``None``.
        swap (bool, optional): Whether to inject pipeline activation swap
            control steps. Supported by ``ScheduleGPipe``, ``Schedule1F1B``,
            and ``ScheduleInterleaved1F1B``. Default ``False``.
        p2p_transport (str, optional): How pipeline send/recv are issued.
            ``"auto"`` (default) — gap-time duplex batching on overlap_b_f
            schedules (``coalesce_p2p``: same-peer send+recv as one
            ``batch_isend_irecv``, TX||RX; hardware-validated and measured a
            net win on real workloads), plain per-op ``isend``/``irecv``
            everywhere else.  ``"plain"`` — force per-op ``isend``/``irecv``
            (escape hatch for transports or topologies where batching
            misbehaves).  ``"batch"`` — duplex batching explicitly (what
            ``"auto"`` picks under overlap_b_f).  ``"boundary"`` —
            EXPERIMENTAL fwd-boundary batching: each overlap's ``F_SEND`` +
            the next slot's recvs go out mid-overlap, right after the forward,
            as per-op solo batches; only ``B_SEND`` waits for the backward.
            Avoids the duplex handle's send-coupling (a2a-friendly) and posts
            the activation send ~half a slot early, but is not yet
            hardware-validated — opt in deliberately.  Must be set identically
            on every rank — HCCL cannot match a batched op against a plain one
            (EI0005). ``"multi_stream"`` — EXPERIMENTAL: the same gap-time duplex
            batching as ``"batch"``, but each adjacent PP peer pair uses its
            own two-rank communication group. MindSpore's default per-group
            communication-stream policy then lets the previous and next PP
            edges progress on separate streams.
    """

    _P2P_TRANSPORTS = ("auto", "plain", "batch", "boundary", "multi_stream")

    def __init__(self,
                 stages,
                 micro_batch_num,
                 args_batch_dim=None,
                 kwargs_batch_dim=None,
                 output_concat_dim=None,
                 overlap_p2p=False,
                 swap=False,
                 p2p_transport="auto"):
        if p2p_transport not in self._P2P_TRANSPORTS:
            raise ValueError(
                f"p2p_transport must be one of {self._P2P_TRANSPORTS}, got "
                f"{p2p_transport!r}"
            )
        self.stages = self._check_stages(stages)
        self.micro_batch_num = micro_batch_num
        self._args_batch_dim = self._normalize_args_batch_dim(args_batch_dim)
        self._kwargs_batch_dim = self._normalize_kwargs_batch_dim(kwargs_batch_dim)
        self._output_concat_dim = output_concat_dim
        self.split_micro_batch = platform.micro_batch(self.micro_batch_num,
                                                      self._args_batch_dim, self._kwargs_batch_dim)
        self.n_local_stages = len(self.stages)
        self._stage_dict = self.convert_stages_dict()
        self.real_stage_num = self.stages[0].stage_num // self.n_local_stages
        self._stage_num = self.stages[0].stage_num
        self._stage_to_rank_index = None
        self._overlap_p2p = overlap_p2p
        self.exec_order = {}
        self._init_stages()
        self._build_stage_to_rank_index()
        self.fwd_handle_cache = {}
        self.bwd_handle_cache = {}
        self._custom_fn_map = {}
        self._pp_swap_enabled = swap
        # Outstanding async send handle groups for the in-flight
        # ``run_microbatches`` call; reset per run and drained at its end.
        self._send_handles = []
        # ``p2p_transport`` resolves in ``build_exec_order`` (it needs the
        # subclass's ``_overlap_b_f``) to one of:
        #
        #  * ``"batch"`` (the ``"auto"`` default on overlap_b_f schedules) —
        #    gap-time duplex via ``coalesce_p2p``: same-peer send+recv as one
        #    ``batch_isend_irecv`` (TX||RX on the full-duplex link).
        #    Hardware-validated and MEASURED a net win on real workloads — the
        #    duplex saving outweighs its known cost (MS's single handle couples
        #    the riding send into the compute-gating recv wait, which can
        #    shave EP a2a overlap).
        #  * ``"plain"`` (the ``"auto"`` default elsewhere) — per-op
        #    ``isend``/``irecv``, the upstream-original path.
        #  * ``"boundary"`` (EXPERIMENTAL, explicit opt-in only) — fwd-boundary
        #    batching.  ``attach_fwd_boundary_p2p`` hangs each steady gap's
        #    ``F_SEND`` (its data is ready when the overlap's forward finishes
        #    — the backward, ~2x FLOPs, is the long pole) plus the next slot's
        #    recvs on the OVERLAP_B_F step; the stage's after-forward hook
        #    fires ``exec_boundary_p2p`` at the fwd/bwd boundary, while the
        #    backward is still running.  The send leaves roughly half a slot
        #    early — the only mode that moves the SENDER's post time — and the
        #    recv handles carry no send (a2a-friendly).  Every op is a per-op
        #    solo batch and ``coalesce_p2p`` is NOT run: a hoisted recv cannot
        #    stay duplexed with a send that is only ready later, and
        #    asymmetric shapes (one end duplex, other end split) hang —
        #    all-solo keeps per-pair batch sequences complementary
        #    ([S,R] vs [R,S]), safe under both candidate HCCL pairing
        #    semantics.  Promote to the auto default only after it earns both
        #    a hardware accuracy pass and a perf win over "batch".
        #  * ``"multi_stream"`` (EXPERIMENTAL, explicit opt-in only) — keeps the
        #    ``"batch"`` schedule and same-peer duplex fusion, but routes each
        #    physical PP edge through a fixed two-rank group. With MindSpore's
        #    ``multi_stream:group`` runtime policy, the previous and next
        #    edges then use separate communication streams. Group count is
        #    bounded by physical PP degree (at most two per rank), not by the
        #    number of micro-batches.
        #
        # Two transport invariants for any future rewrite: plain per-pair
        # streams need the gap's recv-first/send-first complementarity (a
        # send-crossing hoist made both ends recv-first -> rendezvous deadlock,
        # 2026-06), and batch pairing needs per-pair shape mirroring (a
        # one-sided split made 2 solos face 1 duplex -> hang, 2026-06).
        self._p2p_transport = p2p_transport
        # Effective mode + per-op batch gating; set by ``build_exec_order``.
        self._p2p_mode = None
        self._batch_p2p = False
        # ``peer_global_rank -> two-rank process group``. Populated only for
        # the experimental ``p2p_transport="multi_stream"`` mode.
        self._p2p_multi_stream_groups = {}
        # OVERLAP steps whose boundary_p2p was already issued this run (the
        # stage after-forward hook and the post-step safety net are both
        # allowed to call exec_boundary_p2p; reset per run_microbatches call).
        self._boundary_issued = set()
        # (fwd stage_index, micro_index) -> armed OVERLAP step, consumed by the
        # stage after-forward hook to fire the boundary issue mid-overlap.
        self._pending_boundary = {}

    def register_custom_function(self, step_type: MetaStepType, fn) -> None:
        """Register a custom execution function for the given step type.

        When :meth:`run_microbatches` encounters a :class:`MetaStep` whose
        ``type`` matches ``step_type``, it calls ``fn(step, ctx)`` instead
        of the built-in logic.

        Args:
            step_type: The :class:`MetaStepType` to intercept.
            fn: A callable with signature ``(step: MetaStep, ctx: PipelineContext) -> None``.

        Example:
            >>> def my_overlap_callback(step, ctx):
            ...     fwd_step, bwd_step = step.sub_steps
            ...     # custom parallel execution logic
            >>> schedule.register_custom_function(MetaStepType.OVERLAP_F_B, my_overlap_callback)
        """
        self._custom_fn_map[step_type] = fn

    def _inject_local_fsdp_actions(self):
        """Annotate the local rank schedule with optional FSDP control actions."""
        current_rank = self._stage_to_rank_index[self.stages[0].stage_index]
        managed_stage_indices = {
            stage.stage_index
            for stage in self.stages
            if isinstance(stage.submodule, HSDPModule)
        }
        if not managed_stage_indices:
            return
        if len(managed_stage_indices) != len(self.stages):
            raise RuntimeError(
                "When injecting fsdp_action, expect all stages to be HSDPModule. "
                "Check whether all separated modules are wrapped with 'fully_shard'."
            )
        # Pipeline MetaSteps own parameter resharding, so disable the HSDP
        # forward hook once when building the schedule instead of once per
        # micro-batch. The root API applies the setting recursively.
        for stage in self.stages:
            stage.submodule.set_reshard_after_forward(False)
        rank_actions = add_fsdp_unshard_reshard(self.exec_order[current_rank], managed_stage_indices)
        self.exec_order[current_rank] = add_fsdp_reduce_grad(
            rank_actions,
            managed_stage_indices,
            self.micro_batch_num,
        )

    def _prepare_fsdp_backward(self):
        """Disable HSDP post-backward actions once for the current pipeline run.

        Pipeline accumulates gradients through each stage's final backward.
        ``FSDP_REDUCE_GRAD`` then restores gradient sync and launches the
        reduction on the schedule thread. Reset the flags at the next run
        boundary.
        """
        for stage in self.stages:
            if not stage.has_backward:
                continue
            if isinstance(stage.submodule, HSDPModule):
                stage.submodule.set_reshard_after_backward(False)
                stage.submodule.set_requires_gradient_sync(False)

    def _inject_local_pp_swap_actions(self):
        """Annotate the local rank schedule with pipeline activation-swap actions."""
        if not self._pp_swap_enabled:
            return
        current_rank = self._stage_to_rank_index[self.stages[0].stage_index]
        from hyper_parallel.core.pipeline_parallel.pipeline_swap import (  # pylint: disable=C0415
            inject_pipeline_swap_steps,
        )
        self.exec_order[current_rank] = inject_pipeline_swap_steps(self.exec_order[current_rank])

    @abstractmethod
    def _build_stage_to_rank_index(self) -> None:
        """
        Build attribute of  _stage_to_rank_index.
        Each subclass constructs it according to its own schedule style.
        """

    @abstractmethod
    def construct_exec_order(self) -> None:
        """Build exec order, PP cmopute and PP comms(Send/Recv)"""

    def build_exec_order(self) -> None:
        """Build the execution order and inject optional PP-swap/FSDP actions.

        Also resolves ``p2p_transport``: ``"auto"`` becomes ``"batch"`` (the
        measured-beneficial duplex) on schedules running with ``overlap_b_f``
        and ``"plain"`` everywhere else, then the matching order-rewrite pass
        runs last (after swap/FSDP injection) so it sees the final per-rank
        order.
        """
        mode = self._p2p_transport
        if mode == "auto":
            mode = "batch" if getattr(self, "_overlap_b_f", False) else "plain"
        self._p2p_mode = mode
        self._batch_p2p = mode != "plain"
        if mode == "multi_stream":
            self._init_p2p_multi_stream_groups()
        self.construct_exec_order()
        self._inject_local_pp_swap_actions()
        self._inject_local_fsdp_actions()
        if mode == "boundary":
            # fwd-boundary mode: hang the forward's F_SEND + next slot's recvs
            # on the OVERLAP step (issued mid-overlap, right after the
            # forward).  Everything stays per-op solo batches — deliberately
            # NO coalesce_p2p (see __init__).
            self.exec_order = attach_fwd_boundary_p2p(self.exec_order)
        elif mode in ("batch", "multi_stream"):
            # Coalesce contiguous P2P runs into BATCH_SEND_RECV so the runtime
            # issues same-peer send+recv as one duplex batch.  NOTE: couples
            # the riding send into the compute-gating recv wait — see
            # __init__.
            self.exec_order = coalesce_p2p(self.exec_order)

    def _init_p2p_multi_stream_groups(self) -> None:
        """Create adjacent two-rank groups for multi-stream P2P transport."""
        first_stage = self.stages[0]
        mesh = getattr(first_stage, "mesh", None)
        pp_rank_list = (
            list(mesh.rank_list)
            if mesh is not None
            else platform.get_process_group_ranks(first_stage.pp_group)
        )
        self._p2p_multi_stream_groups = platform.create_p2p_multi_stream_groups(
            pp_rank_list,
            include_wrap=self.n_local_stages > 1,
        )

    def _p2p_op(self, op_type, tensor, peer):
        """Build a P2P descriptor, selecting the multi-stream group when enabled."""
        group = None
        if self._p2p_mode == "multi_stream":
            group = self._p2p_multi_stream_groups.get(peer)
            if group is None:
                raise RuntimeError(
                    f"No P2P multi-stream group was created for peer global rank {peer}. "
                    f"Available peers are {sorted(self._p2p_multi_stream_groups)}."
                )
        return platform.p2p_op(op_type, tensor, peer, group)

    def convert_stages_dict(self):
        """convert stages to dict."""
        stage_dict = {}
        for stage in self.stages:
            stage_dict[stage.stage_index] = stage
        return stage_dict

    def split_microbatches(self, args, kwargs):
        """split_microbatches."""
        if args or kwargs:
            args_split, kwargs_split = self.split_micro_batch(args, kwargs)
            return args_split, kwargs_split
        return [[] for _ in range(self.micro_batch_num)], [{} for _ in range(self.micro_batch_num)]

    @staticmethod
    def _to_spec(elem):
        """Normalize one batch-dim entry: ``int`` -> ``BatchDimSpec``.

        ``None`` and ``BatchDimSpec`` pass through unchanged.  ``bool`` is
        rejected even though it is an ``int`` subclass, so ``True``/``False``
        are not silently read as dims 1/0.
        """
        if elem is None or isinstance(elem, BatchDimSpec):
            return elem
        if isinstance(elem, int) and not isinstance(elem, bool):
            return BatchDimSpec(elem)
        raise TypeError(
            f"batch-dim entry must be int, BatchDimSpec or None, but got {type(elem)}.")

    @staticmethod
    def _normalize_args_batch_dim(args_batch_dim):
        """Accept a plain ``int``/``BatchDimSpec`` or a ``list``/``tuple`` of them.

        ``args_batch_dim`` is a per-arg spec indexed by positional-arg position
        (see ``_MicroBatch``).  A single-input model can pass a bare ``int`` /
        ``BatchDimSpec`` instead of the awkward one-element
        ``BatchDimSpec.from_tuple((0,))``; elements may be plain ``int`` (or
        ``None`` to keep the default).  Always returns ``None`` or a
        ``tuple[BatchDimSpec | None]`` so downstream per-arg indexing is
        unchanged.
        """
        if args_batch_dim is None:
            return None
        if isinstance(args_batch_dim, BatchDimSpec) or \
                (isinstance(args_batch_dim, int) and not isinstance(args_batch_dim, bool)):
            args_batch_dim = (args_batch_dim,)
        if isinstance(args_batch_dim, (list, tuple)):
            return tuple(PipelineScheduleRuntime._to_spec(e) for e in args_batch_dim)
        raise TypeError(
            f"args_batch_dim must be int, BatchDimSpec or a list/tuple of them, "
            f"but got {type(args_batch_dim)}.")

    @staticmethod
    def _normalize_kwargs_batch_dim(kwargs_batch_dim):
        """Accept plain ``int`` dict values: ``{\"x\": 0}`` -> ``{\"x\": BatchDimSpec(0)}``.

        ``kwargs_batch_dim`` maps each keyword-arg name to its batch dim.
        Returns ``None`` or a ``dict[str, BatchDimSpec | None]`` so downstream
        per-key indexing is unchanged.
        """
        if kwargs_batch_dim is None:
            return None
        if not isinstance(kwargs_batch_dim, dict):
            raise TypeError(
                f"kwargs_batch_dim must be a dict[str, int | BatchDimSpec], "
                f"but got {type(kwargs_batch_dim)}.")
        return {k: PipelineScheduleRuntime._to_spec(v) for k, v in kwargs_batch_dim.items()}

    def _check_stages(self, stages):
        """check stages type."""
        if isinstance(stages, hyper_parallel.PipelineStage):
            return [stages]
        if isinstance(stages, (list, tuple)):
            for stage in stages:
                if not isinstance(stage, hyper_parallel.PipelineStage):
                    raise TypeError(f"Argument 'stages' must be type of PipelineStage, \
                                     list or tuple of PipelineStage, but got list or tuple of {type(stage)}.")
            return stages
        raise TypeError(f"Argument 'stages' must be type of PipelineStage, \
                         list or tuple of PipelineStage, but got type of {type(stages)}.")

    def _init_stages(self):
        """init stages."""
        for stage in self.stages:
            stage.init(self.n_local_stages)
            # After-forward hook: lets the schedule issue fwd-boundary P2P the
            # moment a forward chunk completes (no-op unless an OVERLAP step
            # with boundary_p2p was armed for that (stage, micro)).
            stage._after_forward_chunk = self._on_forward_chunk_done  # pylint: disable=W0212

    def _on_forward_chunk_done(self, stage_index, micro_index):
        """Stage after-forward hook: fire the armed boundary P2P, if any.

        Runs on the thread executing the forward (the overlap callback's
        ``fwd_fn`` / the main thread), at the fwd/bwd boundary — the paired
        backward is still running, so the boundary ops overlap it.  Keyed by
        the overlap's forward ``(stage_index, micro_index)``; unrelated
        forwards (warm-up steps, recompute re-runs of past micros) miss the
        key and no-op.
        """
        step = self._pending_boundary.pop((stage_index, micro_index), None)
        if step is not None:
            self.exec_boundary_p2p(step)

    def run(self, *args, **kwargs):
        """schedule run."""
        losses = []
        try:
            split_args, split_kwargs = self.split_microbatches(args, kwargs)
            self.run_microbatches(split_args, split_kwargs, losses)
        finally:
            # An exception unwinds past run_microbatches' end-of-iteration send
            # drain, leaving in-flight isend/irecv handles un-waited. Wait them
            # here so the comm contract holds on the error path too. No-op on the
            # normal path. See _drain_inflight_p2p.
            self._drain_inflight_p2p()
        return losses

    def sync_shared_parameters_grad(self):
        """sync_shared_parameters_grad."""
        for stage in self.stages:
            stage.sync_shared_parameters_grad()

    def update_losses(self, stage, loss, losses):
        """update_losses."""
        if stage.is_last_stage:
            losses.append(loss)

    @property
    def enable_dxdw_split(self) -> bool:
        """Whether this schedule splits ``OVERLAP_B_F`` backward into dx/dw."""
        return getattr(self, "_enable_dxdw_split", False)

    def _wait_p2p(self, handles):
        for handle in handles:
            if handle is not None:
                handle.wait()

    def _drain_inflight_p2p(self):
        """Wait every P2P handle still in flight — error-path cleanup.

        run_microbatches waits its deferred sends only in the end-of-iteration
        drain; an exception mid-iteration unwinds past that drain, leaving issued
        isend/irecv handles un-waited in ``_send_handles`` and the recv caches
        (the ``CommHandle destroyed without calling wait()`` warning). run()'s
        finally calls this so every handle is still ``wait()``-ed — honoring the
        comm contract — on the error path too, and pops them so a later run()
        does not re-wait stale handles (the recv caches are never reset per run).
        No-op on the normal path: the drain already emptied ``_send_handles`` and
        every cached recv was consumed.
        """
        while self._send_handles:
            self._wait_p2p(self._send_handles.pop())
        while self.fwd_handle_cache:
            self._wait_p2p(self.fwd_handle_cache.popitem()[1])
        while self.bwd_handle_cache:
            self._wait_p2p(self.bwd_handle_cache.popitem()[1])

    def _batched_issue(self, specs):
        """Launch same-peer P2P ``specs`` as one ``batch_isend_irecv`` group.

        ``specs`` are ``(op_type, tensor, peer_global_rank)`` from the stage's
        ``*_specs`` builders (which carry the meta/bookkeeping side effects).
        Returns ``[handle]`` (the single batch handle) or ``[]`` — shaped like
        the per-op ``exec_*_ops`` return so the cache / drain paths are
        unchanged.  Only the launch is coalesced; matching stays per-peer FIFO.
        """
        if not specs:
            return []
        ops = [self._p2p_op(op_type, tensor, peer) for op_type, tensor, peer in specs]
        handle = platform.batch_isend_irecv(ops)
        return [handle] if handle is not None else []

    # --- P2P step primitives ------------------------------------------------
    # One method per cross-rank comm action, used both by the runtime loop
    # (``_exec_step``) and by OVERLAP callbacks (via ``ctx.schedule``).  With
    # ``overlap_p2p=True`` comm is decoupled from its compute: a recv caches its
    # handles for the consuming step to ``wait_*`` later, and a send defers its
    # handles to the end-of-iteration drain.  With ``overlap_p2p=False`` every
    # op waits inline.

    def recv_fwd(self, stage: "hyper_parallel.PipelineStage", micro_index: int) -> None:
        """Post the FWD recv for ``micro_index``; cache it (overlap_p2p) or wait now."""
        handles = (self._batched_issue(stage.fwd_recv_specs(micro_index))
                   if self._batch_p2p else stage.exec_fwd_recv_ops(micro_index))
        if self._overlap_p2p:
            self.fwd_handle_cache[(stage.stage_index, micro_index)] = handles
        else:
            self._wait_p2p(handles)

    def recv_bwd(self, stage: "hyper_parallel.PipelineStage", micro_index: int) -> None:
        """Post the BWD recv for ``micro_index``; cache it (overlap_p2p) or wait now."""
        handles = (self._batched_issue(stage.bwd_recv_specs(micro_index))
                   if self._batch_p2p else stage.exec_bwd_recv_ops(micro_index))
        if self._overlap_p2p:
            self.bwd_handle_cache[(stage.stage_index, micro_index)] = handles
        else:
            self._wait_p2p(handles)

    def wait_fwd_recv(self, stage_index: int, micro_index: int) -> None:
        """Wait the FWD recv cached by :meth:`recv_fwd`; no-op if nothing is cached."""
        handles = self.fwd_handle_cache.pop((stage_index, micro_index), None)
        if handles:
            self._wait_p2p(handles)

    def wait_bwd_recv(self, stage_index: int, micro_index: int) -> None:
        """Wait the BWD recv cached by :meth:`recv_bwd`; no-op if nothing is cached."""
        handles = self.bwd_handle_cache.pop((stage_index, micro_index), None)
        if handles:
            self._wait_p2p(handles)

    def send_fwd(self, stage: "hyper_parallel.PipelineStage", micro_index: int) -> None:
        """Send this stage's forward output for ``micro_index`` to the next stage."""
        handles = (self._batched_issue(stage.fwd_send_specs(micro_index))
                   if self._batch_p2p else stage.exec_fwd_send_ops(micro_index)) or []
        if self._overlap_p2p:
            # Append the whole handle group: run_microbatches drains _send_handles
            # group by group, so a bare handle would be wrongly iterated as a list.
            self._send_handles.append(handles)
        else:
            self._wait_p2p(handles)

    def send_bwd(self, stage: "hyper_parallel.PipelineStage", micro_index: int) -> None:
        """Send this stage's input-gradient for ``micro_index`` to the previous stage.

        Driven by the scheduler's ``BWD_SEND`` step. It pops the input grad that
        the backward (unified ``backward_one_chunk`` or, under
        ``enable_dxdw_split=True``, ``backward_input_one_chunk``) wrote to the
        stage's ``bwd_cache``. Calling it manually in addition to the scheduled
        ``BWD_SEND`` would double-send the gradient.
        """
        handles = (self._batched_issue(stage.bwd_send_specs(micro_index))
                   if self._batch_p2p else stage.exec_bwd_send_ops(micro_index)) or []
        if self._overlap_p2p:
            self._send_handles.append(handles)
        else:
            self._wait_p2p(handles)

    def _arm_boundary(self, step):
        """Register ``step`` for the stage after-forward hook; return its key.

        No-op (returns ``None``) unless ``step`` carries ``boundary_p2p``.  The
        key is the overlap's forward ``(stage_index, micro_index)`` — exactly
        what the hook receives when that forward chunk completes.
        """
        if not getattr(step, "boundary_p2p", None) or not step.sub_steps:
            return None
        fwd_sub = next((s for s in step.sub_steps
                        if s.type == MetaStepType.FWD), None)
        if fwd_sub is None:
            return None
        key = (fwd_sub.stage_index, fwd_sub.micro_index)
        self._pending_boundary[key] = step
        return key

    def _finish_boundary(self, step, armed_key) -> None:
        """Post-step safety net: issue any boundary P2P the hook missed."""
        self.exec_boundary_p2p(step)
        if armed_key is not None:
            self._pending_boundary.pop(armed_key, None)

    def exec_boundary_p2p(self, step) -> None:
        """Issue ``step.boundary_p2p`` (fwd-boundary P2P) once per run.

        Fired by the stage after-forward hook (``_on_forward_chunk_done``) the
        moment the overlap's forward chunk completes — the backward is still
        running on its own thread, so the F_SEND leaves ~half a slot early and
        the next slot's recvs are already posted when the peers' sends arrive.
        Idempotent per ``run_microbatches`` call: the post-step safety net
        (``_finish_boundary``) also invokes it, so an overlap whose forward
        never went through ``forward_one_chunk`` degrades to gap-time issue
        order instead of dropping the ops.

        Dispatches through the existing per-op helpers (``send_fwd`` /
        ``recv_fwd`` / ``recv_bwd``), so batching, handle caching for
        ``wait_*_recv`` and deferred-send bookkeeping behave exactly like the
        scheduled steps they replace.  No-op for steps without
        ``boundary_p2p``.
        """
        ops = getattr(step, "boundary_p2p", None)
        if not ops or id(step) in self._boundary_issued:
            return
        self._boundary_issued.add(id(step))
        for sub in ops:
            stage = self._stage_dict[sub.stage_index]
            if sub.type == MetaStepType.FWD_SEND:
                self.send_fwd(stage, sub.micro_index)
            elif sub.type == MetaStepType.FWD_RECV:
                self.recv_fwd(stage, sub.micro_index)
            elif sub.type == MetaStepType.BWD_RECV:
                self.recv_bwd(stage, sub.micro_index)
            elif sub.type == MetaStepType.BWD_SEND:
                # attach_fwd_boundary_p2p never hoists BWD_SEND (its grad is
                # produced by the backward still in flight); defensive only.
                self.send_bwd(stage, sub.micro_index)

    # ``op_type -> (specs builder name, route kind)`` for a coalesced sub-step.
    # ``route`` is the recv-cache kind (so wait_*_recv finds the handle), or
    # ``None`` for a send (no local consumer).
    _BATCH_SUB_DISPATCH = {
        MetaStepType.FWD_RECV: ("fwd_recv_specs", "fwd"),
        MetaStepType.BWD_RECV: ("bwd_recv_specs", "bwd"),
        MetaStepType.FWD_SEND: ("fwd_send_specs", None),
        MetaStepType.BWD_SEND: ("bwd_send_specs", None),
    }

    def _exec_batch_send_recv(self, step) -> None:
        """Execute a coalesced P2P run: one ``batch_isend_irecv`` per peer.

        Builds each sub-step's specs (same meta/bookkeeping side effects as the
        per-step ``recv_fwd`` / ``send_fwd`` / ...), groups every op by peer
        global rank, and issues one batch per peer so a same-peer send+recv runs
        duplex.  Handle routing mirrors the per-step path: under
        ``overlap_p2p`` a batch carrying a recv is cached for ``wait_*_recv``
        (its send rides along), a send-only batch defers to ``_send_handles``;
        without ``overlap_p2p`` every batch waits inline.
        """
        # (op_type, tensor, peer, route) per op; route = (kind, stage, micro) for
        # a recv, else None.
        tagged = []
        for sub in step.sub_steps:
            builder_name, kind = self._BATCH_SUB_DISPATCH[sub.type]
            stage = self._stage_dict[sub.stage_index]
            specs = getattr(stage, builder_name)(sub.micro_index)
            route = (kind, sub.stage_index, sub.micro_index) if kind is not None else None
            for op_type, tensor, peer in specs:
                tagged.append((op_type, tensor, peer, route))

        by_peer = {}
        for item in tagged:
            by_peer.setdefault(item[2], []).append(item)

        for items in by_peer.values():
            ops = [self._p2p_op(op_type, tensor, peer) for op_type, tensor, peer, _ in items]
            handle = platform.batch_isend_irecv(ops)
            if handle is None:
                continue
            if not self._overlap_p2p:
                self._wait_p2p([handle])
                continue
            recv_routes = [route for *_, route in items if route is not None]
            if recv_routes:
                for kind, si, mi in recv_routes:
                    cache = self.fwd_handle_cache if kind == "fwd" else self.bwd_handle_cache
                    cache[(si, mi)] = [handle]
            else:
                self._send_handles.append([handle])

    def _assert_in_unshard_if_needed(self, stage, check_step):
        if not isinstance(stage.submodule, HSDPModule):
            return
        submodule_hsdp_scheduler = stage.submodule.hsdp_scheduler
        scheduler_state = submodule_hsdp_scheduler.hsdp_state
        if scheduler_state.is_shard:
            raise RuntimeError(
                f"Executing MetaStep: {check_step}, expected HSDPModule parameters in unsharded "
                f"state, but got sharded parameters."
            )

    def _exec_step(self, cur_step, arg_mbs, kwarg_mbs, losses):
        """Execute one built-in step (non-custom, non-composite).

        Each comm step dispatches to a single P2P primitive; each compute step
        first waits its cached recv (a no-op under ``overlap_p2p=False``) and
        then runs.
        """
        stage = self._stage_dict[cur_step.stage_index]
        micro_index = cur_step.micro_index
        step_type = cur_step.type

        if step_type in (
            MetaStepType.SWAP_SET_GROUP,
            MetaStepType.SWAP_LAUNCH_OFFLOAD,
            MetaStepType.SWAP_WAIT_OFFLOAD,
            MetaStepType.SWAP_LAUNCH_LOAD,
            MetaStepType.SWAP_WAIT_LOAD,
        ):
            self._exec_pipeline_swap_step(cur_step, arg_mbs, kwarg_mbs)

        elif step_type == MetaStepType.FWD_RECV:
            self.recv_fwd(stage, micro_index)

        elif step_type == MetaStepType.FWD:
            self._assert_in_unshard_if_needed(stage, cur_step)
            self.wait_fwd_recv(stage.stage_index, micro_index)
            out = stage.forward_one_chunk(micro_index, arg_mbs[micro_index], kwarg_mbs[micro_index])
            self.update_losses(stage, out, losses)

        elif step_type == MetaStepType.FWD_SEND:
            self.send_fwd(stage, micro_index)

        elif step_type == MetaStepType.BWD_RECV:
            self.recv_bwd(stage, micro_index)

        elif step_type == MetaStepType.BWD_INPUT:
            self._assert_in_unshard_if_needed(stage, cur_step)
            self.wait_bwd_recv(stage.stage_index, micro_index)
            stage.backward_input_one_chunk(micro_index)

        elif step_type == MetaStepType.BWD_WEIGHT:
            self._assert_in_unshard_if_needed(stage, cur_step)
            self.wait_bwd_recv(stage.stage_index, micro_index)
            stage.backward_weight_one_chunk(micro_index)

        elif step_type == MetaStepType.BWD:
            self._assert_in_unshard_if_needed(stage, cur_step)
            self.wait_bwd_recv(stage.stage_index, micro_index)
            stage.backward_one_chunk(micro_index)

        elif step_type == MetaStepType.BWD_SEND:
            self.send_bwd(stage, micro_index)

        else:
            # FSDP control steps dispatch via the handler table; any other type
            # is a no-op here (composite/custom types are handled upstream).
            fsdp_handler = _FSDP_STEP_HANDLERS.get(step_type)
            if fsdp_handler is not None:
                fsdp_handler(stage)

    def _exec_pipeline_swap_step(self, cur_step, arg_mbs, kwarg_mbs):
        """Execute a pipeline activation-swap control step."""
        from hyper_parallel.core.pipeline_parallel.pipeline_swap import (  # pylint: disable=C0415
            swap_launch_load,
            swap_launch_offload,
            swap_set_group,
            swap_wait_load,
            swap_wait_offload,
        )

        if cur_step.type == MetaStepType.SWAP_SET_GROUP:
            swap_set_group(cur_step)
        elif cur_step.type == MetaStepType.SWAP_LAUNCH_OFFLOAD:
            swap_launch_offload(cur_step, self, arg_mbs, kwarg_mbs)
        elif cur_step.type == MetaStepType.SWAP_WAIT_OFFLOAD:
            swap_wait_offload(cur_step)
        elif cur_step.type == MetaStepType.SWAP_LAUNCH_LOAD:
            swap_launch_load(cur_step)
        elif cur_step.type == MetaStepType.SWAP_WAIT_LOAD:
            swap_wait_load(cur_step)

    def run_microbatches(self, arg_mbs: list, kwarg_mbs: list, losses: list) -> None:
        """Execute the schedule step by step.

        Steps whose :attr:`MetaStep.type` has a registered custom function
        are delegated to that function with a :class:`PipelineContext`.
        Composite ``OVERLAP_F_B`` / ``OVERLAP_B_F`` steps without a
        registered handler fall back to executing their ``sub_steps``
        sequentially via :meth:`_exec_step` — correct but without
        comm/compute overlap.  All other steps are executed by
        :meth:`_exec_step`.

        Logs one ``DEBUG`` line per non-bubble step showing the rank's
        progress: ``rank=<r> step=<i>/<n> <MetaStep>``.  Enable with
        ``logging.getLogger('hyper_parallel.core.pipeline_parallel.scheduler')
        .setLevel(logging.DEBUG)`` to trace per-rank schedule advancement
        (handy when diagnosing deadlocks or callback ordering issues).
        """
        real_stage_index = self.stages[0].stage_index % self.real_stage_num
        self._prepare_fsdp_backward()
        self._send_handles = []
        self._boundary_issued = set()
        self._pending_boundary = {}
        ctx = None  # lazily created

        ordered = self.exec_order[real_stage_index]
        total_steps = len(ordered)
        logger.debug(
            "run_microbatches start: rank=%d total_steps=%d micro_batch_num=%d",
            real_stage_index, total_steps, self.micro_batch_num,
        )

        for step_idx, cur_step in enumerate(ordered):
            if cur_step is None:
                continue

            logger.debug(
                "rank=%d step=%d/%d %s",
                real_stage_index, step_idx, total_steps, cur_step,
            )

            # Arm the fwd-boundary hook: when this step carries boundary_p2p,
            # the stage's after-forward hook fires exec_boundary_p2p the moment
            # the overlap's forward chunk completes (works for any callback —
            # no callback cooperation needed).
            armed_key = self._arm_boundary(cur_step)

            # Check for registered custom function
            custom_fn = self._custom_fn_map.get(cur_step.type)
            if custom_fn is not None:
                if ctx is None:
                    ctx = PipelineContext(self, arg_mbs, kwarg_mbs, losses)
                custom_fn(cur_step, ctx)
                # Safety net: if the forward hook never fired (custom fwd path),
                # issue the boundary ops now (gap-time order) instead of
                # dropping them.  Idempotent.
                self._finish_boundary(cur_step, armed_key)
                continue

            # Coalesced P2P block: group sub-steps by peer, issue one
            # batch_isend_irecv per peer (same-peer send+recv -> duplex).
            if cur_step.type == MetaStepType.BATCH_SEND_RECV:
                self._exec_batch_send_recv(cur_step)
                continue

            # Default for composite OVERLAP steps: run sub_steps sequentially.
            # P2P send/recv around these steps are already laid out in two
            # virtual slots by ``add_send_recv``, so sequential execution is
            # semantically equivalent to non-overlapped 1F1B.
            if (cur_step.type in (MetaStepType.OVERLAP_F_B, MetaStepType.OVERLAP_B_F)
                    and cur_step.sub_steps):
                for sub in cur_step.sub_steps:
                    self._exec_step(sub, arg_mbs, kwarg_mbs, losses)
                self._finish_boundary(cur_step, armed_key)
                continue

            self._exec_step(cur_step, arg_mbs, kwarg_mbs, losses)

        logger.debug(
            "run_microbatches end: rank=%d pending_send_handles=%d",
            real_stage_index, len(self._send_handles),
        )
        self.sync_shared_parameters_grad()
        while self._send_handles:
            self._wait_p2p(self._send_handles.pop())


class _OverlapPhantom:
    """Internal marker used by :func:`add_send_recv` to expand an
    ``OVERLAP_F_B`` or ``OVERLAP_B_F`` step into two virtual time slots.

    An overlap step composes two sub-steps (``B + F`` or ``F + B``) that
    execute concurrently on the GPU but occupy **two** logical time slots
    in the column-scan sender timeline — the sender can only finish
    emitting the second sub-step's output after the first sub-step has
    completed.  Treating an overlap step as a single slot places the RECV
    triggered by the second sub-step too early on the receiver.

    Each overlap step is expanded into two phantoms:
      * ``is_first_half=True`` — represents the first sub-step's emission
        slot; the original overlap step is emitted into the output
        schedule here (only once).
      * ``is_first_half=False`` — represents the second sub-step's emission
        slot; only its send/recv comms are inserted.
    """

    __slots__ = ('obf_step', 'sub_step', 'is_first_half')

    def __init__(self, obf_step, sub_step, is_first_half: bool):
        self.obf_step = obf_step
        self.sub_step = sub_step
        self.is_first_half = is_first_half


def _expand_overlap_slots(scheduler, real_stage_num):
    """Expand OVERLAP steps in a per-rank schedule into 2 virtual time slots.

    Returns a new ``{rank: [MetaStep | _OverlapPhantom | None, ...]}`` dict
    where each OVERLAP step is replaced by a pair of phantoms.  Non-OVERLAP
    entries pass through unchanged.
    """
    expanded = {}
    for rank in range(real_stage_num):
        order = scheduler[rank]
        exp = []
        for op in order:
            if (op is not None
                    and op.type in (MetaStepType.OVERLAP_F_B, MetaStepType.OVERLAP_B_F)
                    and op.sub_steps):
                exp.append(_OverlapPhantom(op, op.sub_steps[0], is_first_half=True))
                exp.append(_OverlapPhantom(op, op.sub_steps[1], is_first_half=False))
            else:
                exp.append(op)
        expanded[rank] = exp
    return expanded


def _process_rank_items(real_stage_num, current_items, insert_step_comms, new_schedule):
    """Run ``insert_step_comms`` for each rank's current item, even ranks first.

    Even-before-odd ordering avoids P2P deadlocks between adjacent ranks.
    """
    for rank in range(0, real_stage_num, 2):
        item = current_items.get(rank)
        if item is not None:
            sub = item.sub_step if isinstance(item, _OverlapPhantom) else item
            insert_step_comms(sub, rank, new_schedule)
    for rank in range(1, real_stage_num, 2):
        item = current_items.get(rank)
        if item is not None:
            sub = item.sub_step if isinstance(item, _OverlapPhantom) else item
            insert_step_comms(sub, rank, new_schedule)


def _column_scan_insert_comms(expanded, real_stage_num, insert_step_comms):
    """Column-scan over an OVERLAP-expanded schedule to insert SEND/RECV.

    Processes ``expanded`` one time slot at a time.  Emits the original
    overlap step into ``new_schedule`` only once (at the first-half
    phantom).  Delegates comm insertion to ``insert_step_comms`` for each
    plain step or phantom's underlying sub-step.

    Even ranks are processed before odd ranks at each time step to avoid
    P2P deadlocks between adjacent ranks.

    Args:
        expanded: Result of :func:`_expand_overlap_slots`.
        real_stage_num: Number of physical ranks.
        insert_step_comms: Callable ``(step, rank, new_schedule) -> None``
            that inserts SEND/RECV for a single FWD/BWD step.

    Returns:
        ``{rank: [MetaStep, ...]}`` final schedule.
    """
    max_length = max(len(order) for order in expanded.values())
    new_schedule = {rank: [] for rank in range(real_stage_num)}

    for time_step in range(max_length):
        current_items = {}
        for rank in range(real_stage_num):
            if time_step < len(expanded[rank]):
                item = expanded[rank][time_step]
                current_items[rank] = item
                if item is None:
                    # Preserve bubble slots to keep per-rank time-step
                    # indexing aligned with the column scan.  The runtime
                    # loop skips ``None`` entries, so this is execution-
                    # semantics-neutral.
                    new_schedule[rank].append(None)
                    continue
                if isinstance(item, _OverlapPhantom):
                    # Emit the overlap step only once, at the first-half slot.
                    if item.is_first_half:
                        new_schedule[rank].append(item.obf_step)
                else:
                    new_schedule[rank].append(item)
            else:
                current_items[rank] = None

        _process_rank_items(
            real_stage_num, current_items, insert_step_comms, new_schedule,
        )

    return new_schedule


_P2P_STEP_TYPES = frozenset({
    MetaStepType.FWD_SEND, MetaStepType.FWD_RECV,
    MetaStepType.BWD_SEND, MetaStepType.BWD_RECV,
})


def coalesce_p2p(exec_order):
    """Coalesce maximal contiguous runs of >=2 P2P steps into BATCH_SEND_RECV.

    A *run* is a maximal sequence of consecutive ``FWD_SEND`` / ``FWD_RECV`` /
    ``BWD_SEND`` / ``BWD_RECV`` steps with no compute / overlap / bubble (``None``)
    step between them — so no recv in the run is consumed before the batch is
    issued, and all sends' data is already produced.  Each such run is replaced
    by a single :class:`MetaStep` of type ``BATCH_SEND_RECV`` carrying the run as
    ``sub_steps`` (order preserved, so per-direction FIFO is kept); the runtime
    groups those sub-steps by peer and issues one ``batch_isend_irecv`` per peer
    (same-peer send+recv -> duplex).  Runs of length 1 are left untouched (the
    per-op batched path still batches them, so every transfer is still
    batch-vs-batch).  Pure ``exec_order -> exec_order`` transform.

    Args:
        exec_order: ``{rank: [MetaStep | None, ...]}``.

    Returns:
        A new ``{rank: [...]}`` with contiguous P2P runs coalesced.
    """
    def _flush(run, new):
        if len(run) >= 2:
            new.append(MetaStep(None, MetaStepType.BATCH_SEND_RECV, None, sub_steps=tuple(run)))
        else:
            new.extend(run)

    out = {}
    for rank, order in exec_order.items():
        new = []
        run = []
        for step in order:
            if step is not None and step.type in _P2P_STEP_TYPES:
                run.append(step)
                continue
            _flush(run, new)
            run = []
            new.append(step)
        _flush(run, new)
        out[rank] = new
    return out


_RECV_STEP_TYPES = frozenset({MetaStepType.FWD_RECV, MetaStepType.BWD_RECV})


def attach_fwd_boundary_p2p(exec_order):
    """Hang each overlap gap's boundary-safe P2P on the OVERLAP_B_F step.

    For every ``OVERLAP_B_F`` step, the contiguous P2P run right after it is
    split by data readiness at the overlap's fwd/bwd boundary (forward is the
    short side; backward, ~2x FLOPs, is the long pole):

      * the forward's own ``FWD_SEND`` — its payload exists the moment the
        forward sub-step finishes, no need to wait out the backward;
      * every ``FWD_RECV`` / ``BWD_RECV`` — no data dependency at all;

    are removed from the gap and attached to the OVERLAP step as
    ``boundary_p2p`` (order: ``F_SEND`` first, then the recvs in original
    order), to be issued by :meth:`PipelineScheduleRuntime.exec_boundary_p2p`
    at the boundary, while the backward is still running.  ``BWD_SEND`` (its
    grad is produced by that backward) and any send not produced by this
    overlap's forward stay in the gap.

    Pairing shape (the reason this composition is safe where naive
    hoist+coalesce hung): with every op issued as a per-op solo batch, each
    pair's per-pair batch sequence per slot is ``[F_SEND, B_RECV]`` on the
    prev end and ``[F_RECV, B_SEND]`` on the next end — complementary at every
    position and equal in count, so it matches under both per-direction FIFO
    and per-pair shape-mirroring semantics.  Per-direction FIFO data order is
    preserved (each direction's ops keep their relative order; they all shift
    by the same amount).  Pure ``exec_order -> exec_order`` transform.

    Args:
        exec_order: ``{rank: [MetaStep | None, ...]}``.

    Returns:
        A new ``{rank: [...]}`` with boundary P2P attached to OVERLAP steps.
    """
    out = {}
    for rank, order in exec_order.items():
        new = []
        i = 0
        while i < len(order):
            step = order[i]
            if (step is None or step.type != MetaStepType.OVERLAP_B_F
                    or not step.sub_steps):
                new.append(step)
                i += 1
                continue
            run, j = _p2p_run_after(order, i + 1)
            boundary, leftover = _split_boundary_run(step, run)
            if not boundary:
                new.append(step)
                i += 1
                continue
            new.append(MetaStep(step.micro_index, step.type, step.stage_index,
                                sub_steps=step.sub_steps, boundary_p2p=boundary))
            new.extend(leftover)
            i = j
        out[rank] = new
    return out


def _p2p_run_after(order, start):
    """Collect the contiguous P2P run starting at ``start``.

    Returns ``(run, end)`` where ``end`` is the index of the first step past
    the run (a compute step, ``None`` bubble, or end of order).
    """
    run = []
    j = start
    while j < len(order) and order[j] is not None and order[j].type in _P2P_STEP_TYPES:
        run.append(order[j])
        j += 1
    return run, j


def _split_boundary_run(step, run):
    """Split an overlap's trailing P2P run by fwd/bwd-boundary data readiness.

    Returns ``(boundary, leftover)``: ``boundary`` holds the overlap's own
    forward ``FWD_SEND`` (payload ready at the boundary) first, then every
    recv (no data dependency), keeping original order; ``leftover`` keeps the
    sends produced by the still-running backward, in place.
    """
    fwd_sub = next((s for s in step.sub_steps
                    if s.type == MetaStepType.FWD), None)

    def _is_own_fwd_send(s):
        return (fwd_sub is not None
                and s.type == MetaStepType.FWD_SEND
                and s.stage_index == fwd_sub.stage_index
                and s.micro_index == fwd_sub.micro_index)

    boundary = ([s for s in run if _is_own_fwd_send(s)]
                + [s for s in run if s.type in _RECV_STEP_TYPES])
    taken = {id(s) for s in boundary}
    leftover = [s for s in run if id(s) not in taken]
    return tuple(boundary), leftover


def split_overlap_dxdw(exec_order: dict) -> dict:
    """Split each OVERLAP_B_F backward into dx (in the pair) + dw (after the gap).

    Rewrites ``(BWD, FWD)`` sub_steps to ``(BWD_INPUT, FWD)`` and inserts the
    matching ``BWD_WEIGHT`` after the contiguous P2P run that follows the
    overlap.  The overlap then joins at ``max(dx, fwd)`` instead of
    ``max(dx + dw, fwd)``, so the gap's ``BWD_SEND`` (dx already wrote its
    grad to ``bwd_cache``) and the next slot's recvs are issued a dw earlier;
    under ``overlap_p2p`` they are async and dw computes while they fly.

    Comm placement is untouched: the pass runs after ``add_send_recv`` and
    only moves local compute, so the cross-rank matching order is identical,
    and the P2P run stays contiguous (dw lands after it, not inside) so
    ``coalesce_p2p`` / ``attach_fwd_boundary_p2p`` see the same gap shape.

    First-stage (``stage_index == 0``) backwards stay unified: their dx is a
    no-op (no input grad to compute or send), so splitting would only move
    the whole backward out of the overlap and lose its fwd overlap.

    Args:
        exec_order: ``{rank: [MetaStep | None, ...]}``.

    Returns:
        A new ``{rank: [...]}`` with overlap backwards split into dx/dw.
    """
    out = {}
    for rank, order in exec_order.items():
        new = []
        i = 0
        n = len(order)
        while i < n:
            step = order[i]
            i += 1
            if (step is None or step.type != MetaStepType.OVERLAP_B_F
                    or not step.sub_steps):
                new.append(step)
                continue
            bwd_sub, fwd_sub = step.sub_steps
            if bwd_sub.type != MetaStepType.BWD or bwd_sub.stage_index == 0:
                new.append(step)
                continue
            dx = MetaStep(bwd_sub.micro_index, MetaStepType.BWD_INPUT, bwd_sub.stage_index)
            new.append(MetaStep(step.micro_index, step.type, step.stage_index,
                                sub_steps=(dx, fwd_sub)))
            run, i = _p2p_run_after(order, i)
            new.extend(run)
            new.append(MetaStep(bwd_sub.micro_index, MetaStepType.BWD_WEIGHT,
                                bwd_sub.stage_index))
        out[rank] = new
    return out


def add_send_recv(scheduler, stage_num, real_stage_num, style='loop'):
    """Insert P2P send/recv operations into a per-rank compute schedule.

    For each FWD or BWD step that requires cross-rank communication, a
    ``FWD_SEND`` / ``BWD_SEND`` is appended to the sender's schedule and a
    ``FWD_RECV`` / ``BWD_RECV`` is appended to the receiver's schedule.

    ``OVERLAP_F_B`` / ``OVERLAP_B_F`` composite steps are expanded into
    **two** virtual time slots during the column scan so that the RECV
    triggered by the **second** sub-step lands in the receiver's schedule
    one slot later — matching the fact that the sender can only finish
    emitting the second sub-step's output after the first completes.

    Even ranks are processed before odd ranks at each time step to avoid
    P2P deadlocks between adjacent ranks.

    The resulting per-gap op order (steady state:
    ``[B_RECV, B_SEND, F_RECV, F_SEND]``) is LOAD-BEARING, not cosmetic.
    Each adjacent rank pair shares one comm (HCCL split-by-group) on which
    plain send/recv execute in queue order, and this layout makes one end of
    every pair recv-first while the other is send-first, so the two queue
    heads always match (recv<->send), then the tails match (send<->recv).
    Any later pass that reorders P2P ops relative to EACH OTHER breaks this:
    a (since removed) hoist variant that moved recvs across sends made both
    ends recv-first and deadlocked on hardware (2026-06).
    ``attach_fwd_boundary_p2p`` (the ``"boundary"`` transport) is safe: it keeps
    every per-direction FIFO and runs on the batch transport with per-op solo
    batches, whose per-pair sequences stay complementary.

    Args:
        scheduler: ``{rank: [MetaStep | None, ...]}`` — compute schedule
            with ``None`` for bubble slots.
        stage_num: Total number of virtual pipeline stages.
        real_stage_num: Number of physical ranks.
        style: Topology mapping — ``'loop'`` or ``'v'``.

    Returns:
        ``{rank: [MetaStep, ...]}`` — schedule with communication ops inserted.
    """

    def stage_to_rank(stage_index: int) -> int:
        """Map a virtual stage index to its physical rank."""
        if style == 'loop':
            return stage_index % real_stage_num
        if style == 'v':
            if stage_index < real_stage_num:
                return stage_index
            return stage_num - 1 - stage_index
        raise ValueError(f"Argument 'style' must be 'loop' or 'v', but got {style!r}.")

    def _fwd_peer(stage_index: int):
        """Return the rank that receives this stage's forward output, or None."""
        if stage_index >= stage_num - 1:
            return None
        peer = stage_to_rank(stage_index + 1)
        return peer if peer != stage_to_rank(stage_index) else None

    def _bwd_peer(stage_index: int):
        """Return the rank that receives this stage's backward gradient, or None."""
        if stage_index <= 0:
            return None
        peer = stage_to_rank(stage_index - 1)
        return peer if peer != stage_to_rank(stage_index) else None

    def _insert_comms_for_step(step, rank, new_schedule):
        """Insert send/recv for a single FWD, BWD, or composite OVERLAP step."""
        if step is None:
            return

        if step.type == MetaStepType.FWD:
            peer = _fwd_peer(step.stage_index)
            if peer is not None:
                new_schedule[rank].append(
                    MetaStep(step.micro_index, MetaStepType.FWD_SEND, step.stage_index))
                new_schedule[peer].append(
                    MetaStep(step.micro_index, MetaStepType.FWD_RECV, step.stage_index + 1))

        elif step.type == MetaStepType.BWD:
            peer = _bwd_peer(step.stage_index)
            if peer is not None:
                new_schedule[rank].append(
                    MetaStep(step.micro_index, MetaStepType.BWD_SEND, step.stage_index))
                new_schedule[peer].append(
                    MetaStep(step.micro_index, MetaStepType.BWD_RECV, step.stage_index - 1))

        elif step.type in (MetaStepType.OVERLAP_F_B, MetaStepType.OVERLAP_B_F) and step.sub_steps:
            for sub in step.sub_steps:
                _insert_comms_for_step(sub, rank, new_schedule)

    # --- Main logic: expand OVERLAP steps into 2 virtual slots, then scan ---
    expanded = _expand_overlap_slots(scheduler, real_stage_num)
    return _column_scan_insert_comms(expanded, real_stage_num, _insert_comms_for_step)


_ALIGN_PAD = object()
"""Sentinel marking a forced 1F1B-boundary bubble produced during alignment."""


def _step_dep_ready(step, rank, t, done, stage_num, stage_to_rank):
    """Cross-rank data dependency check used by the alignment simulator.

    A FWD step at stage ``s`` depends on FWD at stage ``s-1`` (on a
    different rank); BWD at stage ``s`` depends on BWD at stage ``s+1``.
    Steps at boundaries or whose producer lives on the same rank are
    always ready.
    """
    si, mi = step.stage_index, step.micro_index
    if step.type == MetaStepType.FWD:
        if si == 0 or stage_to_rank(si - 1) == rank:
            return True
        key = (MetaStepType.FWD, si - 1, mi)
        return key in done and done[key] < t
    if step.type == MetaStepType.BWD:
        if si == stage_num - 1 or stage_to_rank(si + 1) == rank:
            return True
        key = (MetaStepType.BWD, si + 1, mi)
        return key in done and done[key] < t
    return True


def _simulate_aligned_schedule(padded, stage_num, real_stage_num, stage_to_rank):
    """Simulate execution time-step by time-step, inserting bubbles where
    a step is not yet ready (cross-rank dep) or where the cooldown
    rhythm requires it.

    Args:
        padded:          ``{rank: [step | _ALIGN_PAD | None, ...]}`` after
                         1F1B-boundary padding.
        stage_num:       Total number of virtual pipeline stages.
        real_stage_num:  Number of physical ranks.
        stage_to_rank:   Topology mapping from stage to rank.

    Returns:
        ``{rank: [step | None, ...]}`` ready for the column-scan SEND/RECV
        insertion phase.
    """
    remaining_fwd = {
        rank: sum(
            1 for s in padded[rank]
            if s is not _ALIGN_PAD and s is not None and s.type == MetaStepType.FWD
        )
        for rank in range(real_stage_num)
    }
    cursors = {r: 0 for r in range(real_stage_num)}
    aligned = {r: [] for r in range(real_stage_num)}
    done = {}
    last_was_cooldown_bwd = {r: False for r in range(real_stage_num)}
    max_t = sum(len(v) for v in padded.values()) + real_stage_num * 20

    def _emit_bubble(rank):
        aligned[rank].append(None)
        last_was_cooldown_bwd[rank] = False

    def _emit_step(rank, step, t, in_cooldown):
        aligned[rank].append(step)
        done[(step.type, step.stage_index, step.micro_index)] = t
        cursors[rank] += 1
        if step.type == MetaStepType.FWD:
            remaining_fwd[rank] -= 1
        last_was_cooldown_bwd[rank] = in_cooldown and step.type == MetaStepType.BWD

    def _step_rank_at(t, rank):
        if cursors[rank] >= len(padded[rank]):
            return
        item = padded[rank][cursors[rank]]
        if item is _ALIGN_PAD:
            _emit_bubble(rank)
            cursors[rank] += 1
            return
        in_cooldown = remaining_fwd[rank] == 0
        # Cooldown rhythm: alternate None / BWD in pure-BWD phase.
        cooldown_skip = (
            in_cooldown
            and item.type == MetaStepType.BWD
            and last_was_cooldown_bwd[rank]
        )
        if cooldown_skip:
            _emit_bubble(rank)
            return
        if not _step_dep_ready(item, rank, t, done, stage_num, stage_to_rank):
            _emit_bubble(rank)
            return
        _emit_step(rank, item, t, in_cooldown)

    for t in range(max_t):
        if all(cursors[r] >= len(padded[r]) for r in range(real_stage_num)):
            break
        for rank in range(real_stage_num):
            _step_rank_at(t, rank)
    return aligned


def auto_align_and_add_send_recv(scheduler, stage_num, real_stage_num, style='loop'):
    """Auto-insert bubble alignment and P2P send/recv into a pure-compute schedule.

    Unlike :func:`add_send_recv` which requires the caller to pre-insert
    ``None`` bubble slots for time-step alignment, this function accepts a
    **pure compute order** (``FWD`` / ``BWD`` only, no ``None`` needed) and
    automatically determines bubble placement via execution simulation.

    Three constraints are enforced:

    1. **Data dependency** — a ``FWD(stage_k)`` cannot execute until
       ``FWD(stage_{k-1})`` on its source rank has completed (and
       analogously for ``BWD``).
    2. **1F1B transition alignment** — ``real_stage_num - 1 - rank`` padding
       slots are inserted at the warmup → 1F1B boundary (detected as the
       first ``FWD`` immediately followed by a ``BWD`` in the compute order)
       so that all ranks enter the 1F1B steady state in lockstep.
    3. **Cooldown rhythm** — once a rank exhausts its ``FWD`` ops and enters
       pure-``BWD`` cooldown, consecutive ``BWD`` steps are separated by a
       ``None`` slot, maintaining the column-phase-sync property (no rank
       does ``BWD`` while another does ``FWD`` at the same time step).

    After alignment, a column-scan pass inserts ``FWD_SEND`` / ``FWD_RECV``
    and ``BWD_SEND`` / ``BWD_RECV`` with the same prefetch semantics as
    :func:`add_send_recv`.

    Args:
        scheduler: ``{rank: [MetaStep, ...]}`` — pure compute schedule.
            ``None`` entries are silently stripped before processing.
        stage_num: Total number of virtual pipeline stages.
        real_stage_num: Number of physical ranks.
        style: Topology mapping — ``'loop'`` or ``'v'``.

    Returns:
        ``{rank: [MetaStep, ...]}`` — fully aligned schedule with bubbles
        and communication ops inserted.
    """

    # ---- topology helpers (shared with column-scan phase) ----

    def stage_to_rank(stage_index: int) -> int:
        if style == 'loop':
            return stage_index % real_stage_num
        if style == 'v':
            if stage_index < real_stage_num:
                return stage_index
            return stage_num - 1 - stage_index
        raise ValueError(f"Argument 'style' must be 'loop' or 'v', but got {style!r}.")

    def _fwd_peer(stage_index: int):
        if stage_index >= stage_num - 1:
            return None
        peer = stage_to_rank(stage_index + 1)
        return peer if peer != stage_to_rank(stage_index) else None

    def _bwd_peer(stage_index: int):
        if stage_index <= 0:
            return None
        peer = stage_to_rank(stage_index - 1)
        return peer if peer != stage_to_rank(stage_index) else None

    # ---- Phase 1: strip None, detect 1F1B boundary, insert transition padding ----

    def _find_1f1b_boundary(order):
        """Index of the first FWD followed by BWD; ``len(order)`` if absent."""
        for i in range(len(order) - 1):
            if (order[i].type == MetaStepType.FWD
                    and order[i + 1].type == MetaStepType.BWD):
                return i
        return len(order)

    padded = {}
    for rank in range(real_stage_num):
        order = [s for s in scheduler[rank] if s is not None]
        boundary = _find_1f1b_boundary(order)
        pad_count = real_stage_num - 1 - rank
        padded[rank] = order[:boundary] + [_ALIGN_PAD] * pad_count + order[boundary:]

    # ---- Phase 2: simulate execution with data deps + cooldown rhythm ----

    aligned = _simulate_aligned_schedule(padded, stage_num, real_stage_num, stage_to_rank)

    # ---- Phase 3: column-scan SEND/RECV insertion (same as add_send_recv) ----

    def _insert_comms_for_step(step, rank, new_schedule):
        if step is None:
            return
        if step.type == MetaStepType.FWD:
            peer = _fwd_peer(step.stage_index)
            if peer is not None:
                new_schedule[rank].append(
                    MetaStep(step.micro_index, MetaStepType.FWD_SEND, step.stage_index))
                new_schedule[peer].append(
                    MetaStep(step.micro_index, MetaStepType.FWD_RECV, step.stage_index + 1))
        elif step.type == MetaStepType.BWD:
            peer = _bwd_peer(step.stage_index)
            if peer is not None:
                new_schedule[rank].append(
                    MetaStep(step.micro_index, MetaStepType.BWD_SEND, step.stage_index))
                new_schedule[peer].append(
                    MetaStep(step.micro_index, MetaStepType.BWD_RECV, step.stage_index - 1))
        elif step.type in (MetaStepType.OVERLAP_F_B, MetaStepType.OVERLAP_B_F) and step.sub_steps:
            for sub in step.sub_steps:
                _insert_comms_for_step(sub, rank, new_schedule)

    # Expand OVERLAP steps into 2 virtual slots before the column scan so
    # the RECV triggered by an overlap's second sub-step lands one slot
    # later on the receiver — matching the fact that the sender can only
    # finish emitting the second sub-step after the first completes.
    expanded = _expand_overlap_slots(aligned, real_stage_num)
    return _column_scan_insert_comms(expanded, real_stage_num, _insert_comms_for_step)


class ScheduleGPipe(PipelineScheduleRuntime):
    """
    The Gpipe schedule.
    It first executes all forward micro batches and then execute all backward micro batches.
    """
    def __init__(self,
                 stages,
                 micro_batch_num,
                 args_batch_dim=None,
                 kwargs_batch_dim=None,
                 output_concat_dim=None,
                 swap=False):
        super().__init__(stages,
                         micro_batch_num,
                         args_batch_dim=args_batch_dim,
                         kwargs_batch_dim=kwargs_batch_dim,
                         output_concat_dim=output_concat_dim,
                         swap=swap)
        self.build_exec_order()

    def _build_stage_to_rank_index(self) -> None:
        self._stage_to_rank_index = generate_stage_to_rank_mapping(
            self.real_stage_num, self._stage_num, style='loop'
        )

    def construct_exec_order(self):
        """construct_exec_order of Gpipe."""
        for stage_index in range(self.real_stage_num):
            order_list = []
            for mb_index in range(self.micro_batch_num):
                if stage_index != 0:
                    order_list.append(MetaStep(mb_index, MetaStepType.FWD_RECV, stage_index))
                order_list.append(MetaStep(mb_index, MetaStepType.FWD, stage_index))
                if stage_index != self.real_stage_num - 1:
                    order_list.append(MetaStep(mb_index, MetaStepType.FWD_SEND, stage_index))
            for mb_index in range(self.micro_batch_num):
                if stage_index != self.real_stage_num - 1:
                    order_list.append(MetaStep(mb_index, MetaStepType.BWD_RECV, stage_index))
                order_list.append(MetaStep(mb_index, MetaStepType.BWD, stage_index))
                if stage_index != 0:
                    order_list.append(MetaStep(mb_index, MetaStepType.BWD_SEND, stage_index))
            self.exec_order[stage_index] = order_list


class Schedule1F1B(PipelineScheduleRuntime):
    """
    The 1F1B schedule.
    It will perform one forward and one backward on the micro batches in steady state.
    """
    def __init__(self,
                 stages,
                 micro_batch_num,
                 args_batch_dim=None,
                 kwargs_batch_dim=None,
                 output_concat_dim=None,
                 swap=False):
        super().__init__(stages,
                         micro_batch_num,
                         args_batch_dim=args_batch_dim,
                         kwargs_batch_dim=kwargs_batch_dim,
                         output_concat_dim=output_concat_dim,
                         swap=swap)
        self.build_exec_order()

    def _build_stage_to_rank_index(self) -> None:
        self._stage_to_rank_index = generate_stage_to_rank_mapping(
            self.real_stage_num, self._stage_num, style='loop'
        )

    def construct_exec_order(self):
        """construct_exec_order of 1F1B."""
        for stage_index in range(self.real_stage_num):
            order_list = []
            fwd_index = 0
            bwd_index = 0
            # warmup phase
            warmup_micro_batches = min(self.real_stage_num - stage_index, self.micro_batch_num)
            for _ in range(warmup_micro_batches):
                if stage_index != 0:
                    order_list.append(MetaStep(fwd_index, MetaStepType.FWD_RECV, stage_index))
                if stage_index % 2 == 0:
                    order_list.append(MetaStep(fwd_index, MetaStepType.FWD, stage_index))
                    if fwd_index != warmup_micro_batches - 1:
                        order_list.append(MetaStep(fwd_index, MetaStepType.FWD_SEND, stage_index))
                else:
                    if fwd_index > 0:
                        order_list.append(MetaStep(fwd_index - 1, MetaStepType.FWD_SEND, stage_index))
                    order_list.append(MetaStep(fwd_index, MetaStepType.FWD, stage_index))
                fwd_index += 1

            # if warmup phase cannot filled up, then we need to execute fwd send in advance
            if self.real_stage_num - stage_index > self.micro_batch_num:
                order_list.append(MetaStep(fwd_index - 1, MetaStepType.FWD_SEND, stage_index))
                fwd_index += 1
            # steady phase
            steady_micro_batches = self.micro_batch_num - warmup_micro_batches
            for _ in range(steady_micro_batches):
                if stage_index != self.real_stage_num - 1:
                    order_list.append(MetaStep(bwd_index, MetaStepType.BWD_RECV, stage_index))
                    order_list.append(MetaStep(fwd_index - 1, MetaStepType.FWD_SEND, stage_index))
                order_list.append(MetaStep(bwd_index, MetaStepType.BWD, stage_index))

                if stage_index != 0:
                    order_list.append(MetaStep(bwd_index, MetaStepType.BWD_SEND, stage_index))
                    order_list.append(MetaStep(fwd_index, MetaStepType.FWD_RECV, stage_index))
                order_list.append(MetaStep(fwd_index, MetaStepType.FWD, stage_index))
                fwd_index += 1
                bwd_index += 1

            # cooldown phase
            cooldown_micro_batches = warmup_micro_batches
            for _ in range(cooldown_micro_batches):
                if stage_index != self.real_stage_num - 1:
                    order_list.append(MetaStep(bwd_index, MetaStepType.BWD_RECV, stage_index))
                    if bwd_index == self.micro_batch_num - warmup_micro_batches and fwd_index <= self.micro_batch_num:
                        order_list.append(MetaStep(fwd_index - 1, MetaStepType.FWD_SEND, stage_index))
                order_list.append(MetaStep(bwd_index, MetaStepType.BWD, stage_index))

                if stage_index != 0:
                    order_list.append(MetaStep(bwd_index, MetaStepType.BWD_SEND, stage_index))
                bwd_index += 1
            self.exec_order[stage_index] = order_list


class ScheduleInterleaved1F1B(PipelineScheduleRuntime):
    """The Interleaved 1F1B schedule.

    Supports multiple stages per rank.  In steady state, performs one
    forward followed by one backward on each micro-batch.  Handles the
    cases where ``micro_batch_num`` is less than, equal to, or greater
    than the stage count, including non-evenly-divisible micro counts.

    Two orthogonal overlap modes can be enabled via constructor flags:

    * ``overlap_p2p=True``: defer P2P recv ``handle.wait()`` until the
      consuming FWD/BWD step (or the OVERLAP_B_F callback when
      ``overlap_b_f=True``), letting recv overlap with prior compute.
    * ``overlap_b_f=True``: in the 1F1B steady state, pair consecutive
      ``(B_i, F_{i+1})`` steps into ``OVERLAP_B_F`` composite steps so
      a registered callback can drive comm/compute overlap (typically
      via :class:`CommComputeOverlap` for MoE EP A2A).  Users register
      the callback through :meth:`register_custom_function`.
    * ``enable_dxdw_split=True`` (requires ``overlap_b_f``): each
      steady-state pair becomes ``(BWD_INPUT_i, F_{i+1})`` and the
      ``BWD_WEIGHT_i`` runs as its own step after the pair's P2P gap
      (see :func:`split_overlap_dxdw`), so the input-grad send is
      issued once dx and the paired forward finish instead of waiting
      out the full backward.

    The two overlap flags are independent and can be combined.

    Example:
        >>> # Plain interleaved 1F1B
        >>> sched = ScheduleInterleaved1F1B(stages, 8)
        >>> # With B/F overlap (dual-pipe-style comm/compute overlap)
        >>> sched = ScheduleInterleaved1F1B(stages, 8, overlap_b_f=True)
        >>> sched.register_custom_function(MetaStepType.OVERLAP_B_F, callback)
    """
    def __init__(self,
                 stages,
                 micro_batch_num,
                 args_batch_dim=None,
                 kwargs_batch_dim=None,
                 output_concat_dim=None,
                 overlap_p2p=False,
                 overlap_b_f=False,
                 swap=False,
                 enable_dxdw_split=False,
                 p2p_transport="auto"):
        super().__init__(stages,
                         micro_batch_num,
                         args_batch_dim=args_batch_dim,
                         kwargs_batch_dim=kwargs_batch_dim,
                         output_concat_dim=output_concat_dim,
                         overlap_p2p=overlap_p2p,
                         swap=swap,
                         p2p_transport=p2p_transport)
        # _overlap_b_f selects between plain F/B emission and OVERLAP_B_F
        # pairing in the 1F1B steady-state phase.  Must be set before
        # ``construct_stage_exec_order`` is called below.
        self._overlap_b_f = overlap_b_f
        # dx/dw split: ``construct_exec_order`` rewrites each steady-state
        # OVERLAP_B_F pair to ``(BWD_INPUT, FWD)`` and re-emits the matching
        # BWD_WEIGHT after the pair's P2P gap (``split_overlap_dxdw``), so the
        # input-grad send leaves at ``max(dx, fwd)`` and dw overlaps with the
        # in-flight P2P instead of delaying it.
        self._enable_dxdw_split = enable_dxdw_split
        if enable_dxdw_split and not overlap_b_f:
            raise ValueError(
                "enable_dxdw_split=True requires overlap_b_f=True; the split "
                "is only applied to BWD sub-steps inside OVERLAP_B_F composite steps."
            )
        if enable_dxdw_split and swap:
            raise ValueError(
                "enable_dxdw_split=True is incompatible with swap=True: "
                "pipeline-swap injection anchors on unified BWD steps and "
                "does not recognize the split BWD_INPUT/BWD_WEIGHT steps."
            )

        self._init_round_layout()
        self.build_exec_order()

    def _init_round_layout(self):
        """Compute per-round micro-batch counts used by stage-order emission.

        Populates ``n_rounds``, ``n_microbatch_per_round`` and its prefix-sum
        ``n_microbatch_per_round_accu`` from ``micro_batch_num``,
        ``real_stage_num`` and ``n_local_stages``.  Factored out of
        ``__init__`` so the pure schedule-construction path (used by offline
        unit tests) can be exercised without instantiating stages.
        """
        self.n_rounds = max(1, self.micro_batch_num // self.real_stage_num)
        if self.micro_batch_num < self.real_stage_num:
            base = self.micro_batch_num - self.real_stage_num
            remainder = 0
        else:
            n_extra_microbatch = self.micro_batch_num % self.real_stage_num
            base = n_extra_microbatch // self.n_rounds
            remainder = n_extra_microbatch % self.n_rounds
        self.n_microbatch_per_round = \
            [self.real_stage_num + base + 1 if i < remainder else
             self.real_stage_num + base for i in range(self.n_rounds)]
        self.n_microbatch_per_round_accu = \
            [x * self.n_local_stages for x in itertools.accumulate(self.n_microbatch_per_round)]
        self.n_microbatch_per_round_accu.insert(0, 0)

    def construct_exec_order(self):
        for stage_index in range(self.real_stage_num):
            self.exec_order[stage_index] = self.construct_stage_exec_order(stage_index)
        self.exec_order = add_send_recv(self.exec_order, self._stage_num, self.real_stage_num, style='loop')
        if self.enable_dxdw_split:
            self.exec_order = split_overlap_dxdw(self.exec_order)

    def _build_stage_to_rank_index(self) -> None:
        self._stage_to_rank_index = generate_stage_to_rank_mapping(
            self.real_stage_num, self._stage_num, style='loop'
        )

    def warmup_ops(self, stage_index):
        """warmup phase."""
        warmup_ops_last_stage = (self.n_local_stages - 1) * self.n_microbatch_per_round[0]
        warmup_ops = warmup_ops_last_stage + 2 * (self.real_stage_num - 1 - stage_index)
        return min(warmup_ops, self.micro_batch_num * self.n_local_stages)

    def forward_stage_index(self, op_index, stage_index):
        """obtain forward stage_index based on op_index."""
        accu_index = bisect.bisect_right(self.n_microbatch_per_round_accu, op_index) - 1
        local_index = (op_index - self.n_microbatch_per_round_accu[accu_index]) // \
                      self.n_microbatch_per_round[accu_index]
        return (local_index * self.real_stage_num) + stage_index

    def backward_stage_index(self, op_index, stage_index):
        """obtain backward stage_index based on op_index."""
        accu_index = bisect.bisect_right(self.n_microbatch_per_round_accu, op_index) - 1
        local_index = (op_index - self.n_microbatch_per_round_accu[accu_index]) // \
                      self.n_microbatch_per_round[accu_index]
        local_index = self.n_local_stages - 1 - local_index
        return (local_index * self.real_stage_num) + stage_index

    def _short_micro(self) -> bool:
        """True when ``micro_batch_num < real_stage_num`` (extra-bubble regime)."""
        return self.micro_batch_num < self.real_stage_num

    def _trailing_bubble(self) -> int:
        """Bubble count appended after a BWD with ``micro == micro_batch_num - 1``
        in the short-micro regime.
        """
        return self.real_stage_num - self.micro_batch_num

    def _emit_warmup_ops(self, stage_index, warmup_ops, fwd_stage_micro_index):
        """Emit pure-FWD warmup ops with optional short-micro bubble padding."""
        ops = []
        short = self._short_micro()
        last_micro = self.micro_batch_num - 1
        last_stage = self.real_stage_num - 1
        bubble = self._trailing_bubble()
        for op_idx in range(warmup_ops):
            fwd_stage_idx = self.forward_stage_index(op_idx, stage_index)
            fwd_micro_idx = fwd_stage_micro_index[fwd_stage_idx]
            ops.append(MetaStep(fwd_micro_idx, MetaStepType.FWD, fwd_stage_idx))
            need_pad = (
                short
                and fwd_micro_idx == last_micro
                and (op_idx != warmup_ops - 1 or stage_index == last_stage)
            )
            if need_pad:
                ops.extend([None] * bubble)
            fwd_stage_micro_index[fwd_stage_idx] += 1
        return ops

    def _emit_cooldown_ops(self, stage_index, warmup_ops, fwd_bwd_ops, total_ops,
                           bwd_stage_micro_index):
        """Emit pure-BWD cooldown ops (each preceded by a bubble) with
        optional short-micro trailing padding.
        """
        ops = []
        short = self._short_micro()
        last_micro = self.micro_batch_num - 1
        # Double the bubble at each chunk's last-micro BWD: one ``bubble`` covers
        # the missing ``rs - micro`` micros, the second offsets the next chunk
        # by 2 slots so the wrap-around grad (rank 0 stage ``rs`` -> rank
        # last_stage stage ``rs - 1``) lands AFTER its producer in column-scan
        # time.  Matches the +2 cooldown-rhythm offset that non-short Interleaved
        # 1F1B naturally has from extra 1F1B ops on rank last_stage.
        bubble = 2 * self._trailing_bubble()
        for op_idx in range(warmup_ops + fwd_bwd_ops, total_ops):
            ops.append(None)
            bwd_stage_idx = self.backward_stage_index(op_idx - warmup_ops, stage_index)
            bwd_micro_idx = bwd_stage_micro_index[bwd_stage_idx]
            ops.append(MetaStep(bwd_micro_idx, MetaStepType.BWD, bwd_stage_idx))
            if short and bwd_micro_idx == last_micro:
                ops.extend([None] * bubble)
            bwd_stage_micro_index[bwd_stage_idx] += 1
        return ops

    def _emit_1f1b_ops(self, stage_index, warmup_ops, fwd_bwd_ops,
                       fwd_stage_micro_index, bwd_stage_micro_index):
        """Emit interleaved (FWD, BWD) pairs for the 1F1B steady-state phase."""
        ops = []
        short = self._short_micro()
        last_micro = self.micro_batch_num - 1
        last_stage = self.real_stage_num - 1
        # Double the bubble at the 1F1B->cooldown chunk boundary on rank
        # last_stage; see :meth:`_emit_cooldown_ops` for the alignment rationale.
        bubble = 2 * self._trailing_bubble()
        for op_idx in range(warmup_ops, warmup_ops + fwd_bwd_ops):
            fwd_stage_idx = self.forward_stage_index(op_idx, stage_index)
            fwd_micro_idx = fwd_stage_micro_index[fwd_stage_idx]
            ops.append(MetaStep(fwd_micro_idx, MetaStepType.FWD, fwd_stage_idx))
            fwd_stage_micro_index[fwd_stage_idx] += 1
            bwd_stage_idx = self.backward_stage_index(op_idx - warmup_ops, stage_index)
            bwd_micro_idx = bwd_stage_micro_index[bwd_stage_idx]
            ops.append(MetaStep(bwd_micro_idx, MetaStepType.BWD, bwd_stage_idx))
            need_pad = (
                short
                and bwd_micro_idx == last_micro
                and stage_index == last_stage
            )
            if need_pad:
                ops.extend([None] * bubble)
            bwd_stage_micro_index[bwd_stage_idx] += 1
        return ops

    @staticmethod
    def _collect_fwd_bwd_steps(emit_fwd, emit_bwd, fwd_bwd_ops, warmup_ops):
        """Walk the 1F1B range collecting parallel ``fwd_steps`` / ``bwd_steps``.

        ``emit_fwd(op_idx)`` and ``emit_bwd(op_idx)`` build a single
        :class:`MetaStep` and advance their respective per-stage micro
        counters as a side effect.
        """
        fwd_steps = []
        bwd_steps = []
        for op_idx in range(warmup_ops, warmup_ops + fwd_bwd_ops):
            fwd_steps.append(emit_fwd(op_idx))
            bwd_steps.append(emit_bwd(op_idx))
        return fwd_steps, bwd_steps

    @staticmethod
    def _pair_into_overlap_b_f(fwd_steps, bwd_steps):
        """Build ``F₁, [B_i, F_{i+1}], B_n`` ordering with OVERLAP_B_F pairs.

        ``sub_steps`` carry the ``(bwd, fwd)`` tuple — callbacks access
        them via ``step.sub_steps`` to recover per-direction stage /
        micro info.
        """
        ops = []
        if fwd_steps:
            ops.append(fwd_steps[0])  # F₁ runs alone
        for i in range(len(bwd_steps) - 1):
            ops.append(MetaStep(
                None, MetaStepType.OVERLAP_B_F, None,
                sub_steps=(bwd_steps[i], fwd_steps[i + 1]),
            ))
        if bwd_steps:
            ops.append(bwd_steps[-1])  # B_n runs alone
        return ops

    def _emit_1f1b_overlap_ops(self, stage_index, warmup_ops, fwd_bwd_ops,
                               fwd_stage_micro_index, bwd_stage_micro_index):
        """Emit ``F₁, [B_i, F_{i+1}], B_n`` for the 1F1B phase under
        ``overlap_b_f=True``.  Each ``[B_i, F_{i+1}]`` becomes an
        ``OVERLAP_B_F`` composite step; a registered callback drives the
        actual concurrent execution.  Short-micro extra-bubble padding
        on the last rank is appended after ``B_n``.
        """
        def emit_fwd(op_idx):
            fwd_si = self.forward_stage_index(op_idx, stage_index)
            fwd_mi = fwd_stage_micro_index[fwd_si]
            fwd_stage_micro_index[fwd_si] += 1
            return MetaStep(fwd_mi, MetaStepType.FWD, fwd_si)

        def emit_bwd(op_idx):
            bwd_si = self.backward_stage_index(op_idx - warmup_ops, stage_index)
            bwd_mi = bwd_stage_micro_index[bwd_si]
            bwd_stage_micro_index[bwd_si] += 1
            return MetaStep(bwd_mi, MetaStepType.BWD, bwd_si)

        fwd_steps, bwd_steps = self._collect_fwd_bwd_steps(
            emit_fwd, emit_bwd, fwd_bwd_ops, warmup_ops,
        )
        ops = self._pair_into_overlap_b_f(fwd_steps, bwd_steps)

        last_stage = self.real_stage_num - 1
        if self._short_micro() and stage_index == last_stage and bwd_steps:
            if bwd_steps[-1].micro_index == self.micro_batch_num - 1:
                # Double the bubble at the 1F1B->cooldown chunk boundary;
                # see :meth:`_emit_cooldown_ops` for the alignment rationale.
                ops.extend([None] * (2 * self._trailing_bubble()))
        return ops

    def construct_stage_exec_order(self, stage_index):
        """Construct the execution order for ``stage_index``.

        Builds: warmup → bubbles → 1F1B steady state → cooldown.  The
        1F1B segment switches between :meth:`_emit_1f1b_ops` (plain) and
        :meth:`_emit_1f1b_overlap_ops` (OVERLAP_B_F pairing) based on
        the ``overlap_b_f`` constructor flag.
        """
        warmup_ops = self.warmup_ops(stage_index)
        fwd_bwd_ops = self.n_local_stages * self.micro_batch_num - warmup_ops
        total_ops = 2 * warmup_ops + fwd_bwd_ops
        order_list = [None for _ in range(stage_index)]
        fwd_stage_micro_index = defaultdict(int)
        bwd_stage_micro_index = defaultdict(int)
        order_list.extend(self._emit_warmup_ops(stage_index, warmup_ops, fwd_stage_micro_index))
        bubbles_before_1f1b = max(
            0,
            2 * (self.real_stage_num - stage_index - 1) - self.micro_batch_num,
        )
        order_list.extend([None] * bubbles_before_1f1b)
        order_list.extend([None] * (self.real_stage_num - 1 - stage_index))
        if self._overlap_b_f:
            order_list.extend(self._emit_1f1b_overlap_ops(
                stage_index, warmup_ops, fwd_bwd_ops,
                fwd_stage_micro_index, bwd_stage_micro_index,
            ))
        else:
            order_list.extend(self._emit_1f1b_ops(
                stage_index, warmup_ops, fwd_bwd_ops,
                fwd_stage_micro_index, bwd_stage_micro_index,
            ))
        order_list.extend(self._emit_cooldown_ops(
            stage_index, warmup_ops, fwd_bwd_ops, total_ops, bwd_stage_micro_index,
        ))
        return order_list


def detect_cycle_in_graph(ranks_map):
    """
    Detects a cycle in the directed graph constructed from ranks_map.

    Args:
        ranks_map: A dictionary where keys are rank names and values are lists of nodes.

    Returns:
        tuple: (cycle_path, cycle_ranks) where cycle_path is a list of nodes forming the cycle and cycle_ranks
               is a list of rank transitions corresponding to the cycle path.
    """
    graph = defaultdict(list)
    rank_edges = {}

    for rank, nodes in ranks_map.items():
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            graph[u].append(v)
            rank_edges[(u, v)] = rank

    visited = set()
    path = []
    node_indices = {}
    cycle_path = []
    cycle_ranks = []

    stack = []
    for node in list(graph.keys()):
        if node not in visited:
            stack.append((node, False))
            while stack:
                current_node, is_processed = stack.pop()

                if is_processed:
                    path.pop()
                    del node_indices[current_node]
                    continue

                if current_node in node_indices:
                    cycle_start = node_indices[current_node]
                    cycle_path = path[cycle_start:] + [current_node]
                    for i in range(cycle_start, len(path)):
                        u = path[i]
                        v = path[i + 1] if i + 1 < len(path) else current_node
                        cycle_ranks.append(f"{rank_edges[(u, v)]} {u} -> {v}")
                    return cycle_path, cycle_ranks

                if current_node in visited:
                    continue

                visited.add(current_node)
                node_indices[current_node] = len(path)
                path.append(current_node)

                stack.append((current_node, True))
                for neighbor in reversed(graph[current_node]):
                    stack.append((neighbor, False))

    return None, None


def output_cycle_results(cycle_path, cycle_ranks):
    """
    Helper function to output cycle detection results.

    Args:
        cycle_path (list): List of nodes forming a cycle, if any.
        cycle_ranks (list): List of ranks involved in the cycle.

    Returns:
        None: Outputs results to the console.
    """
    if cycle_path:
        logger.error("Cycle detected:")
        path_str = " -> ".join(str(node) for node in cycle_path)
        logger.error("%s -> %s", path_str, cycle_path[0])  # Close the cycle
        logger.error("Involving ranks:")
        for rank in cycle_ranks:
            logger.error(rank)
    else:
        logger.warning("Cycle Check succeeded. There is no cycle in the graph.")


def parse_and_validate(data: dict, all_rank: bool = True):
    """
    Parse and validate execution orders in a directed graph structure.

    This function checks the integrity and consistency of a given dataset, ensuring all required
    keys are present and correctly referenced. It also validates the structure of the input data
    and parses string values to extract meaningful components.

    Args:
        data (dict): A dictionary where keys are string identifiers and values are lists of strings.
                        Each value represents a dependency or reference to other keys.
        all_rank (bool): If True, checks that all elements referenced in the data are present as keys
                            in the dictionary. If False, only checks intersections.

    Returns:
        None: Log error messages to the console if validation fails, otherwise completes silently.

    Raises:
        ValueError: Raised indirectly if `parse_elements` encounters malformed input strings.
        TypeError: Raised indirectly if data contains unexpected types.
    """

    def parse_elements(value: str, max_groups: int = 2) -> set:
        """Extract unique elements inside the first one or two parentheses from a string."""

        groups = re.findall(r'\((\d+)\)', value)
        limited_groups = groups[:max_groups]  # Limit to the first `max_groups` matches

        return {item.strip() for item in limited_groups}

    if not isinstance(data, dict):
        logger.error("Input must be a dictionary with string keys and lists of strings as values.")
        return

    key_to_values = {key: set(values) for key, values in data.items() if
                     isinstance(values, list) and all(isinstance(v, str) for v in values)}

    for key, values in data.items():
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            logger.error("Values for key '%s' must be a list of strings.", key)
            continue

        for value in values:
            try:
                elements = parse_elements(value)
            except (ValueError, TypeError, AttributeError) as e:
                logger.error("Unable to parse elements from value '%s' in key '%s'. Error: %s", value, key, e)
                continue

            # Check for missing keys if all_rank is True
            if all_rank:
                missing_keys = elements - key_to_values.keys()
                if missing_keys:
                    logger.error("The following keys are missing for value '%s': %s", value, missing_keys)
                    continue

            # Check if the value is present in the referenced keys
            for element in elements & key_to_values.keys() if not all_rank else elements:
                if value not in key_to_values[element]:
                    logger.error("Key '%s' is missing the value '%s'.", element, value)


def generate_operations(order_list: dict[int, list[MetaStep]],
                        chunk_num: int,
                        com_type: str = 'loop') -> dict[str, list[str]]:
    """
    Generate formatted operations dictionary from pipeline execution order.

    Args:
        order_list (dict): Dictionary where keys are rank IDs and values are MetaStep execution sequences
        chunk_num (int): Number of chunks (virtual pipeline stages)
        com_type (str): Stage-to-rank mapping type ('loop' for cyclic, 'v' for V-shaped)

    Returns:
        Dictionary where keys are rank IDs (as strings) and values are lists of formatted operation strings
    """

    def stage_to_rank(stage_index, style, stage_num, real_stage_num):
        """Map stage index to rank"""
        if style == 'loop':
            return stage_index % real_stage_num
        if style == 'v':
            if stage_index < real_stage_num:
                return stage_index
            return stage_num - 1 - stage_index
        raise ValueError("Invalid style")

    def find_send_target(stage_idx, op_type):
        """Find target stage for SEND operation"""
        if op_type == MetaStepType.FWD_SEND:
            return forward_comm.get(stage_idx)
        return backward_comm.get(stage_idx)

    def find_recv_source(stage_idx, op_type):
        """Find source stage for RECV operation"""
        if op_type == MetaStepType.FWD_RECV:
            # Reverse lookup in forward_comm
            for src, dst in forward_comm.items():
                if dst == stage_idx:
                    return src
        else:
            # Reverse lookup in backward_comm
            for src, dst in backward_comm.items():
                if dst == stage_idx:
                    return src
        return None

    real_stage = len(order_list)
    total_stages = real_stage * chunk_num

    # Build communication rules
    forward_comm = {}
    backward_comm = {}

    for i in range(total_stages):
        if i + 1 < total_stages:
            forward_comm[i] = i + 1
        if i - 1 >= 0:
            backward_comm[i] = i - 1

    formatted_operations = defaultdict(list)

    for rank, steps in order_list.items():
        operation_counter = defaultdict(int)

        for step in steps:
            if step.type in [MetaStepType.FWD_SEND, MetaStepType.BWD_SEND]:
                target_stage = find_send_target(step.stage_index, step.type)
                if target_stage is not None:
                    target_rank = stage_to_rank(target_stage, com_type, total_stages, real_stage)
                    comm_pair = (rank, target_rank, step.micro_index)
                    operation_counter[comm_pair] += 1
                    count = operation_counter[comm_pair]
                    formatted_op = f"Send_Receive_({rank})->({target_rank})_micro{step.micro_index}_{count}th"
                    formatted_operations[str(rank)].append(formatted_op)

            elif step.type in [MetaStepType.FWD_RECV, MetaStepType.BWD_RECV]:
                source_stage = find_recv_source(step.stage_index, step.type)
                if source_stage is not None:
                    source_rank = stage_to_rank(source_stage, com_type, total_stages, real_stage)
                    comm_pair = (source_rank, rank, step.micro_index)
                    operation_counter[comm_pair] += 1
                    count = operation_counter[comm_pair]
                    formatted_op = f"Send_Receive_({source_rank})->({rank})_micro{step.micro_index}_{count}th"
                    formatted_operations[str(rank)].append(formatted_op)

    # Convert defaultdict to dict
    return dict(formatted_operations)


def validate_pipeline_execution(order_list: dict[int, list[MetaStep]],
                                chunk_num: int,
                                com_type: str = 'loop') -> dict[str, any]:
    """
    Comprehensive validation function for pipeline parallel execution order.

    This function validates the execution order of pipeline parallelism by:
    1. Checking SEND/RECV communication pair matching
    2. Detecting duplicate operations
    3. Detecting cycles in communication graphs
    4. Verifying computation-SEND matching

    Args:
        order_list: Dictionary where keys are rank IDs and values are MetaStep execution sequences
        chunk_num: Number of chunks (virtual pipeline stages)
        com_type: Stage-to-rank mapping type ('loop' for cyclic, 'v' for V-shaped)

    Returns:
        Dictionary containing validation results with the following keys:
        - validation: Communication pair validation results
        - cycle_detection: Cycle detection results
        - computation_send_matching: Computation-SEND matching validation results
        - has_errors: Boolean indicating if any errors were found
        - error_messages: List of all error messages found
        - formatted_operations: Generated formatted operations
    """

    # Generate operations
    formatted_operations = generate_operations(order_list, chunk_num, com_type)

    parse_and_validate(formatted_operations, True)

    # Detect cycles
    cycle_path, cycle_ranks = detect_cycle_in_graph(formatted_operations)

    # Output results
    output_cycle_results(cycle_path, cycle_ranks)

    result = {
        'formatted_operations': formatted_operations,
        'cycle_path': cycle_path,
        'cycle_ranks': cycle_ranks,
        'has_cycle': bool(cycle_path)
    }
    return result


_COMPUTE_META_STEP_TYPES = frozenset({
    MetaStepType.FWD,
    MetaStepType.BWD,
    MetaStepType.BWD_INPUT,
    MetaStepType.BWD_WEIGHT,
})


def _next_active_stage_indices(actions, start_index, max_active_stages, managed_stage_indices):
    """Find the next distinct managed stages that will execute compute work.

    Send/recv and previously injected FSDP control steps are skipped so that the
    lookahead window only counts real compute, otherwise communication-only
    actions would consume the budget and shrink the effective prefetch depth.
    """
    stage_indices = []
    seen = set()
    for action in actions[start_index:]:
        for leaf_step in iter_leaf_meta_steps(action):
            if leaf_step.type not in _COMPUTE_META_STEP_TYPES:
                continue
            if leaf_step.stage_index not in managed_stage_indices or leaf_step.stage_index in seen:
                continue
            seen.add(leaf_step.stage_index)
            stage_indices.append(leaf_step.stage_index)
            if len(stage_indices) == max_active_stages:
                return stage_indices
    return stage_indices


def add_fsdp_unshard_reshard(actions, managed_stage_indices, max_active_stages=3):
    """Insert FSDP unshard/reshard actions for locally managed stages."""
    if not managed_stage_indices:
        return actions

    fsdp_actions = []
    active_stages = []
    for index, action in enumerate(actions):
        next_stage_indices = _next_active_stage_indices(
            actions, index, max_active_stages, managed_stage_indices
        )
        evicted_stages = [stage_index for stage_index in active_stages if stage_index not in next_stage_indices]
        fetched_stages = [stage_index for stage_index in next_stage_indices if stage_index not in active_stages]
        for stage_index in evicted_stages:
            fsdp_actions.append(MetaStep(None, MetaStepType.FSDP_RESHARD, stage_index))
            active_stages.remove(stage_index)
        for stage_index in fetched_stages:
            fsdp_actions.append(MetaStep(None, MetaStepType.FSDP_UNSHARD, stage_index))
            active_stages.append(stage_index)
        fsdp_actions.append(action)

    while active_stages:
        fsdp_actions.append(MetaStep(None, MetaStepType.FSDP_RESHARD, active_stages.pop(0)))
    return fsdp_actions


def add_fsdp_reduce_grad(actions, managed_stage_indices, micro_batch_num):
    """Launch reduction after each final backward and drain after all local stages."""
    if not managed_stage_indices:
        return actions

    fsdp_actions = []
    last_reduced_stage_index = None
    for action in actions:
        fsdp_actions.append(action)
        reduced_stage_indices = []
        for leaf_step in iter_leaf_meta_steps(action):
            if leaf_step.stage_index not in managed_stage_indices:
                continue
            if leaf_step.type not in (MetaStepType.BWD, MetaStepType.BWD_WEIGHT):
                continue
            if leaf_step.micro_index != micro_batch_num - 1:
                continue
            if leaf_step.stage_index not in reduced_stage_indices:
                reduced_stage_indices.append(leaf_step.stage_index)
        for stage_index in reduced_stage_indices:
            fsdp_actions.append(
                MetaStep(None, MetaStepType.FSDP_REDUCE_GRAD, stage_index)
            )
            last_reduced_stage_index = stage_index
    if last_reduced_stage_index is not None:
        fsdp_actions.append(
            MetaStep(
                None,
                MetaStepType.FSDP_WAIT_REDUCE_GRAD,
                last_reduced_stage_index,
            )
        )
    return fsdp_actions
