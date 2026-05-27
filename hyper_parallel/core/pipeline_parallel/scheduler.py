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
    OVERLAP_F_B = auto()
    OVERLAP_B_F = auto()
    FSDP_UNSHARD = auto()
    FSDP_RESHARD = auto()
    FSDP_REDUCE_GRAD = auto()
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
    """
    def __init__(self, micro_index, meta_type, stage_index, sub_steps=None):
        self._type = meta_type
        self._micro_index = micro_index
        self._stage_index = stage_index
        self._sub_steps = sub_steps

    @property
    def micro_index(self):
        return self._micro_index

    @property
    def stage_index(self):
        return self._stage_index

    @property
    def type(self):
        return self._type

    @property
    def sub_steps(self):
        """Sub-steps for composite types: ``(fwd, bwd)`` for OVERLAP_F_B,
        ``(bwd, fwd)`` for OVERLAP_B_F, or ``None``."""
        return self._sub_steps

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
    """Yield leaf MetaSteps, recursively expanding OVERLAP_F_B containers."""
    if step is None:
        return
    if step.type == MetaStepType.OVERLAP_F_B:
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
    """Run the stage's FSDP post-backward gradient reduction."""
    stage.execute_reduce_grad()


# FSDP control MetaStep -> handler(stage).  Membership also marks which
# MetaStepTypes are FSDP control steps, so the runtime loop dispatches with a
# single table lookup instead of re-switching on the step type.
_FSDP_STEP_HANDLERS = {
    MetaStepType.FSDP_UNSHARD: _exec_fsdp_unshard,
    MetaStepType.FSDP_RESHARD: _exec_fsdp_reshard,
    MetaStepType.FSDP_REDUCE_GRAD: _exec_fsdp_reduce_grad,
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
        args_batch_dim (list, optional): Specify the batch dim of the args.
            Default ``None``.
        kwargs_batch_dim (dict, optional): Specify the batch dim of the kwargs.
            Default ``None``.
        swap (bool, optional): Whether to inject pipeline activation swap
            control steps. Supported by ``ScheduleGPipe``, ``Schedule1F1B``,
            and ``ScheduleInterleaved1F1B``. Default ``False``.
    """
    def __init__(self,
                 stages,
                 micro_batch_num,
                 args_batch_dim=None,
                 kwargs_batch_dim=None,
                 output_concat_dim=None,
                 overlap_p2p=False,
                 swap=False):
        self.stages = self._check_stages(stages)
        self.micro_batch_num = micro_batch_num
        self._args_batch_dim = args_batch_dim
        self._kwargs_batch_dim = kwargs_batch_dim
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
        rank_actions = add_fsdp_unshard_reshard(self.exec_order[current_rank], managed_stage_indices)
        self.exec_order[current_rank] = add_fsdp_reduce_grad(
            rank_actions,
            managed_stage_indices,
            self.micro_batch_num,
        )

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
        """Build the execution order and inject optional PP-swap/FSDP actions."""
        self.construct_exec_order()
        self._inject_local_pp_swap_actions()
        self._inject_local_fsdp_actions()

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

    def run(self, *args, **kwargs):
        """schedule run."""
        split_args, split_kwargs = self.split_microbatches(args, kwargs)
        losses = []
        self.run_microbatches(split_args, split_kwargs, losses)
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

    # --- P2P step primitives ------------------------------------------------
    # One method per cross-rank comm action, used both by the runtime loop
    # (``_exec_step``) and by OVERLAP callbacks (via ``ctx.schedule``).  With
    # ``overlap_p2p=True`` comm is decoupled from its compute: a recv caches its
    # handles for the consuming step to ``wait_*`` later, and a send defers its
    # handles to the end-of-iteration drain.  With ``overlap_p2p=False`` every
    # op waits inline.

    def recv_fwd(self, stage: "hyper_parallel.PipelineStage", micro_index: int) -> None:
        """Post the FWD recv for ``micro_index``; cache it (overlap_p2p) or wait now."""
        handles = stage.exec_fwd_recv_ops(micro_index)
        if self._overlap_p2p:
            self.fwd_handle_cache[(stage.stage_index, micro_index)] = handles
        else:
            self._wait_p2p(handles)

    def recv_bwd(self, stage: "hyper_parallel.PipelineStage", micro_index: int) -> None:
        """Post the BWD recv for ``micro_index``; cache it (overlap_p2p) or wait now."""
        handles = stage.exec_bwd_recv_ops(micro_index)
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

    def send_fwd(self, stage: "hyper_parallel.PipelineStage", micro_index: int) -> list:
        """Send this stage's forward output for ``micro_index`` to the next stage."""
        handles = stage.exec_fwd_send_ops(micro_index) or []
        if self._overlap_p2p:
            # Append the whole handle group: run_microbatches drains _send_handles
            # group by group, so a bare handle would be wrongly iterated as a list.
            self._send_handles.append(handles)
        else:
            self._wait_p2p(handles)
        return handles

    def send_bwd(self, stage: "hyper_parallel.PipelineStage", micro_index: int) -> list:
        """Send this stage's input-gradient for ``micro_index`` to the previous stage.

        Driven by the scheduler's ``BWD_SEND`` step. It pops the input grad that
        the backward (unified ``backward_one_chunk`` or, under
        ``enable_dxdw_split=True``, ``backward_input_one_chunk``) wrote to the
        stage's ``bwd_cache``. Calling it manually in addition to the scheduled
        ``BWD_SEND`` would double-send the gradient.
        """
        handles = stage.exec_bwd_send_ops(micro_index) or []
        if self._overlap_p2p:
            self._send_handles.append(handles)
        else:
            self._wait_p2p(handles)
        return handles

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
        self._send_handles = []
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

            # Check for registered custom function
            custom_fn = self._custom_fn_map.get(cur_step.type)
            if custom_fn is not None:
                if ctx is None:
                    ctx = PipelineContext(self, arg_mbs, kwarg_mbs, losses)
                custom_fn(cur_step, ctx)
                continue

            # Default for composite OVERLAP steps: run sub_steps sequentially.
            # P2P send/recv around these steps are already laid out in two
            # virtual slots by ``add_send_recv``, so sequential execution is
            # semantically equivalent to non-overlapped 1F1B.
            if (cur_step.type in (MetaStepType.OVERLAP_F_B, MetaStepType.OVERLAP_B_F)
                    and cur_step.sub_steps):
                for sub in cur_step.sub_steps:
                    self._exec_step(sub, arg_mbs, kwarg_mbs, losses)
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

    The two flags are independent and can be combined.

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
                 enable_dxdw_split=False):
        super().__init__(stages,
                         micro_batch_num,
                         args_batch_dim=args_batch_dim,
                         kwargs_batch_dim=kwargs_batch_dim,
                         output_concat_dim=output_concat_dim,
                         overlap_p2p=overlap_p2p,
                         swap=swap)
        # _overlap_b_f selects between plain F/B emission and OVERLAP_B_F
        # pairing in the 1F1B steady-state phase.  Must be set before
        # ``construct_stage_exec_order`` is called below.
        self._overlap_b_f = overlap_b_f
        # enable dx_dw split in overlap phase.
        self._enable_dxdw_split = enable_dxdw_split
        if enable_dxdw_split and not overlap_b_f:
            raise ValueError(
                "enable_dxdw_split=True requires overlap_b_f=True; the split "
                "is only applied to BWD sub-steps inside OVERLAP_B_F composite steps."
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
    """Insert FSDP reduce-grad actions after the last backward-like action of each stage."""
    if not managed_stage_indices:
        return actions

    fsdp_actions = []
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
            fsdp_actions.append(MetaStep(None, MetaStepType.FSDP_REDUCE_GRAD, stage_index))
    return fsdp_actions
