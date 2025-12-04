# Copyright 2025 Huawei Technologies Co., Ltd
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
from abc import ABC
from enum import Enum, auto
from collections import defaultdict
import itertools
import bisect
from mindspore.parallel.mpmd.pipeline_parallel import PipelineStage
from mindspore.parallel.mpmd.pipeline_parallel._utils import _MicroBatch


class MetaStepType(Enum):
    """Specify the enumeration type for MetaStep."""
    FWD = auto()
    BWD = auto()
    FWD_RECV = auto()
    FWD_SEND = auto()
    BWD_RECV = auto()
    BWD_SEND = auto()


class MetaStep:
    """
    Meta step of PipelineSchedule.
    An execution list composed of MetaStep can be constructed
    and fed into the PipelineSchedule for execution.

    Args:
        micro_index (int): The index of micro-batch.
        type (MetaStepType): Specify the type of current step.
        stage_index(int): Specify the stage index of current step.
    """
    def __init__(self, micro_index, meta_type, stage_index):
        self._type = meta_type
        self._micro_index = micro_index
        self._stage_index = stage_index

    @property
    def micro_index(self):
        return self._micro_index

    @property
    def stage_index(self):
        return self._stage_index

    @property
    def type(self):
        return self._type

    def __eq__(self, value):
        if not isinstance(value, MetaStep):
            return NotImplemented
        return self.type == value.type and \
            self.micro_index == value.micro_index and \
            self.stage_index == value.stage_index

    def __ne__(self, value):
        if not isinstance(value, MetaStep):
            return NotImplemented
        return self.type != value.type or \
            self.micro_index != value.micro_index or \
            self.stage_index != value.stage_index

    def __hash__(self):
        return hash((self.type, self.micro_index, self.stage_index))

    def __str__(self):
        return f"MetaStep(type={self.type}, micro_index={self.micro_index}, stage_index={self.stage_index})"

    def __repr__(self):
        return f"MetaStep(type={self.type}, micro_index={self.micro_index}, stage_index={self.stage_index})"

    @staticmethod
    def from_str(step_str):
        pass


class PipelineScheduleRuntime(ABC):
    """
    Base class for pipeline schedule.
    Implements the `split_microbatches` and `run_microbatches` method.
    Derived classes should implement `run_microbatches` method and `run` method.

    Args:
        stages (list[PipelineStage], PipelineStage):  PipelineStage used to run_microbatches.
        micro_batch_num (int): The number of micro-batch.
        args_batch_dim (list, optional): Specify the batch dim of the args.
            Default ``None``.
        kwargs_batch_dim(dict, optional): Specify the batch dim of the kwargs.
            Default ``None``.
    """
    def __init__(self,
                 stages,
                 micro_batch_num,
                 args_batch_dim=None,
                 kwargs_batch_dim=None,
                 output_concat_dim=None,
                 overlap_p2p=False):
        self.stages = self._check_stages(stages)
        self.micro_batch_num = micro_batch_num
        self._args_batch_dim = args_batch_dim
        self._kwargs_batch_dim = kwargs_batch_dim
        self._output_concat_dim = output_concat_dim
        self.split_micro_batch = _MicroBatch(self.micro_batch_num, self._args_batch_dim, self._kwargs_batch_dim)
        self.n_local_stages = len(self.stages)
        self._stage_dict = self.convert_stages_dict()
        self.real_stage_num = self.stages[0].stage_num // self.n_local_stages
        self._stage_num = self.stages[0].stage_num
        self._overlap_p2p = overlap_p2p
        self.exec_order = {}
        self._init_stages()
        self.fwd_handle_cache = {}
        self.bwd_handle_cache = {}

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
        return [[]] * self.micro_batch_num, [{}] * self.micro_batch_num

    def _check_stages(self, stages):
        """check stages type."""
        if isinstance(stages, PipelineStage):
            return [stages]
        if isinstance(stages, (list, tuple)):
            for stage in stages:
                if not isinstance(stage, PipelineStage):
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
        grads = []
        self.run_microbatches(split_args, split_kwargs, losses, grads)
        return losses, tuple(grads)

    def sync_shared_parameters_grad(self):
        """sync_shared_parameters_grad."""
        for stage in self.stages:
            stage.sync_shared_parameters_grad()

    def update_losses(self, stage, loss, losses):
        """update_losses."""
        if stage.is_last_stage:
            losses.append(loss)

    def _wait_p2p(self, handles):
        for handle in handles:
            if handle is not None:
                handle.wait()

    def run_microbatches(self, arg_mbs, kwarg_mbs, losses, grads):
        """run_microbatches."""
        real_stage_index = self.stages[0].stage_index % self.real_stage_num
        send_handle = []
        for cur_step in self.exec_order[real_stage_index]:
            if cur_step is None:
                continue
            stage = self._stage_dict[cur_step.stage_index]
            stage_index = cur_step.stage_index
            micro_index = cur_step.micro_index
            if cur_step.type == MetaStepType.FWD_RECV:
                comm_handle = stage.exec_fwd_recv_ops(micro_index)
                if not self._overlap_p2p:
                    self._wait_p2p(comm_handle)
                else:
                    key = (stage_index, micro_index)
                    self.fwd_handle_cache[key] = comm_handle
            if cur_step.type == MetaStepType.FWD:
                key = (stage_index, micro_index)
                if self._overlap_p2p and key in self.fwd_handle_cache:
                    comm_handle = self.fwd_handle_cache.pop(key)
                    self._wait_p2p(comm_handle)
                out = stage.forward_one_chunk(micro_index, arg_mbs[micro_index], kwarg_mbs[micro_index])
                self.update_losses(stage, out, losses)
            if cur_step.type == MetaStepType.FWD_SEND:
                comm_handle = stage.exec_fwd_send_ops(micro_index)
                if not self._overlap_p2p:
                    self._wait_p2p(comm_handle)
                else:
                    send_handle.append(comm_handle)
            if cur_step.type == MetaStepType.BWD_RECV:
                comm_handle = stage.exec_bwd_recv_ops(micro_index)
                if not self._overlap_p2p:
                    self._wait_p2p(comm_handle)
                else:
                    key = (stage_index, micro_index)
                    self.bwd_handle_cache[key] = comm_handle
            if cur_step.type == MetaStepType.BWD:
                key = (stage_index, micro_index)
                if self._overlap_p2p and key in self.bwd_handle_cache:
                    comm_handle = self.bwd_handle_cache.pop(key)
                    self._wait_p2p(comm_handle)
                if micro_index == self.micro_batch_num - 1:
                    grad_out = stage.backward_one_chunk(micro_index, True)
                    grads += grad_out
                else:
                    _ = stage.backward_one_chunk(micro_index)
            if cur_step.type == MetaStepType.BWD_SEND:
                comm_handle = stage.exec_bwd_send_ops(micro_index)
                if not self._overlap_p2p:
                    self._wait_p2p(comm_handle)
                else:
                    send_handle.append(comm_handle)
        self.sync_shared_parameters_grad()
        while send_handle:
            self._wait_p2p(send_handle.pop())


def add_send_recv(scheduler, stage_num, real_stage_num, style='loop'):
    """
    Create schedule for each rank and automatically add communication operations

    Args:
        scheduler: Compute schedule table with None
        stage_num: Total number of pipeline stages
        real_stage_num: Number of actual physical stages/ranks
        style: Communication style ('loop' or 'v')

    Returns:
        Complete schedule table for each rank (including communication operations)
    """

    def _need_com(action, style, stage_num):
        """Determine if communication is needed"""
        if action.type == MetaStepType.FWD:
            if action.stage_index == stage_num - 1:
                return False  # Last stage doesn't need forward communication
            next_stage_rank = stage_to_rank(action.stage_index + 1, style, stage_num, real_stage_num)
            current_rank = stage_to_rank(action.stage_index, style, stage_num, real_stage_num)
            return next_stage_rank != current_rank
        if action.type == MetaStepType.BWD:
            if action.stage_index == 0:
                return False  # First stage doesn't need backward communication
            prev_stage_rank = stage_to_rank(action.stage_index - 1, style, stage_num, real_stage_num)
            current_rank = stage_to_rank(action.stage_index, style, stage_num, real_stage_num)
            return prev_stage_rank != current_rank
        return False

    def stage_to_rank(stage_index, style, stage_num, real_stage_num):
        """Map stage index to rank"""
        if style == 'loop':
            return stage_index % real_stage_num
        if style == 'v':
            if stage_index < real_stage_num:
                return stage_index
            return stage_num - 1 - stage_index
        raise ValueError("Invalid style")

    def process_rank_communication(rank, operation, new_schedule, style, stage_num, real_stage_num):
        """Process communication operations for single rank"""
        if operation is None:
            return

        stage_index = operation.stage_index
        pre_rank = stage_to_rank(stage_index - 1, style, stage_num, real_stage_num) if stage_index > 0 else 0
        nxt_rank = stage_to_rank(stage_index + 1, style, stage_num, real_stage_num) if stage_index < stage_num else None

        if (operation.type == MetaStepType.FWD and
                _need_com(operation, style, stage_num) and nxt_rank is not None):
            new_schedule[rank].append(MetaStep(
                micro_index=operation.micro_index,
                meta_type=MetaStepType.FWD_SEND,  # 注意：使用 FWD_SEND 而不是 FWD_SEND
                stage_index=stage_index
            ))
            new_schedule[nxt_rank].append(MetaStep(
                micro_index=operation.micro_index,
                meta_type=MetaStepType.FWD_RECV,  # 注意：使用 FWD_RECV 而不是 FWD_RECV
                stage_index=stage_index + 1
            ))
        elif (operation.type == MetaStepType.BWD and
              _need_com(operation, style, stage_num) and pre_rank is not None):
            new_schedule[rank].append(MetaStep(
                micro_index=operation.micro_index,
                meta_type=MetaStepType.BWD_SEND,  # 注意：使用 BWD_SEND 而不是 BWD_SEND
                stage_index=stage_index
            ))
            new_schedule[pre_rank].append(MetaStep(
                micro_index=operation.micro_index,
                meta_type=MetaStepType.BWD_RECV,  # 注意：使用 BWD_RECV 而不是 BWD_RECV
                stage_index=stage_index - 1
            ))

    # Main logic
    max_length = max(len(schedule) for schedule in scheduler.values())
    new_schedule = {rank: [] for rank in range(real_stage_num)}

    for time_step in range(max_length):
        current_operations = {}
        for rank in range(real_stage_num):
            if time_step < len(scheduler[rank]):
                operation = scheduler[rank][time_step]
                current_operations[rank] = operation
                if operation is not None:
                    new_schedule[rank].append(operation)
            else:
                current_operations[rank] = None

        # Process even rank communication
        for rank in range(0, real_stage_num, 2):
            process_rank_communication(rank, current_operations[rank], new_schedule,
                                       style, stage_num, real_stage_num)

        # Process odd rank communication
        for rank in range(1, real_stage_num, 2):
            process_rank_communication(rank, current_operations[rank], new_schedule,
                                       style, stage_num, real_stage_num)

    return new_schedule

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
                 output_concat_dim=None):
        super().__init__(stages,
                         micro_batch_num,
                         args_batch_dim=args_batch_dim,
                         kwargs_batch_dim=kwargs_batch_dim,
                         output_concat_dim=output_concat_dim)
        self.construct_exec_order()

    def construct_exec_order(self):
        """construct_exec_order of Gpipe."""
        for stage_index in range(self.real_stage_num):
            order_list = []
            for mb_index in range(self.micro_batch_num):
                if stage_index != 0:
                    order_list.append(MetaStep(mb_index, MetaStepType.FWD_RECV, stage_index))
                order_list.append(MetaStep(mb_index, MetaStepType.FWD, stage_index))
                if stage_index != self.stage.stage_num - 1:
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
                 output_concat_dim=None):
        super().__init__(stages,
                         micro_batch_num,
                         args_batch_dim=args_batch_dim,
                         kwargs_batch_dim=kwargs_batch_dim,
                         output_concat_dim=output_concat_dim)
        self.construct_exec_order()

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
    """
    The Interleaved 1F1B schedule.
    Support multiple stages per rank. It will perform one forward and one backward
    on the micro batches in steady state.
    We support cases where num_microbatch is less than or equal, or greater than the
    stage num, as well as cases where num_microbatch can't be evenly divided by the
    stage num.
    """
    def __init__(self,
                 stages,
                 micro_batch_num,
                 args_batch_dim=None,
                 kwargs_batch_dim=None,
                 output_concat_dim=None):
        super().__init__(stages,
                         micro_batch_num,
                         args_batch_dim=args_batch_dim,
                         kwargs_batch_dim=kwargs_batch_dim,
                         output_concat_dim=output_concat_dim)
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
        for stage_index in range(self.real_stage_num):
            self.exec_order[stage_index] = self.construct_stage_exec_order(stage_index)
        self.exec_order = add_send_recv(self.exec_order, self._stage_num, self.real_stage_num, style = 'loop')

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

    def construct_stage_exec_order(self, stage_index):
        """construct the execution order of specified stage_index."""
        warmup_ops = self.warmup_ops(stage_index)
        fwd_bwd_ops = self.n_local_stages * self.micro_batch_num - warmup_ops
        cooldown_ops = warmup_ops
        total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops
        # Pre-padding bubbles, stage starts with no-ops based on the warmup.
        order_list = [None for _ in range(stage_index)]
        fwd_stage_micro_index = defaultdict(int)
        bwd_stage_micro_index = defaultdict(int)
        # WarmUp Phase
        for op_idx in range(warmup_ops):
            fwd_stage_idx = self.forward_stage_index(op_idx, stage_index)
            fwd_micro_idx = fwd_stage_micro_index[fwd_stage_idx]
            order_list.append(MetaStep(fwd_micro_idx, MetaStepType.FWD, fwd_stage_idx))
            # If micro is less than stage num, there will be additional bubbles during warmup phase.
            if self.micro_batch_num < self.real_stage_num and fwd_micro_idx == self.micro_batch_num - 1:
                if op_idx != warmup_ops - 1 or stage_index == self.real_stage_num - 1:
                    order_list.extend([None] * (self.real_stage_num - self.micro_batch_num))
            fwd_stage_micro_index[fwd_stage_idx] += 1
        # If micro is less than 2 * (self.real_stage_num - stage_index - 1),
        # there will be additional bubbles during warmup phase.
        if self.micro_batch_num < 2 * (self.real_stage_num - stage_index - 1):
            order_list.extend([None] * (2 * (self.real_stage_num - stage_index - 1) - self.micro_batch_num))
        # Bubble from the end of warmup to the start of backward.
        order_list.extend([None] * (self.real_stage_num - 1 - stage_index))

        # 1f1b phase
        for op_idx in range(warmup_ops, warmup_ops+fwd_bwd_ops):
            fwd_stage_idx = self.forward_stage_index(op_idx, stage_index)
            fwd_micro_idx = fwd_stage_micro_index[fwd_stage_idx]
            order_list.append(MetaStep(fwd_micro_idx, MetaStepType.FWD, fwd_stage_idx))
            fwd_stage_micro_index[fwd_stage_idx] += 1
            bwd_stage_idx = self.backward_stage_index(op_idx - warmup_ops, stage_index)
            bwd_micro_idx = bwd_stage_micro_index[bwd_stage_idx]
            order_list.append(MetaStep(bwd_micro_idx, MetaStepType.BWD, bwd_stage_idx))
            # If micro is less than 2 * (self.real_stage_num - stage_index - 1),
            # there will be additional bubbles after 1f1b phase in last stage.
            if self.micro_batch_num < self.real_stage_num and bwd_micro_idx == self.micro_batch_num - 1:
                if stage_index == self.real_stage_num - 1:
                    order_list.extend([None] * (self.real_stage_num - self.micro_batch_num))
            bwd_stage_micro_index[bwd_stage_idx] += 1
        # cooldown phase
        for op_idx in range(warmup_ops+fwd_bwd_ops, total_ops):
            order_list.append(None)
            bwd_stage_idx = self.backward_stage_index(op_idx - warmup_ops, stage_index)
            bwd_micro_idx = bwd_stage_micro_index[bwd_stage_idx]
            order_list.append(MetaStep(bwd_micro_idx, MetaStepType.BWD, bwd_stage_idx))
            # If micro is less than 2 * (self.real_stage_num - stage_index - 1),
            # there will be additional bubbles during cooldown phase.
            if self.micro_batch_num < self.real_stage_num and bwd_micro_idx == self.micro_batch_num - 1:
                order_list.extend([None] * (self.real_stage_num - self.micro_batch_num))
            bwd_stage_micro_index[bwd_stage_idx] += 1
        return order_list
