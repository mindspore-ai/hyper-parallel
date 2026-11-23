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
"""Own pipeline synchronization boundaries for deferred HSDP work."""
from collections import deque
from collections.abc import Sequence

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.hsdp_state import HSDPStateV2
from hyper_parallel.core.pipeline_parallel._stage import PipelineStageBase


class FSDPCommController:
    """Coordinate stage-local HSDP gradients on the caller's current device stream."""

    def __init__(self, stages: Sequence[PipelineStageBase], micro_batch_num: int) -> None:
        """Record the stage-owned HSDP units and their settings."""
        if micro_batch_num <= 0:
            raise ValueError("micro_batch_num must be positive")
        self.micro_batch_num = micro_batch_num
        self.stage_schedulers = {}
        self.stage_contexts = {}
        self.user_settings = {}
        self.current_settings = {}
        self.context_settings = {}
        self.completed_microbatches = {stage.stage_index: set() for stage in stages}
        if len(self.completed_microbatches) != len(stages):
            raise ValueError("Local pipeline stages must have distinct stage indices")
        self.dx_microbatches = {stage.stage_index: set() for stage in stages}
        self.reduction_tail_stage = None
        self._handed_off_tail = None
        self.retained_backward_units = set()
        self._validate_stage_ownership(stages)
        for stage in stages:
            schedulers = stage.all_hsdp_schedulers
            if not schedulers:
                continue
            context = schedulers[0].scheduler_ctx
            self.stage_schedulers[stage.stage_index] = schedulers
            self.stage_contexts[stage.stage_index] = context
            for scheduler in schedulers:
                settings = {
                    "reshard_after_forward": scheduler.reshard_after_forward,
                    "reshard_after_backward": scheduler.hsdp_state.reshard_after_backward,
                    "requires_gradient_sync": scheduler.hsdp_state.reduce_grads,
                    "requires_all_reduce": scheduler.hsdp_state.requires_all_reduce,
                }
                self.user_settings[scheduler] = settings
                self.current_settings[scheduler] = settings.copy()

    @staticmethod
    def _validate_stage_ownership(stages: Sequence[PipelineStageBase]) -> None:
        """Ensure a scheduler or mutable parameter is owned by one stage."""
        scheduler_owners = set()
        wrapper_owners = set()
        parameter_owners = {}
        storage_owners = {}
        for stage in stages:
            for scheduler in stage.all_hsdp_schedulers:
                if scheduler in scheduler_owners:
                    raise ValueError("Pipeline stages must not share an HSDP scheduler")
                scheduler_owners.add(scheduler)
                for param in scheduler.hsdp_state.hsdp_params:
                    if param in wrapper_owners:
                        raise ValueError("Pipeline stages must not share HSDP parameter wrappers")
                    wrapper_owners.add(param)

            # Storage identity also catches distinct Parameters created as views.
            for param in stage.submodule.parameters():
                if param in parameter_owners and parameter_owners[param] != stage.stage_index:
                    raise ValueError("Pipeline stages must not share parameter objects")
                parameter_owners[param] = stage.stage_index
                local_param = param.to_local() if isinstance(param, DTensor) else param
                storage = local_param.untyped_storage()
                if storage in storage_owners and storage_owners[storage] != stage.stage_index:
                    raise ValueError("Pipeline stages must not share mutable parameter storage")
                storage_owners[storage] = stage.stage_index

    def begin(self) -> None:
        """Take ownership of root gradient finalization until the run completes.

        An interrupted run keeps its saved settings and communication state;
        starting another run must not overwrite that outstanding work.
        """
        if (
            self.context_settings
            or any(self.completed_microbatches.values())
            or any(self.dx_microbatches.values())
            or self.reduction_tail_stage is not None
            or self._handed_off_tail is not None
        ):
            raise RuntimeError("The previous pipeline run has unfinished HSDP work")
        if any(context.reduce_interval != 1 for context in self.stage_contexts.values()):
            raise ValueError("Deferred gradient finalization requires reduce_interval=1")
        if any(context.root_bp_state for context in self.stage_contexts.values()):
            raise RuntimeError("Set gradient finalization outside backward")
        for stage_index, context in self.stage_contexts.items():
            self.context_settings[stage_index] = context.defer_grad_finalize
            context.defer_grad_finalize = True

    def handoff_reduction_tail(self, backward_stages: set[int]) -> None:
        """Take the previous tail unless this step reduces gradients for its owner."""
        stage_index = self.reduction_tail_stage
        if stage_index is None or stage_index in backward_stages:
            return
        context = self.stage_contexts[stage_index]
        self.reduction_tail_stage = None
        if not (context.pre_reduce_scatter_params or context.pending_all_reduce_groups):
            return
        # This step cannot reuse the owner's parameter slots. Transfer only the queues.
        self._handed_off_tail = (
            stage_index, context.pre_reduce_scatter_params, context.pre_all_reduce_groups,
            context.pending_all_reduce_groups,
        )
        context.pre_reduce_scatter_params = deque()
        context.pre_all_reduce_groups = deque()
        context.pending_all_reduce_groups = []

    def finish_handed_off_tail(self) -> None:
        """Wait and apply the previous tail after the overlapping step returns."""
        if self._handed_off_tail is None:
            return
        stage_index, params, groups, all_reduce = self._handed_off_tail
        context = self.stage_contexts[stage_index]
        if context.pre_reduce_scatter_params or context.pre_all_reduce_groups or context.pending_all_reduce_groups:
            raise RuntimeError("The stage submitted gradients while its previous tail was handed off")
        context.pre_reduce_scatter_params = params
        context.pre_all_reduce_groups = groups
        context.pending_all_reduce_groups = all_reduce
        self._handed_off_tail = None
        self.flush_reduce_scatter(stage_index)
        self.wait_all_reduce(stage_index)
        self.stage_schedulers[stage_index][0].apply_reduced_gradients()

    def finish_backward(self, stage_index: int, micro_index: int) -> None:
        """Launch any blocked root gradients after the chunk's backward returns."""
        if self.reduction_tail_stage is not None and self.reduction_tail_stage != stage_index:
            self.flush_reduce_scatter(self.reduction_tail_stage)
        self.stage_schedulers[stage_index][0].launch_pending_reduce_scatter()
        self.reduction_tail_stage = stage_index
        self.completed_microbatches[stage_index].add(micro_index)

    @staticmethod
    def accumulate_grad_for_all_params_if_needed(hsdp_state: HSDPStateV2) -> None:
        """Accumulate full gradients and retain them locally during no-sync steps."""
        for param in hsdp_state.hsdp_params:
            param.accumulate_unsharded_grad_if_needed()
        if not hsdp_state.reduce_grads:
            for param in hsdp_state.hsdp_params:
                param.to_accumulated_grad_if_needed()

    def reduce_dw_gradients(self, stage_index: int) -> None:
        """Consume old work, then submit every ready unit without waiting on this batch."""
        schedulers = self.stage_schedulers.get(stage_index)
        if not schedulers:
            return

        old_groups = []
        for old_stage_index in dict.fromkeys((self.reduction_tail_stage, stage_index)):
            old_schedulers = self.stage_schedulers.get(old_stage_index)
            if old_schedulers:
                state = old_schedulers[0].hsdp_state
                old_groups.append((state, state.wait_reduce_scatter()))
        for scheduler in schedulers:
            state = scheduler.hsdp_state
            self.accumulate_grad_for_all_params_if_needed(state)
            if state.reshard_after_backward:
                state.shard()
        context = self.stage_contexts[stage_index]
        for scheduler in schedulers:
            state = scheduler.hsdp_state
            if state.reduce_grads and state.hsdp_params:
                state.issue_reduce_scatter_for_current_module()
                context.merge_reduce_scatter_tail()
        for state, groups in old_groups:
            state.issue_prev_fused_all_reduce(groups)
        self.reduction_tail_stage = stage_index

    def flush_reduce_scatter(self, stage_index: int) -> None:
        """Consume only this stage's original queue and submit its ready AllReduce work."""
        schedulers = self.stage_schedulers.get(stage_index)
        if schedulers:
            schedulers[0].flush_reduce_scatter()

    def wait_all_reduce(self, stage_index: int) -> None:
        """Wait already-submitted AllReduce work without launching ReduceScatter."""
        schedulers = self.stage_schedulers.get(stage_index)
        if schedulers:
            schedulers[0].wait_all_reduce()

    def finalize(self) -> None:
        """Finish a successful run, restore user settings, and preserve optimizer gradients.

        Failures leave outstanding state visible and propagate to the caller.
        Callbacks must have joined their compute threads.
        """
        if self.retained_backward_units:
            raise RuntimeError("Overlap callbacks must release retained units")
        for stage_index in self.stage_schedulers:
            if self.dx_microbatches[stage_index]:
                raise RuntimeError(f"Stage {stage_index} has unfinished weight backwards")
            if self.completed_microbatches[stage_index] != set(range(self.micro_batch_num)):
                raise RuntimeError(f"Stage {stage_index} did not complete every backward microbatch")

        self.finish_handed_off_tail()
        # Release outstanding slots before submitting any remaining full gradients.
        for stage_index in self.stage_contexts:
            self.flush_reduce_scatter(stage_index)
        for schedulers in self.stage_schedulers.values():
            for scheduler in schedulers:
                state = scheduler.hsdp_state
                self.accumulate_grad_for_all_params_if_needed(state)
                state.issue_reduce_scatter_for_current_module()
        for stage_index in self.stage_contexts:
            self.flush_reduce_scatter(stage_index)
        for stage_index in self.stage_contexts:
            self.wait_all_reduce(stage_index)
        for stage_index, schedulers in self.stage_schedulers.items():
            schedulers[0].apply_reduced_gradients()
            for scheduler in schedulers:
                scheduler.hsdp_state.shard()
                scheduler.reset_iter_state()
                settings = self.user_settings[scheduler]
                scheduler.set_reshard_after_forward(settings["reshard_after_forward"])
                scheduler.set_reshard_after_backward(settings["reshard_after_backward"])
                scheduler.set_requires_grad_sync(settings["requires_gradient_sync"])
                scheduler.set_requires_all_reduce(settings["requires_all_reduce"])
                self.current_settings[scheduler] = settings.copy()
            self.stage_contexts[stage_index].defer_grad_finalize = self.context_settings[stage_index]
        for completed in self.completed_microbatches.values():
            completed.clear()
        self.reduction_tail_stage = None
        self.context_settings.clear()
