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
"""Own pipeline synchronization boundaries without moving HSDP communication queues."""
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.core.fully_shard.hsdp_state import HSDPStateV2


class FSDPCommController:
    """Coordinate stage-local HSDP units on the caller's current device stream.

    Full backward retains its module hooks and delegates their gradient work
    here during a run. Split weight backward submits only after the complete
    weight calculation. Custom overlap callbacks own thread ordering, parameter
    handles, and stream dependencies; this class adds none.
    """

    def __init__(self, stages, micro_batch_num: int) -> None:
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
        self.retained_backward_units = set()
        self.unshard_handles = {}
        scheduler_owners = {}
        wrapper_owners = {}
        parameter_owners = {}
        storage_owners = {}
        for stage in stages:
            if isinstance(stage.submodule, HSDPModule):
                context = stage.submodule.hsdp_scheduler.scheduler_ctx
                self.stage_contexts[stage.stage_index] = context
                self.context_settings[stage.stage_index] = context.defer_grad_finalize
                self.stage_schedulers[stage.stage_index] = tuple(context.all_hsdp_schedulers)
            else:
                self.stage_schedulers[stage.stage_index] = ()

            for scheduler in self.stage_schedulers[stage.stage_index]:
                if scheduler in scheduler_owners:
                    raise ValueError("Pipeline stages must not share an HSDP scheduler")
                scheduler_owners[scheduler] = stage.stage_index
                settings = {
                    "reshard_after_forward": scheduler.reshard_after_forward,
                    "reshard_after_backward": scheduler.hsdp_state.reshard_after_backward,
                    "requires_gradient_sync": scheduler.hsdp_state.reduce_grads,
                    "requires_all_reduce": scheduler.hsdp_state.requires_all_reduce,
                }
                self.user_settings[scheduler] = settings
                self.current_settings[scheduler] = settings.copy()
                for param in scheduler.hsdp_state.hsdp_params:
                    if param in wrapper_owners:
                        raise ValueError("Pipeline stages must not share HSDP parameter wrappers")
                    wrapper_owners[param] = stage.stage_index

            # Storage identity also catches distinct Parameters created as views.
            # data_ptr() cannot distinguish the empty buffers used by lazy init.
            for param in stage.submodule.parameters():
                if param in parameter_owners and parameter_owners[param] != stage.stage_index:
                    raise ValueError("Pipeline stages must not share parameter objects")
                parameter_owners[param] = stage.stage_index
                local_param = param.to_local() if isinstance(param, DTensor) else param
                storage_id = local_param.untyped_storage()._cdata
                if storage_id in storage_owners and storage_owners[storage_id] != stage.stage_index:
                    raise ValueError("Pipeline stages must not share mutable parameter storage")
                storage_owners[storage_id] = stage.stage_index

    def begin(self) -> None:
        """Take ownership of module gradient processing until successful finalization.

        An interrupted run keeps its handler and communication state installed;
        starting another run must not overwrite that outstanding work.
        """
        if (
            any(self.completed_microbatches.values())
            or any(self.dx_microbatches.values())
            or self.unshard_handles
            or self.reduction_tail_stage is not None
            or any(context.post_backward_handler is not None
                   or context.defer_grad_finalize != self.context_settings[stage_index]
                   for stage_index, context in self.stage_contexts.items())
        ):
            raise RuntimeError("The previous pipeline run has unfinished HSDP work")
        for context in self.stage_contexts.values():
            context.defer_grad_finalize = True
            context.post_backward_handler = self.post_backward

    def post_backward(self, hsdp_state: HSDPStateV2) -> None:
        """Process one ready module without draining communication at empty units.

        Parameter-owning units keep the original wait-old, submit-current,
        submit-old-AllReduce order. An empty root has no gradient to submit,
        so its child tail remains for later computation or finalization.
        """
        hsdp_state.prepare_gradients()
        if not hsdp_state.reduce_grads or not hsdp_state.hsdp_params:
            return
        previous_groups = hsdp_state._wait_prev_reduce_scatter()
        hsdp_state._wait_prev_reduce_scatter_without_all_reduce()
        hsdp_state._issue_reduce_scatter_for_current_module()
        hsdp_state._issue_prev_fused_all_reduce(previous_groups)

    def reduce_dw_gradients(self, stage_index: int) -> None:
        """Consume old work, then submit every ready unit without waiting on this batch."""
        old_stage_indices = [stage_index]
        if self.reduction_tail_stage is not None and self.reduction_tail_stage != stage_index:
            old_stage_indices.insert(0, self.reduction_tail_stage)
        old_groups = []
        for old_stage_index in old_stage_indices:
            if self.stage_schedulers[old_stage_index]:
                state = self.stage_schedulers[old_stage_index][0].hsdp_state
                old_groups.append((state, state._wait_prev_reduce_scatter()))
                state._wait_prev_reduce_scatter_without_all_reduce()
        for scheduler in self.stage_schedulers[stage_index]:
            scheduler.hsdp_state.prepare_gradients()
        for scheduler in self.stage_schedulers[stage_index]:
            if scheduler.hsdp_state.reduce_grads:
                scheduler.hsdp_state._issue_reduce_scatter_for_current_module()
        for state, groups in old_groups:
            state._issue_prev_fused_all_reduce(groups)
        if self.stage_schedulers[stage_index]:
            self.reduction_tail_stage = stage_index

    def flush_reduce_scatter(self, stage_index: int) -> None:
        """Consume only this stage's original queue and submit its ready AllReduce work."""
        if self.stage_schedulers[stage_index]:
            self.stage_schedulers[stage_index][0].flush_reduce_scatter()

    def wait_all_reduce(self, stage_index: int) -> None:
        """Wait already-submitted AllReduce work without launching ReduceScatter."""
        if self.stage_schedulers[stage_index]:
            self.stage_schedulers[stage_index][0].wait_all_reduce()

    def finalize(self) -> None:
        """Finish a successful run, restore user settings, and preserve optimizer gradients.

        Failures leave outstanding state visible and propagate to the caller.
        Callbacks must have joined their compute threads and consumed all handles.
        """
        if self.unshard_handles or self.retained_backward_units:
            raise RuntimeError("Overlap callbacks must consume parameter handles and release retained units")
        for stage_index, schedulers in self.stage_schedulers.items():
            if not schedulers:
                continue
            if self.dx_microbatches[stage_index]:
                raise RuntimeError(f"Stage {stage_index} has unfinished weight backwards")
            if self.completed_microbatches[stage_index] != set(range(self.micro_batch_num)):
                raise RuntimeError(f"Stage {stage_index} did not complete every backward microbatch")

        # Releasing old inputs precedes any attempt to reuse their slots. This
        # also exposes historical shards for parameters unused in the last graph.
        for stage_index in self.stage_contexts:
            self.flush_reduce_scatter(stage_index)
        for schedulers in self.stage_schedulers.values():
            for scheduler in schedulers:
                scheduler.hsdp_state.prepare_gradients()
                scheduler.hsdp_state._issue_reduce_scatter_for_current_module()
        for stage_index in self.stage_contexts:
            self.flush_reduce_scatter(stage_index)
        for stage_index in self.stage_contexts:
            self.wait_all_reduce(stage_index)
        for stage_index, schedulers in self.stage_schedulers.items():
            if not schedulers:
                continue
            schedulers[0]._apply_reduced_gradients()
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
            self.stage_contexts[stage_index].post_backward_handler = None
        for completed in self.completed_microbatches.values():
            completed.clear()
        self.reduction_tail_stage = None
