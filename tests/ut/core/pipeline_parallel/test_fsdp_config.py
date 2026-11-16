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
"""Pipeline HSDP policy, ownership, and communication-boundary tests."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch import nn

from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerContext, HSDPSchedulerV2
from hyper_parallel.core.fully_shard.hsdp_state import HSDPStateV2
from hyper_parallel.core.pipeline_parallel._stage import PipelineStageBase
from hyper_parallel.core.pipeline_parallel.fsdp_comm_controller import FSDPCommController


class _ManagedLinear(HSDPModule, nn.Linear):
    """Keep real parameter identity while substituting the communication backend."""

    def __init__(self, in_features, out_features):
        nn.Linear.__init__(self, in_features, out_features)


def _managed_stage(stage_index=0, unit_count=1):
    """Build real stage bookkeeping with communication-only scheduler doubles."""
    module = _ManagedLinear(2, 2)
    context = HSDPSchedulerContext()
    context.root_module = module
    context.lazy_init_done = True
    schedulers = []
    for _ in range(unit_count):
        scheduler = MagicMock(spec=HSDPSchedulerV2)
        scheduler.scheduler_ctx = context
        scheduler.reshard_after_forward = True
        scheduler.hsdp_state = MagicMock(spec=HSDPStateV2)
        scheduler.hsdp_state.hsdp_params = []
        scheduler.hsdp_state.reshard_after_backward = True
        scheduler.hsdp_state.reduce_grads = True
        scheduler.hsdp_state.requires_all_reduce = True
        schedulers.append(scheduler)
    context.all_hsdp_schedulers = schedulers
    module.hsdp_scheduler = schedulers[0]
    stage = PipelineStageBase(module, stage_index, 4)
    return stage, schedulers


class TestFSDPConfiguration(unittest.TestCase):
    """Exercise stage entry policies without initializing distributed communication."""

    def test_plain_stage_is_no_op(self):
        stage = PipelineStageBase(nn.Linear(2, 2), 0, 1)
        controller = FSDPCommController([stage], 2)
        stage.fsdp_comm_controller = controller
        stage._configure_forward()
        stage._configure_backward(0)
        controller.reduce_dw_gradients(0)
        controller.flush_reduce_scatter(0)
        controller.wait_all_reduce(0)
        controller.finalize()
        self.assertEqual(controller.stage_contexts, {})
        self.assertIsNone(controller.reduction_tail_stage)

    def test_last_backward_uses_completion_order_and_preserves_residency(self):
        stage, (scheduler,) = _managed_stage()
        scheduler.hsdp_state.reduce_grads = False
        scheduler.hsdp_state.reshard_after_backward = False
        controller = FSDPCommController([stage], 3)
        stage.fsdp_comm_controller = controller
        for micro_index in (2, 0, 1):
            stage._configure_backward(micro_index)
            self.assertEqual(scheduler.set_requires_all_reduce.call_args.args[0], micro_index == 1)
            self.assertEqual(controller.current_settings[scheduler]["requires_gradient_sync"], micro_index == 1)
            self.assertFalse(controller.current_settings[scheduler]["reshard_after_backward"])
            stage._finish_backward(micro_index)
        scheduler.set_requires_grad_sync.assert_called_once_with(True)
        scheduler.set_reshard_after_backward.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "already completed"):
            stage._configure_backward(0)

    def test_unit_settings_remain_independent_and_cached(self):
        stage, (root, child) = _managed_stage(unit_count=2)
        root.reshard_after_forward = False
        root.hsdp_state.reshard_after_backward = False
        child.hsdp_state.reduce_grads = False
        controller = FSDPCommController([stage], 3)
        stage.fsdp_comm_controller = controller
        stage._configure_forward()
        stage._configure_forward()
        root.set_reshard_after_forward.assert_not_called()
        child.set_reshard_after_forward.assert_not_called()
        stage._configure_backward(0)
        self.assertTrue(controller.current_settings[root]["requires_gradient_sync"])
        self.assertFalse(controller.current_settings[child]["requires_gradient_sync"])
        self.assertFalse(controller.current_settings[root]["reshard_after_backward"])
        self.assertTrue(controller.current_settings[child]["reshard_after_backward"])

    def test_unfinished_weight_graphs_protect_forward_and_backward_parameters(self):
        stage, (scheduler,) = _managed_stage()
        controller = FSDPCommController([stage], 2)
        stage.fsdp_comm_controller = controller
        stage._configure_input_backward(0)
        stage._configure_input_backward(1)
        self.assertEqual(controller.dx_microbatches[0], {0, 1})
        self.assertEqual(controller.completed_microbatches[0], set())
        stage._configure_forward()
        scheduler.set_reshard_after_forward.assert_called_once_with(False)
        stage._configure_backward(0)
        self.assertFalse(controller.current_settings[scheduler]["reshard_after_backward"])
        controller.dx_microbatches[0].remove(0)
        stage._configure_backward(1)
        self.assertTrue(controller.current_settings[scheduler]["reshard_after_backward"])
        with self.assertRaisesRegex(RuntimeError, "already started"):
            stage._configure_input_backward(1)

    def test_callback_retention_protects_only_the_requested_unit(self):
        stage, (root, child) = _managed_stage(unit_count=2)
        controller = FSDPCommController([stage], 1)
        stage.fsdp_comm_controller = controller
        controller.retained_backward_units.add(root)
        stage._configure_backward(0)
        self.assertFalse(controller.current_settings[root]["reshard_after_backward"])
        self.assertTrue(controller.current_settings[child]["reshard_after_backward"])

    def test_full_backward_consumes_another_stage_after_compute(self):
        previous, (previous_scheduler,) = _managed_stage(0)
        current, (current_scheduler,) = _managed_stage(2)
        controller = FSDPCommController([previous, current], 1)
        current.fsdp_comm_controller = controller
        controller.reduction_tail_stage = 0
        events = ["current backward finished"]
        previous_scheduler.flush_reduce_scatter.side_effect = lambda: events.append("previous reduce scatter")
        current._finish_backward(0)
        self.assertEqual(events, ["current backward finished", "previous reduce scatter"])
        current_scheduler.flush_reduce_scatter.assert_not_called()
        self.assertEqual(controller.reduction_tail_stage, 2)
        self.assertIs(controller.stage_contexts[0], previous_scheduler.scheduler_ctx)
        self.assertIsNot(controller.stage_contexts[0], controller.stage_contexts[2])

    def test_same_stage_full_backward_keeps_its_new_tail(self):
        stage, (scheduler,) = _managed_stage()
        controller = FSDPCommController([stage], 1)
        stage.fsdp_comm_controller = controller
        controller.reduction_tail_stage = 0
        stage._finish_backward(0)
        scheduler.flush_reduce_scatter.assert_not_called()

    def test_weight_batch_submits_all_units_before_old_all_reduce(self):
        stage, schedulers = _managed_stage(unit_count=2)
        controller = FSDPCommController([stage], 2)
        controller.reduction_tail_stage = 0
        events = []
        root_state = schedulers[0].hsdp_state
        root_state._wait_prev_reduce_scatter.side_effect = lambda: events.append("wait old groups") or []
        root_state._wait_prev_reduce_scatter_without_all_reduce.side_effect = lambda: events.append("wait old shards")
        root_state._issue_prev_fused_all_reduce.side_effect = lambda groups: events.append("submit old all reduce")
        for index, scheduler in enumerate(schedulers):
            scheduler.hsdp_state.prepare_gradients.side_effect = lambda index=index: events.append(("prepare", index))
            scheduler.hsdp_state._issue_reduce_scatter_for_current_module.side_effect = (
                lambda index=index: events.append(("submit", index))
            )
        controller.reduce_dw_gradients(0)
        self.assertEqual(events, ["wait old groups", "wait old shards", ("prepare", 0), ("prepare", 1),
                                  ("submit", 0), ("submit", 1), "submit old all reduce"])
        schedulers[0].flush_reduce_scatter.assert_not_called()

    def test_unit_backward_keeps_reduction_order_and_empty_units_do_not_consume(self):
        """Only a parameter-owning PP unit advances the module reduction pipeline."""
        stage, (root, child) = _managed_stage(unit_count=2)
        controller = FSDPCommController([stage], 2)
        child.hsdp_state.hsdp_params = [object()]
        events = []
        child.hsdp_state.prepare_gradients.side_effect = lambda: events.append("prepare")
        child.hsdp_state._wait_prev_reduce_scatter.side_effect = lambda: events.append("wait groups") or []
        child.hsdp_state._wait_prev_reduce_scatter_without_all_reduce.side_effect = lambda: events.append("wait shards")
        child.hsdp_state._issue_reduce_scatter_for_current_module.side_effect = lambda: events.append("submit current")
        child.hsdp_state._issue_prev_fused_all_reduce.side_effect = lambda groups: events.append("submit old all reduce")

        controller.post_backward(root.hsdp_state)
        root.hsdp_state.prepare_gradients.assert_called_once_with()
        root.hsdp_state._wait_prev_reduce_scatter.assert_not_called()
        root.hsdp_state._wait_prev_reduce_scatter_without_all_reduce.assert_not_called()
        root.hsdp_state._issue_reduce_scatter_for_current_module.assert_not_called()
        controller.post_backward(child.hsdp_state)
        self.assertEqual(events, ["prepare", "wait groups", "wait shards", "submit current", "submit old all reduce"])

    def test_finalize_rejects_incomplete_graphs_and_handles(self):
        stage, _ = _managed_stage()
        controller = FSDPCommController([stage], 1)
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            controller.finalize()
        controller.completed_microbatches[0].add(0)
        controller.dx_microbatches[0].add(0)
        with self.assertRaisesRegex(RuntimeError, "unfinished weight"):
            controller.finalize()
        controller.dx_microbatches[0].clear()
        controller.unshard_handles[object()] = object()
        with self.assertRaisesRegex(RuntimeError, "consume parameter handles"):
            controller.finalize()

    def test_finalize_restores_each_unit_and_preserves_context_identity(self):
        stage, (scheduler,) = _managed_stage()
        scheduler.reshard_after_forward = False
        scheduler.hsdp_state.reduce_grads = False
        scheduler.hsdp_state.requires_all_reduce = False
        controller = FSDPCommController([stage], 1)
        stage.fsdp_comm_controller = controller
        context = controller.stage_contexts[0]
        controller.begin()
        self.assertEqual(context.post_backward_handler, controller.post_backward)
        with self.assertRaisesRegex(RuntimeError, "unfinished HSDP work"):
            controller.begin()
        stage._configure_backward(0)
        stage._finish_backward(0)
        controller.finalize()
        scheduler._apply_reduced_gradients.assert_called_once()
        scheduler.set_reshard_after_forward.assert_called_with(False)
        scheduler.set_requires_grad_sync.assert_called_with(False)
        scheduler.set_requires_all_reduce.assert_called_with(False)
        self.assertFalse(context.defer_grad_finalize)
        self.assertIsNone(context.post_backward_handler)
        self.assertEqual(controller.completed_microbatches[0], set())
        self.assertIs(context, scheduler.scheduler_ctx)

    def test_communication_failure_does_not_reset_or_apply_gradients(self):
        stage, (scheduler,) = _managed_stage()
        controller = FSDPCommController([stage], 1)
        controller.begin()
        controller.completed_microbatches[0].add(0)
        scheduler.hsdp_state._issue_reduce_scatter_for_current_module.side_effect = RuntimeError("collective failed")
        with self.assertRaisesRegex(RuntimeError, "collective failed"):
            controller.finalize()
        scheduler.reset_iter_state.assert_not_called()
        scheduler._apply_reduced_gradients.assert_not_called()
        self.assertTrue(controller.stage_contexts[0].defer_grad_finalize)
        self.assertEqual(controller.stage_contexts[0].post_backward_handler, controller.post_backward)
        with self.assertRaisesRegex(RuntimeError, "unfinished HSDP work"):
            controller.begin()

    def test_stages_cannot_share_parameter_storage_or_schedulers(self):
        first = nn.Linear(2, 2)
        second = nn.Linear(2, 2)
        second.weight = nn.Parameter(first.weight.detach())
        stages = [PipelineStageBase(first, 0, 2), PipelineStageBase(second, 1, 2)]
        with self.assertRaisesRegex(ValueError, "storage"):
            FSDPCommController(stages, 1)
        stage, _ = _managed_stage()
        other = SimpleNamespace(stage_index=2, submodule=stage.submodule)
        with self.assertRaisesRegex(ValueError, "share an HSDP scheduler"):
            FSDPCommController([stage, other], 1)

    def test_plain_root_with_managed_children_and_forward_only_managed_stage_rejected(self):
        child, _ = _managed_stage()
        with self.assertRaisesRegex(ValueError, "HSDPModule root"):
            PipelineStageBase(nn.Sequential(child.submodule), 0, 1)
        with self.assertRaisesRegex(ValueError, "require backward"):
            PipelineStageBase(child.submodule, 0, 1, has_backward=False)
