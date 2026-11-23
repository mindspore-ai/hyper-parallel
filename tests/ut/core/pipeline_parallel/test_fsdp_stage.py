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
"""Real autograd coverage for pipeline HSDP lifetimes with a mocked mesh backend."""
import copy
import unittest
from contextlib import ExitStack
from itertools import product
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, clear_device_mesh_cache
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy
from hyper_parallel.core.dtensor.dtensor import SkipDTensorDispatch
from hyper_parallel.core.pipeline_parallel.fsdp_comm_controller import FSDPCommController
from hyper_parallel.core.pipeline_parallel.scheduler import Schedule1F1B, ScheduleGPipe
from hyper_parallel.core.pipeline_parallel.stage import PipelineStage


class _CheckpointModel(nn.Module):
    """Small symmetric model whose peer shards can be simulated on CPU."""

    def __init__(self, checkpointing):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Sigmoid(), nn.Linear(2, 2, bias=False))
        self.checkpointing = checkpointing
        for param in self.parameters():
            nn.init.constant_(param, 0.125)

    def forward(self, inputs):
        """Run both linear layers, optionally inside a non-reentrant checkpoint."""
        if self.checkpointing:
            return checkpoint(self.layers, inputs, use_reentrant=False)
        return self.layers(inputs)


class _ConditionalModel(nn.Module):
    """Exercise a child that first gathers its parameters on a later microbatch."""

    def __init__(self):
        super().__init__()
        self.always = nn.Linear(2, 2, bias=False)
        self.optional = nn.Linear(2, 2, bias=False)

    def forward(self, inputs: torch.Tensor, use_optional: bool) -> torch.Tensor:
        """Run the optional child when selected."""
        output = self.always(inputs)
        return self.optional(output) if use_optional else output


class TestFSDPStageAutograd(unittest.TestCase):
    """Use real parameters and hooks without initializing distributed processes."""

    def setUp(self) -> None:
        """Keep the simulated CPU mesh patches active for the test."""
        clear_device_mesh_cache()
        self.addCleanup(clear_device_mesh_cache)
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        self._patches = ExitStack()
        self.addCleanup(self._patches.close)
        group = MagicMock(spec=torch.distributed.ProcessGroup)
        group.rank.return_value = 0
        group.size.return_value = 1
        self._patches.enter_context(patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0))
        self._patches.enter_context(patch.object(DeviceMesh, "get_group", return_value=group))
        self._patches.enter_context(patch(
            "hyper_parallel.core.fully_shard.api._get_device_from_mesh", return_value=torch.device("cpu"),
        ))
        self.mesh = DeviceMesh("cpu", [0], mesh_dim_names=("fsdp",), _init_backend=False)

    def _run_policy(self, mesh_shape, forward_reshard, backward_reshard, gradient_sync, split, checkpointing, nested,
                    reduce_op="avg", grouped=False):
        """Compare several training steps using real hooks and simulated identical peers."""
        clear_device_mesh_cache()

        def mesh_group(mesh, mesh_dim=None):
            if isinstance(mesh_dim, str):
                mesh_dim = mesh.mesh_dim_names.index(mesh_dim)
            group = MagicMock(spec=torch.distributed.ProcessGroup)
            group.size.return_value = mesh.mesh_shape[mesh_dim or 0]
            group.rank.return_value = 0
            return group

        def all_gather(output, tensor, group, async_op):
            output.copy_(tensor.repeat(group.size()))
            return MagicMock() if async_op else None

        def reduce_scatter(output, tensor, op, group, async_op):
            output.copy_(tensor.reshape(group.size(), -1).sum(0))
            if op == torch.distributed.ReduceOp.AVG:
                output.div_(group.size())
            return MagicMock() if async_op else None

        def all_reduce(tensor, op, group, async_op):
            if op == torch.distributed.ReduceOp.SUM:
                tensor.mul_(group.size())
            return MagicMock() if async_op else None

        with (
            patch.object(DeviceMesh, "get_group", new=mesh_group),
            patch("hyper_parallel.core.fully_shard.hsdp_param.dist.all_gather_into_tensor", side_effect=all_gather),
            patch("hyper_parallel.core.fully_shard.hsdp_param.dist.reduce_scatter_tensor", side_effect=reduce_scatter),
            patch("hyper_parallel.core.fully_shard.param_group.dist.all_reduce", side_effect=all_reduce),
        ):
            mesh = DeviceMesh("cpu", torch.arange(torch.tensor(mesh_shape).prod()).reshape(mesh_shape).tolist(),
                              mesh_dim_names=("replicate", "shard"), _init_backend=False)
            model = _CheckpointModel(checkpointing)
            reference = copy.deepcopy(model)
            if nested:
                if grouped:
                    fully_shard([model.layers[0], model.layers[2]], mesh=mesh)
                else:
                    for layer in (model.layers[0], model.layers[2]):
                        fully_shard(layer, mesh=mesh)
            fully_shard(model, mesh=mesh)
            model.set_reduce_op_type(reduce_op)
            model.set_reshard_after_forward(forward_reshard)
            model.set_reshard_after_backward(backward_reshard)
            model.set_requires_gradient_sync(gradient_sync)
            stage = PipelineStage(model, 1, 3, device=torch.device("cpu"))
            controller = FSDPCommController([stage], 3)
            stage.fsdp_comm_controller = controller
            self.assertEqual(len(controller.stage_schedulers[1]), (2 if grouped else 3) if nested else 1)
            # torch_npu probes the NPU when SGD automatically selects foreach kernels.
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False)
            full_parameters = None

            for training_step in range(3):
                optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
                controller.begin()
                for micro_index in range(3):
                    inputs = torch.full((1, 2), (micro_index + training_step + 1) / 4, requires_grad=True)
                    stage.args_recv_info[micro_index] = [SimpleNamespace(buffer=inputs, requires_grad=True)]
                    stage.grad_recv_info[micro_index] = [SimpleNamespace(buffer=torch.ones(1, 2))]
                    reference_loss = reference(inputs.detach()).sum()
                    if reduce_op == "sum":
                        reference_loss = reference_loss * (mesh_shape[0] * mesh_shape[1])
                    reference_loss.backward()
                    stage.forward_one_chunk(micro_index)
                    if full_parameters is None:
                        full_parameters = stage._get_pp_autograd_params()
                    if split:
                        stage.backward_input_one_chunk(micro_index)
                for micro_index in (2, 0, 1):
                    if split:
                        stage.backward_weight_one_chunk(micro_index)
                    else:
                        stage.backward_one_chunk(micro_index)
                controller.finalize()
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual.grad.to_local(), expected.grad.chunk(mesh_shape[1], dim=0)[0])
                with SkipDTensorDispatch():
                    optimizer.step()
                reference_optimizer.step()
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual.to_local(), expected.chunk(mesh_shape[1], dim=0)[0])
                self.assertIs(stage._get_pp_autograd_params(), full_parameters)
                self.assertFalse(controller.stage_contexts[1].defer_grad_finalize)
                self.assertFalse(controller.context_settings)

    def test_policy_matrix_matches_reference_across_training_steps(self):
        """Independent switches, split graphs, checkpointing, and grouped nested units compose."""
        for mesh_shape, settings, split, checkpointing, nested in product(
            ((1, 1), (1, 2), (2, 1), (2, 2)),
            ((False, False, False), (False, False, True), (True, True, True),
             (True, True, False), (False, True, True), (True, False, True)),
            (False, True), (False, True), (False, True),
        ):
            with self.subTest(mesh_shape=mesh_shape, settings=settings, split=split,
                              checkpointing=checkpointing, nested=nested):
                self._run_policy(mesh_shape, *settings, split, checkpointing, nested)

    def test_sum_reduction_preserves_shard_and_replica_scaling(self):
        """Match the sum-reduced reference across shard and replica groups."""
        for split in (False, True):
            with self.subTest(split=split):
                self._run_policy((2, 2), True, True, True, split, True, True, reduce_op="sum")

    def test_grouped_units_share_one_scheduler_and_stable_parameters(self):
        """Reuse the scheduler and parameter identities shared by grouped modules."""
        for split in (False, True):
            with self.subTest(split=split):
                self._run_policy((2, 2), True, True, True, split, True, True, grouped=True)

    def test_split_backward_uses_full_parameter_first_seen_on_later_forward(self):
        """A late parameter participates in dW and earlier dX group identities survive."""
        model = _ConditionalModel()
        reference = copy.deepcopy(model)
        fully_shard(model.optional, mesh=self.mesh)
        fully_shard(model, mesh=self.mesh)
        model.set_reshard_after_forward(True)
        stage = PipelineStage(model, 1, 3, device=torch.device("cpu"))
        controller = FSDPCommController([stage], 2)
        stage.fsdp_comm_controller = controller
        controller.begin()

        self.assertEqual(stage._get_pp_autograd_params(), ())
        for micro_index, use_optional in enumerate((False, True)):
            inputs = torch.ones(1, 2, requires_grad=True)
            stage.args_recv_info[micro_index] = [SimpleNamespace(buffer=inputs, requires_grad=True)]
            stage.grad_recv_info[micro_index] = [SimpleNamespace(buffer=torch.ones_like(inputs))]
            reference(inputs.detach(), use_optional).sum().backward()
            stage.forward_one_chunk(micro_index, kwargs={"use_optional": use_optional})
            self.assertEqual(len(stage._get_pp_autograd_params()), micro_index + 1)
            stage.backward_input_one_chunk(micro_index)

        for micro_index in (0, 1):
            stage.backward_weight_one_chunk(micro_index)
        controller.finalize()

        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad.to_local(), expected.grad)

    def test_schedule_can_be_built_before_meta_parameters_are_materialized(self):
        """First forward initializes the shards after the caller materializes them."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False, device="meta"))
        fully_shard(model[0], mesh=self.mesh)
        fully_shard(model, mesh=self.mesh)
        stage = PipelineStage(model, 0, 1, device=torch.device("cpu"), mesh=self.mesh)
        schedule = ScheduleGPipe(stage, 2)
        root = model.hsdp_scheduler
        child = model[0].hsdp_scheduler
        self.assertFalse(root.scheduler_ctx.lazy_init_done)

        model.to_empty(device="cpu")
        with SkipDTensorDispatch():
            nn.init.ones_(model[0].weight.to_local())
        schedule.run(torch.ones(2, 2))

        self.assertTrue(root.scheduler_ctx.lazy_init_done)
        self.assertIs(root.scheduler_ctx, child.scheduler_ctx)
        torch.testing.assert_close(model[0].weight.grad.to_local(), torch.full((2, 2), 2.0))

    def test_deferred_finalization_requires_interval_one(self):
        """Reject incompatible settings before initialization and when enabling defer later."""
        model = nn.Linear(2, 2, bias=False)
        fully_shard(model, mesh=self.mesh)
        context = model.hsdp_scheduler.scheduler_ctx
        context.defer_grad_finalize = True
        model.set_reduce_comm_interval(2)

        with self.assertRaisesRegex(ValueError, "reduce_interval=1"):
            model(torch.ones(1, 2))
        self.assertFalse(context.lazy_init_done)

        context.defer_grad_finalize = False
        model(torch.ones(1, 2))
        context.defer_grad_finalize = True
        with self.assertRaisesRegex(ValueError, "reduce_interval=1"):
            model(torch.ones(1, 2))

    def test_forward_overlaps_the_tail_owned_by_the_controller(self):
        """A forward sees an empty scheduler queue and the preceding RS still live."""
        model = nn.Linear(2, 2, bias=False)
        fully_shard(model, mesh=self.mesh)
        stage = PipelineStage(model, 0, 1, device=torch.device("cpu"), mesh=self.mesh)
        schedule = Schedule1F1B(stage, 2)
        scheduler = model.hsdp_scheduler
        param = scheduler.hsdp_state.hsdp_params[0]
        original_forward = model.forward
        observed = []

        def observe_forward(inputs):
            observed.append((bool(scheduler.scheduler_ctx.pre_reduce_scatter_params),
                             param.reduce_scatter_comm_ctx.reduce_scatter_output is not None))
            return original_forward(inputs)

        with patch.object(model, "forward", side_effect=observe_forward):
            schedule.run(torch.ones(2, 2))

        self.assertEqual(observed, [(False, False), (False, True)])
        self.assertIsNone(param.reduce_scatter_comm_ctx.reduce_scatter_output)
        torch.testing.assert_close(model.weight.grad.to_local(), torch.full((2, 2), 2.0))

    def test_deferred_root_leaves_occupied_slots_until_explicit_finalization(self):
        """An input without gradients uses launch-only root handling on both backwards."""
        model = nn.Linear(2, 2, bias=False)
        fully_shard(model, mesh=self.mesh)
        scheduler = model.hsdp_scheduler
        param = scheduler.hsdp_state.hsdp_params[0]
        context = scheduler.scheduler_ctx
        context.defer_grad_finalize = True

        model(torch.ones(1, 2)).sum().backward()
        first_output = param.reduce_scatter_comm_ctx.reduce_scatter_output
        self.assertIsNotNone(first_output)
        model(torch.ones(1, 2)).sum().backward()
        self.assertIs(param.reduce_scatter_comm_ctx.reduce_scatter_output, first_output)

        scheduler._finalize_gradients()
        self.assertFalse(context.pre_reduce_scatter_params)
        torch.testing.assert_close(model.weight.grad.to_local(), torch.full((2, 2), 2.0))

    def test_async_unshard_reuses_work_and_completed_parameters(self):
        """Reuse one all-gather for repeated requests on the same full parameters."""
        group = MagicMock(spec=torch.distributed.ProcessGroup)
        group.rank.return_value = 0
        group.size.return_value = 2
        work = MagicMock()

        def all_gather(output, tensor, **_kwargs):
            output.copy_(tensor.repeat(2))
            return work

        with patch.object(DeviceMesh, "get_group", return_value=group), \
                patch("hyper_parallel.core.fully_shard.hsdp_param.dist.all_gather_into_tensor",
                      side_effect=all_gather) as gather:
            mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("fsdp",), _init_backend=False)
            model = nn.Linear(2, 2, bias=False)
            fully_shard(model, mesh=mesh)
            PipelineStage(model, 0, 1, device=torch.device("cpu"))
            first = model.unshard(async_op=True)
            second = model.unshard(async_op=True)
            self.assertEqual(gather.call_count, 1)
            first.wait()
            second.wait()
            model.unshard()
            gather.assert_called_once()
            work.wait.assert_called_once()

    def test_forward_boundary_fires_after_output_caches_are_ready(self):
        """Expose the forward result in both caches before calling the boundary hook."""
        model = nn.Linear(2, 2)
        stage = PipelineStage(model, 0, 1, device=torch.device("cpu"))
        observed = []

        def after_forward(stage_index, micro_index):
            self.assertEqual(stage_index, 0)
            self.assertIs(stage.fwd_cache[micro_index], stage.fwd_outputs_cache[micro_index])
            observed.append(micro_index)

        stage._after_forward_chunk = after_forward
        stage.forward_one_chunk(0, (torch.ones(1, 2),))
        self.assertEqual(observed, [0])

    def test_no_gradient_input_uses_one_full_backward_at_weight_step(self):
        """Run full backward at the weight step when the input requires no gradient."""
        model = nn.Linear(2, 2, bias=False)
        reference = copy.deepcopy(model)
        fully_shard(model, mesh=self.mesh)
        stage = PipelineStage(model, 1, 3, device=torch.device("cpu"))
        controller = FSDPCommController([stage], 1)
        stage.fsdp_comm_controller = controller
        controller.begin()
        inputs = torch.ones(1, 2)
        stage.args_recv_info[0] = [SimpleNamespace(buffer=inputs, requires_grad=False)]
        stage.grad_recv_info[0] = [SimpleNamespace(buffer=torch.ones(1, 2))]
        stage.forward_one_chunk(0)
        stage.backward_input_one_chunk(0)
        self.assertIn(0, stage.fwd_cache)
        self.assertEqual(controller.completed_microbatches[1], set())
        stage.backward_weight_one_chunk(0)
        self.assertEqual(controller.completed_microbatches[1], {0})
        self.assertEqual(controller.dx_microbatches[1], set())
        controller.finalize()
        reference(inputs).sum().backward()
        torch.testing.assert_close(next(model.parameters()).grad.to_local(), reference.weight.grad)

    def test_ignored_and_frozen_parameters_keep_their_gradient_ownership(self):
        """Accumulate gradients on ignored parameters and leave frozen parameters without gradients."""
        model = _CheckpointModel(False)
        model.layers[2].weight.requires_grad_(False)
        reference = copy.deepcopy(model)
        ignored_param = model.layers[0].weight
        fully_shard(model, mesh=self.mesh, ignored_params={ignored_param})
        stage = PipelineStage(model, 0, 1, device=torch.device("cpu"))
        controller = FSDPCommController([stage], 1)
        stage.fsdp_comm_controller = controller
        controller.begin()
        inputs = torch.ones(1, 2)
        stage.forward_one_chunk(0, (inputs,))
        stage.backward_one_chunk(0)
        controller.finalize()
        reference(inputs).sum().backward()
        self.assertIs(stage._get_pp_autograd_params()[0], ignored_param)
        torch.testing.assert_close(ignored_param.grad, reference.layers[0].weight.grad)
        self.assertIsNone(model.layers[2].weight.grad)

    def test_unsupported_sharding_combinations_fail_during_construction(self):
        """Reject communication fusion and parameter offload for HSDP pipeline stages."""
        for options, message in (
            ({"comm_fusion": True}, "comm_fusion"),
            ({"offload_policy": CPUOffloadPolicy(pin_memory=False)}, "offload"),
        ):
            with self.subTest(options=options):
                model = nn.Linear(2, 2)
                fully_shard(model, mesh=self.mesh, **options)
                with self.assertRaisesRegex(ValueError, message):
                    PipelineStage(model, 0, 1, device=torch.device("cpu"))

    def test_previous_microbatch_consumption_preserves_current_gradient(self):
        """A deferred old reduction must not delete gradients from the next backward."""
        model = nn.Linear(2, 2, bias=False)
        reference = copy.deepcopy(model)
        fully_shard(model, mesh=self.mesh)
        model.set_reduce_op_type("sum")
        stage = PipelineStage(model, 0, 1, device=torch.device("cpu"))
        controller = FSDPCommController([stage], 2)
        stage.fsdp_comm_controller = controller
        controller.begin()

        for micro_index in range(2):
            inputs = torch.full((1, 2), float(micro_index + 1))
            reference(inputs).sum().backward()
            stage.forward_one_chunk(micro_index, (inputs,))
            stage.backward_one_chunk(micro_index)

        controller.finalize()

        torch.testing.assert_close(next(model.parameters()).grad.to_local(), reference.weight.grad)

    def test_empty_root_preserves_child_tail_until_next_backward(self):
        """Deferral preserves the tail across empty root hooks."""
        model = _CheckpointModel(False)
        reference = copy.deepcopy(model)
        for layer in (model.layers[0], model.layers[2]):
            fully_shard(layer, mesh=self.mesh)
        fully_shard(model, mesh=self.mesh)
        model.set_reduce_op_type("sum")
        stage = PipelineStage(model, 1, 3, device=torch.device("cpu"))
        controller = FSDPCommController([stage], 2)
        stage.fsdp_comm_controller = controller
        context = controller.stage_contexts[1]
        self.assertEqual(model.hsdp_scheduler.hsdp_state.hsdp_params, [])
        original_post_backward = self._patches.enter_context(patch.object(
            model.hsdp_scheduler.hsdp_state, "post_backward",
            wraps=model.hsdp_scheduler.hsdp_state.post_backward,
        ))
        controller.begin()
        for micro_index in range(2):
            inputs = torch.full((1, 2), float(micro_index + 1), requires_grad=True)
            stage.args_recv_info[micro_index] = [SimpleNamespace(buffer=inputs, requires_grad=True)]
            stage.grad_recv_info[micro_index] = [SimpleNamespace(buffer=torch.ones(1, 2))]
            stage.forward_one_chunk(micro_index)
            stage.backward_one_chunk(micro_index)
            self.assertEqual(list(context.pre_reduce_scatter_params),
                             [model.layers[0].hsdp_scheduler.hsdp_state.hsdp_params])
            reference(inputs.detach()).sum().backward()

        original_post_backward.assert_not_called()
        controller.finalize()
        self.assertFalse(context.pre_reduce_scatter_params)
        self.assertFalse(context.defer_grad_finalize)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad.to_local(), expected.grad)

        inputs = torch.ones(1, 2, requires_grad=True)
        model(inputs).sum().backward()
        reference(inputs.detach()).sum().backward()
        original_post_backward.assert_called_once_with()
        self.assertFalse(context.pre_reduce_scatter_params)

        context.defer_grad_finalize = True
        model(inputs).sum().backward()
        reference(inputs.detach()).sum().backward()
        self.assertEqual(list(context.pre_reduce_scatter_params),
                         [model.layers[0].hsdp_scheduler.hsdp_state.hsdp_params])

        model.hsdp_scheduler._finalize_gradients()
        self.assertFalse(context.pre_reduce_scatter_params)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad.to_local(), expected.grad)

    def test_full_backward_consumes_the_batch_left_by_weight_backward(self):
        """A full backward can reuse the slots of every unit in a preceding dW batch."""
        model = _CheckpointModel(False)
        reference = copy.deepcopy(model)
        for layer in (model.layers[0], model.layers[2]):
            fully_shard(layer, mesh=self.mesh)
        fully_shard(model, mesh=self.mesh)
        stage = PipelineStage(model, 1, 3, device=torch.device("cpu"))
        controller = FSDPCommController([stage], 2)
        stage.fsdp_comm_controller = controller
        controller.begin()
        for micro_index in range(2):
            inputs = torch.full((1, 2), float(micro_index + 1), requires_grad=True)
            stage.args_recv_info[micro_index] = [SimpleNamespace(buffer=inputs, requires_grad=True)]
            stage.grad_recv_info[micro_index] = [SimpleNamespace(buffer=torch.ones_like(inputs))]
            stage.forward_one_chunk(micro_index)
            reference(inputs.detach()).sum().backward()

        stage.backward_input_one_chunk(0)
        stage.backward_weight_one_chunk(0)
        self.assertEqual([len(params) for params in controller.stage_contexts[1].pre_reduce_scatter_params], [2])
        stage.backward_one_chunk(1)
        controller.finalize()

        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad.to_local(), expected.grad)
