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
from itertools import product
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, clear_device_mesh_cache
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy
from hyper_parallel.core.dtensor.dtensor import SkipDTensorDispatch
from hyper_parallel.core.pipeline_parallel.fsdp_comm_controller import FSDPCommController
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
        if self.checkpointing:
            return checkpoint(self.layers, inputs, use_reentrant=False)
        return self.layers(inputs)


class TestFSDPStageAutograd(unittest.TestCase):
    """Use real parameters and hooks without initializing distributed processes."""

    def setUp(self):
        clear_device_mesh_cache()
        self.addCleanup(clear_device_mesh_cache)
        group = MagicMock(spec=torch.distributed.ProcessGroup)
        group.rank.return_value = 0
        group.size.return_value = 1
        self.enterContext(patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0))
        self.enterContext(patch.object(DeviceMesh, "get_group", return_value=group))
        self.enterContext(patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh",
                                return_value=torch.device("cpu")))
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

        with patch.object(DeviceMesh, "get_group", new=mesh_group), \
                patch("hyper_parallel.core.fully_shard.hsdp_param.dist.all_gather_into_tensor", side_effect=all_gather), \
                patch("hyper_parallel.core.fully_shard.hsdp_param.dist.reduce_scatter_tensor", side_effect=reduce_scatter), \
                patch("hyper_parallel.core.fully_shard.param_group.dist.all_reduce", side_effect=all_reduce):
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
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
            full_parameters = stage._get_trainable_params()

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
                self.assertIs(stage._get_trainable_params(), full_parameters)
                self.assertFalse(controller.stage_contexts[1].defer_grad_finalize)
                self.assertIsNone(controller.stage_contexts[1].post_backward_handler)

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
        for split in (False, True):
            with self.subTest(split=split):
                self._run_policy((2, 2), True, True, True, split, True, True, reduce_op="sum")

    def test_grouped_units_share_one_scheduler_and_stable_parameters(self):
        for split in (False, True):
            with self.subTest(split=split):
                self._run_policy((2, 2), True, True, True, split, True, True, grouped=True)

    def test_async_unshard_reuses_work_and_completed_parameters(self):
        group = MagicMock(spec=torch.distributed.ProcessGroup)
        group.rank.return_value = 0
        group.size.return_value = 2
        work = MagicMock()

        def all_gather(output, tensor, **kwargs):
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
        self.assertIs(stage._get_trainable_params()[0], ignored_param)
        torch.testing.assert_close(ignored_param.grad, reference.layers[0].weight.grad)
        self.assertIsNone(model.layers[2].weight.grad)

    def test_unsupported_sharding_combinations_fail_during_construction(self):
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
        """PP owns empty-root handling while standalone FSDP retains its original path."""
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
        original_post_backward = self.enterContext(patch.object(
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
            self.assertEqual(context.pre_reduce_scatter_params, model.layers[0].hsdp_scheduler.hsdp_state.hsdp_params)
            reference(inputs.detach()).sum().backward()

        original_post_backward.assert_not_called()
        controller.finalize()
        self.assertEqual(context.pre_reduce_scatter_params, [])
        self.assertIsNone(context.post_backward_handler)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad.to_local(), expected.grad)

        inputs = torch.ones(1, 2, requires_grad=True)
        model(inputs).sum().backward()
        reference(inputs.detach()).sum().backward()
        original_post_backward.assert_called_once_with()
        self.assertEqual(context.pre_reduce_scatter_params, [])
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad.to_local(), expected.grad)
