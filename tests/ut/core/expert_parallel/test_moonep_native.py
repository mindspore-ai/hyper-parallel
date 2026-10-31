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
"""Mock-only native MoonEP routing, ownership and lazy-gradient contracts."""
from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.core.expert_parallel.moonep import native, precision
from hyper_parallel.core.expert_parallel.moonep.placement import ExpertPlacement
from hyper_parallel.distributed.expert_parallel import experts as execution
from hyper_parallel.models.deepseek_v41.adapter.distributed import moe_engram_expert_parallel as v41


class _Experts(nn.Module):
    """Use the real bound native experts body with small stacked projections."""

    def __init__(self, rank: int) -> None:
        """Initialize deterministic expert-major source weights."""
        super().__init__()
        self.num_experts = 4
        self.hidden_size = 3
        self.intermediate_size = 2
        generator = torch.Generator().manual_seed(902)
        gate = torch.randn(4, 4, 3, generator=generator) * 0.2
        down = torch.randn(4, 3, 2, generator=generator) * 0.2
        self.gate_up_proj = nn.Parameter(gate[rank * 2:(rank + 1) * 2].clone())
        self.down_proj = nn.Parameter(down[rank * 2:(rank + 1) * 2].clone())


class TestNativeMoonEP(unittest.TestCase):
    """Cover backend-independent ownership and routing invariants on CPU."""


    def test_parameters_are_retrieved_for_each_expert_forward(self) -> None:
        """
        Feature: Current unsharded parameter lookup.
        Description: Invoke the same immutable placement after changing current parameter values.
        Expectation: No cached autograd tensor bypasses the real expert forward boundary.
        """
        experts = _Experts(0)
        plan = ExpertPlacement(torch.zeros(1, 1), torch.zeros(2, 8), torch.zeros(2, 2),
                               ((-1, -1), (0, -1)))
        invocation = native.NativeExpertPlacement(plan, None, 2)
        with patch.object(native, 'borrow_parameters',
                          side_effect=lambda weights, *_: (tuple(weight * 2 for weight in weights),
                                                          (None,) * len(weights))):
            first = invocation.parameters(experts)[0]['gate_up_proj']
            with torch.no_grad():
                experts.gate_up_proj.add_(1)
            second = invocation.parameters(experts)[0]['gate_up_proj']
        torch.testing.assert_close(second - first, torch.full_like(first, 2))
        self.assertIsNot(first.grad_fn, second.grad_fn)

    def test_disabled_dispatch_never_enters_moon_planner(self) -> None:
        """
        Feature: Opt-in native adapter.
        Description: Prepare the ordinary dispatch with MoonEP disabled.
        Expectation: Existing count exchange runs once and no Moon planner or weight access occurs.
        """
        with patch.object(execution, '_select_moonep_route', side_effect=AssertionError('unexpected MoonEP')), \
             patch.object(execution, 'get_static_plan', return_value=None), \
             patch.object(execution, 'store_static_plan'), \
             patch.object(execution, 'ep_exchange_counts', side_effect=lambda counts, *_: counts) as exchange:
            dispatch = execution._prepare_ep_dispatch(
                torch.ones(1, 2, 3), torch.tensor([[0], [2]]), torch.ones(2, 1),
                local_expert_count=2, global_expert_count=4, ep_size=2, ep_group=None)
        self.assertIsNone(dispatch.moon_selection)
        self.assertEqual(exchange.call_count, 1)

    def test_balanced_uses_gathered_counts_without_second_exchange(self) -> None:
        """
        Feature: Balanced native fallback.
        Description: Supply a balanced global table through the opt-in gather.
        Expectation: Native weights remain untouched and both count directions come from that table.
        """
        table = torch.tensor([[1, 0, 1, 0], [0, 2, 0, 2]])

        def gather(output: torch.Tensor, *_args: object, **_kwargs: object) -> None:
            """Publish the complete balanced source-count table."""
            output.copy_(table.reshape(-1))

        with patch.object(native.dist, 'get_world_size', return_value=2), \
             patch.object(native.dist, 'get_rank', return_value=0), \
             patch.object(native.dist, 'all_gather_into_tensor', side_effect=gather) as exchange, \
             patch.object(native, 'build_placement', side_effect=AssertionError('balanced placement')):
            selection = native.prepare_native_placement(
                torch.zeros(2, 3), torch.tensor([[0], [2]]), table[0], ep_group=None,
                hidden_size=3, intermediate_size=2, cube_cores=24, weight_bytes=96)
        self.assertIsNone(selection.invocation)
        self.assertEqual(selection.send_counts, [1, 1])
        self.assertEqual(selection.receive_counts, [1, 2])
        self.assertEqual(exchange.call_count, 1)

    def test_allocator_extent_rejects_before_narrowing(self) -> None:
        """
        Feature: INT32 allocator wire extent.
        Description: Gather an INT64 count table whose global row count exceeds INT32.
        Expectation: All ranks reject explicitly before truncating counts or entering the allocator.
        """
        table = torch.tensor([[1, 0, 0, 0], [2**31, 0, 0, 0]])
        with patch.object(native.dist, 'get_world_size', return_value=2), \
             patch.object(native.dist, 'get_rank', return_value=0), \
             patch.object(native.dist, 'all_gather_into_tensor',
                          side_effect=lambda output, *_args, **_kwargs: output.copy_(table.reshape(-1))), \
             patch.object(native, 'build_placement') as allocator:
            with self.assertRaisesRegex(ValueError, 'INT32'):
                native.prepare_native_placement(
                    torch.zeros(1, 3), torch.zeros(1, 1, dtype=torch.int64), table[0], ep_group=None,
                    hidden_size=3, intermediate_size=2, cube_cores=24, weight_bytes=96)
        allocator.assert_not_called()

    def test_cost_uses_local_geometry_and_fsdp_compute_dtype(self) -> None:
        """
        Feature: Native planner metadata.
        Description: Supply FP32 master metadata with BF16 FSDP compute and ungated biased experts.
        Expectation: Cost uses the home shard and compute dtype without reading unsharded tensor values.
        """
        experts = SimpleNamespace(
            local_expert_count=2, intermediate_size=128, has_gate=False, add_bias=True,
            hsdp_scheduler=SimpleNamespace(mp_policy=SimpleNamespace(param_dtype=torch.bfloat16)),
            parameters=lambda **_: [SimpleNamespace(dtype=torch.float32)])
        with patch.object(native, 'prepare_native_placement') as prepare, \
             patch.object(execution.torch, 'npu', SimpleNamespace(
                 get_device_limit=lambda _: {'cube_core_num': 24}), create=True):
            execution._select_moonep_route(
                experts, torch.zeros(2, 64), torch.tensor([[0], [1]]), 4, None)
        self.assertEqual(prepare.call_args.kwargs['weight_bytes'], 2 * (64 * 128 * 2 + 128 + 64) * 2)
        self.assertEqual(prepare.call_args.kwargs['projection_flops'], 4)

    def test_v41_factory_exposes_native_moon_flag_without_parameter_replacement(self) -> None:
        """
        Feature: User-configurable V4.1 native MoE adapter.
        Description: Resolve the real factory and call its source-compatible local compute entry.
        Expectation: The independent Moon flag is forwarded while registered parameter identities stay unchanged.
        """
        module = nn.Module()
        module.experts = _Experts(0)
        module.experts.limit = 0.0
        module.gate = nn.Linear(3, 4)
        module.shared_experts = nn.Identity()
        module.is_hash = False
        identity = tuple((name, id(parameter)) for name, parameter in module.named_parameters())
        mesh = type('Mesh', (), {
            '__getitem__': lambda self, _: SimpleNamespace(size=lambda: 2),
            'get_group': lambda self, *_: None})()
        with patch.object(v41, '_configure_router_statistics', return_value=()), \
             patch.object(v41, '_routed_and_shared_forward', return_value=torch.ones(1, 2, 3)) as routed:
            compute = v41.deepseek_v41_ep_compute_fn(
                module=module, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=mesh, enable_moonep=True)
            compute(module, torch.zeros(1, 2, 3))
        self.assertTrue(routed.call_args.kwargs['enable_moonep'])
        self.assertEqual(identity, tuple((name, id(parameter)) for name, parameter in module.named_parameters()))

    def test_disabled_pending_state_retains_public_tuple_contract(self) -> None:
        """
        Feature: Ordinary native EP pending-state compatibility.
        Description: Construct the public state using its original eight required arguments.
        Expectation: Length, fields, dictionary keys and nine-value unpack remain unchanged.
        """
        tensor = torch.empty(0)
        state = execution.EPRoutedState(tensor, tensor, tensor, None, None, [], [], (1, 0, 3))
        expected = ('source_token_indices', 'flattened_expert_weights', 'dispatch_order',
                    'received_states', 'received_indices', 'send_counts', 'receive_counts',
                    'output_shape', 'chunks')
        self.assertEqual(len(state), 9)
        self.assertEqual(state._fields, expected)
        self.assertEqual(tuple(state._asdict()), expected)
        first, second, third, fourth, fifth, sixth, seventh, eighth, ninth = state
        self.assertEqual((first, second, third, fourth, fifth, sixth, seventh, eighth, ninth), tuple(state))


class TestNativeGradientScratch(unittest.TestCase):
    """Check gradients and resource cleanup without a real process group."""

    def test_frozen_bias_preserves_dx_without_allocating_scratch(self):
        """An input-only backward must not create an unused bias accumulator."""
        inputs = torch.randn(2, 3, requires_grad=True)
        bias = torch.randn(2, 3)
        target = precision.GradientTarget(tuple(bias.shape), bias.device, requires_grad=False)

        target.add_bias(inputs, bias, torch.tensor([2, 0])).sum().backward()

        torch.testing.assert_close(inputs.grad, torch.ones_like(inputs), rtol=0, atol=0)
        self.assertIsNone(bias.grad)
        self.assertIsNone(target.buffer)

    def test_bias_only_backward_preserves_fp32_reduction(self):
        """Frozen input still gives the same bias gradient and no input gradient."""
        inputs = torch.randn(3, 4)
        bias = torch.randn(2, 4, requires_grad=True)
        target = precision.GradientTarget(tuple(bias.shape), bias.device)
        gradient = torch.randn_like(inputs)

        target.add_bias(inputs, bias, torch.tensor([2, 1])).backward(gradient)

        expected = torch.stack((gradient[:2].sum(0), gradient[2]))
        torch.testing.assert_close(target.buffer, expected)
        self.assertIsNone(inputs.grad)

    def test_weight_only_linear_skips_dx(self):
        """The forward remains unchanged when backward only needs weight gradients."""
        inputs = torch.randn(3, 4)
        weights = torch.randn(2, 5, 4, requires_grad=True)
        target = precision.GradientTarget(tuple(weights.shape), weights.device)

        def add_weight_gradient(output: torch.Tensor, lhs: torch.Tensor, rhs: torch.Tensor,
                                _bounds: torch.Tensor, **_kwargs: object) -> None:
            """Emulate the accumulator kernel without an accelerator backend."""
            output.add_(lhs.transpose(0, 1) @ rhs)

        npu = SimpleNamespace(npu_grouped_matmul_add_=add_weight_gradient)
        with patch.dict(sys.modules, {'torch_npu': npu}), \
                patch.object(precision.functional, 'linear', wraps=torch.nn.functional.linear) as linear:
            precision.expert_linear(inputs, weights, 0, target).sum().backward()

        self.assertEqual(linear.call_count, 1)
        torch.testing.assert_close(target.gradient(weights)[0], inputs.sum(0).expand(5, 4))
        self.assertIsNone(inputs.grad)

    @patch.object(precision, 'dist')
    def test_failed_owner_exchange_releases_scratch_and_retained_graph_recovers(self, mock_dist):
        """A failed exchange cannot leak its partial gradient into a later backward."""
        mock_dist.get_rank.return_value = 0
        mock_dist.get_world_size.return_value = 1
        weight = torch.randn(1, 2, 3, dtype=torch.bfloat16, requires_grad=True)

        def exchange(tensor: torch.Tensor, *_args: object) -> torch.Tensor:
            """Return a differentiable local stand-in for one-rank transport."""
            return tensor.clone()

        with patch.object(precision, 'differentiable_all_to_all_single', side_effect=exchange):
            outputs, targets = precision.borrow_parameters((weight,), ((0,),), object())
            targets[0].main_grad.fill_(9)
            loss = outputs[0].sum()
            with patch.object(precision, 'differentiable_all_to_all_single', side_effect=RuntimeError('owner failure')):
                with self.assertRaisesRegex(RuntimeError, 'owner failure'):
                    loss.backward(retain_graph=True)
            self.assertIsNone(targets[0].buffer)
            self.assertIsNone(weight.grad)

            loss.backward(retain_graph=True)
            first = weight.grad.clone()
            loss.backward()

        torch.testing.assert_close(first, torch.full_like(weight, 2), rtol=0, atol=0)
        torch.testing.assert_close(weight.grad, 2 * first, rtol=0, atol=0)
        self.assertIsNone(targets[0].buffer)
