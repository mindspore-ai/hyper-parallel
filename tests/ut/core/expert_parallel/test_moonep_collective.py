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

"""Single-rank mocked collective checks for the Qwen3 factory integration."""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from hyper_parallel.distributed.expert_parallel.moonep_collective import (
    _assignment_layout,
    _compute_experts,
    _exchange_weights,
    _inverse_permutation,
    collective_expert_forward,
)
from hyper_parallel.distributed.expert_parallel.moonep_collective_plan import (
    build_collective_plan,
)
from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute


class TestCollectiveFactory(unittest.TestCase):
    """Verify the configured path enters expert hooks and preserves gradients."""

    def test_transposed_grouped_weights_and_owner_gradients(self):
        gate = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
        down = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
        experts = SimpleNamespace(gate_up_proj=gate, down_proj=down, is_transposed=True)
        plan = build_collective_plan([[2, 1]])

        def exchange(output, tensor, *_args, **_kwargs):
            output.copy_(tensor)

        with patch("hyper_parallel.distributed.expert_parallel.moonep_collective.dist.all_to_all_single",
                   side_effect=exchange):
            result_gate, result_down, active = _exchange_weights(experts, plan, 0, None, torch.empty(0))
            self.assertEqual(active, (0, 1))
            torch.testing.assert_close(result_gate, gate.transpose(-1, -2))
            torch.testing.assert_close(result_down, down.transpose(-1, -2))
            (result_gate.square().sum() + result_down.square().sum()).backward()
        torch.testing.assert_close(gate.grad, 2 * gate.detach())
        torch.testing.assert_close(down.grad, 2 * down.detach())

    def test_model_gate_semantics_and_gradients(self):
        for custom_gate in (False, True):
            with self.subTest(custom_gate=custom_gate):
                states = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
                gate = torch.randn(2, 6, 2, dtype=torch.float64, requires_grad=True)
                down = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
                apply_gate = lambda x: x.chunk(2, dim=-1)[0].square() * x.chunk(2, dim=-1)[1]
                experts = SimpleNamespace(_ep_use_grouped_gemm=True, _ep_act_fn=torch.nn.functional.gelu,
                                          _ep_apply_gate=apply_gate if custom_gate else None)
                reference = []
                for position, rows in enumerate((states[:1], states[1:])):
                    projected = rows @ gate[position].T
                    left, right = projected.chunk(2, dim=-1)
                    activated = left.square() * right if custom_gate else torch.nn.functional.gelu(left) * right
                    reference.append(activated @ down[position].T)
                expected = torch.cat(reference)
                with patch("hyper_parallel.distributed.expert_parallel.moonep_collective.npu_grouped_swiglu") as fused:
                    actual = _compute_experts(experts, states, gate, down, [1, 2])
                    fused.assert_not_called()
                torch.testing.assert_close(actual, expected)
                actual_grad = torch.autograd.grad(actual.sum(), (states, gate, down), retain_graph=True)
                expected_grad = torch.autograd.grad(expected.sum(), (states, gate, down))
                for result, target in zip(actual_grad, expected_grad):
                    torch.testing.assert_close(result, target)

    def test_factory_hooks_and_router_gradients(self):
        module = nn.Module()
        module.experts = nn.Module()
        module.experts.num_experts = 2
        module.experts.gate_up_proj = nn.Parameter(torch.randn(2, 8, 3))
        module.experts.down_proj = nn.Parameter(torch.randn(2, 3, 4))
        mesh = MagicMock()
        mesh.__getitem__.return_value.size.return_value = 1
        ids = torch.tensor([[0, 1], [1, 0]])
        weights = torch.rand(2, 2, requires_grad=True)
        x = torch.randn(2, 3, requires_grad=True)
        router = MagicMock(return_value=(ids, weights))
        hooks = []
        module.experts.register_forward_pre_hook(lambda *_: hooks.append("pre"))
        module.experts.register_forward_hook(lambda *_: hooks.append("post"))
        compute = build_ep_compute(module, mesh, router_fn=router, archetype_key="test",
                                   expected_attrs=["experts"], combine=lambda _m, _x, y: y,
                                   token_dispatcher="moonep_collective")

        def gather(outputs, tensor, **_kwargs):
            outputs[0].copy_(tensor)

        def exchange(output, tensor, *_args, **_kwargs):
            output.copy_(tensor)

        with patch("hyper_parallel.distributed.expert_parallel.moonep_collective.dist",
                   get_world_size=lambda _g: 1, get_rank=lambda _g: 0,
                   all_gather=gather, all_to_all_single=exchange) as mock_dist, \
                patch("hyper_parallel.distributed.expert_parallel.moonep_collective_dedup.dist", mock_dist), \
                patch("hyper_parallel.distributed.expert_parallel.moonep_collective_packet.dist", mock_dist):
            result = compute(module, x)
            result.sum().backward()
            diagnostic = {}
            collective_expert_forward(module.experts, x, ids[:, 0], group=mesh.get_group("ep"),
                                      diagnostics=diagnostic)
            self.assertEqual(diagnostic["owner_assignments"], [2])
            self.assertEqual(diagnostic["execution_assignments"], [2])
            self.assertEqual(diagnostic["remote_expert_copies"], 0)
            self.assertEqual(diagnostic["weight_forward_bytes"], 0)
            with self.assertRaisesRegex(ValueError, "expert IDs"):
                collective_expert_forward(module.experts, x, torch.tensor([-1, 2]),
                                          group=mesh.get_group("ep"))
        self.assertEqual(hooks, ["pre", "post"])
        router.assert_called_once_with(module, x)
        self.assertEqual(result.shape, x.shape)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weights.grad)
        self.assertIsNotNone(module.experts.gate_up_proj.grad)
        self.assertIsNotNone(module.experts.down_proj.grad)

    def test_invalid_backend_fails_before_binding(self):
        with self.assertRaisesRegex(ValueError, "token_dispatcher"):
            build_ep_compute(SimpleNamespace(), None, router_fn=lambda *_: None,
                             archetype_key="test", expected_attrs=[], combine=lambda *_: None,
                             token_dispatcher="unknown")


class TestAssignmentLayout(unittest.TestCase):
    """Compare batched layouts with independent row-wise routing."""

    def test_layout_matches_stable_assignment_reference(self):
        generator = random.Random(91)
        for size in (1, 2, 4, 8):
            for trial in range(12):
                counts = [[generator.randrange(8) for _ in range(size * 2)] for _ in range(size)]
                if trial == 0:
                    counts = [[0] * (size * 2) for _ in range(size)]
                if trial == 1:
                    counts = [[21] + [0] * (size * 2 - 1) for _ in range(size)]
                plan = build_collective_plan(counts)
                for rank in range(size):
                    with self.subTest(size=size, trial=trial, rank=rank):
                        ids = [expert for expert, count in enumerate(counts[rank]) for _ in range(count)]
                        generator.shuffle(ids)
                        expert_order = torch.argsort(torch.tensor(ids, dtype=torch.int64), stable=True)
                        order, sorting, actual_counts = _assignment_layout(plan, rank, expert_order)
                        offsets = [sum(counts[rank][:expert]) for expert in range(size * 2)]
                        positions, received_ids = [], []
                        for peer in range(size):
                            for segment in plan.segments:
                                if segment.source == rank and segment.destination == peer:
                                    start = offsets[segment.expert] + segment.offset
                                    positions.extend(range(start, start + segment.count))
                                if segment.source == peer and segment.destination == rank:
                                    received_ids.extend([segment.expert] * segment.count)
                        expected_order = expert_order[torch.tensor(positions, dtype=torch.int64)]
                        expected_sort = torch.tensor(sorted(range(len(received_ids)),
                                                            key=received_ids.__getitem__), dtype=torch.int64)
                        torch.testing.assert_close(order, expected_order)
                        torch.testing.assert_close(sorting, expected_sort)
                        self.assertEqual(actual_counts, [received_ids.count(expert)
                                                        for expert in sorted(set(received_ids))])
                        for permutation in (order, sorting):
                            inverse = _inverse_permutation(permutation)
                            torch.testing.assert_close(permutation[inverse], torch.arange(permutation.numel()))

    def test_remote_weight_packing_omits_inactive_local_experts(self):
        gate = torch.randn(16, 8, 3, dtype=torch.float64, requires_grad=True)
        down = torch.randn(16, 3, 4, dtype=torch.float64, requires_grad=True)
        experts = SimpleNamespace(gate_up_proj=gate, down_proj=down)
        plan = build_collective_plan([[3] + [0] * 15])
        sent_shapes = []

        def exchange(output, tensor, *_args, **_kwargs):
            sent_shapes.append(tuple(tensor.shape))
            output.copy_(tensor)

        with patch('hyper_parallel.distributed.expert_parallel.moonep_collective.dist.all_to_all_single',
                   side_effect=exchange):
            received_gate, received_down, active = _exchange_weights(experts, plan, 0, None, torch.empty(0))
            (received_gate.sum() + received_down.sum()).backward()
        self.assertEqual(active, (0,))
        self.assertEqual(sent_shapes, [(0, 36), (0, 36)])
        torch.testing.assert_close(gate.grad[0], torch.ones_like(gate[0]))
        torch.testing.assert_close(down.grad[0], torch.ones_like(down[0]))
        torch.testing.assert_close(gate.grad[1:], torch.zeros_like(gate[1:]))
        torch.testing.assert_close(down.grad[1:], torch.zeros_like(down[1:]))
