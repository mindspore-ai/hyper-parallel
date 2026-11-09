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
"""CPU checks for model-facing MegaKernel migration, named state and precision policies."""

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from hyper_parallel.components.modules.moe import GroupedExperts
from hyper_parallel.core.multicore import MegaGate, MegaMhc, MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_gate import module as gate_module
from hyper_parallel.core.multicore.modules.mega_moe import module as moe_module
from tests.common.multicore.mega_mhc_golden import torch_mega_mhc


class TestFrontendMigration(unittest.TestCase):
    """Check original Parameter ownership and mathematical configuration at replacement boundaries."""

    def test_ep_size_follows_explicit_subgroup_and_rejects_implicit_world(self) -> None:
        """Feature: MegaKernel frontend migration.
        Description: Ep size follows explicit subgroup and rejects implicit world.
        Expectation: A subgroup determines local expert count; a global process group is never guessed.
        """
        group = object()
        with (patch.object(moe_module.dist, "is_initialized", return_value=True),
              patch.object(moe_module.dist, "get_world_size", return_value=2),
              patch.object(moe_module.shmem, "root_members", return_value=(2, 3))):
            layer = MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2, ep_group=group)
            self.addCleanup(layer.close)
            self.assertEqual(layer.ep_size, 2)
            self.assertEqual(layer.local_experts, 2)
            with self.assertRaisesRegex(ValueError, "does not match"):
                MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2,
                               ep_group=group, ep_size=4)
            with self.assertRaisesRegex(ValueError, "ep_group explicitly"):
                MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2)

    def test_moe_migration_converts_each_expert_without_aliasing_source(self) -> None:
        """Feature: MegaKernel frontend migration.
        Description: Convert each expert to EIO and EOI without aliasing noncontiguous source weights.
        Expectation: Both contiguous native layouts preserve the source SwiGLU function.
        """
        source = GroupedExperts(16, 8, 4)
        source.w2 = torch.nn.Parameter(source.w2.transpose(1, 2).contiguous().transpose(1, 2))
        hidden = torch.randn(7, 16)
        for layout in ("EIO", "EOI"):
            with self.subTest(layout=layout):
                converted = MegaMoeExperts.from_module(source, top_k=2, expert_weight_layout=layout)
                self.addCleanup(converted.close)
                for expert in range(4):
                    weight1, weight2 = converted.gate_up_weight[expert], converted.down_weight[expert]
                    if layout == "EOI":
                        weight1, weight2 = weight1.T, weight2.T
                    gate, up = (hidden @ weight1).chunk(2, -1)
                    actual = (torch.nn.functional.silu(gate) * up) @ weight2
                    expected = torch.nn.functional.linear(
                        torch.nn.functional.silu(torch.nn.functional.linear(hidden, source.w1[expert]))
                        * torch.nn.functional.linear(hidden, source.w3[expert]), source.w2[expert])
                    torch.testing.assert_close(actual, expected)
                self.assertNotEqual(converted.gate_up_weight.data_ptr(), source.w1.data_ptr())
                self.assertNotEqual(converted.down_weight.data_ptr(), source.w2.data_ptr())
                self.assertTrue(converted.gate_up_weight.is_contiguous())
                self.assertTrue(converted.down_weight.is_contiguous())
        source.w3.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "gradient policies"):
            MegaMoeExperts.from_module(source, top_k=2)
        with self.assertRaisesRegex(ValueError, "materialize and initialize"):
            MegaMoeExperts.from_module(source.to("meta"), top_k=2)

    def test_gate_migration_and_gradient_parity(self) -> None:
        """Feature: MegaKernel frontend migration.
        Description: Gate migration and gradient parity.
        Expectation: A migrated router keeps Parameters and scoring instead of initializing a new gate.
        """
        source = MegaGate(hidden_size=8, num_experts=4, top_k=2, scoring_func="sigmoid",
                          routed_scaling_factor=1.7)
        source.bias.requires_grad_(False)
        target = MegaGate.from_module(source)
        self.assertIs(target.weight, source.weight)
        self.assertIs(target.bias, source.bias)
        hidden = torch.randn(2, 3, 8, requires_grad=True)
        actual, expected = target(hidden), source(hidden)
        self.assertIs(type(actual), tuple)
        self.assertEqual(tuple(actual[2].shape), (6, 2))
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        probe = torch.randn_like(actual[1])
        first = torch.autograd.grad((actual[1] * probe).sum(), (hidden, target.weight))
        second = torch.autograd.grad((expected[1] * probe).sum(), (hidden, source.weight))
        for got, want in zip(first, second):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "vision_enabled=True"):
            target(hidden, image_mask=torch.ones(2, 3, dtype=torch.bool))
        for source_vision in (False, True):
            original = MegaGate(hidden_size=8, num_experts=4, top_k=2, scoring_func="sigmoid",
                                routed_scaling_factor=1.0, vision_enabled=source_vision)
            for configured in (None, False, True):
                options = {} if configured is None else {"vision_enabled": configured}
                with self.subTest(source_vision=source_vision, configured=configured):
                    if source_vision and configured is False:
                        with self.assertRaisesRegex(ValueError, "source bias_vl"):
                            MegaGate.from_module(original, **options)
                        continue
                    migrated = MegaGate.from_module(original, **options)
                    self.assertIs(migrated.weight, original.weight)
                    self.assertEqual(migrated.bias_vl is not None,
                                     source_vision if configured is None else configured)
                    if source_vision:
                        self.assertIs(migrated.bias_vl, original.bias_vl)

    def test_dtype_conversion_preserves_control_values_and_existing_gradients(self) -> None:
        """Feature: MegaKernel frontend migration.
        Description: Dtype conversion preserves control values and existing gradients.
        Expectation: Global BF16 conversion must never round an FP32 predictor and then cast it back.
        """
        model = torch.nn.Module()
        model.mhc = MegaMhc(128)
        model.gate = MegaGate(hidden_size=128, num_experts=4, top_k=2, scoring_func="sqrtsoftplus",
                              routed_scaling_factor=1.0, vision_enabled=True)
        protected = [model.mhc.phi, model.mhc.alpha, model.mhc.bias, model.gate.bias, model.gate.bias_vl]
        originals = []
        with torch.no_grad():
            for parameter in protected:
                parameter.copy_(torch.randn_like(parameter))
                parameter.grad = torch.randn_like(parameter)
                originals.append((parameter.detach().clone(), parameter.grad.clone()))
        model.bfloat16()
        self.assertEqual(model.mhc.norm_weight.dtype, torch.bfloat16)
        self.assertEqual(model.gate.weight.dtype, torch.bfloat16)
        for parameter, (value, gradient) in zip(protected, originals):
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertEqual(parameter.grad.dtype, torch.float32)
            torch.testing.assert_close(parameter, value, rtol=0, atol=0)
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)

    def test_mhc_migration_keeps_parameters_and_gradients(self) -> None:
        """Feature: MegaKernel frontend migration.
        Description: Compare migrated and source modules with the same five input tensors.
        Expectation: Preserve Parameter identities, outputs and every input/parameter gradient.
        """
        source = MegaMhc(128, hc_eps=2e-6, norm_eps=3e-6)
        target = MegaMhc.from_module(source)
        self.assertIs(type(target), MegaMhc)
        for name, parameter in source.named_parameters():
            self.assertIs(getattr(target, name), parameter)
        self.assertEqual(target.hc_eps, source.hc_eps)
        state = (torch.randn(2, 3, 128, dtype=torch.bfloat16), torch.randn(2, 3, 4, 128, dtype=torch.bfloat16),
                 torch.rand(2, 3, 4), torch.rand(2, 3, 4), torch.rand(2, 3, 4, 4))
        state = tuple(value.requires_grad_() for value in state)

        def golden(*args: torch.Tensor, token_tile: int, **kwargs: Any) -> tuple[torch.Tensor, ...]:
            """Use the independent Torch oracle while accepting the execution-only tile option."""
            del token_tile
            return torch_mega_mhc(*args, **kwargs)

        with patch("hyper_parallel.core.multicore.modules.mega_mhc.module.hyper_mega_mhc", side_effect=golden):
            migrated = target(*state)
            original = source(*state)
        self.assertIs(type(migrated), tuple)
        probe = [torch.randn_like(value) for value in migrated]
        inputs = (*state, *target.parameters())
        actual = torch.autograd.grad(sum((value * weight).sum() for value, weight in zip(migrated, probe)), inputs)
        expected = torch.autograd.grad(sum((value * weight).sum() for value, weight in zip(original, probe)), inputs)
        for got, want in zip((*migrated, *actual), (*original, *expected)):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_gate_dispatch_preserves_scoring_device_and_empty_input_rules(self) -> None:
        """Feature: MegaGate implementation selection.
        Description: Exercise forward selection without loading native plans or running NPU operations.
        Expectation: Only nonempty NPU sqrtsoftplus uses native Route; all other inputs use Torch.
        """
        result = (torch.ones(1, 4), torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.int64))
        cases = (("sqrtsoftplus", "npu", 8, True), ("sqrtsoftplus", "npu", 0, False),
                 ("sqrtsoftplus", "cpu", 8, False), ("softmax", "npu", 8, False),
                 ("sigmoid", "npu", 8, False))
        for scoring, device, elements, native in cases:
            with self.subTest(scoring=scoring, device=device, elements=elements):
                gate = MegaGate(hidden_size=8, num_experts=4, top_k=2, scoring_func=scoring, routed_scaling_factor=1.0)
                hidden = SimpleNamespace(device=SimpleNamespace(type=device), numel=lambda count=elements: count)
                with (patch.object(gate, "_torch_forward", return_value=result) as ordinary,
                      patch.object(gate, "_plan") as plan,
                      patch.object(gate_module, "mega_gate", return_value=result) as fused):
                    self.assertIs(gate(hidden), result)
                self.assertEqual(fused.call_count, int(native))
                self.assertEqual(plan.call_count, int(native))
                self.assertEqual(ordinary.call_count, int(not native))
