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
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.modules.moe import GroupedExperts
from hyper_parallel.core.multicore import (
    GateOutput,
    HyperMegaMhc,
    MegaGate,
    MegaMhc,
    MegaMoeExperts,
    MhcState,
)
from hyper_parallel.core.multicore.modules.mega_moe import module as moe_module
from tests.common.mark_utils import arg_mark
from tests.common.multicore.mega_mhc_golden import torch_mega_mhc


class TestFrontendMigration(unittest.TestCase):
    """Check original Parameter ownership and mathematical configuration at replacement boundaries."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_moe_migration_converts_each_expert_without_aliasing_source(self) -> None:
        """Feature: MegaKernel frontend migration.
        Description: Moe migration converts each expert without aliasing source.
        Expectation: Gate/up concatenation and down transposition preserve the source SwiGLU function.
        """
        source = GroupedExperts(16, 8, 4)
        converted = MegaMoeExperts.from_module(source, top_k=2)
        self.addCleanup(converted.close)
        hidden = torch.randn(7, 16)
        for expert in range(4):
            gate, up = (hidden @ converted.gate_up_weight[expert]).chunk(2, -1)
            actual = (torch.nn.functional.silu(gate) * up) @ converted.down_weight[expert]
            expected = torch.nn.functional.linear(
                torch.nn.functional.silu(torch.nn.functional.linear(hidden, source.w1[expert]))
                * torch.nn.functional.linear(hidden, source.w3[expert]), source.w2[expert])
            torch.testing.assert_close(actual, expected)
        self.assertNotEqual(converted.gate_up_weight.data_ptr(), source.w1.data_ptr())
        self.assertIsNone(converted.max_local_num_tokens)
        self.assertIsNone(converted.local_num_tokens)
        source.w3.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "gradient policies"):
            MegaMoeExperts.from_module(source, top_k=2)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_gate_migration_named_result_and_gradient_parity(self) -> None:
        """Feature: MegaKernel frontend migration.
        Description: Gate migration named result and gradient parity.
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
        self.assertIsInstance(actual, GateOutput)
        self.assertEqual(tuple(actual.ids.shape), (6, 2))
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        probe = torch.randn_like(actual.weights)
        first = torch.autograd.grad((actual.weights * probe).sum(), (hidden, target.weight))
        second = torch.autograd.grad((expected.weights * probe).sum(), (hidden, source.weight))
        for got, want in zip(first, second):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        self.assertEqual(target.backend_for(hidden).name, "torch")
        with self.assertRaisesRegex(ValueError, "vision_enabled=True"):
            target(hidden, image_mask=torch.ones(2, 3, dtype=torch.bool))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mhc_migration_keeps_parameters_and_named_state_gradients(self) -> None:
        """Feature: MegaKernel frontend migration.
        Description: Mhc migration keeps parameters and named state gradients.
        Expectation: Named and legacy calls produce identical outputs and all input/parameter gradients.
        """
        source = MegaMhc(128, hc_eps=2e-6, norm_eps=3e-6)
        target = MegaMhc.from_module(source)
        self.assertIs(HyperMegaMhc, MegaMhc)
        for name, parameter in source.named_parameters():
            self.assertIs(getattr(target, name), parameter)
        self.assertEqual(target.hc_eps, source.hc_eps)
        state = MhcState(*(value.detach().requires_grad_()
                           for value in MhcState.initialize(torch.randn(2, 3, 128, dtype=torch.bfloat16))))

        def golden(*args: torch.Tensor, token_tile: int, **kwargs: Any) -> tuple[torch.Tensor, ...]:
            """Use the independent Torch oracle while accepting the execution-only tile option."""
            del token_tile
            return torch_mega_mhc(*args, **kwargs)

        with patch("hyper_parallel.core.multicore.modules.mega_mhc.module.hyper_mega_mhc", side_effect=golden):
            named = target(state)
            legacy = target(*state)
        probe = [torch.randn_like(value) for value in named]
        inputs = (*state, *target.parameters())
        actual = torch.autograd.grad(sum((value * weight).sum() for value, weight in zip(named, probe)), inputs)
        expected = torch.autograd.grad(sum((value * weight).sum() for value, weight in zip(legacy, probe)), inputs)
        for got, want in zip((*named, *actual), (*legacy, *expected)):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        next_state = named.next_state(named.hidden_states.square())
        self.assertIs(next_state.residual, named.residual)
        self.assertIs(state.residual, state[1])
        self.assertEqual(tuple(next_state.finish().shape), (2, 3, 128))
        with self.assertRaisesRegex(ValueError, "either MhcState"):
            target(state, residual=state.residual)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_named_state_bootstrap_preserves_shape_and_input_gradient(self) -> None:
        """Feature: MegaKernel frontend migration.
        Description: Named state bootstrap preserves shape and input gradient.
        Expectation: Independent states retain the original leading dimensions and flush differentiably.
        """
        hidden = torch.randn(2, 3, 128, requires_grad=True)
        first, second = MhcState.initialize(hidden), MhcState.initialize(hidden * 2)
        torch.testing.assert_close(first.finish(), hidden, rtol=0, atol=0)
        torch.testing.assert_close(second.finish(), hidden * 2, rtol=0, atol=0)
        first.finish().sum().backward()
        torch.testing.assert_close(hidden.grad, torch.ones_like(hidden), rtol=0, atol=0)
        self.assertEqual(first.pre_mix.dtype, torch.float32)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_backend_and_mhc_capabilities_do_not_require_native_initialization(self) -> None:
        """Feature: MegaKernel frontend migration.
        Description: Backend and mhc capabilities do not require native initialization.
        Expectation: Queries expose selection and chip limits instead of requiring users to calculate them.
        """
        gate = MegaGate(hidden_size=8, num_experts=4, top_k=2, scoring_func="sqrtsoftplus",
                        routed_scaling_factor=1.0)
        npu_input = SimpleNamespace(device=SimpleNamespace(type="npu"), numel=lambda: 8)
        self.assertEqual(gate.backend_for(npu_input).name, "native")
        self.assertIsNone(MegaMhc.execution_requirements()["minimum_training_tokens"])
        fake_device = SimpleNamespace(type="npu", index=0)
        fake_npu = SimpleNamespace(get_device_limit=Mock(return_value={"cube_core_num": 24, "vector_core_num": 48}))
        with (patch("hyper_parallel.core.multicore.modules.mega_mhc.module.torch.device", return_value=fake_device),
              patch.object(torch, "npu", fake_npu, create=True)):
            self.assertEqual(MegaMhc.execution_requirements("npu:0")["minimum_training_tokens"], 48)
