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
"""DSV4.1 balances benchmark destinations while preserving learned routing weights."""

import copy
from itertools import product
from types import SimpleNamespace
from typing import Callable
import unittest
from unittest.mock import MagicMock, patch

import torch
import yaml

from hyper_parallel.distributed._builder.rule_resolver import _resolve_local_compute_fn
from hyper_parallel.models.deepseek_v41.adapter.distributed import moe_engram_expert_parallel as ep_adapter
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41TopKRouter
from hyper_parallel.trainer.config.parallelism import PlanOverride, entries_to_plan_overrides
from hyper_parallel.trainer.config.resolver import resolve_component
from tests.common.mark_utils import arg_mark


def _make_module(scoring: str, dtype: torch.dtype, multimodal: bool) -> SimpleNamespace:
    """Keep the real gate and replace only expert transport with CPU stand-ins."""
    gate = DeepseekV41TopKRouter(SimpleNamespace(
        hidden_size=4, num_local_experts=8, num_experts_per_tok=2,
        scoring_func=scoring, routed_scaling_factor=1.25, v41_vision_enabled=multimodal,
    )).to(dtype=dtype)
    with torch.no_grad():
        values = torch.arange(32).reshape(8, 4).float() * 0.04 - 0.5
        gate.weight.copy_(values)
        gate.bias.copy_(torch.tensor([10., 10., 0., 0., 0., 0., 0., 0.]))
        if multimodal:
            gate.bias_vl.copy_(gate.bias.flip(0))
    module = SimpleNamespace(
        gate=gate, experts=SimpleNamespace(limit=0.0, num_experts=8, local_expert_count=2),
        shared_experts=lambda states: states * 0.5, is_hash=False,
        forward=lambda hidden_states, input_ids=None: hidden_states,
    )
    if multimodal:
        module.forward = lambda hidden_states, input_ids=None, image_mask=None: hidden_states
    return module


class TestDeepseekV41FixedRouter(unittest.TestCase):
    """Check destination balancing, native gate gradients, and local image masks."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_fixed_destinations_keep_native_weights_and_gradients(self) -> None:
        """
        Feature: DSV4.1 framework-balanced router.
        Description: Exercise native scores and gradients for FP32/BF16 text and VLM.
        Expectation: Destinations balance while native weights and gradients are preserved.
        """
        for scoring, dtype, multimodal in product(
                ("sqrtsoftplus", "sigmoid", "softmax"), (torch.float32, torch.bfloat16), (False, True)):
            with self.subTest(scoring=scoring, dtype=dtype, multimodal=multimodal):
                self._check_forward_backward(scoring, dtype, multimodal, fixed=True)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_disabled_switch_keeps_native_routing(self) -> None:
        """
        Feature: DSV4.1 learned routing compatibility.
        Description: Execute text and image contracts with the benchmark switch disabled.
        Expectation: Native expert selection, weights and gradients remain unchanged.
        """
        for multimodal in (False, True):
            with self.subTest(multimodal=multimodal):
                self._check_forward_backward("sqrtsoftplus", torch.float32, multimodal, fixed=False)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_yaml_override_transports_both_switches(self) -> None:
        """
        Feature: DSV4.1 typed EP configuration.
        Description: Resolve learned/balanced and eager/grouped YAML for text and VLM.
        Expectation: Both switches reach the factory and routing retains its source signature.
        """
        for grouped, multimodal, fixed in product((False, True), (False, True), (False, True)):
            with self.subTest(grouped=grouped, multimodal=multimodal, fixed=fixed):
                raw = yaml.safe_load(f"""
match: '*.mlp'
when: ep
region_dispatch: false
local_compute_fn:
  _target_: {ep_adapter.__name__}.deepseek_v41_ep_compute_fn
  use_grouped_gemm: {str(grouped).lower()}
  fix_router: {str(fixed).lower()}
""")
                entry = resolve_component(raw, annotation=PlanOverride, path="$.plan_overrides[0]")
                spec = entries_to_plan_overrides([entry], ep_size=4)["*.mlp"]
                self.assertIs(spec.local_compute_fn.fix_router, fixed)
                self.assertIs(spec.local_compute_fn.use_grouped_gemm, grouped)
                self.assertEqual(entries_to_plan_overrides([entry], ep_size=1), {})
                module = _make_module("sqrtsoftplus", torch.float32, multimodal)
                ep_mesh = MagicMock()
                ep_mesh.__getitem__.return_value.size.return_value = 4
                states = torch.ones(1, 10, 4)
                image_mask = torch.zeros(1, 10, dtype=torch.bool) if multimodal else None
                with patch.object(ep_adapter, "bind_local_expert_forward") as bind, patch.object(
                    ep_adapter, "_routed_and_shared_forward", return_value=states,
                ) as routed:
                    compute = _resolve_local_compute_fn(
                        module, spec, mesh=None, mesh_dim_names=(), expert_mesh=ep_mesh,
                    )
                    actual = compute(states, image_mask=image_mask) if multimodal else compute(states)
                self.assertIs(actual, states)
                self.assertIs(bind.call_args.kwargs["use_grouped_gemm"], grouped)
                routed.assert_called_once_with(
                    module, states, image_mask, ep_mesh.get_group.return_value, fix_router=fixed,
                )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_fixed_router_keeps_tp_local_image_mask(self) -> None:
        """
        Feature: DSV4.1 fixed routing with TP sequence slicing.
        Description: Route both TP intervals using a CP-local image mask.
        Expectation: The native gate sees its TP-local mask with either router setting.
        """
        def dispatch(
                target: SimpleNamespace,
                hidden: torch.Tensor,
                *,
                router_fn: Callable,
                ep_group: object,
        ) -> torch.Tensor:
            """Keep native routing weights observable after mask slicing."""
            del ep_group
            return router_fn(target, hidden)[1].sum() * hidden

        for tp_rank, fixed in product((0, 1), (False, True)):
            with self.subTest(tp_rank=tp_rank, fixed=fixed):
                module = _make_module("sqrtsoftplus", torch.float32, True)
                states = torch.ones(1, 5, 4)
                images = torch.tensor([[False, True, False, True, False, True, True, False, False, True]])
                local_images = images[:, tp_rank * 5:(tp_rank + 1) * 5]
                expected = module.gate(states, image_mask=local_images)
                tp_mesh = MagicMock()
                tp_mesh.size.return_value = 2
                tp_mesh.get_local_rank.return_value = tp_rank
                ep_mesh = MagicMock()
                ep_mesh.__getitem__.return_value.size.return_value = 4
                with patch.object(ep_adapter, "bind_local_expert_forward"), patch.object(
                    ep_adapter, "ep_routed_forward", side_effect=dispatch,
                ), patch.object(module.gate, "forward", wraps=module.gate.forward) as gate_forward:
                    compute = ep_adapter.deepseek_v41_ep_compute_fn(
                        module=module, mesh=None, tp_mesh=tp_mesh, cp_mesh=None,
                        ep_mesh=ep_mesh, fix_router=fixed,
                    )
                    actual = compute(module, states, image_mask=images)
                gate_forward.assert_called_once()
                torch.testing.assert_close(gate_forward.call_args.kwargs["image_mask"], local_images)
                torch.testing.assert_close(actual, expected[1].sum() * states + states * 0.5)

    def _check_forward_backward(self, scoring: str, dtype: torch.dtype, multimodal: bool, fixed: bool) -> None:
        """Mock expert transport while exercising the real factory and gate."""
        module = _make_module(scoring, dtype, multimodal)
        reference = copy.deepcopy(module.gate)
        parameters = dict(module.gate.named_parameters())
        checkpoint_keys = set(module.gate.state_dict())
        states = torch.linspace(-1, 1, 40).reshape(1, 10, 4).to(dtype).requires_grad_()
        expected_states = states.detach().clone().requires_grad_()
        image_mask = torch.arange(10).reshape(1, 10) % 3 == 0 if multimodal else None
        local_mask = image_mask
        captured = {}

        def dispatch(
                target: SimpleNamespace,
                inputs: torch.Tensor,
                *,
                router_fn: Callable,
                ep_group: object,
        ) -> torch.Tensor:
            """Make routed output depend on both expert identities and gate weights."""
            del ep_group
            indices, weights = router_fn(target, inputs)
            captured.update(indices=indices, weights=weights)
            factors = (weights * (indices + 1)).sum(-1).reshape(1, 10, 1)
            return inputs * factors

        ep_mesh = MagicMock()
        ep_mesh.__getitem__.return_value.size.return_value = 4
        ep_mesh.get_group.return_value = "ep-group"
        with patch.object(ep_adapter, "bind_local_expert_forward"), patch.object(
            ep_adapter, "ep_routed_forward", side_effect=dispatch,
        ), patch.object(module.gate, "forward", wraps=module.gate.forward) as gate_forward:
            compute = ep_adapter.deepseek_v41_ep_compute_fn(
                module=module, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=ep_mesh, fix_router=fixed,
            )
            actual = compute(module, states, image_mask=image_mask) if multimodal else compute(module, states)
            gate_forward.assert_called_once()
            if multimodal:
                torch.testing.assert_close(gate_forward.call_args.kwargs["image_mask"], local_mask)

        _, expected_weights, learned_indices = reference(expected_states, image_mask=local_mask)
        expected_indices = torch.tensor([
            [0, 2], [4, 6], [1, 3], [5, 7], [0, 2],
            [4, 6], [1, 3], [5, 7], [0, 2], [4, 6],
        ]) if fixed else learned_indices
        expected = expected_states * (
            (expected_weights * (expected_indices + 1)).sum(-1).reshape(1, 10, 1)
        ) + expected_states * 0.5
        torch.testing.assert_close(captured["indices"], expected_indices)
        torch.testing.assert_close(captured["weights"], expected_weights)
        torch.testing.assert_close(actual, expected)
        actual_gradients = torch.autograd.grad(actual.square().sum(), (states, module.gate.weight))
        expected_gradients = torch.autograd.grad(expected.square().sum(), (expected_states, reference.weight))
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertGreater(actual_gradients[1].abs().sum().item(), 0)
        self.assertEqual(set(module.gate.state_dict()), checkpoint_keys)
        self.assertFalse(hasattr(module.gate, "fix_router"))
        for name, parameter in module.gate.named_parameters():
            self.assertIs(parameter, parameters[name])
        if fixed:
            destinations = captured["indices"].flatten() // 2
            torch.testing.assert_close(torch.bincount(destinations, minlength=4), torch.full((4,), 5))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_destination_cycles_cover_partial_and_empty_inputs(self) -> None:
        """
        Feature: DSV4.1 fixed routing across expert-parallel layouts.
        Description: Exercise empty inputs, partial destination cycles and the 384-expert layout.
        Expectation: Expert order follows destination-first cycles and peer loads differ by at most one.
        """
        for experts, peers, top_k, tokens in ((8, 4, 2, 0), (8, 4, 2, 5), (8, 4, 6, 3),
                                            (384, 128, 6, 31), (4, 1, 1, 3)):
            with self.subTest(experts=experts, peers=peers, top_k=top_k, tokens=tokens):
                local_count = experts // peers
                module = SimpleNamespace(
                    gate=SimpleNamespace(num_experts=experts),
                    experts=SimpleNamespace(local_expert_count=local_count),
                )
                weights = torch.rand(tokens, top_k, requires_grad=True)
                real_router = MagicMock(return_value=(torch.zeros(tokens, top_k, dtype=torch.int32), weights))
                states = torch.zeros(tokens, 4)
                indices, actual_weights = ep_adapter._resolve_router(real_router, fix_router=True)(module, states)
                cycle = [rank * local_count + offset for offset in range(local_count) for rank in range(peers)]
                expected = torch.tensor(
                    [cycle[slot % experts] for slot in range(tokens * top_k)], dtype=torch.int32,
                ).view(tokens, top_k)
                torch.testing.assert_close(indices, expected)
                self.assertIs(actual_weights, weights)
                real_router.assert_called_once_with(module, states)
                counts = torch.bincount(indices.flatten().long() // local_count, minlength=peers)
                self.assertLessEqual((counts.max() - counts.min()).item(), 1)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_invalid_expert_layout_fails_before_dispatch(self) -> None:
        """
        Feature: DSV4.1 fixed-router expert layout validation.
        Description: Resolve fixed assignments with zero or indivisible local expert counts.
        Expectation: Invalid layouts raise a descriptive ValueError before expert dispatch.
        """
        for local_count in (0, 3):
            with self.subTest(local_count=local_count), self.assertRaisesRegex(ValueError, "divisible"):
                module = _make_module("sqrtsoftplus", torch.float32, False)
                module.experts.local_expert_count = local_count
                ep_adapter._resolve_router(ep_adapter._router, fix_router=True)(module, torch.ones(1, 5, 4))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_invalid_switch_fails_at_factory_boundary(self) -> None:
        """
        Feature: Strict benchmark flag validation.
        Description: Pass non-boolean values into the local compute factory.
        Expectation: Invalid configuration fails before binding or dispatch.
        """
        for flag in (None, 1, "true"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "boolean"):
                ep_adapter.deepseek_v41_ep_compute_fn(
                    module=_make_module("sqrtsoftplus", torch.float32, False),
                    mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=MagicMock(), fix_router=flag,
                )
