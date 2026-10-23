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
from types import MethodType, SimpleNamespace
from typing import Callable
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch
from torch import nn
from torch.nn import functional
import yaml

from hyper_parallel.distributed._builder.rule_resolver import _resolve_local_compute_fn
from hyper_parallel.models.deepseek_v41.adapter.distributed import moe_engram_expert_parallel as ep_adapter
from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import DeepseekV41MegaMoeExperts
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41TopKRouter,
    _v41_sparse_moe_forward,
)
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
    def test_training_signature_keeps_fixed_destinations_and_auxiliary_gradients(self) -> None:
        """
        Feature: Fixed routing through the auxiliary-aware V4.1 forward contract.
        Description: Bind the real training signature with sequence auxiliary gradients enabled.
        Expectation: Both routing modes preserve outputs and input/router gradients against the oracle.
        """
        for dtype, multimodal, fixed in product((torch.float32, torch.bfloat16), (False, True), (False, True)):
            with self.subTest(dtype=dtype, multimodal=multimodal, fixed=fixed):
                self._check_forward_backward("sqrtsoftplus", dtype, multimodal, fixed, training_contract=True)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_megamoe_keeps_fixed_destinations_and_auxiliary_gradients(self) -> None:
        """
        Feature: MegaMoe benchmark routing with the real auxiliary-aware gate.
        Description: Exercise learned/fixed FP32/BF16 text and VLM routing on a four-rank EP layout.
        Expectation: Indices balance while outputs, input/router gradients and learned load statistics match.
        """
        for dtype, multimodal, fixed in product((torch.float32, torch.bfloat16), (False, True), (False, True)):
            with self.subTest(dtype=dtype, multimodal=multimodal, fixed=fixed):
                self._check_forward_backward(
                    "sqrtsoftplus", dtype, multimodal, fixed, training_contract=True, megamoe=True,
                )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_megamoe_yaml_override_transports_switch(self) -> None:
        """
        Feature: Typed MegaMoe fixed-router configuration.
        Description: Resolve both switch values through the real YAML Target and local-compute builder.
        Expectation: The built training callable changes destinations only when fixed routing is enabled.
        """
        for fixed in (False, True):
            with self.subTest(fixed=fixed):
                entry = resolve_component(yaml.safe_load(f"""
match: '*.mlp'
when: ep
region_dispatch: false
local_compute_fn:
  _target_: {ep_adapter.__name__}.deepseek_v41_megamoe_compute_fn
  fix_router: {str(fixed).lower()}
"""), annotation=PlanOverride, path="$.plan_overrides[0]")
                spec = entries_to_plan_overrides([entry], ep_size=4)["*.mlp"]
                module = _make_module("sqrtsoftplus", torch.float32, True)
                module.forward = MethodType(_v41_sparse_moe_forward, module)
                module.experts = Mock(
                    spec=DeepseekV41MegaMoeExperts, global_experts=8, swiglu_limit=1.5,
                    side_effect=lambda states, indices, weights: states,
                )
                module.shared_experts = Mock(limit=1.5, side_effect=torch.zeros_like)
                ep_mesh = MagicMock()
                ep_mesh.__getitem__.return_value.size.return_value = 4
                compute = _resolve_local_compute_fn(module, spec, mesh=None, mesh_dim_names=(), expert_mesh=ep_mesh)
                states = torch.ones(1, 4, 4)
                _, expected_weights, expected_indices = module.gate(states)
                self.assertIs(spec.local_compute_fn.fix_router, fixed)
                torch.testing.assert_close(compute(states), states)
                expected_indices = torch.tensor([[0, 2], [4, 6], [1, 3], [5, 7]]) if fixed else expected_indices
                torch.testing.assert_close(module.experts.call_args.args[1], expected_indices)
                torch.testing.assert_close(module.experts.call_args.args[2], expected_weights)
                module.experts.configure.assert_called_once_with(ep_mesh.get_group.return_value, 4, 2)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_megamoe_preserves_expert_shared_and_auxiliary_gradients(self) -> None:
        """
        Feature: MegaMoe fixed routing around real model-owned parameters.
        Description: Compare empty/partial/full text and VLM inputs with an independent expert oracle.
        Expectation: Outputs and every input/router/expert/shared gradient match in either routing mode.
        """
        for tokens, multimodal, fixed in product((0, 5, 8), (False, True), (False, True)):
            with self.subTest(tokens=tokens, multimodal=multimodal, fixed=fixed):
                self._check_megamoe_parameter_gradients(tokens, multimodal, fixed)

    def _check_megamoe_parameter_gradients(self, tokens: int, multimodal: bool, fixed: bool) -> None:
        """Retain the real parameter adapter and replace only its NPU executor."""
        torch.manual_seed(103)
        source = nn.Module()
        source.num_experts, source.hidden_dim, source.intermediate_dim = 4, 4, 3
        source.limit, source.act_fn = 1.5, nn.SiLU()
        source.gate_up_proj = nn.Parameter(torch.randn(4, 6, 4))
        source.down_proj = nn.Parameter(torch.randn(4, 4, 3))
        module = nn.Module()
        module.gate = DeepseekV41TopKRouter(SimpleNamespace(
            hidden_size=4, num_local_experts=4, num_experts_per_tok=2,
            scoring_func="sigmoid", routed_scaling_factor=1.25, v41_vision_enabled=multimodal,
            router_aux_loss_coef=0.1, router_bias_update_rate=0.001,
        ))
        with torch.no_grad():
            module.gate.weight.normal_(std=0.1)
            module.gate.bias.copy_(torch.tensor([2., 2., 0., 0.]))
            if multimodal:
                module.gate.bias_vl.copy_(torch.tensor([0., 0., 2., 2.]))
        reference_gate = copy.deepcopy(module.gate)
        module.experts = DeepseekV41MegaMoeExperts(module=source)
        module.shared_experts = nn.Linear(4, 4)
        module.shared_experts.limit = 1.5
        module.is_hash = False
        module.forward = MethodType(_v41_sparse_moe_forward, module)
        checkpoint_keys = set(module.state_dict())
        hidden = torch.randn(1, tokens, 4, requires_grad=True)
        reference_states = hidden.detach().clone().requires_grad_()
        images = torch.arange(tokens).view(1, -1) % 3 == 0 if multimodal else None
        with patch.object(module.experts, "configure") as configure, patch.object(
            module.experts, "_executor", Mock(side_effect=self._execute_mega_experts),
        ):
            compute = ep_adapter.deepseek_v41_megamoe_compute_fn(
                module=module, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=None, fix_router=fixed,
            )
            actual = compute(module, hidden, image_mask=images)
        configure.assert_called_once_with(None, 1, 2)
        _, weights, indices = reference_gate(reference_states, image_mask=images)
        if fixed:
            indices = torch.arange(tokens * 2).view(tokens, 2) % 4
        expected = torch.zeros_like(reference_states.reshape(-1, 4))
        for expert in range(4):
            gate, up = (reference_states.reshape(-1, 4) @ source.gate_up_proj[expert].t()).chunk(2, -1)
            activated = functional.silu(gate.clamp(max=1.5)) * up.clamp(-1.5, 1.5)
            projected = activated @ source.down_proj[expert].t()
            expected = expected + projected * ((indices == expert) * weights).sum(-1, keepdim=True)
        expected = expected.view_as(reference_states) + module.shared_experts(reference_states)
        torch.testing.assert_close(actual, expected)
        targets = (hidden, module.gate.weight, module.experts.gate_up_proj, module.experts.down_proj,
                   *module.shared_experts.parameters())
        reference_targets = (reference_states, reference_gate.weight, source.gate_up_proj, source.down_proj,
                             *module.shared_experts.parameters())
        upstream = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, targets, upstream)
        expected_grads = torch.autograd.grad(expected, reference_targets, upstream)
        for index, (left, right) in enumerate(zip(actual_grads, expected_grads)):
            torch.testing.assert_close(left, right.transpose(1, 2) if index in (2, 3) else right)
        torch.testing.assert_close(module.gate.last_aux_loss, reference_gate.last_aux_loss)
        torch.testing.assert_close(module.gate.tokens_per_expert, reference_gate.tokens_per_expert)
        self.assertEqual(set(module.state_dict()), checkpoint_keys)
        self.assertFalse(hasattr(module.experts, "local_expert_count"))

    @staticmethod
    def _execute_mega_experts(
            inputs: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor,
            *, expert_weights: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Execute the native packed parameter layout using CPU differentiable math."""
        gate_up, down = expert_weights
        states = inputs.reshape(-1, inputs.shape[-1])
        output = torch.zeros_like(states)
        for slot in range(indices.shape[1]):
            gate, up = torch.bmm(states.unsqueeze(1), gate_up[indices[:, slot]]).squeeze(1).chunk(2, -1)
            activated = functional.silu(gate.clamp(max=1.5)) * up.clamp(-1.5, 1.5)
            projected = torch.bmm(activated.unsqueeze(1), down[indices[:, slot]]).squeeze(1)
            output = output + projected * weights[:, slot, None]
        return output.view_as(inputs)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_training_signature_transports_flags_and_partitioned_metadata(self) -> None:
        """
        Feature: Training-aware fixed-router metadata and statistics groups.
        Description: Slice packed sample IDs, image/valid masks and configure CP/TP/DP groups.
        Expectation: Both switches and all local metadata reach the routed helper; EP is not a statistics axis.
        """
        for grouped, fixed, tp_rank in product((False, True), (False, True), (0, 1)):
            with self.subTest(grouped=grouped, fixed=fixed, tp_rank=tp_rank):
                self._check_training_metadata(grouped, fixed, tp_rank)

    def _check_training_metadata(self, grouped: bool, fixed: bool, tp_rank: int) -> None:
        """Exercise the factory using the model's real auxiliary-aware signature."""
        module = _make_module("sqrtsoftplus", torch.float32, True)
        module.forward = MethodType(_v41_sparse_moe_forward, module)
        states = torch.ones(1, 4, 4)
        metadata = {
            "image_mask": torch.tensor([[False, True, False, True, True, False, True, False]]),
            "router_token_mask": torch.tensor([[True, True, False, True, True, False, True, True]]),
            "router_sequence_ids": torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2]]),
        }
        tp_mesh, cp_mesh, ep_mesh, mesh = (MagicMock() for _ in range(4))
        tp_mesh.size.return_value = cp_mesh.size.return_value = 2
        tp_mesh.get_local_rank.return_value = tp_rank
        tp_mesh.get_group.return_value = "tp-group"
        cp_mesh.get_group.return_value = "cp-group"
        ep_mesh.__getitem__.return_value.size.return_value = 4
        ep_mesh.get_group.return_value = "ep-group"
        mesh.mesh_dim_names = ("dp", "ep")
        mesh.__getitem__.return_value.size.return_value = 2
        mesh.__getitem__.return_value.get_group.return_value = "dp-group"
        with patch.object(ep_adapter, "bind_local_expert_forward") as bind, patch.object(
            ep_adapter, "_routed_and_shared_forward", return_value=states,
        ) as routed:
            compute = ep_adapter.deepseek_v41_ep_compute_fn(
                module=module, mesh=mesh, tp_mesh=tp_mesh, cp_mesh=cp_mesh,
                ep_mesh=ep_mesh, use_grouped_gemm=grouped, fix_router=fixed,
            )
            self.assertIs(compute(module, states, **metadata, router_num_sequences=3), states)
        routed.assert_called_once()
        self.assertIs(routed.call_args.args[0], module)
        self.assertIs(routed.call_args.args[1], states)
        self.assertEqual(routed.call_args.args[3], "ep-group")
        self.assertIs(bind.call_args.kwargs["use_grouped_gemm"], grouped)
        self.assertEqual(routed.call_args.kwargs["ep_size"], 4)
        self.assertIs(routed.call_args.kwargs["fix_router"], fixed)
        torch.testing.assert_close(
            routed.call_args.args[2], metadata["image_mask"][:, tp_rank * 4:(tp_rank + 1) * 4],
        )
        for name in ("router_token_mask", "router_sequence_ids"):
            torch.testing.assert_close(
                routed.call_args.kwargs[name], metadata[name][:, tp_rank * 4:(tp_rank + 1) * 4],
            )
        self.assertEqual(routed.call_args.kwargs["router_num_sequences"], 3)
        self.assertEqual(routed.call_args.kwargs["sequence_partition_groups"], ("cp-group", "tp-group"))
        self.assertEqual(module.gate.expert_bias_update_groups, ("dp-group", "cp-group", "tp-group"))

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
                    module, states, image_mask, ep_mesh.get_group.return_value, ep_size=4, fix_router=fixed,
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

    def _check_forward_backward(
            self, scoring: str, dtype: torch.dtype, multimodal: bool, fixed: bool,
            training_contract: bool = False, megamoe: bool = False,
    ) -> None:
        """Mock expert transport while exercising the real factory and gate."""
        module = _make_module(scoring, dtype, multimodal)
        if training_contract:
            module.forward = MethodType(_v41_sparse_moe_forward, module)
            module.gate.aux_loss_coeff = 0.1
        reference = copy.deepcopy(module.gate)
        parameters = dict(module.gate.named_parameters())
        checkpoint_keys = set(module.gate.state_dict())
        states = torch.linspace(-1, 1, 40).reshape(1, 10, 4).to(dtype).requires_grad_()
        expected_states = states.detach().clone().requires_grad_()
        image_mask = torch.arange(10).reshape(1, 10) % 3 == 0 if multimodal else None
        local_mask = image_mask
        captured = {}

        def execute(inputs: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            """Keep expert identities and differentiable weights observable on CPU."""
            captured.update(indices=indices, weights=weights)
            factors = (weights * (indices + 1)).sum(-1).reshape(1, 10, 1)
            return inputs * factors

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
            return execute(inputs, indices, weights)

        factory = ep_adapter.deepseek_v41_ep_compute_fn
        if megamoe:
            module.experts = Mock(spec=DeepseekV41MegaMoeExperts, global_experts=8,
                                  swiglu_limit=1.5, side_effect=execute)
            module.shared_experts = Mock(limit=1.5, side_effect=module.shared_experts)
            factory = ep_adapter.deepseek_v41_megamoe_compute_fn
        ep_mesh = MagicMock()
        ep_mesh.__getitem__.return_value.size.return_value = 4
        ep_mesh.get_group.return_value = "ep-group"
        with patch.object(ep_adapter, "bind_local_expert_forward"), patch.object(
            ep_adapter, "ep_routed_forward", side_effect=dispatch,
        ), patch.object(module.gate, "forward", wraps=module.gate.forward) as gate_forward:
            compute = factory(
                module=module, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=ep_mesh, fix_router=fixed,
            )
            actual = compute(module, states, image_mask=image_mask) if multimodal else compute(module, states)
            gate_forward.assert_called_once()
            if multimodal:
                torch.testing.assert_close(gate_forward.call_args.kwargs["image_mask"], local_mask)

        _, expected_weights, learned_indices = reference(expected_states, image_mask=local_mask)
        if training_contract:
            self.assertIsNotNone(module.gate.last_aux_loss)
            torch.testing.assert_close(module.gate.last_aux_loss, reference.last_aux_loss)
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
        torch.testing.assert_close(module.gate.tokens_per_expert, reference.tokens_per_expert)
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
                indices, actual_weights = ep_adapter._resolve_router(real_router, ep_size=peers, fix_router=True)(
                    module, states,
                )
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
        Description: Resolve fixed assignments with nonpositive or indivisible expert-parallel layouts.
        Expectation: Invalid layouts raise a descriptive ValueError before expert dispatch.
        """
        for expert_count, ep_size in ((0, 1), (8, 0), (8, 3)):
            with self.subTest(experts=expert_count, ep_size=ep_size), self.assertRaisesRegex(ValueError, "divisible"):
                module = _make_module("sqrtsoftplus", torch.float32, False)
                module.gate.num_experts = expert_count
                ep_adapter._resolve_router(ep_adapter._router, fix_router=True, ep_size=ep_size)(
                    module, torch.ones(1, 5, 4),
                )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_invalid_switch_fails_at_factory_boundary(self) -> None:
        """
        Feature: Strict benchmark flag validation.
        Description: Pass non-boolean values into the local compute factory.
        Expectation: Invalid configuration fails before binding or dispatch.
        """
        for flag in (None, 1, "true"):
            for factory in (ep_adapter.deepseek_v41_ep_compute_fn, ep_adapter.deepseek_v41_megamoe_compute_fn):
                with self.subTest(flag=flag, factory=factory.__name__), self.assertRaisesRegex(ValueError, "boolean"):
                    factory(
                        module=_make_module("sqrtsoftplus", torch.float32, False),
                        mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=MagicMock(), fix_router=flag,
                    )
