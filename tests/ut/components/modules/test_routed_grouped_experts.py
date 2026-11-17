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
"""CPU contracts for the routed grouped-expert family."""

from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.components.modules import GroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts import EPGroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import (
    GroupedExpertsConfig,
    GroupedExpertsModule,
    match_grouped_experts_inputs,
)
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.mesh import LogicalDeviceMesh
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec, apply_module_replacements, compile_module_replacements,
)
from hyper_parallel.trainer.config.parallelism import entries_to_module_replacements
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.common.mark_utils import arg_mark


def _cpu_permute(tokens: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Emulate the NPU token permutation and its source-to-permuted map."""
    flat_indices = indices.reshape(-1).to(torch.int64)
    order = flat_indices.argsort(stable=True)
    inverse = torch.empty_like(order, dtype=torch.int32)
    inverse.scatter_(0, order, torch.arange(order.numel(), dtype=torch.int32))
    return tokens.index_select(0, order // indices.shape[1]), inverse


def _cpu_unpermute(
    values: torch.Tensor, inverse: torch.Tensor, probabilities: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Emulate the fused inverse permutation, optional weighting and top-k sum."""
    source_slots = values.index_select(0, inverse.to(torch.int64))
    if probabilities is None:
        return source_slots
    source_slots = source_slots.reshape(*probabilities.shape, values.shape[-1])
    return (source_slots * probabilities.unsqueeze(-1)).sum(dim=1)


def _cpu_grouped_matmul(
    inputs: torch.Tensor, weights: torch.Tensor, *, group_list: torch.Tensor,
) -> torch.Tensor:
    """Apply the same expert row selection as grouped GEMM using CPU batched matmul."""
    counts = torch.diff(group_list, prepend=group_list.new_zeros(1))
    expert_ids = torch.arange(counts.numel(), device=counts.device).repeat_interleave(counts)
    return torch.bmm(inputs.unsqueeze(1), weights.index_select(0, expert_ids)).squeeze(1)


def _reference_forward(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    probabilities: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Evaluate packed SwiGLU experts directly in source-route order."""
    outputs = []
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    for token_index, token in enumerate(flat_hidden):
        token_outputs = []
        for route_index, expert_id in enumerate(expert_ids[token_index]):
            expert = int(expert_id)
            gate, up = F.linear(token, gate_up_proj[expert]).chunk(2, dim=-1)  # pylint: disable=not-callable
            activated = F.silu(gate) * up
            probability = probabilities[token_index, route_index]
            if mode == "before_fc2":
                activated = activated * probability
            output = F.linear(activated, down_proj[expert])  # pylint: disable=not-callable
            if mode == "output":
                output = output * probability
            token_outputs.append(output)
        outputs.append(torch.stack(token_outputs).sum(dim=0))
    return torch.stack(outputs).reshape_as(hidden_states)


class _PackedExperts(nn.Module):
    """Minimal Transformers-style packed expert source."""

    def __init__(self) -> None:
        """Create canonical packed projections and source configuration aliases."""
        super().__init__()
        self.config = SimpleNamespace(
            n_routed_experts=4,
            hidden_size=6,
            moe_intermediate_size=5,
            num_hidden_layers=8,
            initializer_range=0.03,
            hidden_act="silu",
        )
        self.gate_up_proj = nn.Parameter(torch.randn(4, 10, 6))
        self.down_proj = nn.Parameter(torch.randn(4, 6, 5))
        self.act_fn = nn.SiLU()

    def forward(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the source contract without relying on accelerator kernels."""
        return _reference_forward(
            hidden_states, topk_ids, topk_weights, self.gate_up_proj, self.down_proj, "output",
        )


class _RoutedMlp(nn.Module):
    """Parent MoE boundary containing a router and family-owned experts."""

    def __init__(self) -> None:
        """Build the minimum structure needed by semantic boundary inference."""
        super().__init__()
        self.gate = nn.Linear(6, 4, bias=False)
        self.experts = GroupedExperts(
            config=GroupedExpertsConfig(4, 6, 5, num_layers=1),
            context={"tp_size": 2, "ep_size": 2},
        )

    def forward(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate already selected routes to the expert family."""
        return self.experts(hidden_states, topk_ids, topk_weights)


class TestRoutedGroupedExperts(unittest.TestCase):
    """Verify algorithm semantics independently of NPU kernels and communication."""

    def setUp(self) -> None:
        """Use deterministic FP64 inputs so exact semantic errors are visible."""
        torch.manual_seed(17)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_public_yaml_target_selects_local_and_ep_experts(self) -> None:
        """Feature: Public routed-expert import and YAML target.
        Description: Resolve the public target through Trainer config and apply both topology choices.
        Expectation: The new family keeps canonical parameters and selects the local or EP member.
        """
        path = "hyper_parallel.components.modules.GroupedExperts"
        config = resolve_config({
            "model": {"_target_": "torch.nn.Identity"},
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
            "plan_overrides": [{
                "match": "experts",
                "replace": {"kind": "module", "implementation": {"_target_": path}},
            }],
        })
        rules = entries_to_module_replacements(config.plan_overrides)
        for ep_size, expected in ((1, GroupedExpertsModule), (2, EPGroupedExperts)):
            with self.subTest(ep_size=ep_size):
                model = nn.Module()
                source = _PackedExperts()
                model.experts = source
                apply_module_replacements(
                    model, compile_module_replacements(model, rules),
                    weights_mapping=[], context={"ep_size": ep_size},
                )
                self.assertIs(type(model.experts), expected)
                self.assertIs(model.experts.gate_up_proj, source.gate_up_proj)
                self.assertIs(model.experts.down_proj, source.down_proj)
        implementation = config.to_dict()["plan_overrides"][0]["replace"]["implementation"]
        self.assertEqual(implementation["_target_"], path)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_local_forward_and_gradients_match_direct_experts(self) -> None:
        """Feature: Local routed grouped experts.
        Description: Compare both probability placements with a direct expert oracle.
        Expectation: Outputs and all differentiable inputs and parameters agree.
        """
        expert_ids = torch.tensor([[3, 0], [1, 3], [0, 2]], dtype=torch.int64)
        for mode in ("output", "before_fc2"):
            with self.subTest(mode=mode):
                config = GroupedExpertsConfig(4, 6, 5, routing_weight_mode=mode, num_layers=1)
                candidate = GroupedExpertsModule(config=config).to(torch.float64)
                hidden = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
                probabilities = torch.randn(3, 2, dtype=torch.float64).softmax(-1).requires_grad_()
                reference_hidden = hidden.detach().clone().requires_grad_()
                reference_probabilities = probabilities.detach().clone().requires_grad_()
                reference_gate_up = candidate.gate_up_proj.detach().clone().requires_grad_()
                reference_down = candidate.down_proj.detach().clone().requires_grad_()

                with patch(
                    "hyper_parallel.components.modules.moe.grouped_experts.grouped_experts.moe_token_permute",
                    side_effect=_cpu_permute,
                ) as permute, patch(
                    "hyper_parallel.components.modules.moe.grouped_experts.grouped_experts.moe_token_unpermute",
                    side_effect=_cpu_unpermute,
                ), patch(
                    "hyper_parallel.components.modules.moe.grouped_experts.grouped_experts.grouped_matmul",
                    side_effect=_cpu_grouped_matmul,
                ):
                    actual = candidate(hidden, expert_ids, probabilities)
                expected = _reference_forward(
                    reference_hidden,
                    expert_ids,
                    reference_probabilities,
                    reference_gate_up,
                    reference_down,
                    mode,
                )
                actual_grads = torch.autograd.grad(
                    actual.square().sum(),
                    (hidden, probabilities, candidate.gate_up_proj, candidate.down_proj),
                )
                expected_grads = torch.autograd.grad(
                    expected.square().sum(),
                    (reference_hidden, reference_probabilities, reference_gate_up, reference_down),
                )

                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-11)
                self.assertEqual(permute.call_count, 1)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_source_adoption_preserves_parameters_and_configuration(self) -> None:
        """Feature: Packed Transformers source conversion.
        Description: Adopt compatible source weights, aliases, training state and activation child.
        Expectation: No parameter copy or reinitialization occurs and model facts are normalized.
        """
        source = _PackedExperts().eval()
        gate_up = source.gate_up_proj
        down = source.down_proj
        result = GroupedExperts(
            module=source, module_fqn="model.layers.0.mlp.experts", context={"num_layers": 8},
        )

        self.assertIs(result.gate_up_proj, gate_up)
        self.assertIs(result.down_proj, down)
        self.assertIs(result.act_fn, source.act_fn)
        self.assertFalse(result.training)
        self.assertEqual(result.config.num_experts, 4)
        self.assertEqual(result.config.hidden_size, 6)
        self.assertEqual(result.config.intermediate_size, 5)
        self.assertEqual(result.config.num_layers, 8)
        self.assertEqual(result.config.init_method_std, 0.03)
        self.assertEqual(result.config.routing_weight_mode, "output")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_replacement_resolves_depth_before_materialization(self) -> None:
        """Feature: Builder-owned global depth for expert initialization.
        Description: Replace meta experts using source, ancestor, composite and fallback configuration.
        Expectation: Local/EP experts receive global depth and scale FC2 after materialization.
        """
        cases = (
            (8, SimpleNamespace(num_hidden_layers=24), SimpleNamespace(num_layers=48), {"num_layers": 99}, 8),
            (None, SimpleNamespace(num_layers=24), SimpleNamespace(num_hidden_layers=48), {}, 24),
            (None, None, SimpleNamespace(num_hidden_layers=12,
                                        text_config=SimpleNamespace(num_hidden_layers=48)), {"pp_size": 4}, 48),
            (None, None, {"text_config": {"num_layers": 32}}, {}, 32),
            (None, None, SimpleNamespace(get_text_config=lambda: SimpleNamespace(num_hidden_layers=40)), {}, 40),
            (None, None, None, {"num_layers": 16}, 16),
        )
        for ep_size in (1, 2):
            for source_depth, parent_config, root_config, context, expected in cases:
                with self.subTest(ep_size=ep_size, depth=expected):
                    with torch.device("meta"):
                        source = _PackedExperts()
                        source.config.num_hidden_layers = source_depth
                        model = nn.Module()
                        model.config = root_config
                        model.decoder = nn.Module()
                        model.decoder.config = parent_config
                        model.decoder.experts = source
                    plan = compile_module_replacements(model, [ModuleReplacementSpec(
                        match=("decoder.experts",), factory=GroupedExperts,
                    )])
                    apply_module_replacements(
                        model, plan, weights_mapping=[], context={**context, "ep_size": ep_size},
                    )
                    result = model.decoder.experts
                    self.assertEqual(result.config.num_layers, expected)
                    self.assertIsInstance(result, EPGroupedExperts if ep_size > 1 else GroupedExpertsModule)
                    self.assertIs(result.gate_up_proj, source.gate_up_proj)
                    self.assertTrue(result.gate_up_proj.is_meta)
                    result.to_empty(device="cpu")
                    with patch("torch.nn.init.normal_") as initialize:
                        result.reset_parameters()
                    self.assertEqual(initialize.call_args_list[0].kwargs["std"], 0.03)
                    self.assertAlmostEqual(initialize.call_args_list[1].kwargs["std"], 0.03 / (2 * expected) ** 0.5)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_replacement_rejects_invalid_depth_before_installation(self) -> None:
        """Feature: Validate model depth at the replacement boundary.
        Description: Build experts with invalid depth in the model config or fallback context.
        Expectation: Invalid values fail before the original module is replaced.
        """
        for value in (0, -1, True, 2.5, "8"):
            for configured in (True, False):
                with self.subTest(value=value, configured=configured):
                    model = nn.Module()
                    source = _PackedExperts()
                    source.config.num_hidden_layers = value if configured else None
                    model.experts = source
                    plan = compile_module_replacements(model, [ModuleReplacementSpec(
                        match=("experts",), factory=GroupedExperts,
                    )])
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        apply_module_replacements(model, plan, weights_mapping=[], context={"num_layers": value})
                    self.assertIs(model.experts, source)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_canonical_depth_is_preserved_and_standalone_uses_context(self) -> None:
        """Feature: Explicit expert depth and standalone construction.
        Description: Adopt canonical experts with conflicting context and packed experts without a builder.
        Expectation: Canonical configuration is preserved; standalone adoption uses the explicit context depth.
        """
        source = GroupedExperts(config=GroupedExpertsConfig(4, 6, 5, num_layers=12))
        result = GroupedExperts(module=source, context={"num_layers": 48, "ep_size": 2})
        self.assertEqual(result.config.num_layers, 12)
        for ep_size in (1, 2):
            with self.subTest(ep_size=ep_size):
                standalone = GroupedExperts(module=_PackedExperts(), context={"num_layers": 8, "ep_size": ep_size})
                self.assertEqual(standalone.config.num_layers, 8,
                                 f"expected=8, actual={standalone.config.num_layers}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_standalone_requires_explicit_depth(self) -> None:
        """Feature: Trainer-independent expert initialization requires global depth.
        Description: Construct configs and adopt packed experts without a builder-provided layer count.
        Expectation: Missing depth fails with guidance; an explicit single-layer config is accepted.
        """
        for arguments in ({}, {"num_layers": None}):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "num_layers is required"):
                GroupedExpertsConfig(4, 6, 5, **arguments)
        for value in (0, -1, True, 2.5, "8"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "num_layers must be a positive integer"):
                GroupedExpertsConfig(4, 6, 5, num_layers=value)
        for ep_size in (1, 2):
            source = _PackedExperts()
            with self.subTest(ep_size=ep_size), self.assertRaisesRegex(ValueError, "context=.*num_layers"):
                GroupedExperts(module=source, context={"ep_size": ep_size})
            model = nn.Module()
            source.config.num_hidden_layers = None
            model.experts = source
            plan = compile_module_replacements(model, [ModuleReplacementSpec(
                match=("experts",), factory=GroupedExperts,
            )])
            with self.subTest(replacement=ep_size), self.assertRaisesRegex(ValueError, "num_layers is required"):
                apply_module_replacements(model, plan, weights_mapping=[], context={"ep_size": ep_size})
            self.assertIs(model.experts, source, f"expected={source}, actual={model.experts}")
            config = GroupedExpertsConfig(4, 6, 5, num_layers=1)
            with patch("torch.nn.init.normal_") as initialize:
                candidate = GroupedExperts(config=config, context={"ep_size": ep_size})
            self.assertEqual(candidate.config.num_layers, 1, f"expected=1, actual={candidate.config.num_layers}")
            actual_std = initialize.call_args_list[1].kwargs["std"]
            expected_std = 0.02 / 2 ** 0.5
            self.assertAlmostEqual(actual_std, expected_std, msg=f"expected={expected_std}, actual={actual_std}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_input_aliases_retain_identity_and_reject_ambiguity(self) -> None:
        """Feature: Transformers forward keyword compatibility.
        Description: Normalize known aliases without modifying routing tensors.
        Expectation: Tensor identity is retained and missing, duplicate or unknown inputs fail.
        """
        expert_ids = torch.tensor([[0, 1]])
        probabilities = torch.tensor([[0.4, 0.6]])
        actual_ids, actual_probabilities = match_grouped_experts_inputs(
            None,
            None,
            {"selected_experts": expert_ids, "routing_weights": probabilities},
        )
        self.assertIs(actual_ids, expert_ids)
        self.assertIs(actual_probabilities, probabilities)
        for kwargs in (
            {"selected_experts": expert_ids},
            {"selected_experts": expert_ids, "router_indices": expert_ids, "routing_weights": probabilities},
            {"selected_experts": expert_ids, "routing_weights": probabilities, "dense_mask": probabilities},
        ):
            with self.subTest(kwargs=tuple(kwargs)), self.assertRaises(TypeError):
                match_grouped_experts_inputs(None, None, kwargs)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_family_selects_only_implemented_local_and_ep_members(self) -> None:
        """Feature: Family selector matrix.
        Description: Select local and EP implementations and reject reserved execution modes.
        Expectation: The topology chooses the concrete member before runtime resources are bound.
        """
        config = GroupedExpertsConfig(4, 6, 5, num_layers=1)
        local = GroupedExperts(config=config)
        parallel = GroupedExperts(config=config, context={"ep_size": 2})
        self.assertIs(type(local), GroupedExpertsModule)
        self.assertIs(type(parallel), EPGroupedExperts)
        self.assertEqual(parallel.parallel_module_spec().binding_kind, "ep")
        for selectors in ({"megakernel": True}, {"moonep": True, "context": {"ep_size": 2}}):
            with self.subTest(selectors=selectors), self.assertRaises(NotImplementedError):
                GroupedExperts(config=config, **selectors)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_planner_keeps_parent_router_and_family_experts_in_separate_owners(self) -> None:
        """Feature: Default routed-MoE boundary inference.
        Description: Plan a parent router around independently owned EP experts on TP-extend-EP.
        Expectation: The parent uses replicated full outputs and experts retain only EP Shard(0).
        """
        model = nn.Module()
        model.config = SimpleNamespace(model_type="routed_test")
        model.mlp = _RoutedMlp()
        mesh = LogicalDeviceMesh(("tp",), (2,))

        plan = ShardingPlanner().plan(
            model,
            mesh,
            tp_size=2,
            ep_size=2,
            sequence_parallel=False,
        )

        parent = plan.modules["mlp"]
        experts = plan.modules["mlp.experts"]
        self.assertEqual(set(parent.params), {"gate.weight"})
        self.assertTrue(all(isinstance(value, Replicate) for value in parent.params["gate.weight"].values()))
        self.assertFalse(any(isinstance(value, Partial) for axes in parent.out_src.values() for value in axes.values()))
        self.assertEqual(experts.params, {
            "gate_up_proj": {"ep": Shard(0)},
            "down_proj": {"ep": Shard(0)},
        })
        self.assertFalse(experts.is_boundary)


if __name__ == "__main__":
    unittest.main()
