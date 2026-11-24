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
"""CPU boundary tests; the fake executor does not validate native MegaMoe numerics."""

from dataclasses import replace
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from hyper_parallel.components.modules import GroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import (
    GroupedExpertsConfig, GroupedExpertsModule,
)
from hyper_parallel.components.modules.moe.grouped_experts.ep.ep_mega_grouped_experts import EPMegaGroupedExperts
from hyper_parallel.components.modules.megakernel_utils import close_module_runtimes, prepare_module_runtimes
from hyper_parallel.distributed.module_binding import MeshBinding, ParallelBinding, validate_parallel_module
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.models.replacement import apply_module_replacements, compile_module_replacements
from hyper_parallel.trainer.config.parallelism import entries_to_module_replacements
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.common.mark_utils import arg_mark


class _Executor(nn.Module):
    """Parameterless boundary double with differentiable inputs and explicit resources."""

    shared = []

    def __init__(self, **options: Any) -> None:
        super().__init__()
        self.options = options
        self.received_weights = None
        self.closed = False

    def forward(self, hidden: torch.Tensor, ids: torch.Tensor, probabilities: torch.Tensor,
                *, expert_weights: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Retain every differentiable input without emulating the fused algorithm."""
        del ids
        self.received_weights = expert_weights
        return hidden * (probabilities.sum() + sum(weight.sum() for weight in expert_weights))

    @staticmethod
    def share_execution_resources(executors: list) -> None:
        """Record native resource sharing without allocating hardware state."""
        _Executor.shared.append(tuple(executors))

    def close(self) -> None:
        """Record deterministic resource release."""
        self.closed = True


class _PackedSource(nn.Module):
    """Expose the original routed forward signature for replacement validation."""

    def forward(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor,
                topk_weights: torch.Tensor) -> torch.Tensor:
        """Declare the source call contract; the test executes only its replacement."""
        del topk_ids, topk_weights
        return hidden_states


class _Wrapper(nn.Module):
    """Model a transparent activation wrapper that forwards unknown attributes."""

    def __init__(self, child: nn.Module) -> None:
        super().__init__()
        self.child = child

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.child, name)


class TestEPMegaGroupedExperts(unittest.TestCase):
    """Check family selection, borrowed topology, ownership and lifecycle."""

    def setUp(self) -> None:
        """Install a parameterless native boundary double without allocating device resources."""
        self.config = GroupedExpertsConfig(4, 6, 5, num_layers=2, top_k=2)
        package = ModuleType("hyper_multicore")
        package.MegaMoeExperts = _Executor
        package.MegaMoeNumericsOptions = SimpleNamespace
        package.SwiGLUParameters = SimpleNamespace
        experimental = ModuleType("hyper_multicore.experimental")
        experimental.MegaMoeTuningOptions = SimpleNamespace
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"hyper_multicore": package, "hyper_multicore.experimental": experimental}).start()
        _Executor.shared.clear()

    @staticmethod
    def _bind(module: EPMegaGroupedExperts, *, ep: int = 2, tp: int = 1, pp: int = 1,
              sequence_parallel: bool = False, group: Any = None) -> Any:
        group = object() if group is None else group
        axis = Mock()
        axis.get_group.return_value = group
        mesh = SimpleNamespace(ep_size=ep, tp_size=tp, pp_size=pp, sequence_parallel=sequence_parallel,
                               device_mesh={"tp": axis}, fsdp_moe_mesh={"ep": axis})
        module.bind_parallel(ParallelBinding(MeshBinding(mesh)))
        return group

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_public_factory_selects_only_ep_fused_variant(self) -> None:
        """Feature: EP-only Mega expert execution.
        Description: EP>1 selects the fused protocol without acquiring an ordinary EPContext.
        Expectation: Select the EP Mega class with canonical parameters and mesh binding.
        """
        module = GroupedExperts(config=self.config, megakernel=True, context={"ep_size": 2})
        self.assertIs(type(module), EPMegaGroupedExperts)
        self.assertEqual(module.parallel_module_spec().binding_kind, "mesh")
        self.assertEqual(set(dict(module.named_parameters())), {"gate_up_proj", "down_proj"})

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_ep_one_warns_and_falls_back_without_native_dependency(self) -> None:
        """Feature: EP-only Mega expert execution.
        Description: Local fallback ignores Mega tuning and retains ordinary algorithm capabilities.
        Expectation: Warn and construct the ordinary local member without importing the native backend.
        """
        config = replace(self.config, top_k=None, activation=torch.mul)
        with patch.dict(sys.modules, {"hyper_multicore": None}):
            for context in (None, {"ep_size": 1}):
                with self.subTest(context=context), self.assertWarnsRegex(UserWarning, "EP=1 falls back"):
                    module = GroupedExperts(
                        config=config, megakernel=True, context=context, dispatch_mode="pull",
                        initial_capacity_factor=2.0, capacity_growth_factor=1.5,
                    )
                self.assertIs(type(module), GroupedExpertsModule)
                self.assertEqual(module.config, config)
                self.assertFalse(hasattr(module, "_runtime"))
                prepare_module_runtimes(module)
                close_module_runtimes(module)
        self.assertEqual(_Executor.shared, [])

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_yaml_replacement_infers_routing_width_and_keeps_mega_tuning(self) -> None:
        """Feature: EP-only Mega expert execution.
        Description: Resolve actual YAML-style arguments through the public replacement planner.
        Expectation: Preserve parameter identity, infer top-k and route tuning to the selected Mega member.
        """
        config = resolve_config({
            "model": {"_target_": "torch.nn.Identity"},
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
            "plan_overrides": [{
                "match": "experts", "replace": {"kind": "module", "implementation": {
                    "_target_": "hyper_parallel.components.modules.GroupedExperts",
                    "megakernel": True, "dispatch_mode": "pull",
                }},
            }],
        })
        for ep, expected in ((1, GroupedExpertsModule), (2, EPMegaGroupedExperts)):
            with self.subTest(ep=ep):
                model = nn.Module()
                model.experts = _PackedSource()
                model.experts.config = SimpleNamespace(num_experts_per_tok=2, num_hidden_layers=4)
                model.experts.gate_up_proj = nn.Parameter(torch.ones(4, 10, 6))
                model.experts.down_proj = nn.Parameter(torch.ones(4, 6, 5))
                source = model.experts
                rules = entries_to_module_replacements(config.plan_overrides)
                plan = compile_module_replacements(model, rules)
                if ep == 1:
                    with self.assertWarnsRegex(UserWarning, "EP=1 falls back"):
                        apply_module_replacements(model, plan, weights_mapping=[], context={"ep_size": ep})
                else:
                    apply_module_replacements(model, plan, weights_mapping=[], context={"ep_size": ep})
                    self.assertEqual(model.experts._runtime.tuning["dispatch_mode"], "pull")
                self.assertIs(type(model.experts), expected)
                self.assertIs(model.experts.gate_up_proj, source.gate_up_proj)
                self.assertEqual(model.experts.config.top_k, 2)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_both_probability_modes_and_tuning_reach_native_executor(self) -> None:
        """Feature: EP-only Mega expert execution.
        Description: Map stable numerical options and retain the previous push/pull controls.
        Expectation: Forward the probability placement, clamp and transport options unchanged.
        """
        for mode in ("output", "before_fc2"):
            with self.subTest(mode=mode):
                module = GroupedExperts(config=replace(self.config, routing_weight_mode=mode, swiglu_limit=1.5),
                                        megakernel=True, dispatch_mode="pull", context={"ep_size": 2})
                group = self._bind(module)
                prepare_module_runtimes(module)
                options = module._runtime.executor.options
                self.assertIs(options["ep_group"], group)
                self.assertEqual(options["numerics_options"].apply_probs_before_fc2, mode == "before_fc2")
                self.assertFalse(options["activation"].clamp_after_silu)
                self.assertEqual(options["activation"].clamp_limit, 1.5)
                self.assertEqual(options["tuning_options"].dispatch_mode, "pull")
                self.assertEqual(set(module.state_dict()), {"gate_up_proj", "down_proj"})

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_ordinary_members_reject_mega_tuning_before_argument_projection(self) -> None:
        """Feature: EP-only Mega expert execution.
        Description: Signature filtering must preserve validation of public Mega-only options.
        Expectation: Reject Mega-only tuning for ordinary members in both construction paths.
        """
        source = GroupedExperts(config=self.config)
        for ep in (1, 2):
            for tuning in ({"dispatch_mode": "pull"}, {"initial_capacity_factor": 2.0},
                           {"capacity_growth_factor": 1.5}):
                with self.subTest(ep=ep, tuning=tuning):
                    with self.assertRaisesRegex(ValueError, "require megakernel=True"):
                        GroupedExperts(config=self.config, context={"ep_size": ep}, **tuning)
                    config = resolve_config({
                        "model": {"_target_": "torch.nn.Identity"},
                        "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
                        "plan_overrides": [{"match": "experts", "replace": {
                            "kind": "module", "implementation": {
                                "_target_": "hyper_parallel.components.modules.GroupedExperts", **tuning,
                            },
                        }}],
                    })
                    target = config.plan_overrides[0].replace.implementation
                    with self.assertRaisesRegex(ValueError, "require megakernel=True"):
                        target.build(module=source, context={"ep_size": ep})

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_forward_passes_current_weights_without_copy_or_ordinary_stages(self) -> None:
        """Feature: EP-only Mega expert execution.
        Description: FSDP-restored tensors and router gradients reach exactly one fused call.
        Expectation: Use current weights directly in one fused call and retain every input gradient.
        """
        module = GroupedExperts(config=self.config, megakernel=True, context={"ep_size": 2})
        self._bind(module)
        prepare_module_runtimes(module)
        hidden = torch.ones(2, 3, 6, requires_grad=True)
        ids = torch.tensor([[0, 1]] * 6, dtype=torch.int64)
        weights = torch.ones(6, 2, requires_grad=True)
        with patch.object(module, "permute", side_effect=AssertionError("ordinary permute")), \
                patch.object(module, "dispatch", side_effect=AssertionError("ordinary dispatch")):
            output = module(hidden, top_k_index=ids, top_k_weights=weights)
        self.assertEqual(output.shape, hidden.shape)
        received = module._runtime.executor.received_weights
        self.assertIs(received[0], module.gate_up_proj)
        self.assertIs(received[1], module.down_proj)
        output.sum().backward()
        for tensor in (hidden, weights, module.gate_up_proj, module.down_proj):
            self.assertIsNotNone(tensor.grad)
        module.gate_up_proj = nn.Parameter(torch.ones_like(module.gate_up_proj))
        module(hidden, ids, weights)
        self.assertIs(module._runtime.executor.received_weights[0], module.gate_up_proj)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_nonpositive_limit_disables_clamp_without_changing_mega_backend(self) -> None:
        """Feature: EP-only Mega expert execution.
        Description: Zero/negative clamp values warn once at construction and reach Mega as None.
        Expectation: Warn at construction and pass None to the selected Mega executor.
        """
        for limit in (0.0, -1.0):
            with self.subTest(limit=limit):
                with self.assertWarnsRegex(UserWarning, "disables clamping"):
                    config = replace(self.config, swiglu_limit=limit)
                self.assertIsNone(config.swiglu_limit)
                module = GroupedExperts(config=config, megakernel=True, context={"ep_size": 2})
                self.assertIs(type(module), EPMegaGroupedExperts)
                self._bind(module)
                prepare_module_runtimes(module)
                self.assertIsNone(module._runtime.executor.options["activation"].clamp_limit)
                close_module_runtimes(module)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_invalid_limit_still_rejected(self) -> None:
        """Feature: EP-only Mega expert execution.
        Description: Disabling a clamp does not permit nonnumeric or nonfinite configuration.
        Expectation: Reject boolean, nonnumeric and nonfinite limits before constructing an executor.
        """
        for limit in (True, float("nan"), float("inf"), -float("inf"), "1.5"):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "finite number"):
                replace(self.config, swiglu_limit=limit)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_serial_layers_share_once_and_wrappers_do_not_duplicate_owners(self) -> None:
        """Feature: EP-only Mega expert execution.
        Description: Prepare after materialization, share scratch once and release explicitly.
        Expectation: Share compatible workspaces once and release every native executor.
        """
        first = GroupedExperts(config=self.config, megakernel=True, context={"ep_size": 2})
        second = GroupedExperts(config=self.config, megakernel=True, context={"ep_size": 2})
        group = self._bind(first)
        self._bind(second, group=group)
        model = nn.ModuleList([_Wrapper(first), second])
        prepare_module_runtimes(model)
        prepare_module_runtimes(model)
        self.assertEqual(len(_Executor.shared), 1)
        self.assertEqual(len(_Executor.shared[0]), 2)
        executors = _Executor.shared[0]
        close_module_runtimes(model)
        close_module_runtimes(model)
        self.assertTrue(all(executor.closed for executor in executors))
        self.assertIsNone(first._runtime.executor)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_failed_share_releases_new_executors(self) -> None:
        """Feature: EP-only Mega expert execution.
        Description: A failed runtime preparation leaves no half-prepared native holder.
        Expectation: Close newly configured executors and leave no partially prepared runtime.
        """
        module = GroupedExperts(config=self.config, megakernel=True, context={"ep_size": 2})
        self._bind(module)
        with patch.object(_Executor, "share_execution_resources", side_effect=RuntimeError("share failed")):
            with self.assertRaisesRegex(RuntimeError, "share failed"):
                prepare_module_runtimes(module)
        self.assertIsNone(module._runtime.executor)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_parameter_owning_backend_is_rejected_and_closed(self) -> None:
        """Feature: EP-only Mega expert execution.
        Description: An incompatible backend cannot silently allocate a second expert table.
        Expectation: Reject and close a native executor that registers duplicate expert parameters.
        """
        module = GroupedExperts(config=self.config, megakernel=True, context={"ep_size": 2})
        self._bind(module)
        executor = _Executor()
        executor.register_parameter("duplicate", nn.Parameter(torch.ones(1)))
        with patch("hyper_multicore.MegaMoeExperts", return_value=executor):
            with self.assertRaisesRegex(RuntimeError, "parameterless"):
                prepare_module_runtimes(module)
        self.assertTrue(executor.closed)
        self.assertIsNone(module._runtime.executor)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_borrowed_group_rejects_rebinding_and_backend_unsupported_topologies(self) -> None:
        """Feature: EP-only Mega expert execution.
        Description: Preserve existing Mega PP/TP restrictions without constraining ordinary EP.
        Expectation: Reject unsupported topology or group changes while permitting TP with SP.
        """
        for topology in ({"ep": 1}, {"pp": 2}, {"tp": 2}):
            with self.subTest(topology=topology):
                module = GroupedExperts(config=self.config, megakernel=True, context={"ep_size": 2})
                with self.assertRaises(ValueError):
                    self._bind(module, **topology)
        module = GroupedExperts(config=self.config, megakernel=True, context={"ep_size": 2})
        self._bind(module, tp=2, ep=2, sequence_parallel=True)
        with self.assertRaisesRegex(ValueError, "communication owner"):
            self._bind(module, tp=2, ep=2, sequence_parallel=True)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_forward_requires_prepare_and_matching_routing_width(self) -> None:
        """Feature: EP-only Mega expert execution.
        Description: Reject lifecycle and shape mistakes before native execution.
        Expectation: Reject unprepared execution and an inconsistent top-k width before a native call.
        """
        module = GroupedExperts(config=self.config, megakernel=True, context={"ep_size": 2})
        with self.assertRaisesRegex(RuntimeError, "bind_parallel"):
            prepare_module_runtimes(module)
        hidden, ids, weights = torch.ones(2, 6), torch.zeros(2, 2, dtype=torch.int64), torch.ones(2, 2)
        with self.assertRaisesRegex(RuntimeError, "prepare_module_runtimes"):
            module(hidden, ids, weights)
        with self.assertRaisesRegex(ValueError, "top-k width"):
            module(hidden, ids[:, :1], weights[:, :1])

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_layout_validator_requires_canonical_ep_ownership(self) -> None:
        """Feature: EP-only Mega expert execution.
        Description: The fused executor receives complete experts and never expert TP slices.
        Expectation: Accept expert-axis sharding and reject expert-internal TP layouts.
        """
        module = GroupedExperts(config=self.config, megakernel=True, context={"ep_size": 2})
        context = SimpleNamespace(axis_sizes={"ep": 2})
        sharding = ModuleShardingSpec(params={"gate_up_proj": {"ep": Shard(0)}, "down_proj": {"ep": Shard(0)}},
                                      in_src={}, in_dst={}, is_boundary=False)
        validate_parallel_module(module, module.parallel_module_spec(), sharding, context)
        sharding.params["gate_up_proj"] = {"tp": Shard(1)}
        with self.assertRaisesRegex(ValueError, "TP-extend-EP"):
            validate_parallel_module(module, module.parallel_module_spec(), sharding, context)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_topk_aliases_from_source_context_preserve_parameter_identity(self) -> None:
        """Feature: EP-only Mega expert execution.
        Description: Infer native routing width from model config without a second YAML top-k.
        Expectation: Infer every recognized top-k alias without copying expert parameters.
        """
        source = nn.Module()
        source.gate_up_proj = nn.Parameter(torch.ones(4, 10, 6))
        source.down_proj = nn.Parameter(torch.ones(4, 6, 5))
        source.eval()
        for alias in ("num_experts_per_tok", "top_k", "moe_topk"):
            with self.subTest(alias=alias):
                module = GroupedExperts(module=source, megakernel=True,
                                        context={"num_layers": 2, "module_configs": ({alias: 2},), "ep_size": 2})
                self.assertEqual(module.config.top_k, 2)
                self.assertIs(module.gate_up_proj, source.gate_up_proj)
                self.assertFalse(module.training)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_custom_activation_is_rejected_instead_of_silently_replaced(self) -> None:
        """Feature: EP-only Mega expert execution.
        Description: Only declared SwiGLU/clamp semantics can be delegated to the native kernel.
        Expectation: Reject arbitrary activation callbacks before configuring the Mega executor.
        """
        with self.assertRaisesRegex(ValueError, "arbitrary activation"):
            GroupedExperts(config=replace(self.config, activation=torch.mul),
                           megakernel=True, context={"ep_size": 2})
        with self.assertRaisesRegex(ValueError, "top_k"):
            GroupedExperts(config=replace(self.config, top_k=None),
                           megakernel=True, context={"ep_size": 2})
