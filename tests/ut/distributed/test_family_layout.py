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
"""Module-owned layouts, precedence and explicit native-stage integration."""

from dataclasses import replace
import importlib
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.components.modules import family as registry
from hyper_parallel.components.modules.module_family import FamilyShardingContext, build_family
from hyper_parallel.core.dtensor.device_mesh import clear_device_mesh_cache, init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.distributed.module_binding import NativeParallelBindingRequest, ParallelModuleSpec
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from tests.common.module_family_fixtures import LinearConfig, LinearFamily, register_linear_family
from tests.common.mark_utils import arg_mark


def _linear_layout(*, module, module_fqn, member, context):
    del module_fqn, member, context
    return ModuleShardingSpec(
        params={name: {"tp": Shard(0)} for name, _ in module.named_parameters()},
        in_src={"input": {"tp": Replicate()}}, in_dst={"input": {"tp": Replicate()}},
        out_src={"output": {"tp": Shard(-1)}}, out_dst={"output": {"tp": Shard(-1)}},
    )


def _native_layout(*, module, module_fqn, member, context):
    del module_fqn, member, context
    return ModuleShardingSpec(params={name: {"cp": Replicate()} for name, _ in module.named_parameters()},
                              in_src={"inputs": {"cp": Replicate()}}, in_dst={"inputs": {"cp": Replicate()}},
                              out_src={"output": {"cp": Replicate()}}, out_dst={"output": {"cp": Replicate()}})


def _validate_native(*, module, sharding, context):
    del module, context
    if any(isinstance(placement, Shard) for named in sharding.params.values() for placement in named.values()):
        raise ValueError("test native CP protocol requires replicated projection weights")


def _native_protocol():
    return ParallelModuleSpec("test.native.gather", "cp", {"exchange": ("inputs",), "restore": ("output",)},
                              validate=_validate_native)


class _NativeAttention(nn.Module):
    """Native attention fixture with explicit class-defined CP execution stages."""

    def __init__(self) -> None:
        """Register projection parameters and an initially unbound context."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.cp_context = None

    def bind_parallel(self, binding: object) -> None:
        """Bind resources used by stages already defined in the class.

        Args:
            binding: Resource bundle from the common binder.
        """
        self.cp_context = binding.cp_context

    def exchange(self, inputs: torch.Tensor) -> torch.Tensor:
        """Exchange operands through an explicit stage, with a local identity path.

        Args:
            inputs: Prepared local operands.
        """
        return inputs if self.cp_context is None else self.cp_context.gather_sequence(inputs, dim=0)[0]

    def restore(self, output: torch.Tensor) -> torch.Tensor:
        """Preserve local output layout in this test protocol.

        Args:
            output: Computed attention output.
        """
        return output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Use class-defined stages through normal module execution.

        Args:
            inputs: Local attention input.
        """
        return self.restore(self.exchange(inputs) @ self.weight)


def _native_requests(*, model, context):
    del model, context
    return (NativeParallelBindingRequest(("component",), _native_protocol(), _native_layout),)


class TestFamilyLayouts(unittest.TestCase):
    """Planner providers precede templates and retain explicit resource ownership."""

    def setUp(self) -> None:
        """Isolate family and mesh registries between cases."""
        clear_device_mesh_cache()
        self.addCleanup(clear_device_mesh_cache)
        handle = patch.dict(registry._FAMILIES, clear=True)
        handle.start()
        self.addCleanup(handle.stop)
        register_linear_family(sharding=_linear_layout)

    def _mesh(self, axis="tp", size=2):
        return init_device_mesh("cpu", (size,), mesh_dim_names=(axis,), rank_list=tuple(range(size)),
                                init_backend=False)

    def _model(self):
        model = nn.Module()
        model.component = build_family(LinearFamily, config=LinearConfig(output_size=4))
        return model

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_owned_layout_discovery_reuses_family_inventory(self) -> None:
        """Feature: Layout discovery reuses the known family inventory.
        Description: Collect layouts for a mixed family/native model using the planner context.
        Expectation: Family boundaries are collected once and both providers declare complete parameter ownership.
        """
        layouts = importlib.import_module("hyper_parallel.distributed._builder.family_layout")
        model = self._model()
        model.native = _NativeAttention()
        spec = ModelAdapterSpec(
            "MixedModel", "mixed_test", native_parallel=lambda **kwargs: (
                NativeParallelBindingRequest(("native",), _native_protocol(), _native_layout),
            ),
        )
        context = FamilyShardingContext(("tp",), {"tp": 2, "cp": 1, "ep": 1},
                                        {name: tuple(value.shape) for name, value in model.named_parameters()})
        with patch.object(layouts, "collect_family_modules", wraps=layouts.collect_family_modules) as inventory:
            owned = layouts.collect_owned_layouts(model, context, spec)
        self.assertEqual(inventory.call_count, 1)
        self.assertEqual(set(owned.specs["component"].params), {"weight", "bias"})
        self.assertEqual(set(owned.specs["native"].params), {"weight"})

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_family_unknown_structure_works_without_templates_and_overrides_win(self):
        """Feature: Module-owned layout precedence.
        Description: Plan a new family structure with templates enabled or disabled and apply an explicit override.
        Expectation: The family supplies complete ownership and the explicit layout has the highest priority.
        """
        for derive in (True, False):
            model = self._model()
            planner = ShardingPlanner(derive=derive)
            plan = planner.plan(model, self._mesh(), tp_size=2)
            self.assertEqual(plan.modules["component"].params["weight"], {"tp": Shard(0)})
            self.assertEqual(set(plan.modules), {"component"})
            override = ModuleShardingSpec(params={name: {"tp": Replicate()} for name in ("weight", "bias")})
            plan = ShardingPlanner({"component": override}, derive=derive).plan(model, self._mesh(), tp_size=2)
            self.assertEqual(plan.modules["component"].params["weight"], {"tp": Replicate()})

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_coverage_unknown_names_and_storage_axes_fail_before_apply(self):
        """Feature: Complete family layout contracts.
        Description: Remove required ownership or declare an FSDP storage axis from a family provider.
        Expectation: Planning rejects invalid declarations before mutating parameters, including with derive=False.
        """
        model = self._model()
        with self.assertRaisesRegex(ValueError, "required parameter ownership"):
            ShardingPlanner({"component": ModuleShardingSpec(params={})}, derive=False).plan(
                model, self._mesh(), tp_size=2)
        family = registry.get_module_family(LinearFamily)
        def invalid(*, module: nn.Module, module_fqn: str, member: Any,
                    context: Any) -> ModuleShardingSpec:
            """Return an invalid storage-axis declaration for boundary validation."""
            del module, module_fqn, member, context
            return ModuleShardingSpec(params={"weight": {"dp": Shard(0)}, "bias": {}}, in_src={}, in_dst={})
        registry._FAMILIES[id(LinearFamily)] = replace(family, sharding=invalid)
        with self.assertRaisesRegex(ValueError, "storage/unknown"):
            ShardingPlanner(derive=False).plan(self._model(), self._mesh(), tp_size=2)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_shared_aliases_have_one_owner_and_conflicting_placements_fail(self):
        """Feature: Identity-based parameter ownership.
        Description: Discover module aliases and shared parameters with an incompatible explicit placement.
        Expectation: Aliases share one owner and inconsistent placements fail instead of producing independent shards.
        """
        model = self._model()
        model.alias = model.component
        plan = ShardingPlanner(derive=False).plan(model, self._mesh(), tp_size=2)
        self.assertEqual(len(plan.modules), 1)
        self.assertEqual(len(plan.parameter_aliases), 2)
        model.second = build_family(LinearFamily, config=LinearConfig(output_size=4))
        model.second.weight = model.component.weight
        with self.assertRaisesRegex(ValueError, "conflicting layouts"):
            ShardingPlanner({"second": ModuleShardingSpec(params={"weight": {"tp": Replicate()},
                                                                   "bias": {"tp": Shard(0)}})},
                            derive=False).plan(model, self._mesh(), tp_size=2)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_native_binding_preserves_module_parameters_and_class_forward(self):
        """Feature: Native explicit-stage resource binding.
        Description: Bind a native CPU component through its class-defined stages and run forward/backward.
        Expectation: Module and parameter identities, class forward and gradients remain correct without
            hooks or new groups.
        """
        model = nn.Module()
        model.component = _NativeAttention()
        model._hp_model_adapter_spec = ModelAdapterSpec("NativeModel", "native_test", native_parallel=_native_requests)
        component, parameter = model.component, model.component.weight
        forward = type(component).forward
        mesh = self._mesh("cp", 1)
        plan = ShardingPlanner(derive=False).plan(model, mesh)
        with patch("torch.distributed.new_group", side_effect=AssertionError("new group")):
            result, source = apply_sharding_plan(model, plan, SimpleNamespace(device_mesh=mesh, fsdp_moe_mesh=None))
        self.assertIs(result.component, component)
        self.assertIs(result.component.weight, parameter)
        self.assertIs(type(component).forward, forward)
        self.assertNotIn("forward", vars(component))
        self.assertFalse(component._forward_hooks)
        self.assertFalse(component._forward_pre_hooks)
        self.assertIsNone(source)
        inputs = torch.ones(1, 2, requires_grad=True)
        torch.testing.assert_close(result.component(inputs), torch.full((1, 2), 2.))
        result.component(inputs).sum().backward()
        torch.testing.assert_close(inputs.grad, torch.full((1, 2), 2.))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_native_unsupported_layout_is_rejected_before_parameter_mutation(self):
        """Feature: Native protocol capability validation.
        Description: Override a native CP component with a parameter layout unsupported by its protocol.
        Expectation: The final layout is rejected before replacing the original parameter.
        """
        model = nn.Module()
        model.component = _NativeAttention()
        model._hp_model_adapter_spec = ModelAdapterSpec("NativeModel", "native_test", native_parallel=_native_requests)
        before = model.component.weight
        override = ModuleShardingSpec(params={"weight": {"cp": Shard(0)}})
        with self.assertRaisesRegex(ValueError, "replicated"):
            ShardingPlanner({"component": override}, derive=False).plan(model, self._mesh("cp"), cp_size=2)
        self.assertIs(model.component.weight, before)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_context_is_read_only(self):
        """Feature: Immutable layout context.
        Description: Attempt to change the topology metadata exposed to layout providers.
        Expectation: Mutation raises TypeError rather than changing subsequent planning decisions.
        """
        context = FamilyShardingContext(("cp",), {"cp": 2}, {"weight": (4, 4)})
        with self.assertRaises(TypeError):
            context.axis_sizes["cp"] = 4

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_special_handlers_and_parent_templates_do_not_reclaim_family_parameters(self):
        """Feature: Family ownership exclusion from generic handling.
        Description: Plan a family-owned SPECIAL-like parameter and a parent that repeats a nested family parameter.
        Expectation: Generic handlers leave owned parameters alone and duplicate nested ownership is rejected.
        """
        model = self._model()
        model.component.a_log = nn.Parameter(torch.ones(4))
        plan = ShardingPlanner().plan(model, self._mesh(), tp_size=2)
        self.assertIn("a_log", plan.modules["component"].params)
        self.assertEqual(plan.special_handlers, {})
        model.component.child = build_family(LinearFamily, config=LinearConfig(output_size=4))
        with self.assertRaisesRegex(ValueError, "nested module ownership"):
            ShardingPlanner(derive=False).plan(model, self._mesh(), tp_size=2)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_native_cannot_bind_a_family_subtree(self):
        """Feature: Single communication owner.
        Description: Request native resource binding on an already family-owned subtree.
        Expectation: Planning rejects the overlapping ownership before applying any layout.
        """
        model = self._model()
        model._hp_model_adapter_spec = ModelAdapterSpec("NativeModel", "native_test", native_parallel=_native_requests)
        with self.assertRaisesRegex(ValueError, "family-owned"):
            ShardingPlanner(derive=False).plan(model, self._mesh(), tp_size=2)
