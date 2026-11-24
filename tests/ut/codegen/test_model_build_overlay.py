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
"""Family preparation and codegen observation share the native installation contract."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import torch
from torch import nn

from hyper_parallel.codegen.model_build_overlay import (
    _observe_plan, _observe_replacement_installations, _observe_replacements, use_codegen_model_build_overlay,
)
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.codegen.emit.replacement import replacement_records_from_snapshot
from hyper_parallel.components.modules import family as registry
from hyper_parallel.distributed.mesh import DistributedSetup, LogicalDeviceMesh, MeshContext
from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.models import HyperAutoModelForCausalLM, module_build
from hyper_parallel.models._transformers import auto_model, model_builder
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.module_build import attach_build_context, build_module_context
from hyper_parallel.models.replacement import (
    MethodReplacementSpec, ModuleReplacementRequest, ModuleReplacementSpec, apply_module_replacements,
    compile_module_replacements, method_replacement, module_replacement,
)
from tests.common.module_family_fixtures import (
    FamilyModelConfig, LinearFamily, LocalLinear, ParallelLinear, create_family_model, register_linear_family,
)


class _MethodModel(nn.Module):
    """A real model with an explicit non-forward stage for shared policy acceptance."""

    def __init__(self, config, *, context):
        super().__init__()
        del context
        self.config = config
        self.weight = nn.Parameter(torch.ones(()))

    def stage(self, inputs):
        return inputs * self.weight

    def forward(self, inputs):
        return self.stage(inputs)


class TestModelBuildOverlay(unittest.TestCase):
    """Exercise real module identities, meta construction and snapshot contents."""

    def setUp(self):
        handle = patch.dict(registry._FAMILIES, clear=True)
        handle.start()
        self.addCleanup(handle.stop)
        register_linear_family()

    def test_defaults_and_legacy_rules_share_one_plan_and_snapshot_order(self):
        """Default and legacy requests share compilation and preserve observed installation order."""
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(4, 3))
        request = ModuleReplacementRequest(("0",), LinearFamily)
        legacy = ModuleReplacementSpec(("1",), LinearFamily)
        spec = ModelAdapterSpec("Sequential", "toy", replacements=lambda: (request,))
        setup = SimpleNamespace(module_replacements=(legacy,), mesh_context=None)
        attach_build_context(model, build_module_context(setup, model_spec=spec))
        session = CodegenBuildSession()
        session.record_source({"model_type": "toy"})

        with patch.object(module_build, "compile_module_replacements",
                          wraps=compile_module_replacements) as compile_plan:
            result = _observe_replacements(model, setup, None, False, session)

        self.assertIs(result, model)
        self.assertEqual(compile_plan.call_count, 1)
        self.assertIs(type(model[0]), LocalLinear)
        self.assertIs(type(model[1]), LocalLinear)
        session.mark_not_required("snapshot_test")
        replacements = session.finish().to_dict()["replacements"]
        declarations, applications = replacements["declarations"], replacements["applications"]
        self.assertEqual([item["rule_id"] for item in declarations],
                         [item["rule_id"] for item in applications])
        self.assertEqual(declarations[0]["replacement_target"],
                         f"{LinearFamily.__module__}:{LinearFamily.__qualname__}")
        self.assertEqual([item["primary_fqn"] for item in applications], ["0", "1"])

    def test_factory_build_uses_shared_meta_preparation_without_replacement(self):
        """Direct factory builds finalize meta members once without synthetic replacement records."""
        setup = DistributedSetup(mesh_context=MeshContext(dp_size=2, dp_shard_size=2, ep_size=2))
        session = CodegenBuildSession()
        with patch.object(auto_model, "_current_device", return_value=torch.device("cpu")), \
                patch.object(module_build, "_finalize_family_modules",
                             wraps=module_build._finalize_family_modules) as finalize, \
                use_codegen_model_build_overlay(session) as result:
            model = HyperAutoModelForCausalLM.from_config(
                FamilyModelConfig(), model_factory=create_family_model, distributed_setup=setup,
                torch_dtype=torch.float32,
            )
        self.assertEqual(finalize.call_count, 1)
        self.assertTrue(model._hp_components_prepared)
        self.assertIs(type(model.projection), ParallelLinear)
        self.assertEqual(model.projection.weight.device.type, "meta")
        self.assertEqual(result.snapshot.to_dict()["replacements"]["applications"], [])

    def test_failed_atomic_batch_records_no_installations(self):
        """A failed candidate batch preserves the original model and publishes no installations."""
        @module_replacement
        def fail(*, module, module_fqn, context):
            raise ValueError("candidate failed")

        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(4, 3))
        original = tuple(model)
        plan = compile_module_replacements(model, defaults=(
            ModuleReplacementRequest(("0",), LinearFamily), ModuleReplacementRequest(("1",), fail),
        ))
        with _observe_replacement_installations(plan.specs) as applications:
            with self.assertRaisesRegex(ValueError, "candidate failed"):
                apply_module_replacements(model, plan)
        self.assertEqual(applications, [])
        self.assertEqual(tuple(model), original)

    def test_observation_does_not_keep_replaced_sources_alive(self):
        """Codegen observation releases source modules after the shared installer replaces them."""
        references = []

        @module_replacement
        def replace(*, module, module_fqn, context):
            self.assertTrue(all(reference() is None for reference in references))
            references.append(weakref.ref(module))
            return LocalLinear(module=module, module_fqn=module_fqn, context=context)

        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(4, 3))
        plan = compile_module_replacements(model, (ModuleReplacementSpec(("*",), replace),))
        with _observe_replacement_installations(plan.specs) as applications:
            apply_module_replacements(model, plan)
        self.assertEqual(len(applications), 2)
        self.assertTrue(all(reference() is None for reference in references))

    def test_wildcard_user_request_has_distinct_effective_declarations(self):
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(4, 3))
        request = ModuleReplacementRequest(("*",), LinearFamily, declaration_id="user-rule")
        setup = SimpleNamespace(module_replacement_requests=(request,), mesh_context=None)
        session = CodegenBuildSession()
        session.record_source({"model_type": "toy"})
        _observe_replacements(model, setup, None, False, session)
        session.mark_not_required("snapshot_test")
        records = replacement_records_from_snapshot(session.finish())
        self.assertEqual([record["fqn"] for record in records], ["0", "1"])

    def test_size_one_owned_layout_is_planned_during_observation(self):
        """A module-owned layout is planned even when every parallel axis has size one."""
        model = nn.Module()
        model._hp_model_adapter_spec = ModelAdapterSpec("Module", "toy", sharding=lambda **_: {})
        mesh = MeshContext()
        mesh.device_mesh = LogicalDeviceMesh(mesh_dim_names=("dp",), mesh_shape=(1,))
        planner = SimpleNamespace(plan=lambda *_args, **_kwargs: ShardingPlan())
        session = CodegenBuildSession()
        session.record_source({"model_type": "toy"})
        session.record_replacements()
        with patch.object(planner, "plan", wraps=planner.plan) as plan:
            _observe_plan(model, mesh, planner, True, session)
        self.assertEqual(plan.call_count, 1)
        self.assertEqual(session.finish().to_dict()["status"], "derived")

    def test_native_and_observation_share_method_policy_and_prepared_context(self):
        """Native and offline builds reuse the prepared context for identical method replacement policy."""
        events = []

        @method_replacement
        def replace_stage(*, module, method_fqn, original_method, context):
            self.assertIsInstance(module, _MethodModel)
            self.assertEqual(method_fqn, "0.stage")
            self.assertEqual(context["dtype"], "float32")
            events.append(context)

            def scaled(inputs):
                return original_method(inputs) * 2

            return scaled

        setup = DistributedSetup(method_replacements=(MethodReplacementSpec("0.stage", replace_stage),))
        def factory(config, *, context):
            parent = nn.Sequential(_MethodModel(config, context=context))
            parent.config = config
            return parent

        with patch.object(auto_model, "_current_device", return_value=torch.device("cpu")), \
                patch.object(model_builder, "build_module_context",
                             side_effect=AssertionError("prepared context must be reused")):
            model = HyperAutoModelForCausalLM.from_config(
                FamilyModelConfig(), model_factory=factory, distributed_setup=setup, torch_dtype=torch.float32,
            )
            torch.testing.assert_close(model(torch.tensor([3.])), torch.tensor([6.]))
            with use_codegen_model_build_overlay(CodegenBuildSession()):
                observed = HyperAutoModelForCausalLM.from_config(
                    FamilyModelConfig(), model_factory=factory, distributed_setup=setup, torch_dtype=torch.float32,
                )
        self.assertEqual(len(events), 2)
        self.assertIs(events[0]["model_spec"], model._hp_model_adapter_spec)
        self.assertIs(events[1]["model_spec"], observed._hp_model_adapter_spec)
        self.assertEqual(observed[0].weight.device.type, "meta")

    def test_replacement_only_observation_uses_empty_native_plan_projection(self):
        """Replacement-only builds publish an empty layout plan together with installation metadata."""
        model = nn.Sequential(nn.Linear(4, 3))
        setup = SimpleNamespace(module_replacement_requests=(ModuleReplacementRequest(("0",), LinearFamily),),
                                mesh_context=None)
        session = CodegenBuildSession()
        session.record_source({"model_type": "toy"})
        model = _observe_replacements(model, setup, None, False, session)
        mesh = MeshContext()
        mesh.device_mesh = LogicalDeviceMesh(mesh_dim_names=("dp",), mesh_shape=(1,))
        planner = SimpleNamespace(plan=Mock(side_effect=AssertionError("pure DP needs no layout plan")))
        _observe_plan(model, mesh, planner, True, session)
        result = session.finish().to_dict()
        self.assertEqual(result["status"], "derived")
        self.assertEqual(result["plan"]["param_plan"], {})
        self.assertEqual(result["plan"]["injections"], [])
        self.assertEqual(result["plan"]["parallel_dims"]["dp_size"], 1)
        self.assertEqual(len(result["replacements"]["applications"]), 1)

    def test_source_compat_uses_the_effective_per_build_spec(self):
        """An explicit model spec supplies its import policy without a second registry lookup."""
        prepare = Mock()
        provider = Mock(return_value=SimpleNamespace(prepare_model_import=prepare))
        spec = ModelAdapterSpec("FamilyModel", "test_family_model", codegen=provider)
        with patch.object(auto_model, "_current_device", return_value=torch.device("cpu")), \
                use_codegen_model_build_overlay(CodegenBuildSession()):
            model = HyperAutoModelForCausalLM.from_config(
                FamilyModelConfig(), model_factory=create_family_model, model_spec=spec, torch_dtype=torch.float32,
            )
        provider.assert_called_once_with()
        prepare.assert_called_once_with()
        self.assertIs(model._hp_model_adapter_spec, spec)
