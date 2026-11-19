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
"""Shared family preparation and capture replace the former build overlay."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import torch
from torch import nn

from hyper_parallel.components.modules import family as registry
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.models import HyperAutoModelForCausalLM, module_build
from hyper_parallel.models._transformers import auto_model, model_builder
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.build_capture import capture_model_build_plan
from hyper_parallel.models.module_build import (
    attach_build_context,
    build_module_context,
    prepare_model_components,
)
from hyper_parallel.models.preparation_manifest import (
    capture_preparation_manifest,
    collect_replacement_applications,
    get_preparation_manifest,
)
from hyper_parallel.models.replacement import (
    MethodReplacementSpec,
    ModuleReplacementRequest,
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
    method_replacement,
    module_replacement,
)
from tests.common.module_family_fixtures import (
    FamilyModelConfig,
    LinearFamily,
    LocalLinear,
    ParallelLinear,
    create_family_model,
    register_linear_family,
)


class _MethodModel(nn.Module):
    """Minimal model with a replaceable non-forward stage."""

    def __init__(self, config, *, context):
        super().__init__()
        del context
        self.config = config
        self.weight = nn.Parameter(torch.ones(()))

    def stage(self, inputs):
        return inputs * self.weight

    def forward(self, inputs):
        return self.stage(inputs)


class TestSharedModelBuild(unittest.TestCase):
    """Exercise shared preparation without codegen-owned execution paths."""

    def setUp(self):
        handle = patch.dict(registry._FAMILIES, clear=True)
        handle.start()
        self.addCleanup(handle.stop)
        register_linear_family()

    def test_defaults_and_legacy_rules_share_manifest_order(self):
        """Default and legacy requests compile once and retain installation order."""
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(4, 3))
        request = ModuleReplacementRequest(("0",), LinearFamily)
        legacy = ModuleReplacementSpec(("1",), LinearFamily)
        spec = ModelAdapterSpec("Sequential", "toy", replacements=lambda: (request,))
        setup = SimpleNamespace(module_replacements=(legacy,), mesh_context=None)
        context = build_module_context(setup, model_spec=spec)
        attach_build_context(model, context)

        with patch.object(
            module_build,
            "compile_module_replacements",
            wraps=compile_module_replacements,
        ) as compile_plan, capture_preparation_manifest():
            prepared, _ = prepare_model_components(model, context)

        self.assertIs(prepared, model)
        self.assertEqual(compile_plan.call_count, 1)
        self.assertIs(type(model[0]), LocalLinear)
        self.assertIs(type(model[1]), LocalLinear)
        manifest = get_preparation_manifest(model)
        self.assertEqual(
            [item["rule_id"] for item in manifest.declarations],
            [item.declaration_id for item in manifest.replacements],
        )
        self.assertEqual(
            [item.primary_fqn for item in manifest.replacements],
            ["0", "1"],
        )

    def test_factory_capture_uses_shared_meta_preparation(self):
        """Direct factory capture resolves the same family once on meta."""
        setup = DistributedSetup(
            mesh_context=MeshContext(dp_size=2, dp_shard_size=2, ep_size=2)
        )
        with patch.object(auto_model, "_current_device", return_value=torch.device("cpu")), \
                patch.object(
                    module_build,
                    "_finalize_family_modules",
                    wraps=module_build._finalize_family_modules,
                ) as finalize, capture_model_build_plan() as capture:
            model = HyperAutoModelForCausalLM.from_config(
                FamilyModelConfig(),
                model_factory=create_family_model,
                distributed_setup=setup,
                torch_dtype=torch.float32,
            )

        self.assertEqual(finalize.call_count, 1)
        self.assertTrue(capture.complete)
        self.assertTrue(model._hp_components_prepared)
        self.assertIs(type(model.projection), ParallelLinear)
        self.assertEqual(model.projection.weight.device.type, "meta")
        self.assertEqual(capture.manifest.replacements, ())

    def test_failed_atomic_batch_records_no_applications(self):
        """A failed candidate batch preserves the model and commits no application."""
        @module_replacement
        def fail(*, module, module_fqn, context):
            del module, module_fqn, context
            raise ValueError("candidate failed")

        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(4, 3))
        original = tuple(model)
        plan = compile_module_replacements(
            model,
            defaults=(
                ModuleReplacementRequest(("0",), LinearFamily),
                ModuleReplacementRequest(("1",), fail),
            ),
        )
        with collect_replacement_applications() as applications:
            with self.assertRaisesRegex(ValueError, "candidate failed"):
                apply_module_replacements(model, plan)

        self.assertEqual(applications, [])
        self.assertEqual(tuple(model), original)

    def test_application_records_do_not_keep_replaced_sources_alive(self):
        """Stable application records retain no source module references."""
        references = []

        @module_replacement
        def replace(*, module, module_fqn, context):
            self.assertTrue(all(reference() is None for reference in references))
            references.append(weakref.ref(module))
            return LocalLinear(module=module, module_fqn=module_fqn, context=context)

        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(4, 3))
        plan = compile_module_replacements(
            model,
            (ModuleReplacementSpec(("*",), replace),),
        )
        with collect_replacement_applications() as applications:
            apply_module_replacements(model, plan)

        self.assertEqual(len(applications), 2)
        self.assertTrue(all(reference() is None for reference in references))

    def test_native_and_capture_share_method_replacement_order(self):
        """Both builds run method replacement from the same prepared context."""
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

        setup = DistributedSetup(
            method_replacements=(MethodReplacementSpec("0.stage", replace_stage),)
        )

        def factory(config, *, context):
            parent = nn.Sequential(_MethodModel(config, context=context))
            parent.config = config
            return parent

        with patch.object(auto_model, "_current_device", return_value=torch.device("cpu")), \
                patch.object(
                    model_builder,
                    "build_module_context",
                    side_effect=AssertionError("prepared context must be reused"),
                ):
            model = HyperAutoModelForCausalLM.from_config(
                FamilyModelConfig(),
                model_factory=factory,
                distributed_setup=setup,
                torch_dtype=torch.float32,
            )
            torch.testing.assert_close(model(torch.tensor([3.0])), torch.tensor([6.0]))
            with capture_model_build_plan() as capture:
                observed = HyperAutoModelForCausalLM.from_config(
                    FamilyModelConfig(),
                    model_factory=factory,
                    distributed_setup=setup,
                    torch_dtype=torch.float32,
                )

        self.assertTrue(capture.complete)
        self.assertEqual(len(events), 2)
        self.assertIs(events[0]["model_spec"], model._hp_model_adapter_spec)
        self.assertIs(events[1]["model_spec"], observed._hp_model_adapter_spec)
        self.assertEqual(observed[0].weight.device.type, "meta")

    def test_source_compat_uses_effective_build_spec(self):
        """Capture invokes the source hook of the exact model spec once."""
        prepare = Mock()
        provider = Mock(return_value=SimpleNamespace(prepare_model_import=prepare))
        spec = ModelAdapterSpec(
            "FamilyModel",
            "test_family_model",
            codegen=provider,
        )

        def prepare_spec(model_spec):
            model_spec.codegen().prepare_model_import()

        with patch.object(auto_model, "_current_device", return_value=torch.device("cpu")), \
                capture_model_build_plan(prepare_model_spec=prepare_spec):
            model = HyperAutoModelForCausalLM.from_config(
                FamilyModelConfig(),
                model_factory=create_family_model,
                model_spec=spec,
                torch_dtype=torch.float32,
            )

        provider.assert_called_once_with()
        prepare.assert_called_once_with()
        self.assertIs(model._hp_model_adapter_spec, spec)


if __name__ == "__main__":
    unittest.main()
