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
from unittest.mock import patch
import weakref

import torch
from torch import nn

from hyper_parallel.codegen.model_build_overlay import (
    _observe_replacement_installations, _observe_replacements, use_codegen_model_build_overlay,
)
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.components.modules import family as registry
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.models import HyperAutoModelForCausalLM, module_build
from hyper_parallel.models._transformers import auto_model
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.module_build import attach_build_context, build_module_context
from hyper_parallel.models.replacement import (
    ModuleReplacementRequest, ModuleReplacementSpec, apply_module_replacements,
    compile_module_replacements, module_replacement,
)
from tests.common.module_family_fixtures import (
    FamilyModelConfig, LinearFamily, LocalLinear, ParallelLinear, create_family_model, register_linear_family,
)


class TestModelBuildOverlay(unittest.TestCase):
    """Exercise real module identities, meta construction and snapshot contents."""

    def setUp(self):
        handle = patch.dict(registry._FAMILIES, clear=True)
        handle.start()
        self.addCleanup(handle.stop)
        register_linear_family()

    def test_defaults_and_legacy_rules_share_one_plan_and_snapshot_order(self):
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
        @module_replacement
        def fail(*, module, module_fqn, context):
            raise ValueError("candidate failed")

        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(4, 3))
        original = tuple(model)
        plan = compile_module_replacements(model, defaults=(
            ModuleReplacementRequest(("0",), LinearFamily), ModuleReplacementRequest(("1",), fail),
        ))
        with _observe_replacement_installations(model, plan.specs) as applications:
            with self.assertRaisesRegex(ValueError, "candidate failed"):
                apply_module_replacements(model, plan)
        self.assertEqual(applications, [])
        self.assertEqual(tuple(model), original)

    def test_observation_does_not_keep_replaced_sources_alive(self):
        references = []

        @module_replacement
        def replace(*, module, module_fqn, context):
            self.assertTrue(all(reference() is None for reference in references))
            references.append(weakref.ref(module))
            return LocalLinear(module=module, module_fqn=module_fqn, context=context)

        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(4, 3))
        plan = compile_module_replacements(model, (ModuleReplacementSpec(("*",), replace),))
        with _observe_replacement_installations(model, plan.specs) as applications:
            apply_module_replacements(model, plan)
        self.assertEqual(len(applications), 2)
        self.assertTrue(all(reference() is None for reference in references))
