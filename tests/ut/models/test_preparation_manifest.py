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
"""Stable preparation-manifest acceptance without runtime communication resources."""

from dataclasses import replace
from functools import partial
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from torch import nn

from hyper_parallel.components.checkpoint.weight_conversion import Transpose, WeightConverter
from hyper_parallel.components.modules import family as family_registry
from hyper_parallel.components.modules.family import VariantRule
from hyper_parallel.components.modules.module_family import (
    build_prepared_family_replacement,
    get_family_build_info,
)
from hyper_parallel.distributed.module_binding import ParallelModuleSpec
from hyper_parallel.models import PreparationManifest, get_preparation_manifest
from hyper_parallel.models.module_build import build_module_context, prepare_model_components
from hyper_parallel.models.preparation_manifest import (
    WeightConversionRecord,
    collect_replacement_applications,
    record_native_bindings,
)
from hyper_parallel.models.replacement import (
    ModuleReplacementRequest,
    apply_module_replacements,
    compile_module_replacements,
)
from tests.common.module_family_fixtures import (
    FamilyModel,
    FamilyModelConfig,
    LinearBlockFamily,
    LinearConfig,
    LinearFamily,
    create_family_model,
    register_linear_family,
)


class TestPreparationManifest(unittest.TestCase):
    """Validate stable identities, ownership, and atomic application records."""

    def setUp(self) -> None:
        """Start every case from the same small family registry."""
        family_registry._FAMILIES.clear()  # pylint: disable=protected-access
        register_linear_family()

    def tearDown(self) -> None:
        """Do not leak test families into another module."""
        family_registry._FAMILIES.clear()  # pylint: disable=protected-access

    def test_manifest_serializes_importable_callable_and_partial(self) -> None:
        """Callable construction data has structural identities, not repr strings."""
        manifest = PreparationManifest(
            model_spec_id="test:FamilyModel",
            declarations=({
                "factory": create_family_model,
                "power": partial(pow, 2),
            },),
        )

        payload = manifest.to_dict()
        self.assertEqual(
            payload["declarations"][0]["factory"],
            {
                "module": "tests.common.module_family_fixtures",
                "qualname": "create_family_model",
            },
        )
        self.assertEqual(payload["declarations"][0]["power"]["target"]["qualname"], "pow")
        json.dumps(payload)

    def test_manifest_rejects_closure_and_runtime_object(self) -> None:
        """Unimportable decisions and live modules fail instead of leaking through repr."""
        def local_factory():
            return None
        with self.assertRaisesRegex(ValueError, "not stably importable"):
            PreparationManifest("test", declarations=({"factory": local_factory},))
        with self.assertRaisesRegex(TypeError, "unsupported runtime value"):
            PreparationManifest("test", declarations=({"module": nn.Linear(2, 2)},))

    def test_direct_family_manifest_uses_stable_family_identity(self) -> None:
        """Normal preparation publishes the already-resolved direct member and selectors."""
        declaration = family_registry.get_module_family(LinearFamily)
        family_registry._FAMILIES.clear()  # pylint: disable=protected-access
        family_registry.register_module_family(replace(
            declaration,
            family_id="test.linear",
            schema_version=2,
        ))
        model, _ = prepare_model_components(
            FamilyModel(FamilyModelConfig()),
            build_module_context(None),
        )

        manifest = get_preparation_manifest(model)
        component = next(item for item in manifest.components if "projection" in item.aliases)
        self.assertEqual(component.origin, "direct")
        self.assertEqual((component.family_id, component.family_version), ("test.linear", 2))
        self.assertEqual(component.resolved_member.qualname, "LocalLinear")
        self.assertEqual(component.emission_owner_boundary_id, component.boundary_id)
        json.dumps(manifest.to_dict())

    def test_source_replacement_records_only_after_atomic_install(self) -> None:
        """A failed candidate installation cannot publish a replacement application."""
        model = nn.Sequential(nn.Linear(4, 3))
        request = ModuleReplacementRequest(("0",), LinearFamily)
        plan = compile_module_replacements(model, defaults=(request,))

        with collect_replacement_applications() as applications, patch(
            "hyper_parallel.models.replacement._install_module_candidates",
            side_effect=RuntimeError("install failed"),
        ), self.assertRaisesRegex(RuntimeError, "install failed"):
            apply_module_replacements(model, plan, weights_mapping=[])
        self.assertEqual(applications, [])

        model = nn.Sequential(nn.Linear(4, 3))
        context = build_module_context(SimpleNamespace(module_replacement_requests=(request,)))
        model, _ = prepare_model_components(model, context)
        manifest = get_preparation_manifest(model)
        application = manifest.replacements[0]
        component = next(item for item in manifest.components if item.boundary_id == application.boundary_id)
        self.assertEqual(component.origin, "replacement")
        self.assertEqual(component.source_type.qualname, "Linear")
        self.assertEqual(component.resolved_member.qualname, "LocalLinear")

    def test_prepare_replays_local_and_ep_members_without_selecting(self) -> None:
        """Replay the installed member with its exact source and recorded selection."""
        request = ModuleReplacementRequest(("0",), LinearFamily)
        for ep_size, expected_variant in ((1, "default"), (2, "parallel")):
            with self.subTest(ep_size=ep_size):
                model = nn.Sequential(nn.Linear(4, 3))
                context = build_module_context(SimpleNamespace(
                    mesh_context=SimpleNamespace(ep_size=ep_size),
                    module_replacement_requests=(request,),
                ))
                installed, _ = prepare_model_components(model, context)
                application = get_preparation_manifest(installed).replacements[0]
                prepared = application.to_legacy_dict()["family_request"]["selection"]
                source = nn.Linear(4, 3)

                with patch("hyper_parallel.components.modules.module_family.resolve_family",
                           side_effect=AssertionError("member selection should not run")):
                    replayed = build_prepared_family_replacement(module=source, module_fqn="0", prepared=prepared)

                self.assertIs(type(replayed), type(installed[0]))
                self.assertIs(replayed.weight, source.weight)
                self.assertEqual(replayed.module_fqn, "0")
                self.assertEqual(get_family_build_info(replayed).resolved.variant, expected_variant)
                self.assertEqual(get_family_build_info(replayed).resolved.axis_sizes["ep"], ep_size)
                with self.assertRaisesRegex(ValueError, "no longer matches its registration"):
                    build_prepared_family_replacement(
                        module=source, module_fqn="0", prepared={**prepared, "version": -1},
                    )

    def test_nested_component_ownership_stays_hierarchical(self) -> None:
        """A direct child family retains layout ownership below its direct parent."""
        declaration = family_registry.get_module_family(LinearFamily)
        family_registry.register_module_family(replace(
            declaration,
            name="test_linear_block",
            baseline=LinearBlockFamily,
            variants={
                "default": "tests.common.module_family_fixtures.LocalLinearBlock",
                "parallel": "tests.common.module_family_fixtures.ParallelLinearBlock",
            },
            selector_specs={},
            variant_rules=(VariantRule({"parallel": True}, "parallel"),),
            family_id="test.linear_block",
        ))
        model = nn.Module()
        model.block = LinearBlockFamily(config=LinearConfig())
        model.external_alias = model.block.projection
        context = build_module_context(SimpleNamespace(mesh_context=SimpleNamespace(ep_size=2)))
        model, _ = prepare_model_components(model, context)

        components = get_preparation_manifest(model).components
        parent = next(item for item in components if item.family_id == "test.linear_block")
        child = next(item for item in components if item.family_id == "test_linear_family")
        self.assertEqual(child.parent_boundary_id, parent.boundary_id)
        self.assertEqual(child.emission_owner_boundary_id, child.boundary_id)
        self.assertEqual(set(child.aliases), {"block.alias", "block.projection", "external_alias"})

    def test_native_binding_enriches_shared_manifest_without_resources(self) -> None:
        """The shared planner adds protocol facts while retaining family layout ownership."""
        model, _ = prepare_model_components(
            FamilyModel(FamilyModelConfig()),
            build_module_context(None),
        )
        parallel = ParallelModuleSpec(
            protocol="test.linear.local",
            stages={"forward": ("input",)},
            options={"mode": "local"},
        )
        manifest = record_native_bindings(
            model,
            SimpleNamespace(parallel_modules={"projection": parallel}),
        )

        binding = manifest.native_bindings[0]
        component = next(item for item in manifest.components if "projection" in item.aliases)
        self.assertEqual(binding.layout_owner_boundary_id, component.boundary_id)
        self.assertEqual(binding.module_type.qualname, "LocalLinear")
        json.dumps(manifest.to_dict())

    def test_weight_conversion_records_stable_operation_types(self) -> None:
        """Checkpoint transforms retain patterns and operation identities without tensors."""
        transform = WeightConverter(
            source_patterns=["gate.weight", "up.weight"],
            target_patterns="linear_fc1.weight",
            operations=[Transpose(0, 1)],
        )

        record = WeightConversionRecord.from_transform(transform)
        self.assertEqual(record.target_patterns, ("linear_fc1.weight",))
        self.assertEqual(record.operations[0].qualname, "Transpose")
        json.dumps(record.to_dict())


if __name__ == "__main__":
    unittest.main()
