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
"""Real model construction, effective specs, atomic source conversion and codegen."""

from dataclasses import replace
from typing import Any, Mapping, Optional
import importlib
from pathlib import Path
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import BertConfig

from hyper_parallel.components.modules import family as family_registry
from hyper_parallel.components.modules.family import VariantRule
from hyper_parallel.components.modules.module_family import get_family_build_info
from hyper_parallel.models import HyperAutoModelForCausalLM, HyperAutoModelForSequenceClassification, registry
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.module_build import (
    build_module_context, construct_family_model, resolve_effective_model_spec,
    prepare_model_components,
)
from hyper_parallel.models.replacement import (
    ModuleReplacementRequest, ModuleReplacementSpec, _apply_module_replacement_actions, _install_module_candidates,
    _named_tensor_shapes, apply_module_replacements,
    compile_module_replacements, module_replacement,
)
from hyper_parallel.trainer.config.parallelism import normalize_distributed_setup_overrides
from tests.common.mark_utils import arg_mark
from tests.ut.dual_mode_dtensor.test_module_replacement import _AllocatedMappedLinear, _replace_linear
from tests.common.module_family_fixtures import (
    FamilyModel, FamilyModelConfig, LinearBlockFamily, LinearConfig, LinearFamily, LocalLinear, MegaLinear,
    ParallelLinear, ParallelLinearBlock, StatefulLocalLinear, StatefulParallelLinear,
    create_family_model, register_linear_family,
    get_classifier_replacements,
)


class TestModuleBuild(unittest.TestCase):
    """Exercise the complete non-distributed mechanisms with real Torch math."""

    def setUp(self) -> None:
        """Isolate registries, builtin discovery and the random generator."""
        for mapping in (family_registry._FAMILIES, registry.MODEL_ADAPTER_REGISTRY, registry._FAMILY_ALIASES):
            handle = patch.dict(mapping, clear=True)
            handle.start()
            self.addCleanup(handle.stop)
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        register_linear_family()

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_direct_facade_creates_once_without_replacement_engine(self):
        """Feature: Direct module-family models use the actual model facade.
        Description: Construction precedes the shared initialization and parameter pipeline.
        Expectation: Factory runs once, source replacement never runs, and model gradients are real.
        """
        auto = importlib.import_module("hyper_parallel.models._transformers.auto_model")
        builder = importlib.import_module("hyper_parallel.models.module_build")
        with patch.object(auto, "_current_device", return_value=torch.device("cpu")), \
                patch.object(builder, "_apply_module_replacement_actions", side_effect=AssertionError("replacement")), \
                patch.object(auto, "construct_family_model", wraps=construct_family_model) as construct:
            model = HyperAutoModelForCausalLM.from_config(
                FamilyModelConfig(), model_factory=create_family_model, torch_dtype=torch.float32,
            )
        self.assertEqual(construct.call_count, 1)
        x = torch.ones(2, 4, requires_grad=True)
        torch.testing.assert_close(model(x), torch.full((2, 3), 0.75))
        model(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.full((2, 4), 0.375))
        self.assertIs(type(model.projection), LocalLinear)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_public_build_performs_each_preparation_work_once(self) -> None:
        """Feature: One common preparation pass for factory and source entry points.
        Description: Exercise public facades while observing traversal and conversion-provider calls.
        Expectation: Each build finalizes and reads weight mappings once; empty method rules do no work.
        """
        auto = importlib.import_module("hyper_parallel.models._transformers.auto_model")
        builder = importlib.import_module("hyper_parallel.models.module_build")
        infrastructure = importlib.import_module("hyper_parallel.models._transformers.model_builder")
        for direct in (True, False):
            with self.subTest(direct=direct), \
                    patch.object(auto, "_current_device", return_value=torch.device("cpu")), \
                    patch.object(builder, "_finalize_family_modules",
                                 wraps=builder._finalize_family_modules) as finalize, \
                    patch.object(builder, "model_weight_mapping", wraps=builder.model_weight_mapping) as mapping, \
                    patch.object(infrastructure, "build_module_context",
                                 side_effect=AssertionError("empty rules")):
                if direct:
                    model = HyperAutoModelForCausalLM.from_config(
                        FamilyModelConfig(), model_factory=create_family_model, torch_dtype=torch.float32,
                    )
                    torch.testing.assert_close(model(torch.ones(2, 4)), torch.full((2, 3), 0.75))
                else:
                    config = BertConfig.from_dict({
                        "vocab_size": 16, "hidden_size": 8, "num_hidden_layers": 1,
                        "num_attention_heads": 2, "intermediate_size": 16,
                        "architectures": ["BertForSequenceClassification"],
                    })
                    model = HyperAutoModelForSequenceClassification.from_config(config, torch_dtype=torch.float32)
                    self.assertEqual(tuple(model(torch.tensor([[1, 2]])).logits.shape), (1, 2))
                self.assertTrue(model._hp_components_prepared)
                self.assertEqual(finalize.call_count, 1)
                self.assertEqual(mapping.call_count, 1)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_parallel_candidate_reuses_the_resolved_choice(self) -> None:
        """Feature: Resolve each changed family boundary once.
        Description: Prepare an existing local model without re-entering public family selection.
        Expectation: The candidate reuses the resolution and preserves the original weight object.
        """
        builder = importlib.import_module("hyper_parallel.models.module_build")
        family = importlib.import_module("hyper_parallel.components.modules.module_family")
        model = FamilyModel(FamilyModelConfig())
        weight = model.projection.weight
        context = dict(build_module_context(SimpleNamespace(mesh_context=SimpleNamespace(ep_size=2))))
        with patch.object(builder, "resolve_family", wraps=builder.resolve_family) as resolve, \
                patch.object(family, "resolve_family", side_effect=AssertionError("duplicate selection")), \
                patch.object(family, "_constructor_arguments", wraps=family._constructor_arguments) as arguments:
            model, _ = prepare_model_components(model, context)
        self.assertEqual(resolve.call_count, 1)
        constructor_context = arguments.call_args.args[-1]
        self.assertIsInstance(constructor_context, MappingProxyType)
        self.assertIsNot(constructor_context, context)
        with self.assertRaises(TypeError):
            constructor_context["ep_size"] = 4
        self.assertIs(type(model.projection), ParallelLinear)
        self.assertIs(model.projection.weight, weight)
        torch.testing.assert_close(model(torch.ones(2, 4)), torch.full((2, 3), 0.75))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_prebuilt_family_selects_parallel_member_and_retains_weights(self) -> None:
        """Feature: Preconstructed families select CP or EP from the job configuration.
        Description: Prepare a local model with shared, frozen and previously loaded weights.
        Expectation: Parallel members preserve tensor identities, aliases, values and direct ownership.
        """
        for axis in ("cp", "ep"):
            with self.subTest(axis=axis), patch.dict(family_registry._FAMILIES, clear=True):
                register_linear_family(axis=axis)
                model = FamilyModel(FamilyModelConfig()).to(torch.float64).eval()
                model.alias = model.projection
                model.tied = nn.Linear(4, 3, dtype=torch.float64)
                model.tied.weight = model.projection.weight
                model.projection.bias.requires_grad_(False)
                with torch.no_grad():
                    model.projection.weight.fill_(0.7)
                source = model.projection
                identities = {name: id(value) for name, value in model.named_parameters(remove_duplicate=False)}
                inputs = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
                expected = model(inputs).detach()
                context = build_module_context(SimpleNamespace(mesh_context=SimpleNamespace(**{f"{axis}_size": 2})))
                with patch("hyper_parallel.models.module_build._apply_module_replacement_actions",
                           side_effect=AssertionError("source replacement")), \
                        patch("torch.distributed.new_group") as groups:
                    prepared, _ = prepare_model_components(model, context)
                self.assertIs(prepared, model)
                self.assertIs(type(model.projection), ParallelLinear)
                self.assertIsNot(model.projection, source)
                self.assertIs(model.projection, model.alias)
                self.assertIs(model.projection.weight, model.tied.weight)
                self.assertEqual(identities, {name: id(value)
                                              for name, value in model.named_parameters(remove_duplicate=False)})
                self.assertFalse(model.projection.training)
                self.assertFalse(model.projection.bias.requires_grad)
                self.assertEqual(get_family_build_info(model.projection).origin, "direct")
                torch.testing.assert_close(model(inputs), expected)
                model(inputs).sum().backward()
                torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 2.1))
                groups.assert_not_called()
                self.assertIs(prepare_model_components(model, context)[0].projection, model.projection)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_factory_need_not_forward_context_to_nested_family(self) -> None:
        """Feature: Family preparation applies job settings to ordinary model constructors.
        Description: A factory returns a model whose constructor calls the family without context.
        Expectation: Construction remains local, and common preparation selects the configured EP member once.
        """
        calls = []
        def factory(config: FamilyModelConfig, *, context: Mapping[str, Any]) -> nn.Module:
            """Use a model constructor without requiring explicit context propagation."""
            del context
            calls.append(config)
            return FamilyModel(config)
        context = build_module_context(SimpleNamespace(mesh_context=SimpleNamespace(ep_size=2)))
        with patch("hyper_parallel.models.module_build._apply_module_replacement_actions",
                   side_effect=AssertionError("source replacement")):
            model = construct_family_model(factory, FamilyModelConfig(), context=context)
            self.assertIs(type(model.projection), LocalLinear)
            self.assertFalse(getattr(model, "_hp_components_prepared", False))
            model, _ = prepare_model_components(model, context)
        self.assertIs(type(model.projection), ParallelLinear)
        self.assertEqual(len(calls), 1)
        self.assertEqual(get_family_build_info(model.projection).resolved.axis_sizes["ep"], 2)
        torch.testing.assert_close(model(torch.ones(2, 4)), torch.full((2, 3), 0.75))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_factory_cannot_rebind_an_already_prepared_model(self) -> None:
        """Feature: Preparation context remains owned by its original build.
        Description: A factory returns a model already prepared for another job context.
        Expectation: Construction rejects rebinding before changing context, members or parameters.
        """
        model = FamilyModel(FamilyModelConfig())
        original_context = build_module_context()
        model, _ = prepare_model_components(model, original_context)
        weight = model.projection.weight

        def factory(config: FamilyModelConfig, *, context: Mapping[str, Any]) -> nn.Module:
            """Return an existing model to exercise the construction boundary."""
            del config, context
            return model

        new_context = build_module_context(SimpleNamespace(mesh_context=SimpleNamespace(ep_size=2)))
        with self.assertRaisesRegex(ValueError, "another build context"):
            construct_family_model(factory, FamilyModelConfig(), context=new_context)
        self.assertIs(model._hp_module_build_context, original_context)
        self.assertIs(type(model.projection), LocalLinear)
        self.assertIs(model.projection.weight, weight)
        self.assertIs(construct_family_model(factory, FamilyModelConfig(), context=original_context), model)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_root_family_configuration_preserves_source_parameters(self) -> None:
        """Feature: A family can itself be the root model.
        Description: Apply EP configuration to a root created through the local public family.
        Expectation: The returned root selects the parallel class and retains the loaded Parameter objects.
        """
        original = LinearFamily(config=LinearConfig())
        weight = original.weight
        context = build_module_context(SimpleNamespace(mesh_context=SimpleNamespace(ep_size=2)))
        model, _ = prepare_model_components(original, context)
        self.assertIs(type(model), ParallelLinear)
        self.assertIs(model.weight, weight)
        self.assertIs(model._hp_module_build_context, context)
        with self.assertRaisesRegex(ValueError, "another build context"):
            prepare_model_components(model, build_module_context())

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_invalid_family_configuration_does_not_install_any_candidate(self) -> None:
        """Feature: Automatic family selection validates the complete model before installation.
        Description: A later family validator rejects the configured EP job.
        Expectation: Earlier candidates, metadata, loaded values and aliases remain unchanged.
        """
        def validate(*, config: LinearConfig, module: Optional[nn.Module],
                     context: Mapping[str, Any], member: Any) -> None:
            """Reject only the second component when EP is enabled."""
            del module, context
            if config.output_size == 5 and member.selectors["parallel"]:
                raise ValueError("unsupported second component")
        declaration = family_registry.get_module_family(LinearFamily)
        family_registry._FAMILIES.clear()
        family_registry.register_module_family(replace(declaration, validate=validate))
        model = nn.Sequential(LinearFamily(config=LinearConfig()), LinearFamily(config=LinearConfig(output_size=5)))
        originals = tuple(model)
        metadata = tuple(get_family_build_info(child) for child in model)
        context = build_module_context(SimpleNamespace(mesh_context=SimpleNamespace(ep_size=2)))
        with self.assertRaisesRegex(ValueError, "unsupported second component"):
            prepare_model_components(model, context)
        self.assertEqual(tuple(model), originals)
        self.assertEqual(tuple(get_family_build_info(child) for child in model), metadata)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_config_only_member_preserves_buffers_extra_state_and_meta(self) -> None:
        """Feature: Config-only families retain their complete logical state during selection.
        Description: Prepare a real or meta member with a persistent buffer and non-tensor checkpoint state.
        Expectation: EP selection preserves buffers, checkpoint state, weights and the original tensor device.
        """
        declaration = family_registry.get_module_family(LinearFamily)
        variants = {**declaration.variants,
                    "default": "tests.common.module_family_fixtures.StatefulLocalLinear",
                    "parallel": "tests.common.module_family_fixtures.StatefulParallelLinear"}
        family_registry._FAMILIES.clear()
        family_registry.register_module_family(replace(declaration, variants=variants))
        context = build_module_context(SimpleNamespace(mesh_context=SimpleNamespace(ep_size=2)))
        for device in ("cpu", "meta"):
            with self.subTest(device=device), torch.device(device):
                source = LinearFamily(config=LinearConfig())
                self.assertIs(type(source), StatefulLocalLinear)
                source.samples_seen = 17
                with torch.no_grad():
                    source.output_scale.fill_(3)
                result, _ = prepare_model_components(source, context)
                self.assertIs(type(result), StatefulParallelLinear)
                self.assertIs(result.output_scale, source.output_scale)
                self.assertIs(result.weight, source.weight)
                self.assertEqual(result.samples_seen, 17)
                self.assertEqual(result.weight.device.type, device)
                if device == "cpu":
                    torch.testing.assert_close(result(torch.ones(2, 4)), torch.full((2, 3), 2.25))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_nested_families_keep_independent_choices_and_all_aliases(self) -> None:
        """Feature: Automatic preparation supports nested family boundaries.
        Description: Configure a parent family whose child is aliased inside and outside that parent.
        Expectation: Both families select EP members while every child alias and its original weight survive.
        """
        declaration = family_registry.get_module_family(LinearFamily)
        variants = {"default": "tests.common.module_family_fixtures.LocalLinearBlock",
                    "parallel": "tests.common.module_family_fixtures.ParallelLinearBlock"}
        family_registry.register_module_family(replace(
            declaration, name="test_linear_block", baseline=LinearBlockFamily, variants=variants,
            selector_specs={}, variant_rules=(VariantRule({"parallel": True}, "parallel"),),
        ))
        model = nn.Module()
        model.block = LinearBlockFamily(config=LinearConfig())
        model.external_alias = model.block.projection
        weight = model.external_alias.weight
        context = build_module_context(SimpleNamespace(mesh_context=SimpleNamespace(ep_size=2)))
        result, _ = prepare_model_components(model, context)
        self.assertIs(type(result.block), ParallelLinearBlock)
        self.assertIs(type(result.block.projection), ParallelLinear)
        self.assertIs(result.block.projection, result.block.alias)
        self.assertIs(result.block.projection, result.external_alias)
        self.assertIs(result.external_alias.weight, weight)
        torch.testing.assert_close(result.block(torch.ones(2, 4)), torch.full((2, 3), 0.75))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_active_family_cannot_change_topology_after_binding(self) -> None:
        """Feature: Automatic selection finishes before runtime communication starts.
        Description: Attempt to configure another EP job after a family has received a runtime binding.
        Expectation: The active module is retained and preparation reports the lifecycle error.
        """
        model = FamilyModel(FamilyModelConfig())
        model.projection._hp_parallel_binding = object()
        original = model.projection
        context = build_module_context(SimpleNamespace(mesh_context=SimpleNamespace(ep_size=2)))
        with self.assertRaisesRegex(ValueError, "before communication binding"):
            prepare_model_components(model, context)
        self.assertIs(model.projection, original)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_incompatible_parallel_weight_contract_is_rejected(self) -> None:
        """Feature: Selecting a parallel member preserves the family's logical weight representation.
        Description: A defective parallel constructor changes the projection's global width.
        Expectation: Preparation rejects the changed shape and retains the original member and loaded weights.
        """
        class IncompatibleParallelLinear(LocalLinear):
            """Parallel member with a deliberately incorrect logical projection width."""

            def __init__(self, *, config: LinearConfig) -> None:
                """Construct incompatible weights to exercise state-contract validation."""
                config = replace(config, width=config.width + 1)
                super().__init__(config=config)
        model = FamilyModel(FamilyModelConfig())
        original = model.projection
        weight = original.weight
        context = build_module_context(SimpleNamespace(mesh_context=SimpleNamespace(ep_size=2)))
        with patch("tests.common.module_family_fixtures.ParallelLinear", IncompatibleParallelLinear), \
                self.assertRaisesRegex(ValueError, "changed logical parameter shape"):
            prepare_model_components(model, context)
        self.assertIs(model.projection, original)
        self.assertIs(model.projection.weight, weight)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_effective_spec_is_local_and_reaches_model_consumers(self):
        """Feature: Per-build model facts have one source of truth.
        Description: Explicit specs replace builtin providers without modifying global registration.
        Expectation: Model consumers resolve the same explicit object and unrelated jobs retain the builtin.
        """
        config = FamilyModelConfig()
        builtin = ModelAdapterSpec("FamilyModel", config.model_type)
        custom = replace(builtin, replacements=lambda: ())
        registry.register_model_adapter(builtin)
        context = build_module_context(model_spec=resolve_effective_model_spec(config, custom))
        model = construct_family_model(create_family_model, config, context=context)
        self.assertIs(registry.resolve_model_adapter(model), custom)
        self.assertIs(registry.get_model_adapter(config.model_type), builtin)
        self.assertIs(resolve_effective_model_spec(config), builtin)
        with self.assertRaisesRegex(ValueError, "does not match"):
            resolve_effective_model_spec(config, ModelAdapterSpec("Other", "other"))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_materialization_and_external_state_share_effective_spec_resolution(self) -> None:
        """Feature: Model consumers use the shared adapter resolver.
        Description: Exercise builtin, explicit, explicit-None and architecture-fallback identities.
        Expectation: Both consumers honor the effective spec without reintroducing builtin providers.
        """
        infrastructure = importlib.import_module("hyper_parallel.models._transformers.model_builder")
        events = []
        builtin = ModelAdapterSpec(
            "FamilyModel", "test_family_model", materialization=lambda model: events.append("builtin-materialize"),
            external_state=lambda model, context: events.append("builtin-external"),
        )
        custom = replace(builtin, materialization=lambda model: events.append("custom-materialize"),
                         external_state=lambda model, context: events.append("custom-external"))
        registry.register_model_adapter(builtin)
        for choice in ("builtin", "custom", "none", "architecture"):
            with self.subTest(choice=choice):
                events.clear()
                model = nn.Module()
                model.config = FamilyModelConfig()
                if choice in ("custom", "none"):
                    model._hp_model_adapter_spec = custom if choice == "custom" else None
                if choice == "architecture":
                    model.config = SimpleNamespace(model_type="unknown", architectures=("Missing", "FamilyModel"))
                infrastructure._apply_materialization_adapter(model)
                external = infrastructure._create_external_state(
                    model, mesh=None, source_shard_info=None, device=None, is_meta_device=False,
                    model_init_dtype="float32", validate_placement=False, compile_for_execution=False,
                    distributed_setup=None, fsdp2_manager=None,
                )
                self.assertIsNone(external)
                prefix = "custom" if choice == "custom" else "builtin"
                expected = [] if choice == "none" else [prefix + "-materialize", prefix + "-external"]
                self.assertEqual(events, expected)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_defaults_and_user_choices_merge_by_identity(self):
        """Feature: Default source conversion and partial user overrides.
        Description: Aliased paths identify one object, while same-family user parameters replace defaults.
        Expectation: One installed object preserves aliases and shares untouched source parameters.
        """
        model = nn.Module()
        model.first = nn.Linear(4, 3)
        model.alias = model.first
        weight = model.first.weight
        default = ModuleReplacementRequest(("first",), LinearFamily, {"kernel": "megakernel"})
        user = ModuleReplacementRequest(("alias",), LinearFamily, {"precision": "inherit"}, origin="user")
        rules = compile_module_replacements(model, defaults=(default,), overrides=(user,))
        self.assertEqual(len(rules.targets), 1)
        result, _ = _apply_module_replacement_actions(model, rules, context={"ep_size": 1})
        self.assertIs(result.first, result.alias)
        self.assertIs(result.first.weight, weight)
        self.assertIs(type(result.first), MegaLinear)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_direct_boundaries_are_excluded_and_conflicting_ownership_fails(self):
        """Feature: A model may contain direct families and untouched source modules.
        Description: Default globs skip direct boundaries and explicit duplicate ownership is rejected.
        Expectation: Only the remaining source converts; parent or native-like repeated takeover fails early.
        """
        model = nn.Sequential(LinearFamily(config=LinearConfig()), nn.Linear(4, 3))
        default = ModuleReplacementRequest(("*",), LinearFamily)
        rules = compile_module_replacements(model, defaults=(default,))
        self.assertEqual(len(rules.targets), 1)
        self.assertEqual(rules.targets[0].module_fqns, ("1",))
        with self.assertRaisesRegex(ValueError, "directly constructed"):
            compile_module_replacements(model, overrides=(ModuleReplacementRequest(("0",), LinearFamily),))
        parent = nn.Sequential(model)
        with self.assertRaisesRegex(ValueError, "parent replacement"):
            compile_module_replacements(parent, defaults=(ModuleReplacementRequest(("0",), LinearFamily),))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_matching_is_reused_across_request_layers_and_scoped_to_one_build(self) -> None:
        """Feature: One shared pattern expansion for default and user rules.
        Description: Compile equal patterns twice, then compile again after adding another module.
        Expectation: Each build matches each path once and a later build observes the updated inventory.
        """
        replacement = importlib.import_module("hyper_parallel.models.replacement")
        model = nn.ModuleDict({"first": nn.Linear(4, 3), "second": nn.Linear(4, 3)})
        default = ModuleReplacementRequest(("*",), LinearFamily, {"kernel": "megakernel"})
        override = ModuleReplacementRequest(("*",), LinearFamily, {"precision": "inherit"})
        with patch.object(replacement.fnmatch, "fnmatchcase", wraps=replacement.fnmatch.fnmatchcase) as match:
            first = compile_module_replacements(model, defaults=(default,), overrides=(override,))
            self.assertEqual(match.call_count, 2)
            model["third"] = nn.Linear(4, 3)
            final = compile_module_replacements(model, defaults=(default,), overrides=(override,))
            self.assertEqual(match.call_count, 5)
        self.assertEqual(len(first.targets), 2)
        self.assertEqual(len(final.targets), 3)
        with patch.object(replacement, "compile_module_replacements", side_effect=AssertionError("recompiled")):
            model, _ = _apply_module_replacement_actions(model, final)
        for child in model.values():
            self.assertIs(type(child), MegaLinear)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_requests_and_specs_share_matching_and_keep_error_contracts(self) -> None:
        """Feature: Source matching preserves request filtering and executable-spec diagnostics.
        Description: Match concrete, derived and incompatible types plus an unmatched pattern.
        Expectation: Requests filter types; executable specs reject wrong types and every unmatched pattern.
        """
        model = nn.Sequential(nn.Linear(4, 3), nn.LazyLinear(3), nn.Identity())
        request = ModuleReplacementRequest(("*", "missing"), LinearFamily, module_type=nn.Linear)
        plan = compile_module_replacements(model, defaults=(request,))
        self.assertEqual([target.module_fqns for target in plan.targets], [("0",), ("1",)])
        exact = compile_module_replacements(model, defaults=(replace(request, exact_type=True),))
        self.assertEqual([target.module_fqns for target in exact.targets], [("0",)])
        with self.assertRaisesRegex(TypeError, "expected Linear"):
            compile_module_replacements(model, (ModuleReplacementSpec(("*",), LinearFamily, nn.Linear),))
        with self.assertRaisesRegex(ValueError, "matched no final module"):
            compile_module_replacements(model, (ModuleReplacementSpec(("0", "missing"), LinearFamily),))
        with self.assertRaisesRegex(ValueError, "matched no compatible module"):
            compile_module_replacements(model, defaults=(replace(request, match=("2",)),))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_opaque_overrides_reuse_default_matches_and_keep_one_target_per_source(self) -> None:
        """Feature: Opaque specs and neutral requests share the same matcher.
        Description: Repeat an executable spec that overrides defaults with the same pattern.
        Expectation: Matching runs once, the default is superseded and the final plan stays incremental.
        """
        replacement = importlib.import_module("hyper_parallel.models.replacement")
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(4, 3))
        default = ModuleReplacementRequest(("*",), LinearFamily)
        spec = ModuleReplacementSpec(("*",), LinearFamily)
        with patch.object(replacement.fnmatch, "fnmatchcase", wraps=replacement.fnmatch.fnmatchcase) as match:
            plan = compile_module_replacements(model, (spec, spec), defaults=(default,))
        self.assertEqual(match.call_count, 2)
        self.assertEqual(len(plan.targets), 2)
        self.assertFalse(plan.atomic)
        self.assertTrue(all(target.spec is spec for target in plan.targets))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_new_request_batch_validates_all_candidates_before_install(self):
        """Feature: New declarative source conversion is transactional.
        Description: Failure constructing a later sibling must not install an earlier candidate.
        Expectation: The original tree and source Parameter identities remain available after failure.
        """
        @module_replacement
        def choose(*, module: nn.Module, module_fqn: str, context: Mapping[str, Any]) -> nn.Module:
            """Fail on the later sibling to test all-candidate preparation."""
            del context
            if module_fqn == "1":
                raise ValueError("candidate failed")
            return LocalLinear(module=module)
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(4, 3))
        originals = tuple(model)
        rules = compile_module_replacements(model, defaults=(ModuleReplacementRequest(("*",), choose),))
        with self.assertRaisesRegex(ValueError, "candidate failed"):
            _apply_module_replacement_actions(model, rules)
        self.assertIs(model[0], originals[0])
        self.assertIs(model[1], originals[1])

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_shared_installer_handles_root_nested_members_and_external_aliases(self) -> None:
        """Feature: Both assembly paths share root, nesting and alias installation.
        Description: Replace a root, its block and a child also aliased outside that block in one batch.
        Expectation: All slots on the final tree share the selected child; the original tree is untouched.
        """
        model = nn.Module()
        model.block = nn.Sequential(nn.Linear(4, 3))
        model.alias = model.block[0]
        original = model.block[0]
        root = nn.Module()
        root.block = nn.Identity()
        root.alias = nn.Identity()
        block = nn.Sequential(nn.Identity())
        child = nn.Linear(4, 3)
        result = _install_module_candidates(model, (
            (("",), model, root), (("block",), model.block, block),
            (("block.0", "alias"), original, child),
        ))
        self.assertIs(result, root)
        self.assertIs(result.block, block)
        self.assertIs(result.block[0], child)
        self.assertIs(result.alias, child)
        self.assertIs(model.block[0], original)
        self.assertIs(model.alias, original)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_shared_installer_preflights_projected_slots_and_stale_aliases(self) -> None:
        """Feature: Shared installation validates the full batch before writing any slot.
        Description: An invalid final parent or stale late alias follows an otherwise valid replacement.
        Expectation: Neither failure installs an earlier candidate or modifies candidate parents.
        """
        for stale_alias in (False, True):
            with self.subTest(stale_alias=stale_alias):
                model = nn.Module()
                model.first = nn.Identity()
                model.block = nn.Sequential(nn.Identity())
                old_first, old_block = model.first, model.block
                parent = nn.Sequential(nn.Identity()) if stale_alias else nn.Module()
                candidates = [(("first",), old_first, nn.Identity()), (("block",), old_block, parent)]
                if stale_alias:
                    model.alias = nn.Identity()
                    candidates.append((("block.0", "alias"), old_block[0], nn.Identity()))
                else:
                    candidates.append((("block.0",), old_block[0], nn.Identity()))
                with self.assertRaisesRegex(ValueError, "no longer registered|changed while"):
                    _install_module_candidates(model, candidates)
                self.assertIs(model.first, old_first)
                self.assertIs(model.block, old_block)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_source_preparation_compiles_once_and_keeps_literal_module_paths(self) -> None:
        """Feature: Compose requests directly into one executable plan.
        Description: A wildcard matches a registered name containing glob metacharacters.
        Expectation: Matched paths are not parsed again and the shared installer preserves loaded weights.
        """
        builder = importlib.import_module("hyper_parallel.models.module_build")
        replacement = importlib.import_module("hyper_parallel.models.replacement")
        model = nn.ModuleDict({"a[0]": nn.Linear(4, 3)})
        source = model["a[0]"]
        request = ModuleReplacementRequest(("*",), LinearFamily)
        context = build_module_context(SimpleNamespace(module_replacement_requests=(request,)))
        with patch.object(replacement, "_module_aliases", wraps=replacement._module_aliases) as inventory, \
                patch.object(builder, "compile_module_replacements",
                             wraps=compile_module_replacements) as compile_plan, \
                patch.object(replacement, "_install_module_candidates",
                             wraps=replacement._install_module_candidates) as install:
            model, _ = prepare_model_components(model, context)
        self.assertEqual(inventory.call_count, 1)
        self.assertEqual(compile_plan.call_count, 1)
        self.assertEqual(install.call_count, 1)
        self.assertIs(type(model["a[0]"]), LocalLinear)
        self.assertIs(model["a[0]"].weight, source.weight)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_shared_installer_detects_aliases_of_final_parent_slots(self) -> None:
        """Feature: Projected parent aliases cannot overwrite distinct selected members.
        Description: Two candidate parents share an interior container addressed by separate old paths.
        Expectation: Distinct choices fail before writes; identical choices install once and retain aliases.
        """
        for same_candidate in (False, True):
            with self.subTest(same_candidate=same_candidate):
                model = nn.ModuleDict({"left": nn.Sequential(nn.Sequential(nn.Identity())),
                                       "right": nn.Sequential(nn.Sequential(nn.Identity()))})
                old_left, old_right = model["left"], model["right"]
                shared = nn.Sequential(nn.Identity())
                old_slot = shared[0]
                left, right = nn.Sequential(shared), nn.Sequential(shared)
                first = nn.Identity()
                second = first if same_candidate else nn.Identity()
                candidates = (
                    (("left",), old_left, left), (("right",), old_right, right),
                    (("left.0.0",), old_left[0][0], first), (("right.0.0",), old_right[0][0], second),
                )
                if same_candidate:
                    result = _install_module_candidates(model, candidates)
                    self.assertIs(result["left"][0][0], first)
                    self.assertIs(result["right"][0][0], first)
                else:
                    with self.assertRaisesRegex(ValueError, "conflicting replacement candidates"):
                        _install_module_candidates(model, candidates)
                    self.assertIs(model["left"], old_left)
                    self.assertIs(model["right"], old_right)
                    self.assertIs(shared[0], old_slot)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_shared_installer_rejects_aliases_that_redirect_another_target(self) -> None:
        """Feature: Every selected target remains reachable in the projected final tree.
        Description: A new root alias redirects a descendant through another selected ancestor.
        Expectation: Ambiguous candidate ownership fails before changing the source or candidate trees.
        """
        model = nn.ModuleDict({"a": nn.ModuleDict({"x": nn.Sequential(nn.Identity())}),
                               "b": nn.ModuleDict({"x": nn.Sequential(nn.Identity())})})
        shared = nn.ModuleDict({"x": nn.Sequential(nn.Identity())})
        root = nn.ModuleDict({"a": shared, "b": shared})
        old_slot = shared["x"]
        ancestor = nn.Sequential(nn.Identity())
        descendant = nn.Identity()
        with self.assertRaisesRegex(ValueError, "conflicting replacement paths"):
            _install_module_candidates(model, (
                (("",), model, root), (("a.x",), model["a"]["x"], ancestor),
                (("b.x.0",), model["b"]["x"][0], descendant),
            ))
        self.assertIs(shared["x"], old_slot)
        self.assertIsNot(model["a"], model["b"])
        self.assertIsNot(ancestor[0], descendant)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_opaque_factories_keep_incremental_installation(self) -> None:
        """Feature: Legacy opaque factories retain bounded incremental conversion.
        Description: The second factory observes the first installation and then fails.
        Expectation: Earlier conversion remains installed and the failing boundary retains its source.
        """
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(4, 3))
        originals = tuple(model)

        @module_replacement
        def choose(*, module: nn.Module, module_fqn: str, context: Mapping[str, Any]) -> nn.Module:
            """Observe the preceding installation before failing the next conversion."""
            del context
            if module_fqn == "1":
                self.assertIs(type(model[0]), LocalLinear)
                raise ValueError("opaque factory failed")
            return LocalLinear(module=module)

        plan = compile_module_replacements(model, (ModuleReplacementSpec(("*",), choose),))
        with self.assertRaisesRegex(ValueError, "opaque factory failed"):
            _apply_module_replacement_actions(model, plan)
        self.assertIs(type(model[0]), LocalLinear)
        self.assertIs(model[0].weight, originals[0].weight)
        self.assertIs(model[1], originals[1])

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_new_spec_discovery_is_lazy_and_canonical(self):
        """Feature: Filesystem discovery loads one canonical MODEL_SPEC.
        Description: A model directory and its model_type filename may differ.
        Expectation: Scanning imports nothing; lookup imports once and preserves the declared spec object.
        """
        spec = ModelAdapterSpec("ToyModel", "toy_model")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "toy_dir").mkdir()
            (root / "toy_dir" / "toy_model_spec.py").write_text("MODEL_SPEC = None\n")
            fake_path = SimpleNamespace(resolve=lambda: SimpleNamespace(parent=root))
            with patch.object(registry, "_DISCOVERED_PROVIDERS", None), \
                    patch.object(registry, "Path", return_value=fake_path), \
                    patch.object(registry.importlib, "import_module",
                                 return_value=SimpleNamespace(MODEL_SPEC=spec)) as loader:
                registry._discover_family_providers()
                loader.assert_not_called()
                self.assertIs(registry.get_model_adapter("toy_model"), spec)
                self.assertIs(registry.get_model_adapter("ToyModel"), spec)
                self.assertEqual(loader.call_count, 1)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_source_defaults_preserve_effective_spec_and_numerical_behavior(self):
        """Feature: Source models apply their default family declarations without replacement YAML.
        Description: Build a real tiny HF classifier twice with an explicit per-build spec.
        Expectation: Both builds select the family, preserve the spec and agree after loading identical state.
        """
        auto = importlib.import_module("hyper_parallel.models._transformers.auto_model")
        config = BertConfig.from_dict({"vocab_size": 16, "hidden_size": 8, "num_hidden_layers": 1,
                                       "num_attention_heads": 2, "intermediate_size": 16,
                                       "architectures": ["BertForSequenceClassification"],
                                       "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0})
        spec = ModelAdapterSpec("BertForSequenceClassification", "bert", replacements=get_classifier_replacements)
        with patch.object(auto, "_current_device", return_value=torch.device("cpu")), \
                patch.object(auto, "resolve_effective_model_spec", wraps=resolve_effective_model_spec) as resolve_spec:
            model = HyperAutoModelForSequenceClassification.from_config(
                config, model_spec=spec, torch_dtype=torch.float32,
            )
            self.assertIs(type(model.classifier), LocalLinear)
            self.assertIs(model._hp_model_adapter_spec, spec)
            self.assertEqual(resolve_spec.call_count, 1)
            replica = HyperAutoModelForSequenceClassification.from_config(
                config, model_spec=spec, torch_dtype=torch.float32,
            )
            self.assertIs(type(replica.classifier), LocalLinear)
            replica.load_state_dict(model.state_dict())
            model.eval()
            replica.eval()
            inputs = torch.tensor([[1, 2, 3]])
            torch.testing.assert_close(model(inputs).logits, replica(inputs).logits)
            self.assertEqual(resolve_spec.call_count, 2)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_precision_target_validation_waits_for_direct_members(self) -> None:
        """
        Feature: Direct precision selection does not require replacement YAML.
        Description: Validate an enabled precision policy after final member choices are known.
        Expectation: A declared precision member satisfies the policy; an ordinary member is rejected.
        """
        declaration = family_registry.get_module_family(LinearFamily)
        family_registry._FAMILIES.clear()
        family_registry.register_module_family(replace(
            declaration, variants={**declaration.variants, "test_precision": declaration.variants["default"]},
            variant_rules=(*declaration.variant_rules, VariantRule(
                {"kernel": "default", "precision": "test_precision", "parallel": False}, "test_precision")),
        ))
        setup = SimpleNamespace(mesh_context=None, low_precision_config=SimpleNamespace(enabled=True))
        normalize_distributed_setup_overrides(setup, SimpleNamespace(plan_overrides=[]))
        context = build_module_context(setup)
        selected = nn.Sequential(LinearFamily(config=LinearConfig(), precision="test_precision", context=context))
        ordinary = nn.Sequential(LinearFamily(config=LinearConfig(), context=context))
        for model in (selected, ordinary):
            model._hp_module_build_context = context
        prepare_model_components(selected, context)
        with self.assertRaisesRegex(ValueError, "no low-precision family"):
            prepare_model_components(ordinary, context)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_required_keyword_only_forward_contract(self) -> None:
        """Feature: Replacement signatures preserve required keyword-only arguments.
        Description: Convert a stage requiring scale, then reject a replacement that drops it.
        Expectation: Compatible stages execute and differentiate; incompatible stages fail before installation.
        """
        class RequiredKeyword(nn.Module):
            """Minimal real computation with an explicit keyword-only operand."""

            def forward(self, inputs: torch.Tensor, *, scale: float) -> torch.Tensor:
                """Scale the input using the required operand.

                Args:
                    inputs: Local activations.
                    scale: Required scaling factor.
                """
                return inputs * scale

        @module_replacement
        def preserve(*, module: nn.Module, module_fqn: str, context: Mapping[str, Any]) -> nn.Module:
            """Construct the same callable contract without retaining the source."""
            del module, module_fqn, context
            return RequiredKeyword()

        @module_replacement
        def drop(*, module: nn.Module, module_fqn: str, context: Mapping[str, Any]) -> nn.Module:
            """Deliberately omit the required keyword-only operand."""
            del module, module_fqn, context
            return nn.Identity()

        model = nn.Sequential(RequiredKeyword())
        original = model[0]
        apply_module_replacements(model, compile_module_replacements(model, (ModuleReplacementSpec(("0",), preserve),)))
        self.assertIsNot(model[0], original)
        inputs = torch.ones(2, requires_grad=True)
        output = model[0](inputs, scale=3.)
        torch.testing.assert_close(output, torch.full((2,), 3.))
        output.sum().backward()
        torch.testing.assert_close(inputs.grad, torch.full((2,), 3.))
        selected = model[0]
        with self.assertRaisesRegex(ValueError, "incompatible forward signature"):
            apply_module_replacements(model, compile_module_replacements(model, (ModuleReplacementSpec(("0",), drop),)))
        self.assertIs(model[0], selected)

        class OptionalScale(nn.Module):
            """A source stage whose second positional operand may be omitted."""

            def forward(self, inputs: torch.Tensor, scale: float = 1.) -> torch.Tensor:
                """Scale activations, defaulting to identity.

                Args:
                    inputs: Local activations.
                    scale: Optional scaling factor.
                """
                return inputs * scale

        class RequiredScale(nn.Module):
            """An incompatible target that makes the optional operand mandatory."""

            def forward(self, inputs: torch.Tensor, scale: float) -> torch.Tensor:
                """Scale activations using a mandatory operand.

                Args:
                    inputs: Local activations.
                    scale: Required scaling factor.
                """
                return inputs * scale

        @module_replacement
        def strengthen(*, module: nn.Module, module_fqn: str, context: Mapping[str, Any]) -> nn.Module:
            """Deliberately strengthen the source call requirements."""
            del module, module_fqn, context
            return RequiredScale()

        model = nn.Sequential(OptionalScale())
        with self.assertRaisesRegex(ValueError, "incompatible forward signature"):
            apply_module_replacements(
                model, compile_module_replacements(model, (ModuleReplacementSpec(("0",), strengthen),)),
            )
        self.assertIsInstance(model[0], OptionalScale)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_checkpoint_shapes_are_captured_only_for_weight_transforms(self) -> None:
        """Feature: Checkpoint metadata is captured lazily before the first transformed installation.
        Description: Exercise atomic and incremental plans with identity and renamed weights.
        Expectation: Identity-only plans skip the scan; conversion plans capture the original shapes once.
        """
        for atomic in (False, True):
            for mapped in (False, True):
                with self.subTest(atomic=atomic, mapped=mapped):
                    model = nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(4, 8, bias=False))
                    factories = (_replace_linear, _AllocatedMappedLinear if mapped else _replace_linear)
                    if atomic:
                        requests = tuple(ModuleReplacementRequest((str(index),), factory)
                                         for index, factory in enumerate(factories))
                        plan = compile_module_replacements(model, defaults=requests)
                    else:
                        specs = tuple(ModuleReplacementSpec((str(index),), factory)
                                      for index, factory in enumerate(factories))
                        plan = compile_module_replacements(model, specs)
                    with patch("hyper_parallel.models.replacement._named_tensor_shapes",
                               wraps=_named_tensor_shapes) as capture:
                        apply_module_replacements(model, plan, weights_mapping=[])
                    self.assertEqual(capture.call_count, int(mapped))
                    if mapped:
                        self.assertEqual(model._hp_checkpoint_source_shapes, {"0.weight": (8, 4), "1.weight": (8, 4)})
                    else:
                        self.assertFalse(hasattr(model, "_hp_checkpoint_source_shapes"))
