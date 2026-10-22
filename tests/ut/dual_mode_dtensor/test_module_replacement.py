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
"""CPU-only contracts for generic module replacement plans.

Tests are grouped by feature family (success matching, make_transforms,
engine error paths, when gating, YAML desugaring, YAML error paths);
each family runs its atomic checks sequentially with identifying messages.
"""

import unittest
import weakref
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from torch import Tensor, nn

# The generic replacement contract must not depend on a specific transformers
# version: use the WeightRenaming exposed by hyper_parallel itself (which falls
# back to a local implementation when transformers lacks scoped transforms).
from hyper_parallel.components.checkpoint.weight_conversion import WeightRenaming

from hyper_parallel.models.replacement import (
    MethodReplacementSpec,
    ModuleReplacementSpec,
    apply_method_replacements,
    apply_module_replacements,
    compile_module_replacements,
    method_replacement,
    module_replacement,
)
from hyper_parallel.distributed import forward_replacement, inner_wrapper
from hyper_parallel.distributed._builder.rule_resolver import _resolve_local_compute_fn
from hyper_parallel.codegen.emit.replacement import compile_overrides_for_meta
from hyper_parallel.codegen.manager import _apply_generation_replacements
from hyper_parallel.codegen.plan.freeze import freeze_injections
from hyper_parallel.codegen.spec.project import _canonicalize_override
from hyper_parallel.trainer.config.resolver import ConfigResolutionError, resolve_config
from hyper_parallel.trainer.config import (
    PlanOverride,
    ReplaceAction,
    Target,
    _import_module_type,
    entries_to_module_replacements,
    entries_to_method_replacements,
    entries_to_plan_overrides,
    normalize_distributed_setup_overrides,
)


class _ReplacementLinear(nn.Linear):
    """A Linear shell that retains all source state by identity."""


@module_replacement
class _AllocatedMappedLinear(nn.Module):
    """Allocate replacement storage with a checkpoint weight-name mapping."""

    def __init__(self, *, module: nn.Linear, module_fqn: str, context: Mapping[str, Any]) -> None:
        """Copy source values into new storage without retaining the source."""
        super().__init__()
        del module_fqn, context
        self.packed_weight = nn.Parameter(
            module.weight.detach().clone(), requires_grad=module.weight.requires_grad
        )
        self.train(module.training)

    def forward(self, input: Tensor) -> Tensor:  # pylint: disable=redefined-builtin
        """Use the independently allocated weight."""
        return input @ self.packed_weight.t()

    def make_transforms(self) -> list[WeightRenaming]:
        """Describe the source-to-target weight name conversion."""
        return [WeightRenaming("weight", "packed_weight")]


class _NestedModule(nn.Module):
    """A module whose child parameter identity must survive replacement."""

    def __init__(self):
        super().__init__()
        self.child = nn.Linear(4, 8)

    def forward(self, inputs):
        return self.child(inputs)


class _PositionalOnlyModule(nn.Module):
    """Forward uses a positional-only argument."""

    def forward(self, inputs, /, scale=1):
        return inputs * scale


class _VariadicModule(nn.Module):
    """Forward accepts variadic arguments."""

    def forward(self, inputs, *args, **kwargs):
        del args, kwargs
        return inputs


class _DroppedVariadicModule(nn.Module):
    """Replacement intentionally drops variadic call support."""

    def forward(self, inputs):
        return inputs


@module_replacement
def _replace_linear(*, module, module_fqn, context):
    """Build a _ReplacementLinear shell retaining the source parameters by identity."""
    del module_fqn, context
    replacement = _ReplacementLinear.__new__(_ReplacementLinear)  # pylint: disable=no-value-for-parameter
    # Intentional re-init after __new__: nn.Linear.__init__ would allocate fresh
    # parameters, but this shell must keep the source parameter identity.
    nn.Module.__init__(replacement)  # pylint: disable=unnecessary-dunder-call
    replacement.in_features = module.in_features
    replacement.out_features = module.out_features
    replacement.register_parameter("weight", module.weight)
    replacement.register_parameter("bias", module.bias)
    replacement.training = module.training
    return replacement


@module_replacement
def _replace_nested_with_new_child(*, module, module_fqn, context):
    del module, module_fqn, context
    return _NestedModule()


@module_replacement
def _replace_positional_only(*, module, module_fqn, context):
    del module, module_fqn, context
    return _PositionalOnlyModule()


@module_replacement
def _replace_with_dropped_variadic(*, module, module_fqn, context):
    del module, module_fqn, context
    return _DroppedVariadicModule()


@forward_replacement
def _forward_factory(*, mesh, tp_mesh, cp_mesh, ep_mesh, module, scale=1.0):
    """Build a simple differentiable local forward for config tests."""
    del mesh, tp_mesh, cp_mesh, ep_mesh, module

    def compute(current_module, inputs):
        del current_module
        return inputs * scale

    return compute


@inner_wrapper
def _wrapper_factory(*, target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """Build a pass-through CP-style wrapper for config tests."""
    del mesh, tp_mesh, cp_mesh, ep_mesh
    return target_module.forward


class _RouteGate(nn.Module):
    """Small module with a replaceable method distinct from forward."""

    def route(self, inputs):
        """Return the original route result."""
        return inputs + 1

    def forward(self, inputs):
        """Call the replaceable method."""
        return self.route(inputs)


@method_replacement
def _route_factory(*, module, method_fqn, original_method, context, scale=1.0):
    """Build a bound-style method replacement from the original method."""
    del module, method_fqn, context

    def route(inputs):
        return original_method(inputs) * scale

    return route


def _spec(*patterns):
    return ModuleReplacementSpec(
        match=patterns,
        factory=_replace_linear,
        module_type=nn.Linear,
        exact_type=True,
    )


class TestModuleReplacementPlan(unittest.TestCase):
    """Replacement matching and installation stay backend independent."""

    def test_success_matching_and_identity_paths(self):
        """Success family: exact/alias matching, parameter identity, context."""
        # case: replaces_exact_linear_and_preserves_parameter_identity
        linear = nn.Linear(4, 8, bias=False)
        model = nn.Sequential(linear)

        plan = compile_module_replacements(model, [_spec("0")])
        apply_module_replacements(model, plan)

        self.assertIsInstance(
            model[0],
            _ReplacementLinear,
            "case: replaces_exact_linear_and_preserves_parameter_identity",
        )
        self.assertIs(
            model[0].weight,
            linear.weight,
            "case: replaces_exact_linear_and_preserves_parameter_identity",
        )

        # case: aliases_share_one_replacement
        linear = nn.Linear(4, 8, bias=False)
        model = nn.Module()
        model.left = linear
        model.right = linear

        plan = compile_module_replacements(model, [_spec("left")])
        apply_module_replacements(model, plan)

        self.assertIs(model.left, model.right, "case: aliases_share_one_replacement")
        self.assertIsInstance(
            model.left, _ReplacementLinear, "case: aliases_share_one_replacement"
        )

        # case: alias_match_replaces_all_aliases_with_stable_canonical_fqn
        received_fqns = []

        @module_replacement
        def record_fqn(*, module, module_fqn, context):
            received_fqns.append(module_fqn)
            return _replace_linear(
                module=module,
                module_fqn=module_fqn,
                context=context,
            )

        linear = nn.Linear(4, 8, bias=False)
        model = nn.Module()
        model.z_alias = linear
        model.a_alias = linear
        spec = ModuleReplacementSpec(
            match=("z_alias",),
            factory=record_fqn,
            module_type=nn.Linear,
            exact_type=True,
        )

        plan = compile_module_replacements(model, [spec])
        apply_module_replacements(model, plan)

        self.assertEqual(
            received_fqns,
            ["a_alias"],
            "case: alias_match_replaces_all_aliases_with_stable_canonical_fqn",
        )
        self.assertIs(
            model.a_alias,
            model.z_alias,
            "case: alias_match_replaces_all_aliases_with_stable_canonical_fqn",
        )
        self.assertIsInstance(
            model.a_alias,
            _ReplacementLinear,
            "case: alias_match_replaces_all_aliases_with_stable_canonical_fqn",
        )

        # case: apply_passes_generic_context_to_factory
        received_context = []

        @module_replacement
        def record_context(*, module, module_fqn, context):
            received_context.append(context)
            return _replace_linear(module=module, module_fqn=module_fqn, context=context)

        model = nn.Sequential(nn.Linear(4, 8))
        spec = ModuleReplacementSpec(
            match=("0",), factory=record_context, module_type=nn.Linear, exact_type=True
        )
        plan = compile_module_replacements(model, [spec])
        apply_module_replacements(model, plan, context={"policy": "low_precision"})

        self.assertEqual(
            received_context,
            [{"policy": "low_precision"}],
            "case: apply_passes_generic_context_to_factory",
        )
        self.assertIsInstance(
            model[0],
            _ReplacementLinear,
            "case: apply_passes_generic_context_to_factory",
        )

        # case: accepts_positional_only_forward
        model = nn.Sequential(_PositionalOnlyModule())
        spec = ModuleReplacementSpec(
            match=("0",),
            factory=_replace_positional_only,
            module_type=_PositionalOnlyModule,
            exact_type=True,
        )

        plan = compile_module_replacements(model, [spec])
        apply_module_replacements(model, plan)
        self.assertIsInstance(
            model[0], _PositionalOnlyModule, "case: accepts_positional_only_forward"
        )

    def test_make_transforms_weights_mapping(self):
        """Replacement make_transforms takes no model config and feeds mapping."""
        calls = []

        @module_replacement
        class MappedLinear(nn.Module):
            """Replacement shell that packs the weight and declares a weight renaming."""

            def __init__(self, *, module, module_fqn, context):
                super().__init__()
                del module_fqn, context
                self.in_features = module.in_features
                self.out_features = module.out_features
                self.register_parameter("packed_weight", module.weight)

            # The parameter name must match nn.Linear.forward's "input" keyword:
            # _validate_forward_compatibility binds the source signature's keyword
            # names against the replacement signature.
            def forward(self, input):  # pylint: disable=redefined-builtin
                return nn.functional.linear(input, self.packed_weight)

            def make_transforms(self):
                calls.append("make_transforms")
                return [WeightRenaming("weight", "packed_weight")]

        model = nn.Sequential(nn.Linear(4, 8, bias=False))
        spec = ModuleReplacementSpec(
            match=("0",),
            factory=MappedLinear,
            module_type=nn.Linear,
            exact_type=True,
        )
        weights_mapping = []

        plan = compile_module_replacements(model, [spec])
        model, weights_mapping = apply_module_replacements(
            model,
            plan,
            weights_mapping=weights_mapping,
        )

        self.assertEqual(
            calls,
            ["make_transforms"],
            "case: replacement_make_transforms_takes_no_model_config",
        )
        self.assertEqual(
            len(weights_mapping),
            1,
            "case: replacement_make_transforms_takes_no_model_config",
        )
        self.assertEqual(
            weights_mapping[0].source_patterns,
            ["weight"],
            "case: replacement_make_transforms_takes_no_model_config",
        )
        self.assertEqual(
            weights_mapping[0].target_patterns,
            ["packed_weight"],
            "case: replacement_make_transforms_takes_no_model_config",
        )

    def test_engine_error_paths(self):
        """Error family: every rejection keeps its identifying match pattern."""
        # case: each_pattern_must_match
        with self.assertRaisesRegex(
            ValueError, "typo", msg="case: each_pattern_must_match"
        ):
            compile_module_replacements(
                nn.Sequential(nn.Linear(4, 8)), [_spec("0", "typo")]
            )

        # case: rejects_hooks_that_cannot_be_migrated
        model = nn.Sequential(nn.Linear(4, 8))
        model[0].register_full_backward_pre_hook(lambda *_: None)
        plan = compile_module_replacements(model, [_spec("0")])

        with self.assertRaisesRegex(
            ValueError,
            "cannot migrate existing module hooks",
            msg="case: rejects_hooks_that_cannot_be_migrated",
        ):
            apply_module_replacements(model, plan)

        # case: rejects_nested_parameter_identity_changes
        model = nn.Sequential(_NestedModule())
        spec = ModuleReplacementSpec(
            match=("0",),
            factory=_replace_nested_with_new_child,
            module_type=_NestedModule,
            exact_type=True,
        )
        plan = compile_module_replacements(model, [spec])

        with self.assertRaisesRegex(
            ValueError,
            "parameter 'child.weight' identity",
            msg="case: rejects_nested_parameter_identity_changes",
        ):
            apply_module_replacements(model, plan)

        # case: rejects_replacement_that_drops_variadic_forward_support
        model = nn.Sequential(_VariadicModule())
        spec = ModuleReplacementSpec(
            match=("0",),
            factory=_replace_with_dropped_variadic,
            module_type=_VariadicModule,
            exact_type=True,
        )
        plan = compile_module_replacements(model, [spec])

        with self.assertRaisesRegex(
            ValueError,
            "incompatible forward signature",
            msg="case: rejects_replacement_that_drops_variadic_forward_support",
        ):
            apply_module_replacements(model, plan)

        # case: factory_failure_keeps_completed_replacements
        @module_replacement
        def fail_second(*, module: nn.Linear, module_fqn: str, context: Mapping[str, Any]) -> nn.Module:
            """Fail after the first target has already been installed."""
            if module_fqn == "1":
                raise ValueError("factory failed")
            return _replace_linear(module=module, module_fqn=module_fqn, context=context)

        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(4, 8))
        originals = tuple(model.children())
        spec = ModuleReplacementSpec(
            match=("*",), factory=fail_second, module_type=nn.Linear, exact_type=True
        )
        plan = compile_module_replacements(model, [spec])

        with self.assertRaisesRegex(
            ValueError,
            "factory failed",
            msg="case: factory_failure_keeps_completed_replacements",
        ):
            apply_module_replacements(model, plan)
        self.assertIsInstance(
            model[0],
            _ReplacementLinear,
            f"Expected completed replacement, got {type(model[0])}",
        )
        self.assertIs(
            model[1], originals[1],
            f"Expected untouched failing target {originals[1]}, got {model[1]}",
        )

    def test_releases_sources_before_building_next_replacement(self) -> None:
        """Keeping the plan alive must not retain completed source weights."""

        for device in ("cpu", "meta"):
            with self.subTest(device=device):
                model = nn.Module()
                model.left = nn.Linear(4, 8, bias=False, device=device)
                model.right = model.left
                model.tail = nn.Linear(4, 8, bias=False, device=device)

                source_refs = [weakref.ref(model.left), weakref.ref(model.tail)]
                weight_refs = [weakref.ref(model.left.weight), weakref.ref(model.tail.weight)]

                calls = []

                @module_replacement
                def replace_checked(
                    *, module: nn.Linear, module_fqn: str, context: Mapping[str, Any],
                    case_state=(calls, source_refs, weight_refs),
                ) -> nn.Module:
                    """Require previous source storage to be gone before allocating."""
                    calls, source_refs, weight_refs = case_state
                    for index in range(len(calls)):
                        self.assertIsNone(
                            source_refs[index](),
                            f"Expected released source {index}, got {source_refs[index]()}"
                        )
                        self.assertIsNone(
                            weight_refs[index](),
                            f"Expected released weight {index}, got {weight_refs[index]()}"
                        )
                    calls.append(module_fqn)
                    return _AllocatedMappedLinear(module=module, module_fqn=module_fqn, context=context)

                spec = ModuleReplacementSpec(
                    match=("*",), factory=replace_checked, module_type=nn.Linear, exact_type=True
                )

                plan = compile_module_replacements(model, [spec])
                existing_mapping = WeightRenaming("old", "new")
                weights_mapping = [existing_mapping]
                result, result_mapping = apply_module_replacements(model, plan, weights_mapping=weights_mapping)

                self.assertIs(result, model, f"Expected original model object {model}, got {result}")
                self.assertIs(
                    result_mapping, weights_mapping,
                    f"Expected mapping {weights_mapping}, got {result_mapping}",
                )
                self.assertEqual(calls, ["left", "tail"], f"Expected ['left', 'tail'], got {calls}")
                self.assertIs(model.left, model.right, f"Expected shared replacement {model.left}, got {model.right}")

                for source_ref, weight_ref in zip(source_refs, weight_refs):
                    self.assertIsNone(source_ref(), f"Expected released source, got {source_ref()}")
                    self.assertIsNone(weight_ref(), f"Expected released weight, got {weight_ref()}")

                shapes = model._hp_checkpoint_source_shapes  # pylint: disable=protected-access
                expected_shapes = {"left.weight": (8, 4), "right.weight": (8, 4), "tail.weight": (8, 4)}
                self.assertEqual(shapes, expected_shapes, f"Expected {expected_shapes}, got {shapes}")

                scopes = [transform.scope_prefix for transform in weights_mapping[:2]]
                self.assertEqual(scopes, ["left", "tail"], f"Expected ['left', 'tail'], got {scopes}")
                self.assertIs(
                    weights_mapping[-1], existing_mapping,
                    f"Expected existing mapping {existing_mapping}, got {weights_mapping[-1]}",
                )

    def test_changed_plan_target_is_rejected_before_installation(self) -> None:
        """Reject a stale target without undoing earlier replacements."""

        for keep_source_alive in (False, True):
            with self.subTest(keep_source_alive=keep_source_alive):
                model = nn.Sequential(nn.Linear(4, 8), nn.Linear(4, 8))
                plan = compile_module_replacements(model, [_spec("*")])
                previous_source = model[1] if keep_source_alive else None
                model[1] = nn.Linear(4, 8)
                current_source = model[1]

                with self.assertRaisesRegex(ValueError, "replacement target '1'"):
                    apply_module_replacements(model, plan)

                self.assertIsInstance(
                    model[0], _ReplacementLinear,
                    f"Expected completed replacement, got {type(model[0])}",
                )
                self.assertIs(
                    model[1], current_source,
                    f"Expected unchanged current target {current_source}, got {model[1]}",
                )
                del previous_source

    def test_missing_weight_mapping_does_not_install_current_target(self) -> None:
        """A transforming target is validated before it replaces its source."""

        model = nn.Sequential(nn.Linear(4, 8, bias=False))
        source = model[0]
        spec = ModuleReplacementSpec(
            match=("0",), factory=_AllocatedMappedLinear, module_type=nn.Linear, exact_type=True
        )
        plan = compile_module_replacements(model, [spec])

        with self.assertRaisesRegex(ValueError, "weights_mapping is required"):
            apply_module_replacements(model, plan)

        self.assertIs(model[0], source, f"Expected original target {source}, got {model[0]}")


# ==========================================================================
# replace_module YAML config entry (desugared into plan_overrides rules) cases
# ==========================================================================


def _model_target():
    return None


def _optimizer_target():
    return None


def _replacement(*, module, module_fqn, context):
    del module_fqn, context
    return module


@module_replacement
def _declared_replacement(*, module, module_fqn, context):
    return _replacement(module=module, module_fqn=module_fqn, context=context)


def _root(plan_overrides):
    return {
        "model": {"_target_": f"{__name__}._model_target"},
        "optimizer": {"_target_": f"{__name__}._optimizer_target"},
        "plan_overrides": plan_overrides,
    }


class TestModuleReplacementYaml(unittest.TestCase):
    """Replacement actions share the existing plan_overrides YAML transport."""

    def test_yaml_desugars_to_generic_rule(self):
        """YAML desugar family: replace_module entry becomes a generic rule."""
        config = resolve_config(_root([
            {
                "match": ["encoder.*", "decoder.*"],
                "module_type": "torch.nn.Linear",
                "exact_type": True,
                "replace_module": {
                    "_target_": f"{__name__}._declared_replacement",
                },
            },
        ]))

        rules = entries_to_module_replacements(config.plan_overrides)

        self.assertEqual(
            rules[0].match,
            ("encoder.*", "decoder.*"),
            "case: yaml_replacement_desugars_to_generic_rule",
        )
        self.assertEqual(
            rules[0].module_type.__name__,
            "Linear",
            "case: yaml_replacement_desugars_to_generic_rule",
        )
        self.assertTrue(
            rules[0].exact_type, "case: yaml_replacement_desugars_to_generic_rule"
        )

    def test_new_module_action_resolves_and_preserves_parameter_identity(self):
        """A new module action compiles through the existing replacement engine."""
        config = resolve_config(_root([{
            "match": ["0", "1"],
            "replace": {
                "kind": "module",
                "source_type": "torch.nn.Linear",
                "implementation": f"{__name__}._replace_linear",
            },
        }]))
        rule = config.plan_overrides[0]
        self.assertEqual(rule.to_dict()["replace"]["implementation"], f"{__name__}._replace_linear")
        self.assertNotIn("replace_module", rule.to_dict())
        specs = entries_to_module_replacements(config.plan_overrides)
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        original_weights = [module.weight for module in model]
        apply_module_replacements(model, compile_module_replacements(model, specs))
        self.assertTrue(all(isinstance(module, _ReplacementLinear) for module in model))
        self.assertTrue(all(module.weight is weight for module, weight in zip(model, original_weights)))
        records = compile_overrides_for_meta(
            nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)), specs,
            factory_paths=[f"{__name__}._replace_linear"],
        )
        self.assertEqual({record["fqn"] for record in records}, {"0", "1"})
        self.assertEqual(records[0]["factory"], f"{__name__}._replace_linear")
        self.assertIn("replace_module", _canonicalize_override(rule))
        legacy = PlanOverride(
            match=["0", "1"], module_type="torch.nn.Linear",
            replace_module=Target(_replace_linear, target_path=f"{__name__}._replace_linear"),
        )
        self.assertEqual(_canonicalize_override(rule), _canonicalize_override(legacy))

        wrong_type = PlanOverride(match="0", replace=ReplaceAction(
            kind="module", source_type="torch.nn.Conv2d", implementation=f"{__name__}._replace_linear"))
        mismatched = entries_to_module_replacements([wrong_type])
        with self.assertRaisesRegex(TypeError, "replace.kind=module"):
            compile_overrides_for_meta(
                nn.Sequential(nn.Linear(4, 4)), mismatched,
                factory_paths=[f"{__name__}._replace_linear"],
            )

    def test_new_module_action_infers_source_type_from_match(self):
        """A module rule without source_type uses each matched source instance."""
        config = resolve_config(_root([{
            "match": ["0", "1"],
            "replace": {
                "kind": "module",
                "implementation": f"{__name__}._replace_linear",
            },
        }]))
        rule = config.plan_overrides[0]
        self.assertNotIn("source_type", rule.to_dict()["replace"])
        specs = entries_to_module_replacements(config.plan_overrides)
        self.assertIsNone(specs[0].module_type)

        meta_model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        records = compile_overrides_for_meta(
            meta_model, specs, factory_paths=[f"{__name__}._replace_linear"]
        )
        self.assertEqual({record["module_type"] for record in records}, {
            f"{nn.Linear.__module__}.{nn.Linear.__qualname__}"
        })

        generated_model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        dims = SimpleNamespace(tp_size=1, cp_size=1, ep_size=1, pp_size=1, sequence_parallel=False)
        generated_records = _apply_generation_replacements(
            SimpleNamespace(), SimpleNamespace(plan_overrides=config.plan_overrides),
            generated_model, SimpleNamespace(parallel_dims=dims),
        )
        self.assertEqual({record["module_type"] for record in generated_records}, {
            f"{nn.Linear.__module__}.{nn.Linear.__qualname__}"
        })
        self.assertTrue(all(isinstance(module, _ReplacementLinear) for module in generated_model))

        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        original_weights = [module.weight for module in model]
        apply_module_replacements(model, compile_module_replacements(model, specs))
        self.assertTrue(all(isinstance(module, _ReplacementLinear) for module in model))
        self.assertTrue(all(module.weight is weight for module, weight in zip(model, original_weights)))

    def test_new_forward_action_lowers_both_factory_protocols(self):
        """Forward actions use factory metadata, preserving the old execution hosts."""
        config = resolve_config(_root([{
            "match": "*.mlp.forward", "when": "ep",
            "replace": {"kind": "method", "implementation": {
                "_target_": f"{__name__}._forward_factory", "scale": 2.0}},
        }, {
            "match": "*.self_attn.forward", "when": "cp",
            "replace": {"kind": "method", "implementation": f"{__name__}._wrapper_factory"},
        }]))
        ep_rule, cp_rule = config.plan_overrides
        self.assertFalse(ep_rule.region_dispatch)
        self.assertIs(ep_rule.local_compute_fn.callable, _forward_factory)
        self.assertEqual(ep_rule.to_dict()["replace"]["implementation"]["scale"], 2.0)
        self.assertNotIn("region_dispatch", ep_rule.to_dict())
        self.assertEqual(cp_rule.inner_target, "self")
        self.assertIs(cp_rule.inner_wrapper.callable, _wrapper_factory)
        legacy_cp = PlanOverride(
            match="*.self_attn", when="cp", region_dispatch=False,
            inner_target="self", inner_wrapper=Target(
                _wrapper_factory, target_path=f"{__name__}._wrapper_factory"),
        )
        canonical = _canonicalize_override(cp_rule)
        self.assertEqual(canonical.pop("match"), "*.self_attn.forward")
        legacy_canonical = _canonicalize_override(legacy_cp)
        legacy_canonical.pop("match")
        self.assertEqual(canonical, legacy_canonical)
        self.assertEqual(entries_to_plan_overrides(config.plan_overrides, cp_size=1, ep_size=1), {})
        active = entries_to_plan_overrides(config.plan_overrides, cp_size=2, ep_size=2)
        self.assertIs(active["*.mlp"].local_compute_fn, ep_rule.local_compute_fn)
        self.assertIs(active["*.self_attn"].inner_wrapper, cp_rule.inner_wrapper)
        frozen = freeze_injections(SimpleNamespace(modules=active))
        self.assertEqual({entry["match"] for entry in frozen}, {"*.mlp", "*.self_attn"})
        self.assertIn("local_compute_fn", next(entry for entry in frozen if entry["match"] == "*.mlp"))
        self.assertIn("inner_wrapper", next(entry for entry in frozen if entry["match"] == "*.self_attn"))

        model = _NestedModule()
        compute = _resolve_local_compute_fn(model, active["*.mlp"], None, (), None)
        inputs = torch.ones(2, requires_grad=True)
        compute(inputs).sum().backward()
        self.assertTrue(torch.equal(inputs.grad, torch.full_like(inputs, 2.0)))

        strict = PlanOverride(match="*.mlp.forward", region_dispatch=True, replace=ReplaceAction(
            kind="method", implementation=f"{__name__}._forward_factory"))
        self.assertTrue(entries_to_plan_overrides([strict])["*.mlp"].region_dispatch)
        self.assertTrue(strict.to_dict()["region_dispatch"])

    def test_method_path_selects_method_and_preserves_gradient(self):
        """The suffix identifies a method; the owner path keeps glob semantics."""
        config = resolve_config(_root([{
            "match": "*.gate.route", "replace": {"kind": "method", "implementation": {
                "_target_": f"{__name__}._route_factory", "scale": 2.0,
            }},
        }]))
        rule = config.plan_overrides[0]
        self.assertEqual(rule.to_dict()["match"], "*.gate.route")
        self.assertEqual(rule.to_dict()["replace"]["implementation"]["scale"], 2.0)
        self.assertEqual(entries_to_plan_overrides(config.plan_overrides), {})
        specs = entries_to_method_replacements(config.plan_overrides)
        self.assertEqual(len(specs), 1)
        model = nn.Module()
        model.layers = nn.ModuleList([_RouteGate(), _RouteGate()])
        for layer in model.layers:
            layer.gate = _RouteGate()
        original_gate = model.layers[0].gate
        apply_method_replacements(model, specs)
        inputs = torch.ones(2, requires_grad=True)
        result = model.layers[0].gate(inputs) + model.layers[1].gate(inputs)
        result.sum().backward()
        self.assertIs(model.layers[0].gate, original_gate)
        self.assertTrue(torch.equal(result, torch.full_like(result, 8.0)))
        self.assertTrue(torch.equal(inputs.grad, torch.full_like(inputs, 4.0)))
        self.assertIn("replace", _canonicalize_override(rule))

        gated = PlanOverride(match="*.gate.route", when="cp", replace=ReplaceAction(
            kind="method", implementation=f"{__name__}._route_factory"))
        self.assertEqual(entries_to_method_replacements([gated], cp_size=1), ())
        self.assertEqual(len(entries_to_method_replacements([gated], cp_size=2)), 1)

    def test_method_forward_path_reuses_existing_boundary_executor(self):
        """A .forward suffix lowers to the same owner boundary as legacy forward."""
        rule = PlanOverride(match="*.mlp.forward", when="ep", replace=ReplaceAction(
            kind="method", implementation=f"{__name__}._forward_factory"))
        self.assertEqual(rule.to_dict()["match"], "*.mlp.forward")
        self.assertEqual(entries_to_plan_overrides([rule], ep_size=2)["*.mlp"].local_compute_fn,
                         rule.local_compute_fn)
        self.assertEqual(entries_to_method_replacements([rule], ep_size=2), ())
        frozen = freeze_injections(SimpleNamespace(modules=entries_to_plan_overrides([rule], ep_size=2)))
        self.assertEqual(frozen[0]["match"], "*.mlp")

    def test_method_replacement_is_wired_in_native_and_codegen_setup(self):
        """Codegen validates targets; native construction installs the patch."""
        rule = PlanOverride(match="gate.route", replace=ReplaceAction(
            kind="method", implementation=f"{__name__}._route_factory"))
        setup = SimpleNamespace(mesh_context=SimpleNamespace(
            cp_size=1, ep_size=1, sequence_parallel=False), low_precision_config=None)
        normalize_distributed_setup_overrides(setup, SimpleNamespace(plan_overrides=[rule]))
        self.assertEqual(len(setup.method_replacements), 1)
        self.assertEqual(setup.plan_overrides, {})

        model = nn.Module()
        model.gate = _RouteGate()
        dims = SimpleNamespace(tp_size=1, cp_size=1, ep_size=1, pp_size=1, sequence_parallel=False)
        records = _apply_generation_replacements(
            SimpleNamespace(), SimpleNamespace(plan_overrides=[rule]), model,
            SimpleNamespace(parallel_dims=dims),
        )
        self.assertEqual(records, [])
        self.assertNotIn("route", vars(model.gate))
        apply_method_replacements(model, setup.method_replacements)
        self.assertTrue(torch.equal(model.gate(torch.ones(1)), torch.full((1,), 2.0)))

    def test_method_replacement_rejects_missing_and_wrong_targets(self):
        """Fail before patching on invalid suffixes, absent methods, and modules."""
        with self.assertRaisesRegex(ValueError, "literal method name"):
            PlanOverride(match="*.gate.*", replace=ReplaceAction(
                kind="method", implementation=f"{__name__}._route_factory"))
        with self.assertRaisesRegex(TypeError, "@method_replacement"):
            PlanOverride(match="*.gate.route", replace=ReplaceAction(
                kind="method", implementation=f"{__name__}._forward_factory"))
        with self.assertRaisesRegex(ValueError, "boundary contract"):
            PlanOverride(match="*.gate.route", region_dispatch=False, replace=ReplaceAction(
                kind="method", implementation=f"{__name__}._route_factory"))
        specs = (MethodReplacementSpec("*.gate.route", _route_factory),)
        with self.assertRaisesRegex(ValueError, "without a callable"):
            apply_method_replacements(nn.Sequential(nn.ModuleDict({"gate": nn.Linear(2, 2)})), specs)
        with self.assertRaisesRegex(ValueError, "matched no module"):
            apply_method_replacements(nn.Sequential(_RouteGate()), specs)

    def test_method_replacement_rejects_child_module_and_conflict(self):
        """A callable child module cannot be mistaken for an ordinary method."""
        model = nn.Module()
        model.gate = _RouteGate()
        model.gate.child = nn.Linear(2, 2)
        with self.assertRaisesRegex(TypeError, "registered module"):
            apply_method_replacements(model, (MethodReplacementSpec("gate.child", _route_factory),))
        with self.assertRaisesRegex(ValueError, "conflict"):
            apply_method_replacements(model, (
                MethodReplacementSpec("gate.route", _route_factory),
                MethodReplacementSpec("*.route", _route_factory),
            ))

    def test_method_replacement_patches_shared_module_once(self):
        """Alias paths share one module instance and one replacement factory call."""
        calls = []

        @method_replacement
        def factory(*, module, method_fqn, original_method, context):
            del module, context
            calls.append(method_fqn)
            return original_method

        shared = _RouteGate()
        model = nn.Module()
        model.left = shared
        model.right = shared
        apply_method_replacements(model, (MethodReplacementSpec("*.route", factory),))
        self.assertEqual(calls, ["left.route"])
        self.assertIs(model.left, model.right)

    def test_new_action_rejects_invalid_combinations(self):
        """Kind-specific fields and old/new conflicts fail during config loading."""
        invalid = [
            ({"kind": "module", "source_type": "", "implementation": f"{__name__}._replace_linear"},
             "non-empty"),
            ({"kind": "module", "exact_type": True, "implementation": f"{__name__}._replace_linear"},
             "requires replace.source_type"),
            ({"kind": "method", "source_type": "torch.nn.Linear",
              "implementation": f"{__name__}._forward_factory"}, "only valid"),
            ({"kind": "method", "implementation": f"{__name__}._replacement"}, "@forward_replacement"),
        ]
        for action, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex((ValueError, TypeError), message):
                resolve_config(_root([{"match": "x.forward", "replace": action}]))
        with self.assertRaisesRegex(ValueError, "replace.kind must be 'module' or 'method'"):
            ReplaceAction(kind="forward", implementation=f"{__name__}._forward_factory")
        with self.assertRaisesRegex(ConfigResolutionError, "replace.kind.*module.*method"):
            resolve_config(_root([{"match": "x.forward", "replace": {
                "kind": "forward", "implementation": f"{__name__}._forward_factory"}}]))
        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            PlanOverride(match="x", module_type="torch.nn.Linear", replace=ReplaceAction(
                kind="module", source_type="torch.nn.Linear", implementation=f"{__name__}._replace_linear"))
        with self.assertRaisesRegex(ValueError, "region_dispatch"):
            PlanOverride(match="x", region_dispatch=False, replace=ReplaceAction(
                kind="module", source_type="torch.nn.Linear", implementation=f"{__name__}._replace_linear"))
        with self.assertRaisesRegex(ConfigResolutionError, "replace.implementation._target_.*could not be imported"):
            resolve_config(_root([{
                "match": "x.forward",
                "replace": {"kind": "method", "implementation": {"_target_": "missing_hp_package.factory"}},
            }]))
        with self.assertRaisesRegex(ConfigResolutionError, "unknown configuration fields.*target"):
            resolve_config(_root([{
                "match": "x.forward",
                "replace": {"kind": "method", "target": f"{__name__}._forward_factory"},
            }]))

    def test_when_gating_rejected(self):
        """When-gating family: replacements reject any 'when' clause."""
        # case: replacement_rejects_valid_when
        entry = PlanOverride(
            match="encoder.*",
            when="ep",
            module_type="torch.nn.Linear",
            replace_module=Target(
                _declared_replacement,
                target_path=f"{__name__}._declared_replacement",
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "does not support 'when'",
            msg="case: replacement_rejects_valid_when",
        ):
            entries_to_module_replacements([entry])

        # case: replacement_rejects_invalid_when
        entry = PlanOverride(
            match="encoder.*",
            when="xp",
            module_type="torch.nn.Linear",
            replace_module=Target(
                _declared_replacement,
                target_path=f"{__name__}._declared_replacement",
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "does not support 'when'",
            msg="case: replacement_rejects_invalid_when",
        ):
            entries_to_module_replacements([entry])

    def test_yaml_error_paths(self):
        """YAML error family: undeclared factory, bare list match, import error."""
        # case: yaml_replacement_rejects_undeclared_factory_contract
        config = resolve_config(_root([
            {
                "match": "encoder.*",
                "module_type": "torch.nn.Linear",
                "replace_module": {"_target_": f"{__name__}._replacement"},
            },
        ]))

        with self.assertRaisesRegex(
            TypeError,
            "@module_replacement",
            msg="case: yaml_replacement_rejects_undeclared_factory_contract",
        ):
            entries_to_module_replacements(config.plan_overrides)

        # case: list_match_without_replacement_is_a_configuration_error
        with self.assertRaisesRegex(
            ValueError,
            "match lists",
            msg="case: list_match_without_replacement_is_a_configuration_error",
        ):
            entries_to_plan_overrides([PlanOverride(match=["a", "b"])])

        # case: module_type_import_error_has_plan_override_context
        with patch(
            "hyper_parallel.trainer.config.parallelism.importlib.import_module",
            side_effect=ImportError("optional dependency is unavailable"),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "plan_overrides.module_type",
                msg="case: module_type_import_error_has_plan_override_context",
            ):
                _import_module_type("example.module.Type")
