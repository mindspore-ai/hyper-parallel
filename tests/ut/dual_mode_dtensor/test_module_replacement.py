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
from unittest.mock import patch

from torch import nn

# The generic replacement contract must not depend on a specific transformers
# version: use the WeightRenaming exposed by hyper_parallel itself (which falls
# back to a local implementation when transformers lacks scoped transforms).
from hyper_parallel.components.checkpoint.weight_conversion import WeightRenaming

from hyper_parallel.models.replacement import (
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
    module_replacement,
)
from hyper_parallel.trainer.config.resolver import resolve_root
from hyper_parallel.trainer.config import (
    PlanOverride,
    Target,
    _import_module_type,
    entries_to_module_replacements,
    entries_to_plan_overrides,
)


class _ReplacementLinear(nn.Linear):
    """A Linear shell that retains all source state by identity."""


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

        # case: factory_failure_keeps_sources_installed
        @module_replacement
        def fail_second(*, module, module_fqn, context):
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
            msg="case: factory_failure_keeps_sources_installed",
        ):
            apply_module_replacements(model, plan)
        self.assertIs(
            model[0], originals[0], "case: factory_failure_keeps_sources_installed"
        )
        self.assertIs(
            model[1], originals[1], "case: factory_failure_keeps_sources_installed"
        )


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
        config = resolve_root(_root([
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
        config = resolve_root(_root([
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
