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
"""CPU-only contracts for generic module replacement plans."""

import unittest

from torch import nn
from transformers.core_model_loading import WeightRenaming

from hyper_models.components.model_transform import (
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
    module_replacement,
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
    del module_fqn, context
    replacement = _ReplacementLinear.__new__(_ReplacementLinear)
    nn.Module.__init__(replacement)
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

    def test_replaces_exact_linear_and_preserves_parameter_identity(self):
        linear = nn.Linear(4, 8, bias=False)
        model = nn.Sequential(linear)

        plan = compile_module_replacements(model, [_spec("0")])
        apply_module_replacements(model, plan)

        self.assertIsInstance(model[0], _ReplacementLinear)
        self.assertIs(model[0].weight, linear.weight)

    def test_each_pattern_must_match(self):
        with self.assertRaisesRegex(ValueError, "typo"):
            compile_module_replacements(
                nn.Sequential(nn.Linear(4, 8)), [_spec("0", "typo")]
            )

    def test_aliases_share_one_replacement(self):
        linear = nn.Linear(4, 8, bias=False)
        model = nn.Module()
        model.left = linear
        model.right = linear

        plan = compile_module_replacements(model, [_spec("left")])
        apply_module_replacements(model, plan)

        self.assertIs(model.left, model.right)
        self.assertIsInstance(model.left, _ReplacementLinear)

    def test_alias_match_replaces_all_aliases_with_stable_canonical_fqn(self):
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

        self.assertEqual(received_fqns, ["a_alias"])
        self.assertIs(model.a_alias, model.z_alias)
        self.assertIsInstance(model.a_alias, _ReplacementLinear)

    def test_apply_passes_generic_context_to_factory(self):
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

        self.assertEqual(received_context, [{"policy": "low_precision"}])
        self.assertIsInstance(model[0], _ReplacementLinear)

    def test_replacement_make_transforms_takes_no_model_config(self):
        calls = []

        @module_replacement
        class MappedLinear(nn.Module):
            def __init__(self, *, module, module_fqn, context):
                super().__init__()
                del module_fqn, context
                self.in_features = module.in_features
                self.out_features = module.out_features
                self.register_parameter("packed_weight", module.weight)

            def forward(self, input):
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

        self.assertEqual(calls, ["make_transforms"])
        self.assertEqual(len(weights_mapping), 1)
        self.assertEqual(weights_mapping[0].source_patterns, ["weight"])
        self.assertEqual(weights_mapping[0].target_patterns, ["packed_weight"])

    def test_rejects_hooks_that_cannot_be_migrated(self):
        model = nn.Sequential(nn.Linear(4, 8))
        model[0].register_full_backward_pre_hook(lambda *_: None)
        plan = compile_module_replacements(model, [_spec("0")])

        with self.assertRaisesRegex(ValueError, "cannot migrate existing module hooks"):
            apply_module_replacements(model, plan)

    def test_rejects_nested_parameter_identity_changes(self):
        model = nn.Sequential(_NestedModule())
        spec = ModuleReplacementSpec(
            match=("0",),
            factory=_replace_nested_with_new_child,
            module_type=_NestedModule,
            exact_type=True,
        )
        plan = compile_module_replacements(model, [spec])

        with self.assertRaisesRegex(ValueError, "parameter 'child.weight' identity"):
            apply_module_replacements(model, plan)

    def test_accepts_positional_only_forward(self):
        model = nn.Sequential(_PositionalOnlyModule())
        spec = ModuleReplacementSpec(
            match=("0",),
            factory=_replace_positional_only,
            module_type=_PositionalOnlyModule,
            exact_type=True,
        )

        plan = compile_module_replacements(model, [spec])
        apply_module_replacements(model, plan)
        self.assertIsInstance(model[0], _PositionalOnlyModule)

    def test_rejects_replacement_that_drops_variadic_forward_support(self):
        model = nn.Sequential(_VariadicModule())
        spec = ModuleReplacementSpec(
            match=("0",),
            factory=_replace_with_dropped_variadic,
            module_type=_VariadicModule,
            exact_type=True,
        )
        plan = compile_module_replacements(model, [spec])

        with self.assertRaisesRegex(ValueError, "incompatible forward signature"):
            apply_module_replacements(model, plan)

    def test_factory_failure_keeps_sources_installed(self):
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

        with self.assertRaisesRegex(ValueError, "factory failed"):
            apply_module_replacements(model, plan)
        self.assertIs(model[0], originals[0])
        self.assertIs(model[1], originals[1])
