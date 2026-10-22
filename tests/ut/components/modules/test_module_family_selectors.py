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
"""CPU-only coverage for the construction-time Family whitelist."""

import unittest
from dataclasses import replace
from importlib import import_module
from unittest.mock import patch

from hyper_parallel.components.modules import family as registry
from hyper_parallel.components.modules.family import (
    ModuleFamily,
    SelectorSpec,
    VariantRule,
    get_module_family,
    register_module_family,
    resolve_module_family_target,
)
from tests.common.mark_utils import arg_mark


class Baseline:
    """Test constructor with an ordinary, non-selector parameter."""

    def __init__(self, width: int = 8, **kwargs: object) -> None:
        """Record ordinary arguments to detect selector leakage."""
        self.width = width
        self.extra = kwargs


class Alternate(Baseline):
    """Test-only implementation for the first allowed combination."""


class Fused(Baseline):
    """Test-only implementation for the second allowed combination."""


def declaration() -> ModuleFamily:
    """Return a test-local Family with two boolean selectors."""
    return ModuleFamily(
        "example",
        Baseline,
        {"alternate": f"{__name__}.Alternate", "fused": f"{__name__}.Fused"},
        {"enabled": SelectorSpec(bool, False), "fused": SelectorSpec(bool, False)},
        (
            VariantRule({"enabled": True, "fused": False}, "alternate"),
            VariantRule({"fused": True, "enabled": True}, "fused"),
        ),
    )


class TestModuleFamilySelectors(unittest.TestCase):
    """Registrations are isolated between tests; selection is never mocked."""

    def setUp(self) -> None:
        """Restore the exact registry contents after each test."""
        self.registry_patch = patch.dict(registry._FAMILIES, clear=True)
        self.registry_patch.start()
        self.addCleanup(self.registry_patch.stop)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_complete_whitelist_and_defaults(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: Only baseline defaults and fully registered combinations are accepted.
        Expectation: Defaults and declared complete combinations succeed; every other combination raises.
        """
        register_module_family(declaration())
        cases = [
            ({}, Baseline),
            ({"enabled": False}, Baseline),
            ({"fused": False}, Baseline),
            ({"enabled": True}, Alternate),
            ({"enabled": True, "fused": False}, Alternate),
            ({"enabled": True, "fused": True}, Fused),
        ]
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                inputs = {"width": 12, **arguments}
                original = inputs.copy()
                target, kwargs = resolve_module_family_target(Baseline, inputs)
                self.assertIs(target, expected)
                self.assertEqual(kwargs, {"width": 12})
                self.assertEqual(inputs, original)
        for arguments in ({"fused": True}, {"enabled": False, "fused": True}):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "legal combinations"):
                resolve_module_family_target(Baseline, arguments)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_types_and_unknown_arguments_are_rejected(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: Integers/strings cannot masquerade as bool and **kwargs cannot hide typos.
        Expectation: Wrong scalar types and constructor typos raise instead of being coerced or ignored.
        """
        register_module_family(declaration())
        for value in (0, 1, "true", "false", None, [], {}):
            with self.subTest(value=value), self.assertRaises(TypeError):
                resolve_module_family_target(Baseline, {"enabled": value})
        with self.assertRaisesRegex(TypeError, "unknown replacement arguments"):
            resolve_module_family_target(Baseline, {"enabled": True, "unsupported_argument": 8})

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_field_order_is_irrelevant(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: Complete named mappings match regardless of insertion order.
        Expectation: Reordering mapping keys selects exactly the same implementation.
        """
        family = declaration()
        register_module_family(replace(family, selector_specs=dict(reversed(list(family.selector_specs.items())))))
        self.assertIs(resolve_module_family_target(Baseline, {"fused": True, "enabled": True})[0], Fused)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_second_family_uses_its_own_schema(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: Names and types are supplied by a declaration, not global feature branches.
        Expectation: A different scalar schema works without adding feature-specific parser branches.
        """
        register_module_family(declaration())
        register_module_family(
            ModuleFamily(
                "second",
                Alternate,
                {"compact": f"{__name__}.Fused"},
                {"precision": SelectorSpec(str, "native"), "level": SelectorSpec(int, 0)},
                (VariantRule({"level": 2, "precision": "compact"}, "compact"),),
            )
        )
        self.assertIs(resolve_module_family_target(Alternate, {"precision": "compact", "level": 2})[0], Fused)
        with self.assertRaises(ValueError):
            resolve_module_family_target(Alternate, {"precision": "compact", "level": 3})
        with self.assertRaises(TypeError):
            resolve_module_family_target(Alternate, {"level": False})

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_invalid_registrations_are_atomic(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: Missing keys, extra keys, duplicate/default conditions and missing variants fail early.
        Expectation: Invalid declarations raise and leave the baseline unregistered.
        """
        family = declaration()
        invalid_rules = [
            (VariantRule({"enabled": True}, "alternate"),),
            (VariantRule({"enabled": True, "fused": False, "extra": True}, "alternate"),),
            (VariantRule({"enabled": True, "fused": False}, "missing"),),
            (VariantRule({"enabled": False, "fused": False}, "alternate"),),
            family.variant_rules + (VariantRule({"fused": False, "enabled": True}, "fused"),),
            (VariantRule({"enabled": 1, "fused": False}, "alternate"),),
        ]
        for rules in invalid_rules:
            with self.subTest(rules=rules), self.assertRaises((TypeError, ValueError)):
                register_module_family(replace(family, variant_rules=rules))
            self.assertIsNone(get_module_family(Baseline))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_invalid_schema_and_reserved_names(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: Schemas use immutable scalars and cannot consume replacement runtime arguments.
        Expectation: Invalid names, scalar types, defaults and paths fail during registration.
        """
        for name, spec in [
            ("", SelectorSpec(bool, False)),
            ("module", SelectorSpec(bool, False)),
            ("context", SelectorSpec(bool, False)),
            ("_target_", SelectorSpec(str, "x")),
            ("enabled", SelectorSpec(bool, 0)),
            ("enabled", SelectorSpec(list, [])),
            ("ratio", SelectorSpec(float, float("nan"))),
        ]:
            with self.subTest(name=name, spec=spec), self.assertRaises((TypeError, ValueError)):
                register_module_family(replace(declaration(), selector_specs={name: spec}, variant_rules=()))
        with self.assertRaises(ValueError):
            register_module_family(replace(declaration(), variants={"alternate": "not_a_path"}))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_registry_snapshot_is_read_only(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: Mutating caller mappings after registration cannot rewrite checked rules.
        Expectation: Caller mutation cannot change selection and registered mappings reject writes.
        """
        family = declaration()
        register_module_family(family)
        family.variants.clear()
        family.selector_specs.clear()
        family.variant_rules[0].when["enabled"] = False
        registered = get_module_family(Baseline)
        self.assertIs(resolve_module_family_target(Baseline, {"enabled": True})[0], Alternate)
        with self.assertRaises(TypeError):
            registered.variants["alternate"] = "changed.Path"
        with self.assertRaises(TypeError):
            registered.variant_rules[0].when["enabled"] = False

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_duplicate_baseline_is_rejected(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: A later registration cannot replace an existing baseline's contract.
        Expectation: A second registration for the same baseline raises.
        """
        register_module_family(declaration())
        with self.assertRaises(ValueError):
            register_module_family(declaration())

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_lazy_import_and_missing_implementation(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: Registration/default selection do not import variants; missing variants never fall back.
        Expectation: Default selection imports nothing; unavailable selected implementations raise.
        """
        register_module_family(
            replace(
                declaration(),
                variants={
                    "alternate": "unavailable_family_dependency.Alternate",
                    "fused": "unavailable_family_dependency.Fused",
                },
            )
        )
        with patch.object(registry, "import_module", wraps=import_module) as importer:
            self.assertIs(resolve_module_family_target(Baseline, {})[0], Baseline)
            importer.assert_not_called()
            with self.assertRaises(ModuleNotFoundError):
                resolve_module_family_target(Baseline, {"enabled": True})
            importer.assert_called_once_with("unavailable_family_dependency")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_ordinary_target_and_subclass_are_untouched(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: Family selector names do not become global reserved constructor arguments.
        Expectation: Unregistered callables keep their original arguments and remain outside the Family.
        """
        register_module_family(declaration())
        for target in (Alternate, type("Baseline", (Baseline,), {})):
            self.assertIsNone(get_module_family(target))
            args = {"enabled": "ordinary-value", "custom": object()}
            selected, kwargs = resolve_module_family_target(target, args)
            self.assertIs(selected, target)
            self.assertEqual(kwargs, args)
            self.assertIsNot(kwargs, args)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_three_field_declaration_is_baseline_only(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: No selector schema means only baseline defaults; no implicit feature names.
        Expectation: An empty schema selects the baseline and does not infer feature selectors.
        """
        register_module_family(ModuleFamily("plain", Baseline, {}))
        self.assertIs(resolve_module_family_target(Baseline, {"width": 2})[0], Baseline)
        with self.assertRaises(TypeError):
            resolve_module_family_target(Baseline, {"enabled": True})

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_callable_identity_does_not_depend_on_hash_or_equality(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: A callable's registration must not alter a distinct equal or unhashable callable.
        Expectation: Distinct callable instances remain distinct even when equality is overloaded.
        """

        class CallableObject:
            __hash__ = None

            def __call__(self, width: int = 8) -> int:
                """Return a normal constructor argument."""
                return width

            def __eq__(self, other: object) -> bool:
                """Make equality unsuitable for distinguishing the two instances."""
                return isinstance(other, CallableObject)

        first, second = CallableObject(), CallableObject()
        register_module_family(ModuleFamily("callable_object", first, {}))
        self.assertIs(get_module_family(first).baseline, first)
        self.assertIsNone(get_module_family(second))
        self.assertIs(resolve_module_family_target(second, {"width": 4})[0], second)
