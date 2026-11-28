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
"""Catalog validation, rejection diagnostics and ordered backend selection."""

import unittest

import torch

from hyper_parallel.components.ops.registry import OpRegistry
from hyper_parallel.components.ops.spec import OpSpec


class TestRegistryValidation(unittest.TestCase):
    """Reject invalid catalogs before they can influence process backend settings."""

    def setUp(self) -> None:
        """Build an unsealed catalog with one CPU implementation."""
        self.spec = OpSpec(config_key="activation")
        self.registry = OpRegistry()
        self.registry.register_family("activation", specs=(self.spec,))
        self.registry.register_impl(self.spec, name="native", devices=("cpu",), load=lambda: torch.square)

    def test_invalid_spec_names(self) -> None:
        """Configuration keys and variants must each be nonempty strings."""
        for name in (None, "", " "):
            for field in ("config_key", "variant"):
                with self.subTest(field=field, name=name), self.assertRaisesRegex(ValueError, "nonempty string"):
                    OpSpec(**{**{"config_key": "activation"}, field: name})

    def test_family_registration_rejects_invalid_contract_sets(self) -> None:
        """Reject duplicate families, duplicate variants, empty sets and mismatched keys."""
        cases = (
            ("activation", (self.spec,)),
            ("other", ()),
            ("other", (self.spec,)),
            ("activation", (self.spec, self.spec)),
            (" ", (OpSpec(config_key="other"),)),
        )
        for key, specs in cases:
            with self.subTest(key=key, specs=specs), self.assertRaisesRegex(ValueError, "Invalid or duplicate"):
                self.registry.register_family(key, specs=specs)

    def test_implementation_registration_rejects_invalid_metadata(self) -> None:
        """A loader, compatibility predicate, name and device set must all be valid."""
        cases = (
            {"name": "auto"},
            {"name": ""},
            {"devices": ()},
            {"devices": ("cpu", "cpu")},
            {"devices": ("xla",)},
            {"load": None},
            {"name": "native"},
        )
        for invalid in cases:
            settings = {"name": "alternative", "devices": ("cpu",), "load": lambda: torch.neg, **invalid}
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.registry.register_impl(self.spec, **settings)
        self.registry.seal()
        self.assertIs(self.registry.resolve(self.spec, "native", torch.device("cpu"))[1], torch.square)

    def test_foreign_contract_is_rejected(self) -> None:
        """Equal-valued contracts from another catalog cannot be registered or resolved."""
        other = OpSpec(config_key="activation")
        with self.assertRaisesRegex(ValueError, "Unregistered operator contract"):
            self.registry.register_impl(other, name="other", devices=("cpu",), load=lambda: torch.neg)
        with self.assertRaisesRegex(ValueError, "Unregistered operator contract"):
            self.registry.resolve(other, "native", torch.device("cpu"))

    def test_auto_order_rejects_unknown_or_duplicate_candidates(self) -> None:
        """Candidate lists refer to unique registered implementations on a valid device."""
        for device, names in (("xla", ()), ("cpu", ("missing",)), ("cpu", ("native", "native"))):
            with self.subTest(device=device, names=names), self.assertRaisesRegex(ValueError, "Invalid automatic"):
                self.registry.set_auto_order(self.spec, device=device, candidates=names)
        self.registry.set_auto_order(self.spec, device="cpu", candidates=("native",))
        with self.assertRaisesRegex(ValueError, "Invalid automatic"):
            self.registry.set_auto_order(self.spec, device="cpu", candidates=("native",))

    def test_unsealed_selection_and_sealed_mutation_are_rejected(self) -> None:
        """Configuration validation requires a sealed catalog; sealed catalogs are immutable."""
        with self.assertRaisesRegex(ValueError, "sealed registry"):
            self.registry.validate_selections({})
        self.registry.seal()
        with self.assertRaisesRegex(ValueError, "ops must be a mapping"):
            self.registry.validate_selections([])
        with self.assertRaisesRegex(ValueError, "Unknown operator family"):
            self.registry.validate_selections({None: "native"})
        with self.assertRaisesRegex(RuntimeError, "registry is sealed"):
            self.registry.register_family("other", specs=(OpSpec(config_key="other"),))

    def test_auto_order_and_explicit_selection_use_only_contract_and_device(self) -> None:
        """Resolve by registration and configured order without receiving input tensors."""
        self.registry.register_impl(self.spec, name="fused", devices=("cpu",), load=lambda: torch.neg)
        self.registry.set_auto_order(self.spec, device="cpu", candidates=("fused", "native"))
        self.registry.seal()
        self.assertEqual(self.registry.resolve(self.spec, "auto", torch.device("cpu"))[0], "fused")
        self.assertEqual(self.registry.resolve(self.spec, "native", torch.device("cpu"))[0], "native")
        with self.assertRaisesRegex(RuntimeError, "not registered for this contract and device"):
            self.registry.resolve(self.spec, "fused", torch.device("cuda", 1))

    def test_empty_auto_order_and_noncallable_loader_result(self) -> None:
        """Empty auto policies and broken loaders must fail without reporting a selection."""
        self.registry.register_impl(self.spec, name="broken", devices=("cpu",), load=lambda: None)
        self.registry.seal()
        with self.assertRaisesRegex(RuntimeError, "no auto candidates registered"):
            self.registry.resolve(self.spec, "auto", torch.device("cpu"))
        with self.assertRaisesRegex(TypeError, "did not return a callable"):
            self.registry.resolve(self.spec, "broken", torch.device("cpu"))
