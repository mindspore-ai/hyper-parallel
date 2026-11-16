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
"""CPU contracts for factory-selected Muon NS implementations."""

import copy
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.optimizer import get_hyper_optimizer
from hyper_parallel.core.optimizer.muon import _NS_ASYM5_COEFFS, _NS_LEGACY_COEFFS


class TestMuonNSBackendFactory(unittest.TestCase):
    """One keyword must bind effective settings without changing native construction."""

    def setUp(self) -> None:
        """Use real CPU Muon construction and a stand-in optional backend factory."""
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 8, bias=False), torch.nn.Linear(8, 4, bias=False),
        )
        self.parameters = list(self.model.parameters())
        self.callback = Mock(side_effect=lambda inputs, steps: inputs.clone())
        patcher = patch(
            "hyper_parallel.components.functional.gram_newton_schulz.make_gram_newton_schulz",
            return_value=self.callback,
        )
        self.builder = patcher.start()
        self.addCleanup(patcher.stop)

    def test_single_keyword_uses_native_defaults_and_runs_step(self) -> None:
        """Gram selection must use asym5/epsilon defaults and reach the callback."""
        chain = get_hyper_optimizer(self.model, self.parameters, [], muon_ns_backend="gram")
        self.builder.assert_called_once_with(_NS_ASYM5_COEFFS, epsilon=1e-10, use_kernels=False)
        self.assertIs(chain.optimizer.zeropower_fn, self.callback)
        for parameter in self.parameters:
            parameter.grad = torch.ones_like(parameter)
        chain.step()
        self.assertGreater(self.callback.call_count, 0)
        for call in self.callback.call_args_list:
            self.assertEqual(call.kwargs["steps"], 5)

    def test_gram_quack_requests_device_based_kernel_selection(self) -> None:
        """Quack must select kernels per input device so the Torch fallback is available."""
        chain = get_hyper_optimizer(self.model, self.parameters, [], muon_ns_backend="gram_quack")
        self.builder.assert_called_once_with(_NS_ASYM5_COEFFS, epsilon=1e-10, use_kernels=None)
        self.assertIs(chain.optimizer.zeropower_fn, self.callback)

    def test_native_preserves_manual_callback(self) -> None:
        """Existing manual callback configuration must keep working by default."""
        chain = get_hyper_optimizer(
            self.model, self.parameters, [], muon_kwargs={"zeropower_fn": self.callback},
        )
        self.assertIs(chain.optimizer.zeropower_fn, self.callback)
        self.builder.assert_not_called()

    def test_invalid_backend_and_callback_conflict(self) -> None:
        """Typos and conflicting explicit requests must fail before loading the backend."""
        with self.assertRaisesRegex(ValueError, "muon_ns_backend"):
            get_hyper_optimizer(self.model, self.parameters, [], muon_ns_backend="typo")
        for backend in ("gram", "gram_quack"):
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, f"'{backend}'.*zeropower_fn"):
                get_hyper_optimizer(
                    self.model, self.parameters, [], muon_ns_backend=backend,
                    muon_kwargs={"zeropower_fn": self.callback},
                )
        self.builder.assert_not_called()

    def test_prefixed_legacy_settings_are_resolved_without_mutating_input(self) -> None:
        """Normalize legacy-prefixed kwargs and bind repeated coefficients for all steps."""
        configuration = {"muon_ns_variant": "legacy", "muon_ns_steps": 6, "muon_ns_epsilon": 1e-8}
        original = copy.deepcopy(configuration)
        chain = get_hyper_optimizer(
            self.model, self.parameters, [], muon_kwargs=configuration, muon_ns_backend="gram",
        )
        self.builder.assert_called_once_with((_NS_LEGACY_COEFFS,) * 6, epsilon=1e-8, use_kernels=False)
        self.assertEqual(chain.optimizer.param_groups[0]["ns_variant"], "legacy")
        self.assertEqual(configuration, original)

    def test_effective_group_overrides_and_different_step_counts(self) -> None:
        """Read filled groups rather than top-level defaults; share compatible prefixes."""
        coefficients = [(2.0, -1.0, 0.25), (1.5, -0.5, 0.1), (1.0, 0.0, 0.0)]
        groups = [
            {"params": [self.parameters[0]], "ns_steps": 1},
            {"params": [self.parameters[1]], "ns_steps": 3},
        ]
        for group in groups:
            group.update(ns_variant="custom", ns_coefficients=coefficients, ns_epsilon=1e-6)
        get_hyper_optimizer(self.model, groups, [], muon_ns_backend="gram")
        self.builder.assert_called_once_with(tuple(tuple(row) for row in coefficients), epsilon=1e-6, use_kernels=False)

    def test_incompatible_groups_are_rejected_before_loading(self) -> None:
        """A single callback cannot represent different epsilons or coefficient schedules."""
        for override in ({"ns_epsilon": 1e-6}, {"ns_variant": "legacy"}):
            with self.subTest(override=override):
                groups = [
                    {"params": [self.parameters[0]]},
                    {"params": [self.parameters[1]], **override},
                ]
                with self.assertRaisesRegex(ValueError, "across Muon parameter groups"):
                    get_hyper_optimizer(self.model, groups, [], muon_ns_backend="gram")
        self.builder.assert_not_called()

    def test_group_steps_and_custom_schedule_validation(self) -> None:
        """Group overrides must not silently exceed or omit their coefficient schedule."""
        for override in (
            {"ns_steps": 0}, {"ns_steps": True}, {"ns_steps": 1.5}, {"ns_steps": 6},
            {"ns_variant": "custom"}, {"ns_variant": "unknown"},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                get_hyper_optimizer(
                    self.model, [{"params": self.parameters, **override}], [], muon_ns_backend="gram",
                )
        self.builder.assert_not_called()

    def test_empty_muon_does_not_load_optional_backend(self) -> None:
        """Global backend selection must remain harmless for AdamW-only parameter routing."""
        for backend in ("gram", "gram_quack"):
            with self.subTest(backend=backend):
                chain = get_hyper_optimizer(self.model, [], self.parameters, muon_ns_backend=backend)
                self.assertEqual(chain.optimizers_keys, ["adamw"])
        self.builder.assert_not_called()

    def test_dependency_errors_propagate(self) -> None:
        """An explicitly selected missing backend must not silently select native NS."""
        self.builder.side_effect = ImportError("optional dependency missing")
        for backend in ("gram", "gram_quack"):
            with self.subTest(backend=backend), self.assertRaisesRegex(ImportError, "optional dependency missing"):
                get_hyper_optimizer(self.model, self.parameters, [], muon_ns_backend=backend)

    def test_native_factory_import_isolation(self) -> None:
        """The default factory must not even load the optional backend's adapter module."""
        script = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('gram_newton_schulz', 'hyper_parallel.components.functional.gram_newton_schulz'):
            raise RuntimeError('unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import torch
from hyper_parallel.core.optimizer import get_hyper_optimizer
model = torch.nn.Linear(4, 8, bias=False)
optimizer = get_hyper_optimizer(model, list(model.parameters()), [])
assert optimizer.optimizer.zeropower_fn is None
"""
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
