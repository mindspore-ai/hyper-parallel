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
"""CPU regressions for process-wide operator backend selection."""

from __future__ import annotations

import ast
import inspect
import unittest
from importlib import import_module
from inspect import signature
from pkgutil import iter_modules
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import DefaultDeviceType
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from hyper_parallel.components import ops as public_ops
from hyper_parallel.components.ops import impl, ops, swiglu
from hyper_parallel.components.ops.catalog import builtin_registry
from hyper_parallel.components.ops import factory
from hyper_parallel.components.ops.impl.swiglu import SWIGLU
from hyper_parallel.components.ops.registry import ImplementationUnavailable, OpRegistry
from hyper_parallel.components.ops.spec import OpSpec
from hyper_parallel.core.activation_memory.api import checkpoint


def _missing() -> None:
    raise ImplementationUnavailable("optional kernel is missing")


def _identity(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    del dim
    return x


def _square(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    del dim
    return x.square()


class _CheckpointedActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Checkpoint an activation using the process backend configuration."""
        return checkpoint(swiglu, x)


class TestBuiltinFamilyFiles(unittest.TestCase):
    """Keep operator-family filenames aligned with their YAML keys."""

    def test_each_module_declares_its_own_family(self) -> None:
        """Every family file declares variants for its matching configuration key."""
        modules = list(iter_modules(impl.__path__))
        self.assertTrue(modules)
        for module in modules:
            with self.subTest(family=module.name):
                namespace = vars(import_module(f"{impl.__name__}.{module.name}"))
                keys = {spec.config_key for spec in namespace.values() if isinstance(spec, OpSpec)}
                self.assertEqual(keys, {module.name})
                builtin_registry.validate_selections({module.name: "auto"})

    def test_every_variant_has_an_explicit_public_function(self) -> None:
        """Source functions expose real signatures and document every registered variant."""
        self.assertNotIn("__getattr__", vars(public_ops))
        self.assertEqual(len(public_ops.__all__), len(set(public_ops.__all__)))
        for module in iter_modules(impl.__path__):
            namespace = vars(import_module(f"{impl.__name__}.{module.name}"))
            contracts = {name: value for name, value in namespace.items() if isinstance(value, OpSpec)}
            covered = set()
            for name in public_ops.__all__:
                function = getattr(public_ops, name)
                if not inspect.isfunction(function) or function.__module__ != namespace["__name__"]:
                    continue
                with self.subTest(function=name):
                    self.assertIs(function, namespace[name])
                    parameters = signature(function).parameters
                    self.assertNotIn("self", parameters)
                    self.assertNotIn("select", parameters)
                    self.assertNotIn("backend", parameters)
                    self.assertTrue(
                        all(value.annotation is not inspect.Parameter.empty for value in parameters.values())
                    )
                    self.assertIsNot(signature(function).return_annotation, inspect.Signature.empty)
                    tree = ast.parse(inspect.getsource(function))
                    references = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} & contracts.keys()
                    covered.update(contracts[key] for key in references)
                    if references:
                        documentation = inspect.getdoc(function)
                        for section in ("Args:", "Returns:", "Notes:"):
                            self.assertIn(section, documentation)
            self.assertEqual(covered, set(contracts.values()))
        self.assertEqual(next(iter(signature(public_ops.aggregate_hidden).parameters)), "input_tensor")
        self.assertEqual(next(iter(signature(public_ops.grouped_query_attention).parameters)), "module")


class TestBackendSelection(unittest.TestCase):
    """Exercise configuration replacement through imported operator functions."""

    def setUp(self) -> None:
        """Install a CPU-testable catalog and restore process settings after each case."""
        self.registry = OpRegistry()
        self.registry.register_family("swiglu", specs=(SWIGLU,))
        self.registry.register_impl(SWIGLU, name="identity", devices=("cpu", "cuda"), load=lambda: _identity)
        self.registry.register_impl(SWIGLU, name="square", devices=("cpu", "cuda"), load=lambda: _square)
        self.registry.register_impl(SWIGLU, name="missing", devices=("cpu",), load=_missing)
        for device in ("cpu", "cuda"):
            candidates = ("missing", "identity") if device == "cpu" else ("identity",)
            self.registry.set_auto_order(SWIGLU, device=device, candidates=candidates)
        self.registry.seal()
        previous = ops.get_backends()
        patcher = patch.object(factory, "builtin_registry", self.registry)
        patcher.start()
        self.addCleanup(ops.set_backends, previous)
        self.addCleanup(patcher.stop)
        ops.set_backends({})

    def test_settings_are_snapshots_and_failures_preserve_the_active_configuration(self) -> None:
        """Copy input and output mappings, and reject invalid updates atomically."""
        selections = {"swiglu": "square"}
        ops.set_backends(selections)
        selections["swiglu"] = "missing"
        queried = ops.get_backends()
        queried.clear()
        x = torch.full((2,), 3.0)
        torch.testing.assert_close(swiglu(x), x.square())
        self.assertEqual(ops.get_backends(), {"swiglu": "square"})
        for invalid in ({"unknown": "auto"}, {"swiglu": "unknown"}, {"swiglu": None}, []):
            with self.subTest(selections=invalid), self.assertRaises(ValueError):
                ops.set_backends(invalid)
            self.assertEqual(ops.get_backends(), {"swiglu": "square"})
            self.assertEqual(ops.used_backends(), (("swiglu", "standard", "square"),))

    def test_replacing_settings_clears_cached_choices_and_usage(self) -> None:
        """Imported callables observe each complete configuration replacement."""
        x = torch.full((2,), 3.0)
        ops.set_backends({"swiglu": "square"})
        torch.testing.assert_close(swiglu(x), x.square())
        ops.set_backends({})
        self.assertEqual(ops.get_backends(), {})
        self.assertEqual(ops.used_backends(), ())
        torch.testing.assert_close(swiglu(x), x)
        self.assertEqual(ops.used_backends(), (("swiglu", "standard", "identity"),))
        ops.set_backends({"swiglu": "missing"})
        with self.assertRaisesRegex(RuntimeError, "optional kernel is missing"):
            swiglu(x)
        self.assertEqual(ops.used_backends(), ())

    def test_cached_calls_and_threads_share_process_configuration(self) -> None:
        """Worker threads use the configured backend and share a deduplicated report."""
        ops.set_backends({"swiglu": "square"})
        x = torch.full((2,), 3.0)
        with patch.object(self.registry, "resolve", wraps=self.registry.resolve) as resolve:
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(swiglu, [x, x, x]))
        for result in results:
            torch.testing.assert_close(result, x.square())
        resolve.assert_called_once()
        self.assertEqual(ops.used_backends(), (("swiglu", "standard", "square"),))

    def test_torch_and_component_checkpoint_use_process_settings(self) -> None:
        """Ordinary backward replays both checkpoint APIs without model wrappers."""
        # CPU-only checkpoint inputs otherwise inherit the accelerator default.
        previous_device_type = DefaultDeviceType.get_device_type()
        self.addCleanup(DefaultDeviceType.set_device_type, previous_device_type)
        DefaultDeviceType.set_device_type("cpu")
        ops.set_backends({"swiglu": "square"})
        for checkpoint_fn in (checkpoint, torch_checkpoint):
            for reentrant in (True, False):
                with self.subTest(checkpoint=checkpoint_fn.__module__, reentrant=reentrant):
                    x = torch.full((2,), 3.0, requires_grad=True)
                    output = checkpoint_fn(swiglu, x, use_reentrant=reentrant)
                    output.sum().backward()
                    torch.testing.assert_close(output, x.square())
                    torch.testing.assert_close(x.grad, 2 * x)
        model = _CheckpointedActivation()
        x = torch.full((2,), 3.0, requires_grad=True)
        model(x).sum().backward()
        torch.testing.assert_close(x.grad, 2 * x)

    def test_device_key_reuses_choices_across_dtypes_and_shapes(self) -> None:
        """Only execution device separates cache entries; dtype, shape and position do not."""
        def scaled(scale: float, values: torch.Tensor, device: torch.device) -> torch.Tensor:
            operation = factory._select_for_call(SWIGLU, device=device)  # pylint: disable=protected-access
            return operation(values) * scale

        with patch.object(self.registry, "resolve", wraps=self.registry.resolve) as resolve:
            for device in (torch.device("cpu"), torch.device("cuda", 0), torch.device("cuda", 1)):
                for shape, dtype in (((2,), torch.float32), ((3, 4), torch.bfloat16), ((), torch.float64)):
                    values = torch.ones(shape, dtype=dtype)
                    torch.testing.assert_close(scaled(2.0, values, device), values * 2)
            self.assertEqual(resolve.call_count, 3)
        self.assertEqual(ops.used_backends(), (("swiglu", "standard", "identity"),))

    def test_selected_implementation_errors_do_not_trigger_fallback(self) -> None:
        """Input validation remains active on cache hits and cannot change backend."""
        def positive_only(values: torch.Tensor) -> torch.Tensor:
            if values.ndim == 0:
                raise ValueError("scalar tensors are unsupported")
            return values

        self.registry = OpRegistry()
        self.registry.register_family("swiglu", specs=(SWIGLU,))
        self.registry.register_impl(SWIGLU, name="limited", devices=("cpu",), load=lambda: positive_only)
        self.registry.register_impl(SWIGLU, name="native", devices=("cpu",), load=lambda: torch.neg)
        self.registry.set_auto_order(SWIGLU, device="cpu", candidates=("limited", "native"))
        self.registry.seal()
        with patch.object(factory, "builtin_registry", self.registry):
            ops.set_backends({})
            operation = factory._select_for_call(SWIGLU, device=torch.device("cpu"))  # pylint: disable=protected-access
            torch.testing.assert_close(operation(torch.ones(2)), torch.ones(2))
            operation = factory._select_for_call(SWIGLU, device=torch.device("cpu"))  # pylint: disable=protected-access
            with self.assertRaisesRegex(ValueError, "scalar tensors"):
                operation(torch.ones(()))
        self.assertEqual(ops.used_backends(), (("swiglu", "standard", "limited"),))

    def test_compiled_calls_observe_configuration_changes(self) -> None:
        """A full graph captures the selected callable and guards its configuration."""
        self.addCleanup(torch.compiler.reset)
        torch.compiler.reset()
        ops.set_backends({"swiglu": "square"})
        compiled = torch.compile(swiglu, backend="eager", fullgraph=True)
        x = torch.full((2,), 3.0, requires_grad=True)
        output = compiled(x)
        output.sum().backward()
        torch.testing.assert_close(output, x.square())
        torch.testing.assert_close(x.grad, 2 * x)
        self.assertEqual(ops.used_backends(), (("swiglu", "standard", "square"),))
        ops.set_backends({"swiglu": "identity"})
        torch.testing.assert_close(compiled(x), x)
        self.assertEqual(ops.used_backends(), (("swiglu", "standard", "identity"),))
