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
"""CPU regressions for model-local operator dispatch."""

from __future__ import annotations

import unittest
from importlib import import_module
from pkgutil import iter_modules

import torch
from torch import nn

from hyper_parallel.components.ops import families, ops, swiglu
from hyper_parallel.components.ops.catalog import builtin_registry
from hyper_parallel.components.ops.factory import OpRuntime, bind_model_runtime
from hyper_parallel.components.ops.families.swiglu import SWIGLU
from hyper_parallel.components.ops.registry import ImplementationUnavailable, OpBuildContext, OpRegistry
from hyper_parallel.components.ops.spec import OpSpec
from hyper_parallel.core.activation_memory.api import checkpoint


def _missing() -> None:
    raise ImplementationUnavailable("optional kernel is missing")


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _requires_bfloat16(context: OpBuildContext) -> str | None:
    return None if context.input_dtype == torch.bfloat16 else "requires BF16 input"


class _CheckpointedActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Checkpoint an activation using the configured model runtime."""
        return checkpoint(swiglu, x)


class TestBuiltinFamilyFiles(unittest.TestCase):
    """Keep operator-family filenames aligned with their YAML keys."""

    def test_each_module_declares_its_own_family(self) -> None:
        """Every family file declares variants for its matching configuration key."""
        modules = list(iter_modules(families.__path__))
        self.assertTrue(modules)
        for module in modules:
            with self.subTest(family=module.name):
                namespace = vars(import_module(f"{families.__name__}.{module.name}"))
                keys = {spec.config_key for spec in namespace.values() if isinstance(spec, OpSpec)}
                self.assertEqual(keys, {module.name})
                builtin_registry.validate_selections({module.name: "auto"})


class TestOpRuntime(unittest.TestCase):
    """Exercise different selections with the same imported function."""

    def setUp(self) -> None:
        """Build a CPU-only catalog with distinct and unavailable candidates."""
        self.registry = OpRegistry()
        self.registry.register_family("swiglu", specs=(SWIGLU,))
        self.registry.register_impl(SWIGLU, name="identity", devices=("cpu", "cuda"), load=lambda: _identity)
        self.registry.register_impl(SWIGLU, name="square", devices=("cpu",), load=lambda: torch.square)
        self.registry.register_impl(
            SWIGLU, name="square", devices=("cuda",), load=lambda: torch.square, check_compatibility=_requires_bfloat16,
        )
        self.registry.register_impl(SWIGLU, name="missing", devices=("cpu",), load=_missing)
        self.registry.set_auto_order(SWIGLU, device="cpu", candidates=("missing", "identity"))
        self.registry.set_auto_order(SWIGLU, device="cuda", candidates=("square", "identity"))
        self.registry.seal()

    def _runtime(self, **selections: str) -> OpRuntime:
        return ops.configure(registry=self.registry, selections=selections, target_device="cpu")

    def test_two_models_restore_the_prior_runtime(self) -> None:
        """Nested model scopes and failures preserve the outer selection."""
        x = torch.full((2,), 3.0)
        outer = self._runtime()
        inner = self._runtime(swiglu="square")
        missing = self._runtime(swiglu="missing")

        with outer.activate():
            torch.testing.assert_close(swiglu(x), x)
            with inner.activate():
                torch.testing.assert_close(swiglu(x), x.square())
            with self.assertRaisesRegex(RuntimeError, "optional kernel is missing"), missing.activate():
                swiglu(x)
            torch.testing.assert_close(swiglu(x), x)

        self.assertEqual(outer.report(), {("swiglu", "standard"): "identity"})
        self.assertEqual(inner.report(), {("swiglu", "standard"): "square"})
        self.assertEqual(missing.report(), {})

    def test_checkpoint_restores_model_choice_during_backward(self) -> None:
        """A backward outside the model call still replays the original backend."""
        model = _CheckpointedActivation()
        bind_model_runtime(model, self._runtime(swiglu="square"))
        x = torch.full((2,), 3.0, requires_grad=True)

        output = model(x)
        output.sum().backward()

        torch.testing.assert_close(output, x.square())
        torch.testing.assert_close(x.grad, 2 * x)

    def test_cuda_selection_uses_input_dtype(self) -> None:
        """CUDA candidate selection follows activation dtype and caches each choice."""
        runtime = ops.configure(registry=self.registry, selections={}, target_device="cuda", dtype=torch.float32)
        fp32 = torch.full((2,), 3.0)
        bf16 = fp32.to(torch.bfloat16)

        with runtime.activate():
            torch.testing.assert_close(swiglu(fp32), fp32)
            torch.testing.assert_close(swiglu(bf16), bf16.square())
            torch.testing.assert_close(swiglu(fp32), fp32)
        self.assertCountEqual(runtime.used_backends(), (
            ("swiglu", "standard", "identity"),
            ("swiglu", "standard", "square"),
        ))
