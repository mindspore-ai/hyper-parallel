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
"""Numerical and dispatch regressions for lazy concrete-size FX variants."""

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.compile import GraphCompiler, PassConfig
from hyper_parallel.models.build_options import CompileConfig
from tests.common.mark_utils import arg_mark


def _loss(model, x, y, weight):
    loss = (model(x) - y).square().mean() * weight
    return loss, {"mse": loss.detach()}


class TestSizeSpecialization(unittest.TestCase):
    """Exercise real generated variants and general fallback on CPU."""

    def setUp(self) -> None:
        """Create a CPU model for each independent dispatch test."""
        torch.manual_seed(42)
        self.model = torch.nn.Linear(4, 3)

    def _compiler(self, **kwargs):
        options = {"dynamic_arg_dims": {"x": [0, 1], "y": [0, 1]}, "compile_sizes": [7],
                   "compile_size_input": "x", "compile_size_dim": 1}
        options.update(kwargs)
        return GraphCompiler(self.model, _loss, pass_config=PassConfig(fsdp_enabled=False),
                             device=torch.device("cpu"), **options)

    def _step(self, compiler, batch, length, weight=1.0):
        x, y = torch.randn(batch, length, 4), torch.randn(batch, length, 3)
        scale = torch.tensor(weight)
        self.model.zero_grad()
        actual, losses = compiler.forward_backward(x=x, y=y, weight=scale)
        expected, _ = _loss(self.model, x, y, scale)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(losses["mse"], expected.detach())
        grads = torch.autograd.grad(expected, tuple(self.model.parameters()))
        for parameter, gradient in zip(self.model.parameters(), grads):
            torch.testing.assert_close(parameter.grad, gradient)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_lazy_compile_hit_and_general_fallback(self) -> None:
        """
        Feature: Lazy FX specialization across multiple configured sizes.
        Description: Exercise warmup, two hot sizes, repeated hits and cold-size fallback.
        Expectation: One model capture yields two variants with eager loss and gradient parity.
        """
        compiler = self._compiler(compile_sizes=[5, 7])
        with patch.object(compiler, "compile", wraps=compiler.compile) as capture:
            self._step(compiler, 2, 5)
            general_code = compiler._joint_graph.graph_module.code
            self.assertEqual(compiler.specialization_stats["compilations"], 0)
            self._step(compiler, 2, 7)
            entry = next(iter(compiler._size_dispatcher.entries.values()))
            self.assertGreater(entry.folded_nodes, 0)
            self.assertNotIn("sym_size", entry.graph_module.code)
            self.assertNotEqual(entry.graph_module.code, general_code)
            self._step(compiler, 2, 7, weight=0.25)
            self._step(compiler, 2, 9)
            self._step(compiler, 2, 5)
            self._step(compiler, 2, 5, weight=0.5)
            self.assertEqual(capture.call_count, 1)
        stats = compiler.specialization_stats
        self.assertEqual(stats["compilations"], 2)
        self.assertEqual(stats["compiled_sizes"], [5, 7])
        self.assertEqual((stats["general_calls"], stats["specialized_calls"], stats["cache_hits"]), (2, 4, 2))
        self.assertEqual(compiler._joint_graph.graph_module.code, general_code)

    def test_first_hot_call_runs_general(self):
        """The initial execution follows MagiCompiler's general-graph warmup policy."""
        compiler = self._compiler()
        self._step(compiler, 2, 7)
        self.assertEqual(compiler.specialization_stats["compilations"], 0)
        self._step(compiler, 2, 7)
        self.assertEqual(compiler.specialization_stats["compilations"], 1)

    def test_secondary_dimensions_and_capacity(self):
        """Equal dispatch sizes with different batch dimensions require distinct variants."""
        compiler = self._compiler(max_specializations=2)
        for batch, length in [(2, 5), (2, 7), (3, 7), (4, 7), (2, 7)]:
            self._step(compiler, batch, length)
        stats = compiler.specialization_stats
        self.assertEqual(stats["compilations"], 2)
        self.assertEqual(stats["capacity_fallbacks"], 1)
        self.assertEqual(stats["cache_hits"], 1)

    def test_live_weights_optimizer_and_accumulation(self):
        """Cached variants consume updated weights and accumulate gradients normally."""
        compiler = self._compiler()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        for length in (5, 7, 7, 9, 7):
            self._step(compiler, 2, length)
            optimizer.step()
        x, y, weight = torch.randn(2, 7, 4), torch.randn(2, 7, 3), torch.tensor(0.5)
        self.model.zero_grad()
        expected, _ = _loss(self.model, x, y, weight)
        gradients = torch.autograd.grad(expected, tuple(self.model.parameters()))
        for _ in range(2):
            compiler.forward_backward(x=x, y=y, weight=weight)
        for parameter, gradient in zip(self.model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, gradient * 2)

    def test_guards_run_before_cached_dispatch(self):
        """Neither a cache hit nor a configured size may bypass general guards."""
        compiler = self._compiler()
        self._step(compiler, 2, 5)
        self._step(compiler, 2, 7)
        stats = compiler.specialization_stats
        with self.assertRaisesRegex(ValueError, "rank, dtype"):
            compiler.forward_backward(x=torch.randn(2, 7, 4).double(), y=torch.randn(2, 7, 3),
                                      weight=torch.tensor(1.0))
        with self.assertRaisesRegex(ValueError, "shape/stride"):
            compiler.forward_backward(x=torch.randn(2, 7, 4), y=torch.randn(2, 8, 3), weight=torch.tensor(1.0))
        self.assertEqual(compiler.specialization_stats, stats)

    def test_auto_selection_and_nested_explicit_path(self):
        """Automatic selection uses a symbolic axis; explicit selection supports pytree paths."""
        compiler = self._compiler(compile_size_input=None, compile_sizes=[3])
        self._step(compiler, 2, 5)
        self._step(compiler, 3, 5)
        self.assertEqual(compiler.specialization_stats["compiled_sizes"], [3])
        nested = GraphCompiler(self.model, lambda model, batch: model(batch["x"]).square().mean(),
                               pass_config=PassConfig(fsdp_enabled=False), dynamic=True,
                               compile_sizes=[7], compile_size_input="batch.x", compile_size_dim=-2)
        nested.forward_backward(batch={"x": torch.randn(2, 5, 4)})
        nested.forward_backward(batch={"x": torch.randn(2, 7, 4)})
        self.assertEqual(nested.specialization_stats["compiled_sizes"], [7])

    def test_storage_not_retained(self):
        """The variant cache contains metadata and graphs, never real user tensor storage."""
        compiler = self._compiler()
        self._step(compiler, 2, 5)
        x, y = torch.randn(2, 7, 4), torch.randn(2, 7, 3)
        refs = weakref.ref(x), weakref.ref(y)
        compiler.forward_backward(x=x, y=y, weight=torch.tensor(1.0))
        del x, y
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_specialization_does_not_execute_rng(self) -> None:
        """
        Feature: Runtime RNG semantics during size specialization.
        Description: Compare random tensor losses and RNG states with eager execution.
        Expectation: Variant generation consumes no RNG and tensor values remain live.
        """
        def noisy_loss(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            """Compute a loss with runtime random tensor values."""
            return model(x + torch.randn_like(x)).square().mean()

        compiler = GraphCompiler(self.model, noisy_loss, pass_config=PassConfig(fsdp_enabled=False),
                                 dynamic=True, compile_sizes=[7], compile_size_input="x", compile_size_dim=1)
        compiler.forward_backward(x=torch.randn(2, 5, 4))
        for seed in (123, 456):
            x = torch.randn(2, 7, 4)
            self.model.zero_grad()
            torch.manual_seed(seed)
            actual, _ = compiler.forward_backward(x=x)
            state_after_graph = torch.get_rng_state()
            torch.manual_seed(seed)
            expected = noisy_loss(self.model, x)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(torch.get_rng_state(), state_after_graph)
            gradients = torch.autograd.grad(expected, tuple(self.model.parameters()))
            for parameter, gradient in zip(self.model.parameters(), gradients):
                torch.testing.assert_close(parameter.grad, gradient)

    def test_configuration_inheritance_and_disabled_default(self):
        """Trainer-level configuration reaches the compiler without changing defaults."""
        config = CompileConfig(dynamic=True, compile_sizes=[7], compile_size_input="x", compile_size_dim=1)
        compiler = GraphCompiler(self.model, _loss, trainer_config=SimpleNamespace(compile=config),
                                 pass_config=PassConfig(fsdp_enabled=False))
        self._step(compiler, 2, 5)
        self._step(compiler, 2, 7)
        self.assertEqual(compiler.specialization_stats["compilations"], 1)
        disabled = self._compiler(compile_sizes=[])
        self._step(disabled, 2, 5)
        self.assertEqual(disabled.specialization_stats, {})

    def test_invalid_policy_and_selector(self):
        """Reject invalid policy or input selection before parallel transformation."""
        for options in ({"compile_sizes": [True]}, {"compile_sizes": [-1]}, {"compile_sizes": "7"},
                        {"max_specializations": 0}, {"max_specializations": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self._compiler(**options)
        with self.assertRaisesRegex(ValueError, "requires dynamic"):
            self._compiler(dynamic=False, dynamic_arg_dims=None)
        for options in ({"compile_size_input": "missing"}, {"compile_size_dim": 5},
                        {"compile_size_dim": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self._step(self._compiler(**options), 2, 5)
