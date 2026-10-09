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

import copy
import gc
import unittest
import weakref
from unittest.mock import patch

import torch

from tests.common.mark_utils import arg_mark

from hyper_parallel.compile import GraphCompiler, GraphTrainer, PassConfig


def _loss(model, x, y, weight):
    loss = (model(x) - y).square().mean() * weight
    return loss


class TestSizeSpecialization(unittest.TestCase):
    """Exercise real generated variants and general fallback on CPU."""

    def setUp(self) -> None:
        """Create a fresh CPU model for each cache policy test."""
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
        actual = compiler.forward_backward(x=x, y=y, weight=scale)
        expected = _loss(self.model, x, y, scale)
        torch.testing.assert_close(actual, expected)
        grads = torch.autograd.grad(expected, tuple(self.model.parameters()))
        for parameter, gradient in zip(self.model.parameters(), grads):
            torch.testing.assert_close(parameter.grad, gradient)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
              card_mark="onecard", essential_mark="essential")
    def test_lazy_compile_hit_and_general_fallback(self):
        """
        Feature: Lazy size specialization.
        Description: Hot shapes generate real constant shape code once; cold shapes reuse general code.
        Expectation: One capture generates real concrete FX code, reuses it and falls back for cold sizes.
        """
        compiler = self._compiler()
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
            self.assertEqual(capture.call_count, 1)
        stats = compiler.specialization_stats
        self.assertEqual((stats["general_calls"], stats["specialized_calls"], stats["cache_hits"]), (2, 2, 1))
        self.assertEqual(compiler._joint_graph.graph_module.code, general_code)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
              card_mark="onecard", essential_mark="essential")
    def test_first_hot_call_runs_general(self):
        """
        Feature: Lazy size specialization.
        Description: The initial execution follows MagiCompiler's general-graph warmup policy.
        Expectation: The first call stays general and the next eligible call generates one variant.
        """
        compiler = self._compiler()
        self._step(compiler, 2, 7)
        self.assertEqual(compiler.specialization_stats["compilations"], 0)
        self._step(compiler, 2, 7)
        self.assertEqual(compiler.specialization_stats["compilations"], 1)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
              card_mark="onecard", essential_mark="essential")
    def test_secondary_dimensions_and_capacity(self):
        """
        Feature: Lazy size specialization.
        Description: Equal dispatch sizes with different batch dimensions require distinct variants.
        Expectation: Different batches need separate variants and new signatures fall back at capacity.
        """
        compiler = self._compiler(max_specializations=2)
        for batch, length in [(2, 5), (2, 7), (3, 7), (4, 7), (2, 7)]:
            self._step(compiler, batch, length)
        stats = compiler.specialization_stats
        self.assertEqual(stats["compilations"], 2)
        self.assertEqual(stats["capacity_fallbacks"], 1)
        self.assertEqual(stats["cache_hits"], 1)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
              card_mark="onecard", essential_mark="essential")
    def test_live_weights_optimizer_and_accumulation(self):
        """
        Feature: Lazy size specialization.
        Description: Cached variants consume updated weights and accumulate gradients normally.
        Expectation: Cached graphs use updated weights and accumulate the expected gradients.
        """
        compiler = self._compiler()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        for length in (5, 7, 7, 9, 7):
            self._step(compiler, 2, length)
            optimizer.step()
        x, y, weight = torch.randn(2, 7, 4), torch.randn(2, 7, 3), torch.tensor(0.5)
        self.model.zero_grad()
        expected = _loss(self.model, x, y, weight)
        gradients = torch.autograd.grad(expected, tuple(self.model.parameters()))
        for _ in range(2):
            compiler.forward_backward(x=x, y=y, weight=weight)
        for parameter, gradient in zip(self.model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, gradient * 2)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
              card_mark="onecard", essential_mark="essential")
    def test_guards_run_before_cached_dispatch(self):
        """
        Feature: Lazy size specialization.
        Description: Neither a cache hit nor a configured size may bypass general guards.
        Expectation: Invalid inputs raise without changing cache counters or executing a variant.
        """
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
              card_mark="onecard", essential_mark="essential")
    def test_auto_selection_and_nested_explicit_path(self):
        """
        Feature: Lazy size specialization.
        Description: Automatic selection uses a symbolic axis; explicit selection supports pytree paths.
        Expectation: Automatic and nested selectors dispatch on the intended axis.
        """
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
              card_mark="onecard", essential_mark="essential")
    def test_storage_not_retained(self):
        """
        Feature: Lazy size specialization.
        Description: The variant cache contains metadata and graphs, never real user tensor storage.
        Expectation: Variant entries retain no real input tensor storage.
        """
        compiler = self._compiler()
        self._step(compiler, 2, 5)
        x, y = torch.randn(2, 7, 4), torch.randn(2, 7, 3)
        refs = weakref.ref(x), weakref.ref(y)
        compiler.forward_backward(x=x, y=y, weight=torch.tensor(1.0))
        del x, y
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
              card_mark="onecard", essential_mark="essential")
    def test_specialization_does_not_execute_rng(self):
        """
        Feature: Lazy size specialization.
        Description: Generating a variant neither consumes RNG nor freezes random tensor values.
        Expectation: Generation consumes no RNG and execution uses current random values.
        """
        def noisy_loss(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            """Consume live RNG values while computing a scalar training loss."""
            return model(x + torch.randn_like(x)).square().mean()

        compiler = GraphCompiler(self.model, noisy_loss, pass_config=PassConfig(fsdp_enabled=False),
                                 dynamic=True, compile_sizes=[7], compile_size_input="x", compile_size_dim=1)
        compiler.forward_backward(x=torch.randn(2, 5, 4))
        for seed in (123, 456):
            x = torch.randn(2, 7, 4)
            self.model.zero_grad()
            torch.manual_seed(seed)
            actual = compiler.forward_backward(x=x)
            state_after_graph = torch.get_rng_state()
            torch.manual_seed(seed)
            expected = noisy_loss(self.model, x)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(torch.get_rng_state(), state_after_graph)
            gradients = torch.autograd.grad(expected, tuple(self.model.parameters()))
            for parameter, gradient in zip(self.model.parameters(), gradients):
                torch.testing.assert_close(parameter.grad, gradient)


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
              card_mark="onecard", essential_mark="essential")
    def test_invalid_policy_and_selector(self):
        """
        Feature: Lazy size specialization.
        Description: Reject invalid policy or input selection before parallel transformation.
        Expectation: Invalid sizes, capacities, paths and axes raise descriptive errors.
        """
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
              card_mark="onecard", essential_mark="essential")
    def test_graphtrainer_train_loop_and_specialization(self) -> None:
        """
        Feature: Lazy size specialization.
        Description: Compare a complete cached GraphTrainer loop with eager optimizer updates.
        Expectation: Losses and final weights match eager, with one capture and a cache hit.
        """
        reference = copy.deepcopy(self.model)
        optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3, foreach=False)
        batches = [{"x": torch.randn(2, length, 4), "y": torch.randn(2, length, 3),
                    "weight": torch.tensor(weight)}
                   for length, weight in [(5, 1.0), (7, 0.5), (7, 1.25), (9, 0.75)]]
        expected_losses = []
        for inputs in batches:
            expected = _loss(reference, **inputs)
            expected_losses.append(expected.detach().clone())
            expected.backward()
            optimizer.step()
            optimizer.zero_grad()
        trainer = GraphTrainer(
            self.model, _loss, pass_config=PassConfig(fsdp_enabled=False), device=torch.device("cpu"),
            optimizer_config={"lr": 1e-3}, dynamic_arg_dims={"x": [0, 1], "y": [0, 1]},
            compile_sizes=[7], compile_size_input="x", compile_size_dim=1,
        )
        with patch.object(trainer, "compile", wraps=trainer.compile) as capture:
            losses = trainer.train(batches)
        self.assertEqual(capture.call_count, 1)
        for actual, expected in zip(losses, expected_losses):
            torch.testing.assert_close(actual, expected)
        for parameter, expected in zip(self.model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter, expected)
        self.assertEqual(trainer.specialization_stats["compilations"], 1)
        self.assertEqual(trainer.specialization_stats["cache_hits"], 1)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
              card_mark="onecard", essential_mark="essential")
    def test_disabled_policy_keeps_general_graph(self) -> None:
        """
        Feature: Lazy size specialization.
        Description: None and empty size policies retain dynamic execution without a variant cache.
        Expectation: Dynamic shapes work while specialization statistics remain empty.
        """
        for sizes in (None, []):
            with self.subTest(sizes=sizes):
                compiler = self._compiler(compile_sizes=sizes)
                self._step(compiler, 2, 5)
                self._step(compiler, 3, 7)
                self.assertEqual(compiler.specialization_stats, {})
