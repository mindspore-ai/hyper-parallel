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
"""End-to-end symbolic joint graphs, configuration and runtime shape guards."""

import copy
import gc
import unittest
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.compile import GraphCompiler, GraphTrainer, PassConfig
from hyper_parallel.compile.passes.parallel.fsdp_pass import FSDPPass
from hyper_parallel.compile.tracer.dynamic_shapes import normalize_dynamic_arg_dims
from hyper_parallel.compile.tracer.graph_tracer import run_traced_graph, trace_model_graph
from hyper_parallel.models.build_options import CompileConfig
from hyper_parallel.trainer.base import BaseTrainer
from tests.common.mark_utils import arg_mark


def _loss(model, x, y):
    return ((model(x) - y) ** 2).mean()


class TestDynamicShapes(unittest.TestCase):
    """Real forward/backward capture with changing user tensor sizes."""

    def setUp(self) -> None:
        """Create a CPU-only compiler configuration."""
        torch.manual_seed(42)
        self.model = torch.nn.Linear(4, 3)
        self.config = PassConfig(fsdp_enabled=False)

    def _compiler(self, **kwargs):
        return GraphCompiler(self.model, _loss, pass_config=self.config, **kwargs)

    def _assert_step(self, compiler, batch, sequence):
        x = torch.randn(batch, sequence, 4)
        y = torch.randn(batch, sequence, 3)
        self.model.zero_grad()
        actual, _ = compiler.forward_backward(x=x, y=y)
        expected = _loss(self.model, x, y)
        gradients = torch.autograd.grad(expected, tuple(self.model.parameters()))
        torch.testing.assert_close(actual, expected)
        for parameter, gradient in zip(self.model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, gradient)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_batch_and_sequence_reuse_graph(self):
        """
        Feature: Symbolic graph shape contracts.
        Description: One symbolic graph computes loss and gradients for several shapes.
        Expectation: Dynamic dimensions reuse graphs and static dimensions retain guards.
        """
        compiler = self._compiler(dynamic_arg_dims={"x": [0, 1], "y": [0, 1]})
        with patch.object(compiler, "compile", wraps=compiler.compile) as compile_spy:
            for batch, sequence in [(2, 5), (3, 7), (4, 3), (2, 9)]:
                self._assert_step(compiler, batch, sequence)
            self.assertEqual(compile_spy.call_count, 1)
        joint = compiler._joint_graph
        for value in joint.example_inputs[:2]:
            for dim in value.shape:
                self.assertIs(type(dim), int)
        self.assertIsInstance(joint.example_inputs[2].shape[0], torch.SymInt)
        self.assertEqual(joint.example_inputs[2].shape[2], 4)

    def test_automatic_dynamic_mode(self):
        """dynamic=True needs no manual input paths."""
        compiler = self._compiler(dynamic=True)
        self._assert_step(compiler, 2, 5)
        self._assert_step(compiler, 3, 7)

    def test_negative_dims_and_nested_inputs(self):
        """Dictionary/list paths and negative dimension indices are supported."""
        compiler = GraphCompiler(
            self.model, lambda model, batch: _loss(model, batch["items"][0], batch["target"]),
            pass_config=self.config,
            dynamic_arg_dims={"batch.items.0": [-3, -2], "batch.target": [0, 1]},
        )
        for batch, sequence in [(2, 5), (3, 7)]:
            inputs = {"items": [torch.randn(batch, sequence, 4)], "target": torch.randn(batch, sequence, 3)}
            actual, _ = compiler.forward_backward(batch=inputs)
            torch.testing.assert_close(actual, _loss(self.model, inputs["items"][0], inputs["target"]))

    def test_same_sample_sizes_are_independent(self):
        """Unrelated dimensions with equal hints do not acquire equality guards."""
        compiler = GraphCompiler(
            self.model, lambda model, x, z: model(x).mean() + model(z).mean(),
            pass_config=self.config, dynamic_arg_dims={"x": 0, "z": 0},
        )
        compiler.forward_backward(x=torch.randn(3, 4), z=torch.randn(3, 4))
        x, z = torch.randn(5, 4), torch.randn(7, 4)
        actual, _ = compiler.forward_backward(x=x, z=z)
        torch.testing.assert_close(actual, self.model(x).mean() + self.model(z).mean())

    def test_grad_accumulation_across_shapes(self):
        """Microbatches of different sizes still accumulate gradients."""
        compiler = self._compiler(dynamic=True)
        expected_grads = [torch.zeros_like(parameter) for parameter in self.model.parameters()]
        for batch in (2, 5, 3):
            x, y = torch.randn(batch, 4), torch.randn(batch, 3)
            compiler.forward_backward(x=x, y=y)
            grads = torch.autograd.grad(_loss(self.model, x, y), tuple(self.model.parameters()))
            for expected, grad in zip(expected_grads, grads):
                expected.add_(grad)
        for parameter, expected in zip(self.model.parameters(), expected_grads):
            torch.testing.assert_close(parameter.grad, expected)

    def test_optimizer_updates_match_eager(self):
        """GraphTrainer preserves live state across variable-length optimizer steps."""
        reference = copy.deepcopy(self.model)
        optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3, foreach=False)
        trainer = GraphTrainer(self.model, _loss, pass_config=self.config, dynamic=True,
                               optimizer_config={"lr": 1e-3}, device=torch.device("cpu"))
        for batch, sequence in [(2, 5), (3, 7), (4, 3)]:
            x, y = torch.randn(batch, sequence, 4), torch.randn(batch, sequence, 3)
            actual, _ = trainer.train_step(x=x, y=y)
            expected = _loss(reference, x, y)
            expected.backward()
            torch.testing.assert_close(actual, expected)
            trainer.optimizer_step()
            optimizer.step()
            optimizer.zero_grad()
            for parameter, ref in zip(self.model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter, ref)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_python_shape_branch_is_guarded(self):
        """
        Feature: Symbolic graph shape contracts.
        Description: A branch chosen while tracing must not silently run for incompatible sizes.
        Expectation: Dynamic dimensions reuse graphs and static dimensions retain guards.
        """
        def train(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            """Evaluate the parameterized loss for the dynamic input."""
            loss = model(x).sum()
            return loss * 2 if x.shape[0] > 3 else loss * 3

        compiler = GraphCompiler(self.model, train, pass_config=self.config, dynamic_arg_dims={"x": 0})
        compiler.forward_backward(x=torch.randn(5, 4))
        x = torch.randn(7, 4)
        actual, _ = compiler.forward_backward(x=x)
        torch.testing.assert_close(actual, train(self.model, x))
        before = self.model.weight.grad.clone()
        with self.assertRaisesRegex(ValueError, "shape guards"):
            compiler.forward_backward(x=torch.randn(2, 4))
        torch.testing.assert_close(self.model.weight.grad, before)

    def test_unmarked_dimension_is_static(self):
        """Explicit mappings take precedence over automatic dimension selection."""
        compiler = self._compiler(dynamic=True, dynamic_arg_dims={"x": 0, "y": 0})
        self._assert_step(compiler, 2, 5)
        self._assert_step(compiler, 3, 5)
        with self.assertRaisesRegex(ValueError, "shape guards"):
            self._assert_step(compiler, 3, 7)

    def test_related_sizes_are_guarded(self):
        """Broadcast-incompatible label sizes fail before graph execution."""
        compiler = self._compiler(dynamic=True)
        self._assert_step(compiler, 2, 5)
        with self.assertRaisesRegex(ValueError, "shape guards"):
            compiler.forward_backward(x=torch.randn(3, 7, 4), y=torch.randn(2, 7, 3))

    def test_zero_one_specialization_is_guarded(self):
        """Torch's size-one specialization is reported instead of silently reused."""
        compiler = self._compiler(dynamic=True)
        self._assert_step(compiler, 1, 5)
        with self.assertRaisesRegex(ValueError, "sizes 0/1"):
            self._assert_step(compiler, 2, 5)

    def test_metadata_and_constants_are_guarded(self):
        """Dtype, rank and captured Python constants cannot change underneath a graph."""
        compiler = GraphCompiler(self.model, lambda model, x, scale: model(x).mean() * scale,
                                 pass_config=self.config, dynamic=True)
        compiler.forward_backward(x=torch.randn(2, 4), scale=2)
        for inputs in [
            {"x": torch.randn(3, 4), "scale": 3},
            {"x": torch.randn(3, 4, dtype=torch.float64), "scale": 2},
            {"x": torch.randn(3, 2, 4), "scale": 2},
        ]:
            with self.subTest(inputs=inputs):
                with self.assertRaisesRegex(ValueError, "rank, dtype"):
                    compiler.forward_backward(**inputs)

    def test_stride_is_guarded(self):
        """Changing layout cannot invalidate traced view/reshape assumptions."""
        compiler = self._compiler(dynamic=True)
        compiler.forward_backward(x=torch.randn(2, 4), y=torch.randn(2, 3))
        with self.assertRaisesRegex(ValueError, "shape guards"):
            compiler.forward_backward(x=torch.randn(4, 3).t(), y=torch.randn(3, 3))

    def test_alias_marks_merge_and_alias_changes_fail(self):
        """Multiple paths to one tensor merge dimensions and retain identity guards."""
        compiler = GraphCompiler(self.model, lambda model, x, z: (model(x) + model(z)).mean(),
                                 pass_config=self.config, dynamic_arg_dims={"x": 0, "z": 1})
        for shape in [(2, 5, 4), (3, 7, 4)]:
            value = torch.randn(*shape)
            actual, _ = compiler.forward_backward(x=value, z=value)
            torch.testing.assert_close(actual, (2 * self.model(value)).mean())
        with self.assertRaisesRegex(ValueError, "aliasing"):
            compiler.forward_backward(x=value, z=value.clone())

    def test_sample_storage_is_not_retained(self):
        """Runtime guards retain only metadata and fake tensors."""
        compiler = self._compiler(dynamic=True)
        x, y = torch.randn(2, 4), torch.randn(2, 3)
        references = weakref.ref(x), weakref.ref(y)
        compiler.compile(x=x, y=y)
        del x, y
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_static_default_unchanged(self):
        """
        Feature: Symbolic graph shape contracts.
        Description: Opting out retains concrete tensor metadata and no symbolic guards.
        Expectation: Dynamic dimensions reuse graphs and static dimensions retain guards.
        """
        compiler = self._compiler()
        self._assert_step(compiler, 2, 5)
        self.assertIsNone(compiler._joint_graph.input_guards)
        for dim in compiler._joint_graph.example_inputs[2].shape:
            self.assertIs(type(dim), int)

    def test_invalid_mapping(self):
        """Bad paths, non-tensors and invalid dimensions report user-facing errors."""
        for mapping in [{"missing": 0}, {"x": 3}, {"x": -4}, {"scale": 0}]:
            with self.subTest(mapping=mapping):
                compiler = GraphCompiler(self.model, lambda model, x, scale: model(x).mean() * scale,
                                         pass_config=self.config, dynamic_arg_dims=mapping)
                with self.assertRaisesRegex(ValueError, "dynamic_arg_dims"):
                    compiler.compile(x=torch.randn(2, 4), scale=2)
        for mapping in [[], {"": 0}, {"x..a": 0}, {"x": True}, {"x": [1.5]}, {"x": "0"}]:
            with self.subTest(mapping=mapping):
                with self.assertRaisesRegex(ValueError, "dynamic_arg_dims"):
                    normalize_dynamic_arg_dims(mapping)

    def test_pp_rejected_before_tracing(self):
        """Unsupported pipeline mode fails without mutating model state."""
        compiler = GraphCompiler(self.model, _loss, pass_config=PassConfig(pp_enabled=True), dynamic=True)
        with patch("hyper_parallel.compile.compiler.trace_model_graph") as tracer:
            with self.assertRaisesRegex(ValueError, "pipeline parallel"):
                compiler.compile(x=torch.randn(2, 4), y=torch.randn(2, 3))
            tracer.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_buffer_shapes_stay_static(self):
        """
        Feature: Symbolic graph shape contracts.
        Description: Model buffers keep fixed dimensions even when all user dims are symbolic.
        Expectation: Dynamic dimensions reuse graphs and static dimensions retain guards.
        """
        model = torch.nn.Sequential(torch.nn.Linear(4, 3))
        model.register_buffer("offset", torch.ones(3))
        joint = trace_model_graph(model, lambda m, x: (m(x) + m.offset).mean(),
                                  {"x": torch.randn(2, 4)}, dynamic=True)
        for tensor in joint.example_inputs[:joint.graph_module.num_state_inputs]:
            for dim in tensor.shape:
                self.assertIs(type(dim), int)
        x = torch.randn(5, 4)
        actual, _, _ = run_traced_graph(joint, model, {"x": x})
        torch.testing.assert_close(actual, (model(x) + model.offset).mean())

    def test_fsdp_pass_preserves_symbolic_inputs(self):
        """Parameter sharding leaves dynamic user metadata and guards intact."""
        model = torch.nn.Linear(4, 4)
        joint = trace_model_graph(model, lambda m, x: m(x).square().mean(),
                                  {"x": torch.randn(3, 4)}, dynamic_arg_dims={"x": 0})
        with patch("hyper_parallel.compile.passes.parallel.fsdp_pass.dist") as distributed, \
                patch("hyper_parallel.compile.passes.parallel.fsdp_pass._resolve_process_group"):
            distributed.is_initialized.return_value = True
            distributed.get_world_size.return_value = 2
            distributed.get_rank.return_value = 0
            FSDPPass().run(joint.graph_module, PassConfig(fsdp_degree=2), model=model, fsdp_group_name="fsdp")
        joint.graph_module.graph.lint()
        self.assertEqual(tuple(model.weight.shape), (2, 4))
        joint.input_guards.refresh()
        joint.input_guards.validate([torch.randn(7, 4)])
        self.assertTrue(any("all_gather" in str(node.target) for node in joint.graph_module.graph.nodes))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_basetrainer_inherits_dynamic_config(self):
        """
        Feature: Dynamic joint-graph training.
        Description: Existing BaseTrainer construction and step paths pass the new configuration through.
        Expectation: Losses, gradients and public contracts remain correct.
        """
        class KeywordModel(torch.nn.Module):
            def __init__(self) -> None:
                """Create a keyword-compatible linear model."""
                super().__init__()
                self.linear = torch.nn.Linear(4, 3)

            def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
                """Apply the model to a variable-shape input."""
                return self.linear(input_ids)

        base = BaseTrainer.__new__(BaseTrainer)
        base.model_integration = Mock(spec=["record_batch"])
        base.config = SimpleNamespace(
            compile=CompileConfig(enabled=True, use_joint_graph=True,
                                  dynamic_arg_dims={"model_inputs.input_ids": [0, 1], "labels": [0, 1]}),
            accelerator=SimpleNamespace(tp_size=1, sequence_parallel=False, loss_parallel=False),
            fsdp_config=SimpleNamespace(dp_shard_size=1, edp_shard_size=1),
            training=SimpleNamespace(empty_cache_before_backward=False),
        )
        base.model, base.device, base.mesh, base.state = KeywordModel(), torch.device("cpu"), None, None
        base.model_fwd_context = nullcontext()

        def postforward(outputs: torch.Tensor, labels: torch.Tensor) -> tuple:
            """Build the scalar loss and named diagnostics."""
            loss = ((outputs - labels) ** 2).mean()
            return loss, {"main": loss.detach()}

        base.postforward = postforward
        base._build_graph_compiler()
        with patch.object(base.graph_compiler, "compile", wraps=base.graph_compiler.compile) as compile_spy:
            for batch, sequence in [(2, 5), (3, 7)]:
                x, y = torch.randn(batch, sequence, 4), torch.randn(batch, sequence, 3)
                base.model.zero_grad()
                actual, losses = base.forward_backward_step({"input_ids": x}, loss_inputs={"labels": y})
                expected, _ = postforward(base.model(x), y)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(losses["main"], expected.detach())
                gradients = torch.autograd.grad(expected, tuple(base.model.parameters()))
                for parameter, gradient in zip(base.model.parameters(), gradients):
                    torch.testing.assert_close(parameter.grad, gradient)
            self.assertEqual(compile_spy.call_count, 1)


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_basetrainer_token_weights_are_runtime_inputs(self):
        """
        Feature: Dynamic joint-graph training.
        Description: Changing token counts updates the loss and gradients without retracing.
        Expectation: Losses, gradients and public contracts remain correct.
        """
        class KeywordModel(torch.nn.Linear):
            def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
                """Apply the model to a variable-shape input."""
                return super().forward(input_ids)

        base = BaseTrainer.__new__(BaseTrainer)
        base.model_integration = Mock(spec=["record_batch"])
        base.config = SimpleNamespace(training=SimpleNamespace(empty_cache_before_backward=False))
        base.model = KeywordModel(4, 3)
        base.device, base.state, base.mesh = torch.device("cpu"), None, None
        base.model_fwd_context = nullcontext()
        base.loss_fn = lambda model_output, labels: (model_output - labels).square().mean()
        base.graph_compiler = GraphCompiler(base.model, base._graph_train_fn,
                                            pass_config=self.config, dynamic=True)

        def aggregate(loss: torch.Tensor, current: dict, step: dict, device_mesh: object) -> dict:
            """Weight the current loss by live token counts."""
            return {"foundation_loss": loss * current["foundation_tokens"] / step["foundation_tokens"]}

        with patch("hyper_parallel.trainer.base.mean_global_loss", side_effect=aggregate):
            with patch.object(base.graph_compiler, "compile", wraps=base.graph_compiler.compile) as compile_spy:
                for batch, count in [(2, 2), (2, 6), (3, 3)]:
                    x, y = torch.randn(batch, 4), torch.randn(batch, 3)
                    base.current_token_counts = {"foundation_tokens": torch.tensor(count)}
                    base.step_token_counts = {"foundation_tokens": torch.tensor(8)}
                    base.model.zero_grad()
                    actual, losses = base.forward_backward_step({"input_ids": x}, loss_inputs={"labels": y})
                    expected = (base.model(x) - y).square().mean() * count / 8
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(losses["foundation_loss"], expected)
                    gradients = torch.autograd.grad(expected, tuple(base.model.parameters()))
                    for parameter, gradient in zip(base.model.parameters(), gradients):
                        torch.testing.assert_close(parameter.grad, gradient)
                self.assertEqual(compile_spy.call_count, 1)


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_causal_label_padding_preserves_dynamic_length(self):
        """
        Feature: Symbolic graph shape contracts.
        Description: The standard shifted-label CE path must not specialize token length.
        Expectation: Dynamic dimensions reuse graphs and static dimensions retain guards.
        """
        model = torch.nn.Sequential(torch.nn.Embedding(32, 8), torch.nn.Linear(8, 32))

        def causal_loss(module: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Compute cross entropy with dynamically shifted labels."""
            shifted = torch.nn.functional.pad(y, (0, 1), value=-100)[..., 1:].contiguous()
            return torch.nn.functional.cross_entropy(module(x).flatten(0, 1), shifted.flatten())

        compiler = GraphCompiler(model, causal_loss, pass_config=self.config, dynamic=True)
        for length in (5, 7, 11):
            x, y = (torch.randint(0, 32, (1, length)) for _ in range(2))
            model.zero_grad()
            actual, _ = compiler.forward_backward(x=x, y=y)
            expected = causal_loss(model, x, y)
            torch.testing.assert_close(actual, expected)
            gradients = torch.autograd.grad(expected, tuple(model.parameters()))
            for parameter, gradient in zip(model.parameters(), gradients):
                torch.testing.assert_close(parameter.grad, gradient)
        self.assertFalse(any(node.target == torch.ops.aten.constant_pad_nd.default
                             for node in compiler._joint_graph.graph_module.graph.nodes))
