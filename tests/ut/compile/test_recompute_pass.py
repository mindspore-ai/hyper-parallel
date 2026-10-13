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
"""Numerical and graph-safety tests for joint-graph activation recomputation."""

import copy
import operator
import unittest

import torch
from torch import fx, nn

from hyper_parallel.compile import GraphCompiler, GraphParallelPlan, PassConfig, RecomputeConfig
from hyper_parallel.compile.tracer.graph_tracer import trace_model_graph
from hyper_parallel.compile.passes.pipeline import PassPipeline
from hyper_parallel.compile.passes.recompute import RecomputePass
from hyper_parallel.core.activation_memory.policy import CheckpointPolicy
from tests.ut.compile.test_fsdp_reshard import _patch_dist, _run_simulated


class _Block(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        """Build the residual test module."""
        super().__init__()
        self.up = nn.Linear(8, 16, bias=False)
        self.down = nn.Linear(16, 8, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Execute the residual test module."""
        hidden = self.dropout(torch.sigmoid(self.up(value)))
        return value + self.down(hidden).tanh()


class _Model(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        """Build the residual test module."""
        super().__init__()
        self.layers = nn.ModuleList([_Block(dropout), _Block(dropout)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Execute the residual test module."""
        for layer in self.layers:
            value = layer(value)
        return value


def _loss(model, *, value, label):
    return (model(value) - label).square().mean()


def _compiler(model, config):
    return GraphCompiler(model, _loss, PassConfig(
        fsdp_enabled=False, enable_overlap=False, recompute=config), device=torch.device("cpu"))


class TestRecomputePass(unittest.TestCase):
    """Exercise actual GraphCompiler training, not just policy annotations."""

    def _compare_training(self, config, dropout=0.0):
        torch.manual_seed(17)
        original = _Model(dropout)
        model = copy.deepcopy(original)
        compiler = _compiler(model, config)
        value, label = torch.randn(4, 8), torch.randn(4, 8)
        compiler.compile(value=value, label=label)
        eager_optimizer = torch.optim.SGD(original.parameters(), lr=0.01)
        graph_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for step in range(3):
            eager_optimizer.zero_grad(set_to_none=True)
            graph_optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(100 + step)
            expected = _loss(original, value=value, label=label)
            expected.backward()
            expected_rng = torch.get_rng_state()
            torch.manual_seed(100 + step)
            actual, loss_dict = compiler.forward_backward(value=value, label=label)
            self.assertEqual(loss_dict, {})
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            for reference, parameter in zip(original.parameters(), model.parameters()):
                torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)
            eager_optimizer.step()
            graph_optimizer.step()
            for reference, parameter in zip(original.parameters(), model.parameters()):
                torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
        return compiler

    def test_auxiliary_loss_outputs(self):
        """Recompute preserves AutoModel's (loss, loss_dict) output contract."""
        torch.manual_seed(17)
        for mode in ("full", "selective"):
            with self.subTest(mode=mode):
                model = _Model()
                reference = copy.deepcopy(model)

                def train_fn(module, *, value, label):
                    prediction = module(value)
                    return (prediction - label).square().mean(), {"prediction_mean": prediction.mean()}

                compiler = GraphCompiler(model, train_fn, PassConfig(
                    fsdp_enabled=False, enable_overlap=False, recompute=RecomputeConfig(mode=mode)),
                    device=torch.device("cpu"))
                batch = {"value": torch.randn(4, 8), "label": torch.randn(4, 8)}
                expected, expected_dict = train_fn(reference, **batch)
                expected.backward()
                actual, actual_dict = compiler.forward_backward(**batch)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual_dict, expected_dict, rtol=0, atol=0)
                for parameter, original in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(parameter.grad, original.grad, rtol=0, atol=0)

    def test_full_and_selective_training(self):
        """Full and both SAC policies match eager loss, grads and SGD updates."""
        for mode, policy in (("full", "eager"), ("selective", "eager"), ("selective", "save_compute")):
            with self.subTest(mode=mode, policy=policy):
                compiler = self._compare_training(RecomputeConfig(mode=mode, selective_policy=policy))
                self.assertGreater(compiler.recompute_report["recomputed_nodes"], 0)
                report = compiler.recompute_report
                if policy == "eager":
                    self.assertLess(
                        report["retained_activation_bytes_after"], report["retained_activation_bytes_before"])
                else:
                    self.assertLessEqual(
                        report["retained_activation_bytes_after"], report["retained_activation_bytes_before"])

    def test_dropout_preserves_rng_and_gradients(self):
        """Dropout's random and inplace mask chain is never replayed."""
        for mode in ("full", "selective"):
            with self.subTest(mode=mode):
                compiler = self._compare_training(RecomputeConfig(mode=mode), dropout=0.25)
                self.assertIn("mutation_or_alias", compiler.recompute_report["save_reasons"])

    def test_excluded_module_never_replayed(self):
        """An excluded subtree can coexist with recomputation of another layer."""
        compiler = self._compare_training(RecomputeConfig(
            mode="full", exclude_module_patterns=("layers.0",)))
        graph = compiler._joint_graph.graph_module.graph
        copies = [node for node in graph.nodes if "recompute_source" in node.meta]
        self.assertTrue(copies)
        for node in copies:
            paths = [entry[0] for entry in node.meta["nn_module_stack"].values()]
            self.assertNotIn("layers.0", paths)

    def test_custom_policy_and_save_ops(self):
        """Accept native policies and explicit saves without changing gradients."""
        config = RecomputeConfig(mode="full", save_ops=("aten.mm.default",))
        compiler = self._compare_training(config)
        copies = [node for node in compiler._joint_graph.graph_module.graph.nodes
                  if "recompute_source" in node.meta]
        self.assertFalse(any(node.target is torch.ops.aten.mm.default for node in copies))
        config = RecomputeConfig(mode="selective", policy_fn=lambda node:
                                 torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE)
        self._compare_training(config)

    def test_gradient_accumulation(self):
        """Repeated calls before optimizer.step preserve accumulating gradients."""
        torch.manual_seed(11)
        model = _Model()
        reference = copy.deepcopy(model)
        compiler = _compiler(model, RecomputeConfig(mode="full"))
        value, label = torch.randn(2, 8), torch.randn(2, 8)
        for _ in range(2):
            _loss(reference, value=value, label=label).backward()
            compiler.forward_backward(value=value, label=label)
        for expected, actual in zip(reference.parameters(), model.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)

    def test_configuration_errors(self):
        """Bad modes, unmatched selectors and unsupported swap fail explicitly."""
        with self.assertRaisesRegex(ValueError, "recompute.mode"):
            PassConfig(recompute=RecomputeConfig(mode="bad"))
        with self.assertRaisesRegex(ValueError, "pipeline parallelism"):
            PassConfig(pp_enabled=True, recompute=RecomputeConfig(mode="full"))
        configs = (
            (RecomputeConfig(mode="full", module_patterns=("absent.*",)), "matched no traced"),
            (RecomputeConfig(mode="full", save_ops=("aten.nonexistent.default",)), "matched no selected"),
            (RecomputeConfig(mode="full", policy_fn=lambda node: CheckpointPolicy.MUST_SWAP), "swap"),
            (RecomputeConfig(mode="full", policy_fn=lambda node: CheckpointPolicy.MUST_RECOMPUTE), "Cannot"),
        )
        for config, message in configs:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _compiler(_Model(0.2), config).compile(value=torch.randn(2, 8), label=torch.randn(2, 8))

    def test_pipeline_order_and_default_noop(self):
        """FSDP sees newly added backward users before it plans parameter frees."""
        config = PassConfig(recompute=RecomputeConfig(mode="full"))
        names = [type(item).__name__ for item in PassPipeline.from_config(config).passes]
        self.assertLess(names.index("RecomputePass"), names.index("FSDPPass"))
        self.assertNotIn("recompute", [item.name for item in PassPipeline.from_config(PassConfig()).passes])

    def test_shared_dependency_and_tuple_output(self):
        """Shared producers are cloned once and getitem inputs remain topological."""
        graph = fx.Graph()
        value = graph.placeholder("value")
        first = graph.call_function(torch.ops.aten.sin.default, (value,))
        pair = graph.call_function(torch.ops.aten.max.dim, (first, 1))
        part = graph.call_function(operator.getitem, (pair, 0))
        forward = graph.call_function(torch.ops.aten.sum.default, (part,))
        backward = graph.call_function(torch.ops.aten.add.Tensor, (part, part))
        backward.meta["autograd_backward"] = True
        for node in (first, pair, part, forward):
            node.meta["nn_module_stack"] = {"layer": ("layers.0", "Block")}
        graph.output((forward, backward))
        module = fx.GraphModule({}, graph)
        value = torch.randn(3, 4)
        expected = module(value)
        RecomputePass().run(module, PassConfig(recompute=RecomputeConfig(mode="full")))
        torch.testing.assert_close(module(value), expected, rtol=0, atol=0)
        sources = [node.meta["recompute_source"] for node in graph.nodes if "recompute_source" in node.meta]
        self.assertEqual(len(sources), len(set(sources)))
        self.assertIn(first.name, sources)
        module.graph.lint()

    def test_inplace_alias_read_is_saved(self):
        """A read preceding a later alias write cannot be moved into backward."""
        graph = fx.Graph()
        value = graph.placeholder("value")
        copied = graph.call_function(torch.ops.aten.clone.default, (value,))
        view = graph.call_function(torch.ops.aten.view.default, (copied, [-1]))
        read = graph.call_function(torch.ops.aten.sin.default, (view,))
        write = graph.call_function(torch.ops.aten.add_.Tensor, (copied, 1))
        backward = graph.call_function(torch.ops.aten.cos.default, (read,))
        backward.meta["autograd_backward"] = True
        for node in (copied, view, read, write):
            node.meta["nn_module_stack"] = {"layer": ("layers.0", "Block")}
        graph.output((write, backward))
        module = fx.GraphModule({}, graph)
        data = torch.randn(4)
        expected = module(data)
        RecomputePass().run(module, PassConfig(recompute=RecomputeConfig(mode="full")))
        torch.testing.assert_close(module(data), expected, rtol=0, atol=0)
        self.assertEqual(read.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_repeat_pass_rejected(self):
        """Rerunning a pass requires a fresh graph rather than duplicating work."""
        compiler = _compiler(_Model(), RecomputeConfig(mode="full"))
        compiler.compile(value=torch.randn(2, 8), label=torch.randn(2, 8))
        with self.assertRaisesRegex(ValueError, "already run"):
            RecomputePass().run(compiler._joint_graph.graph_module, compiler.pass_config)


    def test_fsdp_reshard_after_recompute(self):
        """Execute real storage frees with simulated two-rank communication."""
        for mode in ("full", "selective"):
            for reshard in (False, True):
                with self.subTest(mode=mode, reshard=reshard):
                    torch.manual_seed(9)
                    reference = _Model().double()
                    live = copy.deepcopy(reference)
                    value, label = torch.randn(4, 8).double(), torch.randn(4, 8).double()
                    graph = trace_model_graph(live, _loss, {"value": value, "label": label}).graph_module
                    config = PassConfig(fsdp_degree=2, fsdp_reshard_after_forward=reshard,
                                        enable_overlap=False, recompute=RecomputeConfig(mode=mode))
                    plan = GraphParallelPlan().fsdp_mark_pattern("*")
                    with _patch_dist():
                        PassPipeline.from_config(config, plan).run(graph, model=live, fsdp_group_name="fsdp")
                    with torch.no_grad():
                        actual_loss, grads = _run_simulated(graph, live, reference, value, label)
                    expected_loss = _loss(reference, value=value, label=label)
                    expected_grads = torch.autograd.grad(expected_loss, list(reference.parameters()))
                    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
                    for actual, expected in zip(grads, expected_grads):
                        torch.testing.assert_close(actual, expected.chunk(2, dim=0)[0], rtol=0, atol=0)


    def test_collective_and_wait_are_not_recomputed(self):
        """Communication nodes stay in forward even under full recomputation."""
        graph = fx.Graph()
        value = graph.placeholder("value")
        launch = graph.call_function(torch.ops._c10d_functional.all_reduce.default,
                                     (value, "sum", "test_group"))
        wait = graph.call_function(torch.ops._c10d_functional.wait_tensor.default, (launch,))
        activation = graph.call_function(torch.ops.aten.sin.default, (wait,))
        loss = graph.call_function(torch.ops.aten.sum.default, (activation,))
        backward = graph.call_function(torch.ops.aten.cos.default, (activation,))
        backward.meta["autograd_backward"] = True
        for node in (launch, wait, activation, loss):
            node.meta["nn_module_stack"] = {"layer": ("layers.0", "Block")}
        graph.output((loss, backward))
        module = fx.GraphModule({}, graph)
        RecomputePass().run(module, PassConfig(recompute=RecomputeConfig(mode="full")))
        self.assertEqual(launch.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(wait.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        copies = [node.meta["recompute_source"] for node in graph.nodes if "recompute_source" in node.meta]
        self.assertEqual(copies, [activation.name])
