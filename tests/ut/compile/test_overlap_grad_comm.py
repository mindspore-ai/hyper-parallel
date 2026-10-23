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
"""Backward gradient overlap: readiness, effects, alias DAGs and numerics."""

import operator
import unittest
from types import SimpleNamespace

import torch
from torch import fx
from torch.ops import _c10d_functional

from hyper_parallel.compile.pass_config import PassConfig, build_pass_config_from_trainer_config
from hyper_parallel.compile.passes.overlap.overlap_grad_comm import overlap_grad_comm
from hyper_parallel.compile.passes.overlap.schedule import FsdpOverlapPass
from hyper_parallel.compile.passes.parallel.fsdp_pass import _free_tensor_storage
from hyper_parallel.models.build_options import CompileConfig
from tests.ut.compile.test_fsdp_reshard import _ChunkedMLP, _MLP, _run_simulated, _trace_and_run


def _positions(module):
    return {node: index for index, node in enumerate(module.graph.nodes)}


def _call(graph, target, args, *, backward=True):
    node = graph.call_function(target, args)
    node.meta["autograd_backward"] = backward
    return node


def _reduce(graph, grad, *, default=False, tagged=True, backward=True, group="fsdp"):
    packet = _c10d_functional.reduce_scatter_tensor
    launch = _call(graph, packet.default if default else packet, (grad, "avg", 2, group), backward=backward)
    if tagged:
        launch.meta.update(comm_type="fsdp_reduce_scatter", comm_group=group)
    wait = _call(graph, _c10d_functional.wait_tensor.default, (launch,), backward=backward)
    return launch, wait


def _graph(*, reverse=False, tagged=True, backward=True):
    """Build two matrix gradients followed by configurable tail reductions."""
    graph = fx.Graph()
    value = graph.placeholder("value")
    weight = graph.placeholder("weight")
    grad0 = _call(graph, torch.ops.aten.mm.default, (value, weight), backward=backward)
    grad1 = _call(graph, torch.ops.aten.mm.default, (grad0, weight), backward=backward)
    ordered = (grad1, grad0) if reverse else (grad0, grad1)
    reductions = [_reduce(graph, grad, default=index == 1, tagged=tagged, backward=backward)
                  for index, grad in enumerate(ordered)]
    graph.output(tuple(wait for _, wait in reductions))
    return fx.GraphModule({}, graph), (grad0, grad1), reductions


def _opaque(tensor):
    return tensor


class TestGradientOverlap(unittest.TestCase):
    """Only owned FSDP reductions move, with complete dependency/effect bounds."""

    def assert_order(self, module: fx.GraphModule, *nodes: fx.Node) -> None:
        """Assert graph order with useful node names and positions on failure."""
        order = _positions(module)
        for left, right in zip(nodes, nodes[1:]):
            self.assertLess(order[left], order[right],
                            f"expected {left.name}@{order[left]} before {right.name}@{order[right]}")

    def test_advance_tail_launch_and_delay_wait_to_output(self):
        """A decomposed mm gradient starts communicating before later backward mm."""
        module, (grad0, grad1), ((launch0, wait0), (launch1, wait1)) = _graph()
        before = set(module.graph.nodes)
        output = next(node for node in module.graph.nodes if node.op == "output")
        returned = output.args
        result = FsdpOverlapPass().run(module, PassConfig())
        self.assertIs(result, module, f"expected same module {module}, got {result}")
        self.assert_order(module, grad0, launch0, grad1, launch1, wait0, wait1, output)
        self.assertEqual(set(module.graph.nodes), before, f"expected nodes {before}, got {set(module.graph.nodes)}")
        self.assertEqual(output.args, returned, f"expected outputs {returned}, got {output.args}")
        stats = module.meta["fsdp_grad_overlap"]
        self.assertEqual(stats["matched"], 2, f"expected 2 reductions, got {stats}")
        self.assertGreater(stats["launches_advanced"], 0, f"expected advanced launches, got {stats}")
        module.graph.lint()

    def test_reverse_output_order_uses_gradient_readiness_and_is_idempotent(self):
        """Readiness can change FSDP launch order without changing output order."""
        module, (grad0, grad1), ((launch1, _), (launch0, _)) = _graph(reverse=True)
        overlap_grad_comm(module, PassConfig())
        self.assert_order(module, grad0, launch0, grad1, launch1)
        before = module.code
        overlap_grad_comm(module, PassConfig())
        self.assertEqual(module.code, before, f"expected stable code {before}, got {module.code}")

    def test_gradient_overlap_runs_without_prefetch_or_legacy_overlap(self):
        """The RS-only path advances communication before later backward compute."""
        module, (grad0, grad1), ((launch, _), _) = _graph()
        FsdpOverlapPass().run(module, PassConfig(
            enable_overlap=False, fsdp_allgather_prefetch=False, fsdp_overlap_grad_reduce=True))
        self.assert_order(module, grad0, launch, grad1)
        self.assertNotIn("fsdp_prefetch", module.meta,
                         f"expected no parameter scheduling statistics, got {module.meta}")

    def test_switches_and_unowned_or_forward_reductions_are_noop(self):
        """Independent toggle permits a prefetch-only baseline; TP remains owned by TP."""
        cases = ({"fsdp_overlap_grad_reduce": False},)
        for options in cases:
            with self.subTest(options=options):
                module, _, _ = _graph()
                before = module.code
                overlap_grad_comm(module, PassConfig(**options))
                self.assertEqual(module.code, before, f"expected unchanged {before}, got {module.code}")
        for options in ({"tagged": False}, {"backward": False}):
            with self.subTest(options=options):
                module, _, _ = _graph(**options)
                before = module.code
                overlap_grad_comm(module, PassConfig())
                self.assertEqual(module.code, before, f"expected unchanged {before}, got {module.code}")

    def test_launch_never_crosses_input_mutation_or_unknown_effect(self):
        """An implicit write to a gradient must precede its async read."""
        for target in (torch.ops.aten.add_.Tensor, _opaque):
            with self.subTest(target=target):
                module, (grad0, grad1), ((launch, _), _) = _graph()
                with module.graph.inserting_before(grad1):
                    args = (grad0, 1) if target is torch.ops.aten.add_.Tensor else (grad0,)
                    effect = _call(module.graph, target, args)
                overlap_grad_comm(module, PassConfig())
                self.assert_order(module, grad0, effect, launch, grad1)

    def test_launch_stays_in_backward_phase(self):
        """A forward-produced constant gradient cannot launch ahead of the backward."""
        module, (grad0, grad1), ((launch, _), _) = _graph()
        grad0.meta["autograd_backward"] = False
        overlap_grad_comm(module, PassConfig())
        self.assert_order(module, grad0, launch, grad1)

    def test_unrelated_parameter_release_does_not_pin_reduction_at_tail(self):
        """Freeing parameter storage does not prevent independent gradient overlap."""
        module, (grad0, grad1), ((launch, _), _) = _graph()
        parameter = next(node for node in module.graph.nodes if node.name == "weight")
        with module.graph.inserting_before(grad1):
            release = _call(module.graph, _free_tensor_storage, (parameter,))
            release.meta["comm_type"] = "fsdp_reshard_free_backward"
        overlap_grad_comm(module, PassConfig())
        self.assert_order(module, grad0, launch, release, grad1)

    def test_wait_remains_before_release_of_async_input_storage(self):
        """A release through an input alias must wait for the async read to finish."""
        graph = fx.Graph()
        value = graph.placeholder("value")
        grad = _call(graph, torch.ops.aten.relu.default, (value,))
        launch, wait = _reduce(graph, grad)
        alias = _call(graph, torch.ops.aten.view.default, (grad, [-1]))
        filler = _call(graph, torch.ops.aten.relu.default, (value,))
        release = _call(graph, _free_tensor_storage, (alias,))
        release.meta["comm_type"] = "fsdp_reshard_free_backward"
        trailing = _call(graph, torch.ops.aten.relu.default, (filler,))
        graph.output((wait, trailing))
        module = fx.GraphModule({}, graph)
        overlap_grad_comm(module, PassConfig())
        self.assert_order(module, grad, launch, alias, filler, wait, release, trailing)

    def test_branched_wait_views_respect_earliest_consumer(self):
        """Split/getitem aliases are sunk per branch, without crossing early reads."""
        graph = fx.Graph()
        value = graph.placeholder("value")
        launch, wait = _reduce(graph, value)
        split = _call(graph, torch.ops.aten.split.Tensor, (wait, 2))
        first = _call(graph, operator.getitem, (split, 0))
        second = _call(graph, operator.getitem, (split, 1))
        filler0 = _call(graph, torch.ops.aten.relu.default, (value,))
        early = _call(graph, torch.ops.aten.relu.default, (first,))
        filler1 = _call(graph, torch.ops.aten.relu.default, (filler0,))
        late = _call(graph, torch.ops.aten.relu.default, (second,))
        graph.output((early, late, filler1))
        module = fx.GraphModule({}, graph)
        overlap_grad_comm(module, PassConfig())
        self.assert_order(module, launch, filler0, wait, split, first, early, filler1, second, late)
        module.graph.lint()

    def test_wait_does_not_cross_mutation_or_unknown_effect(self):
        """Wait synchronization remains before an intervening effect boundary."""
        for target in (torch.ops.aten.add_.Tensor, _opaque):
            with self.subTest(target=target):
                graph = fx.Graph()
                value = graph.placeholder("value")
                launch, wait = _reduce(graph, value)
                filler = _call(graph, torch.ops.aten.relu.default, (value,))
                args = (value, 1) if target is torch.ops.aten.add_.Tensor else (value,)
                effect = _call(graph, target, args)
                consumer = _call(graph, torch.ops.aten.relu.default, (wait,))
                graph.output((consumer, filler, effect))
                module = fx.GraphModule({}, graph)
                overlap_grad_comm(module, PassConfig())
                self.assert_order(module, launch, filler, wait, effect, consumer)

    def test_tp_launch_order_and_hsdp_wait_dependency_are_preserved(self):
        """FSDP RS may cross other groups, while TP order and RS-wait-AR hold."""
        graph = fx.Graph()
        value = graph.placeholder("value")
        grad = _call(graph, torch.ops.aten.relu.default, (value,))
        tp0 = _call(graph, _c10d_functional.all_reduce.default, (value, "sum", "tp"))
        tp1 = _call(graph, _c10d_functional.all_reduce.default, (value, "sum", "tp"))
        filler = _call(graph, torch.ops.aten.relu.default, (grad,))
        launch, wait = _reduce(graph, grad)
        ar = _call(graph, _c10d_functional.all_reduce.default, (wait, "avg", "replicate"))
        ar_wait = _call(graph, _c10d_functional.wait_tensor.default, (ar,))
        graph.output((ar_wait, filler, tp0, tp1))
        module = fx.GraphModule({}, graph)
        overlap_grad_comm(module, PassConfig())
        self.assert_order(module, grad, launch, tp0, tp1, filler, wait, ar, ar_wait)

    def test_unowned_same_group_collective_is_launch_boundary(self):
        """An unrecognized same-group launch keeps its ordering relative to RS."""
        module, (grad0, grad1), ((launch, _), _) = _graph()
        with module.graph.inserting_before(grad1):
            unowned = _call(module.graph, _c10d_functional.all_reduce.default, (grad0, "sum", "fsdp"))
        overlap_grad_comm(module, PassConfig())
        self.assert_order(module, grad0, unowned, launch, grad1)

    def test_reduction_with_multiple_direct_users_is_not_scheduled(self):
        """A noncanonical launch/wait pattern is left intact for its owner."""
        module, _, ((launch, _), _) = _graph()
        output = next(node for node in module.graph.nodes if node.op == "output")
        output.args = (output.args[0] + (launch,),)
        overlap_grad_comm(module, PassConfig())
        self.assertNotIn("fsdp_grad_overlap_anchor", launch.meta,
                         f"expected no move for multi-user launch, got {launch.meta}")

    def test_trainer_config_and_boolean_validation(self):
        """Trainer YAML maps the independent toggle, with strict bool validation."""
        config = SimpleNamespace(
            accelerator=SimpleNamespace(tp_size=2, sequence_parallel=False, loss_parallel=False),
            fsdp_config=SimpleNamespace(dp_shard_size=2, edp_shard_size=1),
            compile=CompileConfig(fsdp_overlap_grad_reduce=False),
        )
        result = build_pass_config_from_trainer_config(config)
        self.assertFalse(result.fsdp_overlap_grad_reduce, f"expected False, got {result.fsdp_overlap_grad_reduce}")
        for value in (1, "true", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "fsdp_overlap_grad_reduce"):
                    PassConfig(fsdp_overlap_grad_reduce=value)
                with self.assertRaisesRegex(TypeError, "fsdp_overlap_grad_reduce"):
                    CompileConfig(fsdp_overlap_grad_reduce=value)


class _TiedNormMLP(torch.nn.Module):
    """A tied weight accumulates gradients from both matmuls before reduction."""

    def __init__(self) -> None:
        """Initialize tied and normalization parameters in double precision."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.float64))
        self.tail = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.float64))
        self.norm = torch.nn.LayerNorm(8, dtype=torch.float64)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Reuse the same parameter, then normalize and project."""
        value = torch.relu(value @ self.weight) @ self.weight
        return self.norm(value) @ self.tail


class TestGradientOverlapNumerics(unittest.TestCase):
    """Run real traced forward/backward and storage frees with simulated collectives."""

    def test_independent_pass_combinations_preserve_loss_and_all_gradient_shards(self):
        """All four AG/RS switch combinations retain eager values and shard ownership."""
        torch.manual_seed(2026)
        for model in (_MLP(3).double(), _ChunkedMLP(), _TiedNormMLP()):
            inputs = torch.randn(4, 8, dtype=torch.float64)
            labels = torch.randn(4, 4, dtype=torch.float64)
            eager_loss = torch.nn.functional.mse_loss(model(inputs), labels)
            eager_grads = torch.autograd.grad(eager_loss, tuple(model.parameters()))
            for reshard in (False, True):
                for prefetch, grad_overlap in ((False, False), (True, False), (False, True), (True, True)):
                    with self.subTest(model=type(model).__name__, reshard=reshard,
                                      prefetch=prefetch, grad_overlap=grad_overlap):
                        baseline, live_baseline = _trace_and_run(model, reshard)
                        optimized, live_optimized = _trace_and_run(model, reshard)
                        FsdpOverlapPass().run(optimized, PassConfig(
                            enable_overlap=False, fsdp_allgather_prefetch=prefetch,
                            fsdp_overlap_grad_reduce=grad_overlap))
                        loss0, grads0 = _run_simulated(baseline, live_baseline, model, inputs, labels)
                        loss1, grads1 = _run_simulated(optimized, live_optimized, model, inputs, labels)
                        torch.testing.assert_close(loss1, loss0, rtol=0, atol=0)
                        torch.testing.assert_close(loss1, eager_loss, rtol=1e-12, atol=1e-12)
                        self.assertEqual(len(grads1), len(grads0),
                                         f"expected {len(grads0)} gradients, got {len(grads1)}")
                        for expected, actual in zip(grads0, grads1):
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        for actual, full in zip(grads1, eager_grads):
                            torch.testing.assert_close(actual, full.chunk(2, dim=0)[0], rtol=1e-12, atol=1e-12)
                        if grad_overlap:
                            stats = optimized.meta["fsdp_grad_overlap"]
                            self.assertGreater(stats["launches_advanced"], 0,
                                               f"expected real RS movement, got {stats}")


if __name__ == "__main__":
    unittest.main()
