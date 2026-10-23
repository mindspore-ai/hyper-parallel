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
"""FSDP prefetch ordering, memory bounds, and joint-graph numerical regression."""

import operator
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import fx
from torch.ops import _c10d_functional

from hyper_parallel.compile.pass_config import PassConfig, build_pass_config_from_trainer_config
from hyper_parallel.compile.passes.overlap import overlap_param_gather as prefetch_module
from hyper_parallel.compile.passes.overlap.schedule import FsdpOverlapPass
from hyper_parallel.compile.passes.parallel.fsdp_pass import _free_tensor_storage
from hyper_parallel.models.build_options import CompileConfig
from tests.ut.compile.test_fsdp_reshard import _ChunkedMLP, _MLP, _run_simulated, _trace_and_run


def _positions(graph_module):
    return {node: index for index, node in enumerate(graph_module.graph.nodes)}


def _gather(graph, shard, *, backward=False, tagged=True, nbytes=64, default=False):
    """Insert a parameter gather/wait pair with configurable FSDP metadata."""
    packet = _c10d_functional.all_gather_into_tensor
    launch = graph.call_function(packet.default if default else packet, (shard, 2, "fsdp"))
    wait = graph.call_function(_c10d_functional.wait_tensor, (launch,))
    if tagged:
        launch.meta.update(comm_type="fsdp_all_gather_backward" if backward else "fsdp_all_gather",
                           param_name=shard.name, fsdp_unsharded_nbytes=nbytes)
    if backward:
        launch.meta["autograd_backward"] = True
        wait.meta["autograd_backward"] = True
    return launch, wait


def _mlp_graph(*, tagged=True, backward=False):
    """Build three gather-backed matrix products within one autograd phase."""
    graph = fx.Graph()
    value = graph.placeholder("x")
    shards = [graph.placeholder(f"shard{index}") for index in range(3)]
    launches, waits, computes = [], [], []
    for index, shard in enumerate(shards):
        launch, wait = _gather(graph, shard, backward=backward, tagged=tagged, default=index % 2 == 0)
        view = graph.call_function(torch.ops.aten.t.default, (wait,))
        value = graph.call_function(torch.ops.aten.mm.default, (value, view))
        view.meta["autograd_backward"] = backward
        value.meta["autograd_backward"] = backward
        launches.append(launch)
        waits.append(wait)
        computes.append(value)
    graph.output(value)
    return fx.GraphModule({}, graph), launches, waits, computes


def _run(graph_module, **kwargs):
    return FsdpOverlapPass().run(graph_module, PassConfig(**kwargs))


class TestPrefetchSchedule(unittest.TestCase):
    """Consumer prefetch preserves phases, effects, and parameter dependencies."""

    def test_forward_and_backward_prefetch_and_wait_sinking(self):
        """Both phases support packet/default collectives and delay wait/view DAGs."""
        for backward in (False, True):
            with self.subTest(backward=backward):
                graph, launches, waits, computes = _mlp_graph(backward=backward)
                _run(graph)
                order = _positions(graph)
                self.assertLess(order[launches[1]], order[computes[0]])
                self.assertLess(order[launches[2]], order[computes[1]])
                self.assertLess(order[launches[1]], order[waits[0]])
                self.assertEqual([node for node in graph.graph.nodes if node in launches], launches)
                self.assertEqual(graph.meta["fsdp_prefetch"]["backward" if backward else "forward"], 2)
                graph.graph.lint()

    def test_non_fsdp_gathers_are_unchanged(self):
        """TP/activation gathers from placeholders must not be mistaken for FSDP."""
        graph, _, _, _ = _mlp_graph(tagged=False)
        before = list(graph.graph.nodes)
        _run(graph)
        self.assertEqual(list(graph.graph.nodes), before)

    def test_disabled_overlap_does_not_mutate_graph(self):
        """The switch disables both prefetch and wait/view sinking."""
        graph, _, _, _ = _mlp_graph()
        before = graph.code
        _run(graph, fsdp_allgather_prefetch=False)
        self.assertEqual(graph.code, before)

    def test_prefetch_is_independent_of_legacy_overlap(self):
        """Both phases still prefetch when the generic overlap switch is off."""
        for backward in (False, True):
            with self.subTest(backward=backward):
                graph, launches, _, computes = _mlp_graph(backward=backward)
                _run(graph, enable_overlap=False, fsdp_overlap_grad_reduce=False)
                order = _positions(graph)
                self.assertLess(order[launches[1]], order[computes[0]],
                                f"expected AG before compute, got positions {order}")

    def test_budget_zero_small_and_unknown_sizes_prevent_launches(self):
        """Budget enforcement fails closed when full replica sizes are unknown."""
        for budget, unknown in ((0, False), (63, False), (128, True)):
            with self.subTest(budget=budget, unknown=unknown):
                graph, launches, _, computes = _mlp_graph()
                if unknown:
                    for launch in launches:
                        launch.meta.pop("fsdp_unsharded_nbytes")
                _run(graph, max_in_flight_gb=budget / (1 << 30))
                self.assertGreater(_positions(graph)[launches[1]], _positions(graph)[computes[0]])

    def test_budget_is_shared_across_overlapping_launch_windows(self):
        """One-replica budget cannot cover two simultaneously prefetched replicas."""
        graph, launches, _, computes = _mlp_graph()
        _run(graph, max_in_flight_gb=64 / (1 << 30))
        order = _positions(graph)
        self.assertLess(order[launches[1]], order[computes[0]],
                        f"expected second gather before first compute, got {order}")
        self.assertGreater(order[launches[2]], order[computes[1]],
                           f"expected third gather after second compute under 64-byte budget, got {order}")

    def test_schedule_is_idempotent(self):
        """Repeated pipeline application does not increase speculative residency."""
        graph, _, _, _ = _mlp_graph()
        _run(graph)
        code = graph.code
        _run(graph)
        self.assertEqual(graph.code, code)

    def test_mutation_is_a_launch_boundary(self):
        """Parameter prefetch never crosses an intervening shard mutation."""
        module, launches, _, computes = _mlp_graph()
        with module.graph.inserting_after(computes[0]):
            barrier = module.graph.call_function(torch.ops.aten.add_.Tensor, (launches[1].args[0], 1))
        _run(module)
        order = _positions(module)
        self.assertGreater(order[launches[1]], order[barrier],
                           f"expected gather after mutation, got positions {order}")
        module.graph.lint()

    def test_backward_gathers_stay_after_forward_storage_release(self):
        """A joint graph never hoists a backward re-gather into forward."""
        module, launches, _, computes = _mlp_graph(backward=True)
        graph = module.graph
        with graph.inserting_before(launches[0]):
            forward = graph.call_function(torch.ops.aten.relu.default, (launches[0].args[0],))
            free = graph.call_function(_free_tensor_storage, (forward,))
            free.meta.update(comm_type="fsdp_reshard_free", param_name=launches[0].args[0].name)
        _run(module)
        order = _positions(module)
        self.assertTrue(all(order[free] < order[launch] for launch in launches))
        self.assertLess(order[launches[1]], order[computes[0]])


class TestPrefetchDepthPolicy(unittest.TestCase):
    """Module patches control each phase without introducing public config fields."""

    def test_module_depths_select_phase_specific_consumer_anchors(self):
        """Each invocation reads the current phase's depth and clamps large lookahead."""
        for forward_depth, backward_depth in ((1, 2), (2, 1), (5, 1), (1, 5)):
            for backward in (False, True):
                with self.subTest(forward_depth=forward_depth, backward_depth=backward_depth, backward=backward):
                    module, launches, _, computes = _mlp_graph(backward=backward)
                    depth = backward_depth if backward else forward_depth
                    expected = computes[max(0, 2 - depth)]
                    with patch.multiple(prefetch_module, fsdp_forward_prefetch_depth=forward_depth,
                                        fsdp_backward_prefetch_depth=backward_depth):
                        _run(module)
                    anchor = launches[2].meta["fsdp_prefetch_anchor"]
                    self.assertEqual(anchor, expected.name,
                                     f"expected depth {depth} anchor {expected.name}, got {anchor}")
                    order = _positions(module)
                    self.assertLess(order[launches[2]], order[expected],
                                    f"expected gather before {expected.name}, got {order}")
                    if depth == 1:
                        self.assertGreater(order[launches[2]], order[computes[0]],
                                           f"expected depth 1 to retain preceding compute, got {order}")
                    module.graph.lint()


class TestPrefetchCollectiveScheduling(unittest.TestCase):
    """Collective reordering retains input, effect, and storage constraints."""

    def test_gather_crosses_independent_tp_launch_in_both_phases(self):
        """Cross-collective prefetch is built in, while TP launches remain in place."""
        for backward in (False, True):
            with self.subTest(backward=backward):
                module, launches, _, computes = _mlp_graph(backward=backward)
                with module.graph.inserting_after(computes[0]):
                    tp = module.graph.call_function(_c10d_functional.all_reduce, (computes[0], "sum", "tp"))
                    tp.meta["autograd_backward"] = backward
                _run(module)
                order = _positions(module)
                self.assertLess(order[launches[1]], order[computes[0]],
                                f"expected AG before compute, got {order}")
                self.assertLess(order[launches[1]], order[tp],
                                f"expected AG before independent TP launch, got {order}")
                module.graph.lint()

    def test_gathers_follow_consumers_when_launch_order_differs(self):
        """An earlier consumer can overtake a gather for a later consumer."""
        module, launches, waits, computes = _mlp_graph()
        launches[1].prepend(launches[2])
        launches[1].prepend(waits[2])
        _run(module)
        order = _positions(module)
        self.assertLess(order[launches[1]], order[launches[2]],
                        f"expected earlier-consumed gather first, got {order}")
        self.assertLess(order[launches[1]], order[computes[0]],
                        f"expected prefetch before preceding compute, got {order}")
        module.graph.lint()

    def test_collective_mode_preserves_unowned_collective_order(self):
        """TP launches remain stationary relative to one another and computation."""
        module, launches, _, computes = _mlp_graph()
        graph = module.graph
        with graph.inserting_after(computes[0]):
            first = graph.call_function(_c10d_functional.all_reduce, (computes[0], "sum", "tp"))
        with graph.inserting_after(first):
            graph.call_function(_c10d_functional.all_reduce, (computes[0], "sum", "other"))
        movable = {node for node in graph.nodes if node in launches or "wait_tensor" in str(node.target)
                   or node.target is torch.ops.aten.t.default}
        stationary = [node for node in graph.nodes if node not in movable]
        _run(module)
        self.assertEqual([node for node in graph.nodes if node not in movable], stationary)

    def test_collective_mode_preserves_mutation_and_unknown_effect_boundaries(self):
        """Removing communication boundaries does not remove storage-write barriers."""
        for mutation in (False, True):
            with self.subTest(mutation=mutation):
                module, launches, _, computes = _mlp_graph()
                with module.graph.inserting_after(computes[0]):
                    barrier = module.graph.call_function(
                        torch.ops.aten.add_.Tensor if mutation else operator.neg,
                        (launches[1].args[0], 1) if mutation else (computes[0],))
                _run(module)
                order = _positions(module)
                self.assertGreater(order[launches[1]], order[barrier])
                module.graph.lint()

    def test_collective_mode_keeps_budget_and_unknown_size_checks(self):
        """Crossing collectives does not permit unbounded speculative replicas."""
        for budget, unknown in ((0, False), (63, False), (128, True)):
            with self.subTest(budget=budget, unknown=unknown):
                module, launches, _, computes = _mlp_graph()
                if unknown:
                    launches[1].meta.pop("fsdp_unsharded_nbytes")
                _run(module, max_in_flight_gb=budget / (1 << 30))
                self.assertGreater(_positions(module)[launches[1]], _positions(module)[computes[0]])


class TestPrefetchNumerics(unittest.TestCase):
    """Execute real joint graphs with mocked collectives and real storage frees."""

    def test_loss_and_parameter_gradients_match_with_and_without_reshard(self):
        """Gather reordering preserves the unsharded reference loss and every gradient."""
        for reshard in (False, True):
            for model_type in (_MLP, _ChunkedMLP):
                with self.subTest(reshard=reshard, model_type=model_type):
                    torch.manual_seed(42)
                    reference = (_MLP(3) if model_type is _MLP else _ChunkedMLP()).double()
                    graph, live = _trace_and_run(reference, reshard=reshard)
                    inputs, labels = torch.randn(4, 8).double(), torch.randn(4, 4).double()
                    reference_loss = torch.nn.functional.mse_loss(reference(inputs), labels)
                    reference_loss.backward()
                    _run(graph)
                    loss, grads = _run_simulated(graph, live, reference, inputs, labels)
                    torch.testing.assert_close(loss, reference_loss, rtol=1e-12, atol=1e-12)
                    for actual, parameter in zip(grads, reference.parameters()):
                        torch.testing.assert_close(actual, parameter.grad.chunk(2, dim=0)[0], rtol=1e-12, atol=1e-12)


class TestPrefetchConfig(unittest.TestCase):
    """Graph controls use their own namespace and validate finite memory limits."""

    def test_trainer_maps_graph_controls_without_eager_depths(self):
        """Module depth settings do not leak into consumer-group scheduling."""
        config = SimpleNamespace(
            accelerator=SimpleNamespace(tp_size=2, sequence_parallel=False, loss_parallel=False),
            fsdp_config=SimpleNamespace(dp_shard_size=2, edp_shard_size=1, forward_prefetch_depth=0,
                                        backward_prefetch_depth=7, reshard_after_forward=False),
            compile=CompileConfig(fsdp_allgather_prefetch=False, max_in_flight_gb=0.5),
        )
        result = build_pass_config_from_trainer_config(config)
        self.assertFalse(result.fsdp_allgather_prefetch, f"expected disabled prefetch, got {result}")
        self.assertTrue(result.fsdp_overlap_grad_reduce, f"expected enabled gradient overlap, got {result}")
        self.assertFalse(result.enable_overlap, f"expected legacy builder default False, got {result}")
        self.assertFalse(result.fsdp_reshard_after_forward, f"expected reshard False, got {result}")
        self.assertEqual(result.max_in_flight_gb, 0.5, f"expected 0.5 GiB, got {result.max_in_flight_gb}")
        enabled = build_pass_config_from_trainer_config(config, enable_overlap=True)
        self.assertTrue(enabled.enable_overlap, f"expected explicit legacy switch True, got {enabled}")
        self.assertFalse(enabled.fsdp_allgather_prefetch, f"legacy switch must not enable prefetch: {enabled}")

    def test_missing_compile_config_keeps_both_fsdp_passes_enabled(self):
        """Older Trainer config objects inherit the independent default-on flags."""
        config = SimpleNamespace(
            accelerator=SimpleNamespace(tp_size=1, sequence_parallel=False, loss_parallel=False),
            fsdp_config=SimpleNamespace(dp_shard_size=2, edp_shard_size=1),
        )
        result = build_pass_config_from_trainer_config(config)
        for field in ("fsdp_allgather_prefetch", "fsdp_overlap_grad_reduce"):
            self.assertTrue(getattr(result, field), f"expected {field}=True, got {result}")
        self.assertEqual(result.max_in_flight_gb, 0.25, f"expected 0.25 GiB, got {result}")

    def test_nonfinite_negative_or_nonnumeric_limits_are_rejected(self):
        """NaN/inf must not bypass memory checks, and bool is not a numeric budget."""
        for invalid in (-1, True, "0.25", None, float("nan"), float("inf")):
            for config_type in (PassConfig, CompileConfig):
                with self.subTest(invalid=invalid, config_type=config_type):
                    with self.assertRaisesRegex(ValueError, "max_in_flight_gb"):
                        config_type(max_in_flight_gb=invalid)

    def test_prefetch_flag_requires_a_boolean(self):
        """Both public configuration boundaries reject ambiguous switch values."""
        for invalid in (0, 1, "true", None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "fsdp_allgather_prefetch"):
                    PassConfig(fsdp_allgather_prefetch=invalid)
                with self.assertRaisesRegex(TypeError, "fsdp_allgather_prefetch"):
                    CompileConfig(fsdp_allgather_prefetch=invalid)
