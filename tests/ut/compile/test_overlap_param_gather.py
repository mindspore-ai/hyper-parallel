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

import torch
from torch import fx
from torch.ops import _c10d_functional

from hyper_parallel.compile.pass_config import PassConfig, build_pass_config_from_trainer_config
from hyper_parallel.compile.passes.overlap.schedule import AutoOverlapPass
from hyper_parallel.compile.passes.parallel.fsdp_pass import _free_tensor_storage
from hyper_parallel.models.build_options import CompileConfig
from tests.ut.compile.test_fsdp_reshard import _ChunkedMLP, _MLP, _run_simulated, _trace_and_run


def _positions(graph_module):
    return {node: index for index, node in enumerate(graph_module.graph.nodes)}


def _gather(graph, shard, *, backward=False, tagged=True, nbytes=64, default=False):
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
    return AutoOverlapPass().run(graph_module, PassConfig(**kwargs))


class TestPrefetchSchedule(unittest.TestCase):
    """Consumer prefetch preserves phases, effects, and collective launch order."""

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
        _run(graph, enable_overlap=False)
        self.assertEqual(graph.code, before)

    def test_zero_depth_disables_only_the_selected_phase(self):
        """Forward/backward depth zero independently disables early launches."""
        for backward in (False, True):
            with self.subTest(backward=backward):
                graph, launches, _, computes = _mlp_graph(backward=backward)
                _run(graph, fsdp_forward_prefetch_depth=0 if not backward else 1,
                     fsdp_backward_prefetch_depth=0 if backward else 1)
                self.assertGreater(_positions(graph)[launches[1]], _positions(graph)[computes[0]])

    def test_budget_zero_small_and_unknown_sizes_prevent_launches(self):
        """Budget enforcement fails closed when full replica sizes are unknown."""
        for budget, unknown in ((0, False), (63, False), (128, True)):
            with self.subTest(budget=budget, unknown=unknown):
                graph, launches, _, computes = _mlp_graph()
                if unknown:
                    for launch in launches:
                        launch.meta.pop("fsdp_unsharded_nbytes")
                _run(graph, fsdp_prefetch_budget_bytes=budget)
                self.assertGreater(_positions(graph)[launches[1]], _positions(graph)[computes[0]])

    def test_budget_is_shared_across_overlapping_launch_windows(self):
        """A larger depth cannot reserve several replicas under a one-replica budget."""
        graph, launches, _, computes = _mlp_graph()
        _run(graph, fsdp_forward_prefetch_depth=2, fsdp_prefetch_budget_bytes=64)
        order = _positions(graph)
        self.assertLess(order[launches[1]], order[computes[0]])
        self.assertGreater(order[launches[2]], order[computes[0]])

    def test_schedule_is_idempotent(self):
        """Repeated pipeline application does not increase speculative residency."""
        graph, _, _, _ = _mlp_graph()
        _run(graph)
        code = graph.code
        _run(graph)
        self.assertEqual(graph.code, code)

    def test_attention_interval_obeys_shared_replica_budget(self):
        """Attention prefetch respects the same byte limit as consumer lookahead."""
        for budget in (64, 128):
            with self.subTest(budget=budget):
                graph = fx.Graph()
                x = graph.placeholder("x")
                shards = [graph.placeholder(f"shard{index}") for index in range(2)]
                attention = graph.call_function(torch.ops.aten.scaled_dot_product_attention.default, (x, x, x))
                launch1, wait1 = _gather(graph, shards[0])
                launch2, wait2 = _gather(graph, shards[1])
                result = graph.call_function(torch.ops.aten.add.Tensor, (wait1, wait2))
                graph.output((attention, result))
                module = fx.GraphModule({}, graph)
                _run(module, fsdp_prefetch_budget_bytes=budget)
                order = _positions(module)
                self.assertLess(order[launch1], order[attention])
                if budget == 64:
                    self.assertGreater(order[launch2], order[attention])
                else:
                    self.assertLess(order[launch2], order[attention])

    def test_branching_and_multi_output_views_sink_safely(self):
        """Wait sinking handles every branch and getitem from multi-output views."""
        graph = fx.Graph()
        x, shard = graph.placeholder("x"), graph.placeholder("shard")
        launch, wait = _gather(graph, shard)
        split = graph.call_function(torch.ops.aten.split.Tensor, (wait, 2))
        first = graph.call_function(operator.getitem, (split, 0))
        second = graph.call_function(torch.ops.aten.view.default, (wait, [-1]))
        independent = graph.call_function(torch.ops.aten.relu.default, (x,))
        user1 = graph.call_function(torch.ops.aten.add.Tensor, (first, independent))
        user2 = graph.call_function(torch.ops.aten.sigmoid.default, (second,))
        graph.output((user1, user2))
        module = fx.GraphModule({}, graph)
        _run(module)
        order = _positions(module)
        self.assertLess(order[independent], order[wait])
        self.assertLess(order[launch], order[wait])
        self.assertLess(order[wait], order[split])
        self.assertLess(order[first], order[user1])
        self.assertLess(order[second], order[user2])
        graph.lint()

    def test_mutation_and_collective_launch_are_ordering_barriers(self):
        """Prefetch cannot cross shard writes or preceding TP collectives."""
        for mutation in (True, False):
            with self.subTest(mutation=mutation):
                module, launches, _, computes = _mlp_graph()
                graph = module.graph
                with graph.inserting_after(computes[0]):
                    if mutation:
                        barrier = graph.call_function(torch.ops.aten.add_.Tensor, (launches[1].args[0], 1))
                    else:
                        barrier = graph.call_function(_c10d_functional.all_reduce, (computes[0], "sum", "tp"))
                _run(module)
                order = _positions(module)
                self.assertGreater(order[launches[1]], order[barrier])
                graph.lint()

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

    def test_private_cast_moves_but_shared_cast_respects_input_readiness(self):
        """Only a private parameter conversion can join an early launch."""
        for shared in (False, True):
            with self.subTest(shared=shared):
                module, launches, _, computes = _mlp_graph()
                graph = module.graph
                launch = launches[1]
                with graph.inserting_before(launch):
                    cast = graph.call_function(torch.ops.prims.convert_element_type.default,
                                               (launch.args[0], torch.float32))
                    if shared:
                        graph.call_function(torch.ops.aten.relu.default, (cast,))
                launch.replace_input_with(launch.args[0], cast)
                _run(module)
                order = _positions(module)
                self.assertLess(order[cast], order[launch])
                if shared:
                    self.assertGreater(order[launch], order[computes[0]])
                else:
                    self.assertLess(order[launch], order[computes[0]])


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
    """Public graph options reject invalid budgets and preserve explicit switches."""

    def test_trainer_projects_overlap_depths_and_budget(self):
        """The YAML-visible switch and eager prefetch depths reach the static pass."""
        config = SimpleNamespace(
            accelerator=SimpleNamespace(tp_size=2, sequence_parallel=False, loss_parallel=False),
            fsdp_config=SimpleNamespace(dp_shard_size=2, edp_shard_size=1, forward_prefetch_depth=2,
                                        backward_prefetch_depth=3, reshard_after_forward=False),
            compile=CompileConfig(enable_overlap=True, fsdp_prefetch_budget_bytes=1024),
        )
        result = build_pass_config_from_trainer_config(config)
        self.assertTrue(result.enable_overlap)
        self.assertFalse(result.fsdp_reshard_after_forward)
        self.assertEqual(result.fsdp_forward_prefetch_depth, 2)
        self.assertEqual(result.fsdp_backward_prefetch_depth, 3)
        self.assertEqual(result.fsdp_prefetch_budget_bytes, 1024)
        self.assertFalse(build_pass_config_from_trainer_config(config, enable_overlap=False).enable_overlap)

    def test_invalid_depth_and_budget(self):
        """Negative, boolean, and fractional limits fail at the configuration boundary."""
        for field in ("fsdp_forward_prefetch_depth", "fsdp_backward_prefetch_depth", "fsdp_prefetch_budget_bytes"):
            for invalid in (-1, True, 1.5):
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, field):
                        PassConfig(**{field: invalid})
        with self.assertRaisesRegex(TypeError, "enable_overlap"):
            CompileConfig(enable_overlap=1)
        with self.assertRaisesRegex(ValueError, "fsdp_prefetch_budget_bytes"):
            CompileConfig(fsdp_prefetch_budget_bytes=-1)
