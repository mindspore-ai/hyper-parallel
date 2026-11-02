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
"""AG/RS packing, overlap and storage lifetimes against actual ATen/eager execution."""

import copy
import operator
import unittest
from itertools import product

import torch
from torch import fx, nn
from torch.ops import _c10d_functional as c10d
from torch.utils._pytree import tree_leaves

from hyper_parallel.compile.pass_config import PassConfig
from hyper_parallel.compile.passes.bucketing.planner import validate_group_plans
from hyper_parallel.compile.passes.bucketing.bucketing_pass import apply_ag_buckets, apply_rs_buckets
from hyper_parallel.compile.passes.overlap.overlap_grad_comm import overlap_grad_comm
from hyper_parallel.compile.passes.overlap.overlap_param_gather import overlap_param_gather
from hyper_parallel.compile.passes.overlap.schedule import FsdpOverlapPass
from hyper_parallel.compile.passes.parallel.fsdp_pass import _free_tensor_storage
from tests.ut.compile.test_bucket_planner import _opaque, bucket_graph, make_plan, staggered_graph
from tests.ut.compile.test_fsdp_reshard import _MLP, _OddWeightMLP, _trace_and_run


def _shard(value, rank):
    if not value.ndim:
        return value
    if value.shape[0] % 2:
        value = torch.cat((value, torch.zeros_like(value[:1])))
    return value.chunk(2)[rank]


class _Transport(fx.Interpreter):
    """Mock only transport; execute real packing, copies, aliases and storage frees."""

    def __init__(self, graph: fx.GraphModule, rank: int = 0, full: nn.Module = None) -> None:
        """Use an eager-model oracle or a synthetic peer offset by 100."""
        super().__init__(graph)
        self.rank, self.full = rank, dict(full.named_parameters()) if full is not None else None
        self.workspaces, self.releases = [], []

    def run_node(self, node: fx.Node) -> object:
        """Reject freed-storage reads and validate the actual packed local shard."""
        args, kwargs = self.fetch_args_kwargs_from_env(node)
        args = tuple(args)
        if node.target is not _free_tensor_storage:
            for value in tree_leaves((args, kwargs)):
                if isinstance(value, torch.Tensor) and value.numel() and not value.untyped_storage().nbytes():
                    raise AssertionError(f"Read after free at {node.name}")
        if self.full is not None and node.meta.get("comm_type") in ("fsdp_all_gather", "fsdp_all_gather_backward"):
            names = node.meta.get("fsdp_bucket_members", (node.meta.get("param_name"),))
            rows = [torch.stack([_shard(self.full[name].detach(), rank).reshape(-1) for rank in (0, 1)])
                    for name in names]
            packed = torch.cat(rows, dim=1)
            torch.testing.assert_close(args[0].reshape(-1), packed[self.rank], rtol=0, atol=0)
            return packed.reshape(node.meta["fsdp_bucket_spec"]["output_shape"]).clone()
        if node.target is _free_tensor_storage:
            storage = args[0].untyped_storage()
            result = super().run_node(node)
            self.releases.append(storage.nbytes())
            return result
        return super().run_node(node)

    def call_function(self, target: object, args: tuple, kwargs: dict) -> object:
        """Simulate two ranks while leaving every ATen operation untouched."""
        if str(target).startswith("_c10d_functional.all_gather_into_tensor"):
            result = torch.cat((args[0], args[0] + 100))
            self.workspaces.append(result)
            return result
        if str(target).startswith("_c10d_functional.reduce_scatter_tensor"):
            return args[0].chunk(2)[self.rank].clone()
        if str(target).startswith("_c10d_functional.wait_tensor"):
            return args[0]
        return super().call_function(target, args, kwargs)


class _LifecycleModel(nn.Module):
    """Small models sharing a regression head but exercising distinct parameter lifetimes."""

    def __init__(self, kind: str) -> None:
        """Build repeated or tied parameter consumers."""
        super().__init__()
        self.kind = kind
        self.tail = nn.Linear(8, 4)
        self.block = nn.Linear(8, 8) if kind == "repeated" else nn.Sequential(
            nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh())
        if kind == "tied":
            self.block = nn.ModuleDict({"embedding": nn.Embedding(8, 8), "head": nn.Linear(8, 8)})
            self.block.head.weight = self.block.embedding.weight
        self.double()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Execute repeated reads or tied lookup/projection."""
        if self.kind == "tied":
            hidden = self.block.head(torch.tanh(self.block.embedding(inputs.argmax(dim=1))))
        else:
            hidden = torch.tanh(self.block(torch.tanh(self.block(inputs))))
        return self.tail(hidden)


class TestBucketRewrite(unittest.TestCase):
    """Pin numerical layouts, scheduling boundaries and independent storage ownership."""

    def _outputs(self, graph, reference, inputs, rank=0):
        actual = _Transport(graph, rank).run(*(value.clone() for value in inputs))
        expected = _Transport(reference, rank).run(*(value.clone() for value in inputs))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return actual

    def _overlap(self, graph, config=None):
        config = config or PassConfig()
        FsdpOverlapPass().run(graph, config)
        code, stats = graph.code, copy.deepcopy(graph.meta)
        FsdpOverlapPass().run(graph, config)
        self.assertEqual(graph.code, code)
        return stats

    def test_layout_and_independent_ag_storage(self) -> None:
        """Unequal shapes, noncontiguous inputs and member reordering preserve both ranks."""
        for kind, dtype, reverse, rank in product(("ag", "rs"), (torch.float32, torch.float64, torch.bfloat16),
                                                   (False, True), (0, 1)):
            with self.subTest(kind=kind, dtype=dtype, reverse=reverse, rank=rank):
                graph, _, _ = bucket_graph(kind=kind, dtype=dtype, reverse=reverse)
                reference = copy.deepcopy(graph)
                apply = apply_ag_buckets if kind == "ag" else apply_rs_buckets
                apply(graph, make_plan(graph))
                inputs = (torch.arange(8, dtype=dtype).reshape(2, 4).t(),
                          torch.arange(12, dtype=dtype).reshape(2, 6).t() + 20)
                runner = _Transport(graph, rank)
                actual, expected = runner.run(*inputs), _Transport(reference, rank).run(*inputs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(graph.meta[f"fsdp_{kind}_bucketing"][f"{kind}_collectives_after"], 1)
                if kind == "ag":
                    owners = [value.untyped_storage().data_ptr() for value in actual]
                    self.assertEqual(len(set(owners)), 2)
                    self.assertNotIn(runner.workspaces[0].untyped_storage().data_ptr(), owners)
                    _free_tensor_storage(actual[0])
                    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_stale_foreign_repeated_and_wrong_kind_plans(self) -> None:
        """Plan validation precedes mutation; no-op calls refresh only invocation stats."""
        for kind, apply, other in (("ag", apply_ag_buckets, apply_rs_buckets),
                                   ("rs", apply_rs_buckets, apply_ag_buckets)):
            with self.subTest(kind=kind):
                graph, _, _ = bucket_graph(kind=kind)
                plan, before = make_plan(graph), graph.code
                other(graph, plan)
                self.assertEqual(graph.code, before)
                plan = make_plan(graph)
                with self.assertRaisesRegex(ValueError, "different graph"):
                    apply(copy.deepcopy(graph), plan)
                apply(graph, plan)
                before = (graph.code, copy.deepcopy([node.meta for node in graph.graph.nodes]))
                with self.assertRaisesRegex(ValueError, "Stale"):
                    apply(graph, plan)
                apply(graph, make_plan(graph))
                self.assertEqual((graph.code, [node.meta for node in graph.graph.nodes]), before)
                self.assertEqual(graph.meta[f"fsdp_{kind}_bucketing"]["buckets"], 0)

    def test_late_inputs_repair_views(self) -> None:
        """Both rewrites place communication after late inputs and preserve early aliases."""
        for kind in ("ag", "rs"):
            graph, sources, pairs = staggered_graph()
            if kind == "ag":
                for launch, _ in pairs:
                    launch.target = c10d.all_gather_into_tensor.default
                    launch.args = (launch.args[0], 2, "fsdp")
                    launch.meta["comm_type"] = "fsdp_all_gather_backward"
                    spec = launch.meta["fsdp_bucket_spec"]
                    spec["output_shape"] = (spec["input_shape"][0] * 2, *spec["input_shape"][1:])
                view = next(node for node in graph.graph.nodes if node.target is torch.ops.aten.view.default)
                view.args = (view.args[0], (8, 2))
            graph.recompile()
            reference = copy.deepcopy(graph)
            apply = apply_ag_buckets if kind == "ag" else apply_rs_buckets
            apply(graph, make_plan(graph, module_bucket_plans=["layers.0"]))
            launch = next(node for node in graph.graph.nodes if node.meta.get("fsdp_bucket_members"))
            nodes = list(graph.graph.nodes)
            self.assertGreater(nodes.index(launch), nodes.index(sources[1]))
            self._overlap(graph)
            for rank in (0, 1):
                self._outputs(graph, reference, (torch.arange(8.).reshape(2, 4).t(),
                                                torch.arange(12.).reshape(2, 6).t()), rank)

    def test_pack_and_launch_advance_together_and_wait_sinks(self) -> None:
        """Pack and launch advance together and wait sinks."""
        graph, sources, _ = bucket_graph()
        apply_rs_buckets(graph, make_plan(graph))
        overlap_grad_comm(graph, PassConfig())
        nodes = list(graph.graph.nodes)
        launch = next(n for n in nodes if n.meta.get("fsdp_bucket_members"))
        work = next(n for n in nodes if n.target is torch.ops.aten.sum.default)
        wait = next(iter(launch.users))
        preparation = [n for n in nodes if n.meta.get("fsdp_bucket_prepare")]
        self.assertLess(nodes.index(sources[-1]), nodes.index(preparation[0]))
        self.assertLess(nodes.index(preparation[-1]), nodes.index(launch))
        self.assertLess(nodes.index(launch), nodes.index(work))
        self.assertLess(nodes.index(work), nodes.index(wait))
        before = graph.code
        overlap_grad_comm(graph, PassConfig())
        apply_rs_buckets(graph, make_plan(graph))
        self.assertEqual(before, graph.code)

    def test_contiguous_bucket_and_singleton_preserve_upstream_scheduling(self) -> None:
        """Fused packs and an unfused RS both retain upstream contiguous preparation."""
        for owned_aten in (False, True):
            with self.subTest(owned_aten=owned_aten):
                graph, sources, reductions = bucket_graph(shapes=((4, 2), (6, 2), (4, 2)))
                reductions[-1][0].meta["param_name"] = "other.weight"
                preparations = []
                for launch, _ in reductions:
                    with graph.graph.inserting_before(launch):
                        if owned_aten:
                            node = graph.graph.call_function(torch.ops.aten.contiguous.default, (launch.args[0],))
                            node.meta.update(fsdp_reduce_scatter_prepare=launch.name, autograd_backward=True)
                        else:
                            node = graph.graph.call_method("contiguous", (launch.args[0],))
                    launch.replace_input_with(launch.args[0], node)
                    preparations.append(node)
                graph.recompile()
                reference = copy.deepcopy(graph)
                plan = make_plan(graph)
                self.assertEqual(sorted(len(bucket.members) for bucket in plan.buckets), [1, 2])
                apply_rs_buckets(graph, plan)
                overlap_grad_comm(graph, PassConfig())
                nodes = list(graph.graph.nodes)
                work = next(node for node in nodes if node.target is torch.ops.aten.sum.default)
                launches = [node for node in nodes if node.meta.get("comm_type") == "fsdp_reduce_scatter"]
                self.assertEqual(len(launches), 2)
                self.assertTrue(all(nodes.index(node) < nodes.index(work) for node in launches))
                self.assertTrue(all(node.target is torch.ops.aten.contiguous.default for node in preparations))
                for source, node in zip(sources, preparations):
                    self.assertLess(nodes.index(source), nodes.index(node))
                    self.assertLess(nodes.index(node), nodes.index(work))
                inputs = tuple(torch.arange(size * 2, dtype=torch.float64).reshape(2, size).t()
                               for size in (4, 6, 4))
                for rank in (0, 1):
                    self._outputs(graph, reference, inputs, rank)
                before = graph.code
                overlap_grad_comm(graph, PassConfig())
                self.assertEqual(graph.code, before)

    def test_unknown_effect_blocks_pack_motion(self) -> None:
        """Unknown effect blocks pack motion."""
        graph, _, reductions = bucket_graph()
        with graph.graph.inserting_before(reductions[0][0]):
            barrier = graph.graph.call_function(_opaque, (reductions[0][0].args[0],))
            barrier.meta["autograd_backward"] = True
        apply_rs_buckets(graph, make_plan(graph))
        overlap_grad_comm(graph, PassConfig())
        order = {n: i for i, n in enumerate(graph.graph.nodes)}
        self.assertTrue(all(order[barrier] < order[n] for n in graph.graph.nodes
                            if n.meta.get("fsdp_bucket_prepare")))

    def test_shared_pack_producer_is_not_moved(self) -> None:
        """A packed tensor exposed elsewhere cannot be moved with its collective."""
        for kind, apply in (("ag", apply_ag_buckets), ("rs", apply_rs_buckets)):
            with self.subTest(kind=kind):
                graph, _, _ = bucket_graph(kind=kind, shapes=((4, 2),) * 4, modules=("a", "a", "b", "b"))
                apply(graph, make_plan(graph))
                launch = [n for n in graph.graph.nodes if n.meta.get("fsdp_bucket_members")][-1]
                pack, output = launch.args[0], list(graph.graph.nodes)[-1]
                output.args = ((*output.args[0], pack),)
                work = next(n for n in graph.graph.nodes if n.target is torch.ops.aten.sum.default)
                self._overlap(graph)
                nodes = list(graph.graph.nodes)
                self.assertLess(nodes.index(work), nodes.index(pack))
                self.assertLess(nodes.index(pack), nodes.index(launch))
                self.assertNotIn("fsdp_prefetch_anchor", launch.meta)

    def test_related_storage_free_stays_after_wait(self) -> None:
        """Related storage free stays after wait."""
        graph, sources, _ = bucket_graph()
        apply_rs_buckets(graph, make_plan(graph))
        output = list(graph.graph.nodes)[-1]
        with graph.graph.inserting_before(output):
            free = graph.graph.call_function(_free_tensor_storage, (sources[0],))
            free.meta.update(comm_type="fsdp_reshard_free_backward", autograd_backward=True)
        overlap_grad_comm(graph, PassConfig())
        nodes = list(graph.graph.nodes)
        wait = next(n for n in nodes if n.target is c10d.wait_tensor.default)
        self.assertLess(nodes.index(wait), nodes.index(free))
        values = _Transport(graph, 0).run(torch.ones(4, 2), torch.ones(6, 2))
        self.assertTrue(all(torch.isfinite(value).all() for value in values))

    def test_unrelated_release_effect_and_foreign_collective_still_split(self) -> None:
        """Only exact original member frees receive the independent-copy exception."""
        for kind in ('other_free', 'source_free', 'effect', 'collective'):
            with self.subTest(kind=kind):
                gm, inputs, pairs = bucket_graph(kind="ag")
                with gm.graph.inserting_before(pairs[1][0]):
                    if kind in ('other_free', 'source_free'):
                        source = inputs[0]
                        if kind == 'other_free':
                            source = gm.graph.call_function(torch.ops.aten.clone.default, (inputs[0],))
                        node = gm.graph.call_function(_free_tensor_storage, (source,))
                        node.meta.update(comm_type="fsdp_reshard_free", param_name="block.p0")
                    elif kind == 'effect':
                        gm.graph.call_function(_opaque, (inputs[0],))
                    else:
                        gm.graph.call_function(c10d.all_reduce.default, (inputs[0], 'sum', 'other'))
                self.assertFalse(any(bucket.supports_ag_rewrite for bucket in make_plan(gm).buckets))

    def test_prefetch_moves_pack_and_respects_whole_bucket_budget(self) -> None:
        """The scheduler sees consumers beyond clones and budgets all member replicas."""
        for budget in (0, 191, 192):
            with self.subTest(budget=budget):
                gm, _, _ = bucket_graph(kind="ag", shapes=((2, 3),) * 4, modules=['a', 'a', 'b', 'b'])
                apply_ag_buckets(gm, make_plan(gm))
                launches = [node for node in gm.graph.nodes if node.meta.get('fsdp_bucket_members')]
                self.assertEqual([node.meta['fsdp_unsharded_nbytes'] for node in launches], [192, 192])
                stats = self._overlap(gm, PassConfig(max_in_flight_gb=budget / 2**30))
                self.assertEqual(stats["fsdp_prefetch"]["forward"], int(budget == 192))
                inputs = tuple(torch.arange(6.).reshape(2, 3) + 10 * i for i in range(4))
                expected = tuple(torch.cat((shard, shard + 100)) for shard in inputs)
                torch.testing.assert_close(_Transport(gm).run(*inputs), expected, rtol=0, atol=0)

    def test_prior_member_free_bounds_launch(self) -> None:
        """Any member's previous materialization bounds the next bucket prefetch."""
        gm, _, _ = bucket_graph(kind="ag", shapes=((2, 3),) * 4, modules=['a', 'a', 'b', 'b'], backward=True)
        apply_ag_buckets(gm, make_plan(gm))
        launches = [node for node in gm.graph.nodes if node.meta.get('fsdp_bucket_members')]
        second = launches[1]
        with gm.graph.inserting_before(second.args[0]):
            extra = gm.graph.call_function(torch.ops.aten.clone.default, (next(iter(gm.graph.nodes)),))
            release = gm.graph.call_function(_free_tensor_storage, (extra,))
            release.meta.update(comm_type='fsdp_reshard_free', param_name='b.p3')
        overlap_param_gather(gm, PassConfig())
        positions = {node: i for i, node in enumerate(gm.graph.nodes)}
        self.assertGreater(positions[second], positions[release])

    def test_repeated_materializations_preserve_each_release(self) -> None:
        """Repeated parameter identities form separate buckets with separate lifetimes."""
        for backward, overlap in product((False, True), repeat=2):
            with self.subTest(backward=backward, overlap=overlap):
                graph, inputs, pairs = bucket_graph(kind="ag", shapes=((2, 3),) * 4, frees=True, backward=backward)
                for index, (launch, wait) in enumerate(pairs):
                    identity = index % 2
                    launch.args = (inputs[identity], 2, "fsdp")
                    launch.meta.update(param_name=f"block.p{identity}", state_idx=identity)
                    release = next(node for node in wait.users if node.target is _free_tensor_storage)
                    release.meta["param_name"] = f"block.p{identity}"
                plan = make_plan(graph)
                self.assertIn("repeated_parameter_instance", [item["reason"] for item in plan.decisions])
                self.assertEqual(sum(bucket.supports_ag_rewrite for bucket in plan.buckets), 2)
                baseline = copy.deepcopy(graph)
                apply_ag_buckets(graph, plan)
                if overlap:
                    self._overlap(graph)
                positions = {node: index for index, node in enumerate(graph.graph.nodes)}
                launches = [node for node in graph.graph.nodes if node.meta.get("fsdp_bucket_members")]
                releases = [node for node in graph.graph.nodes if node.target is _free_tensor_storage]
                self.assertEqual(len(releases), 4)
                for release in releases[:2]:
                    self.assertLess(positions[release], positions[launches[1]])
                shards = tuple(torch.arange(6., dtype=torch.float64).reshape(2, 3) + index for index in range(4))
                self._outputs(graph, baseline, shards)

    def test_split_transpose_aliases_survive_other_member_free(self) -> None:
        """Late noncontiguous aliases remain live after another bucket member is freed."""
        graph, _, pairs = bucket_graph(kind="ag", shapes=((2, 4),) * 2, frees=True)
        wait = pairs[0][1]
        early = next(node for node in wait.users if node.target is torch.ops.aten.sum.default)
        release = next(node for node in wait.users if node.target is _free_tensor_storage)
        output = list(graph.graph.nodes)[-1]
        output.prepend(release)
        with graph.graph.inserting_before(early):
            split = graph.graph.call_function(torch.ops.aten.split.Tensor, (wait, 2, 0))
            left = graph.graph.call_function(operator.getitem, (split, 0))
            right = graph.graph.call_function(operator.getitem, (split, 1))
            transposed = graph.graph.call_function(torch.ops.aten.t.default, (right,))
        early.replace_input_with(wait, left)
        with graph.graph.inserting_before(release):
            late = graph.graph.call_function(torch.ops.aten.sum.default, (transposed,))
        output.args = ((*output.args[0], late),)
        baseline = copy.deepcopy(graph)
        apply_ag_buckets(graph, make_plan(graph))
        self.assertEqual(graph.meta["fsdp_ag_bucketing"]["buckets"], 1)
        self._overlap(graph)
        positions = {node: index for index, node in enumerate(graph.graph.nodes)}
        other_release = next(node for node in graph.graph.nodes
                             if node.target is _free_tensor_storage and node is not release)
        self.assertLess(positions[other_release], positions[late])
        self.assertLess(positions[late], positions[release])
        shards = tuple(torch.arange(8., dtype=torch.float64).reshape(2, 4) + index for index in range(2))
        self._outputs(graph, baseline, shards)

    def test_shard_alias_mutation_is_a_bucket_and_prefetch_boundary(self) -> None:
        """A write through a shard alias separates old and new materializations."""
        graph, inputs, pairs = bucket_graph(kind="ag", shapes=((2, 3),) * 2, frees=True)
        pairs[1][0].args = (inputs[0], 2, "fsdp")
        with graph.graph.inserting_before(pairs[1][0]):
            alias = graph.graph.call_function(torch.ops.aten.t.default, (inputs[0],))
            mutation = graph.graph.call_function(torch.ops.aten.add_.Tensor, (alias, 7))
        baseline = copy.deepcopy(graph)
        plan = make_plan(graph)
        self.assertIn("effect_or_storage_release", [item["reason"] for item in plan.decisions])
        apply_ag_buckets(graph, plan)
        self.assertEqual(graph.meta["fsdp_ag_bucketing"]["buckets"], 0)
        FsdpOverlapPass().run(graph, PassConfig())
        positions = {node: index for index, node in enumerate(graph.graph.nodes)}
        self.assertLess(positions[pairs[0][0]], positions[mutation])
        self.assertLess(positions[mutation], positions[pairs[1][0]])
        shards = (torch.arange(6., dtype=torch.float64).reshape(2, 3), torch.zeros(2, 3, dtype=torch.float64))
        actual = self._outputs(graph, baseline, shards)
        torch.testing.assert_close(actual[1] - actual[0], torch.tensor(84., dtype=torch.float64))

    def test_release_through_alias_stays_conservative(self) -> None:
        """Free through a view is not mistaken for the exact direct-member exception."""
        graph, _, pairs = bucket_graph(kind="ag", shapes=((2, 3),) * 2, frees=True)
        wait = pairs[0][1]
        release = next(node for node in wait.users if node.target is _free_tensor_storage)
        with graph.graph.inserting_before(release):
            alias = graph.graph.call_function(torch.ops.aten.t.default, (wait,))
        release.replace_input_with(wait, alias)
        plan = make_plan(graph)
        self.assertIn("effect_or_storage_release", [item["reason"] for item in plan.decisions])
        baseline = copy.deepcopy(graph)
        apply_ag_buckets(graph, plan)
        self.assertEqual(graph.meta["fsdp_ag_bucketing"]["buckets"], 0)
        FsdpOverlapPass().run(graph, PassConfig())
        shards = tuple(torch.arange(6., dtype=torch.float64).reshape(2, 3) + index for index in range(2))
        self._outputs(graph, baseline, shards)

    def test_training_matches_eager(self) -> None:
        """All variants preserve both rank shards, accumulation and AdamW across four model contracts."""
        for kind, reshard, overlap, variant in product(("mlp", "padded", "repeated", "tied"),
                                                       (False, True), (False, True), ("ag", "rs", "ag_rs")):
            with self.subTest(model=kind, reshard=reshard, overlap=overlap, variant=variant):
                torch.manual_seed(419)
                full = (_MLP(3).double() if kind == "mlp" else
                        _OddWeightMLP() if kind == "padded" else _LifecycleModel(kind))
                graphs, models = zip(*[_trace_and_run(full, reshard=reshard, rank=rank) for rank in (0, 1)])
                modules = ["block", "tail"] if kind in ("repeated", "tied") else None
                for tag, apply in (("ag", apply_ag_buckets), ("rs", apply_rs_buckets)):
                    if tag in variant:
                        plans = [make_plan(graph, module_bucket_plans=modules) for graph in graphs]
                        validate_group_plans(dict(enumerate(plans)), "fsdp")
                        for graph, plan in zip(graphs, plans):
                            apply(graph, plan)
                            self.assertGreater(graph.meta[f"fsdp_{tag}_bucketing"]["buckets"], 0)
                for graph in graphs:
                    if overlap:
                        self._overlap(graph)
                sequences = [[(n.meta.get("comm_type"), n.meta.get("fsdp_bucket_members"), n.meta.get("param_name"))
                              for n in graph.graph.nodes if str(n.target).startswith("_c10d_functional.")]
                             for graph in graphs]
                self.assertEqual(sequences[0], sequences[1])
                if kind == "tied":
                    for model in (full, *models):
                        self.assertIs(model.block.embedding.weight, model.block.head.weight)
                self._train_against_eager(graphs, models, full, reshard)

    def _train_against_eager(self, graphs, models, full, reshard):
        """Compare accumulated gradients, parameters and every AdamW state on both ranks."""
        optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3, foreach=False) for m in (full, *models)]
        for _ in range(3):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            for _ in range(2):
                inputs, labels = torch.randn(4, 8, dtype=torch.float64), torch.randn(4, 4, dtype=torch.float64)
                loss = nn.functional.mse_loss(full(inputs), labels)
                grads = torch.autograd.grad(loss, tuple(full.parameters()))
                for param, grad in zip(full.parameters(), grads):
                    param.grad = grad.clone() if param.grad is None else param.grad + grad
                for rank, (graph, model) in enumerate(zip(graphs, models)):
                    runner = _Transport(graph, rank, full)
                    with torch.no_grad():
                        actual = runner.run(*model.parameters(), inputs, labels)
                    expected = (loss, *(_shard(grad, rank) for grad in grads))
                    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
                    for param, grad in zip(model.parameters(), actual[1:]):
                        param.grad = grad.clone() if param.grad is None else param.grad + grad
                    frees = sum(n.target is _free_tensor_storage for n in graph.graph.nodes)
                    self.assertEqual(runner.releases, [0] * frees)
                    if reshard:
                        self.assertGreater(frees, 0)
            for optimizer in optimizers:
                optimizer.step()
            for rank, (model, optimizer) in enumerate(zip(models, optimizers[1:])):
                for param, full_param in zip(model.parameters(), full.parameters()):
                    torch.testing.assert_close(param, _shard(full_param, rank), rtol=1e-10, atol=1e-12)
                    for key, value in optimizers[0].state[full_param].items():
                        torch.testing.assert_close(optimizer.state[param][key], _shard(value, rank),
                                                   rtol=1e-10, atol=1e-12)
