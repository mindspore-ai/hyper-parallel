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
"""FSDP module selection, safety boundaries and plan/report contracts."""

import copy
import json
import unittest
from unittest.mock import patch

import torch
from torch import fx, nn
from torch.ops import _c10d_functional as c10d

from hyper_parallel.compile.passes.bucketing.planner import (
    Bucket, BucketPlan, BucketPolicy, build_module_bucket_plans,
    normalize_module_bucket_plans, plan_fsdp_buckets, validate_group_plans,
)
from hyper_parallel.compile.passes.bucketing.bucketing_pass import apply_ag_buckets, apply_rs_buckets
from hyper_parallel.compile.passes.parallel.fsdp_pass import _free_tensor_storage, annotate_fsdp_bucket_specs
from tests.ut.compile.test_fsdp_reshard import _MLP, _OddWeightMLP, _trace_and_run


def _opaque(value):
    return value


def bucket_graph(shapes: tuple = ((4, 2), (6, 2)), *, kind: str = "rs", reverse: bool = False,
                 frees: bool = False, backward: bool = False, dtype: torch.dtype = torch.float64,
                 modules: tuple = ()) -> tuple[fx.GraphModule, list, list]:
    """Build real ATen consumers around tagged two-rank AG or RS transport."""
    graph = fx.Graph()
    inputs = [graph.placeholder(f"input{index}") for index in range(len(shapes))]
    sources = inputs
    if kind == "rs":
        backward = True
        sources = [graph.call_function(torch.ops.aten.clone.default, (node,)) for node in inputs]
        for node in sources:
            node.meta["autograd_backward"] = True
    work = graph.call_function(torch.ops.aten.sum.default, (inputs[0],)) if kind == "rs" else None
    if work is not None:
        work.meta["autograd_backward"] = True
    pairs, outputs = [], [work] if work is not None else []
    for index in reversed(range(len(shapes))) if reverse else range(len(shapes)):
        shape = shapes[index]
        target = c10d.reduce_scatter_tensor.default if kind == "rs" else c10d.all_gather_into_tensor.default
        args = (sources[index], "avg", 2, "fsdp") if kind == "rs" else (sources[index], 2, "fsdp")
        launch = graph.call_function(target, args)
        out_shape = (shape[0] // 2 if kind == "rs" else shape[0] * 2, *shape[1:])
        launch.meta.update(comm_type="fsdp_reduce_scatter" if kind == "rs" else
                           "fsdp_all_gather_backward" if backward else "fsdp_all_gather",
                           comm_group="fsdp", autograd_backward=backward, state_idx=index, fsdp_degree=2,
                           param_name=f"{modules[index] if modules else 'block'}.p{index}",
                           fsdp_unsharded_nbytes=torch.empty(out_shape, dtype=dtype).numel() * dtype.itemsize,
                           fsdp_bucket_spec={"input_shape": shape, "output_shape": out_shape,
                                             "dtype": dtype, "device": "cpu"})
        wait = graph.call_function(c10d.wait_tensor.default, (launch,))
        wait.meta["autograd_backward"] = backward
        value = wait
        if kind == "ag":
            value = graph.call_function(torch.ops.aten.sum.default, (wait,))
            value.meta["autograd_backward"] = backward
        outputs.append(value if frees else wait)
        if frees:
            release = graph.call_function(_free_tensor_storage, (wait,))
            release.meta.update(comm_type="fsdp_reshard_free_backward" if backward else "fsdp_reshard_free",
                                param_name=launch.meta["param_name"], autograd_backward=backward)
        pairs.append((launch, wait))
    graph.output(tuple(outputs))
    return fx.GraphModule({}, graph), sources, pairs


def make_plan(graph: fx.GraphModule, **kwargs: object) -> BucketPlan:
    """Plan against an explicitly known two-rank process group."""
    return plan_fsdp_buckets(graph, group_ranks={"fsdp": (0, 1)}, **kwargs)


def reasons(plan: BucketPlan) -> list[str]:
    """Read public skip/split decisions without depending on log prose."""
    return [item["reason"] for item in plan.decisions]


def staggered_graph() -> tuple[fx.GraphModule, list[fx.Node], list[tuple[fx.Node, fx.Node]]]:
    """Build an early output view before the second gradient becomes ready."""
    graph, sources, reductions = bucket_graph()
    reductions[0][1].append(sources[1])
    with graph.graph.inserting_before(sources[1]):
        view = graph.graph.call_function(torch.ops.aten.view.default, (reductions[0][1], (2, 2)))
        view.meta["autograd_backward"] = True
        nested = graph.graph.call_function(torch.ops.aten.transpose.int, (view, 0, 1))
        nested.meta["autograd_backward"] = True
    output = list(graph.graph.nodes)[-1]
    output.args = ((output.args[0][0], nested, reductions[1][1]),)
    reductions[0][0].meta["param_name"] = "layers.0.q.weight"
    reductions[1][0].meta["param_name"] = "layers.0.k.weight"
    graph.graph.lint()
    graph.recompile()
    return graph, sources, reductions


class TestBucketPlanner(unittest.TestCase):
    """Keep identity, layout, compatibility and effect checks independently observable."""

    def test_read_only_export_and_rewrite_lifetime(self) -> None:
        """Planning is deterministic; exported facts outlive mutation and removed nodes."""
        graph, _, _ = bucket_graph(reverse=True)
        before = (graph.code, copy.deepcopy([node.meta for node in graph.graph.nodes]))
        modules = ["block"]
        plan = make_plan(graph, module_bucket_plans=modules)
        saved = plan.to_dict()
        self.assertEqual([m.state_idx for m in plan.buckets[0].members], [0, 1])
        self.assertEqual([m["local_offset"] for m in saved["buckets"][0]["members"]], [0, 4])
        self.assertEqual(saved["buckets"][0]["size_bytes"], 160)
        self.assertEqual((graph.code, [node.meta for node in graph.graph.nodes]), before)
        self.assertEqual(saved, make_plan(graph, module_bucket_plans=modules).to_dict())
        modules.append("unused")
        self.assertEqual(plan.module_bucket_plans, (("block",),))
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_group_plans({0: plan, 1: make_plan(graph, module_bucket_plans=[])}, "fsdp")
        edited = plan.to_dict()
        edited["buckets"][0]["members"][0]["param_name"] = "changed"
        edited["summary"].clear()
        with self.assertLogs("hyper_parallel.compile.passes.bucketing", level="DEBUG") as logs:
            apply_rs_buckets(graph, plan)
        self.assertTrue(any("applied:" in record.getMessage() for record in logs.records))
        self.assertEqual(plan.to_dict(), saved)
        validate_group_plans({0: plan, 1: json.loads(json.dumps(saved))}, "fsdp")

    def test_policy_limits_and_validation(self) -> None:
        """Each limit has a distinct reason and oversized tensors remain singleton."""
        for policy, reason in ((BucketPolicy(cap_bytes=100), "capacity"),
                               (BucketPolicy(max_input_distance=0), "input_distance"),
                               (BucketPolicy(max_consumer_distance=0), "consumer_distance")):
            with self.subTest(reason=reason):
                graph, _, _ = bucket_graph(kind="ag" if reason == "consumer_distance" else "rs")
                self.assertIn(reason, reasons(make_plan(graph, policy=policy)))
        plan = make_plan(bucket_graph()[0], policy=BucketPolicy(cap_bytes=10))
        self.assertEqual(plan.summary["reason_counts"]["singleton"], {"oversized": 2})
        self.assertEqual(plan.summary["singletons"], 2)
        self.assertEqual(len({bucket.bucket_id for bucket in plan.buckets}), 2)
        for options in ({"cap_bytes": 0}, {"max_input_distance": True}, {"max_consumer_distance": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                BucketPolicy(**options)
        with self.assertRaises(ValueError):
            plan_fsdp_buckets(bucket_graph()[0], group_ranks={"fsdp": (0, 0)})

    def test_invalid_facts_and_modes(self) -> None:
        """Malformed identity/spec/phase and unsupported parallel modes never fuse."""
        cases = (("state_idx", None, "missing_parameter_identity"),
                 ("param_name", object(), "missing_parameter_identity"),
                 ("fsdp_bucket_spec", {}, "unknown_or_dynamic_shape"),
                 ("autograd_backward", False, "phase_mismatch"))
        specs = (("output_shape", (7, 2), "invalid_dim0_layout"),
                 ("dtype", None, "unsupported_dtype"), ("device", "invalid", "unknown_device"))
        for spec, mutations in ((False, cases), (True, specs)):
            for field, value, reason in mutations:
                with self.subTest(field=field):
                    graph, _, pairs = bucket_graph()
                    meta = pairs[0][0].meta
                    (meta["fsdp_bucket_spec"] if spec else meta)[field] = value
                    self.assertIn(reason, reasons(make_plan(graph)))
        for options in ({"pp_degree": 2}, {"dp_mode": "hsdp"}):
            plan = make_plan(bucket_graph()[0], **options)
            self.assertFalse(plan.buckets)
            self.assertEqual(set(reasons(plan)), {"unsupported_parallel_mode"})
        graph, _, pairs = bucket_graph()
        pairs[0][0].args = ()
        self.assertIn("unsupported_signature", reasons(make_plan(graph)))

    def test_compatibility_and_repeated_identity(self) -> None:
        """Same-sized tensors still split by scope, dtype, device, operation and PG."""
        for field, value in (("param_name", "other.p1"), ("dtype", torch.float32), ("device", "cpu:1"),
                             ("reduce_op", "sum"), ("group", "other"), ("state_idx", 0)):
            with self.subTest(field=field):
                graph, _, pairs = bucket_graph()
                node = pairs[1][0]
                if field in ("dtype", "device"):
                    node.meta["fsdp_bucket_spec"][field] = value
                elif field in ("reduce_op", "group"):
                    args = list(node.args)
                    args[1 if field == "reduce_op" else -1] = value
                    node.args = tuple(args)
                    if field == "group":
                        node.meta["comm_group"] = value
                else:
                    node.meta[field] = value
                plan = plan_fsdp_buckets(graph, group_ranks={"fsdp": (0, 1), "other": (0, 1)})
                self.assertEqual([len(b.members) for b in plan.buckets], [1, 1])
                if field == "state_idx":
                    self.assertIn("repeated_parameter_instance", reasons(plan))
                if field == "group":
                    self.assertIn("unknown_process_group", reasons(make_plan(graph)))

    def test_effects_and_consumers_bound_delayed_launch(self) -> None:
        """A late input may move views, but cannot cross effects or real consumers."""
        for target in (_opaque, _free_tensor_storage, torch.ops.aten.add_.Tensor, c10d.all_reduce):
            with self.subTest(target=target):
                graph, sources, _ = staggered_graph()
                args = (sources[0], sources[0]) if target is torch.ops.aten.add_.Tensor else (sources[0],)
                if target is c10d.all_reduce:
                    args = (sources[0], "sum", "tp")
                with graph.graph.inserting_before(sources[1]):
                    node = graph.graph.call_function(target, args)
                    node.meta["autograd_backward"] = True
                    if target is _free_tensor_storage:
                        node.meta["comm_type"] = "fsdp_reshard_free_backward"
                reason = "intervening_collective" if target is c10d.all_reduce else "effect_or_storage_release"
                self.assertIn(reason, reasons(make_plan(graph, module_bucket_plans=["layers.0"])))
        for target in (torch.ops.aten.sum.default, c10d.all_reduce):
            graph, sources, pairs = staggered_graph()
            args = (pairs[0][1], "avg", "replicate") if target is c10d.all_reduce else (pairs[0][1],)
            with graph.graph.inserting_before(sources[1]):
                node = graph.graph.call_function(target, args)
                node.meta["autograd_backward"] = True
            self.assertIn("non_output_rs_consumer", reasons(make_plan(graph)))
        graph, inputs, pairs = bucket_graph(kind="ag")
        with graph.graph.inserting_before(pairs[1][0]):
            late = graph.graph.call_function(torch.ops.aten.clone.default, (inputs[1],))
        pairs[1][0].replace_input_with(inputs[1], late)
        self.assertIn("dependency_conflict", reasons(make_plan(graph)))

    def test_contiguous_metadata_preserves_padded_shape(self) -> None:
        """Read through private contiguous without losing the actual padded shape."""
        graph, sources, reductions = bucket_graph(shapes=((6, 2), (4, 2)))
        sources[0].meta["val"] = torch.empty(5, 2, dtype=torch.float64)
        sources[0].meta["fsdp_padded_input_spec"] = {
            "shape": (6, 2), "dtype": torch.float64, "device": "cpu",
        }
        sources[1].meta["val"] = torch.empty(4, 2, dtype=torch.float64)
        for launch, _ in reductions:
            launch.meta.pop("fsdp_bucket_spec")
            with graph.graph.inserting_before(launch):
                node = graph.graph.call_method("contiguous", (launch.args[0],))
            launch.replace_input_with(launch.args[0], node)
        annotate_fsdp_bucket_specs(graph, torch.nn.Module())
        self.assertEqual([launch.meta["fsdp_bucket_spec"]["input_shape"] for launch, _ in reductions],
                         [(6, 2), (4, 2)])
        self.assertEqual([launch.meta["fsdp_bucket_spec"]["output_shape"] for launch, _ in reductions],
                         [(3, 2), (2, 2)])
        self.assertEqual(len(make_plan(graph).buckets), 1)

    def test_nonprivate_or_nondefault_contiguous_remains_a_boundary(self) -> None:
        """Planning must not waive arbitrary methods, ownership or phase boundaries."""
        for case in ("shared", "forward", "owner", "kwargs", "positional", "unknown"):
            with self.subTest(case=case):
                graph, _, reductions = bucket_graph()
                launch = reductions[-1][0]
                with graph.graph.inserting_before(launch):
                    node = graph.graph.call_method("contiguous", (launch.args[0],))
                launch.replace_input_with(launch.args[0], node)
                if case == "shared":
                    output = next(item for item in graph.graph.nodes if item.op == "output")
                    output.args = ((*output.args[0], node),)
                elif case == "forward":
                    node.meta["autograd_backward"] = False
                elif case == "owner":
                    node.meta["fsdp_reduce_scatter_prepare"] = "another_reduction"
                elif case == "kwargs":
                    node.kwargs = {"memory_format": torch.channels_last}
                elif case == "positional":
                    node.args = (*node.args, torch.contiguous_format)
                else:
                    node.target = "clone"
                self.assertEqual(len(make_plan(graph).buckets), 2)
                self.assertEqual(node.op, "call_method")

    def test_cross_rank_digest_ignores_device_index_but_checks_layout_and_rank_order(self) -> None:
        """Cross rank digest ignores device index but checks layout and rank order."""
        graph0, _, _ = bucket_graph()
        graph1, _, reductions = bucket_graph()
        for node, _ in reductions:
            node.meta["fsdp_bucket_spec"]["device"] = "cpu:1"
        plans = {0: make_plan(graph0), 1: make_plan(graph1)}
        validate_group_plans(plans, "fsdp")
        validate_group_plans({rank: json.loads(json.dumps(plan.to_dict())) for rank, plan in plans.items()}, "fsdp")
        reductions[0][0].meta["state_idx"] = 10
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_group_plans({0: plans[0], 1: make_plan(graph1)}, "fsdp")
        with self.assertRaisesRegex(ValueError, "Expected reports"):
            validate_group_plans({0: plans[0]}, "fsdp")
        reversed_plan = plan_fsdp_buckets(graph0, group_ranks={"fsdp": (1, 0)})
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_group_plans({0: plans[0], 1: reversed_plan}, "fsdp")

    def test_module_builder_and_selection(self) -> None:
        """Model scopes use real, unambiguous dot-boundary FQNs; empty means disabled."""
        model = nn.ModuleDict({"embedding": nn.Embedding(8, 4), "norm": nn.LayerNorm(4), "head": nn.Linear(4, 8),
                               "layers": nn.ModuleList([nn.Sequential(nn.Linear(4, 4)) for _ in range(2)])})
        plans = build_module_bucket_plans(model, block_module_types=(nn.Sequential,),
                                          module_bucket_plans=["embedding", ["norm", "head"]])
        self.assertEqual(plans, (("layers.0",), ("layers.1",), ("embedding",), ("norm", "head")))
        model.layers[0].nested = nn.Sequential(nn.Linear(4, 4))
        self.assertEqual(build_module_bucket_plans(model, block_module_types=(nn.Sequential,)),
                         (("layers.0",), ("layers.1",)))
        for options in ({"module_bucket_plans": ["missing"]}, {"block_module_types": (str,)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                build_module_bucket_plans(model, **options)
        for invalid in ("layers.0", ["*"], [""], [[]], [["norm", 3]], ["layers", "layers.0"],
                        [["norm", "norm"]], ["layers..0"], [" layers.0"], ["layers.0", "layers.0"]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_module_bucket_plans(invalid)
        graph, _, pairs = bucket_graph()
        for (node, _), name in zip(pairs, ("layers.1.q.weight", "layers.10.k.weight")):
            node.meta["param_name"] = name
        plan = make_plan(graph, module_bucket_plans=["layers.1"])
        self.assertEqual([m.param_name for b in plan.buckets for m in b.members], ["layers.1.q.weight"])
        self.assertFalse(make_plan(graph, module_bucket_plans=[]).buckets)

    def test_multi_module_group_and_unmatched_parameters(self) -> None:
        """Explicit multi-module groups fuse only selected parameters."""
        graph, _, reductions = bucket_graph(shapes=((4, 2), (6, 2), (4, 2)))
        for (launch, _), name in zip(reductions, ("norm.weight", "head.weight", "other.weight")):
            launch.meta["param_name"] = name
        plan = make_plan(graph, module_bucket_plans=[["norm", "head"]])
        self.assertEqual([len(bucket.members) for bucket in plan.buckets], [2])
        self.assertEqual(plan.buckets[0].members[0].module_fqns, ("norm", "head"))
        self.assertEqual([decision["reason"] for decision in plan.decisions], ["outside_module_plan"])
        apply_rs_buckets(graph, plan)
        self.assertIn(reductions[-1][0], list(graph.graph.nodes))
        self.assertTrue(all(node.meta["fsdp_bucket_module_fqns"] == ("norm", "head")
                            for node in graph.graph.nodes if node.meta.get("fsdp_bucket_id")))

    def test_interleaved_other_module_or_dtype_does_not_reorder_wire(self) -> None:
        """Interleaved compatibility groups retain the original collective order."""
        for other_module in (False, True):
            with self.subTest(other_module=other_module):
                graph, _, reductions = bucket_graph(shapes=((4, 2),) * 3)
                for launch, _ in reductions:
                    launch.meta["param_name"] = "layers.0." + launch.meta["param_name"]
                if other_module:
                    reductions[1][0].meta["param_name"] = "layers.1.q.weight"
                else:
                    reductions[1][0].meta["fsdp_bucket_spec"]["dtype"] = torch.float32
                plan = make_plan(graph, module_bucket_plans=["layers.1", "layers.0"])
                self.assertEqual([len(bucket.members) for bucket in plan.buckets], [1, 1, 1])
                self.assertEqual([bucket.members[0].state_idx for bucket in plan.buckets], [0, 1, 2])
                self.assertIn("intervening_collective", [decision["reason"] for decision in plan.decisions])

    def test_diagnostics_and_application_contract(self) -> None:
        """Report selection separately from eligibility; publish stats only on success."""
        graph, _, pairs = bucket_graph(shapes=((4, 2),) * 4)
        for (node, _), name in zip(pairs, ("norm.weight", "norm.bias", "other.weight", "layers.10.weight")):
            node.meta["param_name"] = name
        pairs[1][0].meta["fsdp_bucket_spec"].pop("dtype")
        with self.assertLogs("hyper_parallel.compile.passes.bucketing", level="DEBUG"):
            plan = make_plan(graph, module_bucket_plans=[["norm", "head"], "layers.1"])
        self.assertEqual([plan.summary[k] for k in ("selected_parameters", "selected_collectives",
                                                   "candidate_collectives")], [2, 2, 1])
        self.assertEqual(plan.summary["unmatched_module_paths"], ["head", "layers.1"])
        self.assertEqual(plan.summary["reason_counts"]["skip"], {"unsupported_dtype": 1, "outside_module_plan": 2})
        graph, _, _ = bucket_graph()
        with patch.object(Bucket, "describe", side_effect=AssertionError("unexpected serialization")), \
                patch.object(BucketPlan, "to_dict", side_effect=AssertionError("unexpected export")):
            apply_rs_buckets(graph, make_plan(graph))
        for kind, apply in (("ag", apply_ag_buckets), ("rs", apply_rs_buckets)):
            with self.subTest(kind=kind):
                graph, _, _ = bucket_graph(kind=kind)
                graph.meta[f"fsdp_{kind}_bucketing"] = {"previous": True}
                with patch.object(graph, "recompile", side_effect=RuntimeError("compile failed")):
                    with self.assertRaisesRegex(RuntimeError, "compile failed"):
                        apply(graph, make_plan(graph))
                self.assertEqual(graph.meta[f"fsdp_{kind}_bucketing"], {"previous": True})

    def test_traced_metadata_and_phase_counts(self) -> None:
        """Real FSDP traces expose padded specs and distinct forward/backward buckets."""
        for model in (_MLP(2), _OddWeightMLP()):
            graph, _ = _trace_and_run(model, reshard=True)
            plan = make_plan(graph)
            self.assertTrue(plan.buckets)
            self.assertFalse([d for d in plan.decisions if d["action"] == "skip"])
            self.assertEqual(plan.summary["selected_parameters"], len(list(model.parameters())))
            self.assertGreater(plan.summary["selected_collectives"], plan.summary["selected_parameters"])
            for bucket in plan.buckets:
                for member in bucket.members:
                    smaller, larger = ((member.output_shape, member.input_shape) if member.reduce_op else
                                       (member.input_shape, member.output_shape))
                    self.assertEqual(larger[0], smaller[0] * 2)
