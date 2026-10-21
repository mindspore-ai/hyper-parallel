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
"""Unit tests for ``hyper_parallel.compile.passes.parallel.fsdp_pass.FSDPPass``.

Covers the execution-layer contract of the pass (previously exercised only at
pipeline-assembly level in ``test_pass_pipeline``):

``run`` guards:
1. Early-returns (graph + live model untouched) when distributed is not
   initialized, and when ``world_size == 1``.
2. Raises ``ValueError`` when the caller omits the live ``model=`` kwarg
   (sharding cannot proceed without it).

``run`` happy path:
3. With ``fsdp_degree`` from the config, shards the live model's params in
   place (dim 0) and inserts AllGather / ReduceScatter / wait_tensor nodes
   into the graph; returns the same graph_module.
4. Pad-shards non-divisible dense parameters, reduces replicated fallbacks,
   and keeps expert communication on its independent EDP group.
5. Uses ``pass_config.fsdp_degree`` over ``world_size`` (a TP+FSDP hybrid
   where the FSDP group is a proper sub-group of the world).

Pure helpers:
6. ``_param_belongs_to_fsdp_module`` matches via exact FQN and wildcard
   pattern; a top-level parameter (no module ancestor) is never matched.
7. ``_get_parent_module_fqn`` strips the trailing parameter leaf.
8. ``_build_trainable_state_indices`` skips buffers and frozen params,
   mirroring the tracer; falls back to all-state when the model is absent.

No real distributed communication happens at pass time: collectives are
inserted as FX nodes only, so ``torch.distributed`` is mocked throughout.
"""

import unittest
from contextlib import contextmanager
from typing import Iterator
from unittest.mock import MagicMock, patch


import torch
from torch import fx, nn

from hyper_parallel.compile.pass_config import PassConfig
from hyper_parallel.compile.passes.parallel.fsdp_pass import FSDPPass
from hyper_parallel.compile.graph_parallel_plan import GraphParallelPlan

_DIST_PATH = "hyper_parallel.compile.passes.parallel.fsdp_pass.dist"
_RESOLVE_PG_PATH = (
    "hyper_parallel.compile.passes.parallel.fsdp_pass._resolve_process_group"
)


@contextmanager
def _patch_dist(
    world_size: int = 2, rank: int = 0, initialized: bool = True
) -> Iterator[MagicMock]:
    """Patch ``dist`` inside ``fsdp_pass`` with a stub returning the given values.

    The pass only calls ``dist.is_initialized`` / ``dist.get_world_size`` /
    ``dist.get_rank`` at pass time (collectives are FX nodes, never executed),
    so a single MagicMock covers every call.  ``_resolve_process_group`` is
    patched as well so ``_shard_live_model_params`` can resolve the FSDP group
    name without a real process group being registered.
    """
    mock_dist = MagicMock()
    mock_dist.is_initialized.return_value = initialized
    mock_dist.get_world_size.return_value = world_size
    mock_dist.get_rank.return_value = rank
    mock_pg = MagicMock()
    with patch(_DIST_PATH, mock_dist), patch(_RESOLVE_PG_PATH, return_value=mock_pg):
        yield mock_dist


def _count_targets(gm: fx.GraphModule, needle: str) -> int:
    """Count ``call_function`` nodes whose ``str(target)`` contains ``needle``."""
    return sum(
        1 for n in gm.graph.nodes if n.op == "call_function" and needle in str(n.target)
    )


def _reduce_ops(gm: fx.GraphModule, needle: str) -> list:
    """Return the reduce-op arg (``args[1]``) of every matching collective node."""
    return [
        n.args[1]
        for n in gm.graph.nodes
        if n.op == "call_function" and needle in str(n.target)
    ]


def _waited_rs_ar_chain(gm: fx.GraphModule) -> int:
    """Count ``all_reduce`` nodes whose input is a waited ``reduce_scatter``.

    HSDP reduce-scatters on the ``fsdp`` group and all-reduces on the
    ``dp_replicate`` group; those run on different process-group streams, so the
    all_reduce must consume a *waited* reduce_scatter output, not the raw async
    tensor (else it could read a partially written shard).
    """
    count = 0
    for node in gm.graph.nodes:
        if node.op != "call_function" or "all_reduce" not in str(node.target):
            continue
        inp = node.args[0]
        if not (isinstance(inp, fx.Node) and "wait_tensor" in str(inp.target)):
            continue
        rs_node = inp.args[0]
        if isinstance(rs_node, fx.Node) and "reduce_scatter_tensor" in str(
            rs_node.target
        ):
            count += 1
    return count


def _linear_joint_graph() -> fx.GraphModule:
    """Build a minimal joint-graph stub for an ``nn.Linear(4, 4)`` model.

    State placeholders (``weight``, ``bias``) lead, then user inputs
    (``x``, ``grad_weight``, ``grad_bias``); output is
    ``[loss, grad_weight, grad_bias]``. The loss depends on both params so
    AllGather user-rewiring is exercised. ``state_fqns`` /
    ``state_is_param`` / ``num_state_inputs`` mirror what
    ``trace_model_graph`` attaches to the real traced GraphModule.
    """
    g = fx.Graph()
    weight = g.placeholder("weight")
    bias = g.placeholder("bias")
    x = g.placeholder("x")
    mm = g.call_function(torch.matmul, args=(weight, x))
    loss = g.call_function(torch.add, args=(mm, bias))
    grad_w = g.placeholder("grad_weight")
    grad_b = g.placeholder("grad_bias")
    g.output([loss, grad_w, grad_b])
    gm = fx.GraphModule({}, g)
    gm.state_fqns = ["weight", "bias"]
    gm.state_is_param = [True, True]
    gm.num_state_inputs = 2
    return gm


class _OddParam(nn.Module):
    """Model with a single param whose dim 0 is not divisible by 2."""

    def __init__(self) -> None:
        """Initialize the non-divisible parameter ``odd``."""
        super().__init__()
        self.odd = nn.Parameter(torch.zeros(3, 4))


class _FrozenParam(nn.Module):
    """Model with one trainable and one frozen FSDP-eligible parameter."""

    def __init__(self) -> None:
        """Initialize the trainable and frozen parameters (both [4, 4])."""
        super().__init__()
        self.trainable = nn.Parameter(torch.zeros(4, 4))
        self.frozen = nn.Parameter(torch.zeros(4, 4), requires_grad=False)


def _frozen_joint_graph() -> fx.GraphModule:
    """Joint-graph stub for ``_FrozenParam``.

    Both params lead as state placeholders and both are read by the forward,
    but the frozen one has no gradient counterpart (the tracer only emits
    grads for ``requires_grad`` params) — output is ``[loss, grad_trainable]``.
    """
    g = fx.Graph()
    trainable = g.placeholder("trainable")
    frozen = g.placeholder("frozen")
    x = g.placeholder("x")
    mm = g.call_function(torch.matmul, args=(trainable, x))
    loss = g.call_function(torch.add, args=(mm, frozen))
    grad_t = g.placeholder("grad_trainable")
    g.output([loss, grad_t])
    gm = fx.GraphModule({}, g)
    gm.state_fqns = ["trainable", "frozen"]
    gm.state_is_param = [True, True]
    gm.num_state_inputs = 2
    return gm


def _odd_joint_graph() -> fx.GraphModule:
    """Joint-graph stub for ``_OddParam`` (state = ``odd`` only).

    ``odd`` is read by the forward so the pass has a live parameter to gather —
    an unused placeholder is treated as dead and skipped.
    """
    g = fx.Graph()
    odd = g.placeholder("odd")
    x = g.placeholder("x")
    mm = g.call_function(torch.matmul, args=(odd, x))
    loss = g.call_function(torch.sum, args=(mm,))
    grad_odd = g.placeholder("grad_odd")
    g.output([loss, grad_odd])
    gm = fx.GraphModule({}, g)
    gm.state_fqns = ["odd"]
    gm.state_is_param = [True]
    gm.num_state_inputs = 1
    return gm


class _MixedState(nn.Module):
    """Model with a trainable param, a frozen param, and a buffer."""

    def __init__(self) -> None:
        """Initialize trainable and frozen params plus a buffer."""
        super().__init__()
        self.trainable = nn.Parameter(torch.zeros(2, 2))
        self.frozen = nn.Parameter(torch.zeros(2, 2), requires_grad=False)
        self.register_buffer("buf", torch.zeros(2))


class _WrappedPlusRootParam(nn.Module):
    """Model with a wrap-eligible submodule plus a root-level param.

    ``lin`` is a named submodule (so a partial ``fsdp_mark("lin")`` plan
    matches it via its module FQN); ``extra`` sits directly on the root, where
    it has no module ancestor. A partial plan wrapping only ``lin`` must shard
    ``lin``'s params and leave ``extra`` full-size on BOTH the graph side and
    the live model.
    """

    def __init__(self) -> None:
        """Initialize the wrap-eligible submodule and the root-level param."""
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.extra = nn.Parameter(torch.zeros(2, 2))


def _lin_bias_extra_joint_graph() -> fx.GraphModule:
    """Joint-graph stub for ``_WrappedPlusRootParam``.

    State placeholders (``lin_weight``, ``lin_bias``, ``extra``) lead, then
    user inputs; output is ``[loss, grad_lin_weight, grad_lin_bias,
    grad_extra]``. ``state_fqns`` holds the real FQNs (with dots) that
    ``FSDPPass`` matches against the plan; the placeholder names themselves
    must stay dot-free because codegen turns them into Python parameter names.
    """
    g = fx.Graph()
    lw = g.placeholder("lin_weight")
    lb = g.placeholder("lin_bias")
    extra = g.placeholder("extra")
    x = g.placeholder("x")
    mm = g.call_function(torch.matmul, args=(lw, x))
    loss = g.call_function(torch.add, args=(mm, lb))
    grad_lw = g.placeholder("grad_lin_weight")
    grad_lb = g.placeholder("grad_lin_bias")
    grad_extra = g.placeholder("grad_extra")
    g.output([loss, grad_lw, grad_lb, grad_extra])
    gm = fx.GraphModule({}, g)
    gm.state_fqns = ["lin.weight", "lin.bias", "extra"]
    gm.state_is_param = [True, True, True]
    gm.num_state_inputs = 3
    return gm


class TestFSDPPassRunGuards(unittest.TestCase):
    """``FSDPPass.run`` early-returns and raises on bad preconditions."""

    def test_run_skips_when_dist_not_initialized(self):
        """Test the pass early-returns when ``dist`` is not initialized."""
        cfg = PassConfig(fsdp_enabled=True, fsdp_degree=2)
        pas = FSDPPass()
        gm = _linear_joint_graph()
        model = nn.Linear(4, 4)
        before_w = tuple(model.weight.shape)
        before_b = tuple(model.bias.shape)

        with _patch_dist(initialized=False):
            result = pas.run(gm, cfg, model=model, fsdp_group_name="fsdp")

        self.assertIs(result, gm, "skipped run should return the same graph_module")
        self.assertEqual(
            tuple(model.weight.shape),
            before_w,
            (
                f"uninitialized dist must not shard the model, "
                f"weight shape={tuple(model.weight.shape)} expected={before_w}"
            ),
        )
        self.assertEqual(
            tuple(model.bias.shape),
            before_b,
            (
                f"uninitialized dist must not shard the model, "
                f"bias shape={tuple(model.bias.shape)} expected={before_b}"
            ),
        )
        self.assertEqual(
            _count_targets(gm, "all_gather_into_tensor"),
            0,
            "skipped run must not insert any AllGather nodes",
        )

    def test_run_skips_when_world_size_one(self):
        """Test the pass early-returns when ``world_size == 1``."""
        cfg = PassConfig(fsdp_enabled=True, fsdp_degree=2)
        pas = FSDPPass()
        gm = _linear_joint_graph()
        model = nn.Linear(4, 4)
        before_w = tuple(model.weight.shape)

        with _patch_dist(world_size=1, initialized=True):
            result = pas.run(gm, cfg, model=model, fsdp_group_name="fsdp")

        self.assertIs(result, gm, "single-card run should return the same graph_module")
        self.assertEqual(
            tuple(model.weight.shape),
            before_w,
            (
                f"world_size=1 must not shard the model, "
                f"weight shape={tuple(model.weight.shape)} expected={before_w}"
            ),
        )
        self.assertEqual(
            _count_targets(gm, "all_gather_into_tensor"),
            0,
            "single-card run must not insert any AllGather nodes",
        )

    def test_run_raises_when_model_kwarg_missing(self):
        """Test the pass raises ``ValueError`` when ``model=`` is not passed.

        The pass needs the live model to shard parameters in place; omitting
        it must fail loudly rather than silently doing nothing.
        """
        cfg = PassConfig(fsdp_enabled=True, fsdp_degree=2)
        pas = FSDPPass()
        gm = _linear_joint_graph()

        with _patch_dist(world_size=2, initialized=True):
            with self.assertRaises(ValueError) as ctx:
                pas.run(gm, cfg, fsdp_group_name="fsdp")

        self.assertIn(
            "model",
            str(ctx.exception),
            f"error should mention the missing model kwarg, got: {ctx.exception}",
        )


class TestFSDPPassRunSharding(unittest.TestCase):
    """``FSDPPass.run`` shards the live model and inserts collectives."""

    def test_run_shards_params_and_inserts_all_gather_and_reduce_scatter(self):
        """Test the happy path: shards params + inserts collectives."""
        cfg = PassConfig(fsdp_enabled=True, fsdp_degree=2)
        pas = FSDPPass()
        gm = _linear_joint_graph()
        model = nn.Linear(4, 4)

        with _patch_dist(world_size=2, rank=0, initialized=True):
            result = pas.run(gm, cfg, model=model, fsdp_group_name="fsdp")

        self.assertIs(result, gm, "run() should return the same graph_module")
        # Live model sharded in place (dim 0, rank 0 of a 2-way shard).
        self.assertEqual(
            tuple(model.weight.shape),
            (2, 4),
            (
                f"weight [4,4] should shard to (2,4) with fsdp_degree=2, "
                f"got {tuple(model.weight.shape)}"
            ),
        )
        self.assertEqual(
            tuple(model.bias.shape),
            (2,),
            (
                f"bias [4] should shard to (2,) with fsdp_degree=2, "
                f"got {tuple(model.bias.shape)}"
            ),
        )
        # AllGather (Shard->Replicate) on each sharded param placeholder.
        self.assertEqual(
            _count_targets(gm, "all_gather_into_tensor"),
            2,
            (
                f"expected 2 AllGather nodes (weight, bias), "
                f"got {_count_targets(gm, 'all_gather_into_tensor')}"
            ),
        )
        # ReduceScatter (Replicate->Shard) on each sharded param's grad.
        self.assertEqual(
            _count_targets(gm, "reduce_scatter_tensor"),
            2,
            (
                f"expected 2 ReduceScatter nodes (grad_weight, grad_bias), "
                f"got {_count_targets(gm, 'reduce_scatter_tensor')}"
            ),
        )
        # One wait after each AllGather and one after each ReduceScatter.
        self.assertEqual(
            _count_targets(gm, "wait_tensor"),
            4,
            (
                f"expected 4 wait_tensor nodes (2 ag + 2 rs), "
                f"got {_count_targets(gm, 'wait_tensor')}"
            ),
        )
        # Gradient reduction averages over the FSDP axis, matching eager
        # ``fully_shard`` (never sums).
        self.assertEqual(
            set(_reduce_ops(gm, "reduce_scatter_tensor")),
            {"avg"},
            "FSDP grads must reduce_scatter with 'avg', not 'sum'",
        )

    def test_run_skips_frozen_params(self):
        """Test frozen params are not all-gathered or sharded.

        The tracer omits frozen params from the gradient list and
        ``_shard_live_model_params`` leaves them full-size, so
        ``_identify_params_in_fsdp_modules`` must skip them too. Otherwise the
        graph would all_gather a full tensor (treating it as a shard) while the
        live model still holds the full shape.
        """
        cfg = PassConfig(fsdp_enabled=True, fsdp_degree=2)
        pas = FSDPPass()
        gm = _frozen_joint_graph()
        model = _FrozenParam()
        before_frozen = tuple(model.frozen.shape)

        with _patch_dist(world_size=2, rank=0, initialized=True):
            result = pas.run(gm, cfg, model=model, fsdp_group_name="fsdp")

        self.assertIs(result, gm, "run() should return the same graph_module")
        # Trainable param sharded, frozen param left full-size in the model.
        self.assertEqual(tuple(model.trainable.shape), (2, 4))
        self.assertEqual(
            tuple(model.frozen.shape),
            before_frozen,
            (
                f"frozen param must stay full-size, "
                f"got {tuple(model.frozen.shape)} expected {before_frozen}"
            ),
        )
        # Only the trainable param is gathered / reduced.
        self.assertEqual(
            _count_targets(gm, "all_gather_into_tensor"),
            1,
            (
                f"only the trainable param should all_gather, "
                f"got {_count_targets(gm, 'all_gather_into_tensor')}"
            ),
        )
        self.assertEqual(_count_targets(gm, "reduce_scatter_tensor"), 1)

    def test_run_pad_shards_non_divisible_params(self):
        """Test params whose dim 0 is not divisible by ``fsdp_degree`` are pad-sharded.

        Mirrors eager ``fully_shard``: dim 0 is zero-padded to a multiple of the
        degree so the parameter is still sharded (uniform shard size) and
        gradient-reduced, instead of being left replicated. The graph gathers the
        padded shard, narrows the gathered tensor back to the real size, and pads
        the gradient back to the padded size before the reduce_scatter.
        """
        cfg = PassConfig(fsdp_enabled=True, fsdp_degree=2)
        pas = FSDPPass()
        gm = _odd_joint_graph()
        model = _OddParam()
        self.assertEqual(tuple(model.odd.shape), (3, 4))

        with _patch_dist(world_size=2, rank=0, initialized=True):
            result = pas.run(gm, cfg, model=model, fsdp_group_name="fsdp")

        self.assertIs(result, gm, "run() should return the same graph_module")
        # 3 rows, degree 2 -> padded to 4 -> this rank owns a uniform 2-row shard.
        self.assertEqual(
            tuple(model.odd.shape),
            (2, 4),
            (
                "non-divisible param must be pad-sharded to a uniform shard, "
                f"got shape={tuple(model.odd.shape)}"
            ),
        )
        self.assertEqual(
            _count_targets(gm, "all_gather_into_tensor"),
            1,
            "pad-sharded param must get one AllGather of the padded shard",
        )
        self.assertEqual(
            _count_targets(gm, "reduce_scatter_tensor"),
            1,
            "pad-sharded param must get one ReduceScatter",
        )
        narrows = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and "narrow" in str(n.target)
        ]
        self.assertEqual(
            len(narrows), 1, "the padded full tensor must be narrowed back once"
        )
        self.assertEqual(
            tuple(narrows[0].args[1:]),
            (0, 0, 3),
            "the narrow must restore the real dim 0 (3 rows)",
        )
        self.assertEqual(
            _count_targets(gm, "constant_pad_nd"),
            1,
            "the [3, 4] gradient must be padded back to [4, 4] before reduction",
        )

    def test_run_rejects_non_divisible_expert_param(self):
        """Do not silently replicate an unsynchronized trainable expert."""
        cfg = PassConfig(fsdp_enabled=True, fsdp_degree=4)
        for shape in ((2, 4), ()):
            with self.subTest(shape=shape):
                gm = _odd_joint_graph()
                model = _OddParam()
                model.odd = nn.Parameter(torch.zeros(shape))
                with _patch_dist(world_size=4, rank=0, initialized=True):
                    with self.assertRaisesRegex(ValueError, "edp_shard degree 4"):
                        FSDPPass().run(
                            gm, cfg, model=model, fsdp_group_name="fsdp",
                            expert_param_names={"odd"}, expert_fsdp_group_name="fsdp_expert",
                        )
                self.assertEqual(tuple(model.odd.shape), shape)
                self.assertEqual(_count_targets(gm, "all_gather_into_tensor"), 0)

    def test_expert_selection_skips_buffers_and_frozen_params(self):
        """Keep mixed state intact, including traces without parameter flags."""
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                model = _FrozenParam()
                model.frozen = nn.Parameter(torch.ones(3), requires_grad=False)
                model.register_buffer("buf", torch.ones(3))
                gm = _frozen_joint_graph()
                with gm.graph.inserting_before(next(node for node in gm.graph.nodes if node.name == "x")):
                    gm.graph.placeholder("buf")
                gm.state_fqns.append("buf")
                gm.state_is_param.append(False)
                gm.num_state_inputs += 1
                if legacy:
                    del gm.state_is_param
                with _patch_dist(world_size=2):
                    FSDPPass().run(
                        gm, PassConfig(fsdp_degree=2), model=model,
                        expert_param_names={"frozen", "buf"},
                    )
                self.assertEqual(tuple(model.trainable.shape), (2, 4))
                torch.testing.assert_close(model.frozen, torch.ones(3), atol=0, rtol=0)
                torch.testing.assert_close(model.buf, torch.ones(3), atol=0, rtol=0)
                self.assertEqual(_count_targets(gm, "all_gather_into_tensor"), 1)
                self.assertEqual(_count_targets(gm, "reduce_scatter_tensor"), 1)

    def test_expert_group_with_dense_padding_and_scalar_reduction(self):
        """Keep dense padding/scalar sync while experts shard or stay local."""
        for degree in (1, 2):
            with self.subTest(edp_degree=degree):
                model = _WrappedPlusRootParam()
                model.lin.weight.data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
                model.lin.bias = nn.Parameter(torch.arange(3, dtype=torch.float32))
                model.extra = nn.Parameter(torch.tensor(2.0))
                weight = torch.detach(model.lin.weight).clone()
                gm = _lin_bias_extra_joint_graph()
                with _patch_dist(world_size=4) as mocked, patch(_RESOLVE_PG_PATH, side_effect=lambda name: name):
                    mocked.get_world_size.side_effect = (
                        lambda group=None, edp_degree=degree: edp_degree if group == "edp" else 4
                    )
                    mocked.get_rank.side_effect = (
                        lambda group=None, edp_degree=degree: edp_degree - 1 if group == "edp" else 3
                    )
                    FSDPPass().run(
                        gm, PassConfig(fsdp_degree=4), model=model, fsdp_group_name="dense",
                        expert_param_names={"lin.weight"}, expert_fsdp_group_name="edp",
                    )
                expected_weight = weight[2:] if degree > 1 else weight
                torch.testing.assert_close(model.lin.weight, expected_weight, atol=0, rtol=0)
                torch.testing.assert_close(model.lin.bias, torch.zeros(1), atol=0, rtol=0)
                self.assertEqual(model.extra.ndim, 0)
                gathers = {node.meta["param_name"]: node.args[1:] for node in gm.graph.nodes
                           if node.meta.get("comm_type") == "fsdp_all_gather"}
                expected_gathers = {"lin.bias": (4, "dense")}
                if degree > 1:
                    expected_gathers["lin.weight"] = (degree, "edp")
                self.assertEqual(gathers, expected_gathers)
                self.assertEqual(_count_targets(gm, "constant_pad_nd"), 1)
                self.assertEqual(_count_targets(gm, "narrow"), 1)
                reductions = [node for node in gm.graph.nodes if node.meta.get("comm_type") == "fsdp_reduce_scatter"]
                self.assertEqual({node.args[2:] for node in reductions}, set(expected_gathers.values()))
                self.assertTrue(all(node.args[0].target == "contiguous" for node in reductions))
                replicated = [node for node in gm.graph.nodes if node.meta.get("comm_type") == "fsdp_all_reduce"]
                self.assertEqual([node.args[2] for node in replicated], ["dense"])
                groups = {node.meta["comm_group"] for node in gm.graph.nodes if "comm_group" in node.meta}
                self.assertEqual(groups, {"dense", "edp"} if degree > 1 else {"dense"})

    def test_unmarked_expert_fallback_uses_edp_group(self):
        """A partial-plan expert fallback must never reduce across dense/EP peers."""
        for degree in (1, 2):
            with self.subTest(edp_degree=degree):
                model = _WrappedPlusRootParam()
                gm = _lin_bias_extra_joint_graph()
                plan = GraphParallelPlan().fsdp_mark("lin")
                with _patch_dist(world_size=4) as mocked, patch(_RESOLVE_PG_PATH, side_effect=lambda name: name):
                    mocked.get_world_size.side_effect = (
                        lambda group=None, edp_degree=degree: edp_degree if group == "edp" else 4
                    )
                    FSDPPass().run(
                        gm, PassConfig(fsdp_degree=4), model=model, parallel_plan=plan,
                        fsdp_group_name="dense", expert_param_names={"extra"}, expert_fsdp_group_name="edp",
                    )
                self.assertEqual(tuple(model.extra.shape), (2, 2))
                replicated = [node for node in gm.graph.nodes if node.meta.get("comm_type") == "fsdp_all_reduce"]
                self.assertEqual([node.args[2] for node in replicated], ["edp"] if degree > 1 else [])

    def test_run_uses_fsdp_degree_from_config_not_world_size(self):
        """Test ``fsdp_degree`` from the config wins over ``world_size``.

        Essential for a TP+FSDP hybrid, where the FSDP group is a proper
        sub-group of the world: sharding by ``world_size`` would over-shard
        along the TP axis. Here ``world_size=4`` but ``fsdp_degree=2`` -> the
        param is split in 2, not 4.
        """
        cfg = PassConfig(fsdp_enabled=True, fsdp_degree=2)
        pas = FSDPPass()
        gm = _linear_joint_graph()
        model = nn.Linear(4, 4)

        with _patch_dist(world_size=4, rank=0, initialized=True):
            pas.run(gm, cfg, model=model, fsdp_group_name="fsdp")

        self.assertEqual(
            tuple(model.weight.shape),
            (2, 4),
            (
                f"config fsdp_degree=2 must win over world_size=4 -> (2,4), "
                f"got {tuple(model.weight.shape)} (world_size path would be (1,4))"
            ),
        )

    def test_partial_plan_shards_only_wrapped_params(self):
        """Test a partial plan shards wrapped params and DDP-falls-back the rest.

        The plan-gate must agree between graph and live model: only ``lin``'s
        params (inside an FSDP-wrapped module) get AllGather/ReduceScatter and
        are physically sharded. The root-level ``extra`` param is unmarked, so it
        stays full-size on both sides and its gradient is instead all-reduced
        over the fsdp group (DDP fallback) — sharding it live while the graph
        holds it replicated (no AllGather) would mismatch at ``run_traced_graph``
        time.
        """
        cfg = PassConfig(fsdp_enabled=True, fsdp_degree=2)
        plan = GraphParallelPlan().fsdp_mark("lin")
        pas = FSDPPass(parallel_plan=plan)
        gm = _lin_bias_extra_joint_graph()
        model = _WrappedPlusRootParam()
        before_extra = tuple(model.extra.shape)

        with _patch_dist(world_size=2, rank=0, initialized=True):
            # parallel_plan auto-filled from the pass (not kwargs), as the
            # pipeline would do.
            result = pas.run(gm, cfg, model=model, fsdp_group_name="fsdp")

        self.assertIs(result, gm, "run() should return the same graph_module")
        # Wrapped submodule sharded in place (dim 0, rank 0 of a 2-way shard)...
        self.assertEqual(
            tuple(model.lin.weight.shape),
            (2, 4),
            (
                f"lin.weight [4,4] should shard to (2,4), "
                f"got {tuple(model.lin.weight.shape)}"
            ),
        )
        self.assertEqual(
            tuple(model.lin.bias.shape),
            (2,),
            (f"lin.bias [4] should shard to (2,), got {tuple(model.lin.bias.shape)}"),
        )
        # ...while the unwrapped root-level param stays replicated.
        self.assertEqual(
            tuple(model.extra.shape),
            before_extra,
            (
                f"extra is not in an FSDP-wrapped module -> must stay full-size, "
                f"got {tuple(model.extra.shape)} expected {before_extra}"
            ),
        )
        # Graph side: exactly the 2 wrapped params get AllGather; extra gets none.
        self.assertEqual(
            _count_targets(gm, "all_gather_into_tensor"),
            2,
            (
                f"only lin.weight/lin.bias should all_gather, "
                f"got {_count_targets(gm, 'all_gather_into_tensor')}"
            ),
        )
        self.assertEqual(
            _count_targets(gm, "reduce_scatter_tensor"),
            2,
            (
                f"only lin.weight/lin.bias grads should reduce_scatter, "
                f"got {_count_targets(gm, 'reduce_scatter_tensor')}"
            ),
        )
        # The unmarked root-level param falls back to DDP: its gradient is
        # all-reduced over the fsdp group (no shard, no gather).
        fallback_ar = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and "all_reduce" in str(n.target)
        ]
        self.assertEqual(
            len(fallback_ar),
            1,
            "unmarked 'extra' must get exactly one fallback all_reduce",
        )
        self.assertEqual(
            fallback_ar[0].meta.get("comm_group"),
            "fsdp",
            "the DDP fallback must reduce over the fsdp group in fsdp mode",
        )


class TestFSDPPassDataParallelModes(unittest.TestCase):
    """DDP (replicate) and HSDP (hybrid_shard) semantics.

    Mirrors simplefsdp's ``data_parallel`` modes: DDP keeps parameters
    replicated and all-reduces gradients; HSDP shards on the ``fsdp`` axis and
    reduce-scatters then all-reduces (average across both axes).
    """

    def test_ddp_replicates_params_and_all_reduces_grads(self):
        """Test DDP: no param sharding / all_gather, grads all-reduced."""
        cfg = PassConfig(fsdp_enabled=True, dp_mode="ddp", dp_replicate_degree=2)
        pas = FSDPPass()
        gm = _linear_joint_graph()
        model = nn.Linear(4, 4)

        with _patch_dist(world_size=2, rank=0, initialized=True):
            result = pas.run(
                gm,
                cfg,
                model=model,
                fsdp_group_name="fsdp",
                dp_replicate_group_name="dp_replicate",
            )

        self.assertIs(result, gm, "run() should return the same graph_module")
        # Parameters stay replicated (no shard axis under DDP).
        self.assertEqual(tuple(model.weight.shape), (4, 4))
        self.assertEqual(tuple(model.bias.shape), (4,))
        self.assertEqual(_count_targets(gm, "all_gather_into_tensor"), 0)
        self.assertEqual(_count_targets(gm, "reduce_scatter_tensor"), 0)
        # One all_reduce (+ wait) per gradient.
        self.assertEqual(_count_targets(gm, "all_reduce"), 2)
        self.assertEqual(_count_targets(gm, "wait_tensor"), 2)
        # DDP grads average over the replicate axis (matching torch DDP).
        self.assertEqual(
            set(_reduce_ops(gm, "all_reduce")),
            {"avg"},
            "DDP grads must all_reduce with 'avg', not 'sum'",
        )

    def test_hsdp_shards_params_and_reduces_both_axes(self):
        """Test HSDP: params shard on fsdp, grads reduce_scatter + all_reduce."""
        cfg = PassConfig(
            fsdp_enabled=True,
            dp_mode="hsdp",
            fsdp_degree=2,
            dp_replicate_degree=2,
        )
        pas = FSDPPass()
        gm = _linear_joint_graph()
        model = nn.Linear(4, 4)

        with _patch_dist(world_size=4, rank=0, initialized=True):
            result = pas.run(
                gm,
                cfg,
                model=model,
                fsdp_group_name="fsdp",
                dp_replicate_group_name="dp_replicate",
            )

        self.assertIs(result, gm, "run() should return the same graph_module")
        # Sharded on the fsdp axis only (replicate axis keeps the same shard).
        self.assertEqual(tuple(model.weight.shape), (2, 4))
        self.assertEqual(tuple(model.bias.shape), (2,))
        self.assertEqual(_count_targets(gm, "all_gather_into_tensor"), 2)
        self.assertEqual(_count_targets(gm, "reduce_scatter_tensor"), 2)
        # One all_reduce per gradient on top of the reduce_scatter.
        self.assertEqual(_count_targets(gm, "all_reduce"), 2)
        # 2 all_gather waits + 2 reduce_scatter waits + 2 all_reduce waits.
        self.assertEqual(_count_targets(gm, "wait_tensor"), 6)
        # Both axes average, so the step gradient is the mean over
        # fsdp_degree * dp_replicate_degree ranks.
        self.assertEqual(set(_reduce_ops(gm, "reduce_scatter_tensor")), {"avg"})
        self.assertEqual(set(_reduce_ops(gm, "all_reduce")), {"avg"})
        # HSDP's reduce_scatter (fsdp) and all_reduce (dp_replicate) run on
        # different streams: the all_reduce must consume a waited reduce_scatter
        # output, otherwise it can race the still-in-flight shard.
        self.assertEqual(
            _waited_rs_ar_chain(gm),
            2,
            "HSDP all_reduce must consume a waited reduce_scatter output",
        )


class TestFSDPPassHelpers(unittest.TestCase):
    """Pure helpers: FQN matching, parent extraction, trainable-state filter."""

    def test_param_belongs_to_fsdp_module_exact_and_pattern(self):
        """Test exact-FQN and wildcard-pattern matching of ancestor modules."""
        exact = FSDPPass(parallel_plan=GraphParallelPlan().fsdp_mark("lin"))
        self.assertTrue(
            exact._param_belongs_to_fsdp_module("lin.weight"),  # pylint: disable=protected-access
            "exact fsdp_mark('lin') should match a param under module 'lin'",
        )
        self.assertFalse(
            exact._param_belongs_to_fsdp_module("other.weight"),  # pylint: disable=protected-access
            "exact wrap must not match an unrelated module",
        )

        pattern = FSDPPass(
            parallel_plan=GraphParallelPlan().fsdp_mark_pattern("layers.*")
        )
        self.assertTrue(
            pattern._param_belongs_to_fsdp_module("layers.0.lin.weight"),  # pylint: disable=protected-access
            "pattern 'layers.*' should match an ancestor like 'layers.0.lin'",
        )
        self.assertFalse(
            pattern._param_belongs_to_fsdp_module("lin.weight"),  # pylint: disable=protected-access
            "pattern 'layers.*' must not match module 'lin'",
        )

    def test_param_belongs_to_fsdp_module_top_level_param_matches(self):
        """Test a top-level (no module ancestor) param is matched by its own name.

        ``_param_belongs_to_fsdp_module`` tests the param FQN itself before
        walking ancestors, so a param directly on the root (``weight``) matches
        a ``*`` plan or an explicit ``fsdp_mark("weight")``.
        """
        pas = FSDPPass(parallel_plan=GraphParallelPlan().fsdp_mark_pattern("*"))
        self.assertTrue(
            pas._param_belongs_to_fsdp_module("weight"),  # pylint: disable=protected-access
            "top-level param 'weight' should match a '*' plan via its own FQN",
        )

    def test_get_parent_module_fqn(self):
        """Test the parent-module FQN is the param FQN minus the leaf."""
        pas = FSDPPass()
        self.assertEqual(
            pas._get_parent_module_fqn("layers.0.attention.wq.weight"),  # pylint: disable=protected-access
            "layers.0.attention.wq",
        )
        # A bare param name (no dot) is returned unchanged.
        self.assertEqual(
            pas._get_parent_module_fqn("weight"),  # pylint: disable=protected-access
            "weight",
        )

    def test_build_trainable_state_indices_skips_buffers_and_frozen(self):
        """Test the trainable-state filter mirrors the tracer's params list."""
        pas = FSDPPass()
        model = _MixedState()
        state_fqns = ["trainable", "frozen", "buf"]
        state_is_param = [True, True, False]

        result = pas._build_trainable_state_indices(  # pylint: disable=protected-access
            state_fqns, num_state_inputs=3, state_is_param=state_is_param, model=model
        )

        self.assertEqual(
            result,
            [0],
            (f"only the trainable param (state idx 0) should be kept, got {result}"),
        )

    def test_build_trainable_state_indices_fallback_without_model(self):
        """Test the all-trainable fallback when the model is unavailable.

        Without the model ``requires_grad`` cannot be inspected; the caller's
        grad-count check raises if the assumption is wrong, so misalignment is
        not silent.
        """
        pas = FSDPPass()
        result = pas._build_trainable_state_indices(  # pylint: disable=protected-access
            state_fqns=["a", "b", "c"],
            num_state_inputs=3,
            state_is_param=None,
            model=None,
        )
        self.assertEqual(
            result,
            [0, 1, 2],
            f"fallback should return all state indices, got {result}",
        )


if __name__ == "__main__":
    unittest.main()
