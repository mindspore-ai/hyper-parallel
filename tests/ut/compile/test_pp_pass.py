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
"""Unit tests for ``hyper_parallel.compile.passes.parallel.pp_pass.PpPass``.

Covers the pipeline-parallel graph-split contract on hand-built joint
graphs (same style as ``test_fsdp_pass`` — no real tracing, no real
communication):

``run`` guards:
1. Early-returns when distributed is not initialized or ``world_size == 1``.
2. Raises ``ValueError`` when the caller omits the live ``model=`` kwarg.
3. Rejects the PP+FSDP hybrid without a mesh (v1 is pure-PP only).

Stage split (2-stage mini LLM, manual plan):
4. Stage 0: state/grads sliced to its params, fwd subgraph ends at the
   boundary activation, live model prunes foreign submodules, schedule
   stub installed with ``call_module``.
5. Stage 1: symmetric — ``pp_act_in_0`` placeholder, trailing boundary grad
   in bwd outputs, opposite submodules pruned.
6. Boundary values: extra crossing tensors ship as additional boundary
   values (multi-value cuts are supported).

Auto split:
7. ``_auto_stage_split``: even fallback split, layer-container split with
   input/output-side weighting, too-few-children error.

Plan / YAML:
8. ``PassPlan.pp_stage`` wildcard + negative-index rejection; manual plan
   length validation; YAML ``pp:`` section parsing.

Schedule driver (mocked subgraphs + mocked dist):
9. ``ScheduleGPipe.forward`` microbatch loop: grad averaging, async send
   bookkeeping, receive buffers sized from recv_spec.

Review follow-ups:
10. Skip-stage dataflow (values crossing 2+ cuts) is rejected up front
    with an actionable message instead of a generic slice-copy failure.
11. Partial manual plans warn about modules no stage declares; ancestor
    containers of declared elements do not.
12. ``_propagate_anchor_stages`` pulls transitive bwd chains in one
    reverse-topological pass.
"""

import logging
import os
import unittest
import warnings
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch
from torch import fx, nn

from tests.common.mark_utils import arg_mark  # pylint: disable=C0413

from hyper_parallel.compile.parallel_config import PassConfig  # pylint: disable=C0413
from hyper_parallel.compile.passes.parallel.pp_pass import (  # pylint: disable=C0413
    PpPass,
    _auto_stage_split,
)
from hyper_parallel.compile.passes.parallel.pp_schedule import (  # pylint: disable=C0413
    ScheduleGPipe,
)
from hyper_parallel.compile.sharding_config import PassPlan  # pylint: disable=C0413
from hyper_parallel.compile.tracer.graph_tracer import (  # pylint: disable=C0413
    run_traced_graph,
    trace_model_graph,
)

_PP_DIST_PATH = "hyper_parallel.compile.passes.parallel.pp_pass.dist"
_SCHED_DIST_PATH = "hyper_parallel.compile.passes.parallel.pp_schedule.dist"

_STAGE_FQNS = ["embed", "lin0", "lin1", "head"]
_ALL_STATE = [f"{m}.{p}" for m in _STAGE_FQNS for p in ("weight", "bias")]


@contextmanager
def _patch_dist(
    world_size: int = 2, rank: int = 0, initialized: bool = True
) -> Iterator[MagicMock]:
    """Patch ``dist`` inside ``pp_pass`` with a stub returning given values."""
    mock_dist = MagicMock()
    mock_dist.is_initialized.return_value = initialized
    mock_dist.get_world_size.return_value = world_size
    mock_dist.get_rank.return_value = rank
    mock_dist.new_group.return_value = MagicMock()
    with patch(_PP_DIST_PATH, mock_dist):
        yield mock_dist


def _stack(*fqns: str) -> dict:
    """``nn_module_stack`` meta: root first, innermost module last."""
    stack = {"": ("", "torch.nn.Module")}
    for fqn in fqns:
        stack[f"k_{fqn}"] = (fqn, f"mod.{fqn}")
    return stack


def _mini_llm_joint_graph(extra_cross: bool = False) -> fx.GraphModule:
    """Joint fwd+bwd graph for a 4-module mini LLM (2-stage split target).

    Forward: x -> embed -> lin0 -> lin1 -> head -> loss (all ``[4, 4]``
    matmul/add chains; op semantics are irrelevant — the pass only does
    graph surgery). Backward nodes carry ``autograd_backward`` and the
    forward's module stack. The single crossing values are ``h1`` (fwd,
    stage0 -> stage1) and ``g_h1`` (bwd, stage1 -> stage0).
    """
    g = fx.Graph()
    ew, eb = g.placeholder("embed_weight"), g.placeholder("embed_bias")
    l0w, l0b = g.placeholder("lin0_weight"), g.placeholder("lin0_bias")
    l1w, l1b = g.placeholder("lin1_weight"), g.placeholder("lin1_bias")
    hw, hb = g.placeholder("head_weight"), g.placeholder("head_bias")
    x = g.placeholder("x")
    # Label placeholder: required by the (state..., input, label) contract
    # even though nothing consumes it in this mini graph.
    g.placeholder("y")

    seq = [0]

    def fwd(target, args, *modules):  # pylint: disable=C9006,C9007
        node = g.call_function(target, args)
        node.meta["nn_module_stack"] = _stack(*modules)
        node.meta["seq_nr"] = seq[0]
        seq[0] += 1
        return node

    def bwd(target, args, seq_nr, *modules):  # pylint: disable=C9006,C9007
        node = g.call_function(target, args)
        node.meta["autograd_backward"] = True
        node.meta["nn_module_stack"] = _stack(*modules)
        # A bwd node carries the seq_nr of the fwd node it differentiates
        # (the same mapping real autograd tracing produces).
        node.meta["seq_nr"] = seq_nr
        return node

    # ---- forward ----
    h0a = fwd(torch.matmul, (x, ew), "embed")
    seq_embed = h0a.meta["seq_nr"]
    h0 = fwd(torch.add, (h0a, eb), "embed")
    h1a = fwd(torch.matmul, (h0, l0w), "lin0")
    seq_lin0 = h1a.meta["seq_nr"]
    h1 = fwd(torch.add, (h1a, l0b), "lin0")
    h2a = fwd(torch.matmul, (h1, l1w), "lin1")
    seq_lin1 = h2a.meta["seq_nr"]
    h2 = fwd(torch.add, (h2a, l1b), "lin1")
    if extra_cross:
        # Second stage-0 -> stage-1 crossing: ships as a second boundary
        # value (multi-value cuts are supported).
        h2 = fwd(torch.add, (h2, h0), "lin1")
    la = fwd(torch.matmul, (h2, hw), "head")
    seq_head = la.meta["seq_nr"]
    logits = fwd(torch.add, (la, hb), "head")
    loss = g.call_function(torch.sum, (logits,))
    loss.meta["nn_module_stack"] = _stack()
    seq_loss = loss.meta["seq_nr"] = 100

    # ---- backward ----
    g_logits = bwd(torch.ones_like, (logits,), seq_loss)
    g_h2 = bwd(torch.matmul, (g_logits, hw), seq_head, "head")
    g_hw = bwd(torch.matmul, (h2, g_logits), seq_head, "head")
    g_hb = bwd(torch.sum, (g_logits,), seq_head, "head")
    g_h1 = bwd(torch.matmul, (g_h2, l1w), seq_lin1, "lin1")
    g_l1w = bwd(torch.matmul, (h1, g_h2), seq_lin1, "lin1")
    g_l1b = bwd(torch.sum, (g_h2,), seq_lin1, "lin1")
    g_h0 = bwd(torch.matmul, (g_h1, l0w), seq_lin0, "lin0")
    g_l0w = bwd(torch.matmul, (h0, g_h1), seq_lin0, "lin0")
    g_l0b = bwd(torch.sum, (g_h1,), seq_lin0, "lin0")
    g_ew = bwd(torch.matmul, (x, g_h0), seq_embed, "embed")
    g_eb = bwd(torch.sum, (g_h0,), seq_embed, "embed")
    h1.meta["val"] = torch.empty(4, 4)
    g_h1.meta["val"] = torch.empty(4, 4)
    if extra_cross:
        h0.meta["val"] = torch.empty(4, 4)

    g.output([loss, g_ew, g_eb, g_l0w, g_l0b, g_l1w, g_l1b, g_hw, g_hb])
    # Real traced graphs carry FakeTensor 'val' meta on every node; the
    # pass sizes P2P receive buffers from it. Mirror that here.
    for node in g.nodes:
        if node.op not in ("placeholder", "output") and "val" not in node.meta:
            node.meta["val"] = torch.empty(4, 4)
    gm = fx.GraphModule({}, g)
    gm.state_fqns = list(_ALL_STATE)
    gm.state_is_param = [True] * len(_ALL_STATE)
    gm.num_state_inputs = len(_ALL_STATE)
    return gm


class MiniLLM(nn.Module):
    """Live model matching ``_mini_llm_joint_graph`` (no layer container)."""

    def __init__(self) -> None:
        """Initialize four leaf linears: embed, lin0, lin1, head."""
        super().__init__()
        self.embed = nn.Linear(4, 4)
        self.lin0 = nn.Linear(4, 4)
        self.lin1 = nn.Linear(4, 4)
        self.head = nn.Linear(4, 4)


class ContainerModel(nn.Module):
    """LLM-shaped model: input module, layer container, output modules."""

    def __init__(self, num_layers: int = 5) -> None:
        """Initialize embedding, a ModuleList layer container, head."""
        super().__init__()
        self.tok_embeddings = nn.Linear(4, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(num_layers)])
        self.norm = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 4)


class ThreeModule(nn.Module):
    """Three leaf modules for the 3-stage skip-dataflow test."""

    def __init__(self) -> None:
        """Initialize m0/m1/m2 leaves."""
        super().__init__()
        self.m0 = nn.Linear(4, 4)
        self.m1 = nn.Linear(4, 4)
        self.m2 = nn.Linear(4, 4)


def _skip_stage_joint_graph() -> fx.GraphModule:
    """3-stage forward graph whose stage-2 add consumes stage 0's output.

    ``a`` (stage 0) feeds both ``b`` (stage 1) and the residual add on
    stage 2 — a skip connection spanning two cuts that v1 cannot ship.
    """
    g = fx.Graph()
    m0w = g.placeholder("m0_weight")
    m1w = g.placeholder("m1_weight")
    g.placeholder("m2_weight")
    x = g.placeholder("x")
    g.placeholder("y")

    def fwd(target, args, module):  # pylint: disable=C9006
        node = g.call_function(target, args)
        node.meta["nn_module_stack"] = _stack(module)
        node.meta["val"] = torch.empty(4, 4)
        return node

    a = fwd(torch.matmul, (x, m0w), "m0")
    b = fwd(torch.matmul, (a, m1w), "m1")
    c = fwd(torch.add, (b, a), "m2")
    loss = g.call_function(torch.sum, (c,))
    loss.meta["nn_module_stack"] = _stack()
    loss.meta["val"] = torch.empty(())
    g.output([loss])
    gm = fx.GraphModule({}, g)
    gm.state_fqns = ["m0.weight", "m1.weight", "m2.weight"]
    gm.state_is_param = [True] * 3
    gm.num_state_inputs = 3
    return gm


def _manual_plan() -> PassPlan:
    """Manual 2-stage plan: (embed, lin0) | (lin1, head)."""
    plan = PassPlan()
    plan.pp_stage(0, ["embed", "lin0"])
    plan.pp_stage(1, ["lin1", "head"])
    return plan


def _run_pp(gm, model, rank, cfg=None, plan=None):
    cfg = cfg or PassConfig(
        fsdp_enabled=False, pp_enabled=True, pp_degree=2, pp_microbatch_size=1
    )
    pas = PpPass(pass_plan=plan if plan is not None else _manual_plan())
    with _patch_dist(world_size=2, rank=rank):
        return pas.run(gm, cfg, model=model), gm, model, pas


class TestPpPassRunGuards(unittest.TestCase):
    """``PpPass.run`` early-returns / raises on bad preconditions."""

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_run_skips_when_dist_not_initialized(self):
        """Test the pass early-returns when ``dist`` is not initialized."""
        cfg = PassConfig(fsdp_enabled=False, pp_enabled=True, pp_degree=2)
        gm = _mini_llm_joint_graph()
        model = MiniLLM()
        with _patch_dist(initialized=False):
            result = PpPass().run(gm, cfg, model=model)
        self.assertIs(result, gm, "skipped run should return the same graph")
        self.assertEqual(
            len(gm.state_fqns),
            8,
            "skipped run must not slice state_fqns",
        )

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_run_skips_when_world_size_one(self):
        """Test the pass early-returns when ``world_size == 1``."""
        cfg = PassConfig(fsdp_enabled=False, pp_enabled=True, pp_degree=2)
        gm = _mini_llm_joint_graph()
        with _patch_dist(world_size=1, initialized=True):
            result = PpPass().run(gm, cfg, model=MiniLLM())
        self.assertIs(result, gm, "single-card run should return the same graph")

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_run_raises_when_model_kwarg_missing(self):
        """Test the pass raises when the live ``model=`` kwarg is omitted."""
        cfg = PassConfig(fsdp_enabled=False, pp_enabled=True, pp_degree=2)
        gm = _mini_llm_joint_graph()
        with _patch_dist(world_size=2, rank=0):
            with self.assertRaises(ValueError) as ctx:
                PpPass().run(gm, cfg)
        self.assertIn("model", str(ctx.exception))

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_run_rejects_fsdp_hybrid_without_mesh(self):
        """Test PP+FSDP without a mesh is rejected (1-D fallback can't host PP)."""
        cfg = PassConfig(fsdp_enabled=True, pp_enabled=True, pp_degree=2)
        gm = _mini_llm_joint_graph()
        with _patch_dist(world_size=2, rank=0):
            with self.assertRaises(ValueError) as ctx:
                PpPass().run(gm, cfg, model=MiniLLM())
        self.assertIn("mesh", str(ctx.exception))

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_run_rejects_partial_pp_degree(self):
        """Test 1 < pp_degree < world_size is rejected (no silent DP replication).

        Multiple ranks would map onto one stage while the group is the whole
        world, funneling every replica's P2P into a single peer rank.
        """
        cfg = PassConfig(fsdp_enabled=False, pp_enabled=True, pp_degree=2)
        gm = _mini_llm_joint_graph()
        with _patch_dist(world_size=4, rank=0):
            with self.assertRaises(ValueError) as ctx:
                PpPass().run(gm, cfg, model=MiniLLM())
        self.assertIn("pure-PP", str(ctx.exception))

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_run_skips_when_pp_degree_one(self):
        """Test ``pp_degree=1`` explicitly disables the pass."""
        cfg = PassConfig(fsdp_enabled=False, pp_enabled=True, pp_degree=1)
        gm = _mini_llm_joint_graph()
        with _patch_dist(world_size=3, rank=0):
            result = PpPass().run(gm, cfg, model=MiniLLM())
        self.assertIs(result, gm, "pp_degree=1 run should return the same graph")
        self.assertEqual(len(gm.state_fqns), 8, "skipped run must not slice state_fqns")


class TestPpPassStage0(unittest.TestCase):
    """Stage 0 split: slicing, pruning, schedule stub."""

    def setUp(self) -> None:
        """Run the pass as rank 0 against a fresh mini-LLM graph."""
        self.gm = _mini_llm_joint_graph()
        self.model = MiniLLM()
        _run_pp(self.gm, self.model, rank=0)

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_state_sliced_to_stage0(self):
        """Test ``state_fqns`` is updated IN PLACE to stage 0's params."""
        self.assertEqual(
            self.gm.state_fqns,
            ["embed.weight", "embed.bias", "lin0.weight", "lin0.bias"],
            f"stage 0 owns embed+lin0 state, got {self.gm.state_fqns}",
        )
        self.assertEqual(self.gm.num_state_inputs, 4)

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_foreign_submodules_pruned(self):
        """Test lin1/head are pruned from the live model, stage modules kept."""
        self.assertIsNone(self.model.lin1, "foreign stage-1 module must be pruned")
        self.assertIsNone(self.model.head, "foreign stage-1 module must be pruned")
        self.assertIsNotNone(self.model.embed, "stage-0 module must survive")
        self.assertIsNotNone(self.model.lin0, "stage-0 module must survive")

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_schedule_stub_installed(self):
        """Test the stub graph dispatches via a call_module node."""
        sched = self.gm.pp_schedule
        self.assertIsInstance(sched, ScheduleGPipe)
        self.assertEqual(sched.stage_idx, 0)
        self.assertEqual(sched.num_state, 4)
        self.assertEqual(sched.num_trainable, 4)
        module_calls = [n for n in self.gm.graph.nodes if n.op == "call_module"]
        self.assertEqual(
            len(module_calls),
            1,
            f"stub must have exactly one call_module, got {module_calls}",
        )
        self.assertEqual(module_calls[0].target, "pp_schedule")

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_fwd_subgraph_ends_at_boundary(self):
        """Test the fwd subgraph computes stage-0 modules and emits h1 first."""
        fwd_g = self.gm.pp_schedule.fwd_gm.graph
        out = next(n for n in fwd_g.nodes if n.op == "output")
        self.assertEqual(
            out.args[0][0].target,
            torch.add,
            "first fwd output must be the boundary activation (h1 = add)",
        )
        # 4 outputs: h1 + saved (lin0.weight round-trip, h0, x)
        self.assertEqual(len(out.args[0]), 4)
        # No stage-1 ops leaked in: lin1's matmul would reference l1w which
        # is not a stage-0 placeholder.
        placeholder_names = [n.name for n in fwd_g.nodes if n.op == "placeholder"]
        self.assertNotIn("lin1_weight", placeholder_names)

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_bwd_subgraph_outputs_only_stage_grads(self):
        """Test bwd outputs are exactly stage 0's 4 param grads (no trailing)."""
        bwd_g = self.gm.pp_schedule.bwd_gm.graph
        out = next(n for n in bwd_g.nodes if n.op == "output")
        self.assertEqual(
            len(out.args[0]),
            4,
            f"stage 0 has 4 trainable params, got {len(out.args[0])} outputs",
        )


class TestPpPassStage1(unittest.TestCase):
    """Stage 1 split: recv placeholder, trailing boundary grad, pruning."""

    def setUp(self) -> None:
        """Run the pass as rank 1 against a fresh mini-LLM graph."""
        self.gm = _mini_llm_joint_graph()
        self.model = MiniLLM()
        _run_pp(self.gm, self.model, rank=1)

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_state_sliced_to_stage1(self):
        """Test ``state_fqns`` is updated IN PLACE to stage 1's params."""
        self.assertEqual(
            self.gm.state_fqns,
            ["lin1.weight", "lin1.bias", "head.weight", "head.bias"],
            f"stage 1 owns lin1+head state, got {self.gm.state_fqns}",
        )

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_foreign_submodules_pruned(self):
        """Test embed/lin0 are pruned, stage-1 modules kept."""
        self.assertIsNone(self.model.embed)
        self.assertIsNone(self.model.lin0)
        self.assertIsNotNone(self.model.lin1)
        self.assertIsNotNone(self.model.head)

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_fwd_takes_act_in_placeholder(self):
        """Test the fwd subgraph receives the boundary activation."""
        sched = self.gm.pp_schedule
        self.assertEqual(sched.stage_idx, 1)
        self.assertEqual(
            sched.recv_spec,
            [("tensor", (4, 4), torch.float32, torch.device("cpu"))],
            "stage 1 needs recv-buffer metadata for the boundary activation",
        )
        fwd_g = sched.fwd_gm.graph
        ph_names = [n.name for n in fwd_g.nodes if n.op == "placeholder"]
        self.assertIn("pp_act_in_0", ph_names)
        out = next(n for n in fwd_g.nodes if n.op == "output")
        self.assertEqual(
            out.args[0][0].target,
            torch.sum,
            "first fwd output on the last stage must be the loss",
        )

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_bwd_emits_trailing_boundary_grad(self):
        """Test bwd outputs carry the boundary gradient for stage 0."""
        sched = self.gm.pp_schedule
        bwd_g = sched.bwd_gm.graph
        out = next(n for n in bwd_g.nodes if n.op == "output")
        self.assertEqual(
            len(out.args[0]),
            5,
            "4 param grads + 1 trailing boundary gradient",
        )
        ph_names = [n.name for n in bwd_g.nodes if n.op == "placeholder"]
        # Stage 1 is the last stage: its single gradient slot is the
        # ones_like(loss) placeholder instead of a received gradient.
        self.assertIn("pp_grad_ones", ph_names)


class TestPpPassErrors(unittest.TestCase):
    """Boundary and plan validation errors."""

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_multiple_crossing_tensors_supported(self):
        """Test extra crossing values ship as additional boundary outputs."""
        gm = _mini_llm_joint_graph(extra_cross=True)
        _run_pp(gm, MiniLLM(), rank=0)
        sched = gm.pp_schedule
        # Stage 0 now ships h1 AND h0 (2 fwd boundary values).
        self.assertEqual(sched.num_send, 2)
        fwd_g = sched.fwd_gm.graph
        out = next(n for n in fwd_g.nodes if n.op == "output")
        self.assertEqual(
            len(out.args[0]),
            4,
            "2 crossing values (h1, h0) + 2 saved (lin0.weight, x) — h0 "
            "crosses so it is deduplicated out of saved",
        )

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_manual_plan_wrong_stage_count_rejected(self):
        """Test a plan declaring fewer stages than pp_degree fails."""
        plan = PassPlan()
        plan.pp_stage(0, ["embed", "lin0"])
        gm = _mini_llm_joint_graph()
        with self.assertRaises(ValueError) as ctx:
            _run_pp(gm, MiniLLM(), rank=0, plan=plan)
        self.assertIn("pp_degree", str(ctx.exception))

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_skip_stage_dataflow_rejected(self):
        """Test a value crossing 2+ cuts fails with an actionable error.

        ``a`` is produced on stage 0 and consumed by stage 2's residual
        add — v1 has no route for it (only neighbouring stages exchange
        boundary values), so the pass must reject it up front instead of
        failing later inside the slice copy with a generic message.
        """
        gm = _skip_stage_joint_graph()
        plan = PassPlan()
        plan.pp_stage(0, ["m0"])
        plan.pp_stage(1, ["m1"])
        plan.pp_stage(2, ["m2"])
        cfg = PassConfig(fsdp_enabled=False, pp_enabled=True, pp_degree=3)
        with _patch_dist(world_size=3, rank=2):
            with self.assertRaises(ValueError) as ctx:
                PpPass(pass_plan=plan).run(gm, cfg, model=ThreeModule())
        self.assertIn("neighbouring stages", str(ctx.exception))
        self.assertIn("stage 0", str(ctx.exception))
        self.assertIn("stage 2", str(ctx.exception))


class TestPruneLiveModel(unittest.TestCase):
    """``_prune_live_model`` handles container elements with descendant cuts."""

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_descendant_cut_recurses_into_container_element(self):
        """Test a cut inside a container element recurses (key-preserving).

        A manual plan declaring ``blocks.0.inner`` must keep that descendant,
        prune its non-stage sibling inside the same element, drop the whole
        foreign element WITHOUT reindexing the surviving key, and prune
        foreign modules outside the container.
        """

        class Nested(nn.Module):
            """Element with two leaf children, only one on the stage."""

            def __init__(self) -> None:
                """Initialize ``inner`` and ``extra`` leaves."""
                super().__init__()
                self.inner = nn.Linear(4, 4)
                self.extra = nn.Linear(4, 4)

        class NestedContainer(nn.Module):
            """Layer container of nested elements plus a head."""

            def __init__(self) -> None:
                """Initialize two container elements and a head."""
                super().__init__()
                self.blocks = nn.ModuleList([Nested(), Nested()])
                self.head = nn.Linear(4, 4)

        model = NestedContainer()
        PpPass()._prune_live_model(model, ["blocks.0.inner"])  # pylint: disable=protected-access
        self.assertIsNotNone(
            model.blocks[0].inner, "declared descendant module must survive"
        )
        self.assertIsNone(
            model.blocks[0].extra, "sibling inside the cut element is pruned"
        )
        self.assertIsNone(model.head, "foreign module outside the container is pruned")
        self.assertEqual(
            list(model.blocks._modules.keys()),  # pylint: disable=protected-access
            ["0"],
            "pruned element key is removed without reindexing survivors",
        )


class TestAutoStageSplit(unittest.TestCase):
    """Default even-by-layers split."""

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_even_fallback_split_without_container(self):
        """Test top-level children split evenly without a ModuleList."""
        stages = _auto_stage_split(MiniLLM(), 2)
        self.assertEqual(stages, [["embed", "lin0"], ["lin1", "head"]])

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_container_split_with_input_output_weighting(self):
        """Test layers are evenly distributed; head/tail modules attached."""
        stages = _auto_stage_split(ContainerModel(num_layers=5), 2)
        self.assertEqual(
            stages,
            [
                ["tok_embeddings", "layers.0", "layers.1", "layers.2"],
                ["layers.3", "layers.4", "norm", "lm_head"],
            ],
        )

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_too_few_children_rejected(self):
        """Test more stages than children is an error without a container."""

        class TwoModule(nn.Module):
            """Two top-level leaf modules, no layer container."""

            def __init__(self) -> None:
                """Initialize modules ``a`` and ``b``."""
                super().__init__()
                self.a = nn.Linear(2, 2)
                self.b = nn.Linear(2, 2)

        with self.assertRaises(ValueError):
            _auto_stage_split(TwoModule(), 4)

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_single_stage_rejected(self):
        """Test ``pp_degree < 2`` cannot produce a split."""
        with self.assertRaises(ValueError):
            _auto_stage_split(MiniLLM(), 1)


class TestPassPlanPpStage(unittest.TestCase):
    """``PassPlan.pp_stage`` builder validation."""

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_wildcard_rejected(self):
        """Test stage cuts must be exact FQNs (no wildcards)."""
        plan = PassPlan()
        with self.assertRaises(ValueError) as ctx:
            plan.pp_stage(0, ["layers.*"])
        self.assertIn("wildcard", str(ctx.exception))

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_negative_stage_rejected(self):
        """Test a negative stage index is rejected."""
        plan = PassPlan()
        with self.assertRaises(ValueError):
            plan.pp_stage(-1, ["embed"])

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_manual_plan_resolves(self):
        """Test a complete manual plan passes plan resolution."""
        pas = PpPass(pass_plan=_manual_plan())
        stages = pas._resolve_stage_plan(MiniLLM(), 2)  # pylint: disable=protected-access
        self.assertEqual(stages, [["embed", "lin0"], ["lin1", "head"]])

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_unknown_module_rejected(self):
        """Test stage FQNs must exist in the model."""
        plan = PassPlan()
        plan.pp_stage(0, ["embed", "lin0"])
        plan.pp_stage(1, ["lin1", "nope"])
        pas = PpPass(pass_plan=plan)
        with self.assertRaises(ValueError) as ctx:
            pas._resolve_stage_plan(MiniLLM(), 2)  # pylint: disable=protected-access
        self.assertIn("nope", str(ctx.exception))

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_partial_plan_warns_on_unassigned_modules(self):
        """Test modules no stage declares are named in a warning."""
        plan = PassPlan()
        plan.pp_stage(0, ["embed", "lin0"])
        plan.pp_stage(1, ["lin1"])
        pas = PpPass(pass_plan=plan)
        with self.assertLogs(
            "hyper_parallel.compile.passes.parallel.pp_pass", level="WARNING"
        ) as logs:
            pas._resolve_stage_plan(MiniLLM(), 2)  # pylint: disable=protected-access
        self.assertIn("head", "".join(logs.output))

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_container_ancestors_do_not_warn(self):
        """Test ancestor containers of declared elements warn nothing."""
        plan = PassPlan()
        plan.pp_stage(0, ["tok_embeddings", "layers.0", "layers.1"])
        plan.pp_stage(1, ["layers.2", "norm", "lm_head"])
        pas = PpPass(pass_plan=plan)
        logger = logging.getLogger("hyper_parallel.compile.passes.parallel.pp_pass")
        with patch.object(logger, "warning") as warn:
            stages = pas._resolve_stage_plan(  # pylint: disable=protected-access
                ContainerModel(num_layers=3), 2
            )
        warn.assert_not_called()
        self.assertEqual(len(stages), 2)


class TestPropagateAnchorStages(unittest.TestCase):
    """Backward arg stage propagation reaches its fixpoint in one pass."""

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_transitive_chain_pulled_in_one_reverse_pass(self):
        """Test a downstream-anchored bwd chain pulls its args transitively.

        a2 is anchored to stage 2 while a0/a1 sit on stage 0: the single
        reverse-topological sweep must pull a1 (direct arg) and a0
        (transitive arg) up to stage 2, and leave the fwd-phase input
        placeholder alone.
        """
        g = fx.Graph()
        x = g.placeholder("x")
        a0 = g.call_function(torch.relu, (x,))
        a1 = g.call_function(torch.relu, (a0,))
        a2 = g.call_function(torch.relu, (a1,))
        g.output([a2])
        node_stage = {x: 0, a0: 0, a1: 0, a2: 2}
        node_phase = {x: "fwd", a0: "bwd", a1: "bwd", a2: "bwd"}
        gm = fx.GraphModule({}, g)
        PpPass._propagate_anchor_stages(  # pylint: disable=protected-access
            gm, node_stage, node_phase
        )
        self.assertEqual(node_stage[a2], 2, "anchored node keeps its stage")
        self.assertEqual(node_stage[a1], 2, "direct arg pulled up")
        self.assertEqual(node_stage[a0], 2, "transitive arg pulled up too")
        self.assertEqual(node_stage[x], 0, "fwd-phase arg untouched")


@contextmanager
def _patch_sched_p2p() -> Iterator[dict]:
    """Patch ``dist`` inside ``pp_schedule`` with an in-process P2P shim.

    ``isend``/``irecv`` append to / pop a single queue instead of touching a
    real backend, so both stages can run sequentially in one process and the
    gradient/scaling arithmetic can be checked numerically. The test drives
    two stages strictly in pipeline order (rank 0 sends forward, rank 1 sends
    its boundary grad back later), so one FIFO is enough; both directions key
    on the fixed sender rank 0.
    """
    store: dict = {}

    def isend(tensor: torch.Tensor, dst: int = 0, group: Any = None) -> MagicMock:
        """Queue a detached copy of the value being sent."""
        del dst, group  # the shim routes by fixed sender, not by peer
        store.setdefault(("send", 0), []).append(tensor.detach().clone())
        return MagicMock()

    def irecv(buffer: torch.Tensor, src: int = 0, group: Any = None) -> MagicMock:
        """Fill ``buffer`` from the queue that ``src`` filled."""
        del group
        channel = store.setdefault(("send", 0), [])
        if not channel:
            raise AssertionError(
                f"test P2P shim: nothing queued to receive from rank {src}"
            )
        item = channel.pop(0)
        if buffer.shape == ():
            buffer.fill_(item.reshape(()).to(torch.int64))
        else:
            buffer.copy_(item)
        return MagicMock()

    mock_dist = MagicMock()
    mock_dist.isend.side_effect = isend
    mock_dist.irecv.side_effect = irecv
    mock_dist.get_global_rank.side_effect = lambda _g, r: r
    with patch(_SCHED_DIST_PATH, mock_dist):
        yield store


def _tiny_lm_joint_graph(batch: int = 4):
    """Trace a tiny LM to a joint graph (for the numeric equivalence test).

    Returns ``(joint_graph, model, x, y)`` where ``x``/``y`` have ``batch``
    rows. Tracing is static-shape, so a PP stage must be traced with a
    MICRO-batch sample and then run on the full batch (the schedule slices
    it back into matching micro-batches), exactly as the example does.
    Seed the RNG before calling for reproducible weights.
    """

    class TinyLM(nn.Module):
        """embed -> lin0 -> relu -> lin1 -> head, next-token CE loss."""

        def __init__(self) -> None:
            """Initialize the four leaf submodules."""
            super().__init__()
            self.embed = nn.Embedding(16, 8)
            self.lin0 = nn.Linear(8, 8)
            self.lin1 = nn.Linear(8, 8)
            self.head = nn.Linear(8, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return vocab logits for ``x``."""
            h = self.embed(x)
            h = torch.relu(self.lin0(h))
            return self.head(self.lin1(h))

    def train_fn(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cross-entropy over the flattened batch (mean-reduced loss)."""
        logits = model(x)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )

    torch.manual_seed(0)
    model = TinyLM().to(torch.float64)  # keep the allclose tolerance tight
    x = torch.randint(0, 16, (batch, 5))
    y = torch.randint(0, 16, (batch, 5))
    with warnings.catch_warnings():
        # Stock torch warns that the autograd-engine hook is absent; the
        # joint graph still captures (the tracer reconstructs the tag).
        warnings.simplefilter("ignore", RuntimeWarning)
        jg = trace_model_graph(model, train_fn, x, y)
    return jg, model, x, y


class TestScheduleGradEquivalence(unittest.TestCase):
    """PP grads equal the non-PP full-batch mean-loss gradient.

    Regression guard for the loss-scaling fix: the backward sweep must
    divide the summed per-microbatch grads by ``num_microbatches`` (the
    hand-written ``ones_like`` seed inside the traced backward graph means
    the sweep produces un-normalized sums, not a mean). Without the final
    division the grads are ``num_microbatches``x too large. The two
    coincide only when there is a single microbatch, so this test uses a
    batch of 4 split into 2 microbatches of 2 to separate them.
    """

    def _stage_model(self):
        """A fresh model matching ``_tiny_lm_joint_graph``'s module layout."""
        torch.manual_seed(0)

        class _LM(nn.Module):
            """Same layout/init as ``_tiny_lm_joint_graph``'s TinyLM."""

            def __init__(self) -> None:
                """Initialize embed/lin0/lin1/head."""
                super().__init__()
                self.embed = nn.Embedding(16, 8)
                self.lin0 = nn.Linear(8, 8)
                self.lin1 = nn.Linear(8, 8)
                self.head = nn.Linear(8, 16)

        return _LM().to(torch.float64)

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_pp_grads_match_full_batch_mean(self):
        """Test 2-stage PP grads match the reference full-batch grads."""
        plan = PassPlan()
        plan.pp_stage(0, ["embed", "lin0"])
        plan.pp_stage(1, ["lin1", "head"])
        cfg = PassConfig(
            fsdp_enabled=False, pp_enabled=True, pp_degree=2, pp_microbatch_size=2
        )

        # Reference: single-card full-batch gradient on the plain joint graph
        # (traced at the FULL batch, since there is no micro-batching).
        ref_jg, ref_model, x, y = _tiny_lm_joint_graph(batch=4)
        _, ref_grads = run_traced_graph(ref_jg, ref_model, x, y)
        ref_fqns = [name for name, p in ref_model.named_parameters() if p.requires_grad]
        ref_by_fqn = {n: g.clone() for n, g in zip(ref_fqns, ref_grads)}

        # PP: one pruned model + schedule per rank, sharing one P2P shim.
        # Each stage is traced with a MICRO-batch sample (static shapes) and
        # then run on the full batch of 4 — the schedule slices it into 2
        # micro-batches of 2.
        models, scheds = {}, {}
        for rank in (0, 1):
            jg, _, _, _ = _tiny_lm_joint_graph(batch=2)
            model = self._stage_model()
            with _patch_dist(world_size=2, rank=rank):
                PpPass(pass_plan=plan).run(jg.graph_module, cfg, model=model)
            models[rank] = model
            scheds[rank] = jg.graph_module.pp_schedule

        with _patch_sched_p2p() as store:
            # Exercise the REAL _forward_sweep / _backward_sweep where the
            # 1/num_microbatches mean normalization lives. Order them as
            # GPipe does: stage 0 fwd (queues the activation) -> stage 1 fwd
            # -> stage 1 bwd (queues the boundary grad) -> stage 0 bwd. The
            # in-process shim carries each boundary value between the two
            # stages' sweeps.
            state0 = [p.detach() for p in models[0].parameters()]
            state1 = [p.detach() for p in models[1].parameters()]
            s0, s1 = scheds[0], scheds[1]
            n_mb = x.shape[0] // s0.microbatch_size

            fwd0 = s0._forward_sweep(state0, x, y, n_mb)  # pylint: disable=protected-access
            fwd1 = s1._forward_sweep(state1, x, y, n_mb)  # pylint: disable=protected-access
            grads1 = s1._backward_sweep(state1, fwd1, n_mb)  # pylint: disable=protected-access
            grads0 = s0._backward_sweep(state0, fwd0, n_mb)  # pylint: disable=protected-access
            store.clear()

        def _fqn_grads(rank, grads):
            """Map a stage's gradient list to FQNs via its model order."""
            fqns = [
                name for name, p in models[rank].named_parameters() if p.requires_grad
            ]
            return dict(zip(fqns, grads))

        pp_by_fqn = {**_fqn_grads(0, grads0), **_fqn_grads(1, grads1)}
        self.assertEqual(
            set(pp_by_fqn), set(ref_by_fqn), "PP must cover every trainable param"
        )
        for name, ref_g in ref_by_fqn.items():
            self.assertTrue(
                torch.allclose(pp_by_fqn[name], ref_g, atol=1e-6, rtol=1e-5),
                f"PP grad for {name}={pp_by_fqn[name]} != reference {ref_g}",
            )


class TestYamlPpSection(unittest.TestCase):
    """YAML ``pp:`` section parsing."""

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_yaml_stages_parsed(self):
        """Test the mapping and list YAML shapes both parse."""
        import tempfile  # pylint: disable=C0415

        from hyper_parallel.compile.sharding_config import (  # pylint: disable=C0415
            create_sharding_plan_from_yaml,
        )

        yaml_text = """
pp:
  stages:
    - stage: 0
      modules: [tok_embeddings, layers.0]
    - stage: 1
      modules: [layers.1, norm, lm_head]
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            path = f.name
        plan = create_sharding_plan_from_yaml(config_path=path)
        self.assertEqual(
            plan.pp_module_fqns_per_stage,
            [
                ["tok_embeddings", "layers.0"],
                ["layers.1", "norm", "lm_head"],
            ],
        )


class TestScheduleGPipe(unittest.TestCase):
    """Schedule driver logic with mocked subgraphs and mocked dist."""

    def _make_sched(self, fwd_ret, bwd_ret, stage_idx=1, pp_degree=3):
        """Build a schedule whose subgraphs and P2P are fully mocked."""
        sched = ScheduleGPipe(
            fwd_gm=MagicMock(return_value=fwd_ret),
            bwd_gm=MagicMock(return_value=bwd_ret),
            stage_idx=stage_idx,
            pp_degree=pp_degree,
            num_state=1,
            num_trainable=len(bwd_ret) - 1,
            num_send=0 if stage_idx == pp_degree - 1 else 1,
            grad_send_count=0 if stage_idx == 0 else 1,
            microbatch_size=2,
            pp_group=MagicMock(),
            recv_spec=(
                [("tensor", (4, 4), torch.float32, torch.device("cpu"))]
                if stage_idx > 0
                else []
            ),
            grad_recv_spec=(
                [("tensor", (4, 4), torch.float32, torch.device("cpu"))]
                if stage_idx < pp_degree - 1
                else []
            ),
        )
        return sched

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_middle_stage_grads_averaged_and_p2p_issued(self):
        """Test microbatch loop: recv -> fwd -> send, grads mean over mb."""
        state = [torch.zeros(4, 4)]
        fwd_ret = (torch.ones(4, 4), torch.zeros(4), torch.zeros(4))
        g1, g2 = torch.ones(4), torch.full((4,), 3.0)
        bwd_ret = (g1, g2, torch.ones(4, 4))  # 2 grads + boundary grad
        sched = self._make_sched(fwd_ret, bwd_ret, stage_idx=1, pp_degree=3)

        with patch(_SCHED_DIST_PATH) as mock_dist:
            mock_dist.isend.return_value = MagicMock()
            mock_dist.irecv.return_value = MagicMock()
            mock_dist.get_global_rank.side_effect = lambda _g, r: r
            input_batch = torch.zeros(4, 8)  # 2 microbatches of 2
            label = torch.zeros(4, 8)
            loss, *grads = sched(*state, input_batch, label)

        # fwd called once per microbatch with (*state, act_in)
        self.assertEqual(sched.fwd_gm.call_count, 2)
        fwd_args = sched.fwd_gm.call_args_list[0].args
        self.assertEqual(len(fwd_args), 2, "(*state, act_in) for a middle stage")
        self.assertEqual(tuple(fwd_args[1].shape), (4, 4))
        # bwd called once per microbatch with (*state, grad_in, *fwd_outs)
        self.assertEqual(sched.bwd_gm.call_count, 2)
        # The traced backward seeds its own loss gradient, so the sweep
        # accumulates un-normalized sums over the 2 microbatches and then
        # divides by num_microbatches: (1+1)/2 and (3+3)/2.
        self.assertTrue(torch.allclose(grads[0], torch.full((4,), 1.0)))
        self.assertTrue(torch.allclose(grads[1], torch.full((4,), 3.0)))
        # non-last stage loss is a zero placeholder
        self.assertTrue(torch.allclose(loss, torch.zeros(())))
        # async sends: act_out + boundary grad, once per microbatch each
        self.assertEqual(mock_dist.isend.call_count, 4)
        # receives: act_in (fwd) + grad_in (bwd), once per microbatch each
        self.assertEqual(mock_dist.irecv.call_count, 4)

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_scalar_boundary_values_packed_as_int64(self):
        """Test sym_size-style scalars ship as 0-d int64 and arrive as ints."""
        state = [torch.zeros(4, 4)]
        fwd_ret = (torch.ones(4, 4), 8, torch.zeros(4))  # act, scalar, saved
        bwd_ret = (torch.ones(4),)  # 1 grad, no boundary grad (stage 0)
        sched = ScheduleGPipe(
            fwd_gm=MagicMock(return_value=fwd_ret),
            bwd_gm=MagicMock(return_value=bwd_ret),
            stage_idx=1,
            pp_degree=2,
            num_state=1,
            num_trainable=1,
            num_send=0,
            grad_send_count=0,
            microbatch_size=1,
            pp_group=MagicMock(),
            recv_spec=[
                ("tensor", (4, 4), torch.float32, torch.device("cpu")),
                ("scalar",),
            ],
            grad_recv_spec=[],
        )
        sched.is_last = True

        with patch(_SCHED_DIST_PATH) as mock_dist:
            mock_dist.isend.return_value = MagicMock()
            mock_dist.irecv.return_value = MagicMock()
            mock_dist.get_global_rank.side_effect = lambda _g, r: r

            def fake_irecv(
                buffer: torch.Tensor, src: int = 0, group: Any = None
            ) -> MagicMock:
                """Fill scalar buffers with 8 so item() returns an int."""
                if buffer.shape == ():
                    buffer.fill_(8)
                return MagicMock()

            mock_dist.irecv.side_effect = fake_irecv
            loss, *grads = sched(*state, torch.zeros(1, 8), torch.zeros(1, 8))

        fwd_args = sched.fwd_gm.call_args_list[0].args
        # (*state, act_in, scalar) — scalar arrived as a Python int
        self.assertIsInstance(fwd_args[2], int)
        self.assertEqual(fwd_args[2], 8)
        # last stage: real loss returned
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(len(grads), 1)

    @arg_mark(
        plat_marks=["cpu_linux"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_batch_not_divisible_rejected(self):
        """Test an indivisible batch size fails loudly."""
        sched = self._make_sched(
            (torch.ones(4, 4),), (torch.ones(4),), stage_idx=0, pp_degree=3
        )
        with self.assertRaises(ValueError):
            sched(torch.zeros(4, 4), torch.zeros(4, 7), torch.zeros(4, 7))


if __name__ == "__main__":
    unittest.main()
