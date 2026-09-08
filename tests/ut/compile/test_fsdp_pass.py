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
4. Skips parameters whose dim 0 is not divisible by ``fsdp_degree`` (they
   stay replicated in both graph and live model — the divisibility gate must
   match between ``_identify_params_in_fsdp_modules`` and
   ``_shard_live_model_params``).
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

import os
import unittest
from contextlib import contextmanager
from typing import Iterator
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch
from torch import fx, nn

from hyper_parallel.compile.parallel_config import PassConfig
from hyper_parallel.compile.passes.parallel.fsdp_pass import FSDPPass
from hyper_parallel.compile.sharding_config import PassPlan

_DIST_PATH = "hyper_parallel.compile.passes.parallel.fsdp_pass.dist"


@contextmanager
def _patch_dist(
    world_size: int = 2, rank: int = 0, initialized: bool = True
) -> Iterator[MagicMock]:
    """Patch ``dist`` inside ``fsdp_pass`` with a stub returning the given values.

    The pass only calls ``dist.is_initialized`` / ``dist.get_world_size`` /
    ``dist.get_rank`` at pass time (collectives are FX nodes, never executed),
    so a single MagicMock covers every call.
    """
    mock_dist = MagicMock()
    mock_dist.is_initialized.return_value = initialized
    mock_dist.get_world_size.return_value = world_size
    mock_dist.get_rank.return_value = rank
    with patch(_DIST_PATH, mock_dist):
        yield mock_dist


def _count_targets(gm: fx.GraphModule, needle: str) -> int:
    """Count ``call_function`` nodes whose ``str(target)`` contains ``needle``."""
    return sum(
        1 for n in gm.graph.nodes if n.op == "call_function" and needle in str(n.target)
    )


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


def _odd_joint_graph() -> fx.GraphModule:
    """Joint-graph stub for ``_OddParam`` (state = ``odd`` only)."""
    g = fx.Graph()
    odd = g.placeholder("odd")
    loss = g.placeholder("loss")
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

    ``lin`` is a named submodule (so a partial ``fsdp_wrap("lin")`` plan
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

    def test_run_skips_non_divisible_params(self):
        """Test params whose dim 0 is not divisible by ``fsdp_degree`` stay replicated.

        The divisibility gate must agree between graph and live model: a
        non-divisible param is skipped on both sides, so no AllGather is
        inserted and the live tensor keeps its shape.
        """
        cfg = PassConfig(fsdp_enabled=True, fsdp_degree=2)
        pas = FSDPPass()
        gm = _odd_joint_graph()
        model = _OddParam()
        before = tuple(model.odd.shape)

        with _patch_dist(world_size=2, rank=0, initialized=True):
            result = pas.run(gm, cfg, model=model, fsdp_group_name="fsdp")

        self.assertIs(result, gm, "run() should return the same graph_module")
        self.assertEqual(
            tuple(model.odd.shape),
            before,
            (
                f"non-divisible param must stay replicated, "
                f"odd shape={tuple(model.odd.shape)} expected={before}"
            ),
        )
        self.assertEqual(
            _count_targets(gm, "all_gather_into_tensor"),
            0,
            (
                f"non-divisible param must get no AllGather, "
                f"got {_count_targets(gm, 'all_gather_into_tensor')}"
            ),
        )
        self.assertEqual(
            _count_targets(gm, "reduce_scatter_tensor"),
            0,
            (
                f"non-divisible param must get no ReduceScatter, "
                f"got {_count_targets(gm, 'reduce_scatter_tensor')}"
            ),
        )

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
        """Test a partial plan shards wrapped params and leaves others full-size.

        The plan-gate must agree between graph and live model: only ``lin``'s
        params (inside an FSDP-wrapped module) get AllGather/ReduceScatter and
        are physically sharded. The root-level ``extra`` param stays full-size
        on both sides — sharding it live while the graph holds it replicated
        (no AllGather) would mismatch at ``run_traced_graph`` time.
        """
        cfg = PassConfig(fsdp_enabled=True, fsdp_degree=2)
        plan = PassPlan().fsdp_wrap("lin")
        pas = FSDPPass(pass_plan=plan)
        gm = _lin_bias_extra_joint_graph()
        model = _WrappedPlusRootParam()
        before_extra = tuple(model.extra.shape)

        with _patch_dist(world_size=2, rank=0, initialized=True):
            # pass_plan auto-filled from the pass (not kwargs), as the
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


class TestFSDPPassHelpers(unittest.TestCase):
    """Pure helpers: FQN matching, parent extraction, trainable-state filter."""

    def test_param_belongs_to_fsdp_module_exact_and_pattern(self):
        """Test exact-FQN and wildcard-pattern matching of ancestor modules."""
        exact = FSDPPass(pass_plan=PassPlan().fsdp_wrap("lin"))
        self.assertTrue(
            exact._param_belongs_to_fsdp_module("lin.weight"),  # pylint: disable=protected-access
            "exact fsdp_wrap('lin') should match a param under module 'lin'",
        )
        self.assertFalse(
            exact._param_belongs_to_fsdp_module("other.weight"),  # pylint: disable=protected-access
            "exact wrap must not match an unrelated module",
        )

        pattern = FSDPPass(pass_plan=PassPlan().fsdp_wrap_pattern("layers.*"))
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
        a ``*`` plan or an explicit ``fsdp_wrap("weight")``.
        """
        pas = FSDPPass(pass_plan=PassPlan().fsdp_wrap_pattern("*"))
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
