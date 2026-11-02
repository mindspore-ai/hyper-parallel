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
"""MoonEP as the routed-experts transport of the framework EP factory.

The kernels and the plan need Ascend hardware, but the *wiring* and the
aggregation arithmetic do not: the factory must pick the MoonEP closure when
``HP_MOONEP_EP`` is set, and the routed branch must weight the combined rows in
token order exactly as the all-to-all path does.  Both are pinned here with a
stub transport that delivers every row to its expert in place.
"""
# pylint: disable=wrong-import-position

from __future__ import annotations

import os
import unittest
from collections.abc import Callable
from types import SimpleNamespace
from unittest import mock

import torch

from hyper_parallel.distributed.expert_parallel import moonep_forward, recipes
from hyper_parallel.distributed.expert_parallel.moonep_forward import (
    moonep_ep_enabled,
    moonep_routed_forward,
)


class _Experts(torch.nn.Module):
    """Expert holder with the attributes the EP compute path reads."""

    def __init__(self, num_experts: int, dim: int, hidden: int) -> None:
        """Create deterministic SwiGLU tables for ``num_experts`` experts."""
        super().__init__()
        self.num_experts = num_experts
        self.local_expert_count = num_experts
        self.w1 = torch.nn.Parameter(torch.randn(num_experts, hidden, dim) * 0.1)
        self.w2 = torch.nn.Parameter(torch.randn(num_experts, dim, hidden) * 0.1)
        self.w3 = torch.nn.Parameter(torch.randn(num_experts, hidden, dim) * 0.1)
        self._ep_act_fn = torch.nn.functional.silu
        self._ep_apply_gate = None


class _Block(torch.nn.Module):
    """Minimal HF-shaped MoE block: a Linear gate plus the expert holder."""

    def __init__(self, num_experts: int, dim: int, hidden: int, top_k: int) -> None:
        """Build the gate and expert tables."""
        super().__init__()
        self.gate = torch.nn.Linear(dim, num_experts, bias=False)
        self.experts = _Experts(num_experts, dim, hidden)
        self.top_k = top_k


class _StubTransport:
    """Delivers every row to its own expert in place, in caller order."""

    def __init__(self) -> None:
        """Record the forwarded rows for later assertions."""
        self.rows = None
        self.route_ids = None
        self.route_weights = None

    def forward(self, rows: torch.Tensor, route_ids: torch.Tensor,
                route_weights: torch.Tensor, expert_forward: Callable | None = None,
                output_dtype: torch.dtype | None = None,
                expert_forward_many: Callable | None = None,
                expert_forward_factory: Callable | None = None) -> torch.Tensor:
        """Run ``expert_forward`` per expert id and keep the row order."""
        if expert_forward_factory is not None:
            # The production factory only needs a copies-shaped argument; the
            # table gather it triggers is mocked out by the caller.
            expert_forward = expert_forward_factory(
                torch.full((1, 1), -1, dtype=torch.int32))
        self.rows = rows
        self.route_ids = route_ids
        self.route_weights = route_weights
        self.expert_forward_many = expert_forward_many
        out = torch.zeros_like(rows, dtype=torch.float32)
        for expert in torch.unique(route_ids):
            mask = route_ids == expert
            out[mask] = expert_forward(rows[mask], int(expert), route_weights[mask]).float()
        return out if output_dtype is None else out.to(output_dtype)


class _EpMesh:
    """Stands in for a 1-D EP mesh: the factory reads both of these."""

    def __init__(self, group: object) -> None:
        """Bind the fake process group."""
        self._group = group

    def get_group(self, name: str) -> object:
        """Return the group for the requested axis."""
        if name != "ep":
            raise KeyError(name)
        return self._group

    def __getitem__(self, name: str) -> SimpleNamespace:
        """Return an axis view exposing ``size()``."""
        if name != "ep":
            raise KeyError(name)
        return SimpleNamespace(size=lambda: 1)


def _reference_expert(experts: _Experts, row: torch.Tensor, expert: int) -> torch.Tensor:
    """First-principles SwiGLU expert output, without routing weights."""
    hidden = torch.nn.functional.silu(torch.nn.functional.linear(row, experts.w1[expert].float()))
    hidden = hidden * torch.nn.functional.linear(row, experts.w3[expert].float())
    return torch.nn.functional.linear(hidden, experts.w2[expert].float())


def _local_tables(experts: _Experts, group: object, copies: torch.Tensor,
                  **_kwargs) -> tuple:
    """Single-rank stand-in for ``_needed_tables``: every expert is local."""
    del group, copies
    return tuple(
        moonep_forward._WeightView(w, None, None, 0, w.shape[0])
        for w in (experts.w1, experts.w3, experts.w2)
    )


class TestMoonEPEnableSwitch(unittest.TestCase):
    """``HP_MOONEP_EP`` is the only thing that selects the transport."""

    def test_truthy_values_select_moonep(self):
        """1/true/yes/on enable the path; anything else leaves it off."""
        for value in ("1", "true", "TRUE", "yes", "on"):
            with self.subTest(value=value), mock.patch.dict(
                    os.environ, {"HP_MOONEP_EP": value}):
                self.assertTrue(moonep_ep_enabled(), f"{value!r} must enable MoonEP")
        for value in ("", "0", "false", "off", "nope"):
            with self.subTest(value=value), mock.patch.dict(
                    os.environ, {"HP_MOONEP_EP": value}):
                self.assertFalse(moonep_ep_enabled(), f"{value!r} must not enable MoonEP")

    def test_the_factory_selects_the_moonep_branch(self):
        """``build_ep_compute`` closes over MoonEP when the switch is on."""
        block = _Block(4, 8, 16, 2)
        mesh = _EpMesh(group=object())
        hidden = torch.randn(2, 3, 8)

        with mock.patch.object(moonep_forward, "moonep_ep_enabled", return_value=True), \
                mock.patch.object(recipes, "moonep_ep_enabled", return_value=True), \
                mock.patch.object(recipes, "moonep_routed_forward",
                                  return_value=torch.ones(2, 3, 8)) as routed, \
                mock.patch.object(recipes, "ep_routed_forward") as a2a:
            compute = recipes.build_ep_compute(
                block, mesh, router_fn=lambda module, states: None,
                archetype_key="routed_only", expected_attrs=["gate", "experts"],
                combine=lambda module, states, out: out,
            )
            out = compute(block, hidden)

        routed.assert_called_once()
        a2a.assert_not_called()
        self.assertTrue(torch.equal(out, torch.ones(2, 3, 8)), f"unexpected output {out}")

    def test_the_factory_keeps_a2a_by_default(self):
        """Without the switch the all-to-all branch is untouched."""
        block = _Block(4, 8, 16, 2)
        mesh = _EpMesh(group=object())
        hidden = torch.randn(2, 3, 8)

        with mock.patch.dict(os.environ, {"HP_MOONEP_EP": ""}), \
                mock.patch.object(recipes, "ep_routed_forward",
                                  return_value=torch.zeros(2, 3, 8)) as a2a, \
                mock.patch.object(recipes, "moonep_routed_forward") as routed:
            compute = recipes.build_ep_compute(
                block, mesh, router_fn=lambda module, states: None,
                archetype_key="routed_only", expected_attrs=["gate", "experts"],
                combine=lambda module, states, out: out,
            )
            compute(block, hidden)

        a2a.assert_called_once()
        routed.assert_not_called()

    def test_moonep_enters_expert_hooks_on_every_step(self):
        """Expert-owned hooks must cover transport math on repeated calls."""
        block = _Block(4, 8, 16, 2)
        events = []
        block.experts.register_forward_pre_hook(lambda *_: events.append("enter"))
        block.experts.register_forward_hook(lambda *_: events.append("exit"))

        def routed(module, states, **kwargs):
            events.append("compute")
            return states * module.experts.w1.sum()

        with mock.patch.object(recipes, "moonep_ep_enabled", return_value=True), \
                mock.patch.object(recipes, "moonep_routed_forward", side_effect=routed):
            compute = recipes.build_ep_compute(
                block, _EpMesh(group=object()), router_fn=lambda *_: None,
                archetype_key="routed_only", expected_attrs=["gate", "experts"],
                combine=lambda module, states, output: output,
            )
            for _ in range(3):
                hidden = torch.randn(2, 3, 8, requires_grad=True)
                compute(block, hidden).sum().backward()
                block.zero_grad()

        expected = ["enter", "compute", "exit"] * 3
        self.assertEqual(events, expected, f"expected={expected}, actual={events}")


class TestMoonEPRoutedForward(unittest.TestCase):
    """The transport-independent half: rows, segment math, weighting, order."""

    def test_auto_fallback_reuses_router_results_and_gradients(self):
        """A stateful router runs once and fallback retains its score graph."""
        hidden = torch.randn(1, 3, 8)
        scores = torch.randn(3, 1, requires_grad=True)
        ids = torch.zeros(3, 1, dtype=torch.long)
        router = mock.Mock(return_value=(ids, scores))

        def fallback(module, states, *, router_fn, ep_group):
            routed_ids, routed_scores = router_fn(module, states)
            self.assertTrue(torch.equal(routed_ids, ids))
            return routed_scores.sum()

        with mock.patch.dict(os.environ, {"HP_MOONEP_EP": "auto", "HP_MOONEP_MIN_ROWS": "100"}), \
                mock.patch.object(moonep_forward, "ep_routed_forward", side_effect=fallback), \
                mock.patch.object(moonep_forward, "_transport_for") as transport:
            result = moonep_routed_forward(object(), hidden, router_fn=router, ep_group=None)
        result.backward()
        router.assert_called_once()
        transport.assert_not_called()
        self.assertTrue(torch.equal(scores.grad, torch.ones_like(scores)))

    def _run(self, block: _Block, hidden: torch.Tensor, transport: _StubTransport):
        """Run the routed branch with a stub transport and single-rank tables."""
        route_ids = torch.tensor([0, 3, 1, 3, 2, 0], dtype=torch.long)
        weights = torch.tensor([0.5, 0.25, 1.0, 0.125, 0.75, 0.375])

        def router(module: torch.nn.Module, states: torch.Tensor) -> tuple:
            """Return the pinned routes and weights for this call."""
            del module, states
            return route_ids.reshape(-1, 1), weights.reshape(-1, 1)

        with mock.patch.object(
                moonep_forward, "_transport_for", return_value=transport), \
                mock.patch.object(
                    moonep_forward, "_needed_tables", side_effect=_local_tables):
            return moonep_routed_forward(block, hidden, router_fn=router, ep_group=None)

    def test_output_is_token_major_and_routing_weighted(self):
        """Every row's expert output comes back to its token, scaled by its weight."""
        torch.manual_seed(0)
        block = _Block(num_experts=4, dim=8, hidden=16, top_k=1)
        hidden = torch.randn(1, 6, 8)
        transport = _StubTransport()
        route_ids = torch.tensor([0, 3, 1, 3, 2, 0])
        weights = torch.tensor([0.5, 0.25, 1.0, 0.125, 0.75, 0.375])

        out = self._run(block, hidden, transport)

        expected = torch.stack([
            # MoonEP moves bf16 rows, so the reference consumes the same rounding.
            _reference_expert(
                block.experts,
                hidden.reshape(-1, 8)[row].to(torch.bfloat16).float(),
                int(expert),
            )
            * float(weight)
            for row, (expert, weight) in enumerate(zip(route_ids, weights))
        ]).reshape(1, 6, 8)
        self.assertTrue(
            torch.allclose(out, expected, atol=3e-4, rtol=1e-3),
            f"weighted rows diverged: max={(out - expected).abs().max().item():.3e}",
        )

    def test_rows_and_ids_reach_the_transport_in_token_order(self):
        """Row ``t * K + k`` must carry token ``t`` and its k-th route."""
        torch.manual_seed(0)
        block = _Block(num_experts=4, dim=8, hidden=16, top_k=2)
        hidden = torch.randn(1, 3, 8)
        transport = _StubTransport()
        route_ids = torch.tensor([[0, 3], [1, 2], [3, 0]])
        weights = torch.tensor([[0.5, 0.25], [1.0, 0.125], [0.75, 0.375]])

        def router(module: torch.nn.Module, states: torch.Tensor) -> tuple:
            """Return the pinned routes and weights for this call."""
            del module, states
            return route_ids, weights

        with mock.patch.object(
                moonep_forward, "_transport_for", return_value=transport), \
                mock.patch.object(
                    moonep_forward, "_needed_tables", side_effect=_local_tables):
            moonep_routed_forward(block, hidden, router_fn=router, ep_group=None)

        expected_rows = hidden.reshape(-1, 8)[torch.arange(3).repeat_interleave(2)]
        self.assertTrue(
            torch.equal(transport.rows, expected_rows.to(torch.bfloat16)),
            f"rows are not in token order: got {transport.rows.shape}",
        )
        self.assertEqual(
            transport.route_ids.tolist(), [0, 3, 1, 2, 3, 0],
            f"route ids must follow the rows, got {transport.route_ids.tolist()}",
        )
        self.assertTrue(
            torch.allclose(transport.route_weights, weights.reshape(-1).float()),
            f"route weights must follow the rows, got {transport.route_weights.tolist()}",
        )

    def test_batched_grouped_path_is_env_gated_off_by_default(self):
        """Production hands the transport a grouped callback, gated by env.

        The grouped local path runs the local-expert block in one GEMM, but
        only when ``HP_MOONEP_GROUPED=1``; without it the transport must keep
        the per-segment default.
        """
        block = _Block(num_experts=4, dim=8, hidden=16, top_k=1)
        transport = _StubTransport()

        self._run(block, torch.randn(1, 6, 8), transport)

        self.assertIsNotNone(
            transport.expert_forward_many,
            "the framework path should hand the transport a grouped callback",
        )
        self.assertNotEqual(
            os.environ.get("HP_MOONEP_GROUPED", "0"), "1",
            "the grouped path must stay opt-in (HP_MOONEP_GROUPED=1)",
        )


class TestWeightView(unittest.TestCase):
    """_WeightView must fail loudly for an expert outside the gathered set."""

    def test_missing_remote_expert_raises(self):
        local = torch.zeros(2, 3)
        compact = torch.ones(4, 3)
        position = torch.tensor([-1, 0, -1, 1])
        view = moonep_forward._WeightView(local, compact, position, lo=0, epn=2)
        self.assertEqual(int(view[0].data_ptr()), int(view.local[0].data_ptr()))
        self.assertEqual(int(view[1].data_ptr()), int(view.local[1].data_ptr()))
        with self.assertRaises(KeyError):
            view[2]
        self.assertEqual(int(view[3].data_ptr()), int(view.compact[1].data_ptr()))


class TestGatherNeededFused(unittest.TestCase):
    """The fused single-collective gather must address the same rows as the
    per-table reference and let gradients reach the contributing shard."""

    @staticmethod
    def _tables() -> tuple:
        """Two SwiGLU tables over E=4 experts (gate [4,3,2], down [4,2,3])."""
        gate = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
        down = 100.0 + torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
        return gate, down

    class _ReplayAllGather:
        """Answers with precomputed per-rank inputs, keeping the live one."""

        def __init__(self, rank: int, per_call_inputs: list) -> None:
            self.rank = rank
            self.per_call = per_call_inputs
            self.call = 0

        def __call__(self, t, group=None):
            del group
            inputs = self.per_call[self.call]
            self.call += 1
            return tuple(t if r == self.rank else inputs[r]
                         for r in range(len(inputs)))

    def _run_rank(self, rank: int, needed: torch.Tensor):
        """Run both gather variants for one rank under a fake 2-rank group."""
        ep_size, epn = 2, 2
        gate, down = self._tables()
        owners = torch.div(needed, epn, rounding_mode="floor")
        counts = torch.zeros(ep_size, dtype=torch.long)
        counts.scatter_add_(0, owners, torch.ones_like(owners))
        width = int(counts.max())

        def contribution(r: int, weight: torch.Tensor) -> torch.Tensor:
            ids = (needed[owners == r] - r * epn).long()
            rows = weight[r * epn:(r + 1) * epn][ids]
            pad = torch.zeros((width - rows.shape[0],) + rows.shape[1:])
            return torch.cat([rows, pad])

        ref_calls = [[contribution(r, w) for r in range(ep_size)]
                     for w in (gate, down)]
        fused_calls = [[torch.cat([contribution(r, gate).reshape(-1),
                                   contribution(r, down).reshape(-1)])
                        for r in range(ep_size)]]
        locals_ = [gate[rank * epn:(rank + 1) * epn].clone().requires_grad_(True),
                   down[rank * epn:(rank + 1) * epn].clone().requires_grad_(True)]
        with mock.patch.object(moonep_forward.dist, "get_world_size",
                               return_value=ep_size), \
                mock.patch.object(moonep_forward.dist, "get_rank",
                                  return_value=rank), \
                mock.patch("torch.distributed.nn.functional.all_gather",
                           self._ReplayAllGather(rank, ref_calls)):
            reference = [moonep_forward._gather_needed(
                w[rank * epn:(rank + 1) * epn], None, needed)
                for w in (gate, down)]
        with mock.patch.object(moonep_forward.dist, "get_world_size",
                               return_value=ep_size), \
                mock.patch.object(moonep_forward.dist, "get_rank",
                                  return_value=rank), \
                mock.patch("torch.distributed.nn.functional.all_gather",
                           self._ReplayAllGather(rank, fused_calls)):
            fused = moonep_forward._gather_needed_fused(locals_, None, needed)
        return reference, fused, locals_

    def test_fused_matches_per_table_rows(self):
        """Every needed expert's rows equal the per-table gather's, both ranks."""
        needed = torch.tensor([1, 2])  # expert 1 owned by rank0, 2 by rank1
        for rank in (0, 1):
            reference, fused, _ = self._run_rank(rank, needed)
            for table in (0, 1):
                for expert in (1, 2):
                    self.assertTrue(
                        torch.equal(fused[table][expert], reference[table][expert]),
                        f"rank {rank} table {table} expert {expert} differs")

    def test_fused_backward_reaches_contributing_shard(self):
        """A read of a compact row must grad into the owner's local shard row."""
        needed = torch.tensor([1, 2])
        _reference, fused, locals_ = self._run_rank(0, needed)
        fused[0][2].sum().backward()  # expert 2 lives on rank 1: zero local grad
        # cat's backward materialises zeros for every differentiable input, so
        # the assertion is on values, not on grad being None.
        self.assertIsNotNone(locals_[0].grad)
        self.assertTrue(torch.equal(locals_[0].grad,
                                    torch.zeros_like(locals_[0].grad)))
        locals_[0].grad = None
        fused[0][1].sum().backward()  # expert 1 is rank0's row 1 of the gate
        self.assertIsNotNone(locals_[0].grad)
        self.assertTrue(torch.equal(locals_[0].grad[0],
                                    torch.zeros_like(locals_[0].grad[0])))
        self.assertTrue(torch.equal(locals_[0].grad[1],
                                    torch.ones_like(locals_[0].grad[1])))


class TestGatherNeededUnequalWidths(TestGatherNeededFused):
    """Qwen-style fused gate/up rows are twice as wide as down rows."""

    @staticmethod
    def _tables() -> tuple:
        """Use distinct values and unequal row widths to expose offset errors."""
        gate = torch.arange(48, dtype=torch.float32).reshape(4, 6, 2)
        down = 100.0 + torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
        return gate, down

    def test_remote_flat_views_share_storage(self):
        """Both tables address the packed result without regroup allocations."""
        _, views, _ = self._run_rank(0, torch.tensor([0, 2, 3]))
        self.assertEqual(views[0].compact.data_ptr(), views[1].compact.data_ptr())
        self.assertEqual(tuple(views[0][3].shape), (6, 2))
        self.assertEqual(tuple(views[1][3].shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
