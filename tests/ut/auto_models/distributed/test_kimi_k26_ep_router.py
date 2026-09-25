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
"""Balanced-router contract of the Kimi-K2.6 EP archetype factory.

``fix_router=True`` is a benchmark-only switch: it must give every destination
rank exactly the same slot count (that is what stops a skewed real router from
dominating step time) while keeping the real gate weights differentiable. These
tests pin the arithmetic and the switch wiring without any distributed setup.
"""

import os
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch  # pylint: disable=wrong-import-position

from tests.common.mark_utils import arg_mark  # pylint: disable=wrong-import-position

from hyper_parallel.distributed.expert_parallel import (  # pylint: disable=wrong-import-position
    routing as routing_module,
)
from hyper_parallel.models.kimi_k26.adapter.distributed import (  # pylint: disable=wrong-import-position
    ep_compute,
)


class _FakeExperts(torch.nn.Module):
    """Stand-in exposing the attributes the balanced router reads."""

    def __init__(self, local_expert_count: int, num_experts: int) -> None:
        """Store the expert layout the router reads back."""
        super().__init__()
        self.local_expert_count = local_expert_count
        self.num_experts = num_experts


class _FakeMoE(torch.nn.Module):
    """Minimal MoE block carrying an ``experts`` child."""

    def __init__(self, local_expert_count: int, num_experts: int) -> None:
        """Expose an ``experts`` child with the given layout."""
        super().__init__()
        self.experts = _FakeExperts(local_expert_count, num_experts)


class TestBalancedExpertIndices(unittest.TestCase):
    """The round-robin assignment itself."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_spreads_slots_evenly_over_destination_ranks(self) -> None:
        """Every destination rank gets the same slot count; experts differ by <=1."""
        tokens, per_token, total, local = 10, 2, 8, 2
        indices = ep_compute._balanced_expert_indices(  # pylint: disable=protected-access
            tokens, per_token, total, local, device=torch.device("cpu"))

        self.assertEqual(tuple(indices.shape), (tokens, per_token))
        self.assertEqual(indices.dtype, torch.int64)
        self.assertTrue(bool((indices >= 0).all() and (indices < total).all()))

        destinations = torch.div(indices.reshape(-1), local, rounding_mode="floor")
        per_destination = torch.bincount(destinations, minlength=total // local)
        self.assertEqual(len(set(per_destination.tolist())), 1,
                         f"destination load is not uniform: {per_destination.tolist()}")

        per_expert = torch.bincount(indices.reshape(-1), minlength=total)
        self.assertLessEqual(int(per_expert.max() - per_expert.min()), 1,
                             f"expert load spread too wide: {per_expert.tolist()}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_indivisible_expert_layout(self) -> None:
        """A global expert count that does not split evenly fails fast."""
        with self.assertRaises(ValueError):
            ep_compute._balanced_expert_indices(  # pylint: disable=protected-access
                4, 1, 6, 4, device=torch.device("cpu"))


class TestBalancedRouter(unittest.TestCase):
    """The router wrapper around the real sigmoid-group adapter."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_keeps_real_gate_weights_and_replaces_destinations(self) -> None:
        """Gate weights stay differentiable; only the chosen experts are replaced."""
        tokens, per_token, total, local = 6, 2, 8, 2
        real_indices = torch.zeros(tokens, per_token, dtype=torch.int64)
        real_weights = torch.arange(
            tokens * per_token, dtype=torch.float32).view(tokens, per_token)

        original = routing_module.MOE_ROUTER_ADAPTERS["deepseekv3"]
        routing_module.MOE_ROUTER_ADAPTERS["deepseekv3"] = (
            lambda module, hidden_states: (real_indices.clone(), real_weights)
        )
        try:
            indices, weights = ep_compute._balanced_router(  # pylint: disable=protected-access
                _FakeMoE(local, total), torch.zeros(1))
        finally:
            routing_module.MOE_ROUTER_ADAPTERS["deepseekv3"] = original

        self.assertTrue(torch.equal(weights, real_weights))
        self.assertFalse(torch.equal(indices, real_indices))
        expected = ep_compute._balanced_expert_indices(  # pylint: disable=protected-access
            tokens, per_token, total, local, device=indices.device)
        self.assertTrue(torch.equal(indices.to(torch.int64), expected))


class TestFactorySwitch(unittest.TestCase):
    """``fix_router`` selects the router the archetype is built with."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fix_router_selects_the_balanced_router(self) -> None:
        """On by switch, off by default; the stub captures what was passed."""
        captured = {}
        original = ep_compute.build_ep_compute

        def _stub(_module, _ep_mesh, *, router_fn, **_kwargs):
            """Record the router the factory selected."""
            captured["router_fn"] = router_fn
            return lambda *args, **kw: None

        ep_compute.build_ep_compute = _stub
        try:
            ep_compute.kimi_k26_ep_compute_fn(
                module=_FakeMoE(2, 8), mesh=None, tp_mesh=None, cp_mesh=None,
                ep_mesh=None, fix_router=True)
            self.assertIs(captured["router_fn"], ep_compute._balanced_router)  # pylint: disable=protected-access

            ep_compute.kimi_k26_ep_compute_fn(
                module=_FakeMoE(2, 8), mesh=None, tp_mesh=None, cp_mesh=None,
                ep_mesh=None)
            self.assertIs(captured["router_fn"],
                          routing_module.MOE_ROUTER_ADAPTERS["deepseekv3"])
        finally:
            ep_compute.build_ep_compute = original


if __name__ == "__main__":
    unittest.main()
