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
"""Switch wiring of the Kimi-K2.6 EP archetype factory.

The factory must forward ``fix_router`` / ``overlap_shared_expert`` /
``capacity_factor`` to the public ``build_ep_compute`` skeleton and must NOT carry its
own copy of the balanced router: the router contract itself (arithmetic, generality,
``<name>_fixed`` registry) is pinned in
``tests/ut/core/expert_parallel/test_ep_balanced_router.py``.
"""

import os
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch  # pylint: disable=wrong-import-position

from tests.common.mark_utils import arg_mark  # pylint: disable=wrong-import-position

from hyper_parallel.models.kimi_k26.adapter.distributed import (  # pylint: disable=wrong-import-position
    ep_compute,
)


# These stub modules never run a forward pass; they only expose the attributes
# the archetype reads, so the abstract ``forward`` hook is left out.
class _FakeExperts(torch.nn.Module):  # pylint: disable=abstract-method
    """Stand-in exposing the expert layout the archetype reads."""

    def __init__(self, local_expert_count: int, num_experts: int) -> None:
        """Store the expert layout the factory reads back."""
        super().__init__()
        self.local_expert_count = local_expert_count
        self.num_experts = num_experts


# Minimal attribute carrier for the archetype; no forward pass is ever executed.
class _FakeMoE(torch.nn.Module):  # pylint: disable=abstract-method
    """Minimal MoE block carrying an ``experts`` child."""

    def __init__(self, local_expert_count: int, num_experts: int) -> None:
        """Expose an ``experts`` child with the given layout."""
        super().__init__()
        self.experts = _FakeExperts(local_expert_count, num_experts)


class TestFactorySwitch(unittest.TestCase):
    """``fix_router`` reaches the framework resolver through the archetype."""

    @staticmethod
    def _capture(**kwargs):
        """Build the archetype with ``build_ep_compute`` stubbed out, return its kwargs."""
        captured = {}
        original = ep_compute.build_ep_compute

        # The leading underscores mark the positional slots as intentionally
        # unused (``ignored-argument-names``); the stub only records the kwargs.
        def _stub(_module, _ep_mesh, **stub_kwargs):  # pylint: disable=invalid-name
            """Record the keyword arguments the archetype passed."""
            captured.clear()
            captured.update(stub_kwargs)
            return lambda *args, **kw: None

        ep_compute.build_ep_compute = _stub
        try:
            ep_compute.kimi_k26_ep_compute_fn(
                module=_FakeMoE(2, 8), mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=None,
                **kwargs)
        finally:
            ep_compute.build_ep_compute = original
        return captured

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fix_router_is_forwarded_to_the_framework(self) -> None:
        """The archetype names its router and asks the skeleton to resolve the switch."""
        default = self._capture()
        self.assertEqual(default["router_fn"], "deepseekv3")
        self.assertFalse(default["fix_router"])

        asked = self._capture(fix_router=True)
        self.assertEqual(asked["router_fn"], "deepseekv3")
        self.assertTrue(asked["fix_router"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_family_carries_no_private_balanced_router(self) -> None:
        """One implementation, in the framework: a family copy would drift from it."""
        self.assertFalse(hasattr(ep_compute, "_balanced_router"))
        self.assertFalse(hasattr(ep_compute, "_balanced_expert_indices"))


class TestOverlapAndCapacitySwitches(unittest.TestCase):
    """``overlap_shared_expert`` / ``capacity_factor`` reach the EP factory.

    The 2-SN reference numbers are measured WITH the shared-expert overlap, and the
    family archetype is the entry point the recipe uses, so the archetype must forward
    both switches; before this was covered, the recipe silently built a non-overlapped
    plan (measured: +2.28 GB peak, +22% step time at GBS 256).
    """

    @staticmethod
    def _capture(**kwargs):
        """Build the archetype with ``build_ep_compute`` stubbed out, return its kwargs."""
        captured = {}
        original = ep_compute.build_ep_compute

        # The leading underscores mark the positional slots as intentionally
        # unused (``ignored-argument-names``); the stub only records the kwargs.
        def _stub(_module, _ep_mesh, **stub_kwargs):  # pylint: disable=invalid-name
            """Record the keyword arguments the archetype passed."""
            captured.clear()
            captured.update(stub_kwargs)
            return lambda *args, **kw: None

        ep_compute.build_ep_compute = _stub
        try:
            ep_compute.kimi_k26_ep_compute_fn(
                module=_FakeMoE(2, 8), mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=None,
                **kwargs)
        finally:
            ep_compute.build_ep_compute = original
        return captured

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_both_switches_default_off_and_are_forwarded(self) -> None:
        """Off by default; on when asked for, as the recipe asks for the overlap."""
        default = self._capture()
        self.assertIsNone(default["overlap_fn"])
        self.assertIsNone(default["capacity_factor"])

        asked = self._capture(overlap_shared_expert=True, capacity_factor=2.0)
        self.assertTrue(callable(asked["overlap_fn"]))
        self.assertEqual(asked["capacity_factor"], 2.0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_combine_uses_the_overlapped_branch_when_it_is_given_one(self) -> None:
        """The overlap is real only if ``combine`` consumes the precomputed branch.

        Expectation: without the overlapped value ``combine`` evaluates
            ``shared_experts`` itself; with it, that branch is used as is -- the
            contract that lets the factory run it while the exchange is in flight.
        """
        captured = self._capture(overlap_shared_expert=True)
        hidden = torch.randn(3, 4)
        shared = torch.randn(3, 4)
        routed = torch.randn(3, 4)

        class _Block:
            """Stand-in whose shared branch is recognisable in the output."""

            @staticmethod
            def shared_experts(states: torch.Tensor) -> torch.Tensor:
                """Return the marker branch offset by the states."""
                return shared + states

        computed = captured["combine"](_Block, hidden, routed)
        # Grouped the way the archetype groups it: the two forms differ only in float
        # addition order, which is not the contract under test.
        self.assertTrue(torch.allclose(computed, routed + (shared + hidden)))
        overlapped = captured["combine"](_Block, hidden, routed, shared)
        self.assertTrue(torch.equal(overlapped, routed + shared))


if __name__ == "__main__":
    unittest.main()
