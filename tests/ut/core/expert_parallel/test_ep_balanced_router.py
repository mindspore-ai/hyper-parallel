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
"""Contract of the generic ``fix_router`` switch.

``make_balanced_router`` wraps *any* real router into a benchmark-only variant that
hands every destination rank the same slot count -- that is what stops a skewed real
router from dominating step time -- while keeping the real gate weights
differentiable. ``resolve_moe_router`` is the one place the switch is turned into a
router, so every archetype factory can expose it without re-implementing the wrapper.

These tests pin the arithmetic, the wrapper's generality and the switch wiring with no
distributed setup.
"""

import os
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch  # pylint: disable=wrong-import-position

from tests.common.mark_utils import arg_mark  # pylint: disable=wrong-import-position

from hyper_parallel.distributed.expert_parallel import (  # pylint: disable=wrong-import-position
    recipes as recipes_module,
)
from hyper_parallel.distributed.expert_parallel import (  # pylint: disable=wrong-import-position
    routing as routing_module,
)


class _FakeExperts(torch.nn.Module):  # pylint: disable=abstract-method
    """Stand-in exposing the expert layout the balanced router reads."""

    def __init__(self, local_expert_count: int, num_experts: int) -> None:
        """Store the layout the router reads back."""
        super().__init__()
        self.local_expert_count = local_expert_count
        self.num_experts = num_experts


class _FakeMoE(torch.nn.Module):  # pylint: disable=abstract-method
    """Minimal MoE block carrying an ``experts`` child."""

    def __init__(self, local_expert_count: int, num_experts: int) -> None:
        """Expose an ``experts`` child with the given layout."""
        super().__init__()
        self.experts = _FakeExperts(local_expert_count, num_experts)


class _StubRouter:
    """Real-router stand-in: records that it ran, returns fixed shapes."""

    def __init__(self, tokens: int, per_token: int) -> None:
        """Fix the shapes and the weights the stub hands back."""
        self.calls = 0
        self.weights = torch.arange(
            tokens * per_token, dtype=torch.float32).view(tokens, per_token)
        self.indices = torch.zeros(tokens, per_token, dtype=torch.int64)

    def __call__(self, module, hidden_states):
        """Return the recorded plan, ignoring the inputs like a fake would."""
        del module, hidden_states
        self.calls += 1
        return self.indices.clone(), self.weights.clone()


class TestBalancedExpertIndices(unittest.TestCase):
    """The round-robin assignment itself."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_spreads_slots_evenly_over_destination_ranks(self) -> None:
        """Every destination rank gets the same slot count; experts differ by <=1."""
        tokens, per_token, total, local = 10, 2, 8, 2
        indices = routing_module._balanced_expert_indices(  # pylint: disable=protected-access
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
            routing_module._balanced_expert_indices(  # pylint: disable=protected-access
                4, 1, 6, 4, device=torch.device("cpu"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_a_non_positive_local_count(self) -> None:
        """A rank that owns no expert cannot take slots; fail instead of dividing by 0."""
        with self.assertRaises(ValueError):
            routing_module._balanced_expert_indices(  # pylint: disable=protected-access
                4, 1, 8, 0, device=torch.device("cpu"))


class TestMakeBalancedRouter(unittest.TestCase):
    """The wrapper is generic: it balances whatever real router it is given."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_wraps_any_real_router_and_keeps_its_weights(self) -> None:
        """The real router still runs; only the destinations are replaced.

        This is the generality that lets a family reuse the switch: the wrapper never
        assumes sigmoid-group routing, it only needs ``(topk_idx, topk_w)``.
        """
        tokens, per_token, total, local = 6, 2, 8, 2
        real = _StubRouter(tokens, per_token)

        balanced = routing_module.make_balanced_router(real)
        indices, weights = balanced(_FakeMoE(local, total), torch.zeros(1))

        self.assertEqual(real.calls, 1, "the real router must still run (gate gradient)")
        self.assertTrue(torch.equal(weights, real.weights),
                        "the real gate weights must be passed through unchanged")
        expected = routing_module._balanced_expert_indices(  # pylint: disable=protected-access
            tokens, per_token, total, local, device=indices.device)
        self.assertTrue(torch.equal(indices.to(torch.int64), expected))

        destinations = torch.div(indices.reshape(-1), local, rounding_mode="floor")
        per_destination = torch.bincount(destinations, minlength=total // local)
        self.assertEqual(len(set(per_destination.tolist())), 1,
                         f"destination load is not uniform: {per_destination.tolist()}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_names_the_wrapper_after_the_real_router(self) -> None:
        """The variant is identifiable in a log line."""
        balanced = routing_module.make_balanced_router(_StubRouter(2, 1))
        self.assertEqual(balanced.__name__, "balanced__StubRouter")


class TestResolveMoeRouter(unittest.TestCase):
    """The single place the ``fix_router`` flag becomes a router."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_name_resolves_plain_and_fixed(self) -> None:
        """A registry name maps to itself, or to its ``_fixed`` variant."""
        self.assertIs(routing_module.resolve_moe_router("mixtral"),
                      routing_module.MOE_ROUTER_ADAPTERS["mixtral"])
        self.assertIs(routing_module.resolve_moe_router("mixtral", fix_router=True),
                      routing_module.MOE_ROUTER_ADAPTERS["mixtral_fixed"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_every_registered_adapter_has_a_fixed_variant(self) -> None:
        """``<name>_fixed`` exists for every alias, and there is no double suffix."""
        for name in routing_module.MOE_ROUTER_ADAPTERS:
            if name.endswith("_fixed"):
                continue
            fixed = routing_module.MOE_ROUTER_ADAPTERS[f"{name}_fixed"]
            self.assertTrue(callable(fixed), f"{name}_fixed is not callable")
        self.assertFalse(any(name.endswith("_fixed_fixed")
                             for name in routing_module.MOE_ROUTER_ADAPTERS))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_deepseekv3_fixed_is_the_archived_router(self) -> None:
        """The benchmark router the recorded 2-SN numbers used is unchanged."""
        self.assertIs(routing_module.MOE_ROUTER_ADAPTERS["deepseekv3_fixed"],
                      routing_module._balanced_router)  # pylint: disable=protected-access
        self.assertIs(routing_module.MOE_ROUTER_ADAPTERS["deepseek_v3_fixed"],
                      routing_module._balanced_router)  # pylint: disable=protected-access

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_callable_is_wrapped_only_when_asked(self) -> None:
        """A family that carries its own adapter still gets the switch for free."""
        real = _StubRouter(2, 1)
        self.assertIs(routing_module.resolve_moe_router(real), real)
        wrapped = routing_module.resolve_moe_router(real, fix_router=True)
        self.assertIsNot(wrapped, real)
        self.assertEqual(wrapped.__name__, "balanced__StubRouter")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unknown_name_and_bad_type_raise(self) -> None:
        """A typo must not silently keep the real router."""
        with self.assertRaises(KeyError):
            routing_module.resolve_moe_router("not-a-router")
        with self.assertRaises(TypeError):
            routing_module.resolve_moe_router(42)


class TestBuildEpComputeSwitch(unittest.TestCase):
    """The skeleton applies the switch, so every archetype inherits it."""

    @staticmethod
    def _build(fix_router):
        """Build a routed-only plan with the dispatch and binder stubbed out."""
        captured = {}
        original_bind = recipes_module.bind_local_expert_forward
        original_forward = recipes_module.ep_routed_forward

        def _bind(*args, **kwargs):
            """Record the binder call, then do nothing (no expert forward needed)."""
            del kwargs
            captured["bound"] = args

        def _forward(module, hidden_states, router_fn=None, ep_group=None):
            """Record the router the skeleton resolved."""
            del module, hidden_states, ep_group
            captured["router_fn"] = router_fn
            return torch.zeros(1)

        class _Axis:
            """The single mesh surface ``build_ep_compute`` touches."""

            @staticmethod
            def size():
                """Expert count along the ``ep`` axis."""
                return 2

        class _Mesh:
            """Minimal EP mesh."""

            @staticmethod
            def get_group(name):
                """Return a placeholder process group."""
                del name
                return "ep-group"

            @staticmethod
            def __getitem__(key):
                """Return the ``ep`` axis stand-in."""
                del key
                return _Axis()

        module = _FakeMoE(local_expert_count=2, num_experts=8)
        module.gate = torch.nn.Linear(1, 1, bias=False)
        recipes_module.bind_local_expert_forward = _bind
        recipes_module.ep_routed_forward = _forward
        try:
            compute = recipes_module.build_ep_compute(
                module, _Mesh(), router_fn="mixtral", archetype_key="test_archetype",
                expected_attrs=["gate", "experts"],
                combine=lambda mod, hidden, routed: routed, fix_router=fix_router)
            compute(module, torch.zeros(1, 1))
        finally:
            recipes_module.bind_local_expert_forward = original_bind
            recipes_module.ep_routed_forward = original_forward
        return captured

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fix_router_selects_the_fixed_adapter(self) -> None:
        """Off keeps the real router; on swaps in the registered variant."""
        off = self._build(fix_router=False)
        self.assertIs(off["router_fn"], routing_module.MOE_ROUTER_ADAPTERS["mixtral"])
        on = self._build(fix_router=True)
        self.assertIs(on["router_fn"], routing_module.MOE_ROUTER_ADAPTERS["mixtral_fixed"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_callable_router_is_balanced_by_the_skeleton(self) -> None:
        """A family passing its own adapter gets the balanced variant too."""
        captured = {}
        original_forward = recipes_module.ep_routed_forward

        def _forward(module, hidden_states, router_fn=None, ep_group=None):
            """Record the resolved router."""
            del module, hidden_states, ep_group
            captured["router_fn"] = router_fn
            return torch.zeros(1)

        real = _StubRouter(2, 1)
        module = _FakeMoE(local_expert_count=2, num_experts=8)
        module.gate = torch.nn.Linear(1, 1, bias=False)

        class _Axis:
            """The mesh axis stand-in."""

            @staticmethod
            def size():
                """Expert count along the ``ep`` axis."""
                return 2

        class _Mesh:
            """Minimal EP mesh."""

            @staticmethod
            def get_group(name):
                """Return a placeholder process group."""
                del name
                return "ep-group"

            @staticmethod
            def __getitem__(key):
                """Return the ``ep`` axis stand-in."""
                del key
                return _Axis()

        recipes_module.bind_local_expert_forward = lambda *args, **kwargs: None
        recipes_module.ep_routed_forward = _forward
        try:
            compute = recipes_module.build_ep_compute(
                module, _Mesh(), router_fn=real, archetype_key="test_archetype",
                expected_attrs=["gate", "experts"],
                combine=lambda mod, hidden, routed: routed, fix_router=True)
            compute(module, torch.zeros(1, 1))
        finally:
            recipes_module.ep_routed_forward = original_forward
        self.assertIsNot(captured["router_fn"], real)
        self.assertEqual(captured["router_fn"].__name__, "balanced__StubRouter")


if __name__ == "__main__":
    unittest.main()
