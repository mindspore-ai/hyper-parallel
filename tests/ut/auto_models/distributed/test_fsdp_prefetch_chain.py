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
"""The FSDP2 prefetch chain must skip units that issue no collectives.

``_order_wrap_modules`` returns units in ``named_modules()`` traversal order,
which interleaves every transformer block with a routed-experts container
nested inside it::

    [L0, L0.experts, L1, L1.experts, ...]

At ``ep == world size`` that container's shard group holds a single rank, so
its unshard and reshard are no-ops -- but it still occupied a slot in the
chain. That made every real prefetch one link late, and backward prefetch was
not merely late but unreachable: a block is prefetched in backward by the unit
registered after it, which is its own nested container, and that container's
backward hook fires after the block's own. The block then unshards
synchronously inside its backward hook.

These tests pin the chain, not the hooks: they assert what each unit is told to
prefetch. Nothing here needs a device or a process group.
"""
# pylint: disable=wrong-import-position

import os
import unittest
from typing import Any, List, Optional

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.models.build_options import FSDP2Config
from tests.common.mark_utils import arg_mark


class _FakeParam:
    """Stands in for ``TorchHSDPParamV2``; only the shard width is read."""

    def __init__(self, shard_world_size: int) -> None:
        """Record the shard width of this parameter."""
        self.shard_world_size = shard_world_size


class _FakeState:
    """Stands in for the HSDP state: a parameter list is all that is read."""

    def __init__(self, shard_world_size: int) -> None:
        """Build a one-parameter state with ``shard_world_size``."""
        self.hsdp_params = [_FakeParam(shard_world_size)]


class _FakeScheduler:
    """Stands in for the HSDP scheduler's state holder."""

    def __init__(self, shard_world_size: int) -> None:
        """Build a scheduler whose state reports ``shard_world_size``."""
        self.hsdp_state = _FakeState(shard_world_size)


class _FakeUnit:
    """An FSDP unit that records the prefetch targets it is given."""

    def __init__(self, name: str, shard_world_size: int) -> None:
        """Build a named unit with the given shard width."""
        self.name = name
        self.hsdp_scheduler = _FakeScheduler(shard_world_size)
        self.forward_prefetch = None
        self.backward_prefetch = None

    def set_modules_to_forward_prefetch(self, modules: Any) -> None:
        """Record the forward prefetch targets."""
        self.forward_prefetch = list(modules)

    def set_modules_to_backward_prefetch(self, modules: Any) -> None:
        """Record the backward prefetch targets."""
        self.backward_prefetch = list(modules)

    def __repr__(self) -> str:
        """Render the unit by name for assertion messages."""
        return f"<{self.name}>"


SHARDED, SINGLE_RANK = 128, 1


def _units(*specs) -> list:
    """Build fake units from ``(name, shard_world_size)`` pairs."""
    return [_FakeUnit(name, width) for name, width in specs]


def _names(units: Any) -> Optional[List]:
    """Return the prefetch target names; ``None`` in stays ``None`` out."""
    if units is None:
        return None
    return [unit.name if unit is not None else None for unit in units]


def _interleaved_chain() -> list:
    """The order ``_order_wrap_modules`` produces for a 3-block MoE model."""
    return _units(
        ("L0", SHARDED), ("L0.experts", SINGLE_RANK),
        ("L1", SHARDED), ("L1.experts", SINGLE_RANK),
        ("L2", SHARDED), ("L2.experts", SINGLE_RANK),
    )


def _configure(units):
    """Run the prefetch configuration under test and return the units."""
    FSDP2Manager(FSDP2Config(), None)._configure_prefetch(units)
    return units


class TestPrefetchChainSkipsInertUnits(unittest.TestCase):
    """Single-rank units are not prefetch links; the sharded ones are."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_forward_prefetch_reaches_the_next_sharded_unit(self):
        """L0 prefetches L1, not the L0.experts no-op that sits between them."""
        chain = _configure(_interleaved_chain())
        dense = [unit for unit in chain if unit.name.endswith("experts") is False]

        self.assertEqual(
            _names(dense[0].forward_prefetch),
            ["L1"],
            f"L0 must prefetch the next sharded unit, got {_names(dense[0].forward_prefetch)}",
        )
        self.assertEqual(
            _names(dense[1].forward_prefetch),
            ["L2"],
            f"L1 must prefetch the next sharded unit, got {_names(dense[1].forward_prefetch)}",
        )
        self.assertEqual(
            _names(dense[2].forward_prefetch),
            [],
            f"the last sharded unit has no target, got {_names(dense[2].forward_prefetch)}",
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_backward_prefetch_reaches_the_previous_sharded_unit(self):
        """L1 prefetches L0's unshard -- the regression this pins.

        With the inert units in the chain this target was ``L0.experts``, whose
        backward hook fires after L1's own, so L1 blocked on an all-gather that
        nobody had issued early.
        """
        chain = _configure(_interleaved_chain())
        dense = [unit for unit in chain if unit.name.endswith("experts") is False]

        self.assertEqual(
            _names(dense[0].backward_prefetch),
            [],
            f"the first sharded unit has no target, got {_names(dense[0].backward_prefetch)}",
        )
        self.assertEqual(
            _names(dense[1].backward_prefetch),
            ["L0"],
            f"L1 must prefetch L0 in backward, got {_names(dense[1].backward_prefetch)}",
        )
        self.assertEqual(
            _names(dense[2].backward_prefetch),
            ["L1"],
            f"L2 must prefetch L1 in backward, got {_names(dense[2].backward_prefetch)}",
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_single_rank_units_are_never_given_targets(self):
        """An inert unit is not a prefetcher either, so it is left untouched."""
        chain = _configure(_interleaved_chain())
        inert = [unit for unit in chain if unit.name.endswith("experts")]

        for unit in inert:
            with self.subTest(unit=unit.name):
                self.assertIsNone(
                    unit.forward_prefetch,
                    f"inert unit {unit.name} must not prefetch, "
                    f"got {_names(unit.forward_prefetch)}",
                )
                self.assertIsNone(
                    unit.backward_prefetch,
                    f"inert unit {unit.name} must not prefetch, "
                    f"got {_names(unit.backward_prefetch)}",
                )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dense_only_models_are_unaffected(self):
        """With every unit sharded the chain is the traversal order, as before."""
        chain = _configure(_units(
            ("L0", SHARDED), ("L1", SHARDED), ("L2", SHARDED),
        ))

        self.assertEqual(
            _names(chain[0].forward_prefetch),
            ["L1"],
            f"dense chain forward prefetch changed: {_names(chain[0].forward_prefetch)}",
        )
        self.assertEqual(
            _names(chain[1].backward_prefetch),
            ["L0"],
            f"dense chain backward prefetch changed: {_names(chain[1].backward_prefetch)}",
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_a_unit_without_scheduler_state_is_treated_as_inert(self):
        """No state means every unshard is a no-op, so it is not a link."""
        units = _units(("L0", SHARDED), ("L1", SHARDED))
        units[1].hsdp_scheduler.hsdp_state = None
        chain = _configure(units)

        self.assertIsNone(
            units[1].forward_prefetch,
            f"a stateless unit must not prefetch, got {_names(units[1].forward_prefetch)}",
        )
        self.assertEqual(
            _names(units[0].forward_prefetch),
            [],
            f"the only sharded unit has no target, got {_names(units[0].forward_prefetch)}",
        )


if __name__ == "__main__":
    unittest.main()
