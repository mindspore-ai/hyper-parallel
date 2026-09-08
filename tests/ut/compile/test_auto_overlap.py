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
"""Unit tests for ``hyper_parallel.compile.passes.overlap.schedule.AutoOverlapPass``.

Covers the execution-layer contract of the pass (previously exercised only at
pipeline-assembly level in ``test_pass_pipeline``):

1. ``run`` early-returns when ``enable_overlap`` is ``False`` (it must not
   even scan the graph).
2. ``run`` with ``enable_overlap=True`` visits every ``wait_tensor`` node;
   today ``_move_wait_later`` is a placeholder no-op, so the graph is
   returned structurally unchanged.
3. ``run`` on a graph with no ``wait_tensor`` nodes is a safe no-op.
4. ``_find_wait_nodes`` returns only ``wait_tensor`` call_function nodes
   (skipping ordinary compute ops), and returns ``[]`` when none are present.
5. ``_move_wait_later`` is a documented no-op today (locks the current
   contract until the overlap scheduler lands).

The pass touches no ``torch.distributed`` state, so no dist mocking is
required -- only FX graph construction.
"""

import os
import unittest
from unittest.mock import patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch
from torch import fx
from torch.ops import _c10d_functional

from hyper_parallel.compile.parallel_config import PassConfig
from hyper_parallel.compile.passes.overlap.schedule import AutoOverlapPass


def _wait_tensor_graph() -> fx.GraphModule:
    """Build a tiny graph ``f(x) = add(wait_tensor(x), x)``.

    The ``wait_tensor`` node uses the same op ``FSDPPass`` inserts, so
    ``AutoOverlapPass._find_wait_nodes`` matches it by the ``wait_tensor``
    substring of ``str(node.target)``. The ``torch.add`` node is an ordinary
    compute op the finder must skip.
    """
    graph = fx.Graph()
    x = graph.placeholder("x")
    wait = graph.call_function(_c10d_functional.wait_tensor, args=(x,))
    out = graph.call_function(torch.add, args=(wait, x))
    graph.output(out)
    return fx.GraphModule({}, graph)


def _plain_graph() -> fx.GraphModule:
    """Build a graph with no ``wait_tensor`` node (``f(x) = x + 1``)."""

    def f(x: torch.Tensor) -> torch.Tensor:
        """Return the input plus one (a wait-free compute graph)."""
        return x + 1

    return torch.fx.symbolic_trace(f)


def _count_targets(gm: fx.GraphModule, needle: str) -> int:
    """Count ``call_function`` nodes whose ``str(target)`` contains ``needle``."""
    return sum(
        1 for n in gm.graph.nodes if n.op == "call_function" and needle in str(n.target)
    )


class TestAutoOverlapRun(unittest.TestCase):
    """``AutoOverlapPass.run`` honours ``enable_overlap`` and visits wait nodes."""

    def test_run_disabled_returns_early_without_scanning(self):
        """Test ``enable_overlap=False`` returns the graph without scanning it.

        The early return must happen before ``_find_wait_nodes`` runs, so a
        graph that *would* contain wait nodes is handed back untouched.
        """
        cfg = PassConfig(enable_overlap=False)
        overlap = AutoOverlapPass()
        gm = _wait_tensor_graph()

        with patch.object(overlap, "_find_wait_nodes") as mock_find:
            result = overlap.run(gm, cfg)
            mock_find.assert_not_called()

        self.assertIs(
            result,
            gm,
            "run() should return the same graph_module it was given",
        )
        self.assertEqual(
            _count_targets(gm, "wait_tensor"),
            1,
            (
                f"disabled pass must not touch the graph, "
                f"got wait_tensor count={_count_targets(gm, 'wait_tensor')}"
            ),
        )

    def test_run_enabled_visits_every_wait_node(self):
        """Test ``enable_overlap=True`` scans the graph and visits each wait.

        ``_move_wait_later`` is a no-op today, so the wait node stays where
        ``FSDPPass`` placed it and the returned graph is the same object.
        """
        cfg = PassConfig(enable_overlap=True)
        overlap = AutoOverlapPass()
        gm = _wait_tensor_graph()

        result = overlap.run(gm, cfg)

        self.assertIs(
            result,
            gm,
            "run() should return the same graph_module it was given",
        )
        self.assertEqual(
            _count_targets(gm, "wait_tensor"),
            1,
            (
                f"no-op move must not drop the wait node, "
                f"got wait_tensor count={_count_targets(gm, 'wait_tensor')}"
            ),
        )

    def test_run_enabled_on_graph_without_wait_nodes_is_noop(self):
        """Test ``run`` on a graph with no wait nodes is a safe no-op."""
        cfg = PassConfig(enable_overlap=True)
        overlap = AutoOverlapPass()
        gm = _plain_graph()

        result = overlap.run(gm, cfg)

        self.assertIs(
            result,
            gm,
            "run() on a wait-free graph should return the same graph_module",
        )
        self.assertEqual(
            _count_targets(gm, "wait_tensor"),
            0,
            (
                f"plain graph has no wait nodes, "
                f"got wait_tensor count={_count_targets(gm, 'wait_tensor')}"
            ),
        )


class TestAutoOverlapFindWaitNodes(unittest.TestCase):
    """``_find_wait_nodes`` returns only ``wait_tensor`` call_function nodes."""

    def test_find_returns_only_wait_tensor_calls(self):
        """Test the finder matches the wait_tensor op and skips ordinary ops."""
        overlap = AutoOverlapPass()
        gm = _wait_tensor_graph()

        waits = overlap._find_wait_nodes(gm.graph)  # pylint: disable=protected-access

        self.assertEqual(
            len(waits),
            1,
            (f"graph has one wait_tensor node, got {len(waits)} matches"),
        )
        self.assertIn(
            "wait_tensor",
            str(waits[0].target),
            (
                f"matched node target should contain 'wait_tensor', "
                f"got {str(waits[0].target)}"
            ),
        )

    def test_find_empty_when_no_wait_nodes(self):
        """Test the finder returns ``[]`` on a graph without wait_tensor nodes."""
        overlap = AutoOverlapPass()
        gm = _plain_graph()

        waits = overlap._find_wait_nodes(gm.graph)  # pylint: disable=protected-access

        self.assertEqual(
            waits,
            [],
            (f"plain graph has no wait nodes, got {len(waits)} matches"),
        )


class TestAutoOverlapMoveWaitLater(unittest.TestCase):
    """``_move_wait_later`` is a documented placeholder no-op today."""

    def test_move_wait_later_is_noop_today(self):
        """Test ``_move_wait_later`` returns the graph unchanged (current contract).

        Locks the current no-op behaviour so the transition to a real overlap
        scheduler is a deliberate, visible change rather than a silent one.
        """
        overlap = AutoOverlapPass()
        cfg: PassConfig = PassConfig(enable_overlap=True)
        gm = _wait_tensor_graph()
        wait = overlap._find_wait_nodes(gm.graph)[0]  # pylint: disable=protected-access

        result = overlap._move_wait_later(gm, wait)  # pylint: disable=protected-access

        self.assertIs(
            result,
            gm,
            "_move_wait_later should return the same graph_module (no-op today)",
        )
        self.assertEqual(
            _count_targets(gm, "wait_tensor"),
            1,
            (
                f"no-op move must keep the wait node in place, "
                f"got wait_tensor count={_count_targets(gm, 'wait_tensor')}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
