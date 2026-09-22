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
"""Unit tests for MegaMoe route preparation."""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe import route as route_module
from hyper_parallel.core.multicore.modules.mega_moe.route import (
    _expert_capacity,
    _resolve_counts,
    _validate_bounded_capacity,
    prepare_topk_route,
)
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec


class TestMegaMoeRoute(unittest.TestCase):
    """Validate trusted Router counts and bounded capacity without hardware."""

    @staticmethod
    def _spec(
        *,
        ep_size: int = 1,
        rank_id: int = 0,
        expert_capacity_factor: float | None = None,
        receive_capacity: int = 128,
    ) -> MegaMoeSpec:
        """Build a small CPU-only route specification."""
        return MegaMoeSpec(
            local_num_tokens=2,
            hidden_size=4,
            intermediate_size=2,
            num_experts=4,
            top_k=2,
            expert_capacity_factor=expert_capacity_factor,
            receive_capacity=receive_capacity,
            ep_size=ep_size,
            ep_group=None,
            rank_id=rank_id,
            num_cube_cores=24,
        )

    def test_router_counts_are_reused_or_computed_only_when_omitted(self) -> None:
        """Reuse supplied counts and compute a histogram only when omitted."""
        spec = self._spec()
        flat_ids = torch.tensor([0, 1, 1, 3], dtype=torch.int32)
        supplied_counts = torch.tensor([4, 0, 0, 0], dtype=torch.int32)

        with patch.object(torch, "bincount") as mock_bincount:
            supplied = _resolve_counts(flat_ids, supplied_counts, spec)

        mock_bincount.assert_not_called()
        self.assertIs(supplied, supplied_counts)
        computed = _resolve_counts(flat_ids, None, spec)
        expected = torch.tensor([1, 2, 0, 1], dtype=torch.int32)
        self.assertTrue(
            torch.equal(computed, expected),
            f"computed counts mismatch: expected={expected}, got={computed}",
        )

    def test_capacity_checks_only_explicit_bounded_mode(self) -> None:
        """Reuse the gathered maximum and reject explicit bounded overflow."""
        spec = self._spec(ep_size=2, expert_capacity_factor=None)
        _validate_bounded_capacity(8, spec)
        bounded_spec = self._spec(
            ep_size=2,
            expert_capacity_factor=1.0,
            receive_capacity=4,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "configured_capacity=4, actual_maximum=8",
        ):
            _validate_bounded_capacity(8, bounded_spec)

    def test_async_count_gather_overlaps_permute_and_builds_metadata(self) -> None:
        """Wait after permutation and derive every native offset from counts."""
        spec = self._spec(ep_size=2, rank_id=1)
        hidden_states = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32)
        supplied_counts = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
        global_counts = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            dtype=torch.int32,
        )
        routed_tokens = hidden_states.repeat_interleave(2, dim=0)
        unpermute_mapping = torch.arange(4, dtype=torch.int32)
        events = []
        workspace = Mock()
        workspace.wait_for_reuse.side_effect = lambda: events.append("reuse")
        work = Mock()
        work.wait.side_effect = lambda: events.append("wait")

        def gather_counts(output: Any, input_tensor: Any, **kwargs: Any) -> Any:
            """Populate the mocked rank-major gather output."""
            self.assertIs(input_tensor, supplied_counts)
            self.assertTrue(kwargs["async_op"])
            events.append("gather")
            output.copy_(global_counts.reshape(-1))
            return work

        def permute_input(
            input_hidden_states: Any,
            input_topk_ids: Any,
        ) -> tuple[Any, Any]:
            """Record the mocked payload permutation launch."""
            self.assertIs(input_hidden_states, hidden_states)
            self.assertIs(input_topk_ids, topk_ids)
            events.append("permute")
            return routed_tokens, unpermute_mapping

        with (
            patch.object(
                route_module.dist,
                "all_gather_into_tensor",
                side_effect=gather_counts,
            ) as mock_gather,
            patch.object(
                route_module,
                "_permute_topk_input",
                side_effect=permute_input,
            ),
        ):
            route = prepare_topk_route(
                hidden_states,
                topk_ids,
                topk_weights,
                spec,
                supplied_counts,
                workspace=workspace,
            )

        self.assertEqual(events, ["reuse", "gather", "permute", "wait"])
        self.assertTrue(mock_gather.call_args.kwargs["async_op"])
        self.assertIs(route.routed_tokens, routed_tokens)
        self.assertIs(route.unpermute_mapping, unpermute_mapping)
        self.assertTrue(
            torch.equal(
                route.received_counts,
                torch.tensor([[3, 4], [7, 8]], dtype=torch.int32),
            )
        )
        metadata = route.metadata
        self.assertTrue(
            torch.equal(metadata.dispatch_src_off, torch.tensor([0, 5, 11, 18]))
        )
        self.assertTrue(
            torch.equal(metadata.dispatch_target_off, torch.tensor([1, 8, 3, 14]))
        )
        self.assertTrue(torch.equal(metadata.dispatch_size, supplied_counts))
        self.assertTrue(
            torch.equal(metadata.combine_src_off, torch.tensor([0, 10, 3, 14]))
        )
        self.assertTrue(
            torch.equal(metadata.combine_target_off, torch.tensor([3, 6, 11, 18]))
        )
        self.assertTrue(torch.equal(metadata.combine_size, torch.tensor([3, 4, 7, 8])))
        self.assertTrue(torch.equal(metadata.group_list, torch.tensor([10, 22])))
        self.assertEqual(metadata.expert_capacity, 22)

    def test_local_capacity_tracks_routes_below_source_size_and_empty_ranks(self) -> None:
        """Shrink destination intermediates independently of source or SHMEM size."""
        spec = self._spec(ep_size=2, receive_capacity=256)
        routes = (
            ([[1, 1, 1, 1], [1, 1, 1, 1]], (4, 4)),
            ([[2, 1, 1, 0], [2, 1, 1, 0]], (6, 2)),
            ([[2, 2, 0, 0], [2, 2, 0, 0]], (8, 1)),
            ([[0, 0, 2, 2], [0, 0, 2, 2]], (1, 8)),
        )
        for counts, capacities in routes:
            for rank, expected in enumerate(capacities):
                with self.subTest(counts=counts, rank=rank):
                    rank_spec = replace(spec, rank_id=rank)
                    actual = _expert_capacity(torch.tensor(counts, dtype=torch.int32), rank_spec)
                    self.assertEqual(actual, expected)
                    self.assertEqual(rank_spec.receive_capacity, 256)
                    self.assertEqual(rank_spec.routed_slots, 4)

    def test_overflow_is_rejected_on_every_rank_including_empty_destinations(self) -> None:
        """A cold rank must report the same hot-rank overflow before execution."""
        counts = torch.tensor([[4, 0, 0, 0], [4, 0, 0, 0]], dtype=torch.int32)
        for rank in range(2):
            spec = self._spec(ep_size=2, rank_id=rank, expert_capacity_factor=1.0, receive_capacity=4)
            with (
                self.subTest(rank=rank),
                self.assertRaisesRegex(RuntimeError, "configured_capacity=4, actual_maximum=8"),
            ):
                _expert_capacity(counts, spec)


if __name__ == "__main__":
    unittest.main()
