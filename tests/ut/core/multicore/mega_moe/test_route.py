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
from typing import Any
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe import route as route_module
from hyper_parallel.core.multicore.modules.mega_moe.route import (
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
        """Skip default scalar sync and reject explicit bounded overflow."""
        spec = self._spec(ep_size=2, expert_capacity_factor=None)
        counts = torch.ones((2, 4), dtype=torch.int32)

        with patch.object(route_module, "_maximum_destination_load") as mock_maximum:
            _validate_bounded_capacity(counts, spec)

        mock_maximum.assert_not_called()
        bounded_spec = self._spec(
            ep_size=2,
            expert_capacity_factor=1.0,
            receive_capacity=4,
        )
        counts = torch.tensor(
            [[4, 0, 0, 0], [4, 0, 0, 0]],
            dtype=torch.int32,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "configured_capacity=4, actual_maximum=8",
        ):
            _validate_bounded_capacity(counts, bounded_spec)

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
            )

        self.assertEqual(events, ["gather", "permute", "wait"])
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
        self.assertEqual(metadata.expert_capacity, 128)


if __name__ == "__main__":
    unittest.main()
