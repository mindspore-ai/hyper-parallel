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
    prepare_topk_route,
)
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec


class TestMegaMoeRoute(unittest.TestCase):
    """Validate trusted Router counts and bounded capacity without hardware."""

    def test_route_counts_match_native_abi_after_int64_dynamic_gather(self) -> None:
        """Both transports must lower dynamic INT64 counts to the kernel's INT32 ABI."""
        for mode in ('push', 'pull'):
            for dtype in (torch.int32, torch.int64):
                with self.subTest(mode=mode, dtype=dtype):
                    counts = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtype)
                    spec = replace(self._spec(ep_size=2), dispatch_mode=mode)
                    metadata = route_module._compute_route_metadata(counts[0], counts, counts[:, :2], spec, 14)
                    self.assertEqual(metadata.dispatch_size.dtype, torch.int32)
                    self.assertEqual(metadata.combine_size.dtype, torch.int32)
                    self.assertEqual(metadata.dispatch_src_off.dtype, torch.int64)
                    self.assertEqual(metadata.dispatch_target_off.dtype, torch.int64)
                    expected = [1, 2, 3, 4] if mode == 'push' else [1, 2, 5, 6]
                    self.assertEqual(metadata.dispatch_size.tolist(), expected)

    def test_dynamic_ragged_counts_preserve_exact_round_trip_intervals(self) -> None:
        """Include an empty source receiving work and a source with an empty destination."""
        counts = torch.tensor([[0, 0, 0, 0], [3, 1, 2, 0]], dtype=torch.int64)
        header = torch.tensor([[0, 0, 15], [0, 3, 15]], dtype=torch.int64)
        payload = torch.cat((header, counts), dim=1)
        for rank in range(2):
            spec = replace(self._spec(ep_size=2, rank_id=rank), max_local_num_tokens=257)
            capacity, plan_tokens, maximum = route_module._dynamic_summary(payload, spec)
            self.assertEqual((capacity, plan_tokens), ((4, 128) if rank == 0 else (2, 128)))
            self.assertEqual(maximum, 4)
            received = counts[:, rank * 2:rank * 2 + 2]
            meta = route_module._compute_route_metadata(counts[rank], counts, received, spec, capacity)
            self.assertEqual(meta.dispatch_src_off.tolist(), [0, 0, 0, 0] if rank == 0 else [0, 3, 4, 6])
            self.assertEqual(meta.group_list.tolist(), [3, 4] if rank == 0 else [2, 2])
            self.assertEqual(int(meta.dispatch_size.sum()), 0 if rank == 0 else 6)

    def test_dynamic_invalid_calls_fail_on_every_rank(self) -> None:
        """Reject over-limit tokens, inconsistent gradients/counts, and receive overflow together."""
        valid = torch.tensor([[0, 1, 15, 1, 1, 0, 0], [0, 2, 15, 2, 2, 0, 0]], dtype=torch.int64)
        for column, value, message in ((0, 1, "token capacity"), (1, 258, "token capacity"),
                                       (2, 0, "autograd participation"), (3, 9, "counts")):
            invalid = valid.clone()
            invalid[1, column] = value
            for rank in range(2):
                spec = replace(self._spec(ep_size=2, rank_id=rank), max_local_num_tokens=257)
                with self.subTest(column=column, rank=rank), self.assertRaisesRegex(RuntimeError, message):
                    route_module._dynamic_summary(invalid, spec)
        spec = replace(self._spec(ep_size=2), max_local_num_tokens=257, receive_capacity=4)
        self.assertEqual(route_module._dynamic_summary(valid, spec)[2], 6)

    def test_empty_dynamic_route_keeps_autograd_without_native_permute(self) -> None:
        """Zero logical rows remain empty, including router and expert gradient connections."""
        spec = replace(self._spec(), max_local_num_tokens=128)
        hidden = torch.empty(0, 4, requires_grad=True)
        ids = torch.empty(0, 2, dtype=torch.int32)
        weights = torch.empty(0, 2, requires_grad=True)
        with patch.object(route_module.torch_npu, "npu_moe_token_permute") as permute:
            route = prepare_topk_route(hidden, ids, weights, spec, None, autograd_mask=15)
        permute.assert_not_called()
        self.assertEqual(route.routed_tokens.shape[0], 0)
        self.assertEqual(route.metadata.expert_capacity, 1)
        self.assertEqual(route.metadata.plan_tokens, 128)
        output = route_module.restore_topk_output(hidden, route.unpermute_mapping, weights)
        output.sum().backward()
        self.assertEqual(tuple(hidden.grad.shape), (0, 4))
        self.assertEqual(tuple(weights.grad.shape), (0, 2))

    def test_dynamic_gather_keeps_variable_payload_without_token_padding(self) -> None:
        """A rank with 129 tokens shares a 256 plan with an empty peer using one gather."""
        spec = replace(self._spec(ep_size=2), max_local_num_tokens=257)
        hidden = torch.randn(129, 4)
        ids = torch.tensor([[0, 1]] * 129, dtype=torch.int32)
        weights = torch.ones(129, 2)
        spec = replace(spec, receive_capacity=512, local_num_tokens=384)
        payload = torch.tensor([[0, 129, 15, 129, 129, 0, 0], [0, 0, 15, 0, 0, 0, 0]])

        def gather(output: Any, value: Any, **kwargs: Any) -> Mock:
            """Emulate one asynchronous count exchange."""
            self.assertEqual(value[:3].tolist(), [0, 129, 15])
            self.assertTrue(kwargs["async_op"])
            output.copy_(payload.flatten())
            return Mock()

        with (
            patch.object(route_module.dist, "all_gather_into_tensor", side_effect=gather),
            patch.object(route_module, "_permute_topk_input",
                         return_value=(hidden.repeat_interleave(2, dim=0), torch.arange(258))) as permute,
        ):
            route = prepare_topk_route(hidden, ids, weights, spec, None, autograd_mask=15)
        self.assertIs(permute.call_args.args[0], hidden)
        self.assertEqual(tuple(route.routed_tokens.shape), (258, 4))
        self.assertEqual(route.metadata.plan_tokens, 256)
        self.assertEqual(route.metadata.local_num_tokens, 129)

    @staticmethod
    def _spec(
        *,
        ep_size: int = 1,
        rank_id: int = 0,
        initial_capacity_factor: float | None = None,
        receive_capacity: int = 128,
    ) -> MegaMoeSpec:
        """Build a small CPU-only route specification."""
        return MegaMoeSpec(
            local_num_tokens=2,
            hidden_size=4,
            intermediate_size=2,
            num_experts=4,
            top_k=2,
            initial_capacity_factor=initial_capacity_factor,
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
        self.assertEqual((spec.dispatch_split, spec.combine_split, spec.swiglu_split), (128, 128, 128))
        events.clear()
        workspace.in_use, workspace.source_buffer = True, torch.empty_like(routed_tokens)
        workspace.wait_for_reuse.reset_mock()
        pull_spec = replace(spec, dispatch_mode="pull")

        def _permute_out(tokens, indices, output, mapping):
            self.assertIs(tokens, hidden_states)
            self.assertIs(output, workspace.source_buffer)
            self.assertTrue(indices.is_contiguous())
            self.assertEqual(mapping.dtype, torch.int32)
            events.append("permute-out")
            output.copy_(routed_tokens)
            mapping.copy_(unpermute_mapping)

        with (patch.object(route_module.dist, "all_gather_into_tensor", side_effect=gather_counts) as gather,
              patch.object(route_module.multicore_ops, "moe_token_permute_out", side_effect=_permute_out),
              patch.object(route_module, "_permute_topk_input") as allocating):
            first, pull = [prepare_topk_route(hidden_states, topk_ids.T.contiguous().T, topk_weights,
                                              pull_spec, supplied_counts, workspace) for _ in range(2)]
            self.assertIs(pull.routed_tokens, workspace.source_buffer)
            self.assertNotEqual(first.unpermute_mapping.data_ptr(), pull.unpermute_mapping.data_ptr())
            torch.testing.assert_close(first.unpermute_mapping, unpermute_mapping)
            for active, source in ((False, workspace.source_buffer), (True, None)):
                workspace.in_use, workspace.source_buffer = active, source
                gather.reset_mock()
                with self.assertRaisesRegex(RuntimeError, "initialized, claimed workspace"):
                    prepare_topk_route(hidden_states, topk_ids, topk_weights, pull_spec, supplied_counts, workspace)
                gather.assert_not_called()
        allocating.assert_not_called()
        workspace.wait_for_reuse.assert_not_called()
        self.assertEqual(events, ["gather", "permute-out", "wait"] * 2)
        self.assertEqual(pull.metadata.dispatch_src_off.tolist(), [3, 6, 11, 18])
        self.assertEqual(pull.metadata.dispatch_target_off.tolist(), [0, 10, 3, 14])
        self.assertEqual(pull.metadata.dispatch_size.tolist(), [3, 4, 7, 8])
        self.assertEqual(pull.maximum_received_slots, 22)

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
                    self.assertEqual(actual, (expected, max(capacities)))
                    self.assertEqual(rank_spec.receive_capacity, 256)
                    self.assertEqual(rank_spec.routed_slots, 4)

if __name__ == "__main__":
    unittest.main()
