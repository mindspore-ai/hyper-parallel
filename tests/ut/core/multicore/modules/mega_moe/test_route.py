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
from types import SimpleNamespace
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
from tests.common.mark_utils import arg_mark


class TestMegaMoeRoute(unittest.TestCase):
    """Validate trusted Router counts and bounded capacity without hardware."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_automatic_route_grows_before_writing_pull_source(self) -> None:
        """Feature: MegaMoe route preparation.

        Description: One count exchange establishes capacity before a pull view can reference SHMEM.
        Expectation: Count agreement precedes growth and every pull write uses the new source buffer.
        """
        for mode in ("push", "pull"):
            with self.subTest(mode=mode):
                self._automatic_route_grows_before_writing_source(mode)

    def _automatic_route_grows_before_writing_source(self, mode: str) -> None:
        """Check one transport with callbacks bound outside the test loop."""
        spec = replace(self._spec(ep_size=2), auto_tokens=True, dispatch_mode=mode)
        hidden = torch.randn(257, 4)
        ids = torch.tensor([[0, 1]] * 257, dtype=torch.int32)
        weights = torch.ones(257, 2)
        payload = torch.tensor([[0, 257, 15, 257, 257, 0, 0], [0, 0, 15, 0, 0, 0, 0]])
        workspace = Mock(in_use=False, source_buffer=None)
        order = []

        def gather(output: Any, unused_value: Any, **_kwargs: Any) -> Mock:
            """Record completion before permitting the capacity callback."""
            output.copy_(payload.flatten())
            return Mock(wait=lambda: order.append("wait"))

        def grow(tokens: int, received: int) -> None:
            """Install the larger source allocation only after count agreement."""
            self.assertEqual(order, ["wait"] if mode == "pull" else ["permute", "wait"])
            self.assertEqual((tokens, received), (384, 514))
            workspace.source_buffer = torch.empty(tokens * 2, 4)
            workspace.in_use = True
            order.append("grow")

        def permute(*args: Any) -> tuple:
            """Observe the new allocation rather than an invalidated old view."""
            self.assertEqual(order, ["wait", "grow"] if mode == "pull" else [])
            order.append("permute")
            output = args[2] if mode == "pull" else hidden.repeat_interleave(2, dim=0)
            return output, torch.arange(514, dtype=torch.int32)

        with (
            patch.object(route_module.dist, "all_gather_into_tensor", side_effect=gather) as exchange,
            patch.object(route_module, "_permute_topk_input", side_effect=permute),
            patch.object(route_module, "_permute_topk_input_out", side_effect=permute),
        ):
            route = prepare_topk_route(hidden, ids, weights, spec, None, workspace,
                                       autograd_mask=15, prepare_workspace=grow)
        self.assertEqual(exchange.call_count, 1)
        self.assertEqual(order, ["wait", "grow", "permute"] if mode == "pull" else ["permute", "wait", "grow"])
        self.assertEqual(tuple(route.routed_tokens.shape), (514, 4))
        self.assertEqual(route.metadata.expert_capacity, 514)


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_automatic_pull_route_never_allocates_or_writes_source(self) -> None:
        """Feature: MegaMoe route preparation.

        Description: All ranks reject inconsistent participation before the allocation callback can run.
        Expectation: Invalid peers cannot trigger capacity allocation or a pull source write.
        """
        spec = replace(self._spec(ep_size=2), auto_tokens=True, dispatch_mode="pull")
        hidden, ids, weights = torch.ones(1, 4), torch.tensor([[0, 1]], dtype=torch.int32), torch.ones(1, 2)
        payload = torch.tensor([[0, 1, 15, 1, 1, 0, 0], [0, 1, 0, 1, 1, 0, 0]])

        def gather(output: Any, unused_value: Any, **_kwargs: Any) -> Mock:
            """Inject a peer that chose incompatible autograd participation."""
            output.copy_(payload.flatten())
            return Mock()

        callback = Mock()
        with (patch.object(route_module.dist, "all_gather_into_tensor", side_effect=gather),
              patch.object(route_module, "_permute_topk_input_out") as permute,
              self.assertRaisesRegex(RuntimeError, "autograd participation")):
            prepare_topk_route(hidden, ids, weights, spec, None, prepare_workspace=callback)
        callback.assert_not_called()
        permute.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_staged_pull_publishes_only_after_capacity_agreement(self) -> None:
        """Feature: Automatic MegaMoe route workspace reuse.

        Description: Ordinary permutation may overlap agreement while SHMEM writes use the committed allocation.
        Expectation: The old source stays untouched and publication uses the newly committed allocation.
        """
        spec = replace(self._spec(ep_size=2), auto_tokens=True, dispatch_mode="pull")
        hidden = torch.randn(257, 4)
        ids = torch.tensor([[0, 1]] * 257, dtype=torch.int32)
        wire = torch.tensor([[0, 257, 15, 257, 257, 0, 0], [0, 0, 15, 0, 0, 0, 0]])
        old_buffer = torch.zeros(1, 4)
        workspace = SimpleNamespace(in_use=False, source_buffer=old_buffer, route_cache=None,
                                    route_header_cache=None)
        order = []
        staged = hidden.repeat_interleave(2, dim=0)
        original_publish = route_module._publish_staged_pull

        def gather(output: torch.Tensor, value: torch.Tensor, **_kwargs: object) -> Mock:
            """Complete agreement after ordinary-memory permutation was submitted."""
            output.copy_(wire.flatten())
            return Mock(wait=lambda: order.append("wait"))

        def stage(*_args: object) -> tuple:
            """Keep the old SHMEM allocation untouched before agreement."""
            self.assertEqual(order, [])
            self.assertFalse(workspace.in_use)
            order.append("stage")
            return staged, torch.arange(514, dtype=torch.int32)

        def grow(tokens: int, received: int) -> None:
            """Replace the undersized allocation after collective validation."""
            self.assertEqual(order, ["stage", "wait"])
            self.assertEqual((tokens, received), (384, 514))
            workspace.source_buffer = torch.empty(tokens * 2, 4)
            workspace.in_use = True
            order.append("grow")

        def publish(*args: object) -> tuple:
            """Copy into the newly committed source buffer with its lease held."""
            self.assertEqual(order, ["stage", "wait", "grow"])
            order.append("publish")
            return original_publish(*args)

        with (patch.object(route_module.dist, "all_gather_into_tensor", side_effect=gather),
              patch.object(route_module, "_can_stage_pull", return_value=True),
              patch.object(route_module, "_permute_topk_input", side_effect=stage),
              patch.object(route_module, "_permute_topk_input_out") as direct,
              patch.object(route_module, "_publish_staged_pull", side_effect=publish)):
            prepared = prepare_topk_route(hidden, ids, torch.ones(257, 2), spec, None, workspace,
                                          autograd_mask=15, prepare_workspace=grow)
        self.assertEqual(order, ["stage", "wait", "grow", "publish"])
        direct.assert_not_called()
        self.assertEqual(prepared.routed_tokens.data_ptr(), workspace.source_buffer.data_ptr())
        torch.testing.assert_close(prepared.routed_tokens, staged, rtol=0, atol=0)
        torch.testing.assert_close(old_buffer, torch.zeros_like(old_buffer), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_small_pull_staging_keeps_collective_rejection_ahead_of_publication(self) -> None:
        """Feature: Automatic MegaMoe route workspace reuse.

        Description: A failing peer may discard ordinary staging but must never publish to SHMEM.
        Expectation: A bad peer prevents capacity changes and every SHMEM publication.
        """
        spec = replace(self._spec(ep_size=2), auto_tokens=True, dispatch_mode="pull")
        hidden, ids = torch.ones(1, 4), torch.tensor([[0, 1]], dtype=torch.int32)
        wire = torch.tensor([[0, 1, 15, 1, 1, 0, 0], [0, 1, 0, 1, 1, 0, 0]])

        def gather(output: torch.Tensor, value: torch.Tensor, **_kwargs: object) -> Mock:
            """Inject an incompatible peer mask through the real wire-validation path."""
            output.copy_(wire.flatten())
            return Mock()

        grow = Mock()
        workspace = SimpleNamespace(in_use=False, source_buffer=None, route_header_cache=None)
        with (patch.object(route_module.dist, "all_gather_into_tensor", side_effect=gather),
              patch.object(route_module, "_can_stage_pull", return_value=True),
              patch.object(route_module, "_permute_topk_input", return_value=(hidden, ids.flatten())) as stage,
              patch.object(route_module, "_publish_staged_pull") as publish,
              self.assertRaisesRegex(RuntimeError, "autograd participation")):
            prepare_topk_route(hidden, ids, torch.ones(1, 2), spec, None, workspace,
                               autograd_mask=15, prepare_workspace=grow)
        stage.assert_called_once()
        grow.assert_not_called()
        publish.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_pull_staging_is_limited_by_actual_tensor_bytes(self) -> None:
        """Feature: Automatic MegaMoe route workspace reuse.

        Description: Limit temporary Top-K expansion by bytes across hidden sizes and token counts.
        Expectation: Only automatic NPU Pull inputs within the byte budget use staging.
        """
        spec = replace(self._spec(ep_size=2), auto_tokens=True, dispatch_mode="pull", top_k=8)
        for tokens, expected in ((256, True), (512, True), (1024, False), (4096, False)):
            tensor = SimpleNamespace(device=SimpleNamespace(type="npu"), numel=lambda tokens=tokens: tokens * 5120,
                                     element_size=lambda: 2)
            with self.subTest(tokens=tokens):
                self.assertEqual(route_module._can_stage_pull(tensor, spec), expected)


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dynamic_ragged_counts_preserve_exact_round_trip_intervals(self) -> None:
        """Feature: MegaMoe route preparation.

        Description: Include an empty source receiving work and a source with an empty destination.
        Expectation: Source and destination intervals describe only the actual routed token rows.
        """
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dynamic_invalid_calls_fail_on_every_rank(self) -> None:
        """Feature: MegaMoe route preparation.

        Description: Reject over-limit tokens, inconsistent gradients/counts, and receive overflow together.
        Expectation: Each rank rejects invalid source lengths, counts and participation.
        """
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_empty_dynamic_route_keeps_autograd_without_native_permute(self) -> None:
        """Feature: MegaMoe route preparation.

        Description: Zero logical rows remain empty, including router and expert gradient connections.
        Expectation: Empty outputs retain gradients while native permutation is skipped.
        """
        spec = replace(self._spec(), max_local_num_tokens=128)
        hidden = torch.empty(0, 4, requires_grad=True)
        ids = torch.empty(0, 2, dtype=torch.int32)
        weights = torch.empty(0, 2, requires_grad=True)
        with patch.object(route_module, "load_torch_npu") as backend:
            route = prepare_topk_route(hidden, ids, weights, spec, None, autograd_mask=15)
        backend.assert_not_called()
        self.assertEqual(route.routed_tokens.shape[0], 0)
        self.assertEqual(route.metadata.expert_capacity, 1)
        self.assertEqual(route.metadata.plan_tokens, 128)
        output = route_module.restore_topk_output(hidden, route.unpermute_mapping, weights)
        output.sum().backward()
        self.assertEqual(tuple(hidden.grad.shape), (0, 4))
        self.assertEqual(tuple(weights.grad.shape), (0, 2))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dynamic_gather_keeps_variable_payload_without_token_padding(self) -> None:
        """Feature: MegaMoe route preparation.

        Description: A rank with 129 tokens shares a 256 plan with an empty peer using one gather.
        Expectation: One count exchange preserves actual rows without padding source tensors.
        """
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_router_counts_are_reused_or_computed_only_when_omitted(self) -> None:
        """Feature: MegaMoe route preparation.

        Description: Reuse supplied counts and compute a histogram only when omitted.
        Expectation: Trusted contiguous INT32 counts are reused and omitted counts are computed.
        """
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_async_count_gather_overlaps_permute_and_builds_metadata(self) -> None:
        """Feature: MegaMoe route preparation.

        Description: Wait after permutation and derive every native offset from counts.
        Expectation: Permutation overlaps the count exchange and pull mappings retain their own storage.
        """
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_local_capacity_tracks_routes_below_source_size_and_empty_ranks(self) -> None:
        """Feature: MegaMoe route preparation.

        Description: Shrink destination intermediates independently of source or SHMEM size.
        Expectation: Each rank uses its actual destination load with non-null storage for empty destinations.
        """
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


class TestMegaMoeDynamicMetadata(unittest.TestCase):
    """Keep dynamic validation and the native count ABI independent of count-table policy."""

    @staticmethod
    def _spec(experts: int = 24, rank: int = 0, automatic: bool = True) -> MegaMoeSpec:
        """Construct a four-rank CPU specification without a process group."""
        return MegaMoeSpec(
            local_num_tokens=512,
            hidden_size=16,
            intermediate_size=8,
            num_experts=experts,
            top_k=2,
            initial_capacity_factor=None,
            receive_capacity=512,
            ep_size=4,
            ep_group=None,
            rank_id=rank,
            num_cube_cores=24,
            auto_tokens=automatic,
            max_local_num_tokens=None if automatic else 512,
        )

    @staticmethod
    def _payload(experts: int, dtype: torch.dtype) -> torch.Tensor:
        """Generate distinct Top-K histograms including an empty source."""
        rows = []
        for rank, tokens in enumerate((0, 1, 129, 257)):
            ids = (torch.arange(tokens * 2) + rank * 3) % experts
            counts = torch.bincount(ids, minlength=experts)
            rows.append([0, tokens, 31, *counts.tolist()])
        return torch.tensor(rows, dtype=dtype)

    @staticmethod
    def _reference(payload: torch.Tensor, spec: MegaMoeSpec) -> tuple[int, int, int]:
        """Use tensor reductions to independently check the Host list implementation."""
        counts = payload[:, 3:]
        loads = counts.reshape(spec.ep_size, spec.ep_size, spec.local_experts).sum((0, 2))
        maximum_tokens = int(payload[:, 1].max())
        schedule = max(128, (maximum_tokens + 127) // 128 * 128)
        return max(1, int(loads[spec.rank_id])), schedule, int(loads.max())

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_host_and_device_summary_policies_match_exact_route_counts(self) -> None:
        """Feature: dynamic count summary.

        Description: Cross table sizes, integer widths, transports and empty-source ranks.
        Expectation: Every policy preserves exact destination loads and aligned schedules.
        """
        for experts in (24, 256, 260, 512, 1024, 2048):
            for dtype in (torch.int32, torch.int64):
                payload = self._payload(experts, dtype)
                for rank in range(4):
                    spec = self._spec(experts, rank)
                    with self.subTest(experts=experts, dtype=dtype, rank=rank):
                        self.assertEqual(
                            route_module._read_dynamic_route(payload, spec)[0], self._reference(payload, spec)
                        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_large_summary_never_copies_the_complete_count_table_to_host(self) -> None:
        """Feature: bounded summary transfer.

        Description: Prohibit raw readback for a count matrix above the Host policy limit.
        Expectation: Device reduction reads back a compact summary with the same results.
        """
        payload = self._payload(2048, torch.int64)
        spec = self._spec(2048)
        expected = self._reference(payload, spec)
        original = torch.Tensor.tolist

        def tolist(tensor: torch.Tensor) -> list:
            """Distinguish a full-table transfer from the compact device summary."""
            if tensor is payload:
                raise AssertionError(f"large raw count table copied to Host: shape={tuple(tensor.shape)}")
            return original(tensor)

        with patch.object(torch.Tensor, "tolist", tolist):
            self.assertEqual(route_module._read_dynamic_route(payload, spec)[0], expected)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_summary_validation_remains_collective_for_both_table_policies(self) -> None:
        """Feature: distributed dynamic validation.

        Description: Inject native overflow, mixed gradients and invalid histogram values.
        Expectation: Small and large tables reject invalid metadata on every rank.
        """
        for experts in (24, 2048):
            valid = self._payload(experts, torch.int64)
            for column, value, message in ((0, 1, "INT32"), (2, 0, "autograd participation"), (3, -1, "counts")):
                invalid = valid.clone()
                invalid[1, column] = value
                for rank in range(4):
                    with self.subTest(experts=experts, column=column, rank=rank):
                        with self.assertRaisesRegex(RuntimeError, message):
                            route_module._read_dynamic_route(invalid, self._spec(experts, rank))
            invalid = valid.clone()
            invalid[1, 1] = 2**31
            invalid[1, 3:] = 0
            invalid[1, 3:5] = 2**31
            with self.assertRaisesRegex(RuntimeError, "INT32"):
                route_module._read_dynamic_route(invalid, self._spec(experts))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_host_descriptors_match_independent_tensor_prefixes(self) -> None:
        """Feature: packed native route descriptors.

        Description: Compare every field across empty sources, uneven counts and both transports.
        Expectation: INT64 offsets, INT32 sizes and cumulative expert loads match tensor scans exactly.
        """
        for ep in (1, 2, 4, 8, 16):
            for dtype in (torch.int32, torch.int64):
                for local in (2, 6, 64):
                    experts = ep * local
                    generator = torch.Generator().manual_seed(experts)
                    counts = torch.randint(0, 1024, (ep, experts), dtype=dtype, generator=generator)
                    counts[0] = 0
                    raw = torch.cat((torch.zeros(ep, 3, dtype=torch.int32), counts), dim=1).tolist()
                    for rank in range(ep):
                        for mode in ("push", "pull"):
                            spec = replace(self._spec(experts), ep_size=ep, rank_id=rank, dispatch_mode=mode)
                            received = counts[:, rank * local : (rank + 1) * local].contiguous()
                            expected = route_module._compute_route_metadata(counts[rank], counts, received, spec, 32768)
                            actual = route_module._host_route_metadata(raw, spec, 32768, torch.device("cpu"))
                            with self.subTest(ep=ep, local=local, rank=rank, mode=mode):
                                for field, value in vars(expected).items():
                                    observed = getattr(actual, field)
                                    if isinstance(value, torch.Tensor):
                                        expected_dtype = torch.int32 if field.endswith("size") else torch.int64
                                        self.assertEqual(value.dtype, expected_dtype)
                                        torch.testing.assert_close(observed, value, rtol=0, atol=0)
                                        self.assertTrue(observed.is_contiguous())
                                        offset = observed.data_ptr() - observed.untyped_storage().data_ptr()
                                        self.assertEqual(offset % 512, 0)
                                    else:
                                        self.assertEqual(observed, value)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cached_routes_reuse_descriptors_but_permute_current_ids(self) -> None:
        """Feature: Automatic MegaMoe route workspace reuse.

        Description: Reuse metadata for unchanged histograms while consuming each call's token assignments.
        Expectation: Identical tables reuse descriptors, changed tables replace them, and token mappings stay
        current.
        """
        for mode in ("push", "pull"):
            for experts in (24, 384):
                with self.subTest(mode=mode, experts=experts):
                    self._check_cached_route_permutations(mode, experts)

    def _check_cached_route_permutations(self, mode: str, experts: int) -> None:
        """Exercise descriptor hits and replacement with an independent tensor-prefix oracle."""
        spec = replace(self._spec(experts, rank=1), dispatch_mode=mode)
        hidden = torch.ones(1, 16)
        ids = torch.tensor([[3, 4]], dtype=torch.int32)
        probabilities = torch.ones(1, 2)
        payload = self._payload(experts, torch.int32)
        workspace = SimpleNamespace(route_cache=None, in_use=True, source_buffer=torch.empty(1024, 16),
                                    wait_for_reuse=lambda: None)
        observed_ids = []

        def gather(output: torch.Tensor, value: torch.Tensor, **_kwargs: object) -> Mock:
            """Return all peers' current counts without a real process group."""
            output.copy_(payload.flatten())
            return Mock()

        def permute(hidden_states: torch.Tensor, current_ids: torch.Tensor, *_args: object) -> tuple:
            """Retain evidence that token order is never replaced by a cached permutation."""
            observed_ids.append(current_ids.clone())
            return hidden_states.repeat_interleave(2, dim=0), torch.arange(2, dtype=torch.int32)

        grow = Mock()
        with (patch.object(route_module.dist, "all_gather_into_tensor", side_effect=gather) as exchange,
              patch.object(route_module, "_permute_topk_input", side_effect=permute),
              patch.object(route_module, "_permute_topk_input_out", side_effect=permute),
              patch.object(route_module, "_dynamic_route_metadata", wraps=route_module._dynamic_route_metadata) as build,
              patch.object(route_module, "_host_route_metadata", wraps=route_module._host_route_metadata) as host_build,
              patch.object(route_module, "_compute_route_metadata", wraps=route_module._compute_route_metadata)
              as device_build):
            first = prepare_topk_route(hidden, ids, probabilities, spec, None, workspace,
                                       autograd_mask=31, prepare_workspace=grow)
            first_fields = {key: value.clone() for key, value in vars(first.metadata).items()
                            if isinstance(value, torch.Tensor)}
            second = prepare_topk_route(hidden, ids.flip(1), probabilities, spec, None, workspace,
                                        autograd_mask=31, prepare_workspace=grow)
            self.assertEqual(build.call_count, 2)
            self.assertNotEqual(first.metadata.dispatch_src_off.data_ptr(),
                                second.metadata.dispatch_src_off.data_ptr())
            again = prepare_topk_route(hidden, ids, probabilities, spec, None, workspace,
                                       autograd_mask=31, prepare_workspace=grow)
            self.assertEqual(build.call_count, 2)
            self.assertEqual(second.metadata.dispatch_src_off.data_ptr(),
                             again.metadata.dispatch_src_off.data_ptr())
            changed_ids = torch.tensor([[4, 5]], dtype=torch.int32)
            payload[1, 3:] = torch.bincount(changed_ids.flatten().long(), minlength=experts).to(torch.int32)
            third = prepare_topk_route(hidden, changed_ids, probabilities, spec, None, workspace,
                                       autograd_mask=31, prepare_workspace=grow)
            self.assertEqual(build.call_count, 3)
            self.assertEqual(exchange.call_count, 4)
            self.assertEqual(grow.call_count, 4)
            host_policy = spec.ep_size * experts <= route_module._HOST_DESCRIPTOR_MAX_COUNTS
            self.assertEqual(host_build.call_count, 3 if host_policy else 0)
            self.assertEqual(device_build.call_count, 0 if host_policy else 3)
        for observed, expected in zip(observed_ids, (ids, ids.flip(1), ids, changed_ids)):
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        counts = payload[:, 3:]
        start = spec.rank_id * spec.local_experts
        received = counts[:, start:start + spec.local_experts].contiguous()
        expected = route_module._compute_route_metadata(counts[1], counts, received, spec,
                                                        third.metadata.expert_capacity)
        for key, value in vars(expected).items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(getattr(third.metadata, key), value, rtol=0, atol=0)
                torch.testing.assert_close(getattr(first.metadata, key), first_fields[key], rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cached_pull_still_rejects_changed_peer_before_workspace_write(self) -> None:
        """Feature: Automatic MegaMoe route workspace reuse.

        Description: A previous valid descriptor must not hide new peer participation failures.
        Expectation: A previously cached route cannot bypass fresh peer validation.
        """
        spec = replace(self._spec(24, rank=1), dispatch_mode="pull")
        payload = self._payload(24, torch.int32)
        hidden, ids = torch.ones(1, 16), torch.tensor([[3, 4]], dtype=torch.int32)
        workspace = SimpleNamespace(route_cache=None, in_use=True, source_buffer=torch.empty(1024, 16))
        grow, permute = Mock(), Mock(return_value=(hidden.repeat_interleave(2, dim=0), torch.arange(2)))

        def gather(output: torch.Tensor, value: torch.Tensor, **_kwargs: object) -> Mock:
            """Supply valid and invalid wire tables in consecutive calls."""
            output.copy_(payload.flatten())
            return Mock()

        with (patch.object(route_module.dist, "all_gather_into_tensor", side_effect=gather),
              patch.object(route_module, "_permute_topk_input_out", permute)):
            prepare_topk_route(hidden, ids, torch.ones(1, 2), spec, None, workspace,
                               autograd_mask=31, prepare_workspace=grow)
            previous = workspace.route_cache
            payload[2, 2] = 0
            grow.reset_mock()
            permute.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "autograd participation"):
                prepare_topk_route(hidden, ids, torch.ones(1, 2), spec, None, workspace,
                                   autograd_mask=31, prepare_workspace=grow)
            grow.assert_not_called()
            permute.assert_not_called()
            self.assertIs(workspace.route_cache, previous)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cached_descriptors_wait_and_retain_storage_on_another_stream(self) -> None:
        """Feature: Automatic MegaMoe route workspace reuse.

        Description: A consumer stream must wait for upload and retain the shared descriptor allocation.
        Expectation: A different consumer waits for upload and retains both cached allocations.
        """
        order = []
        producer, consumer, ready = Mock(), Mock(), Mock()
        consumer.wait_event.side_effect = lambda _event: order.append("wait")
        tensors = [Mock(spec=torch.Tensor, record_stream=Mock(side_effect=lambda _stream: order.append("record")))
                   for _ in range(7)]
        metadata = SimpleNamespace(**{f"field{index}": tensor for index, tensor in enumerate(tensors)}, capacity=1)
        cached = route_module._HostRouteCache((), [], (1, 128, 1), metadata, producer, ready)
        with patch.object(torch, "npu", create=True) as mock_npu:
            mock_npu.current_stream.return_value = consumer
            cached.wait("npu:0")
        expected_order = ["wait"] + ["record"] * len(tensors)
        self.assertEqual(order, expected_order)
        consumer.wait_event.assert_called_once_with(ready)
        for tensor in tensors:
            tensor.record_stream.assert_called_once_with(consumer)
        with patch.object(torch, "npu", create=True) as mock_npu:
            mock_npu.current_stream.return_value = producer
            cached.wait("npu:0")
        self.assertEqual(order, expected_order)
        tensor = tensors[0]
        header = route_module._RouteHeaderCache((), tensor, producer, ready)
        order.clear()
        with patch.object(torch, "npu", create=True) as mock_npu:
            mock_npu.current_stream.return_value = consumer
            header.wait("npu:0")
        self.assertEqual(order, ["wait", "record"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dynamic_headers_reuse_immutable_storage_and_track_mask_changes(self) -> None:
        """Feature: Automatic MegaMoe route workspace reuse.

        Description: Control headers must update on shape or gradient-policy changes without mutating an in-flight
        header.
        Expectation: Identical keys reuse storage while token and mask changes allocate a new immutable header.
        """
        workspace = SimpleNamespace(route_header_cache=None)
        with patch.object(route_module, "_upload_route_tensor", wraps=route_module._upload_route_tensor) as upload:
            first = route_module._dynamic_header((0, 129, 31), torch.int32, "cpu", workspace)
            again = route_module._dynamic_header((0, 129, 31), torch.int32, "cpu", workspace)
            self.assertIsNot(again, first)
            confirmed = route_module._dynamic_header((0, 129, 31), torch.int32, "cpu", workspace)
            self.assertIs(confirmed, again)
            self.assertEqual(upload.call_count, 2)
            changed = route_module._dynamic_header((0, 129, 15), torch.int32, "cpu", workspace)
            resized = route_module._dynamic_header((0, 257, 15), torch.int32, "cpu", workspace)
            self.assertEqual(upload.call_count, 4)
        torch.testing.assert_close(first, torch.tensor([0, 129, 31], dtype=torch.int32), rtol=0, atol=0)
        torch.testing.assert_close(changed, torch.tensor([0, 129, 15], dtype=torch.int32), rtol=0, atol=0)
        torch.testing.assert_close(resized, torch.tensor([0, 257, 15], dtype=torch.int32), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_changed_tables_do_not_create_descriptor_cache_events(self) -> None:
        """Feature: Automatic MegaMoe descriptor cache admission.

        Description: Continuously changed tables must not pay for unused device cache readiness events.
        Expectation: Only the first repeated table retains descriptors and records one readiness event.
        """
        spec = self._spec()
        raw = self._payload(24, torch.int32).tolist()
        changed = [row.copy() for row in raw]
        changed[1][3 + 3] -= 1
        changed[1][3 + 5] += 1
        metadata = route_module._host_route_metadata(raw, spec, 128, torch.device("cpu"))
        changed_metadata = route_module._host_route_metadata(changed, spec, 128, torch.device("cpu"))
        workspace = SimpleNamespace(route_cache=None)
        producer, ready = Mock(), Mock()
        with (patch.object(torch, "npu", create=True) as mock_npu,
              patch.object(torch, "device", return_value=SimpleNamespace(type="npu", index=0))):
            current, event = mock_npu.current_stream, mock_npu.Event
            current.return_value, event.return_value = producer, ready
            route_module._cache_host_route(workspace, spec, raw, (1, 128, 1), metadata, "npu:0")
            self.assertIsNone(workspace.route_cache.metadata)
            route_module._cache_host_route(workspace, spec, changed, (1, 128, 1), changed_metadata, "npu:0")
            self.assertIsNone(workspace.route_cache.metadata)
            current.assert_not_called()
            event.assert_not_called()
            route_module._cache_host_route(workspace, spec, changed, (1, 128, 1), changed_metadata, "npu:0")
            self.assertIs(workspace.route_cache.metadata, changed_metadata)
            current.assert_called_once()
            event.assert_called_once()
            ready.record.assert_called_once_with(producer)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_changed_headers_do_not_create_unused_readiness_events(self) -> None:
        """Feature: Automatic MegaMoe control header cache admission.

        Description: New token lengths upload headers without recording events for unconfirmed reuse.
        Expectation: A repeated header records one event and subsequent identical headers reuse its tensor.
        """
        workspace = SimpleNamespace(route_header_cache=None)
        producer, ready = Mock(), Mock()
        with (patch.object(route_module, "_upload_route_tensor", side_effect=lambda value, _device: value) as upload,
              patch.object(torch, "npu", create=True) as mock_npu,
              patch.object(torch, "device", return_value=SimpleNamespace(type="npu", index=0))):
            current, event = mock_npu.current_stream, mock_npu.Event
            current.return_value, event.return_value = producer, ready
            route_module._dynamic_header((0, 129, 31), torch.int32, "npu:0", workspace)
            self.assertIsNone(workspace.route_header_cache.tensor)
            route_module._dynamic_header((0, 257, 31), torch.int32, "npu:0", workspace)
            self.assertIsNone(workspace.route_header_cache.tensor)
            current.assert_not_called()
            event.assert_not_called()
            confirmed = route_module._dynamic_header((0, 257, 31), torch.int32, "npu:0", workspace)
            again = route_module._dynamic_header((0, 257, 31), torch.int32, "npu:0", workspace)
            self.assertIs(confirmed, again)
            self.assertEqual(upload.call_count, 3)
            self.assertEqual(current.call_count, 2)
            event.assert_called_once()
            ready.record.assert_called_once_with(producer)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cached_device_dispatch_sizes_own_their_count_snapshot(self) -> None:
        """Feature: Automatic MegaMoe cached descriptor ownership.

        Description: A Router may reuse its count buffer after earlier invocations have completed.
        Expectation: Cached dispatch sizes remain equal to the gathered snapshot after caller mutation.
        """
        spec = self._spec(384, rank=1)
        payload = self._payload(384, torch.int32)
        summary, raw = route_module._read_dynamic_route(payload, spec)
        supplied = payload[spec.rank_id, 3:].clone()
        workspace = SimpleNamespace(route_cache=None)
        for _ in range(2):
            metadata = route_module._dynamic_route_metadata(supplied, payload, raw, spec, summary[0])
            route_module._cache_host_route(workspace, spec, raw, summary, metadata, "cpu")
        self.assertIsNotNone(workspace.route_cache.metadata)
        supplied.zero_()
        _, _, cached = route_module._read_cached_route(payload, spec, workspace, "cpu")
        self.assertIsNotNone(cached)
        torch.testing.assert_close(cached.dispatch_size, payload[spec.rank_id, 3:], rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_changing_wide_tables_back_off_and_recover_on_repeated_counts(self) -> None:
        """Feature: Automatic MegaMoe route cache reprobing.

        Description: Changing wide tables temporarily use device summaries without reading the full Host table.
        Expectation: Eight fallback calls preserve validation, then repeated counts recover immutable descriptors.
        """
        spec = self._spec(384, rank=1)
        payload = self._payload(384, torch.int32)
        workspace = SimpleNamespace(route_cache=None, route_cache_probe_after=0)
        summary, raw, _ = route_module._read_cached_route(payload, spec, workspace, "cpu")
        metadata = route_module._dynamic_route_metadata(payload[1, 3:], payload, raw, spec, summary[0])
        route_module._cache_host_route(workspace, spec, raw, summary, metadata, "cpu")
        payload[1, 3 + 3] -= 1
        payload[1, 3 + 5] += 1
        changed, raw, _ = route_module._read_cached_route(payload, spec, workspace, "cpu")
        metadata = route_module._dynamic_route_metadata(payload[1, 3:], payload, raw, spec, changed[0])
        route_module._cache_host_route(workspace, spec, raw, changed, metadata, "cpu")
        self.assertEqual(workspace.route_cache_probe_after, route_module._ROUTE_CACHE_REPROBE_STEPS)
        with (patch.object(payload, "tolist", side_effect=AssertionError("unexpected full Host table read")),
              patch.object(route_module, "_dynamic_summary", wraps=route_module._dynamic_summary) as device_summary):
            for remaining in range(route_module._ROUTE_CACHE_REPROBE_STEPS - 1, -1, -1):
                observed, host, cached = route_module._read_cached_route(payload, spec, workspace, "cpu")
                self.assertEqual(observed, changed)
                self.assertIsNone(host)
                self.assertIsNone(cached)
                self.assertEqual(workspace.route_cache_probe_after, remaining)
            self.assertEqual(device_summary.call_count, route_module._ROUTE_CACHE_REPROBE_STEPS)
        _, raw, cached = route_module._read_cached_route(payload, spec, workspace, "cpu")
        self.assertIsNone(cached)
        route_module._cache_host_route(workspace, spec, raw, changed, metadata, "cpu")
        _, _, cached = route_module._read_cached_route(payload, spec, workspace, "cpu")
        self.assertIs(cached, metadata)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_automatic_header_is_int32_and_bounded_header_remains_int64(self) -> None:
        """Feature: dynamic count wire format.

        Description: Exchange ragged source rows through both automatic and bounded routes.
        Expectation: Automatic metadata uses INT32 while the bounded format stays INT64.
        """
        for automatic in (False, True):
            spec = self._spec(automatic=automatic)
            hidden = torch.ones(129, 16)
            ids = torch.tensor([[0, 1]] * 129, dtype=torch.int32)
            counts = torch.tensor([129, 129] + [0] * 22, dtype=torch.int32)
            peer = torch.zeros(24, dtype=torch.int32)
            expected_dtype = torch.int32 if automatic else torch.int64
            header = torch.tensor(
                [[0, 129, 31, *counts.tolist()]] + [[0, 0, 31, *peer.tolist()]] * 3,
                dtype=expected_dtype).reshape(-1)

            def gather(output: torch.Tensor, value: torch.Tensor, payload: torch.Tensor = header,
                       **_kwargs: object) -> Mock:
                """Inspect the real locally packed header before providing peer counts."""
                self.assertEqual(value.dtype, payload.dtype)
                self.assertEqual(output.dtype, value.dtype)
                self.assertEqual(value[:3].tolist(), [0, 129, 31])
                output.copy_(payload)
                return Mock()

            with (patch.object(route_module.dist, "all_gather_into_tensor", side_effect=gather),
                  patch.object(route_module, "_permute_topk_input",
                               return_value=(hidden.repeat_interleave(2, dim=0), torch.arange(258)))):
                prepared = route_module.prepare_topk_route(hidden, ids, torch.ones(129, 2), spec, counts,
                                                          autograd_mask=31, prepare_workspace=Mock())
            self.assertEqual(prepared.metadata.local_num_tokens, 129)
            self.assertEqual(prepared.metadata.dispatch_size.dtype, torch.int32)
            self.assertEqual(prepared.metadata.combine_size.dtype, torch.int32)
            self.assertEqual(prepared.metadata.dispatch_src_off.dtype, torch.int64)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unrepresentable_automatic_source_still_joins_rejection_collective(self) -> None:
        """Feature: coordinated automatic overflow.

        Description: Submit a mocked source shape too large for an INT32 wire header.
        Expectation: Send an invalid flag with a representable sentinel before every rank rejects.
        """
        hidden = SimpleNamespace(shape=(2**31 + 128, 16), device=torch.device("cpu"))
        ids = torch.empty((0, 2), dtype=torch.int32)
        callback = Mock()

        def gather(output: torch.Tensor, value: torch.Tensor, **_kwargs: object) -> Mock:
            """Require overflow rejection to occur after the same collective on all ranks."""
            self.assertEqual(value.dtype, torch.int32)
            self.assertEqual(output.dtype, value.dtype)
            self.assertEqual(value[:3].tolist(), [1, 2**31 - 1, 31])
            output.copy_(value.repeat(4))
            return Mock()

        with (patch.object(route_module.dist, "all_gather_into_tensor", side_effect=gather) as exchange,
              patch.object(route_module, "_permute_topk_input") as permute,
              self.assertRaisesRegex(RuntimeError, "INT32")):
            route_module._prepare_dynamic_route(hidden, ids, ids.reshape(-1), self._spec(), None, None, 31,
                                                prepare_workspace=callback)
        exchange.assert_called_once()
        callback.assert_not_called()
        permute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
