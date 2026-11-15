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
"""CPU contracts for compact Index K selection and owner-only KL gradients."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import accumulate
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.distributed.context_parallel import sequence_dispatch
from hyper_parallel.distributed.context_parallel.index_key_dispatch import index_key_plan
from hyper_parallel.distributed.context_parallel.indexer_balance import indexer_task_plan


def _options(mode: str) -> dict:
    return {"block_size": 2 if mode in ("greedy", "hybrid") else None,
            "zigzag_min_length": 8 if mode == "hybrid" else None}


def _document_oracle(lengths: tuple, size: int, ratio: int, mode: str) -> tuple:
    """Enumerate raw queries and completed blocks without using compressed routing helpers."""
    boundaries = (0, *accumulate(lengths))
    local = boundaries[-1] // size
    documents = tuple(document for document, length in enumerate(lengths) for _ in range(length))
    blocks = [(document, end) for document, (first, last) in enumerate(zip(boundaries, boundaries[1:]))
              for end in range(first + ratio, last + 1, ratio)]

    def visible(queries: Iterable[int]) -> tuple[int, ...]:
        """Select complete same-document blocks visible to at least one raw query."""
        return tuple(sorted({index for query in queries for index, (document, end) in enumerate(blocks)
                             if documents[query] == document and end <= query + 1}))

    owned = [tuple(index for index, (_, end) in enumerate(blocks) if (end - 1) // local == rank)
             for rank in range(size)]
    owner = [visible(range(rank * local, (rank + 1) * local)) for rank in range(size)]
    worker = owner
    if mode != "none":
        tasks = indexer_task_plan(lengths, size, ratio, mode, **_options(mode))
        worker = [visible(boundaries[task.document] + query for task in assigned
                          for query in range(task.start, task.end)) for assigned in tasks]
    demands = [tuple(sorted(set(original) | set(selected))) for original, selected in zip(owner, worker)]
    return owned, owner, demands, blocks


class _CopyOnWait:
    """Expose the mock receive buffer only once the implementation waits."""

    def __init__(self, output: torch.Tensor, values: torch.Tensor) -> None:
        """Retain the receive buffer and its expected payload until wait."""
        self.output = output
        self.values = values

    def wait(self) -> None:
        """Publish remote values without a device or process group."""
        self.output.copy_(self.values)


class TestIndexKeyDispatch(unittest.TestCase):
    """Cover document isolation, global coordinates and restricted reverse dispatch."""

    def test_demand_union_matches_completed_document_blocks(self):
        """Worker selection and original-owner KL jointly determine each compact bank."""
        for lengths in ((3, 5, 24), (*([1] * 8), 9, 15)):
            for ratio in (1, 2):
                for mode in ("none", "zigzag", "greedy", "hybrid"):
                    owned, owner, demands, _ = _document_oracle(lengths, 4, ratio, mode)
                    plans = [index_key_plan(lengths, 4, rank, ratio, mode, **_options(mode)) for rank in range(4)]
                    for rank, plan in enumerate(plans):
                        with self.subTest(lengths=lengths, ratio=ratio, mode=mode, rank=rank):
                            positions = tuple(row for first, last, _ in plan.prefixes for row in range(first, last))
                            self.assertEqual(positions, demands[rank])
                            self.assertEqual(plan.global_positions(torch.device("cpu")).tolist(), list(positions))
                            self.assertEqual(plan.dispatch.local_length, len(owned[rank]))
                            first, last, offset = plan.owner_range
                            self.assertEqual(tuple(range(first, last)), owner[rank])
                            self.assertEqual(positions[offset:offset + last - first], owner[rank])
                            self.assertEqual(plan.gradient.length, len(owner[rank]))
                            sent = tuple(row for start, count in plan.dispatch.owner_ranges()
                                         for row in owned[rank][start:start + count])
                            expected = tuple(row for peer in range(4) if peer != rank
                                             for row in owned[rank] if row in demands[peer])
                            self.assertEqual(sent, expected)
                            for peer in range(4):
                                count = sum(row in demands[peer] for row in owned[rank]) if peer != rank else 0
                                self.assertEqual(plan.dispatch.input_splits[peer], count)
                                self.assertEqual(plan.dispatch.input_splits[peer],
                                                 plans[peer].dispatch.output_splits[rank])

    def test_topk_and_owner_view_retain_global_document_coordinates(self):
        """Compact indices round-trip per query, including invalid slots and empty prefixes."""
        lengths, ratio = (3, 5, 24), 2
        boundaries = (0, *accumulate(lengths))
        _, owner, demands, blocks = _document_oracle(lengths, 4, ratio, "greedy")
        assignments = indexer_task_plan(lengths, 4, ratio, "greedy", block_size=2)
        for rank, tasks in enumerate(assignments):
            plan = index_key_plan(lengths, 4, rank, ratio, "greedy", block_size=2)
            compact = torch.tensor(demands[rank], dtype=torch.float32).view(1, -1, 1)
            torch.testing.assert_close(plan.owner_view(compact).flatten(), torch.tensor(owner[rank]).float())
            positions = {global_row: compact_row for compact_row, global_row in enumerate(demands[rank])}
            segments, expected, cursor = [], [], 0
            for task in tasks:
                base = sum(document < task.document for document, _ in blocks)
                count = sum(document == task.document and end <= boundaries[task.document] + task.end
                            for document, end in blocks)
                segments.append((cursor, cursor + task.length, base, base + count, task.end % ratio))
                for query in range(task.start, task.end):
                    visible = [index for index, (document, end) in enumerate(blocks)
                               if document == task.document and end <= boundaries[document] + query + 1]
                    expected.append((visible[0], visible[-1], -1) if visible else (-1, -1, -1))
                cursor += task.length
            segments = tuple(segments)
            global_ids = torch.tensor(expected, dtype=torch.int64).unsqueeze(0)
            compact_ids = torch.tensor([[positions.get(index, -1) for index in row] for row in expected]).unsqueeze(0)
            torch.testing.assert_close(plan.global_indices(compact_ids, segments), global_ids, rtol=0, atol=0)
            for original, mapped in zip(segments, plan.map_segments(segments)):
                self.assertEqual((mapped[0], mapped[1], mapped[4]), (original[0], original[1], original[4]))
                self.assertEqual(demands[rank][mapped[2]:mapped[3]], tuple(range(original[2], original[3])))

    def test_empty_compression_does_not_launch_a_collective(self):
        """Length-one documents have no r2 keys even when the query work is balanced."""
        for mode in ("none", "zigzag", "greedy", "hybrid"):
            plan = index_key_plan((1,) * 16, 4, 0, 2, mode, **_options(mode))
            local = torch.empty((1, 0, 2), requires_grad=True)
            with patch.object(sequence_dispatch.dist, "all_to_all_single") as collective:
                output = sequence_dispatch.async_sequence_dispatch(
                    local, 1, plan.dispatch, None, gradient=plan.gradient).wait_with_gradient()
                selection, owner_key = output
                self.assertFalse(selection.requires_grad)
                owner_key.sum().backward()
                collective.assert_not_called()
            self.assertEqual(plan.prefixes, ())
            self.assertEqual(plan.global_positions(torch.device("cpu")).numel(), 0)
            self.assertEqual(local.grad.shape, local.shape)

    def _check_owner_gradients(self, mode: str, rank: int, lengths: tuple = (3, 29), separate=False, batch=1) -> None:
        """Compare async payloads and owner sums to independently enumerated consumers."""
        owned, owner, demands, blocks = _document_oracle(lengths, 4, 2, mode)
        plan = index_key_plan(lengths, 4, rank, 2, mode, **_options(mode))
        bank = torch.arange(len(blocks) * batch * 2).reshape(-1, batch, 2).bfloat16()
        coefficients = (256., 1., -256., .5)
        gradients = [torch.full((len(rows), batch, 2), value, dtype=torch.bfloat16)
                     for rows, value in zip(owner, coefficients)]
        local = bank[list(owned[rank])].movedim(0, 1).clone().requires_grad_(True)
        calls = []

        def values(rows: tuple[int, ...], peer: int | None = None) -> torch.Tensor:
            """Read canonical bank values or independently generated owner gradients."""
            if peer is None:
                return bank[list(rows)]
            return gradients[peer][[owner[peer].index(row) for row in rows]]

        def exchange(output: torch.Tensor, send: torch.Tensor, output_split_sizes: list[int],
                     input_split_sizes: list[int], group: None, async_op: bool) -> _CopyOnWait:
            """Validate peer payloads before publishing asynchronous receive values."""
            self.assertIsNone(group)
            self.assertTrue(async_op)
            backward = bool(calls)
            requested = owner if backward else demands
            outgoing, incoming = [], []
            for peer in range(4):
                sent = tuple(row for row in owned[rank] if row in requested[peer] and peer != rank)
                received = tuple(row for row in owned[peer] if row in requested[rank] and peer != rank)
                outgoing.append(values(received, rank) if backward else values(sent))
                incoming.append(values(sent, peer) if backward else values(received))
            self.assertEqual(input_split_sizes, [part.shape[0] for part in outgoing])
            self.assertEqual(output_split_sizes, [part.shape[0] for part in incoming])
            torch.testing.assert_close(send, torch.cat(outgoing), rtol=0, atol=0)
            calls.append(backward)
            return _CopyOnWait(output, torch.cat(incoming))

        with patch.object(sequence_dispatch.dist, "all_to_all_single", side_effect=exchange):
            handle = sequence_dispatch.async_sequence_dispatch(local, 1, plan.dispatch, None, gradient=plan.gradient)
            if separate:
                output, owner_key = handle.wait_with_gradient()
                self.assertFalse(output.requires_grad)
                torch.testing.assert_close(owner_key, plan.owner_view(output), rtol=0, atol=0)
                # Full/Reindex accumulate on one edge before a single reverse collective.
                loss = (owner_key * 0.25 + owner_key * 0.75)
            else:
                output = handle.wait()
                loss = plan.owner_view(output)
            torch.testing.assert_close(output, bank[list(demands[rank])].movedim(0, 1), rtol=0, atol=0)
            loss.backward(gradients[rank].movedim(0, 1))
            with self.assertRaisesRegex(RuntimeError, "cannot be waited twice"):
                handle.wait()
        expected = torch.tensor([sum(coefficients[peer] for peer in range(4) if row in owner[peer])
                                 for row in owned[rank]], dtype=torch.bfloat16).view(1, -1, 1).expand_as(local)
        torch.testing.assert_close(local.grad, expected, rtol=0, atol=0)
        self.assertEqual(calls, [False, True])
        self.assertTrue(plan.gradient.route.accumulate_fp32)

    def test_backward_returns_only_owner_kl_gradients_with_fp32_sum(self):
        """Selection-only K rows are excluded from reverse traffic and BF16 sums retain small terms."""
        for mode in ("zigzag", "greedy"):
            for rank in range(4):
                with self.subTest(mode=mode, rank=rank):
                    self._check_owner_gradients(mode, rank)
                    self._check_owner_gradients(mode, rank, separate=True)
        self._check_owner_gradients("greedy", 0, (*([1] * 8), 9, 15), separate=True)
        self._check_owner_gradients("greedy", 1, separate=True, batch=2)

    def test_uncovered_consumer_prefix_is_rejected(self):
        """A reused bank must cover the requested document before indices are remapped."""
        plan = index_key_plan((32,), 4, 0, 2)
        with self.assertRaisesRegex(ValueError, "does not cover"):
            plan.map_segments(((0, 4, 0, 16, 0),))
